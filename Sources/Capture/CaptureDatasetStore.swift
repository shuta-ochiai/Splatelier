// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Disk storage for CapturedScene (shared by resume after a training crash and the validation
// dataset (.valset)). Resume means retraining from saved frames/poses, not a mid-training
// checkpoint. Layout: <dir>/manifest.json + <dir>/frames/NNNNN.rgb (raw uint8 RGB, row-major
// w*h*3). PNG is not used because encoding is heavy. Do not change the layout so existing .valset
// files stay readable. CaptureDiagnostics is not saved (load returns nil).
// Frames saved or loaded here can drop their in-memory images and reload them from the directory
// (`CapturedScene.releaseSavedImages`), so training keeps only its views resident.

import Foundation
import MLX
import simd
import Trainer

/// A saved dataset directory that frame images reload from. Shared by the frames
/// `CaptureDatasetStore.save` / `load` attach it to, which keep it alive.
public final class CaptureDatasetFiles: @unchecked Sendable {
    private let lock = NSLock()
    private var directory: URL
    private var retired = false

    init(directory: URL) {
        self.directory = directory
    }

    /// Reads a frame image (under the lock, so `retire` cannot move the directory mid-read).
    func readImage(_ relativePath: String, height: Int, width: Int) throws -> MLXArray {
        lock.lock()
        defer { lock.unlock() }
        return try CaptureDatasetStore.readRawRGB8(
            url: directory.appendingPathComponent(relativePath), width: width, height: height)
    }

    /// The dataset is no longer needed as saved data (the crash-recovery copy after training
    /// finished). Moves it out of Documents at once, so recovery and the Files app no longer see
    /// it, and deletes it when the last frame reading from it is released.
    public func retire() {
        lock.lock()
        defer { lock.unlock() }
        guard !retired else { return }
        retired = true
        let fm = FileManager.default
        let destination = Self.retiredRoot.appendingPathComponent(UUID().uuidString, isDirectory: true)
        do {
            try fm.createDirectory(at: Self.retiredRoot, withIntermediateDirectories: true)
            try fm.moveItem(at: directory, to: destination)
            directory = destination
        } catch {
            // Left in place; still deleted on release.
        }
    }

    deinit {
        if retired { try? FileManager.default.removeItem(at: directory) }
    }

    static var retiredRoot: URL { CaptureScratch.directory("retired-datasets") }
}

public enum CaptureDatasetStore {
    public static let version = 2

    public enum StoreError: Swift.Error { case corruptManifest(String) }

    struct Manifest: Codable {
        var version: Int
        var createdAt: TimeInterval
        var frames: [FrameRecord]
        var sparsePoints: [Float]       // flattened xyz
        var baRefinedPoints: [Float]?
        var lidarPoints: [Float]
    }

    struct FrameRecord: Codable {
        var imageFile: String           // relative path "frames/00000.rgb"
        var viewmat: [Float]            // 16, row-major
        var intrinsics: [Float]         // [fx, fy, cx, cy]
        var width: Int
        var height: Int
        var timestamp: TimeInterval
        var anchorIdentifier: String?
        // ARKit feature observations (nil in old datasets).
        // IDs exceed 2^53 and lose precision as JSON doubles, so they are stored as strings.
        var featureTrackIDs: [String]?
        var featureCameraPoints: [Float]?   // flattened xyz
    }

    /// Saves `scene` to `directory`. Written to a temp directory and moved, so a partial write is
    /// never mistaken for a complete one. The frames can then release their images to it
    /// (`CapturedScene.releaseSavedImages`); undistorted frames, whose saved image differs from the
    /// captured one, stay in memory.
    @discardableResult
    public static func save(_ scene: CapturedScene, to directory: URL) throws -> CaptureDatasetFiles {
        let fm = FileManager.default
        let parent = directory.deletingLastPathComponent()
        let tmp = parent.appendingPathComponent(".tmp." + directory.lastPathComponent)
        if fm.fileExists(atPath: tmp.path) { try fm.removeItem(at: tmp) }
        try fm.createDirectory(at: tmp.appendingPathComponent("frames"),
                               withIntermediateDirectories: true)

        var records: [FrameRecord] = []
        records.reserveCapacity(scene.frames.count)
        for (i, f) in scene.frames.enumerated() {
            let rel = String(format: "frames/%05d.rgb", i)
            try writeRawRGB8(f.image, to: tmp.appendingPathComponent(rel))
            records.append(FrameRecord(
                imageFile: rel,
                viewmat: f.viewmat,
                intrinsics: [f.intrinsics.fx, f.intrinsics.fy, f.intrinsics.cx, f.intrinsics.cy],
                width: f.width, height: f.height,
                timestamp: f.timestamp,
                anchorIdentifier: f.anchorIdentifier?.uuidString,
                featureTrackIDs: f.featureObservations.isEmpty
                    ? nil : f.featureObservations.map { String($0.trackID) },
                featureCameraPoints: f.featureObservations.isEmpty
                    ? nil : packPoints(f.featureObservations.map(\.cameraPoint))
            ))
        }

        let manifest = Manifest(
            version: version,
            createdAt: scene.frames.first?.timestamp ?? 0,
            frames: records,
            sparsePoints: packPoints(scene.sparsePoints),
            baRefinedPoints: scene.baRefinedPoints.map(packPoints),
            lidarPoints: packPoints(scene.lidarPoints)
        )
        let data = try JSONEncoder().encode(manifest)
        try data.write(to: tmp.appendingPathComponent("manifest.json"))

        if fm.fileExists(atPath: directory.path) { try fm.removeItem(at: directory) }
        try fm.moveItem(at: tmp, to: directory)

        let files = CaptureDatasetFiles(directory: directory)
        for (f, r) in zip(scene.frames, records) where f.lensUndistortion == nil {
            f.imageStorage.attach(files, path: r.imageFile)
        }
        return files
    }

    /// Metadata read from the manifest only (frames/*.rgb are not read).
    public struct Info: Sendable {
        public let frameCount: Int
        public let createdAt: TimeInterval
        public let sparsePointCount: Int
    }

    public static func peek(at directory: URL) throws -> Info {
        let data = try Data(contentsOf: directory.appendingPathComponent("manifest.json"))
        let manifest = try JSONDecoder().decode(Manifest.self, from: data)
        return Info(frameCount: manifest.frames.count,
                    createdAt: manifest.createdAt,
                    sparsePointCount: manifest.sparsePoints.count / 3)
    }

    /// Restores a CapturedScene from a directory written by `save`.
    public static func load(from directory: URL) throws -> CapturedScene {
        let data = try Data(contentsOf: directory.appendingPathComponent("manifest.json"))
        let manifest = try JSONDecoder().decode(Manifest.self, from: data)

        let files = CaptureDatasetFiles(directory: directory)
        var frames: [CapturedFrame] = []
        frames.reserveCapacity(manifest.frames.count)
        for r in manifest.frames {
            guard r.intrinsics.count == 4 else {
                throw StoreError.corruptManifest("intrinsics count \(r.intrinsics.count)")
            }
            guard r.viewmat.count == 16 else {
                throw StoreError.corruptManifest("viewmat count \(r.viewmat.count)")
            }
            var featureObs: [ARKitFeatureObservation] = []
            if let ids = r.featureTrackIDs, let flat = r.featureCameraPoints {
                guard flat.count == ids.count * 3 else {
                    throw StoreError.corruptManifest(
                        "featureCameraPoints count \(flat.count) != 3 × \(ids.count)")
                }
                let pts = unpackPoints(flat)
                featureObs.reserveCapacity(ids.count)
                for (idStr, p) in zip(ids, pts) {
                    guard let id = UInt64(idStr) else {
                        throw StoreError.corruptManifest("featureTrackID \(idStr)")
                    }
                    featureObs.append(ARKitFeatureObservation(trackID: id, cameraPoint: p))
                }
            }
            let img = try readRawRGB8(
                url: directory.appendingPathComponent(r.imageFile),
                width: r.width, height: r.height
            )
            let frame = CapturedFrame(
                image: img,
                viewmat: r.viewmat,
                intrinsics: (r.intrinsics[0], r.intrinsics[1], r.intrinsics[2], r.intrinsics[3]),
                width: r.width, height: r.height,
                timestamp: r.timestamp,
                anchorIdentifier: r.anchorIdentifier.flatMap(UUID.init(uuidString:)),
                featureObservations: featureObs
            )
            frame.imageStorage.attach(files, path: r.imageFile)
            frames.append(frame)
        }

        return CapturedScene(
            frames: frames,
            sparsePoints: unpackPoints(manifest.sparsePoints),
            baRefinedPoints: manifest.baRefinedPoints.map(unpackPoints),
            lidarPoints: unpackPoints(manifest.lidarPoints),
            diagnostics: nil
        )
    }

    /// Writes uint8 [H, W, 3] as raw bytes as is.
    private static func writeRawRGB8(_ image: MLXArray, to url: URL) throws {
        let h = image.shape[0], w = image.shape[1]
        let bytes = image.reshaped([h * w * 3]).asArray(UInt8.self)
        try Data(bytes).write(to: url)
    }

    /// Inverse of `writeRawRGB8`. Returns uint8 [H, W, 3].
    static func readRawRGB8(url: URL, width: Int, height: Int) throws -> MLXArray {
        let data = try Data(contentsOf: url)
        let n = width * height * 3
        guard data.count == n else {
            throw StoreError.corruptManifest(
                "raw RGB byte count \(data.count) != \(n) for \(url.lastPathComponent)")
        }
        return MLXArray(data, [height, width, 3], type: UInt8.self)
    }

    private static func packPoints(_ pts: [SIMD3<Float>]) -> [Float] {
        var out = [Float]()
        out.reserveCapacity(pts.count * 3)
        for p in pts { out.append(p.x); out.append(p.y); out.append(p.z) }
        return out
    }

    private static func unpackPoints(_ flat: [Float]) -> [SIMD3<Float>] {
        var out = [SIMD3<Float>]()
        out.reserveCapacity(flat.count / 3)
        var i = 0
        while i + 2 < flat.count {
            out.append(SIMD3<Float>(flat[i], flat[i + 1], flat[i + 2]))
            i += 3
        }
        return out
    }
}
