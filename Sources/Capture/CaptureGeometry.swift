// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// What training and meshing share up to MVS: the frames after intake and BA, and the MVS result
// (the seed point cloud for training, the depth maps for meshing). Training builds it
// (`CaptureTraining.prepareGeometry`) and meshing after training reuses it instead of redoing
// intake, BA and MVS, so the mesh and the splats also share the same poses.

import Foundation
import MLX
import Trainer

public final class CaptureGeometry: @unchecked Sendable {
    /// Frames of the capture it was built from, including those intake dropped.
    let capturedFrameCount: Int
    /// Frames after pose-outlier and quality filtering and BA (undistorted if the lens distortion
    /// is significant).
    public let scene: CapturedScene
    /// Per-frame training sampling weights from optical flow, keyed by timestamp.
    let weightByTimestamp: [TimeInterval: Float]?
    let sceneScale: Float
    /// Voxel the MVS points were deduplicated at (the seed fusion matches it).
    let mvsVoxelSize: Float?
    /// MVS depth maps behind the point cloud, for meshing (nil if they could not be written).
    private let depthMaps: DepthMapFile?

    private let lock = NSLock()
    private var seedCloud: CaptureTraining.SeedCloud?

    init(capturedFrameCount: Int, scene: CapturedScene, weightByTimestamp: [TimeInterval: Float]?,
         sceneScale: Float, mvsVoxelSize: Float?, seedCloud: CaptureTraining.SeedCloud?,
         depthMaps: DepthMapFile?) {
        self.capturedFrameCount = capturedFrameCount
        self.scene = scene
        self.weightByTimestamp = weightByTimestamp
        self.sceneScale = sceneScale
        self.mvsVoxelSize = mvsVoxelSize
        self.seedCloud = seedCloud
        self.depthMaps = depthMaps
    }

    /// The MVS depth maps (read back from disk on each call).
    public func depthViews() throws -> [CostVolumeMVS.DepthView] {
        guard let depthMaps else { throw DepthMapFile.FileError.unavailable }
        return try depthMaps.read()
    }

    /// The MVS seed point cloud, handed over once: training only needs it for initialization, so
    /// it is not kept (millions of points) once taken.
    func takeSeedCloud() -> CaptureTraining.SeedCloud? {
        lock.lock()
        defer { lock.unlock() }
        let cloud = seedCloud
        seedCloud = nil
        return cloud
    }
}

/// MVS depth maps written to a temporary file as MVS produces them (7 bytes per MVS pixel, about
/// 205MB for 600 frames), so they are never all resident, and read back when meshing. The file
/// goes with this object.
final class DepthMapFile: @unchecked Sendable {
    enum FileError: Error { case unavailable, truncated }

    let url: URL
    private var handle: FileHandle?
    private(set) var count = 0
    private(set) var byteCount = 0
    private var writeError: Error?

    init() throws {
        let directory = CaptureScratch.directory("mvs-depth")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        url = directory.appendingPathComponent(UUID().uuidString + ".bin")
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
            throw CocoaError(.fileWriteUnknown)
        }
        handle = try FileHandle(forWritingTo: url)
    }

    deinit {
        try? handle?.close()
        try? FileManager.default.removeItem(at: url)
    }

    /// Per view: width, height (Int32), viewmat (16) and fx, fy, cx, cy (Float32), depth (w·h
    /// Float32), rgb (w·h·3 bytes).
    func append(_ v: CostVolumeMVS.DepthView) {
        guard writeError == nil, let handle else { return }
        var data = Data(capacity: Self.headerBytes + v.depth.count * 4 + v.rgb.count)
        withUnsafeBytes(of: Int32(v.width)) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: Int32(v.height)) { data.append(contentsOf: $0) }
        let camera = v.viewmat + [v.intrinsics.fx, v.intrinsics.fy, v.intrinsics.cx, v.intrinsics.cy]
        camera.withUnsafeBytes { data.append(contentsOf: $0) }
        v.depth.withUnsafeBytes { data.append(contentsOf: $0) }
        data.append(contentsOf: v.rgb)
        do {
            try handle.write(contentsOf: data)
            count += 1
            byteCount += data.count
        } catch {
            writeError = error
        }
    }

    /// Closes the file; throws the first write error.
    func finishWriting() throws {
        try handle?.close()
        handle = nil
        if let writeError { throw writeError }
    }

    private static let headerBytes = 8 + 20 * 4

    func read() throws -> [CostVolumeMVS.DepthView] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        func read(_ n: Int) throws -> Data {
            guard let data = try handle.read(upToCount: n), data.count == n else {
                throw FileError.truncated
            }
            return data
        }
        func floats(_ data: Data, at offset: Int, count: Int) -> [Float] {
            [Float](unsafeUninitializedCapacity: count) { buffer, initialized in
                data.withUnsafeBytes { raw in
                    UnsafeMutableRawBufferPointer(buffer).copyMemory(
                        from: UnsafeRawBufferPointer(rebasing: raw[offset ..< offset + count * 4]))
                }
                initialized = count
            }
        }
        var views: [CostVolumeMVS.DepthView] = []
        views.reserveCapacity(count)
        for _ in 0..<count {
            // FileHandle hands back autoreleased buffers; without a pool per view they all (205MB
            // for 600 frames) stayed alive into training on the same thread.
            try autoreleasepool {
                let header = try read(Self.headerBytes)
                let w = Int(header.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: Int32.self) })
                let h = Int(header.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 4, as: Int32.self) })
                let camera = floats(header, at: 8, count: 20)
                let depth = floats(try read(w * h * 4), at: 0, count: w * h)
                let rgb = [UInt8](try read(w * h * 3))
                views.append(CostVolumeMVS.DepthView(
                    width: w, height: h, depth: depth, rgb: rgb, viewmat: Array(camera[0..<16]),
                    intrinsics: (fx: camera[16], fy: camera[17], cx: camera[18], cy: camera[19])))
            }
        }
        return views
    }
}

extension CaptureTraining {

    /// Intake (pose outliers, quality filter, optical-flow weights) → BA → MVS. The defaults are
    /// the app's configuration (`MVSMode`).
    public static func prepareGeometry(
        scene: CapturedScene,
        // ORB feature cap and FAST threshold. More gives more BA points and a denser seed.
        orbMaxKeypointsPerFrame: Int = 1500,
        orbFastThreshold: Float = 0.05,
        // Estimate radial lens distortion (k1, k2) in BA; if significant, run MVS and training on
        // undistorted images.
        lensDistortion: LensDistortionEstimation.Parameters = .init(),
        frameQualityFilter: CaptureFrameQualityFilter.Parameters = .init(),
        opticalFlowDiagnostics: CaptureOpticalFlow.Parameters = .init(),
        flowSoftWeight: CaptureOpticalFlow.WeightParameters = .init(),
        // MVS dense seed (PocketGS Operator G). See MVSMode.
        mvsParameters: CostVolumeMVS.Parameters = MVSMode.mvsParameters,
        // Downscale factor of the MVS input (box average, intrinsics also 1/s). nil = the smallest
        // integer factor making the long side ≤ mvsTargetLongSide px, 1 = full size. At full size
        // Census 5x5 cannot absorb the ARKit pose epipolar error (1-4px) and it breaks down.
        mvsDownscale: Int? = nil,
        // Far clip for the dense seed: drop points beyond factor × sceneScale from the camera
        // centroid (guards against far floaters from depth outliers). 0 disables. The depth maps
        // for meshing are not clipped.
        seedFarClipFactor: Float = 6.0,
        // For pose accuracy evaluation: write post-BA (and pre-BA) viewmats / intrinsics /
        // timestamps as JSON.
        poseDumpURL: URL? = nil,
        log: (String) -> Void = { print($0) }
    ) -> CaptureGeometry {
        precondition(!scene.frames.isEmpty, "CapturedScene has no frames")
        logIntake(scene: scene, log: log)

        // [timing] Diagnostic log of per-phase preprocessing wall-clock.
        let t0 = Date()
        var tMark = t0
        func phaseMs(_ name: String) {
            let now = Date()
            log(String(format: "[timing] %@ = %.2fs", name, now.timeIntervalSince(tMark)))
            tMark = now
        }

        let intake = filterFrames(
            scene: scene, frameQualityFilter: frameQualityFilter,
            opticalFlowDiagnostics: opticalFlowDiagnostics, flowSoftWeight: flowSoftWeight, log: log)
        phaseMs("intake(outlier+quality+flow)")

        let workingScene = refinePoses(
            intake.scene, preBaScale: intake.preBaScale,
            orbFastThreshold: orbFastThreshold, orbMaxKeypointsPerFrame: orbMaxKeypointsPerFrame,
            // Add multi-view tracks from ARKit feature-point IDs to the BA observations
            // (CaptureBA.buildARKitTracks).
            useARKitTracks: true, lensDistortion: lensDistortion, log: log)
        phaseMs("BA+geomrefine(ORB/match/loop/triangulate/BA/undistort)")
        if let poseDumpURL {
            writePoseDump(before: intake.scene, after: workingScene, to: poseDumpURL, log: log)
        }

        // From here MVS reads each frame once (at 1/4 size) and training reads only its views.
        // Release the captured images a saved dataset backs (a .valset; the app saves one before
        // training) so the full-size frames are not resident through MVS; MVS, the training views,
        // held-out eval and so on reload them from disk.
        let released = scene.releaseSavedImages()
        if released > 0 {
            log("released \(released) captured image(s) from memory to the saved dataset")
        }

        let sceneScale = sceneScale(workingScene.frames)
        let farClip = SeedFarClip(frames: workingScene.frames, sceneScale: sceneScale, factor: seedFarClipFactor)
        // The depth maps go to disk as MVS produces them (for meshing after training).
        var depthMaps: DepthMapFile?
        do {
            depthMaps = try DepthMapFile()
        } catch {
            log("[warn] could not create a file for the MVS depth maps (\(error)); meshing is unavailable")
        }
        let cloud = mvsSeedCloud(
            scene: workingScene, parameters: mvsParameters, downscale: mvsDownscale, farClip: farClip,
            depthMaps: depthMaps.map { file in { file.append($0) } }, log: log)
        if let file = depthMaps {
            do {
                try file.finishWriting()
                log(String(format: "MVS depth maps (%d views, %.0f MB) kept on disk for meshing",
                           file.count, Double(file.byteCount) / 1_048_576))
            } catch {
                log("[warn] could not write the MVS depth maps (\(error)); meshing is unavailable")
                depthMaps = nil
            }
        }
        log(String(format: "[timing] === geometry total (intake+BA+MVS) = %.2fs ===",
                   Date().timeIntervalSince(t0)))
        return CaptureGeometry(
            capturedFrameCount: scene.frames.count, scene: workingScene,
            weightByTimestamp: intake.weightByTimestamp,
            sceneScale: sceneScale, mvsVoxelSize: mvsParameters.voxelDedupSize, seedCloud: cloud,
            depthMaps: depthMaps)
    }
}
