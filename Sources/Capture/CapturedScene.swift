// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// In-memory scene built from an ARKit capture (counterpart of `ColmapScene`). Each frame holds its
// ARAnchor ID so post-loop-closure pose updates can be applied.

import Foundation
import simd
import MLX
import Trainer

/// One observation of an ARKit feature point (`rawFeaturePoints`). The same `trackID` is the same
/// 3D point across frames. `cameraPoint` is in OpenCV camera coordinates at the capture-time pose.
/// It is pose-independent, so it projects with intrinsics alone.
public struct ARKitFeatureObservation: Sendable, Equatable {
    public let trackID: UInt64
    public let cameraPoint: SIMD3<Float>
    public init(trackID: UInt64, cameraPoint: SIMD3<Float>) {
        self.trackID = trackID
        self.cameraPoint = cameraPoint
    }
}

/// Lens undistortion spec. The captured image has `distortion` under the `source` intrinsics.
public struct LensUndistortionSpec {
    public let source: Intrinsics
    public let distortion: RadialDistortion
    public init(source: Intrinsics, distortion: RadialDistortion) {
        self.source = source
        self.distortion = distortion
    }
}

/// The captured image of one frame, shared by every copy of the frame (`withViewmat`,
/// `undistorted`, and the scenes the app and training keep). Holds the image in memory and, once
/// the capture is saved (`CaptureDatasetStore`), the file it can be reloaded from. Releasing the
/// in-memory copy (`CapturedScene.releaseSavedImages`) affects every copy; later reads load the
/// file each time (not cached).
final class CapturedImageStorage: @unchecked Sendable {
    private let lock = NSLock()
    private var resident: MLXArray?
    private var saved: (dataset: CaptureDatasetFiles, path: String)?
    private let shape: [Int]

    init(image: MLXArray) {
        resident = image
        shape = image.shape
    }

    var image: MLXArray {
        lock.lock()
        defer { lock.unlock() }
        if let resident { return resident }
        guard let saved else { preconditionFailure("captured image released without a saved copy") }
        do {
            return try saved.dataset.readImage(saved.path, height: shape[0], width: shape[1])
        } catch {
            fatalError("could not reload captured image \(saved.path): \(error)")
        }
    }

    var savedDataset: CaptureDatasetFiles? {
        lock.lock()
        defer { lock.unlock() }
        return saved?.dataset
    }

    /// Records that `path` in `dataset` holds this image byte for byte (replaces an earlier one).
    func attach(_ dataset: CaptureDatasetFiles, path: String) {
        lock.lock()
        defer { lock.unlock() }
        saved = (dataset, path)
    }

    /// Drops the in-memory image if a saved file backs it. Returns whether it dropped one.
    func releaseIfSaved() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard resident != nil, saved != nil else { return false }
        resident = nil
        return true
    }
}

/// One recorded LiDAR depth sample of a frame.
public struct LiDARSample: Sendable, Equatable {
    /// OpenCV camera coordinates (m).
    public let cameraPoint: SIMD3<Float>
    /// ARConfidenceLevel raw value (0=low / 1=medium / 2=high).
    public let confidenceLevel: UInt8
    /// Largest depth difference to the adjacent depth pixels (m); large at depth edges.
    public let depthSpread: Float

    public init(cameraPoint: SIMD3<Float>, confidenceLevel: UInt8, depthSpread: Float) {
        self.cameraPoint = cameraPoint
        self.confidenceLevel = confidenceLevel
        self.depthSpread = depthSpread
    }

    /// Fusion confidence in 0...1, the product of three terms in the manner of the ToF confidence
    /// of Marin et al. (ECCV 2016, "Reliable Fusion of ToF and Stereo Depth Driven by Confidence
    /// Measures"): ARKit's confidence level, a mixed-pixel term that falls to 0 as the spread to
    /// the neighbors reaches `edgeRelative` of the depth, and a range term falling to 0 at
    /// `maxRange`.
    public func confidence(edgeRelative: Float = 0.1, maxRange: Float = 5) -> Float {
        let z = cameraPoint.z
        guard z > 0 else { return 0 }
        let levelTerm: Float = confidenceLevel >= 2 ? 1 : confidenceLevel == 1 ? 0.5 : 0.25
        let edgeTerm = Swift.max(0, 1 - depthSpread / (edgeRelative * z))
        let rangeTerm = Swift.max(0, 1 - z / maxRange)
        return levelTerm * edgeTerm * rangeTerm
    }
}

/// Capture result for one frame. `intrinsics` are in pixels at the image resolution. Not Sendable
/// because MLXArray is not thread-safe. Synchronize on the constructing side when crossing threads.
/// The camera's exposure and white balance when a frame was taken, to check that they stayed locked
/// through a capture (the app locks them after a short warm-up). The exposure comes from ARKit;
/// the rest from the capture device, nil where it was not available.
public struct CameraSettings: Codable, Sendable, Equatable {
    /// Exposure time, seconds.
    public var exposureDuration: Double
    /// Exposure offset (EV) ARKit applied.
    public var exposureOffset: Float
    public var iso: Float?
    /// Device white balance gains, R G B.
    public var whiteBalanceGains: [Float]?
    public var exposureLocked: Bool?
    public var whiteBalanceLocked: Bool?

    public init(exposureDuration: Double, exposureOffset: Float, iso: Float? = nil,
                whiteBalanceGains: [Float]? = nil, exposureLocked: Bool? = nil,
                whiteBalanceLocked: Bool? = nil) {
        self.exposureDuration = exposureDuration
        self.exposureOffset = exposureOffset
        self.iso = iso
        self.whiteBalanceGains = whiteBalanceGains
        self.exposureLocked = exposureLocked
        self.whiteBalanceLocked = whiteBalanceLocked
    }

    /// One log line with the range of each setting over `frames` and how many were locked; nil
    /// when no frame has them.
    public static func summary(of frames: [CapturedFrame]) -> String? {
        let s = frames.compactMap(\.cameraSettings)
        guard !s.isEmpty else { return nil }
        func range(_ v: [Float], _ f: String) -> String {
            guard let lo = v.min(), let hi = v.max() else { return "-" }
            return String(format: "\(f)–\(f)", lo, hi)
        }
        let ms = s.map { Float($0.exposureDuration * 1000) }
        var line = "[camera] \(s.count)/\(frames.count) frames: exposure " + range(ms, "%.2f") + " ms"
            + ", offset " + range(s.map(\.exposureOffset), "%+.2f") + " EV"
        let iso = s.compactMap(\.iso)
        if !iso.isEmpty { line += ", ISO " + range(iso, "%.0f") }
        let wb = s.compactMap(\.whiteBalanceGains).filter { $0.count == 3 }
        if !wb.isEmpty {
            line += ", WB gains R " + range(wb.map { $0[0] }, "%.3f") + " G " + range(wb.map { $0[1] }, "%.3f")
                + " B " + range(wb.map { $0[2] }, "%.3f")
        }
        let ae = s.compactMap(\.exposureLocked), awb = s.compactMap(\.whiteBalanceLocked)
        if !ae.isEmpty { line += ", exposure locked \(ae.filter { $0 }.count)/\(ae.count)" }
        if !awb.isEmpty { line += ", WB locked \(awb.filter { $0 }.count)/\(awb.count)" }
        return line
    }
}

/// How fast the camera moved when a frame was taken, from the ARKit pose just before it (about
/// 1/60 s earlier). With the exposure duration this gives the motion blur in pixels whatever the
/// frame shows, which an image measure of sharpness cannot (a plain surface looks soft to it).
/// Velocities are in the ARKit camera's axes: x right, y up, z towards the viewer.
public struct CameraMotion: Codable, Sendable, Equatable {
    /// Angular velocity, rad/s.
    public var angularVelocity: [Float]
    /// Linear velocity, m/s.
    public var linearVelocity: [Float]
    /// Time between the two poses, seconds.
    public var interval: Double

    public init(angularVelocity: [Float], linearVelocity: [Float], interval: Double) {
        self.angularVelocity = angularVelocity
        self.linearVelocity = linearVelocity
        self.interval = interval
    }

    /// The motion from the camera-to-world pose `from` at `t0` to `to` at `t1`; nil when the
    /// interval is not positive.
    public init?(from: simd_float4x4, at t0: TimeInterval, to: simd_float4x4, at t1: TimeInterval) {
        let dt = t1 - t0
        guard dt > 0 else { return nil }
        func rotation(_ m: simd_float4x4) -> simd_float3x3 {
            simd_float3x3(SIMD3(m.columns.0.x, m.columns.0.y, m.columns.0.z),
                          SIMD3(m.columns.1.x, m.columns.1.y, m.columns.1.z),
                          SIMD3(m.columns.2.x, m.columns.2.y, m.columns.2.z))
        }
        let r0 = rotation(from), r1 = rotation(to)
        // The rotation from the first camera to the second, in the first camera's axes.
        let q = simd_quatf(r0.transpose * r1)
        let angle = q.angle
        let axis = angle > 1e-7 ? q.axis : SIMD3<Float>(0, 0, 0)
        let omega = axis * (angle > Float.pi ? angle - 2 * Float.pi : angle) / Float(dt)
        let world = SIMD3(to.columns.3.x - from.columns.3.x, to.columns.3.y - from.columns.3.y,
                          to.columns.3.z - from.columns.3.z) / Float(dt)
        let v = r1.transpose * world
        self.init(angularVelocity: [omega.x, omega.y, omega.z], linearVelocity: [v.x, v.y, v.z],
                  interval: dt)
    }

    /// Blur in pixels from turning the camera across the image (pitch and yaw) during `exposure`
    /// seconds, at focal length `focal` px. Roll is left out (it smears only towards the edges).
    public func rotationalBlurPixels(exposure: Double, focal: Float) -> Float {
        guard angularVelocity.count == 3 else { return 0 }
        return focal * Float(exposure) * (angularVelocity[0] * angularVelocity[0]
                                          + angularVelocity[1] * angularVelocity[1]).squareRoot()
    }

    /// Blur in pixels from moving across the image during `exposure` seconds, for a surface `depth`
    /// metres away.
    public func translationalBlurPixels(exposure: Double, focal: Float, depth: Float) -> Float {
        guard linearVelocity.count == 3, depth > 0 else { return 0 }
        return focal * Float(exposure) * (linearVelocity[0] * linearVelocity[0]
                                          + linearVelocity[1] * linearVelocity[1]).squareRoot() / depth
    }

    /// One log line with the angular speed and, where the exposure was recorded, the rotational
    /// blur over `frames`; nil when no frame has a motion.
    public static func summary(of frames: [CapturedFrame]) -> String? {
        let withMotion = frames.filter { $0.cameraMotion != nil }
        guard !withMotion.isEmpty else { return nil }
        func median(_ v: [Float]) -> Float { v.sorted()[v.count / 2] }
        func p90(_ v: [Float]) -> Float { v.sorted()[Swift.min(v.count - 1, v.count * 9 / 10)] }
        let speeds = withMotion.compactMap { $0.cameraMotion }.map { m -> Float in
            m.angularVelocity.count == 3 ? (m.angularVelocity.map { $0 * $0 }.reduce(0, +)).squareRoot() : 0
        }
        var line = String(format: "[motion] %d/%d frames: angular speed median %.2f p90 %.2f rad/s",
                          withMotion.count, frames.count, median(speeds), p90(speeds))
        let blur = withMotion.compactMap { f -> Float? in
            guard let m = f.cameraMotion, let s = f.cameraSettings else { return nil }
            return m.rotationalBlurPixels(exposure: s.exposureDuration, focal: f.intrinsics.fx)
        }
        if !blur.isEmpty {
            line += String(format: ", rotational blur median %.2f p90 %.2f px, over 2 px %d",
                           median(blur), p90(blur), blur.filter { $0 > 2 }.count)
        }
        return line
    }
}

public struct CapturedFrame {
    /// [H, W, 3] uint8 RGB as captured. Kept resident at 1/4 the memory of float32 until released
    /// to its saved copy.
    let imageStorage: CapturedImageStorage
    private var capturedImage: MLXArray { imageStorage.image }
    /// Undistortion spec. If present, `image` is built from `capturedImage` on every use.
    public let lensUndistortion: LensUndistortionSpec?
    /// [H, W, 3] uint8 RGB consistent with `intrinsics`. With an undistortion spec it is built on
    /// the GPU on every use and not cached (so it is not resident alongside the captured image).
    /// Call sites that use it repeatedly should read it once and reuse it.
    public var image: MLXArray {
        guard let spec = lensUndistortion else { return capturedImage }
        return LensUndistortion.undistortImage(
            capturedImage, source: spec.source, target: intrinsics, distortion: spec.distortion)
    }
    /// The image as captured (distorted) and its matching intrinsics. For evaluation against raw
    /// images as ground truth.
    public var originalImage: MLXArray { capturedImage }
    public var originalIntrinsics: Intrinsics { lensUndistortion?.source ?? intrinsics }
    /// float [0,1] version. Generated only at the moment of use, not cached. Float input is
    /// returned as is.
    public var imageFloat: MLXArray {
        let image = self.image
        return image.dtype == .uint8 ? image.asType(.float32) / 255.0 : image
    }
    public let viewmat: [Float]                                   // row-major 4×4 world→camera
    public let intrinsics: Intrinsics
    public let width: Int
    public let height: Int
    public let timestamp: TimeInterval
    /// ID used to rebuild the viewmat from the anchor's final transform in `stopCapture`.
    public let anchorIdentifier: UUID?
    /// ARKit feature observations. Used by `CaptureBA` to build multi-view tracks. Empty for
    /// non-ARKit paths.
    public let featureObservations: [ARKitFeatureObservation]
    /// LiDAR depth samples of this frame (empty without LiDAR). Moved to world with the frame's
    /// final pose, so they follow BA and loop closure.
    public let lidarSamples: [LiDARSample]
    /// The camera's exposure and white balance when the frame was taken (nil for frames not from
    /// an ARKit capture, and in datasets saved before it was recorded).
    public let cameraSettings: CameraSettings?
    /// How fast the camera moved when the frame was taken (nil for frames not from an ARKit
    /// capture, and in datasets saved before it was recorded).
    public let cameraMotion: CameraMotion?

    public init(
        image: MLXArray, viewmat: [Float],
        intrinsics: Intrinsics,
        width: Int, height: Int,
        timestamp: TimeInterval, anchorIdentifier: UUID? = nil,
        featureObservations: [ARKitFeatureObservation] = [],
        lidarSamples: [LiDARSample] = [],
        cameraSettings: CameraSettings? = nil,
        cameraMotion: CameraMotion? = nil
    ) {
        precondition(viewmat.count == 16, "viewmat must be a row-major 4×4")
        precondition(image.shape.count == 3 && image.shape[2] == 3,
                     "image must be [H, W, 3]")
        self.imageStorage = CapturedImageStorage(image: image)
        self.lensUndistortion = nil
        self.viewmat = viewmat
        self.intrinsics = intrinsics
        self.width = width
        self.height = height
        self.timestamp = timestamp
        self.anchorIdentifier = anchorIdentifier
        self.featureObservations = featureObservations
        self.lidarSamples = lidarSamples
        self.cameraSettings = cameraSettings
        self.cameraMotion = cameraMotion
    }

    private init(copying f: CapturedFrame, viewmat: [Float], intrinsics: Intrinsics,
                 lensUndistortion: LensUndistortionSpec?) {
        self.imageStorage = f.imageStorage
        self.lensUndistortion = lensUndistortion
        self.viewmat = viewmat
        self.intrinsics = intrinsics
        self.width = f.width
        self.height = f.height
        self.timestamp = f.timestamp
        self.anchorIdentifier = f.anchorIdentifier
        self.featureObservations = f.featureObservations
        self.lidarSamples = f.lidarSamples
        self.cameraSettings = f.cameraSettings
        self.cameraMotion = f.cameraMotion
    }

    /// Frame with only the pose replaced (image and undistortion spec unchanged).
    public func withViewmat(_ viewmat: [Float]) -> CapturedFrame {
        precondition(viewmat.count == 16, "viewmat must be a row-major 4×4")
        return CapturedFrame(copying: self, viewmat: viewmat, intrinsics: intrinsics,
                             lensUndistortion: lensUndistortion)
    }

    /// Frame that views the captured image undistorted by `distortion` with the `target`
    /// intrinsics. Holds only the spec, not an image.
    public func undistorted(target: Intrinsics, distortion: RadialDistortion) -> CapturedFrame {
        precondition(lensUndistortion == nil, "frame is already undistorted")
        return CapturedFrame(
            copying: self, viewmat: viewmat, intrinsics: target,
            lensUndistortion: LensUndistortionSpec(source: intrinsics, distortion: distortion))
    }
}

/// Final output of a capture session. Three point clouds:
/// - `sparsePoints`: BA-refined ORB points + raw ARKit points (only raw ARKit if BA was not run).
/// - `baRefinedPoints`: only BA points that passed IGR verification. Excludes wide-area raw ARKit
///   points such as floor and walls. nil if BA was not run.
/// - `lidarPoints`: world points unprojected from sceneDepth with the capture-time ARKit poses. Only
///   in datasets saved before the frames carried `lidarSamples`; empty otherwise.
public struct CapturedScene {
    public let frames: [CapturedFrame]
    public let sparsePoints: [SIMD3<Float>]
    public let baRefinedPoints: [SIMD3<Float>]?
    public let lidarPoints: [SIMD3<Float>]
    /// Capture-time diagnostic counters (ARKit / LiDAR). Non-nil only for scenes from capture.
    public let diagnostics: CaptureDiagnostics?

    public init(
        frames: [CapturedFrame],
        sparsePoints: [SIMD3<Float>],
        baRefinedPoints: [SIMD3<Float>]? = nil,
        lidarPoints: [SIMD3<Float>] = [],
        diagnostics: CaptureDiagnostics? = nil
    ) {
        self.frames = frames
        self.sparsePoints = sparsePoints
        self.baRefinedPoints = baRefinedPoints
        self.lidarPoints = lidarPoints
        self.diagnostics = diagnostics
    }

    /// Scene with only the frames replaced (point clouds and diagnostics unchanged).
    public func withFrames(_ frames: [CapturedFrame]) -> CapturedScene {
        CapturedScene(frames: frames, sparsePoints: sparsePoints, baRefinedPoints: baRefinedPoints,
                      lidarPoints: lidarPoints, diagnostics: diagnostics)
    }

    /// A LiDAR sample in world coordinates with its fusion confidence, the depth it was seen at
    /// and its color (0..1; mid-grey unless taken from the image).
    public typealias WorldLiDARSample = (position: SIMD3<Float>, confidence: Float, depth: Float, color: SIMD3<Float>)

    /// LiDAR samples in world coordinates, moved with each frame's current pose (the BA pose once
    /// the scene is refined). Datasets saved before frames carried their samples fall back to
    /// `lidarPoints` at medium confidence, seen from 1 m.
    public var lidarWorldSamples: [WorldLiDARSample] { lidarWorldSamples(colors: false) }

    /// `lidarWorldSamples`; with `colors`, each sample takes the color of the pixel it projects to
    /// in its frame's image (reading every frame image once).
    public func lidarWorldSamples(colors: Bool) -> [WorldLiDARSample] {
        let grey = SIMD3<Float>(repeating: 0.5)
        guard frames.contains(where: { !$0.lidarSamples.isEmpty }) else {
            let medium = LiDARSample(cameraPoint: SIMD3(0, 0, 1), confidenceLevel: 1, depthSpread: 0)
            return lidarPoints.map { ($0, medium.confidence(), 1, grey) }
        }
        var out: [WorldLiDARSample] = []
        out.reserveCapacity(frames.reduce(0) { $0 + $1.lidarSamples.count })
        for f in frames where !f.lidarSamples.isEmpty {
            // viewmat = [R | t] (world → camera), so world = Rᵀ (p − t)
            let v = f.viewmat
            let rt = simd_float3x3(rows: [
                SIMD3(v[0], v[4], v[8]), SIMD3(v[1], v[5], v[9]), SIMD3(v[2], v[6], v[10]),
            ])
            let t = SIMD3<Float>(v[3], v[7], v[11])
            var rgb = [SIMD3<Float>](repeating: grey, count: f.lidarSamples.count)
            if colors {
                // Pixel each sample projects to with the frame's (undistorted) intrinsics.
                let k = f.intrinsics
                let pix = f.lidarSamples.map { s -> Int32 in
                    let p = s.cameraPoint
                    let u = Swift.min(f.width - 1, Swift.max(0, Int(k.fx * p.x / p.z + k.cx)))
                    let vv = Swift.min(f.height - 1, Swift.max(0, Int(k.fy * p.y / p.z + k.cy)))
                    return Int32(vv * f.width + u)
                }
                let img = f.image.reshaped([f.width * f.height, 3])
                let picked = img.take(MLXArray(pix), axis: 0).asType(.float32).asArray(Float.self)
                for i in 0..<rgb.count { rgb[i] = SIMD3(picked[i * 3], picked[i * 3 + 1], picked[i * 3 + 2]) / 255 }
            }
            for (i, s) in f.lidarSamples.enumerated() {
                out.append((rt * (s.cameraPoint - t), s.confidence(), s.cameraPoint.z, rgb[i]))
            }
        }
        return out
    }

    /// Positions of `lidarWorldSamples`.
    public var lidarWorldPoints: [SIMD3<Float>] { lidarWorldSamples.map(\.position) }

    /// The saved dataset the frame images reload from (`CaptureDatasetStore.save` / `load`), nil
    /// while they exist only in memory.
    public var savedDataset: CaptureDatasetFiles? { frames.first?.imageStorage.savedDataset }

    /// Drops the in-memory copy of every image a saved dataset backs; later reads load it from
    /// disk. Affects every copy of these frames, including scenes other code keeps. Returns the
    /// number of images released.
    @discardableResult
    public func releaseSavedImages() -> Int {
        frames.reduce(0) { $0 + ($1.imageStorage.releaseIfSaved() ? 1 : 0) }
    }
}
