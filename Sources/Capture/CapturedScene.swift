// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// In-memory scene built from an ARKit capture (counterpart of `ColmapScene`). Each frame holds its
// ARAnchor ID so post-loop-closure pose updates can be applied.

import Foundation
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

/// Capture result for one frame. `intrinsics` are in pixels at the image resolution. Not Sendable
/// because MLXArray is not thread-safe. Synchronize on the constructing side when crossing threads.
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

    public init(
        image: MLXArray, viewmat: [Float],
        intrinsics: Intrinsics,
        width: Int, height: Int,
        timestamp: TimeInterval, anchorIdentifier: UUID? = nil,
        featureObservations: [ARKitFeatureObservation] = []
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
/// - `lidarPoints`: world points unprojected from sceneDepth. Empty on devices without LiDAR.
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
