// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import Trainer
import simd
import Testing
@testable import Capture

private func datasetFrame(index: Int, width: Int, height: Int) -> CapturedFrame {
    // CapturedFrame.image is uint8 [H,W,3].
    var pixels = [UInt8](repeating: 0, count: width * height * 3)
    for i in 0..<(width * height) {
        pixels[i * 3 + 0] = UInt8((i * 7 + index * 3) % 256)
        pixels[i * 3 + 1] = UInt8((i * 13 + index) % 256)
        pixels[i * 3 + 2] = UInt8((i * 5 + index * 11) % 256)
    }
    return CapturedFrame(
        image: MLXArray(pixels, [height, width, 3]),
        viewmat: [
            1, 0, 0, -Float(index) * 0.05,
            0, 1, 0, 0.1,
            0, 0, 1, 0.2,
            0, 0, 0, 1,
        ],
        intrinsics: (fx: 123.5, fy: 124.5, cx: Float(width) / 2, cy: Float(height) / 2),
        width: width, height: height,
        timestamp: TimeInterval(index) + 0.25,
        anchorIdentifier: UUID()
    )
}

@Test
func datasetStoreRoundTrips() throws {
    let w = 6, h = 4
    let scene = CapturedScene(
        frames: [datasetFrame(index: 0, width: w, height: h),
                 datasetFrame(index: 1, width: w, height: h),
                 datasetFrame(index: 2, width: w, height: h)],
        sparsePoints: [SIMD3<Float>(0.1, 0.2, 0.3), SIMD3<Float>(-1, 2, -3)],
        baRefinedPoints: [SIMD3<Float>(0.5, 0.5, 0.5)],
        lidarPoints: [],
        diagnostics: nil
    )

    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("dataset_rt_\(UUID().uuidString)")
        .appendingPathComponent("capture_test.dataset")
    defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }

    try CaptureDatasetStore.save(scene, to: dir)
    #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("manifest.json").path))
    #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("frames/00000.rgb").path))

    let loaded = try CaptureDatasetStore.load(from: dir)
    #expect(loaded.frames.count == 3)
    #expect(loaded.sparsePoints.count == 2)
    #expect(loaded.baRefinedPoints?.count == 1)
    #expect(loaded.lidarPoints.isEmpty)

    // pose / intrinsics / size / points match exactly.
    let f0 = loaded.frames[0]
    #expect(f0.width == w && f0.height == h)
    #expect(f0.viewmat == scene.frames[0].viewmat)
    #expect(f0.intrinsics.fx == 123.5 && f0.intrinsics.cy == Float(h) / 2)
    #expect(abs(f0.timestamp - 0.25) < 1e-6)
    #expect(f0.anchorIdentifier == scene.frames[0].anchorIdentifier)

    #expect(loaded.sparsePoints[0] == SIMD3<Float>(0.1, 0.2, 0.3))
    #expect(loaded.baRefinedPoints?[0] == SIMD3<Float>(0.5, 0.5, 0.5))

    // uint8 raw writes the bytes as is, so it is fully lossless.
    let orig = scene.frames[1].image.asArray(UInt8.self)
    let back = loaded.frames[1].image.asArray(UInt8.self)
    #expect(orig.count == back.count)
    #expect(orig == back, "uint8 image round-trip must be exact")
}

/// The camera's exposure and white balance round-trip, a frame without them stays nil, and the
/// intake summary covers only the frames that have them.
@Test
func datasetStoreRoundTripsCameraSettings() throws {
    let base = datasetFrame(index: 0, width: 4, height: 4)
    let settings = CameraSettings(exposureDuration: 1.0 / 120, exposureOffset: -0.25, iso: 320,
                                  whiteBalanceGains: [1.95, 1, 2.125], exposureLocked: true,
                                  whiteBalanceLocked: false)
    let withSettings = CapturedFrame(
        image: base.image, viewmat: base.viewmat, intrinsics: base.intrinsics,
        width: base.width, height: base.height, timestamp: base.timestamp,
        anchorIdentifier: base.anchorIdentifier, cameraSettings: settings
    )
    let scene = CapturedScene(
        frames: [withSettings, datasetFrame(index: 1, width: 4, height: 4)], sparsePoints: []
    )
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("dataset_camera_\(UUID().uuidString)")
    let dir = root.appendingPathComponent("capture_test.dataset")
    defer { try? FileManager.default.removeItem(at: root) }

    try CaptureDatasetStore.save(scene, to: dir)
    let loaded = try CaptureDatasetStore.load(from: dir)
    #expect(loaded.frames[0].cameraSettings == settings)
    #expect(loaded.frames[1].cameraSettings == nil)
    // A pose update keeps them.
    #expect(loaded.frames[0].withViewmat(base.viewmat).cameraSettings == settings)

    let summary = try #require(CameraSettings.summary(of: loaded.frames))
    #expect(summary.hasPrefix("[camera] 1/2 frames"), "\(summary)")
    #expect(summary.contains("ISO 320–320") && summary.contains("exposure locked 1/1")
            && summary.contains("WB locked 0/1"), "\(summary)")
    #expect(CameraSettings.summary(of: [base]) == nil)
}

/// A yaw of 0.02 rad and a 1 cm step to the right over 1/60 s give the expected velocities in the
/// camera's axes, and the blur in pixels follows from them.
@Test
func cameraMotionFromTwoPoses() throws {
    let dt = 1.0 / 60
    let yaw: Float = 0.02
    // A camera rotated 90° about world z, so world and camera axes differ.
    let base = simd_float4x4(simd_quatf(angle: .pi / 2, axis: SIMD3(0, 0, 1)))
    var next = base * simd_float4x4(simd_quatf(angle: yaw, axis: SIMD3(0, 1, 0)))
    let right = base * SIMD4<Float>(0.01, 0, 0, 0)     // 1 cm along the camera's x, in world
    next.columns.3 = base.columns.3 + right
    let m = try #require(CameraMotion(from: base, at: 10, to: next, at: 10 + dt))
    #expect(abs(m.angularVelocity[1] - yaw / Float(dt)) < 1e-3, "\(m.angularVelocity)")
    #expect(abs(m.angularVelocity[0]) < 1e-4 && abs(m.angularVelocity[2]) < 1e-4)
    // The translation is read in the second camera's axes, turned by the yaw.
    #expect(abs(m.linearVelocity[0] - 0.01 * cos(yaw) / Float(dt)) < 1e-3, "\(m.linearVelocity)")
    #expect(abs(m.interval - dt) < 1e-9)
    // 1.2 rad/s for 4 ms at 800 px is 3.84 px; 0.6 m/s at 1 m depth is 1.92 px.
    #expect(abs(m.rotationalBlurPixels(exposure: 0.004, focal: 800) - 3.84) < 0.01)
    #expect(abs(m.translationalBlurPixels(exposure: 0.004, focal: 800, depth: 1) - 1.92) < 0.01)
    #expect(CameraMotion(from: base, at: 10, to: next, at: 10) == nil)
}

/// The motion round-trips through the dataset and the intake summary reports the blur.
@Test
func datasetStoreRoundTripsCameraMotion() throws {
    let base = datasetFrame(index: 0, width: 4, height: 4)
    let motion = CameraMotion(angularVelocity: [0, 1.25, 0.5], linearVelocity: [0.25, 0, -0.5],
                              interval: 1.0 / 60)
    let settings = CameraSettings(exposureDuration: 0.004, exposureOffset: 0)
    let frame = CapturedFrame(
        image: base.image, viewmat: base.viewmat, intrinsics: base.intrinsics,
        width: base.width, height: base.height, timestamp: base.timestamp,
        anchorIdentifier: base.anchorIdentifier, cameraSettings: settings, cameraMotion: motion
    )
    let scene = CapturedScene(frames: [frame, datasetFrame(index: 1, width: 4, height: 4)], sparsePoints: [])
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("dataset_motion_\(UUID().uuidString)")
    let dir = root.appendingPathComponent("capture_test.dataset")
    defer { try? FileManager.default.removeItem(at: root) }

    try CaptureDatasetStore.save(scene, to: dir)
    let loaded = try CaptureDatasetStore.load(from: dir)
    #expect(loaded.frames[0].cameraMotion == motion)
    #expect(loaded.frames[1].cameraMotion == nil)
    #expect(loaded.frames[0].withViewmat(base.viewmat).cameraMotion == motion)

    // 1.25 rad/s × 4 ms × fx 123.5 = 0.62 px.
    let summary = try #require(CameraMotion.summary(of: loaded.frames))
    #expect(summary.hasPrefix("[motion] 1/2 frames"), "\(summary)")
    #expect(summary.contains("rotational blur median 0.62"), "\(summary)")
    #expect(CameraMotion.summary(of: [base]) == nil)
}

/// The streamed capture-time pose round-trips through the dataset and survives a pose change;
/// frames without one load without it.
@Test
func datasetStoreRoundTripsCaptureViewmat() throws {
    let base = datasetFrame(index: 0, width: 4, height: 4)
    var stream = base.viewmat
    stream[3] += 0.1   // the anchor moved 10 cm after the frame was taken
    let frame = CapturedFrame(
        image: base.image, viewmat: base.viewmat, intrinsics: base.intrinsics,
        width: base.width, height: base.height, timestamp: base.timestamp,
        anchorIdentifier: base.anchorIdentifier, captureViewmat: stream
    )
    let scene = CapturedScene(frames: [frame, datasetFrame(index: 1, width: 4, height: 4)], sparsePoints: [])
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("dataset_stream_\(UUID().uuidString)")
    let dir = root.appendingPathComponent("capture_test.dataset")
    defer { try? FileManager.default.removeItem(at: root) }

    try CaptureDatasetStore.save(scene, to: dir)
    let loaded = try CaptureDatasetStore.load(from: dir)
    #expect(loaded.frames[0].captureViewmat == stream)
    #expect(loaded.frames[0].viewmat == base.viewmat)
    #expect(loaded.frames[1].captureViewmat == nil)
    #expect(loaded.frames[0].withViewmat(stream).captureViewmat == stream)
}

@Test
func datasetStoreRoundTripsFeatureObservations() throws {
    // ARKit feature point observation IDs (including UInt64 beyond JSON number precision) and
    // camera coordinates round-trip. Frames without observations stay empty (same path as a dataset
    // with no observations).
    let base = datasetFrame(index: 0, width: 4, height: 4)
    let observations = [
        ARKitFeatureObservation(trackID: UInt64.max - 3, cameraPoint: SIMD3<Float>(0.1, -0.2, 1.5)),
        ARKitFeatureObservation(trackID: 42, cameraPoint: SIMD3<Float>(-0.3, 0.4, 2.25)),
    ]
    let withObs = CapturedFrame(
        image: base.image, viewmat: base.viewmat, intrinsics: base.intrinsics,
        width: base.width, height: base.height, timestamp: base.timestamp,
        anchorIdentifier: base.anchorIdentifier, featureObservations: observations
    )
    let scene = CapturedScene(
        frames: [withObs, datasetFrame(index: 1, width: 4, height: 4)], sparsePoints: []
    )
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("dataset_feat_\(UUID().uuidString)")
    let dir = root.appendingPathComponent("capture_test.dataset")
    defer { try? FileManager.default.removeItem(at: root) }

    try CaptureDatasetStore.save(scene, to: dir)
    let loaded = try CaptureDatasetStore.load(from: dir)
    #expect(loaded.frames[0].featureObservations == observations)
    #expect(loaded.frames[1].featureObservations.isEmpty)
}

@Test
func datasetStoreRoundTripsLiDARSamplesAndMovesThemWithThePose() throws {
    // Per-frame LiDAR samples round-trip, and lidarWorldPoints moves them with each frame's
    // current pose, so a pose change (BA) moves them too.
    let base = datasetFrame(index: 0, width: 4, height: 4)
    let samples = [LiDARSample(cameraPoint: SIMD3(0.1, -0.2, 1.5), confidenceLevel: 2, depthSpread: 0.01),
                   LiDARSample(cameraPoint: SIMD3(-0.3, 0.4, 2.25), confidenceLevel: 1, depthSpread: 0)]
    let withLiDAR = CapturedFrame(
        image: base.image, viewmat: base.viewmat, intrinsics: base.intrinsics,
        width: base.width, height: base.height, timestamp: base.timestamp,
        anchorIdentifier: base.anchorIdentifier, lidarSamples: samples
    )
    let scene = CapturedScene(
        frames: [withLiDAR, datasetFrame(index: 1, width: 4, height: 4)], sparsePoints: []
    )
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("dataset_lidar_\(UUID().uuidString)")
    let dir = root.appendingPathComponent("capture_test.dataset")
    defer { try? FileManager.default.removeItem(at: root) }

    try CaptureDatasetStore.save(scene, to: dir)
    let loaded = try CaptureDatasetStore.load(from: dir)
    #expect(loaded.frames[0].lidarSamples == samples)
    #expect(loaded.frames[1].lidarSamples.isEmpty)

    // 90° about +Y, then translation: world = Rᵀ (p − t) inverts p = R·world + t.
    let world = SIMD3<Float>(0.5, -0.25, 2)
    let R = simd_float3x3(rows: [SIMD3(0, 0, -1), SIMD3(0, 1, 0), SIMD3(1, 0, 0)])
    let t = SIMD3<Float>(0.1, 0.2, 0.3)
    let p = R * world + t
    let moved = loaded.withFrames([CapturedFrame(
        image: base.image,
        viewmat: [0, 0, -1, t.x, 0, 1, 0, t.y, 1, 0, 0, t.z, 0, 0, 0, 1],
        intrinsics: base.intrinsics, width: base.width, height: base.height,
        timestamp: base.timestamp,
        lidarSamples: [LiDARSample(cameraPoint: p, confidenceLevel: 2, depthSpread: 0)]
    )])
    #expect(simd_distance(moved.lidarWorldPoints[0], world) < 1e-5)

    // Old datasets without per-frame samples fall back to the scene's world points.
    let old = CapturedScene(frames: [base], sparsePoints: [], lidarPoints: [world])
    #expect(old.lidarWorldPoints == [world])
}

@Test
func datasetStoreSaveIsAtomic() throws {
    // Overwriting an existing dataset replaces it cleanly (no .tmp left behind).
    let w = 4, h = 4
    let scene = CapturedScene(
        frames: [datasetFrame(index: 0, width: w, height: h)],
        sparsePoints: []
    )
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("dataset_atomic_\(UUID().uuidString)")
    let dir = root.appendingPathComponent("capture_test.dataset")
    defer { try? FileManager.default.removeItem(at: root) }

    try CaptureDatasetStore.save(scene, to: dir)
    try CaptureDatasetStore.save(scene, to: dir)  // Overwrite

    let entries = try FileManager.default.contentsOfDirectory(atPath: root.path)
    #expect(entries.contains("capture_test.dataset"))
    #expect(!entries.contains(where: { $0.hasPrefix(".tmp") }))
    let loaded = try CaptureDatasetStore.load(from: dir)
    #expect(loaded.frames.count == 1)
}

@Test
func savedImagesReleaseAndReloadExactly() throws {
    // After save, frames drop their in-memory images and read them back from the dataset; every
    // copy of a frame shares that. Undistorted frames (saved image differs from the captured one)
    // and unsaved frames stay in memory.
    let w = 6, h = 4
    let frames = (0..<3).map { datasetFrame(index: $0, width: w, height: h) }
    let originals = frames.map { $0.image.asArray(UInt8.self) }
    let undistorted = datasetFrame(index: 3, width: w, height: h)
        .undistorted(target: frames[0].intrinsics, distortion: RadialDistortion(k1: 0.01, k2: 0))
    let scene = CapturedScene(frames: frames + [undistorted], sparsePoints: [])
    let copy = scene.withFrames(scene.frames.map { $0.withViewmat($0.viewmat) })
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("dataset_release_\(UUID().uuidString)")
    let dir = root.appendingPathComponent("capture_test.dataset")
    defer { try? FileManager.default.removeItem(at: root) }

    #expect(scene.releaseSavedImages() == 0, "nothing is saved yet")
    try CaptureDatasetStore.save(scene, to: dir)
    #expect(scene.savedDataset != nil)
    #expect(copy.releaseSavedImages() == 3)
    #expect(scene.releaseSavedImages() == 0, "copies share the released storage")
    for i in 0..<3 {
        #expect(scene.frames[i].image.asArray(UInt8.self) == originals[i])
        #expect(copy.frames[i].originalImage.asArray(UInt8.self) == originals[i])
    }

    let loaded = try CaptureDatasetStore.load(from: dir)
    #expect(loaded.releaseSavedImages() == 4)
    #expect(loaded.frames[1].image.asArray(UInt8.self) == originals[1])
}

@Test
func retiredDatasetStaysReadableUntilFramesAreReleased() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("dataset_retire_\(UUID().uuidString)")
    let dir = root.appendingPathComponent("capture_test.dataset")
    defer { try? FileManager.default.removeItem(at: root) }
    let fm = FileManager.default
    let before = Set((try? fm.contentsOfDirectory(atPath: CaptureDatasetFiles.retiredRoot.path)) ?? [])

    var scene: CapturedScene? = CapturedScene(
        frames: [datasetFrame(index: 0, width: 4, height: 4)], sparsePoints: [])
    let original = scene!.frames[0].image.asArray(UInt8.self)
    try CaptureDatasetStore.save(scene!, to: dir).retire()
    scene!.releaseSavedImages()

    // Out of its original place at once, still readable from where it moved.
    #expect(!fm.fileExists(atPath: dir.path))
    let after = Set((try? fm.contentsOfDirectory(atPath: CaptureDatasetFiles.retiredRoot.path)) ?? [])
    let moved = after.subtracting(before)
    #expect(moved.count == 1)
    #expect(scene!.frames[0].image.asArray(UInt8.self) == original)

    // Deleted with the last frame that reads from it.
    scene = nil
    let movedURL = CaptureDatasetFiles.retiredRoot.appendingPathComponent(moved.first ?? "-")
    #expect(!fm.fileExists(atPath: movedURL.path))
}
