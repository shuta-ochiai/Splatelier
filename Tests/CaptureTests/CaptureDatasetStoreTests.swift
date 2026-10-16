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
