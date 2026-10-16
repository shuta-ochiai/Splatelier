// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// CaptureTraining.filterPoseOutliers: guards against a single frame's anchor.transform turning into
// garbage on the order of 1e7 and blowing up the scene scale.

import Foundation
import MLX
import simd
import Testing
@testable import Capture

/// w2c viewmat for camera position pos (row-major, R = I, t = -pos).
private func viewmatAtPosition(_ pos: SIMD3<Float>) -> [Float] {
    return [
        1, 0, 0, -pos.x,
        0, 1, 0, -pos.y,
        0, 0, 1, -pos.z,
        0, 0, 0,  1,
    ]
}

private func makeFrame(pos: SIMD3<Float>) -> CapturedFrame {
    CapturedFrame(
        image: MLXArray([Float](repeating: 0, count: 2 * 2 * 3), [2, 2, 3]),
        viewmat: viewmatAtPosition(pos),
        intrinsics: (fx: 100, fy: 100, cx: 1, cy: 1),
        width: 2, height: 2, timestamp: 0,
        anchorIdentifier: UUID()
    )
}

@Test
func poseFilterPassesAllInRangeFrames() {
    // 5 frames within [0, 2 m] (a normal room).
    let positions: [SIMD3<Float>] = [
        SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(0, 1, 0),
        SIMD3(2, 0.5, 1), SIMD3(0.5, -0.5, 1.5),
    ]
    let scene = CapturedScene(
        frames: positions.map { makeFrame(pos: $0) },
        sparsePoints: []
    )
    let filtered = CaptureTraining.filterPoseOutliers(scene: scene) { _ in }
    #expect(filtered.frames.count == 5)
}

@Test
func poseFilterDropsSingleGiantOutlier() {
    // 5 normal + 1 corrupt (on the order of 1e7 m).
    let positions: [SIMD3<Float>] = [
        SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(0, 1, 0),
        SIMD3(2, 0.5, 1), SIMD3(0.5, -0.5, 1.5),
        SIMD3(2.4e7, 0, 0),                            // Corrupt anchor
    ]
    let scene = CapturedScene(
        frames: positions.map { makeFrame(pos: $0) },
        sparsePoints: []
    )
    let filtered = CaptureTraining.filterPoseOutliers(scene: scene) { _ in }
    #expect(filtered.frames.count == 5, "should drop the 1e7 outlier")
    // The remaining frames are the original first 5
    let resultCenters = CaptureTraining.cameraCenters(filtered.frames)
    for c in resultCenters {
        #expect(abs(c.x) < 100 && abs(c.y) < 100 && abs(c.z) < 100)
    }
}

@Test
func poseFilterDropsMultipleOutliers() {
    // 4 normal + 2 distant outliers. Both are dropped.
    let positions: [SIMD3<Float>] = [
        SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(1, 1, 0),
        SIMD3(0, 0, 1e6),
        SIMD3(-1e7, 0, 0),
    ]
    let scene = CapturedScene(
        frames: positions.map { makeFrame(pos: $0) },
        sparsePoints: []
    )
    let filtered = CaptureTraining.filterPoseOutliers(scene: scene) { _ in }
    #expect(filtered.frames.count == 4)
}

@Test
func poseFilterKeepsScalesLessThanThreshold() {
    // 5 frames spread over about 30 m (a large room). Distance from the median position is within
    // the 50 m absolute threshold, so none are dropped.
    let positions: [SIMD3<Float>] = [
        SIMD3(0, 0, 0), SIMD3(30, 0, 0), SIMD3(0, 30, 0),
        SIMD3(15, 15, 15), SIMD3(-10, -10, -10),
    ]
    let scene = CapturedScene(
        frames: positions.map { makeFrame(pos: $0) },
        sparsePoints: []
    )
    let filtered = CaptureTraining.filterPoseOutliers(scene: scene) { _ in }
    #expect(filtered.frames.count == 5)
}

@Test
func poseFilterEmptyAndSmallScenes() {
    let empty = CapturedScene(frames: [], sparsePoints: [])
    let filteredEmpty = CaptureTraining.filterPoseOutliers(scene: empty) { _ in }
    #expect(filteredEmpty.frames.count == 0)

    // With < 3 frames the median is undefined, so pass through
    let two = CapturedScene(
        frames: [makeFrame(pos: SIMD3(0, 0, 0)),
                 makeFrame(pos: SIMD3(1e7, 0, 0))],
        sparsePoints: []
    )
    let filteredTwo = CaptureTraining.filterPoseOutliers(scene: two) { _ in }
    #expect(filteredTwo.frames.count == 2,
            "filter requires >= 3 frames for median; smaller scenes pass through")
}
