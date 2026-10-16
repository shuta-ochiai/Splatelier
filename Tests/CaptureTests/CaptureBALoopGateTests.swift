// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Depth gate for loop matches (in scenes with a repeated pattern, discards mismatches to the
// neighboring copy of the pattern on the epipolar line).

import Foundation
import Testing
import simd
import MLX
@testable import Trainer
@testable import Capture

private let gateK = (fx: Float(500), fy: Float(500), cx: Float(320), cy: Float(240))

/// Camera with identity rotation and center (x, 0, 0) (+Z forward).
private func gateFrame(x: Float) -> CapturedFrame {
    CapturedFrame(
        image: MLXArray.zeros([2, 2, 3]),
        viewmat: [1, 0, 0, -x, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        intrinsics: gateK, width: 640, height: 480, timestamp: 0
    )
}

/// Integer keypoint of a point projected into the camera centered at (cx, 0, 0) (pixel center +0.5
/// convention).
private func gateKeypoint(_ p: SIMD3<Float>, camX: Float) -> (x: Int, y: Int) {
    let u = gateK.fx * (p.x - camX) / p.z + gateK.cx
    let v = gateK.fy * p.y / p.z + gateK.cy
    return (Int((u - 0.5).rounded()), Int((v - 0.5).rounded()))
}

private func gateFeatures(_ coords: [(x: Int, y: Int)]) -> ORBPipeline.FrameFeatures {
    ORBPipeline.FrameFeatures(coords: coords, descriptors: MLXArray.zeros([coords.count, 8], dtype: .int32))
}

/// A capture moving sideways along a wall (z = 2 m). Frames 0 and 1 are consecutive (8 cm
/// baseline); frame 2 is the loop partner 50 cm further. Frame 2 has another copy of the same
/// pattern 60 cm to the right (appearing 150 px off along the epipolar line); matches to that copy
/// are discarded by the depth gate and correct matches are kept.
@Test
func loopDepthGateRejectsRepeatedPatternOnEpipolarLine() {
    let frames = [gateFrame(x: 0), gateFrame(x: 0.08), gateFrame(x: 0.5)]
    let wall: [SIMD3<Float>] = [
        SIMD3(0.30, 0.10, 2), SIMD3(0.40, -0.20, 2), SIMD3(0.55, 0.05, 2),
        SIMD3(0.25, -0.05, 2), SIMD3(0.50, 0.20, 2),
    ]
    let dup = SIMD3<Float>(0.6, 0, 0)
    let f0 = gateFeatures(wall.map { gateKeypoint($0, camX: 0) })
    let f1 = gateFeatures(wall.map { gateKeypoint($0, camX: 0.08) })
    // frame 2: [correct position × 5 | neighboring copy of the pattern × 5]
    let f2 = gateFeatures(wall.map { gateKeypoint($0, camX: 0.5) } + wall.map { gateKeypoint($0 + dup, camX: 0.5) })
    let features = [f0, f1, f2]

    let consecutive = wall.indices.map { (a: 0, b: 1, ia: $0, ib: $0) }
    let depths = CaptureBA.keypointDepths(matches: consecutive, features: features, frames: frames)
    for k in wall.indices {
        let d = try! #require(depths[0][k])
        #expect(abs(1 / d.rho - 2) < 0.15)
    }
    #expect(depths[2].isEmpty)

    let correct = wall.indices.map { ORBPipeline.Match(queryIdx: $0, trainIdx: $0, distance: 0) }
    let repeated = wall.indices.map { ORBPipeline.Match(queryIdx: $0, trainIdx: wall.count + $0, distance: 0) }
    var stats = CaptureBA.LoopGateStats()
    let (kept, rejected) = CaptureBA.loopDepthGate(
        matches: correct + repeated, a: 0, b: 2, features: features, depths: depths,
        frames: frames, radius: 25, stats: &stats)
    #expect(kept == correct)
    #expect(rejected == repeated.count)
    #expect(stats.noDepth == 0)

    // The reverse direction (no depth on side a) predicts from side b's depth.
    var stats2 = CaptureBA.LoopGateStats()
    let reversed = (correct + repeated).map { ORBPipeline.Match(queryIdx: $0.trainIdx, trainIdx: $0.queryIdx, distance: 0) }
    let (kept2, rejected2) = CaptureBA.loopDepthGate(
        matches: reversed, a: 2, b: 0, features: features, depths: depths,
        frames: frames, radius: 25, stats: &stats2)
    #expect(kept2.count == correct.count && rejected2 == repeated.count)
}

/// Matches whose keypoint has no depth are checked against the frame's depth range: with the wall's
/// range (z = 2 m), correct matches are kept and matches to the same pattern 60 cm over (150 px
/// along the epipolar line) are discarded.
@Test
func loopDepthGateFallsBackToFrameDepthRange() {
    let frames = [gateFrame(x: 0), gateFrame(x: 0.5)]
    let p = SIMD3<Float>(0.3, 0.1, 2)
    let features = [
        gateFeatures([gateKeypoint(p, camX: 0)]),
        gateFeatures([gateKeypoint(p, camX: 0.5), gateKeypoint(p + SIMD3(0.6, 0, 0), camX: 0.5)]),
    ]
    // Depth from frame 0's other keypoints (1.9-2.1 m) → range 1.52-2.63 m.
    var known: [Int: CaptureBA.KeypointDepth] = [:]
    for k in 0..<40 { known[100 + k] = CaptureBA.KeypointDepth(rho: 1 / (1.9 + 0.2 * Float(k) / 39), sigma: 0.01) }
    let ranges = CaptureBA.frameDepthRanges([known, [:]])
    #expect(ranges[0] != nil && ranges[1] == nil)
    var stats = CaptureBA.LoopGateStats()
    let correct = ORBPipeline.Match(queryIdx: 0, trainIdx: 0, distance: 0)
    let repeated = ORBPipeline.Match(queryIdx: 0, trainIdx: 1, distance: 0)
    let (kept, rejected) = CaptureBA.loopDepthGate(
        matches: [correct, repeated], a: 0, b: 1, features: features, depths: [known, [:]],
        frameRanges: ranges, frames: frames, radius: 25, stats: &stats)
    #expect(kept == [correct] && rejected == 1)
    #expect(stats.rangeOnly == 2 && stats.noDepth == 0)
}

/// A match where neither keypoint has depth and there is no frame depth range cannot be checked, so
/// it is discarded.
@Test
func loopDepthGateDropsMatchesWithoutDepth() {
    let frames = [gateFrame(x: 0), gateFrame(x: 0.5)]
    let p = SIMD3<Float>(0.3, 0.1, 2)
    let features = [gateFeatures([gateKeypoint(p, camX: 0)]), gateFeatures([gateKeypoint(p, camX: 0.5)])]
    var stats = CaptureBA.LoopGateStats()
    let (kept, rejected) = CaptureBA.loopDepthGate(
        matches: [ORBPipeline.Match(queryIdx: 0, trainIdx: 0, distance: 0)], a: 0, b: 1,
        features: features, depths: [[:], [:]], frames: frames, radius: 25, stats: &stats)
    #expect(kept.isEmpty && rejected == 0 && stats.noDepth == 1)
}
