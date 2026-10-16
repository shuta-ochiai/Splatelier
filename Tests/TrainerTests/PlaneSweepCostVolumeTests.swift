// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Plane Sweep Cost Volume (PocketGS Operator G). Checked with known relative poses and random
// textures.

import Foundation
import MLX
import Testing
@testable import Trainer

/// Census transform of a random texture.
private func randomTexturedCensus(width: Int, height: Int, seed: UInt32) -> MLXArray {
    var lcg = TestLCG(seed: seed)
    var pixels: [Float] = []
    for _ in 0..<(height * width) {
        pixels.append(lcg.next())
    }
    let luma = MLXArray(pixels, [height, width])
    return CensusTransform.compute(luma: luma)
}

// MARK: - relativePose

/// Identical viewmats give an identity relative pose (R=I, t=0).
@Test
func relativePoseIdentity() {
    let vm: [Float] = [
        1, 0, 0, 2,
        0, 1, 0, 3,
        0, 0, 1, 4,
        0, 0, 0, 1,
    ]
    let rt = PlaneSweepCostVolume.relativePose(targetViewmat: vm, refViewmat: vm)
    // R = I
    let eye: [Float] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    for i in 0..<9 {
        #expect(abs(rt[i] - eye[i]) < 1e-5,
            "R[\(i)] = \(rt[i]) (expected \(eye[i]))")
    }
    // t = 0
    for i in 9..<12 {
        #expect(abs(rt[i]) < 1e-5, "t[\(i - 9)] = \(rt[i])")
    }
}

/// With target = identity and ref w2c translation (1, 0, 0), the target→ref relative pose is R = I,
/// t = (1, 0, 0).
@Test
func relativePosePureTranslation() {
    let targetVM: [Float] = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
    let refVM: [Float] = [
        1, 0, 0, 1,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
    let rt = PlaneSweepCostVolume.relativePose(targetViewmat: targetVM, refViewmat: refVM)
    let expectedR: [Float] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    let expectedT: [Float] = [1, 0, 0]
    for i in 0..<9 { #expect(abs(rt[i] - expectedR[i]) < 1e-5) }
    for i in 0..<3 { #expect(abs(rt[9 + i] - expectedT[i]) < 1e-5) }
}

// MARK: - plane sweep kernel: same viewpoint

/// From the same viewpoint every depth hypothesis samples the same pixel, so the Hamming cost
/// inside the frustum is 0.
@Test
func planeSweepIdentityCostIsZero() {
    let w = 16, h = 16
    let census = randomTexturedCensus(width: w, height: h, seed: 7)
    let vm: [Float] = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
    let rt = PlaneSweepCostVolume.relativePose(targetViewmat: vm, refViewmat: vm)
    let intr: [Float] = [20, 20, 8, 8]
    let depths: [Float] = [1.0, 2.0, 5.0, 10.0]
    let cv = PlaneSweepCostVolume.compute(
        targetCensus: census, refCensus: census,
        intrinsicsTarget: intr, intrinsicsRef: intr,
        relativeRT: rt, depths: depths
    )
    let arr = cv.asArray(Int32.self)
    // Projections can fall outside near the border, so only check the central region that is surely
    // inside the frustum.
    var zeros = 0
    var total = 0
    for y in 4..<(h - 4) {
        for x in 4..<(w - 4) {
            for k in 0..<depths.count {
                let cost = arr[(y * w + x) * depths.count + k]
                if cost == 0 { zeros += 1 }
                total += 1
            }
        }
    }
    #expect(zeros == total,
        "identity-pose center cost should be all 0; got \(zeros)/\(total)")
}

// MARK: - plane sweep kernel: minimum cost at the true depth

/// Simulate horizontal-baseline rectified stereo by shifting the image sideways; minimum cost lands
/// near Z = fx·B / shift.
@Test
func planeSweepMinimumNearTrueDepth() {
    let w = 32, h = 16
    let fx: Float = 50, fy: Float = 50, cx: Float = Float(w) / 2, cy: Float = Float(h) / 2
    let baseline: Float = 1.0    // 1.0 m along x
    let trueDepth: Float = 5.0
    let shiftPixels = Int((fx * baseline / trueDepth).rounded())   // = 10

    var lcg = TestLCG(seed: 13)
    // The ref camera is on the world -x side, so the same point appears at u_r = u_t + fx·B/Z = u_t
    // + 10. Reproduce this by slicing target from the center and ref from left of center of a
    // shared canvas.
    let extW = w + shiftPixels + 4
    var pixels: [Float] = []
    for _ in 0..<(h * extW) { pixels.append(lcg.next()) }
    let extLuma = MLXArray(pixels, [h, extW])
    let targetLuma = extLuma[0..., shiftPixels..<(shiftPixels + w)]
    let refLuma = extLuma[0..., 0..<w]
    let targetCensus = CensusTransform.compute(luma: targetLuma)
    let refCensus = CensusTransform.compute(luma: refLuma)

    // ref w2c translation +baseline = camera center (-baseline, 0, 0)
    let targetVM: [Float] = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
    let refVM: [Float] = [
        1, 0, 0, baseline,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
    let rt = PlaneSweepCostVolume.relativePose(targetViewmat: targetVM, refViewmat: refVM)

    // Depth hypotheses: a wide set containing the true value 5.0
    let depths: [Float] = [2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 8.0, 12.0]
    let intr: [Float] = [fx, fy, cx, cy]
    let cv = PlaneSweepCostVolume.compute(
        targetCensus: targetCensus, refCensus: refCensus,
        intrinsicsTarget: intr, intrinsicsRef: intr,
        relativeRT: rt, depths: depths
    )
    let arr = cv.asArray(Int32.self)

    // Mean cost per depth hypothesis over the central region inside the frustum.
    var meanCostPerDepth = [Float](repeating: 0, count: depths.count)
    var countPerDepth = 0
    for y in 4..<(h - 4) {
        for x in 8..<(w - 8) {
            for k in 0..<depths.count {
                meanCostPerDepth[k] += Float(arr[(y * w + x) * depths.count + k])
            }
            countPerDepth += 1
        }
    }
    for k in 0..<depths.count {
        meanCostPerDepth[k] /= Float(countPerDepth)
    }

    var minK = 0
    var minVal = meanCostPerDepth[0]
    for k in 1..<depths.count {
        if meanCostPerDepth[k] < minVal {
            minVal = meanCostPerDepth[k]
            minK = k
        }
    }
    // The minimum lands at the true value 5.0 ± 0.5 (index 3..5).
    #expect([3, 4, 5].contains(minK),
        "argmin should be near true depth 5.0; got depth index \(minK), depth=\(depths[minK]), profile=\(meanCostPerDepth)")
}
