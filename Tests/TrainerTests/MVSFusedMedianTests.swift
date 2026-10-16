// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Checks that PlaneSweepCostVolume.computeMedian (fused kernel: cost + median over the reference
// axis) exactly matches the old path (compute per reference → CostVolumeMVS.elementwiseMedian).
// Poses are randomized so projections span inside/outside the image and behind the camera (Z<=0).

import Foundation
import MLX
import Testing
@testable import Trainer

private func randomPose(_ rng: inout SystemRandomNumberGenerator) -> [Float] {
    // Small rotation (±0.3 rad per axis) + translation (±0.3). target→ref [R(9) | t(3)].
    func r() -> Float { Float.random(in: -0.3...0.3, using: &rng) }
    let (a, b, c) = (r(), r(), r())
    let (ca, sa, cb, sb, cc, sc) = (cos(a), sin(a), cos(b), sin(b), cos(c), sin(c))
    // R = Rz(c) · Ry(b) · Rx(a)
    let R: [Float] = [
        cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa,
        sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa,
        -sb, cb * sa, cb * ca,
    ]
    return R + [r(), r(), r()]
}

@Test(arguments: [1, 2, 3, 4, 5, 6, 7, 8])
func fusedMedianMatchesPerRefVolumesAndElementwiseMedian(m: Int) {
    var rng = SystemRandomNumberGenerator()
    let (h, w, k) = (30, 41, 16)
    func census(_ hh: Int, _ ww: Int) -> MLXArray {
        MLXArray((0..<(hh * ww)).map { _ in Int32.random(in: 0..<(1 << 25), using: &rng) }, [hh, ww])
    }
    let target = census(h, w)
    let refs = (0..<m).map { _ in census(h, w) }
    let intrT: [Float] = [40, 41, 20.5, 15.2]
    let intrRs: [[Float]] = (0..<m).map { _ in
        [Float.random(in: 35...45, using: &rng), Float.random(in: 35...45, using: &rng),
         Float.random(in: 18...22, using: &rng), Float.random(in: 13...17, using: &rng)]
    }
    let rts = (0..<m).map { _ in randomPose(&rng) }
    // Include hypotheses that end up behind the camera (negative depth) to exercise the max-cost
    // path for out-of-image / Z<=0.
    let depths: [Float] = (0..<k).map { -0.2 + Float($0) * 0.25 }

    let perRef = (0..<m).map {
        PlaneSweepCostVolume.compute(
            targetCensus: target, refCensus: refs[$0],
            intrinsicsTarget: intrT, intrinsicsRef: intrRs[$0],
            relativeRT: rts[$0], depths: depths)
    }
    let expected = CostVolumeMVS.elementwiseMedian(perRef)
    let got = PlaneSweepCostVolume.computeMedian(
        targetCensus: target, refCensuses: refs,
        intrinsicsTarget: intrT, intrinsicsRefs: intrRs,
        relativeRTs: rts, depths: depths)
    #expect(got.shape == [h, w, k])
    let mismatch = MLX.notEqual(expected, got).asType(.int32).sum().item(Int.self)
    #expect(mismatch == 0, "M=\(m): \(mismatch) mismatches")
}
