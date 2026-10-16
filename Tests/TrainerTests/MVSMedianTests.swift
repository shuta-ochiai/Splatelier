// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// CostVolumeMVS.elementwiseMedian exactly matches "stack + the n/2-th after sorting along the
// reference axis". Compared over 1..8 references (odd and even) with small-integer,
// Hamming-distance-like costs that have many ties.

import MLX
import Testing
@testable import Trainer

@Test
func elementwiseMedianMatchesSortedReference() {
    MLXRandom.seed(7)
    for n in 1...8 {
        // [H, W, K] = [13, 17, 9], values 0..24 (the Hamming distance range of Census 5x5), many
        // ties.
        let vols = (0..<n).map { _ in MLXRandom.randInt(low: 0, high: 25, [13, 17, 9]).asType(.int32) }
        let ref = MLX.sorted(MLX.stacked(vols, axis: 0), axis: 0)[n / 2]
        let got = CostVolumeMVS.elementwiseMedian(vols)
        let mismatch = MLX.notEqual(ref, got).asType(.int32).sum().item(Int.self)
        #expect(mismatch == 0, "n=\(n): \(mismatch) mismatches")
        #expect(got.shape == [13, 17, 9])
    }
}

@Test
func elementwiseMedianMatchesOnFloatInput() {
    // Float input also matches the sorted reference.
    MLXRandom.seed(11)
    let vols = (0..<5).map { _ in MLXRandom.randInt(low: 0, high: 25, [8, 8, 64]).asType(.float32) }
    let ref = MLX.sorted(MLX.stacked(vols, axis: 0), axis: 0)[2]
    let got = CostVolumeMVS.elementwiseMedian(vols)
    #expect(MLX.notEqual(ref, got).asType(.int32).sum().item(Int.self) == 0)
}
