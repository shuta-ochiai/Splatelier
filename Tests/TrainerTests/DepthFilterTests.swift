// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Depth filter (PocketGS Operator G). confidence = 1 - best/second, where second is searched
// outside winner ± excludeRadius (default 2).

import Foundation
import MLX
import Testing
@testable import Trainer

/// A clear winner is accepted. The winner's neighbor bins are set close (5 vs 4) to mimic an SGM
/// valley; without neighbor exclusion it would be rejected at conf=0.2. With exclusion, the 8 three
/// bins away is second, conf=0.5.
@Test
func depthFilterClearWinnerPasses() {
    let h = 4, w = 4
    let depths: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    let k = depths.count

    // best=4 (idx1), excluding k=0..3 gives second=8 → conf = 1 - 4/8 = 0.5 ≥ 0.4.
    // Neighbor costs are symmetric (5, 5), so sub-pixel δ=0 and depth is the bin value 2.0.
    var costH: [Int32] = []
    for _ in 0..<(h * w) {
        costH += [5, 4, 5, 20, 8, 20, 20, 20]
    }
    let aggCost = MLXArray(costH, [h, w, k])

    let result = DepthFilter.filter(aggregatedCost: aggCost, depths: depths, threshold: 0.4)
    let depth = result.depthMap.asArray(Float.self)
    let conf = result.confidence.asArray(Float.self)
    for i in 0..<(h * w) {
        #expect(abs(depth[i] - 2.0) < 1e-5, "pixel \(i) depth=\(depth[i]), expected 2.0")
        #expect(abs(conf[i] - 0.5) < 1e-3, "pixel \(i) conf=\(conf[i]), expected ~0.5")
    }
}

/// A truly ambiguous case with a close rival outside excludeRadius is rejected, and depth is -1.
@Test
func depthFilterLowConfidenceRejected() {
    let h = 2, w = 2
    let depths: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    let k = depths.count

    // best=4 (idx1), excluding k=0..3 gives second=5 (idx4) → conf = 1 - 4/5 = 0.2 < 0.4.
    var costH: [Int32] = []
    for _ in 0..<(h * w) {
        costH += [6, 4, 6, 6, 5, 20, 20, 20]
    }
    let aggCost = MLXArray(costH, [h, w, k])

    let result = DepthFilter.filter(aggregatedCost: aggCost, depths: depths, threshold: 0.4)
    let depth = result.depthMap.asArray(Float.self)
    let conf = result.confidence.asArray(Float.self)
    for i in 0..<(h * w) {
        #expect(depth[i] == -1.0, "should be rejected; got depth=\(depth[i])")
        #expect(abs(conf[i] - 0.2) < 1e-3, "conf=\(conf[i])")
    }
}

/// With threshold 0, even confidence 0 is not rejected.
@Test
func depthFilterZeroThresholdAcceptsAll() {
    let h = 2, w = 2
    let depths: [Float] = [1.0, 2.0, 3.0]
    var costH: [Int32] = []
    for _ in 0..<(h * w) {
        costH += [5, 4, 5]   // K=3: every bin is within the exclusion range, conf=0. Symmetric, so δ=0
    }
    let aggCost = MLXArray(costH, [h, w, depths.count])
    let result = DepthFilter.filter(aggregatedCost: aggCost, depths: depths, threshold: 0.0)
    let depth = result.depthMap.asArray(Float.self)
    for v in depth {
        #expect(abs(v - 2.0) < 1e-5, "depth=\(v), expected 2.0 (idx 1)")
    }
}

/// K=1 has no second, so confidence is 0 and it is rejected.
@Test
func depthFilterSingleDepthRejected() {
    let h = 2, w = 2
    let depths: [Float] = [5.0]
    var costH: [Int32] = []
    for _ in 0..<(h * w) { costH += [3] }
    let aggCost = MLXArray(costH, [h, w, 1])
    let result = DepthFilter.filter(aggregatedCost: aggCost, depths: depths, threshold: 0.4)
    let depth = result.depthMap.asArray(Float.self)
    for v in depth {
        #expect(v == -1.0, "single-depth pixel should be rejected; got \(v)")
    }
}

/// Even with a different winner per pixel, WTA gives the correct depth independently per pixel.
/// Non-winners are all 10 and both neighbors are symmetric (δ=0), so depth is exactly the bin
/// value.
@Test
func depthFilterPerPixelDifferentWinners() {
    let h = 2, w = 2
    let depths: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    let k = depths.count
    // winner idx 0/2/5/7 (cost 1/2/3/4), second 10 → conf 0.9/0.8/0.7/0.6
    let costH: [Int32] = [
         1, 10, 10, 10, 10, 10, 10, 10,
        10, 10,  2, 10, 10, 10, 10, 10,
        10, 10, 10, 10, 10,  3, 10, 10,
        10, 10, 10, 10, 10, 10, 10,  4,
    ]
    let aggCost = MLXArray(costH, [h, w, k])
    let result = DepthFilter.filter(aggregatedCost: aggCost, depths: depths, threshold: 0.4)
    let depth = result.depthMap.asArray(Float.self)
    let expected: [Float] = [1.0, 3.0, 6.0, 8.0]
    for i in 0..<4 {
        #expect(abs(depth[i] - expected[i]) < 1e-5,
            "pixel \(i): got \(depth[i]), expected \(expected[i])")
    }
}

/// Sub-pixel refinement (parabola fit). Hypotheses are evenly spaced in inverse depth (0.2-1.2).
/// - Interior bin [10,4,8] → δ=(cm-cp)/(2(cm-2c0+cp))=0.1, w=0.4+0.1·0.2=0.42 → z≈2.381 (pulled
///   toward the lower cost).
/// - If an edge bin wins, no refinement; the bin value is kept.
@Test
func depthFilterSubPixelRefinement() {
    let h = 1, w = 2
    let depths: [Float] = [5.0, 2.5, 1.0 / 0.6, 1.25, 1.0, 1.0 / 1.2]
    let k = depths.count
    let costH: [Int32] = [
        10, 4, 8, 12, 8, 20,    // interior bin idx1 wins, second=8 (idx4), conf=0.5
         4, 10, 12, 10, 20, 20, // edge bin idx0 wins, second=10 (idx3), conf=0.6
    ]
    let aggCost = MLXArray(costH, [h, w, k])
    let result = DepthFilter.filter(aggregatedCost: aggCost, depths: depths, threshold: 0.4)
    let depth = result.depthMap.asArray(Float.self)
    #expect(abs(depth[0] - 2.380952) < 1e-3,
            "interior sub-pixel: got \(depth[0]), expected ≈2.381")
    #expect(abs(depth[1] - 5.0) < 1e-5,
            "boundary winner must skip refinement: got \(depth[1]), expected 5.0")
}
