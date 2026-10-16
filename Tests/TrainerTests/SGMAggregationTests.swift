// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// SGM Aggregation (PocketGS Operator G).

import Foundation
import MLX
import Testing
@testable import Trainer

/// If every pixel and disparity has the same cost, the 1-path SGM output stays L = C
/// (the start is L = C, then L = C + min(L_prev) - minLprev = 3 + 3 - 3).
@Test
func sgmSinglePathConstantInput() {
    let h = 8, w = 8, k = 4
    let costH = [Int32](repeating: 3, count: h * w * k)
    let cost = MLXArray(costH, [h, w, k])
    let pen = SGMAggregation.Penalties(p1: 1, p2: 5)
    let out = SGMAggregation.aggregateSinglePath(
        costVolume: cost, direction: 0, penalties: pen
    )
    let arr = out.asArray(Int32.self)
    for v in arr {
        #expect(v == 3, "expected uniform L=3, got \(v)")
    }
}

/// If the cost volume is minimal at the same disparity for every pixel, argmin stays there after
/// 4-path SGM.
@Test
func sgmFourPathPreservesGlobalMinimum() {
    let h = 16, w = 16, k = 8
    let kTrue: Int32 = 3
    // cost[y, x, d] = |d - kTrue|
    var costH: [Int32] = []
    for _ in 0..<h {
        for _ in 0..<w {
            for d in 0..<k {
                costH.append(abs(Int32(d) - kTrue))
            }
        }
    }
    let cost = MLXArray(costH, [h, w, k])
    let pen = SGMAggregation.Penalties(p1: 1, p2: 5)
    let agg = SGMAggregation.aggregate(costVolume: cost, penalties: pen)
    let arr = agg.asArray(Int32.self)

    for y in 0..<h {
        for x in 0..<w {
            var minVal = arr[(y * w + x) * k + 0]
            var minIdx = 0
            for d in 1..<k {
                let v = arr[(y * w + x) * k + d]
                if v < minVal { minVal = v; minIdx = d }
            }
            #expect(minIdx == Int(kTrue),
                "pixel (\(y), \(x)) argmin = \(minIdx), expected \(kTrue)")
        }
    }
}

/// When the surroundings are stable at k=2 and only the center slightly favors k=5 (4 vs 5), the
/// jump from the surroundings to k=5 incurs the P2 penalty, so after SGM k=2 flips to being
/// favored.
@Test
func sgmFourPathReducesGapAgainstNeighbours() {
    let h = 9, w = 9, k = 8
    let kBackground: Int32 = 2
    let kAnomaly: Int32 = 5
    var costH: [Int32] = []
    for y in 0..<h {
        for x in 0..<w {
            for d in 0..<k {
                var v: Int32
                if y == 4 && x == 4 {
                    // Center pixel: make the anomaly side slightly lower, leaving room for
                    // smoothing to flip it.
                    v = (Int32(d) == kAnomaly) ? 4 : (Int32(d) == kBackground ? 5 : 12)
                } else {
                    // Surrounding pixels: minimum at background, others increase linearly
                    v = abs(Int32(d) - kBackground) * 2
                }
                costH.append(v)
            }
        }
    }
    let cost = MLXArray(costH, [h, w, k])
    let pen = SGMAggregation.Penalties(p1: 1, p2: 10)
    let agg = SGMAggregation.aggregate(costVolume: cost, penalties: pen)
    let arr = agg.asArray(Int32.self)

    let center = (4 * w + 4) * k
    let costAnomaly = arr[center + Int(kAnomaly)]
    let costBackground = arr[center + Int(kBackground)]
    let msg = "SGM should make background depth competitive at the center pixel: agg cost background=\(costBackground), anomaly=\(costAnomaly)"
    #expect(costBackground < costAnomaly, "\(msg)")
}

/// A single path's result differs by scan direction. With only the top-left pixel at low cost,
/// L→R propagates it to the right side and R→L does not reach it.
@Test
func sgmSinglePathDirectionMatters() {
    let h = 4, w = 8, k = 4
    var costH: [Int32] = []
    for y in 0..<h {
        for x in 0..<w {
            for d in 0..<k {
                let isCorner = (y == 0 && x == 0)
                let isMinDisparity = (d == 0)
                let v: Int32 = (isCorner && isMinDisparity) ? 0 : 10
                costH.append(v)
            }
        }
    }
    let cost = MLXArray(costH, [h, w, k])
    let pen = SGMAggregation.Penalties(p1: 1, p2: 3)
    let lr = SGMAggregation.aggregateSinglePath(
        costVolume: cost, direction: 0, penalties: pen
    )
    let rl = SGMAggregation.aggregateSinglePath(
        costVolume: cost, direction: 1, penalties: pen
    )
    let lrArr = lr.asArray(Int32.self)
    let rlArr = rl.asArray(Int32.self)
    // Each direction's start (left edge for L→R, right edge for R→L) stays the raw cost.
    #expect(lrArr[0] == 0)
    #expect(rlArr[(0 * w + w - 1) * k] == 10)
    // Only L→R carries the corner's information to the right edge (y=0, x=7).
    var lrSumRight: Int32 = 0
    var rlSumRight: Int32 = 0
    for d in 0..<k {
        lrSumRight += lrArr[(0 * w + 7) * k + d]
        rlSumRight += rlArr[(0 * w + 7) * k + d]
    }
    // Do not check exact values; only confirm the two differ.
    #expect(lrSumRight != rlSumRight,
        "directions should produce different L_r (L→R sum=\(lrSumRight), R→L sum=\(rlSumRight))")
}

/// Bit-exact comparison of all 8 directions against a CPU reference (Hirschmüller 2008 eq. 13). W≠H
/// and 1-pixel-wide inputs catch mixed-up diagonal scan starts and lengths. The simd version
/// distributes depth indices across lanes, so K = 1 / 31 / 32 / 33 / 64 also check lane boundaries
/// and out-of-range slots.
@Test(arguments: [
    (7, 12, 6), (12, 7, 6), (1, 9, 6), (9, 1, 6),
    (5, 8, 1), (6, 9, 31), (9, 6, 32), (4, 7, 33), (8, 5, 64),
])
func sgmAllDirectionsMatchCPUReference(dimsHWK: (Int, Int, Int)) {
    let (h, w, k) = dimsHWK
    var rng = SystemRandomNumberGenerator()
    let costH = (0..<(h * w * k)).map { _ in Int32.random(in: 0...25, using: &rng) }
    let cost = MLXArray(costH, [h, w, k])
    let p1: Int32 = 2, p2: Int32 = 7
    let pen = SGMAggregation.Penalties(p1: p1, p2: p2)
    let steps: [(Int, Int)] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1)]
    for (dir, (dx, dy)) in steps.enumerated() {
        var ref = [Int32](repeating: 0, count: h * w * k)
        // Scan order: iterate dy/dx so the previous pixel (x-dx, y-dy) is finalized first.
        let ys = dy >= 0 ? Array(0..<h) : Array((0..<h).reversed())
        let xs = dx >= 0 ? Array(0..<w) : Array((0..<w).reversed())
        for y in ys {
            for x in xs {
                let px = x - dx, py = y - dy
                let base = (y * w + x) * k
                guard px >= 0, px < w, py >= 0, py < h else {
                    for d in 0..<k { ref[base + d] = costH[base + d] }
                    continue
                }
                let pb = (py * w + px) * k
                let prev = Array(ref[pb..<(pb + k)])
                let mn = prev.min()!
                for d in 0..<k {
                    var m = min(prev[d], mn + p2)
                    if d > 0 { m = min(m, prev[d - 1] + p1) }
                    if d < k - 1 { m = min(m, prev[d + 1] + p1) }
                    ref[base + d] = costH[base + d] + m - mn
                }
            }
        }
        for impl in [SGMAggregation.Implementation.simd, .scalar] {
            let got = SGMAggregation.aggregateSinglePath(
                costVolume: cost, direction: dir, penalties: pen, implementation: impl
            ).asArray(Int32.self)
            #expect(got == ref, "\(impl) direction \(dir) mismatch for H=\(h) W=\(w) K=\(k)")
        }
    }
}

/// On a cost volume close to real MVS size (270×202×64, Hamming 0..25), the 8-path sum of the simd
/// version exactly matches the sequential version. P1/P2 at the defaults (2/48) and at a strongly
/// smoothing combination.
@Test(arguments: [(Int32(2), Int32(48)), (Int32(10), Int32(120))])
func sgmSimdMatchesScalarOnMVSSizedVolume(pen: (Int32, Int32)) {
    MLXRandom.seed(3)
    let cost = MLXRandom.randInt(low: 0, high: 26, [202, 270, 64]).asType(.int32)
    let penalties = SGMAggregation.Penalties(p1: pen.0, p2: pen.1)
    let simd = SGMAggregation.aggregate(costVolume: cost, penalties: penalties, implementation: .simd)
    let scalar = SGMAggregation.aggregate(costVolume: cost, penalties: penalties, implementation: .scalar)
    #expect(MLX.notEqual(simd, scalar).asType(.int32).sum().item(Int.self) == 0)
}
