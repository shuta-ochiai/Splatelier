// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// ORB pipeline (FAST + BRIEF + matcher + symmetric check).

import Foundation
import MLX
import Testing
@testable import Trainer

/// Extracting the same image twice self-matches every keypoint at Hamming distance 0.
@Test
func orbPipelineSameImageMatchesAllKeypoints() {
    let h = 64, w = 64
    // Random texture with a fixed seed
    var lcg = TestLCG(seed: 31)
    var pixels = [Float](repeating: 0, count: h * w)
    for i in 0..<(h * w) { pixels[i] = lcg.next() }
    let luma = MLXArray(pixels, [h, w])

    let f1 = ORBPipeline.extract(luma: luma, fastThreshold: 0.05, maxKeypoints: 200)
    let f2 = ORBPipeline.extract(luma: luma, fastThreshold: 0.05, maxKeypoints: 200)

    #expect(f1.coords.count > 0, "should detect at least 1 keypoint")
    #expect(f1.coords.count == f2.coords.count, "deterministic extraction")

    let matches = ORBPipeline.match(a: f1, b: f2, loweRatio: 0.8)
    // d1=0, d2>0, so it always passes the Lowe ratio, and the symmetric check passes trivially.
    #expect(matches.count == f1.coords.count, "all keypoints should match self with d=0")
    for m in matches {
        #expect(m.distance == 0,
            "self-match should have Hamming distance 0, got \(m.distance)")
        #expect(m.queryIdx == m.trainIdx,
            "self-match queryIdx==trainIdx; got \(m.queryIdx) vs \(m.trainIdx)")
    }
}

/// Two different random textures barely match.
@Test
func orbPipelineRandomImagesFewMatches() {
    let h = 64, w = 64
    func makeRandom(seed: UInt32) -> [Float] {
        var lcg = TestLCG(seed: seed)
        return (0..<(h * w)).map { _ in lcg.next() }
    }
    let lumaA = MLXArray(makeRandom(seed: 11), [h, w])
    let lumaB = MLXArray(makeRandom(seed: 99), [h, w])

    let fA = ORBPipeline.extract(luma: lumaA, fastThreshold: 0.05, maxKeypoints: 200)
    let fB = ORBPipeline.extract(luma: lumaB, fastThreshold: 0.05, maxKeypoints: 200)
    let matches = ORBPipeline.match(a: fA, b: fB, loweRatio: 0.8)
    // Loosely require "at most 30% of the keypoint count" (stable with a fixed seed).
    let cap = max(fA.coords.count, fB.coords.count) * 30 / 100
    #expect(matches.count <= cap,
        "random images should yield few matches: got \(matches.count) of \(fA.coords.count)/\(fB.coords.count)")
}

/// For a purely translated image, correspondences come out mostly correct.
@Test
func orbPipelineTranslationMatches() {
    let h = 64, w = 64
    // Crop 2 images offset by 4 px from a shared canvas.
    var lcg = TestLCG(seed: 31)
    var canvas = [Float](repeating: 0, count: h * (w + 8))
    for i in 0..<(h * (w + 8)) { canvas[i] = lcg.next() }
    var slabA = [Float](repeating: 0, count: h * w)
    var slabB = [Float](repeating: 0, count: h * w)
    for y in 0..<h {
        for x in 0..<w {
            slabA[y * w + x] = canvas[y * (w + 8) + x]
            slabB[y * w + x] = canvas[y * (w + 8) + x + 4]
        }
    }
    let lumaA = MLXArray(slabA, [h, w])
    let lumaB = MLXArray(slabB, [h, w])
    let fA = ORBPipeline.extract(luma: lumaA, fastThreshold: 0.05, maxKeypoints: 200)
    let fB = ORBPipeline.extract(luma: lumaB, fastThreshold: 0.05, maxKeypoints: 200)
    let matches = ORBPipeline.match(a: fA, b: fB, loweRatio: 0.8)

    // Correspondences satisfy (xA, yA) ≈ (xB + 4, yB), so the mean dx is close to 4.
    var dxSum = 0
    var dxCount = 0
    for m in matches {
        let pa = fA.coords[m.queryIdx]
        let pb = fB.coords[m.trainIdx]
        let dy = pa.y - pb.y
        // There is no y offset, so only use matches with |dy| ≤ 2.
        if abs(dy) <= 2 {
            dxSum += (pa.x - pb.x)
            dxCount += 1
        }
    }
    #expect(dxCount > 0, "should have at least 1 dy-consistent match")
    if dxCount > 0 {
        let meanDx = Double(dxSum) / Double(dxCount)
        #expect(abs(meanDx - 4) < 2,
            "expected mean dx ≈ 4 (translation), got \(meanDx) over \(dxCount) matches")
    }
}

/// Rotation invariance (steered BRIEF). An image rotated exactly 90° by index remap matches
/// consistently with the rotated geometry. Non-rotated BRIEF gets nearly 0 matches.
@Test
func orbPipelineRotationInvariance() {
    let n = 64
    var lcg = TestLCG(seed: 71)
    var a = [Float](repeating: 0, count: n * n)
    for i in 0..<(n * n) { a[i] = lcg.next() }
    // B = A rotated 90° clockwise: B[r'][c'] = A[N-1-c'][r'], i.e. A's (x, y) → B's (N-1-y, x).
    var b = [Float](repeating: 0, count: n * n)
    for rp in 0..<n {
        for cp in 0..<n {
            b[rp * n + cp] = a[(n - 1 - cp) * n + rp]
        }
    }
    let fA = ORBPipeline.extract(luma: MLXArray(a, [n, n]), fastThreshold: 0.05, maxKeypoints: 300)
    let fB = ORBPipeline.extract(luma: MLXArray(b, [n, n]), fastThreshold: 0.05, maxKeypoints: 300)
    let matches = ORBPipeline.match(a: fA, b: fB, loweRatio: 0.8)

    // Count matches consistent with (xB, yB) ≈ (N-1-yA, xA).
    var consistent = 0
    for m in matches {
        let pa = fA.coords[m.queryIdx]
        let pb = fB.coords[m.trainIdx]
        let predX = (n - 1) - pa.y
        let predY = pa.x
        if abs(pb.x - predX) <= 2 && abs(pb.y - predY) <= 2 { consistent += 1 }
    }
    // Threshold kept low to be safe.
    #expect(consistent >= 10,
        "rotation-invariant matches expected; got \(consistent) consistent of \(matches.count) matches")
}

/// matchBatch returns the same results as per-pair matching, in pair order (covering pairs with and
/// without a band, pairs with no keypoints, and more pairs than fit in one evaluation).
@Test
func orbPipelineMatchBatchEqualsPerPairMatch() {
    let h = 64, w = 64
    var lcg = TestLCG(seed: 97)
    let frames: [ORBPipeline.FrameFeatures] = (0..<4).map { _ in
        ORBPipeline.extract(luma: MLXArray((0..<(h * w)).map { _ in lcg.next() }, [h, w]),
                            fastThreshold: 0.05, maxKeypoints: 150)
    }
    let empty = ORBPipeline.FrameFeatures(coords: [], descriptors: MLXArray.zeros([0, 8], dtype: .int32))
    let guide = ORBPipeline.EpipolarGuide(fundamentalAB: [0, 0, 0, 0, 0, -1, 0, 1, 0], bandPx: 6)
    var pairs: [(a: ORBPipeline.FrameFeatures, b: ORBPipeline.FrameFeatures, epipolar: ORBPipeline.EpipolarGuide?)] = []
    for i in 0..<4 { for j in 0..<4 { pairs.append((frames[i], frames[j], (i + j) % 2 == 0 ? guide : nil)) } }
    pairs.append((frames[0], empty, nil))
    let batched = ORBPipeline.matchBatch(pairs, loweRatio: 0.8, pairsPerEval: 5)
    #expect(batched.count == pairs.count)
    for (p, b) in zip(pairs, batched) {
        #expect(ORBPipeline.match(a: p.a, b: p.b, loweRatio: 0.8, epipolar: p.epipolar) == b)
    }
    #expect(batched[0].count == frames[0].coords.count, "an image matched against itself matches every keypoint")
    #expect(batched.last!.isEmpty)
}
