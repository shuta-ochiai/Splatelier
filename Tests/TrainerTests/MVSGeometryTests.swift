// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// MVS geometry helpers (ReferenceSelection / DepthRangeEstimator, PocketGS Operator G).

import Foundation
import Testing
@testable import Trainer

// MARK: - ReferenceSelection.score

/// If target == ref, baseline 0 = bTarget gives baselineTerm = 1 and α = 0 gives angleTerm = max(0,
/// 1) = 1 → score 1 (rejecting target == ref is selectTopK's job).
@Test
func referenceScoreZeroBaselineZeroAngle() {
    let center = SIMD3<Float>(0, 0, 0)
    let scene = SIMD3<Float>(0, 0, 5)
    let params = ReferenceSelection.Parameters(bTarget: 0, sigmaB: 1, alphaMin: 0.1)
    let s = ReferenceSelection.score(
        targetCenter: center, refCenter: center,
        sceneCenter: scene, parameters: params
    )
    #expect(abs(s - 1.0) < 1e-5, "got \(s)")
}

/// With target = (-d/2, 0, 0), ref = (d/2, 0, 0), scene = (0, 0, z): baseline = bTarget = d, and
/// if α = 2·atan(d/2 / z) = alphaMin, both terms are 1 and score is 1.
@Test
func referenceScorePreferredBaseline() {
    let d: Float = 0.5
    let z: Float = 5.0
    let target = SIMD3<Float>(-d / 2, 0, 0)
    let ref = SIMD3<Float>(d / 2, 0, 0)
    let scene = SIMD3<Float>(0, 0, z)
    let alpha = 2 * atan((d / 2) / z)
    let params = ReferenceSelection.Parameters(
        bTarget: d, sigmaB: 1, alphaMin: alpha
    )
    let s = ReferenceSelection.score(
        targetCenter: target, refCenter: ref,
        sceneCenter: scene, parameters: params
    )
    #expect(abs(s - 1.0) < 1e-4, "got \(s)")
}

// MARK: - ReferenceSelection.selectTopK

/// top-3 is in descending score order, with idx 2 (b = bTarget) first.
@Test
func referenceSelectTopKReturnsScoreDescending() {
    let target = SIMD3<Float>(0, 0, 0)
    let nearRef = SIMD3<Float>(0.05, 0, 0)
    let goodRef = SIMD3<Float>(0.5, 0, 0)
    let farRef = SIMD3<Float>(3.0, 0, 0)
    let scene = SIMD3<Float>(0, 0, 5)
    let centers = [target, nearRef, goodRef, farRef]
    let params = ReferenceSelection.Parameters(bTarget: 0.5, sigmaB: 0.3, alphaMin: 0.01)
    let top3 = ReferenceSelection.selectTopK(
        targetIndex: 0, cameraCenters: centers, sceneCenter: scene, k: 3, parameters: params
    )
    #expect(top3.count == 3)
    #expect(top3[0] == 2, "top-1 must be good baseline (idx 2)")
    // The order of the remaining 2 (too close / too far) is set by score comparison.
    let s1 = ReferenceSelection.score(
        targetCenter: target, refCenter: nearRef, sceneCenter: scene, parameters: params)
    let s3 = ReferenceSelection.score(
        targetCenter: target, refCenter: farRef, sceneCenter: scene, parameters: params)
    if s1 > s3 {
        #expect(top3[1] == 1 && top3[2] == 3)
    } else {
        #expect(top3[1] == 3 && top3[2] == 1)
    }
}

/// If k exceeds the candidate count, return every candidate except target (cameras.count - 1).
@Test
func referenceSelectTopKClampsToAvailable() {
    let target = SIMD3<Float>(0, 0, 0)
    let ref1 = SIMD3<Float>(0.5, 0, 0)
    let ref2 = SIMD3<Float>(-0.5, 0, 0)
    let scene = SIMD3<Float>(0, 0, 5)
    let centers = [target, ref1, ref2]
    let picked = ReferenceSelection.selectTopK(
        targetIndex: 0, cameraCenters: centers, sceneCenter: scene, k: 10
    )
    #expect(picked.count == 2, "should clamp to (cameras.count - 1)")
    #expect(Set(picked) == Set([1, 2]))
}

/// With only one camera, selectTopK returns an empty array.
@Test
func referenceSelectTopKSingleCameraReturnsEmpty() {
    let centers = [SIMD3<Float>(0, 0, 0)]
    let picked = ReferenceSelection.selectTopK(
        targetIndex: 0, cameraCenters: centers, sceneCenter: SIMD3<Float>(0, 0, 5), k: 5
    )
    #expect(picked.isEmpty)
}

// MARK: - ReferenceSelection covisibility

/// A camera facing the same way but shifted sideways sees almost all of target's sparse points; one
/// facing the opposite way sees none.
@Test
func referenceCovisibilityDependsOnViewingDirection() {
    let intr: Intrinsics = (fx: 100, fy: 100, cx: 50, cy: 50)
    func view(_ r: [Float], _ t: SIMD3<Float>) -> ReferenceSelection.View {
        // viewmat = [R | t] (world → camera, row-major)
        ReferenceSelection.View(
            viewmat: [r[0], r[1], r[2], t.x, r[3], r[4], r[5], t.y, r[6], r[7], r[8], t.z, 0, 0, 0, 1],
            intrinsics: intr, width: 100, height: 100)
    }
    let identity: [Float] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    let target = view(identity, .zero)
    let shifted = view(identity, SIMD3<Float>(-0.5, 0, 0))                  // stands at x=+0.5 looking along +z
    let opposite = view([-1, 0, 0, 0, 1, 0, 0, 0, -1], .zero)                // same position, looking along -z
    // Grid points inside target's view (z=5, ±2m).
    var points: [SIMD3<Float>] = []
    for i in 0..<20 { for j in 0..<20 {
        points.append(SIMD3<Float>(Float(i) * 0.2 - 1.9, Float(j) * 0.2 - 1.9, 5))
    } }
    let cov = ReferenceSelection.covisibility(
        targetIndex: 0, views: [target, shifted, opposite], points: points, maxSamples: 64)!
    #expect(cov[0] == 1)
    #expect(cov[1] > 0.8, "got \(cov[1])")
    #expect(cov[2] == 0)
    // nil if no points are visible in target.
    #expect(ReferenceSelection.covisibility(
        targetIndex: 2, views: [target, shifted, opposite], points: points) == nil)
}

/// Candidates meeting the covisibility floor are picked first in score order (even the top-score
/// candidate goes later if its covisibility is low).
@Test
func referenceSelectTopKPrefersCovisibleCandidates() {
    let centers = [SIMD3<Float>(0, 0, 0), SIMD3<Float>(0.5, 0, 0),
                   SIMD3<Float>(0.45, 0, 0), SIMD3<Float>(0.3, 0, 0)]   // scores are 1 > 2 > 3
    let scene = SIMD3<Float>(0, 0, 5)
    var params = ReferenceSelection.Parameters(bTarget: 0.5, sigmaB: 0.3, alphaMin: 0.01)
    let cov: [Float] = [1, 0, 0.9, 0.7]
    let plain = ReferenceSelection.selectTopK(
        targetIndex: 0, cameraCenters: centers, sceneCenter: scene, k: 2, parameters: params,
        covisibility: cov)
    #expect(plain == [1, 2], "covisibility is ignored when minCovisibility is unset")
    params.minCovisibility = 0.6
    let picked = ReferenceSelection.selectTopK(
        targetIndex: 0, cameraCenters: centers, sceneCenter: scene, k: 2, parameters: params,
        covisibility: cov)
    #expect(picked == [2, 3])
}

/// If too few candidates meet the floor, fill in covisibility × score order.
@Test
func referenceSelectTopKCovisibilityFallsBackToWeightedScore() {
    let centers = [SIMD3<Float>(0, 0, 0), SIMD3<Float>(0.5, 0, 0),
                   SIMD3<Float>(0.45, 0, 0), SIMD3<Float>(0.3, 0, 0)]
    let params = ReferenceSelection.Parameters(
        bTarget: 0.5, sigmaB: 0.3, alphaMin: 0.01, minCovisibility: 0.6)
    let picked = ReferenceSelection.selectTopK(
        targetIndex: 0, cameraCenters: centers, sceneCenter: SIMD3<Float>(0, 0, 5), k: 2,
        parameters: params, covisibility: [1, 0.1, 0.5, 0.2])
    #expect(picked == [2, 1])
}

// MARK: - OrderStatistic

/// quickselect matches a[k] of the sorted array (including arrays with many equal values, 1
/// element, and sub-ranges).
@Test
func orderStatisticMatchesSortedArray() {
    var state: UInt64 = 0x9E3779B97F4A7C15
    func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state >> 33
    }
    for n in [1, 2, 3, 7, 64, 1000] {
        for distinct in [1, 3, 50, 100_000] {
            let base = (0..<n).map { _ in Float(next() % UInt64(distinct)) * 0.25 }
            let sorted = base.sorted()
            for k in Set([0, n / 2, n - 1, Int(Float(n) * 0.05), Int(Float(n) * 0.95)].map { min(n - 1, $0) }) {
                var a = base
                #expect(OrderStatistic.select(&a, k) == sorted[k], "n=\(n) distinct=\(distinct) k=\(k)")
                #expect(a[k] == sorted[k])
                #expect(a[..<k].allSatisfy { $0 <= sorted[k] } && a[(k + 1)...].allSatisfy { $0 >= sorted[k] })
                // Pick the larger rank first, then a smaller rank from the range before it (same
                // usage as DepthRangeEstimator)
                let lowK = k / 3
                if lowK < k {
                    #expect(OrderStatistic.select(&a, lowK, in: 0..<k) == sorted[lowK])
                }
            }
        }
    }
}

// MARK: - DepthRangeEstimator

/// With 100 points at z = 1..100 on the optical axis, the range is quantiles 0.05 / 0.95 times the
/// default factors 0.5 / 1.5.
@Test
func depthRangeQuantilesOnSyntheticPoints() {
    let vm: [Float] = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
    // All points project to the image center (X = Y = 0)
    var pts: [SIMD3<Float>] = []
    for i in 1...100 { pts.append(SIMD3<Float>(0, 0, Float(i))) }
    let intr: Intrinsics = (100, 100, 64, 64)
    let range = DepthRangeEstimator.estimate(
        viewmat: vm, imageWidth: 128, imageHeight: 128,
        intrinsics: intr, points: pts
    )
    guard let r = range else {
        Issue.record("range should not be nil")
        return
    }
    // index Int(100·0.05) = 5 → depth 6, Int(100·0.95) = 95 → depth 96
    #expect(abs(r.minDepth - 6.0 * 0.5) < 1e-4, "minDepth=\(r.minDepth)")
    #expect(abs(r.maxDepth - 96.0 * 1.5) < 1e-4, "maxDepth=\(r.maxDepth)")
}

/// Returns nil when every point is outside the frustum (z < 0).
@Test
func depthRangeNoValidPointsReturnsNil() {
    let vm: [Float] = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
    let pts: [SIMD3<Float>] = [SIMD3<Float>(0, 0, -5)]  // behind
    let range = DepthRangeEstimator.estimate(
        viewmat: vm, imageWidth: 128, imageHeight: 128,
        intrinsics: (100, 100, 64, 64), points: pts
    )
    #expect(range == nil)
}

/// Points outside the image are excluded. X=1000, z=10 gives u = 100*1000/10 + 64 = 10064 →
/// outside.
@Test
func depthRangeExcludesOutOfFrustum() {
    let vm: [Float] = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
    let pts: [SIMD3<Float>] = [
        SIMD3<Float>(0, 0, 5),        // inside the image
        SIMD3<Float>(1000, 0, 10),     // outside the image
    ]
    let range = DepthRangeEstimator.estimate(
        viewmat: vm, imageWidth: 128, imageHeight: 128,
        intrinsics: (100, 100, 64, 64), points: pts
    )
    // Only the z = 5 point is valid, so range = (5·0.5, 5·1.5).
    #expect(range != nil)
    if let r = range {
        #expect(abs(r.minDepth - 2.5) < 1e-4, "minDepth=\(r.minDepth)")
        #expect(abs(r.maxDepth - 7.5) < 1e-4, "maxDepth=\(r.maxDepth)")
    }
}
