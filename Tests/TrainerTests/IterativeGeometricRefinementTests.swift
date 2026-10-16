// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Iterative Geometric Refinement (PocketGS Operator G, end of Sec. III-A 2).

import Foundation
import simd
import Testing
@testable import Trainer

/// viewmat with camera center c and rotation I (row-major, t = -c).
private func makeViewmat(c: SIMD3<Float>) -> [Float] {
    return [
        1, 0, 0, -c.x,
        0, 1, 0, -c.y,
        0, 0, 1, -c.z,
        0, 0, 0, 1,
    ]
}

private func project(P: SIMD3<Float>, vm: [Float], fx: Float, fy: Float, cx: Float, cy: Float) -> SIMD2<Float>? {
    let xc = vm[0] * P.x + vm[1] * P.y + vm[2]  * P.z + vm[3]
    let yc = vm[4] * P.x + vm[5] * P.y + vm[6]  * P.z + vm[7]
    let zc = vm[8] * P.x + vm[9] * P.y + vm[10] * P.z + vm[11]
    guard zc > 1e-3 else { return nil }
    return SIMD2<Float>(fx * xc / zc + cx, fy * yc / zc + cy)
}

// MARK: - Basic DLT re-triangulation

/// With 2 cameras + 1 point and noise-free observations, DLT re-triangulation returns to GT from a
/// far-off initial value.
@Test
func retriangulateRecoversGroundTruth() {
    let intr: [Intrinsics] = [
        (200, 200, 128, 128), (200, 200, 128, 128),
    ]
    let viewmats = [
        makeViewmat(c: SIMD3<Float>(-0.5, 0, 0)),
        makeViewmat(c: SIMD3<Float>( 0.5, 0, 0)),
    ]
    let truePt = SIMD3<Float>(0.2, -0.1, 3.0)
    var observations: [BundleAdjustment.Observation] = []
    for cIdx in 0..<2 {
        if let p = project(P: truePt, vm: viewmats[cIdx],
                           fx: intr[cIdx].fx, fy: intr[cIdx].fy,
                           cx: intr[cIdx].cx, cy: intr[cIdx].cy) {
            observations.append(BundleAdjustment.Observation(
                cameraIndex: cIdx, pointIndex: 0, pixel: p
            ))
        }
    }
    #expect(observations.count == 2)

    // initial estimate deliberately far off
    let state = BundleAdjustment.State(
        viewmats: viewmats,
        points: [SIMD3<Float>(0, 0, 1)]
    )
    let triangulated = IterativeGeometricRefinement.retriangulate(
        state: state, intrinsics: intr, observations: observations
    )
    let est = triangulated[0]
    let err = ((est.x - truePt.x) * (est.x - truePt.x)
             + (est.y - truePt.y) * (est.y - truePt.y)
             + (est.z - truePt.z) * (est.z - truePt.z)).squareRoot()
    #expect(err < 1e-3, "triangulation error = \(err) at \(est) vs \(truePt)")
}

// MARK: - Drop observations behind the camera (z ≤ 0)

@Test
func filterBehindCameraDropsBehindObs() {
    let viewmats = [
        makeViewmat(c: SIMD3<Float>(0, 0, 0)),
    ]
    let pts = [
        SIMD3<Float>(0, 0,  3),   // in front
        SIMD3<Float>(0, 0, -3),   // behind
    ]
    let state = BundleAdjustment.State(viewmats: viewmats, points: pts)
    let obs = [
        BundleAdjustment.Observation(cameraIndex: 0, pointIndex: 0, pixel: .zero),
        BundleAdjustment.Observation(cameraIndex: 0, pointIndex: 1, pixel: .zero),
    ]
    let filtered = IterativeGeometricRefinement.filterBehindCamera(
        state: state, observations: obs
    )
    #expect(filtered.count == 1)
    #expect(filtered[0].pointIndex == 0)
}

// MARK: - Reprojection error filter

@Test
func filterReprojectionDropsBadObs() {
    let intr: [Intrinsics] = [
        (100, 100, 50, 50),
    ]
    let viewmats = [makeViewmat(c: SIMD3<Float>(0, 0, 0))]
    let state = BundleAdjustment.State(
        viewmats: viewmats,
        points: [SIMD3<Float>(0, 0, 1)]    // projects to (50, 50)
    )
    let obs = [
        BundleAdjustment.Observation(cameraIndex: 0, pointIndex: 0, pixel: SIMD2<Float>(50, 50)),   // OK
        BundleAdjustment.Observation(cameraIndex: 0, pointIndex: 0, pixel: SIMD2<Float>(60, 50)),   // error 10 px → dropped
        BundleAdjustment.Observation(cameraIndex: 0, pointIndex: 0, pixel: SIMD2<Float>(51, 51)),   // error ~1.4 px → kept
    ]
    let filtered = IterativeGeometricRefinement.filterReprojection(
        state: state, intrinsics: intr, observations: obs, thresholdPx: 2.0
    )
    #expect(filtered.count == 2)
    #expect(!filtered.contains { $0.pixel.x == 60 })
}

// MARK: - Triangulation angle filter

@Test
func filterTriangulationAngleDropsLowAngle() {
    let intr: [Intrinsics] = [
        (100, 100, 50, 50), (100, 100, 50, 50),
    ]
    // 2 cameras at nearly the same position and the point 100 m away → triangulation angle ≈ 0
    let viewmats = [
        makeViewmat(c: SIMD3<Float>(0, 0, 0)),
        makeViewmat(c: SIMD3<Float>(0.001, 0, 0)),
    ]
    let pts = [SIMD3<Float>(0, 0, 100)]
    let obs = [
        BundleAdjustment.Observation(cameraIndex: 0, pointIndex: 0, pixel: SIMD2<Float>(50, 50)),
        BundleAdjustment.Observation(cameraIndex: 1, pointIndex: 0, pixel: SIMD2<Float>(50, 50)),
    ]
    // under 1° → drop
    let filtered = IterativeGeometricRefinement.filterTriangulationAngle(
        points: pts, intrinsics: intr, observations: obs, viewmats: viewmats,
        minAngleRadians: 0.0174    // 1°
    )
    #expect(filtered.isEmpty, "low-angle point should be dropped")
}

@Test
func filterTriangulationAngleKeepsAdequateAngle() {
    let intr: [Intrinsics] = [
        (100, 100, 50, 50), (100, 100, 50, 50),
    ]
    let viewmats = [
        makeViewmat(c: SIMD3<Float>(-1, 0, 0)),
        makeViewmat(c: SIMD3<Float>( 1, 0, 0)),
    ]
    let pts = [SIMD3<Float>(0, 0, 3)]   // baseline 2 m, depth 3 m → 2·atan(1/3) ≈ 36.9° > 1°
    let obs = [
        BundleAdjustment.Observation(cameraIndex: 0, pointIndex: 0, pixel: SIMD2<Float>(50, 50)),
        BundleAdjustment.Observation(cameraIndex: 1, pointIndex: 0, pixel: SIMD2<Float>(50, 50)),
    ]
    let filtered = IterativeGeometricRefinement.filterTriangulationAngle(
        points: pts, intrinsics: intr, observations: obs, viewmats: viewmats,
        minAngleRadians: 0.0174
    )
    #expect(filtered.count == 2, "adequate angle should keep observations")
}

// MARK: - End-to-end refine loop

/// 3 cameras + 8 points + 1 outlier observation. Iterating drops the outlier and the final BA
/// residual is under 1 px.
@Test
func iterativeRefinementConvergesAndDropsOutlier() {
    let intr: [Intrinsics] = Array(
        repeating: (300, 300, 128, 128), count: 3
    )
    let viewmats = [
        makeViewmat(c: SIMD3<Float>(-0.6, 0, 0)),
        makeViewmat(c: SIMD3<Float>( 0.0, 0, 0)),
        makeViewmat(c: SIMD3<Float>( 0.6, 0, 0)),
    ]
    var pts: [SIMD3<Float>] = []
    for i in 0..<8 {
        let theta = Float(i) / 8.0 * .pi * 2
        pts.append(SIMD3<Float>(
            0.4 * cos(theta), 0.4 * sin(theta), 3 + Float(i % 3) * 0.5
        ))
    }
    var observations: [BundleAdjustment.Observation] = []
    for (cIdx, vm) in viewmats.enumerated() {
        for (pIdx, p) in pts.enumerated() {
            if let pix = project(P: p, vm: vm,
                                 fx: intr[cIdx].fx, fy: intr[cIdx].fy,
                                 cx: intr[cIdx].cx, cy: intr[cIdx].cy) {
                observations.append(BundleAdjustment.Observation(
                    cameraIndex: cIdx, pointIndex: pIdx, pixel: pix
                ))
            }
        }
    }
    // make one observation a strong outlier (+30 px)
    observations[0] = BundleAdjustment.Observation(
        cameraIndex: observations[0].cameraIndex,
        pointIndex: observations[0].pointIndex,
        pixel: SIMD2<Float>(observations[0].pixel.x + 30, observations[0].pixel.y)
    )

    // noise on the initial positions of cams 1 and 2
    var noisyViewmats = viewmats
    noisyViewmats[1][3] += 0.02
    noisyViewmats[2][3] -= 0.01

    let result = IterativeGeometricRefinement.refine(
        state: BundleAdjustment.State(viewmats: noisyViewmats, points: pts),
        intrinsics: intr, observations: observations,
        parameters: IterativeGeometricRefinement.Parameters(
            maxRounds: 3, reprojectionThresholdPx: 3.0
        ),
        log: { _ in }
    )

    let initialObs = observations.count
    let finalObs = result.observations.count
    #expect(finalObs < initialObs, "iterative refinement should drop outlier observations: \(initialObs) → \(finalObs)")
    if let lastRound = result.rounds.last {
        #expect(lastRound.baResidualPx < 1.0,
            "final BA residual should drop below 1 px, got \(lastRound.baResidualPx)")
    }
}
