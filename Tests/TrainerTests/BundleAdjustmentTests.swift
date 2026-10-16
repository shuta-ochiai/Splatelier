// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Bundle Adjustment (PocketGS Operator G). Adds pose noise to observations of synthetic cameras +
// points and runs BA. Camera 0 is gauge-fixed, so what is checked is relative-pose consistency, not
// absolute pose.

import Foundation
import Testing
@testable import Trainer

private func makeIdentityViewmat() -> [Float] {
    [1, 0, 0, 0,
     0, 1, 0, 0,
     0, 0, 1, 0,
     0, 0, 0, 1]
}

/// Builds a viewmat from camera center c and w2c rotation R (3×3 row-major) (t = -R·c).
private func makeViewmat(camCenter: SIMD3<Float>, rotation: [Float]) -> [Float] {
    let r = rotation
    let tx = -(r[0] * camCenter.x + r[1] * camCenter.y + r[2] * camCenter.z)
    let ty = -(r[3] * camCenter.x + r[4] * camCenter.y + r[5] * camCenter.z)
    let tz = -(r[6] * camCenter.x + r[7] * camCenter.y + r[8] * camCenter.z)
    return [
        r[0], r[1], r[2], tx,
        r[3], r[4], r[5], ty,
        r[6], r[7], r[8], tz,
        0, 0, 0, 1,
    ]
}

private func projectPoint(
    point: SIMD3<Float>, viewmat: [Float],
    fx: Float, fy: Float, cx: Float, cy: Float
) -> SIMD2<Float>? {
    let pcx = viewmat[0] * point.x + viewmat[1] * point.y + viewmat[2]  * point.z + viewmat[3]
    let pcy = viewmat[4] * point.x + viewmat[5] * point.y + viewmat[6]  * point.z + viewmat[7]
    let pcz = viewmat[8] * point.x + viewmat[9] * point.y + viewmat[10] * point.z + viewmat[11]
    guard pcz > 1e-3 else { return nil }
    return SIMD2<Float>(fx * pcx / pcz + cx, fy * pcy / pcz + cy)
}

/// With 3 cameras + 12 points, adding noise to the translations of cams 1 and 2 (and to the points)
/// and running 5 iters drops the residual below 1/10, while camera 0 stays fixed.
@Test
func bundleAdjustmentConvergesOnSynthetic3Cam12Pt() {
    // ----- True poses -----
    let identity: [Float] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    let trueVMs: [[Float]] = [
        makeViewmat(camCenter: SIMD3<Float>(-1, 0, 0), rotation: identity),
        makeViewmat(camCenter: SIMD3<Float>( 0, 0, 0), rotation: identity),
        makeViewmat(camCenter: SIMD3<Float>( 1, 0, 0), rotation: identity),
    ]
    let intrinsics: [Intrinsics] =
        Array(repeating: (200, 200, 128, 128), count: 3)

    // ----- True 3D points (12 points ahead at z = 3-5) -----
    var truePoints: [SIMD3<Float>] = []
    for i in 0..<12 {
        let theta = Float(i) / 12.0 * .pi * 2
        let r: Float = 0.6
        truePoints.append(SIMD3<Float>(r * cos(theta), r * sin(theta), 3 + Float(i % 3) * 0.5))
    }

    // ----- Generate observations -----
    var observations: [BundleAdjustment.Observation] = []
    for (camIdx, vm) in trueVMs.enumerated() {
        for (ptIdx, p) in truePoints.enumerated() {
            if let pix = projectPoint(
                point: p, viewmat: vm,
                fx: intrinsics[camIdx].fx, fy: intrinsics[camIdx].fy,
                cx: intrinsics[camIdx].cx, cy: intrinsics[camIdx].cy
            ) {
                observations.append(BundleAdjustment.Observation(
                    cameraIndex: camIdx, pointIndex: ptIdx, pixel: pix
                ))
            }
        }
    }

    // ----- Noise on cam 1, 2 translations (camera 0 stays at ground truth, gauge-fixed) -----
    var noisyVMs = trueVMs
    noisyVMs[1][3] += 0.05; noisyVMs[1][7] += 0.03
    noisyVMs[2][3] -= 0.08; noisyVMs[2][11] += 0.06

    // Light noise on the points too
    var noisyPoints = truePoints
    for i in 0..<noisyPoints.count {
        noisyPoints[i].x += 0.02
        noisyPoints[i].y -= 0.01
    }

    let state0 = BundleAdjustment.State(viewmats: noisyVMs, points: noisyPoints)
    let params = BundleAdjustment.Parameters(iterations: 5, freezeFirstCamera: true)
    let result = BundleAdjustment.refine(
        state: state0, intrinsics: intrinsics,
        observations: observations, parameters: params
    )

    // The residual drops below 1/10 of the initial value.
    let firstResidual = result.residualNorms.first ?? 0
    let lastResidual = result.residualNorms.last ?? 0
    #expect(firstResidual > 0.1, "noise should give meaningful initial residual; got \(firstResidual)")
    #expect(lastResidual < firstResidual * 0.1,
        "BA should reduce residual by 10×; firstResidual=\(firstResidual), lastResidual=\(lastResidual)")
    // Camera 0 is unchanged due to the gauge fix.
    for k in 0..<16 {
        #expect(abs(result.state.viewmats[0][k] - trueVMs[0][k]) < 1e-5,
            "camera 0 should be frozen; element \(k) drifted")
    }
}

/// In the minimal setup (2 cams, 4 points, noise on cam1), one iter improves the residual.
@Test
func bundleAdjustmentSingleStepReducesResidual() {
    let identity: [Float] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    let trueVMs: [[Float]] = [
        makeViewmat(camCenter: SIMD3<Float>(-0.5, 0, 0), rotation: identity),
        makeViewmat(camCenter: SIMD3<Float>( 0.5, 0, 0), rotation: identity),
    ]
    let intrinsics: [Intrinsics] = [
        (150, 150, 64, 64), (150, 150, 64, 64),
    ]
    let truePoints: [SIMD3<Float>] = [
        SIMD3<Float>(0, 0, 3),
        SIMD3<Float>(0.4, 0.2, 4),
        SIMD3<Float>(-0.3, 0.5, 3.5),
        SIMD3<Float>(0.1, -0.4, 4.2),
    ]
    var observations: [BundleAdjustment.Observation] = []
    for (camIdx, vm) in trueVMs.enumerated() {
        for (ptIdx, p) in truePoints.enumerated() {
            if let pix = projectPoint(
                point: p, viewmat: vm,
                fx: intrinsics[camIdx].fx, fy: intrinsics[camIdx].fy,
                cx: intrinsics[camIdx].cx, cy: intrinsics[camIdx].cy
            ) {
                observations.append(BundleAdjustment.Observation(
                    cameraIndex: camIdx, pointIndex: ptIdx, pixel: pix
                ))
            }
        }
    }
    var noisyVMs = trueVMs
    noisyVMs[1][3] += 0.03    // small noise on cam 1 translation
    let state0 = BundleAdjustment.State(viewmats: noisyVMs, points: truePoints)
    let params = BundleAdjustment.Parameters(iterations: 1, freezeFirstCamera: true)
    let result = BundleAdjustment.refine(
        state: state0, intrinsics: intrinsics, observations: observations, parameters: params
    )
    let before = result.residualNorms[0]
    let after = result.residualNorms[1]
    #expect(before > 0.1, "noise should produce measurable residual; got \(before)")
    #expect(after < before * 0.5,
        "single GN step should halve residual; before=\(before), after=\(after)")
}

// MARK: - Huber / LM / scale-aware gauge

/// With one observation turned into a +200 px outlier, Huber gives a pose closer to ground truth
/// than pure L2.
@Test
func bundleAdjustmentHuberSurpressesOutlier() {
    let identity: [Float] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    let trueVMs: [[Float]] = [
        makeViewmat(camCenter: SIMD3<Float>(-1, 0, 0), rotation: identity),
        makeViewmat(camCenter: SIMD3<Float>( 0, 0, 0), rotation: identity),
        makeViewmat(camCenter: SIMD3<Float>( 1, 0, 0), rotation: identity),
    ]
    let intr: [Intrinsics] =
        Array(repeating: (200, 200, 128, 128), count: 3)
    var truePoints: [SIMD3<Float>] = []
    for i in 0..<12 {
        let theta = Float(i) / 12.0 * .pi * 2
        truePoints.append(SIMD3<Float>(0.6 * cos(theta), 0.6 * sin(theta), 3 + Float(i % 3) * 0.5))
    }
    var observations: [BundleAdjustment.Observation] = []
    for (camIdx, vm) in trueVMs.enumerated() {
        for (ptIdx, p) in truePoints.enumerated() {
            if let pix = projectPoint(
                point: p, viewmat: vm,
                fx: intr[camIdx].fx, fy: intr[camIdx].fy,
                cx: intr[camIdx].cx, cy: intr[camIdx].cy
            ) {
                observations.append(BundleAdjustment.Observation(
                    cameraIndex: camIdx, pointIndex: ptIdx, pixel: pix
                ))
            }
        }
    }
    // Make one observation a strong outlier (shift its image coordinates by +200 px)
    if !observations.isEmpty {
        let last = observations.removeFirst()
        observations.insert(
            BundleAdjustment.Observation(
                cameraIndex: last.cameraIndex, pointIndex: last.pointIndex,
                pixel: SIMD2<Float>(last.pixel.x + 200, last.pixel.y + 200)
            ), at: 0
        )
    }
    var noisyVMs = trueVMs
    noisyVMs[1][3] += 0.02
    noisyVMs[2][3] -= 0.02
    let state0 = BundleAdjustment.State(viewmats: noisyVMs, points: truePoints)

    // Compare after one GN step (multi-iteration L2 breaks linearization on the outlier, and
    // results vary with GPU atomic add order).
    func refine(huber: Float?) -> BundleAdjustment.Result {
        BundleAdjustment.refine(
            state: state0, intrinsics: intr, observations: observations,
            parameters: BundleAdjustment.Parameters(
                iterations: 1, freezeFirstCamera: true,
                huberThreshold: huber, lmInitialDamping: 0.01
            )
        )
    }
    let resPureL2 = refine(huber: nil)
    let resHuber = refine(huber: 2.0)

    // Compare by error against the true cam 2 translation.
    let trueT2 = SIMD3<Float>(trueVMs[2][3], trueVMs[2][7], trueVMs[2][11])
    func tDiff(_ vms: [[Float]]) -> Float {
        let dx = vms[2][3] - trueT2.x
        let dy = vms[2][7] - trueT2.y
        let dz = vms[2][11] - trueT2.z
        return (dx * dx + dy * dy + dz * dz).squareRoot()
    }
    let errL2 = tDiff(resPureL2.state.viewmats)
    let errHuber = tDiff(resHuber.state.viewmats)
    #expect(errHuber < errL2,
        "Huber should beat pure L2 in presence of outlier; errHuber=\(errHuber), errL2=\(errL2)")
}

/// Even with a large initial offset where linearization is poor (cam 1 translation 0.3 m), LM
/// damping (decaying from λ=10) does not diverge, and the residual decreases nearly monotonically
/// to convergence.
@Test
func bundleAdjustmentLMDampingHandlesLargePerturbation() {
    let identity: [Float] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    let trueVMs: [[Float]] = [
        makeViewmat(camCenter: SIMD3<Float>(-0.7, 0, 0), rotation: identity),
        makeViewmat(camCenter: SIMD3<Float>( 0.7, 0, 0), rotation: identity),
    ]
    let intr: [Intrinsics] = [(200, 200, 128, 128), (200, 200, 128, 128)]
    var truePoints: [SIMD3<Float>] = []
    for i in 0..<10 {
        let theta = Float(i) / 10.0 * .pi * 2
        truePoints.append(SIMD3<Float>(0.5 * cos(theta), 0.5 * sin(theta), 3.5))
    }
    var observations: [BundleAdjustment.Observation] = []
    for (camIdx, vm) in trueVMs.enumerated() {
        for (ptIdx, p) in truePoints.enumerated() {
            if let pix = projectPoint(
                point: p, viewmat: vm,
                fx: intr[camIdx].fx, fy: intr[camIdx].fy,
                cx: intr[camIdx].cx, cy: intr[camIdx].cy
            ) {
                observations.append(BundleAdjustment.Observation(
                    cameraIndex: camIdx, pointIndex: ptIdx, pixel: pix
                ))
            }
        }
    }
    var noisyVMs = trueVMs
    // Shift cam 1 translation by 0.3 m (far larger than a ~1 px residual)
    noisyVMs[1][3] += 0.3
    let state0 = BundleAdjustment.State(viewmats: noisyVMs, points: truePoints)

    let resLM = BundleAdjustment.refine(
        state: state0, intrinsics: intr, observations: observations,
        parameters: BundleAdjustment.Parameters(
            iterations: 6, freezeFirstCamera: true,
            lmInitialDamping: 10.0, lmDecay: 0.4
        )
    )
    let firstRes = resLM.residualNorms.first ?? 0
    let lastRes = resLM.residualNorms.last ?? 0
    #expect(firstRes > 1.0, "0.3m noise should produce large initial residual; got \(firstRes)")
    #expect(lastRes < firstRes * 0.2,
        "LM should still converge from large perturbation; firstRes=\(firstRes), lastRes=\(lastRes)")
    // Nearly monotonic decrease (up to ×1.1 allowed). Pairs both below the floor (1e-4) are float
    // noise, so skip.
    let convergenceFloor: Float = 1e-4
    for i in 1..<resLM.residualNorms.count {
        let prev = resLM.residualNorms[i - 1]
        let cur = resLM.residualNorms[i]
        if cur < convergenceFloor && prev < convergenceFloor { continue }
        #expect(cur < prev * 1.1, "iter \(i): residual increased from \(prev) to \(cur)")
    }
}

/// The scale-aware gauge removes only the baseline-axis (cam0→cam1) component from camera 1's
/// translation update. With camera 0 fixed, the only gauge freedom left is the global scale ("shift
/// cam 1 along the baseline and scale all point depths proportionally", a null direction for
/// reprojection error); the two orthogonal components are set by the data and converge to ground
/// truth. Baseline on the x axis; noise on cam 1's x (gauge) and y/z (orthogonal).
@Test
func bundleAdjustmentScaleAwareGaugeConstrainsCam1Translation() {
    let identity: [Float] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    let trueVMs: [[Float]] = [
        makeViewmat(camCenter: SIMD3<Float>(-0.5, 0, 0), rotation: identity),
        makeViewmat(camCenter: SIMD3<Float>( 0.5, 0, 0), rotation: identity),
    ]
    let intr: [Intrinsics] = [(150, 150, 64, 64), (150, 150, 64, 64)]
    var truePoints: [SIMD3<Float>] = []
    for i in 0..<8 {
        truePoints.append(SIMD3<Float>(Float(i % 3 - 1) * 0.3, Float(i / 3 - 1) * 0.3, 3.0))
    }
    var observations: [BundleAdjustment.Observation] = []
    for (camIdx, vm) in trueVMs.enumerated() {
        for (ptIdx, p) in truePoints.enumerated() {
            if let pix = projectPoint(
                point: p, viewmat: vm,
                fx: intr[camIdx].fx, fy: intr[camIdx].fy,
                cx: intr[camIdx].cx, cy: intr[camIdx].cy
            ) {
                observations.append(BundleAdjustment.Observation(
                    cameraIndex: camIdx, pointIndex: ptIdx, pixel: pix
                ))
            }
        }
    }
    var noisyVMs = trueVMs
    noisyVMs[1][3] += 0.05    // t.x += 0.05  (baseline axis = scale gauge direction)
    noisyVMs[1][7] += 0.05    // t.y += 0.05  (orthogonal)
    noisyVMs[1][11] += 0.05   // t.z += 0.05  (orthogonal)

    let state0 = BundleAdjustment.State(viewmats: noisyVMs, points: truePoints)
    let result = BundleAdjustment.refine(
        state: state0, intrinsics: intr, observations: observations,
        parameters: BundleAdjustment.Parameters(
            iterations: 3, freezeFirstCamera: true, scaleAwareGauge: true
        )
    )
    let initTx = noisyVMs[1][3]
    let finalTx = result.state.viewmats[1][3]
    // The true cam1 viewmat translation is t = -R·c = -(0.5, 0, 0), so the true y/z are 0.
    let trueTy = trueVMs[1][7]
    let trueTz = trueVMs[1][11]
    let finalTy = result.state.viewmats[1][7]
    let finalTz = result.state.viewmats[1][11]

    // (1) The 0.05 baseline-axis noise is removed as gauge and remains (if it got canceled, the
    // gauge is not working). However, the y/z noise tilts the baseline ~3° off world x (bhat ≈
    // (0.997, -0.052, -0.052)), and the y/z correction and rotation terms leak a few mm into x, so
    // it is not perfectly fixed. Threshold is 1/5 of the noise (1e-2).
    #expect(abs(finalTx - initTx) < 1e-2,
        "gauge should pin cam1 t.x (baseline axis); init=\(initTx), final=\(finalTx)")

    // (2) The orthogonal components are set by the data and converge to the true 0. With perfect
    // data this reaches near machine precision, but the threshold is 1e-4 (1/500 of the noise) to
    // allow for float32 GN platform differences.
    #expect(abs(finalTy - trueTy) < 1e-4,
        "orthogonal t.y must converge to truth; true=\(trueTy), final=\(finalTy)")
    #expect(abs(finalTz - trueTz) < 1e-4,
        "orthogonal t.z must converge to truth; true=\(trueTz), final=\(finalTz)")

    // (3) The real invariant: the scale gauge fixes the baseline length (the global scene scale).
    // t.x is only a proxy, so check directly that the baseline length is preserved across BA.
    let c0Final = FrameSubsampler.cameraCenter(viewmat: result.state.viewmats[0])
    let c1Final = FrameSubsampler.cameraCenter(viewmat: result.state.viewmats[1])
    let c0Init = FrameSubsampler.cameraCenter(viewmat: noisyVMs[0])
    let c1Init = FrameSubsampler.cameraCenter(viewmat: noisyVMs[1])
    func baselineLength(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
        let d = SIMD3<Float>(b.x - a.x, b.y - a.y, b.z - a.z)
        return (d.x * d.x + d.y * d.y + d.z * d.z).squareRoot()
    }
    let initBaseline = baselineLength(c0Init, c1Init)
    let finalBaseline = baselineLength(c0Final, c1Final)
    // For the same reason as (1) it is not exactly preserved (a few mm). Without the gauge the x
    // noise passes through and changes it by ~5%, so 1e-2 cleanly separates working from not
    // working.
    #expect(abs(finalBaseline - initBaseline) < 1e-2,
        "scale gauge must preserve baseline length; init=\(initBaseline), final=\(finalBaseline)")
}

/// The left SE(3) perturbation (Rodrigues) keeps the rotation orthonormal and adds v to the
/// translation (checks sign/axis convention).
@Test
func bundleAdjustmentLeftPerturbationRoundTrip() {
    let vm: [Float] = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
    let omega = SIMD3<Float>(0.1, -0.2, 0.05)
    let v = SIMD3<Float>(0.3, 0.1, -0.2)
    let perturbed = BundleAdjustment.applyLeftSE3Perturbation(viewmat: vm, omega: omega, v: v)
    // Row norms of R stay 1.
    let r00 = perturbed[0], r01 = perturbed[1], r02 = perturbed[2]
    let lenRow0 = sqrt(r00 * r00 + r01 * r01 + r02 * r02)
    #expect(abs(lenRow0 - 1.0) < 1e-4, "rotation row 0 norm = \(lenRow0)")
    // Translation is a plain add (small angle V≈I)
    #expect(abs(perturbed[3] - v.x) < 1e-4)
    #expect(abs(perturbed[7] - v.y) < 1e-4)
    #expect(abs(perturbed[11] - v.z) < 1e-4)
}
