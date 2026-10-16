// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Regression test: on a synthetic scene at capture-data scale (focal length ~760px, subject ~1m,
// over 1000 points = GPU Schur path), the IGR-default Marquardt LM accepts steps and moves poses
// toward ground truth. With the old setting (adding an absolute λ=1 to the reduced system's
// diagonal), at this scale the damping was effectively zero against Hessian diagonals of ~1e6,
// every step was rejected and poses never moved (2026-10-08).

import Foundation
import Testing
@testable import Trainer

private struct LCG {
    var s: UInt64
    mutating func next() -> Float {
        s = s &* 6364136223846793005 &+ 1442695040888963407
        return Float(s >> 40) / Float(1 << 24)
    }
    mutating func gauss() -> Float {
        let u = max(next(), 1e-7), v = next()
        return (-2 * log(u)).squareRoot() * cos(2 * .pi * v)
    }
}

/// Rotation matrix about axis a by deg degrees (row-major 3×3).
private func rotation(axis a: SIMD3<Float>, deg: Float) -> [Float] {
    let n = a / (a * a).sum().squareRoot()
    let t = deg * .pi / 180, c = cos(t), s = sin(t), C = 1 - c
    return [
        c + n.x * n.x * C, n.x * n.y * C - n.z * s, n.x * n.z * C + n.y * s,
        n.y * n.x * C + n.z * s, c + n.y * n.y * C, n.y * n.z * C - n.x * s,
        n.z * n.x * C - n.y * s, n.z * n.y * C + n.x * s, c + n.z * n.z * C,
    ]
}

private func mul3(_ a: [Float], _ b: [Float]) -> [Float] {
    var r = [Float](repeating: 0, count: 9)
    for i in 0..<3 { for j in 0..<3 { for k in 0..<3 { r[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j] } } }
    return r
}

/// Builds a row-major viewmat from world→camera rotation R and camera center c.
private func viewmat(R: [Float], center c: SIMD3<Float>) -> [Float] {
    let t = SIMD3<Float>(
        -(R[0] * c.x + R[1] * c.y + R[2] * c.z),
        -(R[3] * c.x + R[4] * c.y + R[5] * c.z),
        -(R[6] * c.x + R[7] * c.y + R[8] * c.z))
    return [R[0], R[1], R[2], t.x, R[3], R[4], R[5], t.y, R[6], R[7], R[8], t.z, 0, 0, 0, 1]
}

/// Relative rotation R_b·R_aᵀ of viewmats a→b, returned as a rotation-only viewmat.
private func relRot(_ a: [Float], _ b: [Float]) -> [Float] {
    var r = [Float](repeating: 0, count: 16)
    for i in 0..<3 { for j in 0..<3 { for k in 0..<3 { r[i * 4 + j] += b[i * 4 + k] * a[j * 4 + k] } } }
    r[15] = 1
    return r
}

/// Rotation difference between two viewmats (degrees).
private func rotationErrorDeg(_ a: [Float], _ b: [Float]) -> Float {
    // trace(Ra · Rbᵀ)
    var tr: Float = 0
    for i in 0..<3 { for k in 0..<3 { tr += a[i * 4 + k] * b[i * 4 + k] } }
    return acos(max(-1, min(1, (tr - 1) / 2))) * 180 / .pi
}

@Test
func marquardtBARefinesPhoneScalePosesOnGPUSchurPath() {
    var rng = LCG(s: 42)
    let fx: Float = 760, cx: Float = 540, cy: Float = 405
    let nCam = 12, nPt = 3000
    // Cameras lie on an arc looking at the subject (near the origin, depth ~0.8 m). View is +z,
    // rotating gradually about y.
    var trueVMs: [[Float]] = []
    for i in 0..<nCam {
        let a = (Float(i) - Float(nCam - 1) / 2) * 2.5          // degrees
        let R = rotation(axis: SIMD3(0, 1, 0), deg: a)
        let c = SIMD3<Float>(-0.8 * sin(a * .pi / 180), 0.01 * Float(i % 3), 0.8 - 0.8 * cos(a * .pi / 180))
        trueVMs.append(viewmat(R: R, center: c))
    }
    // Scatter points in a box at depth 0.6-1.1 m.
    var truePts: [SIMD3<Float>] = []
    for _ in 0..<nPt {
        truePts.append(SIMD3((rng.next() - 0.5) * 0.6, (rng.next() - 0.5) * 0.45, 0.6 + rng.next() * 0.5))
    }
    // As with real ORB matches, half the points are seen only by 2 consecutive cameras (small
    // parallax, so the per-point Hessian H_pp is nearly singular). The other half are tracks over 4
    // consecutive cameras (like ARKit tracks / loop matches; they pin the poses down uniquely). 5%
    // of observations are outliers off by 20-40 px.
    var obs: [BundleAdjustment.Observation] = []
    for (pi, p) in truePts.enumerated() {
        let span = pi % 2 == 0 ? 2 : 4
        let c0 = (pi / 2) % (nCam - span + 1)
        for ci in c0..<(c0 + span) {
            let v = trueVMs[ci]
            let x = v[0] * p.x + v[1] * p.y + v[2] * p.z + v[3]
            let y = v[4] * p.x + v[5] * p.y + v[6] * p.z + v[7]
            let z = v[8] * p.x + v[9] * p.y + v[10] * p.z + v[11]
            guard z > 0.1 else { continue }
            var u = fx * x / z + cx + 0.3 * rng.gauss(), w = fx * y / z + cy + 0.3 * rng.gauss()
            if rng.next() < 0.05 { u += 20 + 20 * rng.next(); w -= 20 + 20 * rng.next() }
            guard u >= 0, u < 1080, w >= 0, w < 810 else { continue }
            obs.append(.init(cameraIndex: ci, pointIndex: pi, pixel: SIMD2(u, w)))
        }
    }
    // 0.3° rotation and 5 mm translation noise on every pose except camera 0 (gauge-fixed). 5 mm on
    // points too.
    var initVMs = trueVMs
    for i in 1..<nCam {
        let dR = rotation(axis: SIMD3(rng.gauss(), rng.gauss(), rng.gauss()), deg: 0.3)
        let v = trueVMs[i]
        let R = mul3(dR, [v[0], v[1], v[2], v[4], v[5], v[6], v[8], v[9], v[10]])
        let c = SIMD3<Float>(
            -(v[0] * v[3] + v[4] * v[7] + v[8] * v[11]),
            -(v[1] * v[3] + v[5] * v[7] + v[9] * v[11]),
            -(v[2] * v[3] + v[6] * v[7] + v[10] * v[11])) + 0.005 * SIMD3(rng.gauss(), rng.gauss(), rng.gauss())
        initVMs[i] = viewmat(R: R, center: c)
    }
    let initPts = truePts.map { $0 + 0.005 * SIMD3(rng.gauss(), rng.gauss(), rng.gauss()) }
    let intr = Array(repeating: (fx: fx, fy: fx, cx: cx, cy: cy), count: nCam)

    // Run the same IGR as production (BA → re-triangulation → outlier removal, up to 3 rounds).
    let params = IterativeGeometricRefinement.Parameters()
    #expect(params.baParameters.dampingMode == .marquardt)
    let r0 = BundleAdjustment.residualNorm(
        state: .init(viewmats: initVMs, points: initPts), intrinsics: intr, observations: obs)
    let result = IterativeGeometricRefinement.refine(
        state: .init(viewmats: initVMs, points: initPts),
        intrinsics: intr, observations: obs, parameters: params, log: { _ in })
    let r1 = BundleAdjustment.residualNorm(
        state: result.state, intrinsics: intr, observations: result.observations)
    #expect(r1 < 0.5 * r0, "residual \(r0) → \(r1)")
    #expect(r1 < 0.6, "final residual \(r1) px (noise 0.3 px)")
    // Poses actually moved (with the old setting every step was rejected and not a single bit
    // changed).
    let moved = (1..<nCam).map { rotationErrorDeg(result.state.viewmats[$0], initVMs[$0]) }.max() ?? 0
    #expect(moved > 0.05, "max pose change \(moved)°")
    // The relative rotation error between adjacent cameras improves. With a narrow field of view
    // and points linked only across nearby frames, absolute rotation vs ground truth can accumulate
    // error along the chain (several solutions have nearly the same residual), so evaluate the
    // relative rotation between neighbors, which the problem determines.
    func relErr(_ vms: [[Float]]) -> Float {
        (0..<(nCam - 1)).map { i -> Float in
            rotationErrorDeg(relRot(vms[i], vms[i + 1]), relRot(trueVMs[i], trueVMs[i + 1]))
        }.reduce(0, +) / Float(nCam - 1)
    }
    let relBefore = relErr(initVMs), relAfter = relErr(result.state.viewmats)
    #expect(relAfter < 0.5 * relBefore, "mean adjacent relative rotation error \(relBefore)° → \(relAfter)°")
}

/// A camera with very few, very noisy observations has an orientation the data alone cannot fix.
/// With the weak prior on ARKit poses (IGR default), that camera stays near its prior pose.
@Test
func posePriorHoldsWeaklyObservedCamera() {
    var rng = LCG(s: 11)
    let fx: Float = 760, cx: Float = 540, cy: Float = 405
    let nCam = 6
    var trueVMs: [[Float]] = []
    for i in 0..<nCam {
        let a = (Float(i) - 2.5) * 4
        trueVMs.append(viewmat(R: rotation(axis: SIMD3(0, 1, 0), deg: a),
                               center: SIMD3(-0.8 * sin(a * .pi / 180), 0, 0.8 - 0.8 * cos(a * .pi / 180))))
    }
    var pts: [SIMD3<Float>] = []
    for _ in 0..<1500 { pts.append(SIMD3((rng.next() - 0.5) * 0.6, (rng.next() - 0.5) * 0.45, 0.6 + rng.next() * 0.5)) }
    var obs: [BundleAdjustment.Observation] = []
    for (pi, p) in pts.enumerated() {
        for ci in 0..<nCam {
            // Camera 5 observes only the first 6 points, with 3 px noise (weak constraint).
            let weak = ci == nCam - 1
            if weak && pi >= 6 { continue }
            let v = trueVMs[ci]
            let x = v[0] * p.x + v[1] * p.y + v[2] * p.z + v[3]
            let y = v[4] * p.x + v[5] * p.y + v[6] * p.z + v[7]
            let z = v[8] * p.x + v[9] * p.y + v[10] * p.z + v[11]
            let sd: Float = weak ? 3 : 0.3
            obs.append(.init(cameraIndex: ci, pointIndex: pi,
                             pixel: SIMD2(fx * x / z + cx + sd * rng.gauss(), fx * y / z + cy + sd * rng.gauss())))
        }
    }
    // Initial poses: only camera 5 is off ground truth, by 0.4°. The prior poses are ground truth.
    var initVMs = trueVMs
    let v = trueVMs[nCam - 1]
    let R = mul3(rotation(axis: SIMD3(1, 0.3, -0.2), deg: 0.4), [v[0], v[1], v[2], v[4], v[5], v[6], v[8], v[9], v[10]])
    initVMs[nCam - 1] = viewmat(R: R, center: SIMD3(
        -(v[0] * v[3] + v[4] * v[7] + v[8] * v[11]),
        -(v[1] * v[3] + v[5] * v[7] + v[9] * v[11]),
        -(v[2] * v[3] + v[6] * v[7] + v[10] * v[11])))
    let intr = Array(repeating: (fx: fx, fy: fx, cx: cx, cy: cy), count: nCam)
    func run(_ p: BundleAdjustment.Parameters) -> Float {
        let r = BundleAdjustment.refine(state: .init(viewmats: initVMs, points: pts), intrinsics: intr,
                                        observations: obs, parameters: p, priorViewmats: trueVMs)
        return rotationErrorDeg(r.state.viewmats[nCam - 1], trueVMs[nCam - 1])
    }
    var withPrior = IterativeGeometricRefinement.Parameters().baParameters
    withPrior.iterations = 20
    var noPrior = withPrior
    noPrior.priorRotationSigmaDeg = nil; noPrior.priorPositionSigma = nil
    let eWith = run(withPrior), eWithout = run(noPrior)
    // Without the prior, the 6 noisy observations can pull it worse than the initial 0.4°. With the
    // prior it is held toward the prior pose (= ground truth): smaller error than without the
    // prior, and no worse than initial.
    #expect(eWith < 0.8 * eWithout, "with prior \(eWith)° vs without \(eWithout)°")
    #expect(eWith < 0.4, "weak camera rotation error with prior: \(eWith)° (initial 0.4°)")
}
