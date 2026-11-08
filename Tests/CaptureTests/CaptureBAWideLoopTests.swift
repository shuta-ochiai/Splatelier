// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Wide-angle loop closure: the relative pose fit near a prior, and the pose graph that closes the
// loop before bundle adjustment.

import Foundation
import Testing
import simd
@testable import Trainer
@testable import Capture

private func rotationY(_ deg: Double) -> simd_double3x3 {
    let t = deg * .pi / 180
    return simd_double3x3(rows: [SIMD3(cos(t), 0, sin(t)), SIMD3(0, 1, 0), SIMD3(-sin(t), 0, cos(t))])
}

private func rotation(axis: SIMD3<Double>, deg: Double) -> simd_double3x3 {
    let k = simd_normalize(axis), t = deg * .pi / 180
    let K = simd_double3x3(rows: [SIMD3(0, -k.z, k.y), SIMD3(k.z, 0, -k.x), SIMD3(-k.y, k.x, 0)])
    return matrix_identity_double3x3 + sin(t) * K + (1 - cos(t)) * (K * K)
}

private func angleDeg(_ a: simd_double3x3, _ b: simd_double3x3) -> Double {
    let m = a.transpose * b
    return acos(max(-1, min(1, (m[0][0] + m[1][1] + m[2][2] - 1) / 2))) * 180 / .pi
}

/// Row-major world→camera viewmat from rotation R (world→camera) and center C.
private func viewmat(_ R: simd_double3x3, _ C: SIMD3<Double>) -> [Float] {
    let t = -(R * C)
    return [Float(R[0][0]), Float(R[1][0]), Float(R[2][0]), Float(t.x),
            Float(R[0][1]), Float(R[1][1]), Float(R[2][1]), Float(t.y),
            Float(R[0][2]), Float(R[1][2]), Float(R[2][2]), Float(t.z), 0, 0, 0, 1]
}

/// Two cameras on opposite sides of a floor patch, looking down at it; matches of the patch, plus
/// a second group that only a pose turned 180° explains (the same lettering on the far face of a
/// sign). With a prior 3° off the truth, the fit keeps the patch and drops the other group.
@Test
func relativePoseFitKeepsMatchesNearThePriorAndDropsAFlippedGroup() throws {
    let K = (fx: Float(750), fy: Float(750), cx: Float(540), cy: Float(405))
    // Camera a looks along +z tilted down; camera b is camera a turned 160° about the vertical
    // through the patch center.
    let P = SIMD3<Double>(0, 1, 2)
    let Ra = rotation(axis: SIMD3(1, 0, 0), deg: -35)
    let Ca = SIMD3<Double>(0, 0, 0)
    let turn = rotationY(160)
    let Rb = Ra * turn.transpose
    let Cb = P + turn * (Ca - P)
    func project(_ R: simd_double3x3, _ C: SIMD3<Double>, _ X: SIMD3<Double>) -> SIMD2<Float> {
        let x = R * (X - C)
        return SIMD2(Float(Double(K.fx) * x.x / x.z + Double(K.cx)), Float(Double(K.fy) * x.y / x.z + Double(K.cy)))
    }
    var rng = SplitMix64(seed: 7)
    var uvA: [SIMD2<Float>] = [], uvB: [SIMD2<Float>] = []
    // 60 points on the floor patch (y = 1 plane), seen by both.
    for _ in 0..<60 {
        let X = P + SIMD3(Double.random(in: -0.3...0.3, using: &rng), 0, Double.random(in: -0.3...0.3, using: &rng))
        let noise = { SIMD2<Float>(Float.random(in: -0.5...0.5, using: &rng), Float.random(in: -0.5...0.5, using: &rng)) }
        uvA.append(project(Ra, Ca, X) + noise()); uvB.append(project(Rb, Cb, X) + noise())
    }
    // 40 wrong matches: a's points paired with the pixels a camera on a's own side would see
    // (b's image of the "other face"), i.e. consistent with b at a's pose shifted sideways.
    let Cfake = Ca + SIMD3(0.4, 0, 0)
    for _ in 0..<40 {
        let X = P + SIMD3(Double.random(in: -0.3...0.3, using: &rng), Double.random(in: -0.6...(-0.1), using: &rng), 0.35)
        uvA.append(project(Ra, Ca, X)); uvB.append(project(Ra, Cfake, X))
    }
    let trueR = Rb * Ra.transpose
    let trueT = Rb * (Ca - Cb)   // x_b = R x_a + t
    let priorR = rotation(axis: SIMD3(0.2, 1, 0.1), deg: 3) * trueR
    let fit = try #require(CaptureBA.fitRelativePoseNearPrior(
        uvA: uvA, uvB: uvB, intrA: K, intrB: K, priorR: priorR, priorT: trueT + SIMD3(0.03, 0, 0)))
    let correct = fit.inliers.filter { $0 < 60 }.count
    let wrong = fit.inliers.count - correct
    #expect(correct >= 55, "kept \(correct) of 60 patch matches")
    #expect(wrong <= 2, "kept \(wrong) of 40 flipped matches")
    // The matches lie on a small plane, which leaves the rotation weakly determined (about 1°); the
    // fit still moves closer to the truth than the 3° prior.
    #expect(angleDeg(fit.R, trueR) < 2, "fitted rotation \(angleDeg(fit.R, trueR))° off the truth")
}

/// A ring of 120 cameras with drift that builds up along the trajectory, as ARKit's does between
/// passes (8 cm and 1.5° at the end, well within σ in each frame-to-frame step), tied across the ring
/// by a few loop edges with the true relative poses: the pose graph spreads the correction over the
/// steps and brings the far half most of the way back (the loop edges are loose, σ 3 cm: BA, with
/// the loop pairs' matches, finishes the job).
@Test
func poseGraphClosesALoopAgainstDrift() {
    let n = 120
    var truth: [[Float]] = []
    for k in 0..<n {
        let a = Double(k) / Double(n) * 360
        let R = rotationY(-a)
        let C = SIMD3(1.5 * sin(a * .pi / 180), 0, -1.5 * cos(a * .pi / 180))
        truth.append(viewmat(R, C))
    }
    // Drifted: camera k moved rigidly by k/(n-1) of (1.5° about y around the origin, + 8 cm x)
    var drifted = truth
    for k in 0..<n {
        let f = Double(k) / Double(n - 1)
        let W = rotationY(1.5 * f)
        let v = truth[k]
        let R = simd_double3x3(rows: [SIMD3(Double(v[0]), Double(v[1]), Double(v[2])),
                                      SIMD3(Double(v[4]), Double(v[5]), Double(v[6])),
                                      SIMD3(Double(v[8]), Double(v[9]), Double(v[10]))])
        let C = -(R.transpose * SIMD3(Double(v[3]), Double(v[7]), Double(v[11])))
        drifted[k] = viewmat(R * W.transpose, W * C + SIMD3(0.08 * f, 0, 0))
    }
    let odometry = (0..<(n - 1)).map {
        PoseGraph.Edge.between($0, $0 + 1, viewmats: drifted, sigmaRotationRad: 0.1 * .pi / 180,
                               sigmaPosition: 0.0025, huber: 2)
    }
    let loops = [(10, 70), (20, 80), (40, 100), (50, 110), (0, 119)].map { (i, j) in
        PoseGraph.Edge.between(i, j, viewmats: truth, sigmaRotationRad: 0.3 * .pi / 180, sigmaPosition: 0.03, huber: 2)
    }
    func centerError(_ vms: [[Float]]) -> Float {
        (60..<n).map { simd_distance(FrameSubsampler.cameraCenter(viewmat: vms[$0]),
                                     FrameSubsampler.cameraCenter(viewmat: truth[$0])) }.max()!
    }
    let before = centerError(drifted)
    let result = PoseGraph.optimize(viewmats: drifted, edges: odometry + loops)
    let after = centerError(result.viewmats)
    #expect(result.finalCost < result.initialCost)
    #expect(after < 0.5 * before, "far half off the truth by \(after) m after (\(before) m before)")
}
