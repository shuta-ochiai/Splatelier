// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Loop edges between frames that look at the same place from very different directions (the two
// faces of a double-sided sign meeting at its ends, the far side of an object orbit): the relative
// pose explaining a pair's matches, fitted starting from ARKit's, and the pose graph edge made from
// it. A pose rotated far from ARKit's (e.g. the same lettering on both faces of a sign matched
// across) is not explained, so such matches drop out.

import Foundation
import simd
import MLX
import Trainer

extension CaptureBA {

    struct RelativePoseFit {
        let inliers: [Int]
        let rotationDeg: Double
        /// x_b = R x_a + t (t is a unit direction).
        let R: simd_double3x3
        let t: SIMD3<Double>
    }

    /// RANSAC over 5-match samples, each solved by Gauss-Newton on the Sampson error starting from
    /// the prior pose; hypotheses rotated more than `maxRotationDeg` from the prior are discarded.
    /// The best one is refined on its inliers. nil if fewer than `minInliers` remain.
    static func fitRelativePoseNearPrior(
        uvA: [SIMD2<Float>], uvB: [SIMD2<Float>], intrA: Intrinsics, intrB: Intrinsics,
        priorR: simd_double3x3, priorT: SIMD3<Double>,
        maxRotationDeg: Double = 10, rotationPriorDeg: Double = 0, thresholdPx: Double = 2,
        minInliers: Int = 20, iterations: Int = 200, seed: UInt64 = 1
    ) -> RelativePoseFit? {
        let m = uvA.count
        guard m >= minInliers, m == uvB.count else { return nil }
        let Ka = simd_double3x3(rows: [SIMD3(Double(intrA.fx), 0, Double(intrA.cx)),
                                       SIMD3(0, Double(intrA.fy), Double(intrA.cy)), SIMD3(0, 0, 1)])
        let Kb = simd_double3x3(rows: [SIMD3(Double(intrB.fx), 0, Double(intrB.cx)),
                                       SIMD3(0, Double(intrB.fy), Double(intrB.cy)), SIMD3(0, 0, 1)])
        let KaInv = Ka.inverse, KbInvT = Kb.inverse.transpose
        let xa = uvA.map { SIMD3(Double($0.x), Double($0.y), 1) }
        let xb = uvB.map { SIMD3(Double($0.x), Double($0.y), 1) }
        let t0 = simd_length(priorT) > 1e-9 ? simd_normalize(priorT) : SIMD3(1, 0, 0)
        // Orthonormal basis of the tangent plane at t0.
        let helper = abs(t0.x) < 0.9 ? SIMD3<Double>(1, 0, 0) : SIMD3<Double>(0, 1, 0)
        let e1 = simd_normalize(simd_cross(t0, helper)), e2 = simd_cross(t0, e1)

        func rotation(_ w: SIMD3<Double>) -> simd_double3x3 {
            let th = simd_length(w)
            guard th > 1e-12 else { return matrix_identity_double3x3 }
            let k = w / th
            let K = simd_double3x3(rows: [SIMD3(0, -k.z, k.y), SIMD3(k.z, 0, -k.x), SIMD3(-k.y, k.x, 0)])
            return matrix_identity_double3x3 + sin(th) * K + (1 - cos(th)) * (K * K)
        }
        /// Parameters: ω (rotation from the prior, 3) and the translation direction's tangent
        /// coordinates (2).
        func fundamental(_ p: [Double]) -> simd_double3x3 {
            let R = rotation(SIMD3(p[0], p[1], p[2])) * priorR
            let t = simd_normalize(t0 + p[3] * e1 + p[4] * e2)
            let tx = simd_double3x3(rows: [SIMD3(0, -t.z, t.y), SIMD3(t.z, 0, -t.x), SIMD3(-t.y, t.x, 0)])
            return KbInvT * (tx * R) * KaInv
        }
        /// Signed Sampson distance (px) of match k under F.
        func residual(_ F: simd_double3x3, _ k: Int) -> Double {
            let Fa = F * xa[k], Ftb = F.transpose * xb[k]
            let num = simd_dot(xb[k], Fa)
            let den = Fa.x * Fa.x + Fa.y * Fa.y + Ftb.x * Ftb.x + Ftb.y * Ftb.y
            return den > 1e-18 ? num / den.squareRoot() : 0
        }
        // Rows ω/σ that pull the rotation toward the prior (1σ weighs like 1 px): with a small field
        // of view or a planar patch, rotation and translation direction trade off (bas-relief
        // ambiguity), and without it the fit drifts to a rotation degrees off the true one.
        let priorWeight = rotationPriorDeg > 0 ? 1 / (rotationPriorDeg * .pi / 180) : 0
        func gaussNewton(_ start: [Double], _ idx: [Int], steps: Int) -> [Double] {
            var p = start
            let h = 1e-6
            for _ in 0..<steps {
                let F = fundamental(p)
                let r = idx.map { residual(F, $0) }
                var J = [[Double]](repeating: [Double](repeating: 0, count: 5), count: idx.count)
                for c in 0..<5 {
                    var q = p; q[c] += h
                    let Fq = fundamental(q)
                    for (row, k) in idx.enumerated() { J[row][c] = (residual(Fq, k) - r[row]) / h }
                }
                // (JᵀJ + λI) δ = −Jᵀr
                var A5 = [Double](repeating: 0, count: 25), g = [Double](repeating: 0, count: 5)
                for row in 0..<idx.count {
                    for i in 0..<5 {
                        g[i] -= J[row][i] * r[row]
                        for j in 0..<5 { A5[i * 5 + j] += J[row][i] * J[row][j] }
                    }
                }
                for i in 0..<3 where priorWeight > 0 {
                    // row r = w·ω_i, J = w·e_i
                    A5[i * 5 + i] += priorWeight * priorWeight
                    g[i] -= priorWeight * priorWeight * p[i]
                }
                for i in 0..<5 { A5[i * 5 + i] += 1e-6 * (1 + A5[i * 5 + i]) }
                guard let d = solve5(A5, g) else { break }
                for i in 0..<5 { p[i] += d[i] }
                if d.map(abs).max()! < 1e-9 { break }
            }
            return p
        }
        func inliers(_ p: [Double]) -> [Int] {
            let F = fundamental(p)
            return (0..<m).filter { abs(residual(F, $0)) <= thresholdPx }
        }
        let maxRot = maxRotationDeg * .pi / 180
        var state = seed &* 6364136223846793005 &+ 1442695040888963407
        func nextIndex() -> Int {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Int((state >> 33) % UInt64(m))
        }
        var best: (p: [Double], inl: [Int])? = nil
        for _ in 0..<iterations {
            var sample: [Int] = []
            while sample.count < 5 {
                let k = nextIndex()
                if !sample.contains(k) { sample.append(k) }
            }
            let p = gaussNewton([0, 0, 0, 0, 0], sample, steps: 8)
            guard simd_length(SIMD3(p[0], p[1], p[2])) <= maxRot else { continue }
            let inl = inliers(p)
            if inl.count > (best?.inl.count ?? 0) { best = (p, inl) }
        }
        guard var (p, inl) = best, inl.count >= minInliers else { return nil }
        // Refine on the inliers, then recount.
        p = gaussNewton(p, inl, steps: 10)
        let rot = simd_length(SIMD3(p[0], p[1], p[2]))
        guard rot <= maxRot else { return nil }
        inl = inliers(p)
        guard inl.count >= minInliers else { return nil }
        return RelativePoseFit(inliers: inl, rotationDeg: rot * 180 / .pi,
                               R: rotation(SIMD3(p[0], p[1], p[2])) * priorR,
                               t: simd_normalize(t0 + p[3] * e1 + p[4] * e2))
    }

    /// Solves the 5×5 system A x = b (row-major) by Gaussian elimination with partial pivoting.
    private static func solve5(_ a: [Double], _ b: [Double]) -> [Double]? {
        var A = a, x = b
        for c in 0..<5 {
            var piv = c
            for r in (c + 1)..<5 where abs(A[r * 5 + c]) > abs(A[piv * 5 + c]) { piv = r }
            guard abs(A[piv * 5 + c]) > 1e-15 else { return nil }
            if piv != c {
                for k in 0..<5 { A.swapAt(c * 5 + k, piv * 5 + k) }
                x.swapAt(c, piv)
            }
            for r in (c + 1)..<5 {
                let f = A[r * 5 + c] / A[c * 5 + c]
                for k in c..<5 { A[r * 5 + k] -= f * A[c * 5 + k] }
                x[r] -= f * x[c]
            }
        }
        for c in stride(from: 4, through: 0, by: -1) {
            for k in (c + 1)..<5 { x[c] -= A[c * 5 + k] * x[k] }
            x[c] /= A[c * 5 + c]
        }
        return x
    }

    /// Relative pose b ← a of two row-major world→camera viewmats: R = R_b R_aᵀ, t = t_b − R t_a.
    static func relativePose(_ va: [Float], _ vb: [Float]) -> (R: simd_double3x3, t: SIMD3<Double>) {
        func rt(_ v: [Float]) -> (simd_double3x3, SIMD3<Double>) {
            (simd_double3x3(rows: [SIMD3(Double(v[0]), Double(v[1]), Double(v[2])),
                                   SIMD3(Double(v[4]), Double(v[5]), Double(v[6])),
                                   SIMD3(Double(v[8]), Double(v[9]), Double(v[10]))]),
             SIMD3(Double(v[3]), Double(v[7]), Double(v[11])))
        }
        let (Ra, ta) = rt(va), (Rb, tb) = rt(vb)
        let R = Rb * Ra.transpose
        return (R, tb - R * ta)
    }

    /// Pose graph edge from a fitted relative pose, with the baseline length taken from `priorT`
    /// (the fit gives only its direction).
    static func loopEdge(_ i: Int, _ j: Int, fit: RelativePoseFit, priorT: SIMD3<Double>,
                         sigmaRotationRad: Double, sigmaPosition: Double, huber: Double) -> PoseGraph.Edge {
        // Camera j's center in camera i's frame: 0 = R·O + t·|t_prior|.
        let offset = -(fit.R.transpose * (fit.t * simd_length(priorT)))
        return PoseGraph.Edge(i: i, j: j, rotation: fit.R, offset: offset,
                              sigmaRotationRad: sigmaRotationRad, sigmaPosition: sigmaPosition, huber: huber)
    }
}
