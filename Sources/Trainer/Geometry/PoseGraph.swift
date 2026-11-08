// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Pose graph over camera poses: relative-pose edges (odometry between consecutive frames, loop
// closures between frames matched across passes) solved by Gauss-Newton. Used to close large loops
// before bundle adjustment: BA's robust kernel and outlier filter treat loop observations that are
// off by tens of pixels as outliers instead of moving the cameras, so a loop that far open never
// closes in BA alone.

import Foundation
import simd

public enum PoseGraph {

    /// Relative pose of camera j seen from camera i, in the form of `BundleAdjustment.Parameters`'
    /// relative prior: rotation R_j R_iᵀ and offset R_i(O_j − O_i) (O = camera center). Residuals
    /// are whitened by the σs and Huber-robust at `huber` σ (0 = quadratic).
    public struct Edge: Sendable {
        public let i: Int
        public let j: Int
        public let rotation: simd_double3x3
        public let offset: SIMD3<Double>
        public let sigmaRotationRad: Double
        public let sigmaPosition: Double
        public let huber: Double

        public init(i: Int, j: Int, rotation: simd_double3x3, offset: SIMD3<Double>,
                    sigmaRotationRad: Double, sigmaPosition: Double, huber: Double = 0) {
            self.i = i; self.j = j
            self.rotation = rotation; self.offset = offset
            self.sigmaRotationRad = sigmaRotationRad; self.sigmaPosition = sigmaPosition
            self.huber = huber
        }

        /// The edge between two row-major world→camera viewmats as they are.
        public static func between(_ i: Int, _ j: Int, viewmats: [[Float]],
                                   sigmaRotationRad: Double, sigmaPosition: Double, huber: Double = 0) -> Edge {
            typealias P = BundleAdjustment.PosePrior
            let Ri = P.rot(viewmats[i]), Rj = P.rot(viewmats[j])
            let Oi = P.center(viewmats[i]), Oj = P.center(viewmats[j])
            return Edge(i: i, j: j, rotation: Rj * Ri.transpose, offset: Ri * (Oj - Oi),
                        sigmaRotationRad: sigmaRotationRad, sigmaPosition: sigmaPosition, huber: huber)
        }
    }

    public struct Result: Sendable {
        public let viewmats: [[Float]]
        /// Whitened cost before and after.
        public let initialCost: Double
        public let finalCost: Double
    }

    /// Optimizes the poses (camera `fixed` stays put) for `iterations` LM steps.
    public static func optimize(viewmats: [[Float]], edges: [Edge], fixed: Int = 0, iterations: Int = 15) -> Result {
        typealias P = BundleAdjustment.PosePrior
        let n = viewmats.count
        var R = viewmats.map { P.rot($0) }
        var O = viewmats.map { P.center($0) }
        func residual(_ e: Edge, _ Ri: simd_double3x3, _ Oi: SIMD3<Double>,
                      _ Rj: simd_double3x3, _ Oj: SIMD3<Double>) -> [Double] {
            let eR = P.logSO3(e.rotation.transpose * (Rj * Ri.transpose)) / e.sigmaRotationRad
            let eD = (Ri * (Oj - Oi) - e.offset) / e.sigmaPosition
            return [eR.x, eR.y, eR.z, eD.x, eD.y, eD.z]
        }
        /// Huber weight factors (rotation group, position group) and cost of a whitened residual.
        func robust(_ r: [Double], _ h: Double) -> (wR: Double, wD: Double, cost: Double) {
            func one(_ z2: Double) -> (Double, Double) {
                guard h > 0 else { return (1, 0.5 * z2) }
                let z = z2.squareRoot()
                return z <= h ? (1, 0.5 * z2) : (h / z, h * z - 0.5 * h * h)
            }
            let a = one(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
            let b = one(r[3] * r[3] + r[4] * r[4] + r[5] * r[5])
            return (a.0, b.0, a.1 + b.1)
        }
        func cost() -> Double {
            edges.reduce(0) { $0 + robust(residual($1, R[$1.i], O[$1.i], R[$1.j], O[$1.j]), $1.huber).cost }
        }
        let initialCost = cost()
        var current = initialCost
        var lambda = 1e-4
        // Index of each free camera in the system.
        let free = (0..<n).filter { $0 != fixed }
        var slot = [Int](repeating: -1, count: n)
        for (k, c) in free.enumerated() { slot[c] = k }
        let D = 6 * free.count
        guard D > 0, !edges.isEmpty else {
            return Result(viewmats: viewmats, initialCost: initialCost, finalCost: initialCost)
        }
        for _ in 0..<iterations {
            var H = [Double](repeating: 0, count: D * D)
            var g = [Double](repeating: 0, count: D)
            let h = 1e-6
            for e in edges {
                let r = residual(e, R[e.i], O[e.i], R[e.j], O[e.j])
                let w = robust(r, e.huber)
                let wv = [w.wR, w.wR, w.wR, w.wD, w.wD, w.wD]
                var J = [[Double]](repeating: [Double](repeating: 0, count: 6), count: 12)
                for c in 0..<12 {
                    var d = [Double](repeating: 0, count: 6); d[c % 6] = h
                    var dm = d; for k in 0..<6 { dm[k] = -d[k] }
                    let plus: [Double], minus: [Double]
                    if c < 6 {
                        let (Rp, Op) = P.perturbed(R[e.i], O[e.i], d), (Rm, Om) = P.perturbed(R[e.i], O[e.i], dm)
                        plus = residual(e, Rp, Op, R[e.j], O[e.j]); minus = residual(e, Rm, Om, R[e.j], O[e.j])
                    } else {
                        let (Rp, Op) = P.perturbed(R[e.j], O[e.j], d), (Rm, Om) = P.perturbed(R[e.j], O[e.j], dm)
                        plus = residual(e, R[e.i], O[e.i], Rp, Op); minus = residual(e, R[e.i], O[e.i], Rm, Om)
                    }
                    for k in 0..<6 { J[c][k] = (plus[k] - minus[k]) / (2 * h) }
                }
                func row(_ c: Int) -> Int? {
                    let cam = c < 6 ? e.i : e.j
                    return slot[cam] < 0 ? nil : slot[cam] * 6 + c % 6
                }
                for c1 in 0..<12 {
                    guard let r1 = row(c1) else { continue }
                    var gs = 0.0
                    for k in 0..<6 { gs += J[c1][k] * wv[k] * r[k] }
                    g[r1] -= gs
                    for c2 in 0..<12 {
                        guard let r2 = row(c2) else { continue }
                        var hs = 0.0
                        for k in 0..<6 { hs += J[c1][k] * wv[k] * J[c2][k] }
                        H[r1 * D + r2] += hs
                    }
                }
            }
            for k in 0..<D { H[k * D + k] += lambda * (H[k * D + k] + 1e-9) + 1e-12 }
            let step = BundleAdjustment.solveReducedSystem(lhs: H.map(Float.init), rhs: g.map(Float.init), dimension: D)
            var R2 = R, O2 = O
            for c in free {
                let k = slot[c] * 6
                let d = (0..<6).map { Double(step[k + $0]) }
                (R2[c], O2[c]) = P.perturbed(R[c], O[c], d)
            }
            let (Rs, Os) = (R, O)
            R = R2; O = O2
            let next = cost()
            if next < current {
                current = next
                lambda = max(lambda / 3, 1e-9)
            } else {
                R = Rs; O = Os
                lambda *= 4
                if lambda > 1e6 { break }
            }
        }
        let out: [[Float]] = (0..<n).map { c in
            let t = -(R[c] * O[c])
            return [Float(R[c][0][0]), Float(R[c][1][0]), Float(R[c][2][0]), Float(t.x),
                    Float(R[c][0][1]), Float(R[c][1][1]), Float(R[c][2][1]), Float(t.y),
                    Float(R[c][0][2]), Float(R[c][1][2]), Float(R[c][2][2]), Float(t.z),
                    0, 0, 0, 1]
        }
        return Result(viewmats: out, initialCost: initialCost, finalCost: current)
    }
}
