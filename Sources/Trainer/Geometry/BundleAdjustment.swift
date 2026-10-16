// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Bundle Adjustment (PocketGS Operator G 3c, Sec. III-A 2). GN / LM + Huber + scale-aware gauge.
// Reference: Triggs et al. 2000, "Bundle Adjustment — A Modern Synthesis".
//
// Schur complement (Triggs sec. 4.3 / PocketGS eq. 3). H_pp is block-diagonal with per-point 3×3
// blocks:
//   (H_tt - H_tp H_pp⁻¹ H_tpᵀ) δ_cam = b_t - H_tp H_pp⁻¹ b_p
//   δ_pt_j = H_pp_j⁻¹ (b_p_j - Σ_i H_tp_{ij}ᵀ δ_cam_i)
// Pose update is a left perturbation T ← exp([δ_rot, δ_trans]) · T (small-step approximation, Sola
// 2018 eq. 95-102).

import Foundation
import simd
import MLX
import MLXFast

public enum BundleAdjustment {

    /// Camera i observed point j at `pixel`. `pixel` is continuous, with pixel centers at integer +
    /// 0.5.
    public struct Observation: Sendable, Equatable {
        public let cameraIndex: Int
        public let pointIndex: Int
        public let pixel: SIMD2<Float>
        public init(cameraIndex: Int, pointIndex: Int, pixel: SIMD2<Float>) {
            self.cameraIndex = cameraIndex
            self.pointIndex = pointIndex
            self.pixel = pixel
        }
    }

    /// viewmats are row-major 4×4 world→camera; points are in world coordinates.
    public struct State: Sendable {
        public var viewmats: [[Float]]
        public var points: [SIMD3<Float>]
        public init(viewmats: [[Float]], points: [SIMD3<Float>]) {
            self.viewmats = viewmats
            self.points = points
        }
    }

    public struct Parameters: Sendable {
        public var iterations: Int
        public var freezeFirstCamera: Bool   // fix camera 0 (gauge fix)
        /// Scale-Aware Gauge Fixing (PocketGS Sec. III-A 2). Fixes the global-scale freedom left
        /// after fixing camera 0 by removing the baseline-axis component from camera 1's
        /// translation update (the 2 orthogonal components stay free).
        public var scaleAwareGauge: Bool
        /// Huber threshold (px). nil means L2. The paper does not disclose the value.
        public var huberThreshold: Float?
        /// Initial LM damping λ (added to the reduced system's diagonal; 0 is pure GN). When not
        /// adaptive, multiplied by `lmDecay` each iteration.
        public var lmInitialDamping: Float
        public var lmDecay: Float
        /// If the residual drops, accept and λ /= 3; if it rises, reject and λ *= 2 (simplified
        /// Triggs sec. 4.6 / Madsen-Nielsen eq. 3.36). Fixed decay tends to diverge on small
        /// systems with outliers.
        public var adaptiveLM: Bool
        /// How damping is applied. `.absolute` is the old scheme that adds λ directly to the
        /// diagonal of the reduced camera system (the right λ depends on the problem's scale).
        /// `.marquardt` scales the diagonals of H_tt and H_pp by (1+λ), making λ a dimensionless
        /// relative value. With `.marquardt` and `adaptiveLM`, steps are accepted or rejected on
        /// the actual objective (Huber cost, with a penalty for observations that end up behind the
        /// camera).
        public var dampingMode: DampingMode
        /// Weak constraint toward prior poses (`refine(priorViewmats:)`). Given an orientation σ
        /// (degrees) and position σ (m), adds 0.5·(σpx/σθ)²·|Log(R_priorᵀR)|² + 0.5·(σpx/σp)²·|O −
        /// O_prior|² to the cost (σpx = `priorPixelSigma`, same scale as the pixel residuals).
        /// Cameras with many observations follow the data; cameras with few stay near their prior
        /// pose. nil disables.
        public var priorRotationSigmaDeg: Float? = nil
        public var priorPositionSigma: Float? = nil
        public var priorPixelSigma: Float = 1.0

        public enum DampingMode: Sendable { case absolute, marquardt }

        public init(
            iterations: Int = 5,
            freezeFirstCamera: Bool = true,
            scaleAwareGauge: Bool = false,
            huberThreshold: Float? = nil,
            lmInitialDamping: Float = 0,
            lmDecay: Float = 0.5,
            adaptiveLM: Bool = false,
            dampingMode: DampingMode = .absolute,
            priorRotationSigmaDeg: Float? = nil,
            priorPositionSigma: Float? = nil
        ) {
            precondition(iterations > 0)
            precondition(lmInitialDamping >= 0)
            precondition(lmDecay > 0)
            self.iterations = iterations
            self.freezeFirstCamera = freezeFirstCamera
            self.scaleAwareGauge = scaleAwareGauge
            self.huberThreshold = huberThreshold
            self.lmInitialDamping = lmInitialDamping
            self.lmDecay = lmDecay
            self.adaptiveLM = adaptiveLM
            self.dampingMode = dampingMode
            self.priorRotationSigmaDeg = priorRotationSigmaDeg
            self.priorPositionSigma = priorPositionSigma
        }
    }

    public struct Result: Sendable {
        public let state: State
        public let residualNorms: [Float]   // RMS residual (px): initial + after each iteration
    }

    /// Takes `parameters.iterations` GN / LM steps. Intrinsics are fixed.
    public static func refine(
        state: State,
        intrinsics: [Intrinsics],
        observations: [Observation],
        parameters: Parameters = Parameters(),
        priorViewmats: [[Float]]? = nil
    ) -> Result {
        precondition(state.viewmats.count == intrinsics.count,
                     "viewmats and intrinsics counts must match")
        precondition(!observations.isEmpty)

        var current = state
        var residualNorms: [Float] = []
        var lambda = parameters.lmInitialDamping
        residualNorms.append(residualNorm(state: current, intrinsics: intrinsics, observations: observations))
        let marquardt = parameters.dampingMode == .marquardt
        let prior = PosePrior(parameters: parameters, viewmats: priorViewmats)
        var currentCost = marquardt
            ? robustCost(state: current, intrinsics: intrinsics, observations: observations,
                         huberThreshold: parameters.huberThreshold) + (prior?.cost(current) ?? 0)
            : 0
        for _ in 0..<parameters.iterations {
            let candidate = singleGaussNewtonStep(
                state: current, intrinsics: intrinsics,
                observations: observations,
                freezeFirstCamera: parameters.freezeFirstCamera,
                scaleAwareGauge: parameters.scaleAwareGauge,
                huberThreshold: parameters.huberThreshold,
                lmDamping: lambda,
                dampingMode: parameters.dampingMode,
                prior: prior
            )
            if marquardt && parameters.adaptiveLM {
                // Accept/reject on the objective (Huber cost). Accept: λ/3; reject: λ×4.
                let candCost = robustCost(
                    state: candidate, intrinsics: intrinsics, observations: observations,
                    huberThreshold: parameters.huberThreshold) + (prior?.cost(candidate) ?? 0)
                if candCost < currentCost {
                    current = candidate
                    currentCost = candCost
                    residualNorms.append(residualNorm(state: current, intrinsics: intrinsics, observations: observations))
                    lambda = max(lambda / 3, 1e-9)
                } else {
                    residualNorms.append(residualNorms.last ?? 0)
                    lambda *= 4
                    if lambda > 1e6 { break }   // more damping won't lower it = converged
                }
            } else if parameters.adaptiveLM {
                // Accept/reject on RMS residual (simple Madsen-Nielsen). Accept: λ/3; reject: λ×2.
                let beforeRes = residualNorms.last ?? Float.infinity
                let afterRes = residualNorm(state: candidate, intrinsics: intrinsics, observations: observations)
                if afterRes < beforeRes {
                    current = candidate
                    residualNorms.append(afterRes)
                    lambda = max(lambda * (1.0 / 3.0), 1e-8)
                } else {
                    // Reject and leave the state unchanged
                    residualNorms.append(beforeRes)
                    lambda = lambda > 0 ? lambda * 2.0 : 1e-4
                    if lambda > 1e10 { break }
                }
            } else {
                current = candidate
                residualNorms.append(residualNorm(state: current, intrinsics: intrinsics, observations: observations))
                lambda *= parameters.lmDecay
            }
        }
        return Result(state: current, residualNorms: residualNorms)
    }

    // MARK: - Internals

    /// One GN / LM step. Solves the reduced camera system via the Schur complement and updates the
    /// state.
    static func singleGaussNewtonStep(
        state: State,
        intrinsics: [Intrinsics],
        observations: [Observation],
        freezeFirstCamera: Bool,
        scaleAwareGauge: Bool = false,
        huberThreshold: Float? = nil,
        lmDamping: Float = 0,
        dampingMode: Parameters.DampingMode = .absolute,
        prior: PosePrior? = nil
    ) -> State {
        let nCam = state.viewmats.count
        let nPt = state.points.count
        let firstFree = freezeFirstCamera ? 1 : 0
        let nFree = nCam - firstFree
        guard nFree > 0 else { return state }

        // --- 1. Build H and b on the GPU ---
        // H_tt [nCam][6×6] (block-diagonal), H_pp [nPt][3×3], b_t [nCam][6], b_p [nPt][3],
        // H_tp is a 6×3 block per observation (htpBlocks).
        var Htt: [Float]
        var Hpp: [Float]
        var bt: [Float]
        var bp: [Float]
        var htpBlocks: [HtpBlock]
        (Htt, Hpp, bt, bp, htpBlocks) = computeJacobianAndAccumulateGPU(
            state: state, intrinsics: intrinsics,
            observations: observations, huberThreshold: huberThreshold,
            nCam: nCam, nPt: nPt
        )
        // Prior-pose constraint (camera blocks only).
        if let prior { prior.accumulate(state: state, Htt: &Htt, bt: &bt, skipFirst: freezeFirstCamera) }
        // Marquardt damping: scale the diagonals of H_tt and H_pp by (1+λ). Applying it on the
        // point side too keeps the inverse of low-parallax points (H_pp nearly singular) from
        // blowing up. Components with a zero diagonal (cameras without observations, etc.) get a
        // small floor to stay non-singular.
        if dampingMode == .marquardt && lmDamping > 0 {
            let s = 1 + lmDamping
            for c in 0..<nCam {
                for r in 0..<6 {
                    let k = c * 36 + r * 7
                    Htt[k] = Htt[k] * s + 1e-9
                }
            }
            for j in 0..<nPt {
                for r in 0..<3 {
                    let k = j * 9 + r * 4
                    Hpp[k] = Hpp[k] * s + 1e-9
                }
            }
        }

        // --- 2. H_pp⁻¹ + Schur reduction --- On the GPU, differing atomic-add order causes
        // cancellation in small systems (breaks tests with few points). Use the GPU only when there
        // are enough points, otherwise the CPU.
        let D = 6 * nFree
        let GPU_SCHUR_NPT_THRESHOLD = 1000
        var redLHSMut: [Float]
        var redRHSMut: [Float]
        var HppInv: [Float]
        if nPt >= GPU_SCHUR_NPT_THRESHOLD {
            (redLHSMut, redRHSMut, HppInv) = runHppInvertAndSchurGPU(
                Hpp: Hpp, bp: bp, Htt: Htt, bt: bt, htpBlocks: htpBlocks,
                nCam: nCam, nPt: nPt, nFree: nFree, firstFree: firstFree
            )
        } else {
            HppInv = [Float](repeating: 0, count: nPt * 9)
            for j in 0..<nPt {
                invert3x3(
                    a: Array(Hpp[(j * 9)..<((j + 1) * 9)]),
                    into: &HppInv, offset: j * 9
                )
            }
            redLHSMut = [Float](repeating: 0, count: D * D)
            redRHSMut = [Float](repeating: 0, count: D)
            for camI in firstFree..<nCam {
                let fI = camI - firstFree
                for r in 0..<6 {
                    redRHSMut[fI * 6 + r] = bt[camI * 6 + r]
                    for c in 0..<6 {
                        redLHSMut[(fI * 6 + r) * D + (fI * 6 + c)] = Htt[camI * 36 + r * 6 + c]
                    }
                }
            }
            cpuSchurReduce(
                redLHS: &redLHSMut, redRHS: &redRHSMut,
                HppInv: HppInv, bp: bp, htpBlocks: htpBlocks,
                D: D, nPt: nPt, firstFree: firstFree, freezeFirstCamera: freezeFirstCamera
            )
        }
        // Per-point htpBlocks indices (for CPU back-substitution)
        var byPoint: [[Int]] = Array(repeating: [], count: nPt)
        for (idx, b) in htpBlocks.enumerated() {
            byPoint[b.pt].append(idx)
        }

        // --- 3. LM damping: (H + λI) δ = -Jᵀ r (Triggs sec. 4.4) ---
        if dampingMode == .absolute && lmDamping > 0 {
            for i in 0..<D {
                redLHSMut[i * D + i] += lmDamping
            }
        }

        // --- 4. Solve the reduced system (D×D is a few hundred, so CPU stream) ---
        let reducedLHSArr = MLXArray(redLHSMut, [D, D])
        let reducedRHSArr = MLXArray(redRHSMut, [D, 1])
        let solution = MLX.solve(reducedLHSArr, reducedRHSArr, stream: .cpu)
        let deltaCamFlat = solution.asArray(Float.self)

        // δ_cam of fixed cameras is 0
        var deltaCam = [Float](repeating: 0, count: nCam * 6)
        for camI in firstFree..<nCam {
            let fI = camI - firstFree
            for r in 0..<6 {
                deltaCam[camI * 6 + r] = deltaCamFlat[fI * 6 + r]
            }
        }

        // --- 5. Scale-aware gauge: remove the baseline-axis component from camera 1's translation
        // update --- The baseline is in world coordinates but δ_cam's translation (v of the left
        // perturbation) is in camera 1 coordinates, so rotate the baseline by camera 1's rotation
        // before the dot product.
        if scaleAwareGauge && freezeFirstCamera && nCam >= 2 {
            let c0 = FrameSubsampler.cameraCenter(viewmat: state.viewmats[0])
            let c1 = FrameSubsampler.cameraCenter(viewmat: state.viewmats[1])
            let bvec = SIMD3<Float>(c1.x - c0.x, c1.y - c0.y, c1.z - c0.z)
            let blen = (bvec.x * bvec.x + bvec.y * bvec.y + bvec.z * bvec.z).squareRoot()
            if blen > 1e-6 {
                let bhatWorld = SIMD3<Float>(bvec.x / blen, bvec.y / blen, bvec.z / blen)
                let vm1 = state.viewmats[1]
                // R1 · bhatWorld (a rotation, so still unit length)
                let bhat = SIMD3<Float>(
                    vm1[0] * bhatWorld.x + vm1[1] * bhatWorld.y + vm1[2] * bhatWorld.z,
                    vm1[4] * bhatWorld.x + vm1[5] * bhatWorld.y + vm1[6] * bhatWorld.z,
                    vm1[8] * bhatWorld.x + vm1[9] * bhatWorld.y + vm1[10] * bhatWorld.z
                )
                let dx = deltaCam[1 * 6 + 3]
                let dy = deltaCam[1 * 6 + 4]
                let dz = deltaCam[1 * 6 + 5]
                let proj = dx * bhat.x + dy * bhat.y + dz * bhat.z
                deltaCam[1 * 6 + 3] = dx - proj * bhat.x
                deltaCam[1 * 6 + 4] = dy - proj * bhat.y
                deltaCam[1 * 6 + 5] = dz - proj * bhat.z
            }
        }

        // --- 6. Back-substitute δ_pt (GPU / CPU with the same threshold as Schur) ---
        let deltaPt: [Float]
        if nPt >= GPU_SCHUR_NPT_THRESHOLD {
            deltaPt = runBackSubstituteGPU(
                HppInv: HppInv, bp: bp, htpBlocks: htpBlocks,
                deltaCam: deltaCam, nCam: nCam, nPt: nPt
            )
        } else {
            var deltaPtMut = [Float](repeating: 0, count: nPt * 3)
            for j in 0..<nPt {
                let attachedIndices = byPoint[j]
                if attachedIndices.isEmpty { continue }
                var rhs = SIMD3<Float>(bp[j * 3 + 0], bp[j * 3 + 1], bp[j * 3 + 2])
                for idx in attachedIndices {
                    let b = htpBlocks[idx]
                    let dc = SIMD6Floats(
                        deltaCam[b.cam * 6 + 0], deltaCam[b.cam * 6 + 1],
                        deltaCam[b.cam * 6 + 2], deltaCam[b.cam * 6 + 3],
                        deltaCam[b.cam * 6 + 4], deltaCam[b.cam * 6 + 5]
                    )
                    rhs -= matVec3x6Transpose(m6x3: b.m, v6: dc)
                }
                let delta = matVec3x3(matrixOffset: j * 9, m: HppInv, v: rhs)
                deltaPtMut[j * 3 + 0] = delta.x
                deltaPtMut[j * 3 + 1] = delta.y
                deltaPtMut[j * 3 + 2] = delta.z
            }
            deltaPt = deltaPtMut
        }

        // --- 7. Update state ---
        var newViewmats = state.viewmats
        for camI in 0..<nCam {
            let omega = SIMD3<Float>(
                deltaCam[camI * 6 + 0], deltaCam[camI * 6 + 1], deltaCam[camI * 6 + 2]
            )
            let v = SIMD3<Float>(
                deltaCam[camI * 6 + 3], deltaCam[camI * 6 + 4], deltaCam[camI * 6 + 5]
            )
            newViewmats[camI] = applyLeftSE3Perturbation(
                viewmat: state.viewmats[camI], omega: omega, v: v
            )
        }
        var newPoints = state.points
        for j in 0..<nPt {
            newPoints[j].x += deltaPt[j * 3 + 0]
            newPoints[j].y += deltaPt[j * 3 + 1]
            newPoints[j].z += deltaPt[j * 3 + 2]
        }
        return State(viewmats: newViewmats, points: newPoints)
    }

    /// CPU Schur reduction for small systems. Assumes redLHS / redRHS are initialized with the H_tt
    /// diagonal blocks + b_t.
    static func cpuSchurReduce(
        redLHS: inout [Float], redRHS: inout [Float],
        HppInv: [Float], bp: [Float],
        htpBlocks: [HtpBlock],
        D: Int, nPt: Int, firstFree: Int, freezeFirstCamera: Bool
    ) {
        var byPoint: [[Int]] = Array(repeating: [], count: nPt)
        for (idx, b) in htpBlocks.enumerated() {
            byPoint[b.pt].append(idx)
        }
        for j in 0..<nPt {
            let attachedIndices = byPoint[j]
            if attachedIndices.isEmpty { continue }
            let invBase = j * 9
            let bp_j = SIMD3<Float>(bp[j * 3 + 0], bp[j * 3 + 1], bp[j * 3 + 2])
            let HppInvbp = matVec3x3(matrixOffset: invBase, m: HppInv, v: bp_j)
            for idx in attachedIndices {
                let b = htpBlocks[idx]
                if freezeFirstCamera && b.cam == 0 { continue }
                let fI = b.cam - firstFree
                let contribRHS = matVec6x3(m6x3: b.m, v: HppInvbp)
                for r in 0..<6 { redRHS[fI * 6 + r] -= contribRHS[r] }
                for idx2 in attachedIndices {
                    let b2 = htpBlocks[idx2]
                    if freezeFirstCamera && b2.cam == 0 { continue }
                    let fK = b2.cam - firstFree
                    let HppInvHtpkT = matMul3x3_3x6T(matrixOffset: invBase, m: HppInv, n6x3: b2.m)
                    let M = matMul6x3_3x6(a6x3: b.m, b3x6: HppInvHtpkT)
                    for r in 0..<6 {
                        for c in 0..<6 {
                            redLHS[(fI * 6 + r) * D + (fK * 6 + c)] -= M[r * 6 + c]
                        }
                    }
                }
            }
        }
    }

    // MARK: - Cost

    /// Objective cost: sum of per-observation Huber ρ(|r|) (0.5·|r|² if `huberThreshold` is nil).
    /// Observations that end up behind the camera (z ≤ 1e-3) are charged the cost of a
    /// `behindPenaltyPx` residual (excluding them would make steps that push points behind the
    /// camera look like cost drops).
    static func robustCost(
        state: State,
        intrinsics: [Intrinsics],
        observations: [Observation],
        huberThreshold: Float?,
        behindPenaltyPx: Float = 100
    ) -> Float {
        guard !observations.isEmpty else { return 0 }
        let (residualsFlat, zCam) = computeResidualsGPU(
            state: state, intrinsics: intrinsics, observations: observations
        )
        let n = observations.count
        let res2 = residualsFlat.reshaped([n, 2])
        let mag = MLX.sqrt((res2 * res2).sum(axis: 1))
        func rho(_ r: MLXArray) -> MLXArray {
            guard let k = huberThreshold else { return 0.5 * r * r }
            return MLX.which(r .<= k, 0.5 * r * r, k * (r - 0.5 * k))
        }
        let valid = zCam .> Float(1e-3)
        let penalty = rho(MLXArray(behindPenaltyPx))
        let per = MLX.which(valid, rho(mag), penalty)
        let total = per.sum()
        eval(total)
        return total.item(Float.self)
    }

    /// Prior-pose constraint term. With the left perturbation (R' = exp(ω)R, t' = exp(ω)t + v):
    ///   orientation: e_R = Log(R_pᵀ R), ∂e_R/∂ω ≈ Rᵀ  →  H_ωω += w_R·I,  b_ω += −w_R·R·e_R
    ///   position:    e_O = O − O_p,     ∂e_O/∂v = −Rᵀ  →  H_vv += w_O·I,  b_v += w_O·R·e_O
    /// (b follows the existing kernels' convention b = −Σ w Jᵀ r).
    struct PosePrior {
        let wRot: Double
        let wPos: Double
        let priorR: [simd_double3x3]
        let priorO: [SIMD3<Double>]

        init?(parameters p: Parameters, viewmats: [[Float]]?) {
            guard let vms = viewmats, p.priorRotationSigmaDeg != nil || p.priorPositionSigma != nil else { return nil }
            let spx = Double(p.priorPixelSigma)
            wRot = p.priorRotationSigmaDeg.map { let s = Double($0) * .pi / 180; return (spx / s) * (spx / s) } ?? 0
            wPos = p.priorPositionSigma.map { let s = Double($0); return (spx / s) * (spx / s) } ?? 0
            priorR = vms.map { PosePrior.rot($0) }
            priorO = vms.map { PosePrior.center($0) }
        }

        static func rot(_ v: [Float]) -> simd_double3x3 {
            simd_double3x3(rows: [
                SIMD3(Double(v[0]), Double(v[1]), Double(v[2])),
                SIMD3(Double(v[4]), Double(v[5]), Double(v[6])),
                SIMD3(Double(v[8]), Double(v[9]), Double(v[10])),
            ])
        }
        static func center(_ v: [Float]) -> SIMD3<Double> {
            -(rot(v).transpose * SIMD3(Double(v[3]), Double(v[7]), Double(v[11])))
        }
        static func logSO3(_ R: simd_double3x3) -> SIMD3<Double> {
            let c = max(-1, min(1, (R[0][0] + R[1][1] + R[2][2] - 1) / 2))
            let th = acos(c)
            let v = SIMD3(R[1][2] - R[2][1], R[2][0] - R[0][2], R[0][1] - R[1][0])
            return th < 1e-9 ? 0.5 * v : v * (th / (2 * sin(th)))
        }

        func cost(_ state: State) -> Float {
            var c = 0.0
            for (i, v) in state.viewmats.enumerated() {
                let R = PosePrior.rot(v)
                c += 0.5 * wRot * simd_length_squared(PosePrior.logSO3(priorR[i].transpose * R))
                c += 0.5 * wPos * simd_length_squared(PosePrior.center(v) - priorO[i])
            }
            return Float(c)
        }

        func accumulate(state: State, Htt: inout [Float], bt: inout [Float], skipFirst: Bool) {
            for (i, v) in state.viewmats.enumerated() where !(skipFirst && i == 0) {
                let R = PosePrior.rot(v)
                let eR = PosePrior.logSO3(priorR[i].transpose * R)
                let eO = PosePrior.center(v) - priorO[i]
                let bw = -(R * eR) * wRot
                let bv = (R * eO) * wPos
                for k in 0..<3 {
                    Htt[i * 36 + k * 7] += Float(wRot)
                    Htt[i * 36 + (k + 3) * 7] += Float(wPos)
                    bt[i * 6 + k] += Float(bw[k])
                    bt[i * 6 + 3 + k] += Float(bv[k])
                }
            }
        }
    }

    /// RMS residual (px) over observations with z_cam > 1e-3.
    static func residualNorm(
        state: State,
        intrinsics: [Intrinsics],
        observations: [Observation]
    ) -> Float {
        guard !observations.isEmpty else { return 0 }
        let (residualsFlat, zCam) = computeResidualsGPU(
            state: state, intrinsics: intrinsics, observations: observations
        )
        let valid = (zCam .> Float(1e-3)).asType(.float32)
        let n = observations.count
        let res2 = residualsFlat.reshaped([n, 2])
        let sqMag = (res2 * res2).sum(axis: 1)   // [N]
        let sumSq = (sqMag * valid).sum()
        let count = valid.sum()
        eval(sumSq, count)
        let cs = count.item(Float.self)
        if cs <= 0 { return 0 }
        return (sumSq.item(Float.self) / cs).squareRoot()
    }
}
