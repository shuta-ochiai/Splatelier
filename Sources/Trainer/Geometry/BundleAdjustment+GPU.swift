// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXFast

// GPU kernel wrappers for BA (Jacobian / H and b accumulation, Schur reduction, back-substitution,
// residuals).
extension BundleAdjustment {

    /// Per-observation residual kernel (shared by `residualNorm` and `filterReprojection`).
    static let residualKernel = SlangCompiledKernel.required("ba_residual")

    /// Per-observation Jacobian + atomic accumulation of H / b (1 thread / obs). It mixes in plain
    /// writes of Htp_blocks, so it cannot use `atomicOutputs`; the atomic adds go through an unsafe
    /// cast (same as rasterize_bwd). Small errors from addition order are acceptable.
    static let jacobianKernel = SlangCompiledKernel.required("ba_jacobian_accumulate")

    /// Per-observation H_tp blocks (camera, point, 6×3 row-major).
    struct HtpBlock { let cam: Int; let pt: Int; var m: [Float] }

    static let hppInvertKernel = SlangCompiledKernel.required("ba_hpp_invert")

    /// Loops over observation pairs per point and atomically accumulates into the reduced system.
    /// Writes concentrate on the same cells, so `atomicOutputs: true` is required (without it the
    /// adds go through an unsafe cast and values drift under contention).
    static let schurReduceKernel = SlangCompiledKernel.required("ba_schur_reduce", atomicOutputs: true)

    static let backSubstituteKernel = SlangCompiledKernel.required("ba_back_substitute")

    /// Computes H_pp⁻¹ and the Schur reduction on the GPU and returns the reduced system and HppInv
    /// to the host. Small systems suffer cancellation from differing atomic addition order, so
    /// callers use this only when there are many points.
    static func runHppInvertAndSchurGPU(
        Hpp: [Float], bp: [Float], Htt: [Float], bt: [Float],
        htpBlocks: [HtpBlock],
        nCam: Int, nPt: Int, nFree: Int, firstFree: Int
    ) -> (redLHS: [Float], redRHS: [Float], HppInv: [Float]) {
        let D = 6 * nFree
        precondition(D > 0, "nFree must be > 0")

        let hppArr = MLXArray(Hpp, [nPt * 9])
        let hppDimsArr = MLXArray([Int32(nPt)], [1])
        let hppInvOut = hppInvertKernel(
            [hppArr, hppDimsArr],
            grid: (nPt, 1, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [[nPt * 9]],
            outputDTypes: [.float32]
        )
        let HppInvArr = hppInvOut[0]
        eval(HppInvArr)

        // The H_tt diagonal blocks and b_t stay on the host and are summed with the kernel's delta
        // at the end.
        var redLHSHost = [Float](repeating: 0, count: D * D)
        var redRHSHost = [Float](repeating: 0, count: D)
        for camI in firstFree..<nCam {
            let fI = camI - firstFree
            for r in 0..<6 {
                redRHSHost[fI * 6 + r] = bt[camI * 6 + r]
                for c in 0..<6 {
                    redLHSHost[(fI * 6 + r) * D + (fI * 6 + c)] = Htt[camI * 36 + r * 6 + c]
                }
            }
        }

        // htpBlocks is passed in its original order; the per-point ordering is expressed by an
        // index array (sortedHtpIdx) and per-point (start, count).
        let nValid = htpBlocks.count
        if nValid == 0 {
            // No valid observations: no Schur contribution.
            return (redLHS: redLHSHost, redRHS: redRHSHost, HppInv: HppInvArr.asArray(Float.self))
        }
        var sortedHtpIdx = Array(0..<nValid)
        sortedHtpIdx.sort { htpBlocks[$0].pt < htpBlocks[$1].pt }
        var pointStarts = [Int32](repeating: 0, count: nPt)
        var pointCounts = [Int32](repeating: 0, count: nPt)
        for k in 0..<nValid { pointCounts[htpBlocks[sortedHtpIdx[k]].pt] += 1 }
        var running: Int32 = 0
        for j in 0..<nPt {
            pointStarts[j] = running
            running += pointCounts[j]
        }

        var htpFlat = [Float](repeating: 0, count: nValid * 18)
        var htpCamIdx = [Int32](repeating: 0, count: nValid)
        for k in 0..<nValid {
            let b = htpBlocks[k]
            htpCamIdx[k] = Int32(b.cam)
            for x in 0..<18 { htpFlat[k * 18 + x] = b.m[x] }
        }
        var sortedHtpIdxI32 = [Int32](repeating: 0, count: nValid)
        for k in 0..<nValid { sortedHtpIdxI32[k] = Int32(sortedHtpIdx[k]) }

        // The kernel atomically subtracts into a zero-initialized delta buffer.
        let bpArr = MLXArray(bp, [nPt * 3])
        let htpFlatArr = MLXArray(htpFlat, [nValid * 18])
        let htpCamArr = MLXArray(htpCamIdx, [nValid])
        let sortedIdxArr = MLXArray(sortedHtpIdxI32, [nValid])
        let pointStartsArr = MLXArray(pointStarts, [nPt])
        let pointCountsArr = MLXArray(pointCounts, [nPt])
        let schurDimsArr = MLXArray([Int32(nPt), Int32(D), Int32(firstFree)], [3])

        let schurOut = schurReduceKernel(
            [HppInvArr, bpArr, htpFlatArr, htpCamArr,
             sortedIdxArr, pointStartsArr, pointCountsArr, schurDimsArr],
            grid: (nPt, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[D * D], [D]],
            outputDTypes: [.float32, .float32],
            initValue: 0.0
        )
        let lhsDelta = schurOut[0]
        let rhsDelta = schurOut[1]
        eval(lhsDelta, rhsDelta)

        let lhsDeltaHost = lhsDelta.asArray(Float.self)
        let rhsDeltaHost = rhsDelta.asArray(Float.self)
        // Near convergence H_tt ≈ Σ(Schur terms) cancels, so the sum is done in double.
        for i in 0..<(D * D) {
            redLHSHost[i] = Float(Double(redLHSHost[i]) + Double(lhsDeltaHost[i]))
        }
        for i in 0..<D {
            redRHSHost[i] = Float(Double(redRHSHost[i]) + Double(rhsDeltaHost[i]))
        }

        return (redLHS: redLHSHost, redRHS: redRHSHost, HppInv: HppInvArr.asArray(Float.self))
    }

    /// GPU back-substitution: δ_pt[j] = H_pp⁻¹ (b_p[j] - Σ_i H_tp[i,j]ᵀ δ_cam_i).
    /// htpBlocks ordering follows the same convention as `runHppInvertAndSchurGPU`.
    static func runBackSubstituteGPU(
        HppInv: [Float], bp: [Float],
        htpBlocks: [HtpBlock], deltaCam: [Float],
        nCam: Int, nPt: Int
    ) -> [Float] {
        if nPt == 0 { return [] }
        let nValid = htpBlocks.count
        if nValid == 0 {
            // No valid observations: δ_pt = HppInv · bp_j
            var deltaPt = [Float](repeating: 0, count: nPt * 3)
            for j in 0..<nPt {
                let bp0 = bp[j * 3 + 0], bp1 = bp[j * 3 + 1], bp2 = bp[j * 3 + 2]
                let hi = j * 9
                deltaPt[j * 3 + 0] = HppInv[hi + 0] * bp0 + HppInv[hi + 1] * bp1 + HppInv[hi + 2] * bp2
                deltaPt[j * 3 + 1] = HppInv[hi + 3] * bp0 + HppInv[hi + 4] * bp1 + HppInv[hi + 5] * bp2
                deltaPt[j * 3 + 2] = HppInv[hi + 6] * bp0 + HppInv[hi + 7] * bp1 + HppInv[hi + 8] * bp2
            }
            return deltaPt
        }

        var sortedHtpIdx = Array(0..<nValid)
        sortedHtpIdx.sort { htpBlocks[$0].pt < htpBlocks[$1].pt }
        var pointStarts = [Int32](repeating: 0, count: nPt)
        var pointCounts = [Int32](repeating: 0, count: nPt)
        for k in 0..<nValid { pointCounts[htpBlocks[sortedHtpIdx[k]].pt] += 1 }
        var running: Int32 = 0
        for j in 0..<nPt {
            pointStarts[j] = running
            running += pointCounts[j]
        }

        var htpFlat = [Float](repeating: 0, count: nValid * 18)
        var htpCamIdx = [Int32](repeating: 0, count: nValid)
        for k in 0..<nValid {
            let b = htpBlocks[k]
            htpCamIdx[k] = Int32(b.cam)
            for x in 0..<18 { htpFlat[k * 18 + x] = b.m[x] }
        }
        var sortedHtpIdxI32 = [Int32](repeating: 0, count: nValid)
        for k in 0..<nValid { sortedHtpIdxI32[k] = Int32(sortedHtpIdx[k]) }

        let HppInvArr = MLXArray(HppInv, [nPt * 9])
        let bpArr = MLXArray(bp, [nPt * 3])
        let htpFlatArr = MLXArray(htpFlat, [nValid * 18])
        let htpCamArr = MLXArray(htpCamIdx, [nValid])
        let sortedIdxArr = MLXArray(sortedHtpIdxI32, [nValid])
        let pointStartsArr = MLXArray(pointStarts, [nPt])
        let pointCountsArr = MLXArray(pointCounts, [nPt])
        let deltaCamArr = MLXArray(deltaCam, [nCam * 6])
        let bsDimsArr = MLXArray([Int32(nPt)], [1])

        let bsOut = backSubstituteKernel(
            [HppInvArr, bpArr, htpFlatArr, htpCamArr,
             sortedIdxArr, pointStartsArr, pointCountsArr, deltaCamArr, bsDimsArr],
            grid: (nPt, 1, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [[nPt * 3]],
            outputDTypes: [.float32]
        )
        let deltaPtArr = bsOut[0]
        eval(deltaPtArr)
        return deltaPtArr.asArray(Float.self)
    }

    /// Builds H / b on the GPU and returns what is needed from Schur onward to the host.
    /// Observations with z ≤ 1e-3 come back from the kernel with Htp_valid = 0 and are dropped from
    /// htpBlocks.
    static func computeJacobianAndAccumulateGPU(
        state: State,
        intrinsics: [Intrinsics],
        observations: [Observation],
        huberThreshold: Float?,
        nCam: Int, nPt: Int
    ) -> (Htt: [Float], Hpp: [Float], bt: [Float], bp: [Float], htpBlocks: [HtpBlock]) {
        let nObs = observations.count
        if nObs == 0 {
            return (
                Htt: [Float](repeating: 0, count: nCam * 36),
                Hpp: [Float](repeating: 0, count: nPt * 9),
                bt: [Float](repeating: 0, count: nCam * 6),
                bp: [Float](repeating: 0, count: nPt * 3),
                htpBlocks: []
            )
        }

        var vmFlat = [Float](repeating: 0, count: nCam * 16)
        for (i, vm) in state.viewmats.enumerated() {
            precondition(vm.count == 16, "viewmat must be 16 floats")
            for k in 0..<16 { vmFlat[i * 16 + k] = vm[k] }
        }
        var intrFlat = [Float](repeating: 0, count: nCam * 4)
        for (i, intr) in intrinsics.enumerated() {
            intrFlat[i * 4 + 0] = intr.fx
            intrFlat[i * 4 + 1] = intr.fy
            intrFlat[i * 4 + 2] = intr.cx
            intrFlat[i * 4 + 3] = intr.cy
        }
        var ptsFlat = [Float](repeating: 0, count: nPt * 3)
        for (i, p) in state.points.enumerated() {
            ptsFlat[i * 3 + 0] = p.x
            ptsFlat[i * 3 + 1] = p.y
            ptsFlat[i * 3 + 2] = p.z
        }
        var camIdx = [Int32](repeating: 0, count: nObs)
        var ptIdx = [Int32](repeating: 0, count: nObs)
        var pixels = [Float](repeating: 0, count: nObs * 2)
        for (i, obs) in observations.enumerated() {
            camIdx[i] = Int32(obs.cameraIndex)
            ptIdx[i] = Int32(obs.pointIndex)
            pixels[i * 2 + 0] = obs.pixel.x
            pixels[i * 2 + 1] = obs.pixel.y
        }
        // jac_params = [huber_k (-1 disables), unused]
        let huberK = huberThreshold ?? -1
        let params = [Float(huberK), 0]

        let vmArr = MLXArray(vmFlat, [nCam * 16])
        let intrArr = MLXArray(intrFlat, [nCam * 4])
        let ptsArr = MLXArray(ptsFlat, [nPt * 3])
        let camIdxArr = MLXArray(camIdx, [nObs])
        let ptIdxArr = MLXArray(ptIdx, [nObs])
        let pixelsArr = MLXArray(pixels, [nObs * 2])
        let paramsArr = MLXArray(params, [2])
        let dimsArr = MLXArray([Int32(nObs)], [1])

        // Zero-initialize the atomic accumulation targets.
        let outputs = jacobianKernel(
            [vmArr, intrArr, ptsArr, camIdxArr, ptIdxArr, pixelsArr, paramsArr, dimsArr],
            grid: (nObs, 1, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [
                [nCam * 36],   // Htt
                [nPt * 9],     // Hpp
                [nCam * 6],    // bt
                [nPt * 3],     // bp
                [nObs * 18],   // Htp_blocks
                [nObs],        // Htp_valid
            ],
            outputDTypes: [.float32, .float32, .float32, .float32, .float32, .int32],
            initValue: 0.0
        )
        let HttArr = outputs[0]
        let HppArr = outputs[1]
        let btArr = outputs[2]
        let bpArr = outputs[3]
        let HtpArr = outputs[4]
        let HtpValidArr = outputs[5]
        eval(HttArr, HppArr, btArr, bpArr, HtpArr, HtpValidArr)

        let HttHost = HttArr.asArray(Float.self)
        let HppHost = HppArr.asArray(Float.self)
        let btHost = btArr.asArray(Float.self)
        let bpHost = bpArr.asArray(Float.self)
        let HtpHost = HtpArr.asArray(Float.self)
        let HtpValidHost = HtpValidArr.asArray(Int32.self)

        var htpBlocks: [HtpBlock] = []
        htpBlocks.reserveCapacity(nObs)
        for i in 0..<nObs {
            if HtpValidHost[i] == 0 { continue }
            let m = Array(HtpHost[(i * 18)..<((i + 1) * 18)])
            htpBlocks.append(HtpBlock(
                cam: observations[i].cameraIndex,
                pt: observations[i].pointIndex,
                m: m
            ))
        }
        return (Htt: HttHost, Hpp: HppHost, bt: btHost, bp: bpHost, htpBlocks: htpBlocks)
    }

    /// Computes per-observation residuals (du, dv) and z_cam on the GPU and returns them as
    /// MLXArrays.
    static func computeResidualsGPU(
        state: State,
        intrinsics: [Intrinsics],
        observations: [Observation]
    ) -> (residuals: MLXArray, zCam: MLXArray) {
        let nCam = state.viewmats.count
        let nPt = state.points.count
        let nObs = observations.count
        precondition(nObs > 0, "computeResidualsGPU: observations must be non-empty")
        precondition(intrinsics.count == nCam, "intrinsics / viewmats count mismatch")

        var vmFlat = [Float](repeating: 0, count: nCam * 16)
        for (i, vm) in state.viewmats.enumerated() {
            precondition(vm.count == 16, "viewmat must be 16 floats")
            for k in 0..<16 { vmFlat[i * 16 + k] = vm[k] }
        }
        var intrFlat = [Float](repeating: 0, count: nCam * 4)
        for (i, intr) in intrinsics.enumerated() {
            intrFlat[i * 4 + 0] = intr.fx
            intrFlat[i * 4 + 1] = intr.fy
            intrFlat[i * 4 + 2] = intr.cx
            intrFlat[i * 4 + 3] = intr.cy
        }
        var ptsFlat = [Float](repeating: 0, count: nPt * 3)
        for (i, p) in state.points.enumerated() {
            ptsFlat[i * 3 + 0] = p.x
            ptsFlat[i * 3 + 1] = p.y
            ptsFlat[i * 3 + 2] = p.z
        }
        var camIdx = [Int32](repeating: 0, count: nObs)
        var ptIdx = [Int32](repeating: 0, count: nObs)
        var pixels = [Float](repeating: 0, count: nObs * 2)
        for (i, obs) in observations.enumerated() {
            camIdx[i] = Int32(obs.cameraIndex)
            ptIdx[i] = Int32(obs.pointIndex)
            pixels[i * 2 + 0] = obs.pixel.x
            pixels[i * 2 + 1] = obs.pixel.y
        }

        let vmArr = MLXArray(vmFlat, [nCam * 16])
        let intrArr = MLXArray(intrFlat, [nCam * 4])
        let ptsArr = MLXArray(ptsFlat, [nPt * 3])
        let camIdxArr = MLXArray(camIdx, [nObs])
        let ptIdxArr = MLXArray(ptIdx, [nObs])
        let pixelsArr = MLXArray(pixels, [nObs * 2])
        let dimsArr = MLXArray([Int32(nObs)], [1])

        let outputs = residualKernel(
            [vmArr, intrArr, ptsArr, camIdxArr, ptIdxArr, pixelsArr, dimsArr],
            grid: (nObs, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[nObs * 2], [nObs]],
            outputDTypes: [.float32, .float32]
        )
        return (residuals: outputs[0], zCam: outputs[1])
    }
}
