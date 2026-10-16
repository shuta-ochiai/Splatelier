// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Plane sweep cost volume (PocketGS Operator G 3b-2). Census Hamming cost of the target view
// against one ref.

import Foundation
import MLX
import MLXFast

public enum PlaneSweepCostVolume {

    private static let kernel = SlangCompiledKernel.required("plane_sweep_cost")

    private static let medianKernel = SlangCompiledKernel.required("plane_sweep_cost_median")

    /// Maximum number of references the fused kernel handles (must match `MEDIAN_M_MAX` in
    /// `plane_sweep_cost_median.slang`).
    public static let medianMaxRefs = 8

    /// Relative pose from target camera coordinates → ref camera coordinates = ref_viewmat ·
    /// target_viewmat⁻¹ (row-major 4x4).
    /// - Returns: `[12]` (R 9 + t 3, row-major)
    public static func relativePose(
        targetViewmat: [Float], refViewmat: [Float]
    ) -> [Float] {
        precondition(targetViewmat.count == 16 && refViewmat.count == 16)
        // The inverse of the rigid transform [R | t] is [Rᵀ | -Rᵀ t].
        let tr = targetViewmat
        let rT_r0 = tr[0]; let rT_r1 = tr[4]; let rT_r2 = tr[8]
        let rT_r3 = tr[1]; let rT_r4 = tr[5]; let rT_r5 = tr[9]
        let rT_r6 = tr[2]; let rT_r7 = tr[6]; let rT_r8 = tr[10]
        let tx = tr[3]; let ty = tr[7]; let tz = tr[11]
        let mTx = -(rT_r0 * tx + rT_r1 * ty + rT_r2 * tz)
        let mTy = -(rT_r3 * tx + rT_r4 * ty + rT_r5 * tz)
        let mTz = -(rT_r6 * tx + rT_r7 * ty + rT_r8 * tz)
        let rv = refViewmat
        func dot3(_ a0: Float, _ a1: Float, _ a2: Float,
                  _ b0: Float, _ b1: Float, _ b2: Float) -> Float {
            a0 * b0 + a1 * b1 + a2 * b2
        }
        // R_ref · R_targetᵀ
        let R00 = dot3(rv[0], rv[1], rv[2], rT_r0, rT_r3, rT_r6)
        let R01 = dot3(rv[0], rv[1], rv[2], rT_r1, rT_r4, rT_r7)
        let R02 = dot3(rv[0], rv[1], rv[2], rT_r2, rT_r5, rT_r8)
        let R10 = dot3(rv[4], rv[5], rv[6], rT_r0, rT_r3, rT_r6)
        let R11 = dot3(rv[4], rv[5], rv[6], rT_r1, rT_r4, rT_r7)
        let R12 = dot3(rv[4], rv[5], rv[6], rT_r2, rT_r5, rT_r8)
        let R20 = dot3(rv[8], rv[9], rv[10], rT_r0, rT_r3, rT_r6)
        let R21 = dot3(rv[8], rv[9], rv[10], rT_r1, rT_r4, rT_r7)
        let R22 = dot3(rv[8], rv[9], rv[10], rT_r2, rT_r5, rT_r8)
        // t_ref + R_ref · (-Rᵀ t_target)
        let tx_rel = rv[3]  + dot3(rv[0], rv[1], rv[2],  mTx, mTy, mTz)
        let ty_rel = rv[7]  + dot3(rv[4], rv[5], rv[6],  mTx, mTy, mTz)
        let tz_rel = rv[11] + dot3(rv[8], rv[9], rv[10], mTx, mTy, mTz)
        return [
            R00, R01, R02,
            R10, R11, R12,
            R20, R21, R22,
            tx_rel, ty_rel, tz_rel,
        ]
    }

    /// Plane sweep cost volume. Pixel centers follow the i + 0.5 convention (continuous coordinate
    /// → index is floor).
    /// - Parameters:
    ///   - relativeRT: Output of `relativePose` (target → ref)
    ///   - depths: `[K]` depth hypotheses (z in target camera coordinates)
    /// - Returns: `[H, W, K]` int32 Hamming distances. Hypotheses outside the ref view or behind
    ///   the camera get the max cost 25 (`CENSUS_MAX_BITS`).
    public static func compute(
        targetCensus: MLXArray, refCensus: MLXArray,
        intrinsicsTarget: [Float], intrinsicsRef: [Float],
        relativeRT: [Float],
        depths: [Float]
    ) -> MLXArray {
        precondition(targetCensus.shape.count == 2, "targetCensus must be [H, W]")
        precondition(refCensus.shape.count == 2, "refCensus must be [H_ref, W_ref]")
        precondition(intrinsicsTarget.count == 4 && intrinsicsRef.count == 4)
        precondition(relativeRT.count == 12)
        precondition(!depths.isEmpty)

        let h = targetCensus.shape[0]
        let w = targetCensus.shape[1]
        let hRef = refCensus.shape[0]
        let wRef = refCensus.shape[1]
        let k = depths.count

        let dims = MLXArray(
            [Int32(w), Int32(h), Int32(wRef), Int32(hRef), Int32(k)], [5]
        )
        let intrT = MLXArray(intrinsicsTarget, [4])
        let intrR = MLXArray(intrinsicsRef, [4])
        let rt = MLXArray(relativeRT, [12])
        let depthsArr = MLXArray(depths, [k])
        let outputs = kernel(
            [
                targetCensus.reshaped([h * w]),
                refCensus.reshaped([hRef * wRef]),
                intrT, intrR, rt, depthsArr, dims,
            ],
            grid: (((w + 31) / 32) * 32, ((h + 7) / 8) * 8, 1),
            threadGroup: (32, 8, 1),
            outputShapes: [[h * w * k]],
            outputDTypes: [.int32]
        )
        return outputs[0].reshaped([h, w, k])
    }

    /// Computes, in one kernel, the median along the reference axis (ascending index M/2) of the
    /// plane sweep costs (same formula as `compute`) for each of M references. Gives the same
    /// result as calling `compute` M times and taking `CostVolumeMVS.elementwiseMedian` (verified
    /// by tests).
    /// - Parameters:
    ///   - refCensuses: M `[H_ref, W_ref]` int32 (all the same resolution)
    ///   - intrinsicsRefs / relativeRTs: `[4]` / `[12]` per reference
    /// - Returns: `[H, W, K]` int32
    public static func computeMedian(
        targetCensus: MLXArray, refCensuses: [MLXArray],
        intrinsicsTarget: [Float], intrinsicsRefs: [[Float]],
        relativeRTs: [[Float]],
        depths: [Float]
    ) -> MLXArray {
        let m = refCensuses.count
        precondition(m >= 1 && m <= medianMaxRefs, "M=\(m) must be in 1...\(medianMaxRefs)")
        precondition(intrinsicsRefs.count == m && relativeRTs.count == m)
        precondition(targetCensus.shape.count == 2)
        let refShape = refCensuses[0].shape
        precondition(refCensuses.allSatisfy { $0.shape == refShape },
                     "all reference census maps must share one resolution")
        precondition(intrinsicsRefs.allSatisfy { $0.count == 4 } && relativeRTs.allSatisfy { $0.count == 12 })

        let h = targetCensus.shape[0]
        let w = targetCensus.shape[1]
        let hRef = refShape[0]
        let wRef = refShape[1]
        let k = depths.count

        let dims = MLXArray(
            [Int32(w), Int32(h), Int32(wRef), Int32(hRef), Int32(k), Int32(m)], [6]
        )
        let refs = MLX.concatenated(refCensuses.map { $0.reshaped([hRef * wRef]) }, axis: 0)
        let outputs = medianKernel(
            [
                targetCensus.reshaped([h * w]),
                refs,
                MLXArray(intrinsicsTarget, [4]),
                MLXArray(intrinsicsRefs.flatMap { $0 }, [m * 4]),
                MLXArray(relativeRTs.flatMap { $0 }, [m * 12]),
                MLXArray(depths, [k]),
                dims,
            ],
            grid: (((w + 31) / 32) * 32, ((h + 7) / 8) * 8, 1),
            threadGroup: (32, 8, 1),
            outputShapes: [[h * w * k]],
            outputDTypes: [.int32]
        )
        return outputs[0].reshaped([h, w, k])
    }
}
