// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// BA Schur complement stage (PocketGS Sec. III-A 2). All 1 thread / point:
//  (1) ba_hpp_invert: H_pp⁻¹  (2) ba_schur_reduce: atomic add into the reduced system
//   (3) ba_back_substitute: δ_pt = H_pp⁻¹ (b_p - Σ H_tpᵀ δ_cam)
// Per-point observation lists (sorted_obs_idx / starts / counts) are built by the host.
// htp_blocks is the output of ba_jacobian_accumulate (observation order).

// ===========================================================================
// (1) H_pp 3×3 inverse
// ===========================================================================
StructuredBuffer<float> Hpp_in;      // [nPt * 9] row-major
StructuredBuffer<int>   hpp_dims;    // [1] : nPt

RWStructuredBuffer<float> Hpp_inv;   // [nPt * 9]

[shader("compute")]
[numthreads(128, 1, 1)]
void ba_hpp_invert(uint3 tid : SV_DispatchThreadID)
{
    int j = int(tid.x);
    int N = hpp_dims[0];
    if (j >= N) { return; }

    int base = j * 9;
    float a00 = Hpp_in[base + 0], a01 = Hpp_in[base + 1], a02 = Hpp_in[base + 2];
    float a10 = Hpp_in[base + 3], a11 = Hpp_in[base + 4], a12 = Hpp_in[base + 5];
    float a20 = Hpp_in[base + 6], a21 = Hpp_in[base + 7], a22 = Hpp_in[base + 8];

    // Cramer's rule. Like the CPU invert3x3, returns identity when degenerate
    float c00 =  (a11 * a22 - a12 * a21);
    float c01 = -(a10 * a22 - a12 * a20);
    float c02 =  (a10 * a21 - a11 * a20);
    float det = a00 * c00 + a01 * c01 + a02 * c02;
    if (abs(det) < 1e-12) {
        Hpp_inv[base + 0] = 1.0; Hpp_inv[base + 1] = 0.0; Hpp_inv[base + 2] = 0.0;
        Hpp_inv[base + 3] = 0.0; Hpp_inv[base + 4] = 1.0; Hpp_inv[base + 5] = 0.0;
        Hpp_inv[base + 6] = 0.0; Hpp_inv[base + 7] = 0.0; Hpp_inv[base + 8] = 1.0;
        return;
    }
    float invDet = 1.0 / det;
    // Transposed cofactor matrix / det
    Hpp_inv[base + 0] = c00 * invDet;
    Hpp_inv[base + 1] = -(a01 * a22 - a02 * a21) * invDet;
    Hpp_inv[base + 2] =  (a01 * a12 - a02 * a11) * invDet;
    Hpp_inv[base + 3] = c01 * invDet;
    Hpp_inv[base + 4] =  (a00 * a22 - a02 * a20) * invDet;
    Hpp_inv[base + 5] = -(a00 * a12 - a02 * a10) * invDet;
    Hpp_inv[base + 6] = c02 * invDet;
    Hpp_inv[base + 7] = -(a00 * a21 - a01 * a20) * invDet;
    Hpp_inv[base + 8] =  (a00 * a11 - a01 * a10) * invDet;
}

// ===========================================================================
// (2) Schur reduction
// ===========================================================================
// For each observation pair (i, k) of point j:
//  redLHS[fI, fK] -= H_tp[i] · H_pp⁻¹ · H_tp[k]ᵀ, redRHS[fI] -= H_tp[i] · H_pp⁻¹ · b_p[j]
// Skips cam < firstFree (fixed cameras). redLHS / redRHS must be zero-initialized.

StructuredBuffer<float> Hpp_inv_in;       // [nPt * 9]
StructuredBuffer<float> bp_in;            // [nPt * 3]
StructuredBuffer<float> htp_blocks;       // [nObs * 18] 6×3 block per observation
StructuredBuffer<int>   htp_cam_idx;      // [nObs]
StructuredBuffer<int>   sorted_obs_idx;   // [nObs] observation indices sorted by point
StructuredBuffer<int>   point_obs_starts; // [nPt]
StructuredBuffer<int>   point_obs_counts; // [nPt]
StructuredBuffer<int>   schur_dims;       // [3] : nPt, D (= 6*nFree), firstFree

RWStructuredBuffer<float> redLHS;         // [D*D] atomic
RWStructuredBuffer<float> redRHS;         // [D]   atomic

[shader("compute")]
[numthreads(64, 1, 1)]
void ba_schur_reduce(uint3 tid : SV_DispatchThreadID)
{
    int j = int(tid.x);
    int nPt = schur_dims[0];
    if (j >= nPt) { return; }

    int D = schur_dims[1];
    int firstFree = schur_dims[2];

    int count = point_obs_counts[j];
    if (count == 0) { return; }
    int start = point_obs_starts[j];

    int hppBase = j * 9;
    float bp0 = bp_in[j * 3 + 0];
    float bp1 = bp_in[j * 3 + 1];
    float bp2 = bp_in[j * 3 + 2];
    float hi00 = Hpp_inv_in[hppBase + 0], hi01 = Hpp_inv_in[hppBase + 1], hi02 = Hpp_inv_in[hppBase + 2];
    float hi10 = Hpp_inv_in[hppBase + 3], hi11 = Hpp_inv_in[hppBase + 4], hi12 = Hpp_inv_in[hppBase + 5];
    float hi20 = Hpp_inv_in[hppBase + 6], hi21 = Hpp_inv_in[hppBase + 7], hi22 = Hpp_inv_in[hppBase + 8];
    float HppInvbp0 = hi00 * bp0 + hi01 * bp1 + hi02 * bp2;
    float HppInvbp1 = hi10 * bp0 + hi11 * bp1 + hi12 * bp2;
    float HppInvbp2 = hi20 * bp0 + hi21 * bp1 + hi22 * bp2;

    for (int k = 0; k < count; k++) {
        int obsI = sorted_obs_idx[start + k];
        int camI = htp_cam_idx[obsI];
        if (camI < firstFree) { continue; }   // fixed camera
        int fI = camI - firstFree;

        int bI = obsI * 18;
        [unroll]
        for (int r = 0; r < 6; r++) {
            float c = htp_blocks[bI + r * 3 + 0] * HppInvbp0
                    + htp_blocks[bI + r * 3 + 1] * HppInvbp1
                    + htp_blocks[bI + r * 3 + 2] * HppInvbp2;
            InterlockedAdd(redRHS[fI * 6 + r], -c);
        }

        for (int k2 = 0; k2 < count; k2++) {
            int obsK = sorted_obs_idx[start + k2];
            int camK = htp_cam_idx[obsK];
            if (camK < firstFree) { continue; }
            int fK = camK - firstFree;
            int bK = obsK * 18;

            // Expand htX_c = (H_pp⁻¹ · H_tp[k]ᵀ)[x, c] (3×6) into registers
            float ht0_0 = hi00 * htp_blocks[bK + 0 * 3 + 0] + hi01 * htp_blocks[bK + 0 * 3 + 1] + hi02 * htp_blocks[bK + 0 * 3 + 2];
            float ht0_1 = hi00 * htp_blocks[bK + 1 * 3 + 0] + hi01 * htp_blocks[bK + 1 * 3 + 1] + hi02 * htp_blocks[bK + 1 * 3 + 2];
            float ht0_2 = hi00 * htp_blocks[bK + 2 * 3 + 0] + hi01 * htp_blocks[bK + 2 * 3 + 1] + hi02 * htp_blocks[bK + 2 * 3 + 2];
            float ht0_3 = hi00 * htp_blocks[bK + 3 * 3 + 0] + hi01 * htp_blocks[bK + 3 * 3 + 1] + hi02 * htp_blocks[bK + 3 * 3 + 2];
            float ht0_4 = hi00 * htp_blocks[bK + 4 * 3 + 0] + hi01 * htp_blocks[bK + 4 * 3 + 1] + hi02 * htp_blocks[bK + 4 * 3 + 2];
            float ht0_5 = hi00 * htp_blocks[bK + 5 * 3 + 0] + hi01 * htp_blocks[bK + 5 * 3 + 1] + hi02 * htp_blocks[bK + 5 * 3 + 2];
            float ht1_0 = hi10 * htp_blocks[bK + 0 * 3 + 0] + hi11 * htp_blocks[bK + 0 * 3 + 1] + hi12 * htp_blocks[bK + 0 * 3 + 2];
            float ht1_1 = hi10 * htp_blocks[bK + 1 * 3 + 0] + hi11 * htp_blocks[bK + 1 * 3 + 1] + hi12 * htp_blocks[bK + 1 * 3 + 2];
            float ht1_2 = hi10 * htp_blocks[bK + 2 * 3 + 0] + hi11 * htp_blocks[bK + 2 * 3 + 1] + hi12 * htp_blocks[bK + 2 * 3 + 2];
            float ht1_3 = hi10 * htp_blocks[bK + 3 * 3 + 0] + hi11 * htp_blocks[bK + 3 * 3 + 1] + hi12 * htp_blocks[bK + 3 * 3 + 2];
            float ht1_4 = hi10 * htp_blocks[bK + 4 * 3 + 0] + hi11 * htp_blocks[bK + 4 * 3 + 1] + hi12 * htp_blocks[bK + 4 * 3 + 2];
            float ht1_5 = hi10 * htp_blocks[bK + 5 * 3 + 0] + hi11 * htp_blocks[bK + 5 * 3 + 1] + hi12 * htp_blocks[bK + 5 * 3 + 2];
            float ht2_0 = hi20 * htp_blocks[bK + 0 * 3 + 0] + hi21 * htp_blocks[bK + 0 * 3 + 1] + hi22 * htp_blocks[bK + 0 * 3 + 2];
            float ht2_1 = hi20 * htp_blocks[bK + 1 * 3 + 0] + hi21 * htp_blocks[bK + 1 * 3 + 1] + hi22 * htp_blocks[bK + 1 * 3 + 2];
            float ht2_2 = hi20 * htp_blocks[bK + 2 * 3 + 0] + hi21 * htp_blocks[bK + 2 * 3 + 1] + hi22 * htp_blocks[bK + 2 * 3 + 2];
            float ht2_3 = hi20 * htp_blocks[bK + 3 * 3 + 0] + hi21 * htp_blocks[bK + 3 * 3 + 1] + hi22 * htp_blocks[bK + 3 * 3 + 2];
            float ht2_4 = hi20 * htp_blocks[bK + 4 * 3 + 0] + hi21 * htp_blocks[bK + 4 * 3 + 1] + hi22 * htp_blocks[bK + 4 * 3 + 2];
            float ht2_5 = hi20 * htp_blocks[bK + 5 * 3 + 0] + hi21 * htp_blocks[bK + 5 * 3 + 1] + hi22 * htp_blocks[bK + 5 * 3 + 2];

            // M = H_tp[i] · (H_pp⁻¹ · H_tp[k]ᵀ) (6×6)
            [unroll]
            for (int r = 0; r < 6; r++) {
                float hi0 = htp_blocks[bI + r * 3 + 0];
                float hi1 = htp_blocks[bI + r * 3 + 1];
                float hi2 = htp_blocks[bI + r * 3 + 2];
                float m0 = hi0 * ht0_0 + hi1 * ht1_0 + hi2 * ht2_0;
                float m1 = hi0 * ht0_1 + hi1 * ht1_1 + hi2 * ht2_1;
                float m2 = hi0 * ht0_2 + hi1 * ht1_2 + hi2 * ht2_2;
                float m3 = hi0 * ht0_3 + hi1 * ht1_3 + hi2 * ht2_3;
                float m4 = hi0 * ht0_4 + hi1 * ht1_4 + hi2 * ht2_4;
                float m5 = hi0 * ht0_5 + hi1 * ht1_5 + hi2 * ht2_5;
                int rowBase = (fI * 6 + r) * D + fK * 6;
                InterlockedAdd(redLHS[rowBase + 0], -m0);
                InterlockedAdd(redLHS[rowBase + 1], -m1);
                InterlockedAdd(redLHS[rowBase + 2], -m2);
                InterlockedAdd(redLHS[rowBase + 3], -m3);
                InterlockedAdd(redLHS[rowBase + 4], -m4);
                InterlockedAdd(redLHS[rowBase + 5], -m5);
            }
        }
    }
}

// ===========================================================================
// (3) Back substitution: δ_pt[j] = H_pp⁻¹ (b_p[j] - Σ_i H_tp[i,j]ᵀ δ_cam_i)
// ===========================================================================

StructuredBuffer<float> Hpp_inv_bs;       // [nPt * 9]
StructuredBuffer<float> bp_bs;            // [nPt * 3]
StructuredBuffer<float> htp_blocks_bs;    // [nObs * 18]
StructuredBuffer<int>   htp_cam_idx_bs;   // [nObs]
StructuredBuffer<int>   sorted_obs_idx_bs;// [nObs]
StructuredBuffer<int>   point_obs_starts_bs; // [nPt]
StructuredBuffer<int>   point_obs_counts_bs; // [nPt]
StructuredBuffer<float> deltaCam_bs;      // [nCam * 6]  (0 in frozen cam slots)
StructuredBuffer<int>   bs_dims;          // [1] : nPt

RWStructuredBuffer<float> deltaPt_out;    // [nPt * 3]

[shader("compute")]
[numthreads(128, 1, 1)]
void ba_back_substitute(uint3 tid : SV_DispatchThreadID)
{
    int j = int(tid.x);
    int N = bs_dims[0];
    if (j >= N) { return; }

    deltaPt_out[j * 3 + 0] = 0.0;
    deltaPt_out[j * 3 + 1] = 0.0;
    deltaPt_out[j * 3 + 2] = 0.0;

    int count = point_obs_counts_bs[j];
    if (count == 0) { return; }
    int start = point_obs_starts_bs[j];

    float rhs0 = bp_bs[j * 3 + 0];
    float rhs1 = bp_bs[j * 3 + 1];
    float rhs2 = bp_bs[j * 3 + 2];

    for (int k = 0; k < count; k++) {
        int obsI = sorted_obs_idx_bs[start + k];
        int camI = htp_cam_idx_bs[obsI];
        int dcBase = camI * 6;
        float dc0 = deltaCam_bs[dcBase + 0];
        float dc1 = deltaCam_bs[dcBase + 1];
        float dc2 = deltaCam_bs[dcBase + 2];
        float dc3 = deltaCam_bs[dcBase + 3];
        float dc4 = deltaCam_bs[dcBase + 4];
        float dc5 = deltaCam_bs[dcBase + 5];
        // H_tp[i]ᵀ · dc (3)
        int bI = obsI * 18;
        float c0 = htp_blocks_bs[bI + 0 * 3 + 0] * dc0
                 + htp_blocks_bs[bI + 1 * 3 + 0] * dc1
                 + htp_blocks_bs[bI + 2 * 3 + 0] * dc2
                 + htp_blocks_bs[bI + 3 * 3 + 0] * dc3
                 + htp_blocks_bs[bI + 4 * 3 + 0] * dc4
                 + htp_blocks_bs[bI + 5 * 3 + 0] * dc5;
        float c1 = htp_blocks_bs[bI + 0 * 3 + 1] * dc0
                 + htp_blocks_bs[bI + 1 * 3 + 1] * dc1
                 + htp_blocks_bs[bI + 2 * 3 + 1] * dc2
                 + htp_blocks_bs[bI + 3 * 3 + 1] * dc3
                 + htp_blocks_bs[bI + 4 * 3 + 1] * dc4
                 + htp_blocks_bs[bI + 5 * 3 + 1] * dc5;
        float c2 = htp_blocks_bs[bI + 0 * 3 + 2] * dc0
                 + htp_blocks_bs[bI + 1 * 3 + 2] * dc1
                 + htp_blocks_bs[bI + 2 * 3 + 2] * dc2
                 + htp_blocks_bs[bI + 3 * 3 + 2] * dc3
                 + htp_blocks_bs[bI + 4 * 3 + 2] * dc4
                 + htp_blocks_bs[bI + 5 * 3 + 2] * dc5;
        rhs0 -= c0;
        rhs1 -= c1;
        rhs2 -= c2;
    }

    int hi = j * 9;
    float hi00 = Hpp_inv_bs[hi + 0], hi01 = Hpp_inv_bs[hi + 1], hi02 = Hpp_inv_bs[hi + 2];
    float hi10 = Hpp_inv_bs[hi + 3], hi11 = Hpp_inv_bs[hi + 4], hi12 = Hpp_inv_bs[hi + 5];
    float hi20 = Hpp_inv_bs[hi + 6], hi21 = Hpp_inv_bs[hi + 7], hi22 = Hpp_inv_bs[hi + 8];

    deltaPt_out[j * 3 + 0] = hi00 * rhs0 + hi01 * rhs1 + hi02 * rhs2;
    deltaPt_out[j * 3 + 1] = hi10 * rhs0 + hi11 * rhs1 + hi12 * rhs2;
    deltaPt_out[j * 3 + 2] = hi20 * rhs0 + hi21 * rhs1 + hi22 * rhs2;
}
