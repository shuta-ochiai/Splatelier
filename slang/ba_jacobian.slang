// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Per-observation BA Jacobian and accumulation of normal-equation blocks (1 thread / obs). PocketGS
// Sec. III-A 2, Triggs "Bundle Adjustment — A Modern Synthesis" Sec. 4.5. r = π(R·P + t) - (u, v),
// J_cam = (∂π/∂Pc)·[-[Pc]_× | I] (left perturbation), Huber weight w = k/|r|. H_tt / H_pp / b_t /
// b_p use atomic adds, so outputs must be zero-initialized. Summation order is nondeterministic.
// Observations with z <= 1e-3 get Htp_valid = 0 and contribute nothing.

StructuredBuffer<float> viewmats;        // [num_cameras * 16]
StructuredBuffer<float> intrinsics_buf;  // [num_cameras * 4] (fx, fy, cx, cy)
StructuredBuffer<float> points;          // [num_points * 3]
StructuredBuffer<int>   obs_cam_idx;     // [num_obs]
StructuredBuffer<int>   obs_pt_idx;      // [num_obs]
StructuredBuffer<float> obs_pixels;      // [num_obs * 2]
StructuredBuffer<float> jac_params;      // [2] : huber_threshold (<=0 disables), unused
StructuredBuffer<int>   jac_dims;        // [1] : num_obs

RWStructuredBuffer<float> Htt;           // [num_cameras * 36]   atomic
RWStructuredBuffer<float> Hpp;           // [num_points * 9]     atomic
RWStructuredBuffer<float> bt;            // [num_cameras * 6]    atomic
RWStructuredBuffer<float> bp;            // [num_points * 3]     atomic
RWStructuredBuffer<float> Htp_blocks;    // [num_obs * 18]       one private slot per obs
RWStructuredBuffer<int>   Htp_valid;     // [num_obs]            1 if z > 1e-3

[shader("compute")]
[numthreads(128, 1, 1)]
void ba_jacobian_accumulate(uint3 tid : SV_DispatchThreadID)
{
    int i = int(tid.x);
    int N = jac_dims[0];
    if (i >= N) { return; }

    int cam = obs_cam_idx[i];
    int pt = obs_pt_idx[i];
    int vmBase = cam * 16;
    int inBase = cam * 4;
    int ptBase = pt * 3;

    float fx = intrinsics_buf[inBase + 0];
    float fy = intrinsics_buf[inBase + 1];
    float cx = intrinsics_buf[inBase + 2];
    float cy = intrinsics_buf[inBase + 3];

    float Px = points[ptBase + 0];
    float Py = points[ptBase + 1];
    float Pz = points[ptBase + 2];

    // P_c = R · P + t  (row-major viewmat)
    float pcx = viewmats[vmBase + 0] * Px + viewmats[vmBase + 1] * Py + viewmats[vmBase + 2]  * Pz + viewmats[vmBase + 3];
    float pcy = viewmats[vmBase + 4] * Px + viewmats[vmBase + 5] * Py + viewmats[vmBase + 6]  * Pz + viewmats[vmBase + 7];
    float pcz = viewmats[vmBase + 8] * Px + viewmats[vmBase + 9] * Py + viewmats[vmBase + 10] * Pz + viewmats[vmBase + 11];

    if (pcz <= 1e-3) {
        Htp_valid[i] = 0;
        // Zero-fill so stale values do not survive buffer reuse
        [unroll]
        for (int k = 0; k < 18; k++) { Htp_blocks[i * 18 + k] = 0.0; }
        return;
    }

    float invZ = 1.0 / pcz;
    float predU = fx * pcx * invZ + cx;
    float predV = fy * pcy * invZ + cy;
    float rU = predU - obs_pixels[i * 2 + 0];
    float rV = predV - obs_pixels[i * 2 + 1];

    // ∂π/∂Pc (2×3): [[fx/z, 0, -fx·x/z²], [0, fy/z, -fy·y/z²]]
    float dpi_00 = fx * invZ;
    float dpi_02 = -fx * pcx * invZ * invZ;
    float dpi_11 = fy * invZ;
    float dpi_12 = -fy * pcy * invZ * invZ;

    // J_point (2×3) = (∂π/∂Pc) · R
    float R00 = viewmats[vmBase + 0];
    float R01 = viewmats[vmBase + 1];
    float R02 = viewmats[vmBase + 2];
    float R10 = viewmats[vmBase + 4];
    float R11 = viewmats[vmBase + 5];
    float R12 = viewmats[vmBase + 6];
    float R20 = viewmats[vmBase + 8];
    float R21 = viewmats[vmBase + 9];
    float R22 = viewmats[vmBase + 10];
    float Jp[6];   // 2×3 row-major
    Jp[0] = dpi_00 * R00 + dpi_02 * R20;
    Jp[1] = dpi_00 * R01 + dpi_02 * R21;
    Jp[2] = dpi_00 * R02 + dpi_02 * R22;
    Jp[3] = dpi_11 * R10 + dpi_12 * R20;
    Jp[4] = dpi_11 * R11 + dpi_12 * R21;
    Jp[5] = dpi_11 * R12 + dpi_12 * R22;

    // J_cam (2×6): columns 0..2 rotation, 3..5 translation
    float Jc[12];   // 2×6 row-major
    Jc[0] = dpi_00 * 0.0   + dpi_02 * pcy;
    Jc[1] = dpi_00 * pcz   + dpi_02 * (-pcx);
    Jc[2] = dpi_00 * (-pcy) + dpi_02 * 0.0;
    Jc[3] = dpi_00;
    Jc[4] = 0.0;
    Jc[5] = dpi_02;
    Jc[6] = dpi_11 * (-pcz) + dpi_12 * pcy;
    Jc[7] = dpi_11 * 0.0    + dpi_12 * (-pcx);
    Jc[8] = dpi_11 * pcx    + dpi_12 * 0.0;
    Jc[9] = 0.0;
    Jc[10] = dpi_11;
    Jc[11] = dpi_12;

    float w = 1.0;
    float huberK = jac_params[0];
    if (huberK > 0.0) {
        float absR = sqrt(rU * rU + rV * rV);
        if (absR > huberK) {
            w = huberK / max(absR, 1e-9);
        }
    }

    Htp_valid[i] = 1;

    // H_tt[cam] += w · Jcᵀ Jc (6×6)
    int httBase = cam * 36;
    [unroll]
    for (int ii = 0; ii < 6; ii++) {
        [unroll]
        for (int jj = 0; jj < 6; jj++) {
            float v = w * (Jc[0 * 6 + ii] * Jc[0 * 6 + jj] + Jc[1 * 6 + ii] * Jc[1 * 6 + jj]);
            InterlockedAdd(Htt[httBase + ii * 6 + jj], v);
        }
    }

    // H_pp[pt] += w · Jpᵀ Jp (3×3)
    int hppBase = pt * 9;
    [unroll]
    for (int ii = 0; ii < 3; ii++) {
        [unroll]
        for (int jj = 0; jj < 3; jj++) {
            float v = w * (Jp[0 * 3 + ii] * Jp[0 * 3 + jj] + Jp[1 * 3 + ii] * Jp[1 * 3 + jj]);
            InterlockedAdd(Hpp[hppBase + ii * 3 + jj], v);
        }
    }

    // b_t[cam] += -w · Jcᵀ r (6)
    int btBase = cam * 6;
    [unroll]
    for (int ii = 0; ii < 6; ii++) {
        float v = -w * (Jc[0 * 6 + ii] * rU + Jc[1 * 6 + ii] * rV);
        InterlockedAdd(bt[btBase + ii], v);
    }

    // b_p[pt] += -w · Jpᵀ r (3)
    int bpBase = pt * 3;
    [unroll]
    for (int ii = 0; ii < 3; ii++) {
        float v = -w * (Jp[0 * 3 + ii] * rU + Jp[1 * 3 + ii] * rV);
        InterlockedAdd(bp[bpBase + ii], v);
    }

    // H_tp_block[obs] = w · Jcᵀ Jp (6×3). Each obs has a private slot, so no atomics needed
    int blkBase = i * 18;
    [unroll]
    for (int ii = 0; ii < 6; ii++) {
        [unroll]
        for (int jj = 0; jj < 3; jj++) {
            Htp_blocks[blkBase + ii * 3 + jj] =
                w * (Jc[0 * 6 + ii] * Jp[0 * 3 + jj] + Jc[1 * 6 + ii] * Jp[1 * 3 + jj]);
        }
    }
}
