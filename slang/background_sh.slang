// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Forward / backward of the directional SH background (splatfacto-W style). The basis is computed
// per pixel in registers.
// The math must match BackgroundSH.swift:
//   dir = normalize(((x+0.5-cx)/fx, (y+0.5-cy)/fy, 1) · R), bg = Σ_b basis_b(dir)·coeffs[b]
//   comp = render + (1-alpha)·bg
// gs_math.sh_eval_deg3 includes +0.5 and clamping, so it cannot be used (bg is raw linear SH).
// Coefficient layout: coeffs[b*3 + c] (same as BackgroundSH's [bands, 3]). Do not pass
// StructuredBuffers to helpers (Slang lowers them to mutable pointers, which clash with MLX const
// inputs).

// Same SH normalization constants as gs_math.slang sh_eval_deg3.
static const float BG_C0   = 0.28209479177387814;
static const float BG_C1   = 0.4886025119029199;
static const float BG_C2_0 = 1.0925484305920792;
static const float BG_C2_1 = -1.0925484305920792;
static const float BG_C2_2 = 0.31539156525252005;
static const float BG_C2_3 = -1.0925484305920792;
static const float BG_C2_4 = 0.5462742152960396;
static const float BG_C3_0 = -0.5900435899266435;
static const float BG_C3_1 = 2.890611442640554;
static const float BG_C3_2 = -0.4570457994644658;
static const float BG_C3_3 = 0.3731763325901154;
static const float BG_C3_4 = -0.4570457994644658;
static const float BG_C3_5 = 1.445305721320277;
static const float BG_C3_6 = -0.5900435899266435;
// degree 4 (9 bands). Keep equal to BackgroundSH.swift.
static const float BG_C4_0 = 2.5033429417967046;
static const float BG_C4_1 = -1.7701307697799304;
static const float BG_C4_2 = 0.9461746957575601;
static const float BG_C4_3 = -0.6690465435572892;
static const float BG_C4_4 = 0.10578554691520431;
static const float BG_C4_5 = -0.6690465435572892;
static const float BG_C4_6 = 0.47308734787878004;
static const float BG_C4_7 = -1.7701307697799304;
static const float BG_C4_8 = 0.6258357354491761;

// SH basis of a normalized dir (up to degree 4 = 25 bands). Bands above deg are 0.
void bgBasis(float3 dir, int deg, out float bs[25])
{
    [unroll] for (uint k = 0; k < 25; ++k) { bs[k] = 0.0; }
    float x = dir.x, y = dir.y, z = dir.z;
    bs[0] = BG_C0;
    if (deg >= 1) {
        bs[1] = -BG_C1 * y;
        bs[2] =  BG_C1 * z;
        bs[3] = -BG_C1 * x;
    }
    if (deg >= 2) {
        float xx = x * x, yy = y * y, zz = z * z;
        bs[4] = BG_C2_0 * (x * y);
        bs[5] = BG_C2_1 * (y * z);
        bs[6] = BG_C2_2 * (2.0 * zz - xx - yy);
        bs[7] = BG_C2_3 * (x * z);
        bs[8] = BG_C2_4 * (xx - yy);
    }
    if (deg >= 3) {
        float xx = x * x, yy = y * y, zz = z * z;
        bs[9]  = BG_C3_0 * y * (3.0 * xx - yy);
        bs[10] = BG_C3_1 * (x * y) * z;
        bs[11] = BG_C3_2 * y * (4.0 * zz - xx - yy);
        bs[12] = BG_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
        bs[13] = BG_C3_4 * x * (4.0 * zz - xx - yy);
        bs[14] = BG_C3_5 * z * (xx - yy);
        bs[15] = BG_C3_6 * x * (xx - 3.0 * yy);
    }
    if (deg >= 4) {
        float xx = x * x, yy = y * y, zz = z * z;
        bs[16] = BG_C4_0 * (x * y) * (xx - yy);
        bs[17] = BG_C4_1 * (y * z) * (3.0 * xx - yy);
        bs[18] = BG_C4_2 * (x * y) * (7.0 * zz - 1.0);
        bs[19] = BG_C4_3 * (y * z) * (7.0 * zz - 3.0);
        bs[20] = BG_C4_4 * (35.0 * zz * zz - 30.0 * zz + 3.0);
        bs[21] = BG_C4_5 * (x * z) * (7.0 * zz - 3.0);
        bs[22] = BG_C4_6 * (xx - yy) * (7.0 * zz - 1.0);
        bs[23] = BG_C4_7 * (x * z) * (xx - 3.0 * yy);
        bs[24] = BG_C4_8 * (xx * (xx - 3.0 * yy) - yy * (3.0 * xx - yy));
    }
}

// ---- Forward ------------------------------------------------------------------

StructuredBuffer<float> render;        // [P, 3]  foreground render (black background)
StructuredBuffer<float> render_alpha;  // [P]      foreground alpha
StructuredBuffer<float> coeffs_fwd;    // [(deg+1)² * 3] bg SH coefficients (band-major)
StructuredBuffer<float> viewmat_fwd;   // [16]     row-major 4×4
StructuredBuffer<float> intr_fwd;      // [4]      fx, fy, cx, cy
StructuredBuffer<int> dims_fwd;        // [3]      W, H, degree
RWStructuredBuffer<float> composited;  // [P, 3]   out: render + (1-alpha)·bg
RWStructuredBuffer<float> bg_out;      // [P, 3]   out: bg (for vAlpha in backward)

[shader("compute")]
[numthreads(64, 1, 1)]
void background_sh_fwd(uint3 tid : SV_DispatchThreadID)
{
    uint W = uint(dims_fwd[0]);
    uint H = uint(dims_fwd[1]);
    int  deg = dims_fwd[2];
    uint p = tid.x;
    uint P = W * H;
    if (p >= P) { return; }

    uint x = p % W;
    uint y = p / W;

    // Match BackgroundSH.rayDirections
    float fx = intr_fwd[0], fy = intr_fwd[1], cx = intr_fwd[2], cy = intr_fwd[3];
    float cxd = (float(x) + 0.5 - cx) / fx;
    float cyd = (float(y) + 0.5 - cy) / fy;
    float3 w3 = float3(
        cxd * viewmat_fwd[0] + cyd * viewmat_fwd[4] + viewmat_fwd[8],
        cxd * viewmat_fwd[1] + cyd * viewmat_fwd[5] + viewmat_fwd[9],
        cxd * viewmat_fwd[2] + cyd * viewmat_fwd[6] + viewmat_fwd[10]);
    float3 dir = w3 * rsqrt(max(dot(w3, w3), 1e-20));

    float bs[25];
    bgBasis(dir, deg, bs);

    // The coefficient buffer only holds (deg+1)² bands, so stop at nb (avoids OOB)
    uint nb = uint((deg + 1) * (deg + 1));
    if (nb > 25u) { nb = 25u; }
    float3 bg = float3(0.0, 0.0, 0.0);
    for (uint b = 0; b < nb; ++b) {
        bg += bs[b] * float3(coeffs_fwd[b * 3 + 0], coeffs_fwd[b * 3 + 1], coeffs_fwd[b * 3 + 2]);
    }

    float oma = 1.0 - render_alpha[p];
    uint o = p * 3;
    composited[o + 0] = render[o + 0] + oma * bg.x;
    composited[o + 1] = render[o + 1] + oma * bg.y;
    composited[o + 2] = render[o + 2] + oma * bg.z;
    bg_out[o + 0] = bg.x;
    bg_out[o + 1] = bg.y;
    bg_out[o + 2] = bg.z;
}

// ---- Backward (coefficient gradients only) ------------------------------------
// dL/dcoeff_{b,c} = Σ_px basis_b · vRender_c · (1-alpha). Reduced within each threadgroup and
// written to partials; the sum across groups is done on the MLX side. dL/dalpha is computed on the
// MLX side from bg_out.

StructuredBuffer<float> cot_color_bwd;    // [P, 3]  vRender = dL/dcomposited
StructuredBuffer<float> render_alpha_bwd; // [P]
StructuredBuffer<float> viewmat_bwd;      // [16]
StructuredBuffer<float> intr_bwd;         // [4]
StructuredBuffer<int> dims_bwd;           // [3]  W, H, degree
RWStructuredBuffer<float> partials;       // [numGroups, 75]  per-threadgroup partial sums (25 bands × 3)

// Per-simdgroup partial sums (64 threads, at most 4 simdgroups when laneCount >= 16).
groupshared float gsm[4 * 75];

[shader("compute")]
[numthreads(64, 1, 1)]
void background_sh_bwd(uint3 tid : SV_DispatchThreadID, uint lidx : SV_GroupIndex)
{
    uint W = uint(dims_bwd[0]);
    uint H = uint(dims_bwd[1]);
    int  deg = dims_bwd[2];
    uint P = W * H;
    uint p = tid.x;
    bool active = (p < P);

    float gsh[75];
    [unroll] for (uint k = 0; k < 75; ++k) { gsh[k] = 0.0; }

    if (active) {
        uint x = p % W;
        uint y = p / W;

        float fx = intr_bwd[0], fy = intr_bwd[1], cx = intr_bwd[2], cy = intr_bwd[3];
        float cxd = (float(x) + 0.5 - cx) / fx;
        float cyd = (float(y) + 0.5 - cy) / fy;
        float3 w3 = float3(
            cxd * viewmat_bwd[0] + cyd * viewmat_bwd[4] + viewmat_bwd[8],
            cxd * viewmat_bwd[1] + cyd * viewmat_bwd[5] + viewmat_bwd[9],
            cxd * viewmat_bwd[2] + cyd * viewmat_bwd[6] + viewmat_bwd[10]);
        float3 dir = w3 * rsqrt(max(dot(w3, w3), 1e-20));

        float bs[25];
        bgBasis(dir, deg, bs);

        uint o = p * 3;
        float3 vR = float3(cot_color_bwd[o + 0], cot_color_bwd[o + 1], cot_color_bwd[o + 2]);
        float oma = 1.0 - render_alpha_bwd[p];
        float3 cot = vR * oma;                       // dL/dbg

        // Bands above deg get bs = 0. The MLX side slices partials to bands*3.
        [unroll] for (uint b = 0; b < 25; ++b) {
            gsh[b * 3 + 0] = bs[b] * cot.x;
            gsh[b * 3 + 1] = bs[b] * cot.y;
            gsh[b * 3 + 2] = bs[b] * cot.z;
        }
    }

    // pass 1: reduce within the simdgroup
    uint laneCount = WaveGetLaneCount();
    uint simdId = lidx / laneCount;
    uint numSimd = (64u + laneCount - 1u) / laneCount;
    [unroll] for (uint k = 0; k < 75; ++k) {
        float ws = WaveActiveSum(gsh[k]);
        if (WaveGetLaneIndex() == 0u) { gsm[simdId * 75u + k] = ws; }
    }
    GroupMemoryBarrierWithGroupSync();

    // pass 2: thread 0 sums the simdgroup partials
    if (lidx == 0u) {
        uint group = tid.x / 64u;     // assumes numthreads(64)
        for (uint k = 0; k < 75; ++k) {
            float t = 0.0;
            for (uint s = 0; s < numSimd; ++s) { t += gsm[s * 75u + k]; }
            partials[group * 75u + k] = t;
        }
    }
}
