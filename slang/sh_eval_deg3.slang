// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Forward / backward of foreground SH color (up to degree 3). The math is in gs_math.slang.
// degree_buf is [active degree (0-3), rest bands per Gaussian in shN (3, 8 or 15)]; the caller
// clamps the degree to what shN holds. Coefficients come in 2 buffers, sh0 and shN, to avoid
// concatenation. shN is not read at degree 0.

import gs_math;

// ---- Forward ------------------------------------------------------------------

StructuredBuffer<float> dirs;
StructuredBuffer<float> sh0;            // [N, 3]   band 0 (DC)
StructuredBuffer<float> shN;            // [N, R*3] rest bands, shN[i*R*3 + (k-1)*3 + ch]
StructuredBuffer<int> degree_buf;       // [2] : (degree, R)
StructuredBuffer<int> n_buf;            // [1] : N
StructuredBuffer<int> radii;           // [N, 2] (rx, ry). 0 means culled in this view
// Evaluation is fp32, only storage is half. The caller (GaussianRenderer) converts back to fp32 and
// passes it to rasterize.
RWStructuredBuffer<half> colors;

[shader("compute")]
[numthreads(64, 1, 1)]
void sh_eval_deg3_fwd(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (int(i) >= n_buf[0]) { return; }
    uint oBase = i * 3;
    // Culled ones (radii = 0) are not rasterized, so skip evaluation and write 0
    if (radii[i * 2 + 0] <= 0 || radii[i * 2 + 1] <= 0) {
        colors[oBase + 0] = half(0.0);
        colors[oBase + 1] = half(0.0);
        colors[oBase + 2] = half(0.0);
        return;
    }
    uint dBase = i * 3;
    uint s0 = i * 3;
    uint rest = uint(degree_buf[1]);
    uint sN = i * rest * 3;
    int deg = degree_buf[0];

    float3 dir = float3(dirs[dBase + 0], dirs[dBase + 1], dirs[dBase + 2]);

    SHBands16 sh;
    sh.b0  = float3(sh0[s0 + 0], sh0[s0 + 1], sh0[s0 + 2]);
    // Inactive bands are 0 (no contribution)
    sh.b1 = float3(0.0); sh.b2 = float3(0.0); sh.b3 = float3(0.0);
    sh.b4 = float3(0.0); sh.b5 = float3(0.0); sh.b6 = float3(0.0);
    sh.b7 = float3(0.0); sh.b8 = float3(0.0);
    sh.b9 = float3(0.0); sh.b10 = float3(0.0); sh.b11 = float3(0.0); sh.b12 = float3(0.0);
    sh.b13 = float3(0.0); sh.b14 = float3(0.0); sh.b15 = float3(0.0);
    if (deg >= 1) {
        sh.b1 = float3(shN[sN + 0],  shN[sN + 1],  shN[sN + 2]);
        sh.b2 = float3(shN[sN + 3],  shN[sN + 4],  shN[sN + 5]);
        sh.b3 = float3(shN[sN + 6],  shN[sN + 7],  shN[sN + 8]);
    }
    if (deg >= 2) {
        sh.b4 = float3(shN[sN + 9],  shN[sN + 10], shN[sN + 11]);
        sh.b5 = float3(shN[sN + 12], shN[sN + 13], shN[sN + 14]);
        sh.b6 = float3(shN[sN + 15], shN[sN + 16], shN[sN + 17]);
        sh.b7 = float3(shN[sN + 18], shN[sN + 19], shN[sN + 20]);
        sh.b8 = float3(shN[sN + 21], shN[sN + 22], shN[sN + 23]);
    }
    if (deg >= 3) {
        sh.b9  = float3(shN[sN + 24], shN[sN + 25], shN[sN + 26]);
        sh.b10 = float3(shN[sN + 27], shN[sN + 28], shN[sN + 29]);
        sh.b11 = float3(shN[sN + 30], shN[sN + 31], shN[sN + 32]);
        sh.b12 = float3(shN[sN + 33], shN[sN + 34], shN[sN + 35]);
        sh.b13 = float3(shN[sN + 36], shN[sN + 37], shN[sN + 38]);
        sh.b14 = float3(shN[sN + 39], shN[sN + 40], shN[sN + 41]);
        sh.b15 = float3(shN[sN + 42], shN[sN + 43], shN[sN + 44]);
    }

    float3 c = sh_eval_deg3(deg, dir, sh);
    colors[oBase + 0] = half(c.x);
    colors[oBase + 1] = half(c.y);
    colors[oBase + 2] = half(c.z);
}

// ---- Backward -----------------------------------------------------------------

StructuredBuffer<float> dirs_bwd;
StructuredBuffer<float> sh0_bwd;          // [N, 3]
StructuredBuffer<float> shN_bwd;          // [N, R*3]
StructuredBuffer<int> degree_bwd;       // [2] : (degree, R)
StructuredBuffer<int> n_bwd;            // [1] : N
StructuredBuffer<float> cot_color;
StructuredBuffer<int> radii_bwd;       // [N, 2] (rx, ry)
RWStructuredBuffer<float> grad_dirs;
RWStructuredBuffer<float> grad_sh0;       // [N, 3]
RWStructuredBuffer<float> grad_shN;       // [N, R*3]

[shader("compute")]
[numthreads(64, 1, 1)]
void sh_eval_deg3_bwd(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (int(i) >= n_bwd[0]) { return; }
    uint dBase = i * 3;
    uint s0 = i * 3;
    uint rest = uint(degree_bwd[1]);
    uint sN = i * rest * 3;
    // Culled ones get zero gradient. Outputs are not zero-initialized, so write explicitly
    if (radii_bwd[i * 2 + 0] <= 0 || radii_bwd[i * 2 + 1] <= 0) {
        grad_dirs[dBase + 0] = 0.0;
        grad_dirs[dBase + 1] = 0.0;
        grad_dirs[dBase + 2] = 0.0;
        grad_sh0[s0 + 0] = 0.0; grad_sh0[s0 + 1] = 0.0; grad_sh0[s0 + 2] = 0.0;
        for (uint k = 0; k < rest * 3; ++k) { grad_shN[sN + k] = 0.0; }
        return;
    }
    int deg = degree_bwd[0];

    float3 dir = float3(dirs_bwd[dBase + 0], dirs_bwd[dBase + 1], dirs_bwd[dBase + 2]);

    SHBands16 sh;
    sh.b0  = float3(sh0_bwd[s0 + 0], sh0_bwd[s0 + 1], sh0_bwd[s0 + 2]);
    sh.b1 = float3(0.0); sh.b2 = float3(0.0); sh.b3 = float3(0.0);
    sh.b4 = float3(0.0); sh.b5 = float3(0.0); sh.b6 = float3(0.0);
    sh.b7 = float3(0.0); sh.b8 = float3(0.0);
    sh.b9 = float3(0.0); sh.b10 = float3(0.0); sh.b11 = float3(0.0); sh.b12 = float3(0.0);
    sh.b13 = float3(0.0); sh.b14 = float3(0.0); sh.b15 = float3(0.0);
    if (deg >= 1) {
        sh.b1 = float3(shN_bwd[sN + 0],  shN_bwd[sN + 1],  shN_bwd[sN + 2]);
        sh.b2 = float3(shN_bwd[sN + 3],  shN_bwd[sN + 4],  shN_bwd[sN + 5]);
        sh.b3 = float3(shN_bwd[sN + 6],  shN_bwd[sN + 7],  shN_bwd[sN + 8]);
    }
    if (deg >= 2) {
        sh.b4 = float3(shN_bwd[sN + 9],  shN_bwd[sN + 10], shN_bwd[sN + 11]);
        sh.b5 = float3(shN_bwd[sN + 12], shN_bwd[sN + 13], shN_bwd[sN + 14]);
        sh.b6 = float3(shN_bwd[sN + 15], shN_bwd[sN + 16], shN_bwd[sN + 17]);
        sh.b7 = float3(shN_bwd[sN + 18], shN_bwd[sN + 19], shN_bwd[sN + 20]);
        sh.b8 = float3(shN_bwd[sN + 21], shN_bwd[sN + 22], shN_bwd[sN + 23]);
    }
    if (deg >= 3) {
        sh.b9  = float3(shN_bwd[sN + 24], shN_bwd[sN + 25], shN_bwd[sN + 26]);
        sh.b10 = float3(shN_bwd[sN + 27], shN_bwd[sN + 28], shN_bwd[sN + 29]);
        sh.b11 = float3(shN_bwd[sN + 30], shN_bwd[sN + 31], shN_bwd[sN + 32]);
        sh.b12 = float3(shN_bwd[sN + 33], shN_bwd[sN + 34], shN_bwd[sN + 35]);
        sh.b13 = float3(shN_bwd[sN + 36], shN_bwd[sN + 37], shN_bwd[sN + 38]);
        sh.b14 = float3(shN_bwd[sN + 39], shN_bwd[sN + 40], shN_bwd[sN + 41]);
        sh.b15 = float3(shN_bwd[sN + 42], shN_bwd[sN + 43], shN_bwd[sN + 44]);
    }

    float3 cot = float3(cot_color[i * 3 + 0], cot_color[i * 3 + 1], cot_color[i * 3 + 2]);

    ShEvalGrad g = sh_eval_deg3_grad(deg, dir, sh, cot);

    grad_dirs[dBase + 0] = g.dir.x;
    grad_dirs[dBase + 1] = g.dir.y;
    grad_dirs[dBase + 2] = g.dir.z;

    grad_sh0[s0 + 0] = g.sh.b0.x; grad_sh0[s0 + 1] = g.sh.b0.y; grad_sh0[s0 + 2] = g.sh.b0.z;
    // Autodiff returns 0 for inactive bands too, so write all R*3 elements (Adam reads every
    // element)
    grad_shN[sN + 0]  = g.sh.b1.x;  grad_shN[sN + 1]  = g.sh.b1.y;  grad_shN[sN + 2]  = g.sh.b1.z;
    grad_shN[sN + 3]  = g.sh.b2.x;  grad_shN[sN + 4]  = g.sh.b2.y;  grad_shN[sN + 5]  = g.sh.b2.z;
    grad_shN[sN + 6]  = g.sh.b3.x;  grad_shN[sN + 7]  = g.sh.b3.y;  grad_shN[sN + 8]  = g.sh.b3.z;
    if (rest >= 8) {
        grad_shN[sN + 9]  = g.sh.b4.x;  grad_shN[sN + 10] = g.sh.b4.y;  grad_shN[sN + 11] = g.sh.b4.z;
        grad_shN[sN + 12] = g.sh.b5.x;  grad_shN[sN + 13] = g.sh.b5.y;  grad_shN[sN + 14] = g.sh.b5.z;
        grad_shN[sN + 15] = g.sh.b6.x;  grad_shN[sN + 16] = g.sh.b6.y;  grad_shN[sN + 17] = g.sh.b6.z;
        grad_shN[sN + 18] = g.sh.b7.x;  grad_shN[sN + 19] = g.sh.b7.y;  grad_shN[sN + 20] = g.sh.b7.z;
        grad_shN[sN + 21] = g.sh.b8.x;  grad_shN[sN + 22] = g.sh.b8.y;  grad_shN[sN + 23] = g.sh.b8.z;
    }
    if (rest >= 15) {
        grad_shN[sN + 24] = g.sh.b9.x;  grad_shN[sN + 25] = g.sh.b9.y;  grad_shN[sN + 26] = g.sh.b9.z;
        grad_shN[sN + 27] = g.sh.b10.x; grad_shN[sN + 28] = g.sh.b10.y; grad_shN[sN + 29] = g.sh.b10.z;
        grad_shN[sN + 30] = g.sh.b11.x; grad_shN[sN + 31] = g.sh.b11.y; grad_shN[sN + 32] = g.sh.b11.z;
        grad_shN[sN + 33] = g.sh.b12.x; grad_shN[sN + 34] = g.sh.b12.y; grad_shN[sN + 35] = g.sh.b12.z;
        grad_shN[sN + 36] = g.sh.b13.x; grad_shN[sN + 37] = g.sh.b13.y; grad_shN[sN + 38] = g.sh.b13.z;
        grad_shN[sN + 39] = g.sh.b14.x; grad_shN[sN + 40] = g.sh.b14.y; grad_shN[sN + 41] = g.sh.b14.z;
        grad_shN[sN + 42] = g.sh.b15.x; grad_shN[sN + 43] = g.sh.b15.y; grad_shN[sN + 44] = g.sh.b15.z;
    }
}
