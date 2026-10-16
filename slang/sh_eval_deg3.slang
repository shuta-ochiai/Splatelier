// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Forward / backward of foreground SH color (up to degree 1). The math is in gs_math.slang.
// degree_buf is 0 or 1 (the caller clamps). The deg3 in the file name is a leftover from the old
// spec. Coefficients come in 2 buffers, sh0 and shN, to avoid concatenation. shN is not read at
// degree 0.

import gs_math;

// ---- Forward ------------------------------------------------------------------

StructuredBuffer<float> dirs;
StructuredBuffer<float> sh0;            // [N, 3]   band 0 (DC)
StructuredBuffer<float> shN;            // [N, 9]   bands 1-3, shN[i*9 + (k-1)*3 + ch]
StructuredBuffer<int> degree_buf;
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
    uint sN = i * 9;
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

    float3 c = sh_eval_deg3(deg, dir, sh);
    colors[oBase + 0] = half(c.x);
    colors[oBase + 1] = half(c.y);
    colors[oBase + 2] = half(c.z);
}

// ---- Backward -----------------------------------------------------------------

StructuredBuffer<float> dirs_bwd;
StructuredBuffer<float> sh0_bwd;          // [N, 3]
StructuredBuffer<float> shN_bwd;          // [N, 9]
StructuredBuffer<int> degree_bwd;
StructuredBuffer<int> n_bwd;            // [1] : N
StructuredBuffer<float> cot_color;
StructuredBuffer<int> radii_bwd;       // [N, 2] (rx, ry)
RWStructuredBuffer<float> grad_dirs;
RWStructuredBuffer<float> grad_sh0;       // [N, 3]
RWStructuredBuffer<float> grad_shN;       // [N, 9]

[shader("compute")]
[numthreads(64, 1, 1)]
void sh_eval_deg3_bwd(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (int(i) >= n_bwd[0]) { return; }
    uint dBase = i * 3;
    uint s0 = i * 3;
    uint sN = i * 9;
    // Culled ones get zero gradient. Outputs are not zero-initialized, so write explicitly
    if (radii_bwd[i * 2 + 0] <= 0 || radii_bwd[i * 2 + 1] <= 0) {
        grad_dirs[dBase + 0] = 0.0;
        grad_dirs[dBase + 1] = 0.0;
        grad_dirs[dBase + 2] = 0.0;
        grad_sh0[s0 + 0] = 0.0; grad_sh0[s0 + 1] = 0.0; grad_sh0[s0 + 2] = 0.0;
        for (uint k = 0; k < 9; ++k) { grad_shN[sN + k] = 0.0; }
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

    float3 cot = float3(cot_color[i * 3 + 0], cot_color[i * 3 + 1], cot_color[i * 3 + 2]);

    ShEvalGrad g = sh_eval_deg3_grad(deg, dir, sh, cot);

    grad_dirs[dBase + 0] = g.dir.x;
    grad_dirs[dBase + 1] = g.dir.y;
    grad_dirs[dBase + 2] = g.dir.z;

    grad_sh0[s0 + 0] = g.sh.b0.x; grad_sh0[s0 + 1] = g.sh.b0.y; grad_sh0[s0 + 2] = g.sh.b0.z;
    // Autodiff returns 0 for inactive bands too, so write all 9 elements (Adam reads every element)
    grad_shN[sN + 0]  = g.sh.b1.x;  grad_shN[sN + 1]  = g.sh.b1.y;  grad_shN[sN + 2]  = g.sh.b1.z;
    grad_shN[sN + 3]  = g.sh.b2.x;  grad_shN[sN + 4]  = g.sh.b2.y;  grad_shN[sN + 5]  = g.sh.b2.z;
    grad_shN[sN + 6]  = g.sh.b3.x;  grad_shN[sN + 7]  = g.sh.b3.y;  grad_shN[sN + 8]  = g.sh.b3.z;
}
