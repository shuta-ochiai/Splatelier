// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Forward / backward of quat_scale_to_cov3d. The math is in gs_math.slang.

import gs_math;

// ---- Forward ------------------------------------------------------------------

StructuredBuffer<float> quats;
StructuredBuffer<float> scales;
RWStructuredBuffer<float> cov3ds;

[shader("compute")]
[numthreads(1, 1, 1)]
void quat_scale_to_cov3d_fwd(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    uint qBase = i * 4;
    uint sBase = i * 3;
    uint cBase = i * 6;

    SymMat3 C = quat_scale_to_cov3d(
        quats[qBase + 0], quats[qBase + 1], quats[qBase + 2], quats[qBase + 3],
        scales[sBase + 0], scales[sBase + 1], scales[sBase + 2]
    );

    cov3ds[cBase + 0] = C.c00;
    cov3ds[cBase + 1] = C.c01;
    cov3ds[cBase + 2] = C.c02;
    cov3ds[cBase + 3] = C.c11;
    cov3ds[cBase + 4] = C.c12;
    cov3ds[cBase + 5] = C.c22;
}

// ---- Backward -----------------------------------------------------------------

StructuredBuffer<float> quats_bwd;
StructuredBuffer<float> scales_bwd;
StructuredBuffer<float> cot_cov;
RWStructuredBuffer<float> grad_quats;
RWStructuredBuffer<float> grad_scales;

[shader("compute")]
[numthreads(1, 1, 1)]
void quat_scale_to_cov3d_bwd(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    uint qBase = i * 4;
    uint sBase = i * 3;
    uint cBase = i * 6;

    SymMat3 cot;
    cot.c00 = cot_cov[cBase + 0];
    cot.c01 = cot_cov[cBase + 1];
    cot.c02 = cot_cov[cBase + 2];
    cot.c11 = cot_cov[cBase + 3];
    cot.c12 = cot_cov[cBase + 4];
    cot.c22 = cot_cov[cBase + 5];

    QuatScaleGrad g = quat_scale_to_cov3d_grad(
        quats_bwd[qBase + 0], quats_bwd[qBase + 1],
        quats_bwd[qBase + 2], quats_bwd[qBase + 3],
        scales_bwd[sBase + 0], scales_bwd[sBase + 1], scales_bwd[sBase + 2],
        cot
    );

    grad_quats[qBase + 0] = g.qw;
    grad_quats[qBase + 1] = g.qx;
    grad_quats[qBase + 2] = g.qy;
    grad_quats[qBase + 3] = g.qz;
    grad_scales[sBase + 0] = g.sx;
    grad_scales[sBase + 1] = g.sy;
    grad_scales[sBase + 2] = g.sz;
}
