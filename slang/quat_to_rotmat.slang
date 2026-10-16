// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Forward / backward of quat_to_rotmat. The math and the bwd_diff wrappers are in gs_math.slang.

import gs_math;

// ---- Forward ------------------------------------------------------------------

StructuredBuffer<float> quats;
RWStructuredBuffer<float> rotmats;

[shader("compute")]
[numthreads(1, 1, 1)]
void quat_to_rotmat_fwd(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    uint qBase = i * 4;
    uint rBase = i * 9;
    Mat3 R = quat_to_rotmat(
        quats[qBase + 0],
        quats[qBase + 1],
        quats[qBase + 2],
        quats[qBase + 3]
    );
    rotmats[rBase + 0] = R.r00;
    rotmats[rBase + 1] = R.r01;
    rotmats[rBase + 2] = R.r02;
    rotmats[rBase + 3] = R.r10;
    rotmats[rBase + 4] = R.r11;
    rotmats[rBase + 5] = R.r12;
    rotmats[rBase + 6] = R.r20;
    rotmats[rBase + 7] = R.r21;
    rotmats[rBase + 8] = R.r22;
}

// ---- Backward -----------------------------------------------------------------

StructuredBuffer<float> quats_bwd;
StructuredBuffer<float> cot_R;
RWStructuredBuffer<float> grad_quats;

[shader("compute")]
[numthreads(1, 1, 1)]
void quat_to_rotmat_bwd(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    uint qBase = i * 4;
    uint rBase = i * 9;

    Mat3 cot;
    cot.r00 = cot_R[rBase + 0]; cot.r01 = cot_R[rBase + 1]; cot.r02 = cot_R[rBase + 2];
    cot.r10 = cot_R[rBase + 3]; cot.r11 = cot_R[rBase + 4]; cot.r12 = cot_R[rBase + 5];
    cot.r20 = cot_R[rBase + 6]; cot.r21 = cot_R[rBase + 7]; cot.r22 = cot_R[rBase + 8];

    QuatGrad g = quat_to_rotmat_grad(
        quats_bwd[qBase + 0], quats_bwd[qBase + 1],
        quats_bwd[qBase + 2], quats_bwd[qBase + 3],
        cot
    );

    grad_quats[qBase + 0] = g.w;
    grad_quats[qBase + 1] = g.x;
    grad_quats[qBase + 2] = g.y;
    grad_quats[qBase + 3] = g.z;
}
