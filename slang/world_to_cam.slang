// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Forward / backward of world_to_cam. The math is in gs_math.slang.

import gs_math;

// ---- Forward ------------------------------------------------------------------

StructuredBuffer<float> means3d_world;
StructuredBuffer<float> cov3ds_world;
StructuredBuffer<float> viewmat;
RWStructuredBuffer<float> means3d_cam;
RWStructuredBuffer<float> cov3ds_cam;

[shader("compute")]
[numthreads(1, 1, 1)]
void world_to_cam_fwd(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    uint mBase = i * 3;
    uint cBase = i * 6;

    SymMat3 covW;
    covW.c00 = cov3ds_world[cBase + 0];
    covW.c01 = cov3ds_world[cBase + 1];
    covW.c02 = cov3ds_world[cBase + 2];
    covW.c11 = cov3ds_world[cBase + 3];
    covW.c12 = cov3ds_world[cBase + 4];
    covW.c22 = cov3ds_world[cBase + 5];

    CamPose r = world_to_cam(
        means3d_world[mBase + 0],
        means3d_world[mBase + 1],
        means3d_world[mBase + 2],
        covW,
        viewmat[0], viewmat[1], viewmat[2],
        viewmat[4], viewmat[5], viewmat[6],
        viewmat[8], viewmat[9], viewmat[10],
        viewmat[3], viewmat[7], viewmat[11]
    );

    means3d_cam[mBase + 0] = r.mx;
    means3d_cam[mBase + 1] = r.my;
    means3d_cam[mBase + 2] = r.mz;
    cov3ds_cam[cBase + 0] = r.cov.c00;
    cov3ds_cam[cBase + 1] = r.cov.c01;
    cov3ds_cam[cBase + 2] = r.cov.c02;
    cov3ds_cam[cBase + 3] = r.cov.c11;
    cov3ds_cam[cBase + 4] = r.cov.c12;
    cov3ds_cam[cBase + 5] = r.cov.c22;
}

// ---- Backward -----------------------------------------------------------------

StructuredBuffer<float> means3d_world_bwd;
StructuredBuffer<float> cov3ds_world_bwd;
StructuredBuffer<float> viewmat_bwd;
StructuredBuffer<float> cot_means3d_cam;
StructuredBuffer<float> cot_cov3ds_cam;
RWStructuredBuffer<float> grad_means3d_world;
RWStructuredBuffer<float> grad_cov3ds_world;

[shader("compute")]
[numthreads(1, 1, 1)]
void world_to_cam_bwd(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    uint mBase = i * 3;
    uint cBase = i * 6;

    SymMat3 covW;
    covW.c00 = cov3ds_world_bwd[cBase + 0];
    covW.c01 = cov3ds_world_bwd[cBase + 1];
    covW.c02 = cov3ds_world_bwd[cBase + 2];
    covW.c11 = cov3ds_world_bwd[cBase + 3];
    covW.c12 = cov3ds_world_bwd[cBase + 4];
    covW.c22 = cov3ds_world_bwd[cBase + 5];

    CamPose cot;
    cot.mx = cot_means3d_cam[mBase + 0];
    cot.my = cot_means3d_cam[mBase + 1];
    cot.mz = cot_means3d_cam[mBase + 2];
    cot.cov.c00 = cot_cov3ds_cam[cBase + 0];
    cot.cov.c01 = cot_cov3ds_cam[cBase + 1];
    cot.cov.c02 = cot_cov3ds_cam[cBase + 2];
    cot.cov.c11 = cot_cov3ds_cam[cBase + 3];
    cot.cov.c12 = cot_cov3ds_cam[cBase + 4];
    cot.cov.c22 = cot_cov3ds_cam[cBase + 5];

    WorldToCamGrad g = world_to_cam_grad(
        means3d_world_bwd[mBase + 0],
        means3d_world_bwd[mBase + 1],
        means3d_world_bwd[mBase + 2],
        covW,
        viewmat_bwd[0], viewmat_bwd[1], viewmat_bwd[2],
        viewmat_bwd[4], viewmat_bwd[5], viewmat_bwd[6],
        viewmat_bwd[8], viewmat_bwd[9], viewmat_bwd[10],
        viewmat_bwd[3], viewmat_bwd[7], viewmat_bwd[11],
        cot
    );

    grad_means3d_world[mBase + 0] = g.wx;
    grad_means3d_world[mBase + 1] = g.wy;
    grad_means3d_world[mBase + 2] = g.wz;
    grad_cov3ds_world[cBase + 0] = g.covW.c00;
    grad_cov3ds_world[cBase + 1] = g.covW.c01;
    grad_cov3ds_world[cBase + 2] = g.covW.c02;
    grad_cov3ds_world[cBase + 3] = g.covW.c11;
    grad_cov3ds_world[cBase + 4] = g.covW.c12;
    grad_cov3ds_world[cBase + 5] = g.covW.c22;
}
