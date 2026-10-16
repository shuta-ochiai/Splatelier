// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Forward / backward of persp_proj. The math is in gs_math.slang.

import gs_math;

// ---- Forward ------------------------------------------------------------------

StructuredBuffer<float> means3d_cam;
StructuredBuffer<float> cov3ds_cam;
StructuredBuffer<float> intrinsics;
StructuredBuffer<float> image_dims;     // [imageWidth, imageHeight]
RWStructuredBuffer<float> means2d;
RWStructuredBuffer<float> cov2ds;

[shader("compute")]
[numthreads(1, 1, 1)]
void persp_proj_fwd(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    uint mBase = i * 3;
    uint cBase = i * 6;
    uint pBase = i * 2;
    uint qBase = i * 3;

    SymMat3 cov3d;
    cov3d.c00 = cov3ds_cam[cBase + 0];
    cov3d.c01 = cov3ds_cam[cBase + 1];
    cov3d.c02 = cov3ds_cam[cBase + 2];
    cov3d.c11 = cov3ds_cam[cBase + 3];
    cov3d.c12 = cov3ds_cam[cBase + 4];
    cov3d.c22 = cov3ds_cam[cBase + 5];

    ProjResult r = persp_proj(
        means3d_cam[mBase + 0],
        means3d_cam[mBase + 1],
        means3d_cam[mBase + 2],
        cov3d,
        intrinsics[0], intrinsics[1],
        intrinsics[2], intrinsics[3],
        image_dims[0], image_dims[1]
    );

    means2d[pBase + 0] = r.px;
    means2d[pBase + 1] = r.py;
    cov2ds[qBase + 0] = r.cov2d.c00;
    cov2ds[qBase + 1] = r.cov2d.c01;
    cov2ds[qBase + 2] = r.cov2d.c11;
}

// ---- Backward -----------------------------------------------------------------

StructuredBuffer<float> means3d_cam_bwd;
StructuredBuffer<float> cov3ds_cam_bwd;
StructuredBuffer<float> intrinsics_bwd;
StructuredBuffer<float> image_dims_bwd;
StructuredBuffer<float> cot_mean2d;
StructuredBuffer<float> cot_cov2d;
RWStructuredBuffer<float> grad_means3d;
RWStructuredBuffer<float> grad_cov3d;

[shader("compute")]
[numthreads(1, 1, 1)]
void persp_proj_bwd(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    uint mBase = i * 3;
    uint cBase = i * 6;
    uint pBase = i * 2;
    uint qBase = i * 3;

    SymMat3 cov3d;
    cov3d.c00 = cov3ds_cam_bwd[cBase + 0];
    cov3d.c01 = cov3ds_cam_bwd[cBase + 1];
    cov3d.c02 = cov3ds_cam_bwd[cBase + 2];
    cov3d.c11 = cov3ds_cam_bwd[cBase + 3];
    cov3d.c12 = cov3ds_cam_bwd[cBase + 4];
    cov3d.c22 = cov3ds_cam_bwd[cBase + 5];

    ProjResult cot;
    cot.px = cot_mean2d[pBase + 0];
    cot.py = cot_mean2d[pBase + 1];
    cot.cov2d.c00 = cot_cov2d[qBase + 0];
    cot.cov2d.c01 = cot_cov2d[qBase + 1];
    cot.cov2d.c11 = cot_cov2d[qBase + 2];

    PerspProjGrad g = persp_proj_grad(
        means3d_cam_bwd[mBase + 0],
        means3d_cam_bwd[mBase + 1],
        means3d_cam_bwd[mBase + 2],
        cov3d,
        intrinsics_bwd[0], intrinsics_bwd[1],
        intrinsics_bwd[2], intrinsics_bwd[3],
        image_dims_bwd[0], image_dims_bwd[1],
        cot
    );

    grad_means3d[mBase + 0] = g.mx;
    grad_means3d[mBase + 1] = g.my;
    grad_means3d[mBase + 2] = g.mz;
    grad_cov3d[cBase + 0] = g.cov3d.c00;
    grad_cov3d[cBase + 1] = g.cov3d.c01;
    grad_cov3d[cBase + 2] = g.cov3d.c02;
    grad_cov3d[cBase + 3] = g.cov3d.c11;
    grad_cov3d[cBase + 4] = g.cov3d.c12;
    grad_cov3d[cBase + 5] = g.cov3d.c22;
}
