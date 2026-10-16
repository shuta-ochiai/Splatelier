// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Projection of world-space (mean3d, quat, scale) → (means2d, conic, depth, radii).
// The computation itself is project_to_screen in gs_math.slang.

import gs_math;

// ---- Forward ------------------------------------------------------------------

StructuredBuffer<float> means3d_world;  // [N, 3]
StructuredBuffer<float> quats;          // [N, 4] (w, x, y, z)
StructuredBuffer<float> scales;         // [N, 3]
StructuredBuffer<float> viewmat;        // [16] row-major 4x4
StructuredBuffer<float> intrinsics;     // [4] (fx, fy, cx, cy)
StructuredBuffer<float> image_dims;     // [imageWidth, imageHeight]
StructuredBuffer<float> opacities;      // [N] after sigmoid
StructuredBuffer<int>   n_buf;          // [1] : N
RWStructuredBuffer<float> means2d;      // [N, 2]
RWStructuredBuffer<float> conics;       // [N, 3] upper triangle (a, b, c) of Σ_2d⁻¹
RWStructuredBuffer<float> depths;       // [N] camera-space z
RWStructuredBuffer<int> radii;          // [N, 2] (radius_x, radius_y), 0 if culled

[shader("compute")]
[numthreads(64, 1, 1)]
void project_to_screen_fwd(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (int(i) >= n_buf[0]) { return; }
    uint wBase = i * 3;
    uint qBase = i * 4;
    uint sBase = i * 3;
    uint mBase = i * 2;
    uint cBase = i * 3;

    ProjectedScreen out = project_to_screen(
        means3d_world[wBase + 0], means3d_world[wBase + 1], means3d_world[wBase + 2],
        quats[qBase + 0], quats[qBase + 1], quats[qBase + 2], quats[qBase + 3],
        scales[sBase + 0], scales[sBase + 1], scales[sBase + 2],
        viewmat[0], viewmat[1], viewmat[2],
        viewmat[4], viewmat[5], viewmat[6],
        viewmat[8], viewmat[9], viewmat[10],
        viewmat[3], viewmat[7], viewmat[11],
        intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3],
        image_dims[0], image_dims[1]
    );

    means2d[mBase + 0] = out.px;
    means2d[mBase + 1] = out.py;
    conics[cBase + 0] = out.conic_a;
    conics[cBase + 1] = out.conic_b;
    conics[cBase + 2] = out.conic_c;
    depths[i] = out.depth;

    // Per-axis radii (same as gsplat ProjectionEWA3DGSFused.cu):
    //  opacity < 1/255 is culled (also dropped from binning / bwd so near-dead Gaussians get no
    //  gradient noise).
    //  extend = min(3.33, sqrt(2·log(opac/α_th))) shrinks the bbox for lower opacity.
    const float ALPHA_THRESHOLD = 1.0 / 255.0;
    const float GAUSSIAN_EXTEND = 3.33;
    uint rBase = i * 2;
    float opac = opacities[i];
    if (opac < ALPHA_THRESHOLD) {
        radii[rBase + 0] = 0;
        radii[rBase + 1] = 0;
    } else {
        float a = out.cov2d_a;
        float c = out.cov2d_c;
        float extend = min(GAUSSIAN_EXTEND, sqrt(2.0 * log(opac / ALPHA_THRESHOLD)));
        int rx = int(ceil(extend * sqrt(max(0.0, a))));
        int ry = int(ceil(extend * sqrt(max(0.0, c))));
        // Radius over 4x the long side of the screen = degenerate Gaussian near the near plane. It
        // covers every tile while contributing almost nothing, so cull it for this view only.
        // Normal close Gaussians under the scale cap stay below this.
        float maxR = 4.0 * max(float(image_dims[0]), float(image_dims[1]));
        if (float(rx) > maxR || float(ry) > maxR) { rx = 0; ry = 0; }
        radii[rBase + 0] = rx;
        radii[rBase + 1] = ry;
    }
}

// ---- Backward -----------------------------------------------------------------
// Gradients of means3d / quats / scales from the means2d / conics cotangents.

StructuredBuffer<float> means3d_world_bwd;
StructuredBuffer<float> quats_bwd;
StructuredBuffer<float> scales_bwd;
StructuredBuffer<float> viewmat_bwd;
StructuredBuffer<float> intrinsics_bwd;
StructuredBuffer<float> image_dims_bwd;
StructuredBuffer<int>   n_bwd;          // [1] : N
StructuredBuffer<float> cot_means2d;
StructuredBuffer<float> cot_conics;
RWStructuredBuffer<float> grad_means3d_world;
RWStructuredBuffer<float> grad_quats;
RWStructuredBuffer<float> grad_scales;

[shader("compute")]
[numthreads(64, 1, 1)]
void project_to_screen_bwd(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (int(i) >= n_bwd[0]) { return; }
    uint wBase = i * 3;
    uint qBase = i * 4;
    uint sBase = i * 3;
    uint mBase = i * 2;
    uint cBase = i * 3;

    ProjectedScreen cot;
    cot.px = cot_means2d[mBase + 0];
    cot.py = cot_means2d[mBase + 1];
    cot.conic_a = cot_conics[cBase + 0];
    cot.conic_b = cot_conics[cBase + 1];
    cot.conic_c = cot_conics[cBase + 2];
    // depth / cov2d are not differentiated
    cot.depth = 0.0;
    cot.cov2d_a = 0.0;
    cot.cov2d_b = 0.0;
    cot.cov2d_c = 0.0;

    ProjectScreenGrad g = project_to_screen_grad(
        means3d_world_bwd[wBase + 0], means3d_world_bwd[wBase + 1], means3d_world_bwd[wBase + 2],
        quats_bwd[qBase + 0], quats_bwd[qBase + 1], quats_bwd[qBase + 2], quats_bwd[qBase + 3],
        scales_bwd[sBase + 0], scales_bwd[sBase + 1], scales_bwd[sBase + 2],
        viewmat_bwd[0], viewmat_bwd[1], viewmat_bwd[2],
        viewmat_bwd[4], viewmat_bwd[5], viewmat_bwd[6],
        viewmat_bwd[8], viewmat_bwd[9], viewmat_bwd[10],
        viewmat_bwd[3], viewmat_bwd[7], viewmat_bwd[11],
        intrinsics_bwd[0], intrinsics_bwd[1], intrinsics_bwd[2], intrinsics_bwd[3],
        image_dims_bwd[0], image_dims_bwd[1],
        cot
    );

    grad_means3d_world[wBase + 0] = g.wx;
    grad_means3d_world[wBase + 1] = g.wy;
    grad_means3d_world[wBase + 2] = g.wz;
    grad_quats[qBase + 0] = g.qw;
    grad_quats[qBase + 1] = g.qx;
    grad_quats[qBase + 2] = g.qy;
    grad_quats[qBase + 3] = g.qz;
    grad_scales[sBase + 0] = g.sx;
    grad_scales[sBase + 1] = g.sy;
    grad_scales[sBase + 2] = g.sz;
}
