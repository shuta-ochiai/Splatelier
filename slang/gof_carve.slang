// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Sign of the Gaussian opacity field at 3D points. GOF (Yu et al., TOG 2024, eq. 10) defines a
// point's opacity as the minimum over the training views that observe it; implemented from the
// paper only. A view's ray opacity is non-decreasing in depth (see gof_depth.slang), so that view's
// opacity at the point is ≥ 0.5 exactly when the point lies at or behind the view's 0.5-crossing
// depth. The point is inside (field ≥ 0.5) when no observing view sees it in front of that depth.
//
// GOF evaluates each point on its exact ray; here a view's depth map is sampled at the point's
// projection. Within a smooth surface the 4 surrounding pixel depths are interpolated bilinearly
// (nearest-pixel lookup erodes surfaces seen at grazing angles, since one carving view suffices).
// Across a depth discontinuity (the 4 depths differ by more than the gap) or next to pixels that
// never reach 0.5, the nearest of the valid depths is used, so a view carves only what lies in
// front of everything it sees there. The caller may also tolerate a few carving views (pose error);
// with 0 tolerated this is the GOF field. 1 thread = 1 point, looping over all views; every output
// is written exactly once. Points are either the voxel centers of 8³ blocks (mode 0, same layout as
// tsdf_integrate) or an explicit list (mode 1).

StructuredBuffer<float> depth;           // [Σ H*W] 0.5-crossing depths of all views, <= 0: never reaches 0.5
StructuredBuffer<uint>  rgb_packed;      // [Σ H*W] r | g<<8 | b<<16 (rendered color)
StructuredBuffer<float> viewmats;        // [F*16] row-major world→camera
StructuredBuffer<float> intrinsics_buf;  // [F*4] (fx, fy, cx, cy)
StructuredBuffer<int>   frame_meta;      // [F*3] (W, H, pixel offset)
StructuredBuffer<float> carve_params;    // [5] (depthToleranceAbs, depthToleranceRel, colorBand, gapAbs, gapRel)
StructuredBuffer<int>   block_coords;    // [nBlocks*3] (mode 0)
StructuredBuffer<float> grid_params;     // [4] (voxelSize, originX, originY, originZ) (mode 0)
StructuredBuffer<float> points;          // [P*3] (mode 1)
StructuredBuffer<int>   carve_range;     // [4] (mode, start, count, frameCount)

RWStructuredBuffer<float> carve_out;     // [count] views that see the point in front of their surface
RWStructuredBuffer<float> observed_out;  // [count] views whose frustum contains the point
RWStructuredBuffer<uint>  color_out;     // [count] packed RGB8 near the surface, 0 if none
RWStructuredBuffer<float> support_out;   // [count] views whose surface depth is within the color band

[shader("compute")]
[numthreads(256, 1, 1)]
void gof_carve(uint3 tid : SV_DispatchThreadID)
{
    int local = int(tid.x);
    if (local >= carve_range[2]) { return; }
    int g = carve_range[1] + local;
    float wx;
    float wy;
    float wz;
    if (carve_range[0] == 0) {
        int b = g >> 9;
        int l = g & 511;
        float voxelSize = grid_params[0];
        wx = grid_params[1] + (float(block_coords[b * 3 + 0] * 8 + (l & 7)) + 0.5) * voxelSize;
        wy = grid_params[2] + (float(block_coords[b * 3 + 1] * 8 + ((l >> 3) & 7)) + 0.5) * voxelSize;
        wz = grid_params[3] + (float(block_coords[b * 3 + 2] * 8 + (l >> 6)) + 0.5) * voxelSize;
    } else {
        wx = points[g * 3 + 0];
        wy = points[g * 3 + 1];
        wz = points[g * 3 + 2];
    }

    float tolAbs = carve_params[0];
    float tolRel = carve_params[1];
    float band = carve_params[2];
    float carve = 0.0;
    float observed = 0.0;
    float cr = 0.0;
    float cg = 0.0;
    float cb = 0.0;
    float cw = 0.0;
    float support = 0.0;
    int frameCount = carve_range[3];
    for (int f = 0; f < frameCount; f++) {
        int vo = f * 16;
        float z = viewmats[vo + 8] * wx + viewmats[vo + 9] * wy + viewmats[vo + 10] * wz + viewmats[vo + 11];
        if (z <= 1e-4) { continue; }
        float xc = viewmats[vo + 0] * wx + viewmats[vo + 1] * wy + viewmats[vo + 2] * wz + viewmats[vo + 3];
        float yc = viewmats[vo + 4] * wx + viewmats[vo + 5] * wy + viewmats[vo + 6] * wz + viewmats[vo + 7];
        float u = intrinsics_buf[f * 4 + 0] * xc / z + intrinsics_buf[f * 4 + 2];
        float v = intrinsics_buf[f * 4 + 1] * yc / z + intrinsics_buf[f * 4 + 3];
        if (u < 0.0 || v < 0.0) { continue; }
        int W = frame_meta[f * 3 + 0];
        int H = frame_meta[f * 3 + 1];
        int ix = int(u);
        int iy = int(v);
        if (ix >= W || iy >= H) { continue; }
        observed += 1.0;
        int base = frame_meta[f * 3 + 2];
        int pix = base + iy * W + ix;

        // Bilinear neighborhood of the pixel centers (i + 0.5) around (u, v).
        float fu = u - 0.5;
        float fv = v - 0.5;
        int x0 = int(floor(fu));
        int y0 = int(floor(fv));
        float ax = fu - float(x0);
        float ay = fv - float(y0);
        int xa = clamp(x0, 0, W - 1);
        int xb = clamp(x0 + 1, 0, W - 1);
        int ya = clamp(y0, 0, H - 1);
        int yb = clamp(y0 + 1, 0, H - 1);
        float d00 = depth[base + ya * W + xa];
        float d10 = depth[base + ya * W + xb];
        float d01 = depth[base + yb * W + xa];
        float d11 = depth[base + yb * W + xb];
        float dmin = 1e30;
        float dmax = 0.0;
        if (d00 > 0.0) { dmin = min(dmin, d00); dmax = max(dmax, d00); }
        if (d10 > 0.0) { dmin = min(dmin, d10); dmax = max(dmax, d10); }
        if (d01 > 0.0) { dmin = min(dmin, d01); dmax = max(dmax, d01); }
        if (d11 > 0.0) { dmin = min(dmin, d11); dmax = max(dmax, d11); }
        // No neighboring ray reaches opacity 0.5: the rays are free space in this view.
        if (dmax <= 0.0) { carve += 1.0; continue; }
        float d = dmin;
        bool allValid = d00 > 0.0 && d10 > 0.0 && d01 > 0.0 && d11 > 0.0;
        if (allValid && dmax - dmin <= carve_params[3] + carve_params[4] * dmin) {
            d = (1.0 - ay) * ((1.0 - ax) * d00 + ax * d10) + ay * ((1.0 - ax) * d01 + ax * d11);
        }
        if (z < d - (tolAbs + tolRel * d)) { carve += 1.0; continue; }
        float off = abs(z - d);
        if (off < band) {
            support += 1.0;
            float wc = 1.0 - off / band;
            uint p = rgb_packed[pix];
            cr += wc * float(p & 0xFF);
            cg += wc * float((p >> 8) & 0xFF);
            cb += wc * float((p >> 16) & 0xFF);
            cw += wc;
        }
    }
    carve_out[local] = carve;
    observed_out[local] = observed;
    support_out[local] = support;
    uint packed = 0;
    if (cw > 0.0) {
        uint rr = uint(min(255.0, cr / cw + 0.5));
        uint gg = uint(min(255.0, cg / cw + 0.5));
        uint bb = uint(min(255.0, cb / cw + 0.5));
        packed = rr | (gg << 8) | (bb << 16);
    }
    color_out[local] = packed;
}
