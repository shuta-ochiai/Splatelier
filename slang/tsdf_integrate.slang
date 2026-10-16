// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// TSDF fusion (Curless & Levoy 1996). 1 thread = 1 voxel loops over all frames and averages the
// projective SDF (d - z) against the depth maps, normalized by the truncation distance. Voxels are
// allocated in blocks of 8³; voxel number g is in block g / 512, at in-block index g % 512 (x
// fastest). Each output is written exactly once per voxel, so no init needed.

StructuredBuffer<int>   block_coords;    // [nBlocks*3] block coordinates (voxel coordinate = block*8 + local)
StructuredBuffer<float> depth;           // [Σ H*W] all frames concatenated, <= 0 is invalid
StructuredBuffer<uint>  rgb_packed;      // [Σ H*W] r | g<<8 | b<<16
StructuredBuffer<float> viewmats;        // [F*16] row-major world→camera
StructuredBuffer<float> intrinsics_buf;  // [F*4] (fx, fy, cx, cy)
StructuredBuffer<int>   frame_meta;      // [F*3] (W, H, pixel offset)
StructuredBuffer<float> params_f;        // [5] (voxelSize, truncation, originX, originY, originZ)
StructuredBuffer<int>   params_i;        // [3] (voxelStart, voxelCount, frameCount)

RWStructuredBuffer<float> tsdf_out;      // [voxelCount] normalized TSDF (-1..1, positive = free space on the camera side)
RWStructuredBuffer<float> weight_out;    // [voxelCount] observation count
RWStructuredBuffer<uint>  color_out;     // [voxelCount] packed RGB8

[shader("compute")]
[numthreads(256, 1, 1)]
void tsdf_integrate(uint3 tid : SV_DispatchThreadID)
{
    int local = int(tid.x);
    if (local >= params_i[1]) { return; }
    int g = params_i[0] + local;
    int b = g >> 9;
    int l = g & 511;
    float voxelSize = params_f[0];
    float trunc = params_f[1];
    // World coordinates of the voxel center
    float wx = params_f[2] + (float(block_coords[b * 3 + 0] * 8 + (l & 7)) + 0.5) * voxelSize;
    float wy = params_f[3] + (float(block_coords[b * 3 + 1] * 8 + ((l >> 3) & 7)) + 0.5) * voxelSize;
    float wz = params_f[4] + (float(block_coords[b * 3 + 2] * 8 + (l >> 6)) + 0.5) * voxelSize;

    float sdfSum = 0.0;
    float wSum = 0.0;
    float cr = 0.0;
    float cg = 0.0;
    float cb = 0.0;
    float cw = 0.0;
    int frameCount = params_i[2];
    for (int f = 0; f < frameCount; f++) {
        int vo = f * 16;
        float z = viewmats[vo + 8] * wx + viewmats[vo + 9] * wy + viewmats[vo + 10] * wz + viewmats[vo + 11];
        if (z <= 1e-4) { continue; }
        float xc = viewmats[vo + 0] * wx + viewmats[vo + 1] * wy + viewmats[vo + 2] * wz + viewmats[vo + 3];
        float yc = viewmats[vo + 4] * wx + viewmats[vo + 5] * wy + viewmats[vo + 6] * wz + viewmats[vo + 7];
        float u = intrinsics_buf[f * 4 + 0] * xc / z + intrinsics_buf[f * 4 + 2];
        float v = intrinsics_buf[f * 4 + 1] * yc / z + intrinsics_buf[f * 4 + 3];
        // Pixel center = i + 0.5 convention, so index is floor
        if (u < 0.0 || v < 0.0) { continue; }
        int W = frame_meta[f * 3 + 0];
        int H = frame_meta[f * 3 + 1];
        int ix = int(u);
        int iy = int(v);
        if (ix >= W || iy >= H) { continue; }
        int pix = frame_meta[f * 3 + 2] + iy * W + ix;
        float d = depth[pix];
        if (d <= 0.0) { continue; }
        float sdf = d - z;
        // Behind the surface (beyond the truncation distance) is occluded and cannot be observed
        if (sdf < -trunc) { continue; }
        sdfSum += min(1.0, sdf / trunc);
        wSum += 1.0;
        if (abs(sdf) < trunc) {
            // Color averages only observations near the surface, weighting closer ones more
            float wc = 1.0 - abs(sdf) / trunc;
            uint p = rgb_packed[pix];
            cr += wc * float(p & 0xFF);
            cg += wc * float((p >> 8) & 0xFF);
            cb += wc * float((p >> 16) & 0xFF);
            cw += wc;
        }
    }
    tsdf_out[local] = wSum > 0.0 ? sdfSum / wSum : 1.0;
    weight_out[local] = wSum;
    uint packed = 0;
    if (cw > 0.0) {
        uint r = uint(min(255.0, cr / cw + 0.5));
        uint gg = uint(min(255.0, cg / cw + 0.5));
        uint bb = uint(min(255.0, cb / cw + 0.5));
        packed = r | (gg << 8) | (bb << 16);
    }
    color_out[local] = packed;
}
