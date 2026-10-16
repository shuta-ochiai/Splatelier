// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Fused SSIM backward (follows rahul-goel/fused-ssim, MIT, Copyright (c) 2024 Rahul Goel).
//   dL/dimg1 = conv(w, dm_dmu1·dL) + 2·img1·conv(w, dm_dsigma1_sq·dL) + img2·conv(w,
//   dm_dsigma12·dL)
// The Gaussian window is symmetric, so gather = the forward convolution. Out-of-image taps are 0
// (matches the forward zero-fill).

StructuredBuffer<float> img1;          // [H*W*C]  differentiated
StructuredBuffer<float> img2;          // [H*W*C]  target
StructuredBuffer<float> dL_dmap;       // [H*W*C]  dL/d(ssim_map)
// Caches written as half by ssim_fwd. Accumulation is float.
StructuredBuffer<half> dm_dmu1;       // [H*W*C]
StructuredBuffer<half> dm_dsigma1_sq; // [H*W*C]
StructuredBuffer<half> dm_dsigma12;   // [H*W*C]
StructuredBuffer<float> gauss;         // [2*rad + 1]
StructuredBuffer<int>   dims;          // [H, W, C, rad]
// Also adds the L1-term cotangent (1-λ)·sign(img1-img2)/N. [0, 0] gives the pure SSIM gradient
// (used in tests).
StructuredBuffer<float> lossParams;    // [1-λ, 1/(H*W*C)]
RWStructuredBuffer<float> dL_dimg1;    // [H*W*C]

[shader("compute")]
[numthreads(16, 16, 1)]
void ssim_bwd(uint3 tid : SV_DispatchThreadID)
{
    int H   = dims[0];
    int W   = dims[1];
    int C   = dims[2];
    int rad = dims[3];

    int x = int(tid.x);
    int y = int(tid.y);
    if (x >= W || y >= H) { return; }

    for (int c = 0; c < C; ++c) {
        float sum0 = 0.0, sum1 = 0.0, sum2 = 0.0;
        for (int dy = -rad; dy <= rad; ++dy) {
            int yy = y + dy;
            if (yy < 0 || yy >= H) { continue; }
            float wy = gauss[dy + rad];
            for (int dx = -rad; dx <= rad; ++dx) {
                int xx = x + dx;
                if (xx < 0 || xx >= W) { continue; }
                float w = wy * gauss[dx + rad];
                int idx = (yy * W + xx) * C + c;
                float dL = dL_dmap[idx];
                sum0 += w * dm_dmu1[idx]       * dL;
                sum1 += w * dm_dsigma1_sq[idx] * dL;
                sum2 += w * dm_dsigma12[idx]   * dL;
            }
        }
        int o = (y * W + x) * C + c;
        float p1 = img1[o];
        float p2 = img2[o];
        dL_dimg1[o] = sum0 + 2.0 * p1 * sum1 + p2 * sum2
                    + lossParams[0] * sign(p1 - p2) * lossParams[1];
    }
}

// ---- Backward (tiled + separable) ---------------------------------------------
// Same math as ssim_bwd. Assumptions and border handling are the same as ssim_fwd_tiled (rad <= 5,
// halo zero-filled, and because of the barrier, the grid is padded to a multiple of 16 to fill
// threadgroups).

static const int SSIMB_TILE = 16;
static const int SSIMB_HALO = 26;

groupshared float sb_P0[SSIMB_HALO * SSIMB_HALO];   // dm_dmu1 * dL
groupshared float sb_P1[SSIMB_HALO * SSIMB_HALO];   // dm_dsigma1_sq * dL
groupshared float sb_P2[SSIMB_HALO * SSIMB_HALO];   // dm_dsigma12 * dL
groupshared float sb_h0[SSIMB_TILE * SSIMB_HALO];
groupshared float sb_h1[SSIMB_TILE * SSIMB_HALO];
groupshared float sb_h2[SSIMB_TILE * SSIMB_HALO];

[shader("compute")]
[numthreads(16, 16, 1)]
void ssim_bwd_tiled(
    uint3 gid  : SV_DispatchThreadID,
    uint3 ltid : SV_GroupThreadID,
    uint3 grp  : SV_GroupID)
{
    int H   = dims[0];
    int W   = dims[1];
    int C   = dims[2];
    int rad = dims[3];

    int lx = int(ltid.x), ly = int(ltid.y);
    int lidx = ly * SSIMB_TILE + lx;
    int tx0 = int(grp.x) * SSIMB_TILE;
    int ty0 = int(grp.y) * SSIMB_TILE;
    int gx = int(gid.x), gy = int(gid.y);
    int taps = 2 * rad + 1;

    for (int c = 0; c < C; ++c) {
        // 1. halo load: P_k = dm_k * dL_dmap
        for (int i = lidx; i < SSIMB_HALO * SSIMB_HALO; i += SSIMB_TILE * SSIMB_TILE) {
            int col = i % SSIMB_HALO;
            int row = i / SSIMB_HALO;
            int hx = tx0 - rad + col;
            int hy = ty0 - rad + row;
            bool inb = (hx >= 0 && hx < W && hy >= 0 && hy < H);
            int gidx = (hy * W + hx) * C + c;
            float dL = inb ? dL_dmap[gidx] : 0.0;
            sb_P0[i] = inb ? dm_dmu1[gidx]       * dL : 0.0;
            sb_P1[i] = inb ? dm_dsigma1_sq[gidx] * dL : 0.0;
            sb_P2[i] = inb ? dm_dsigma12[gidx]   * dL : 0.0;
        }
        GroupMemoryBarrierWithGroupSync();

        // 2. horizontal pass
        for (int j = lidx; j < SSIMB_TILE * SSIMB_HALO; j += SSIMB_TILE * SSIMB_TILE) {
            int c0 = j % SSIMB_TILE;
            int r  = j / SSIMB_TILE;
            int base = r * SSIMB_HALO + c0;
            float a0 = 0.0, a1 = 0.0, a2 = 0.0;
            for (int k = 0; k < taps; ++k) {
                float w = gauss[k];
                a0 += w * sb_P0[base + k];
                a1 += w * sb_P1[base + k];
                a2 += w * sb_P2[base + k];
            }
            sb_h0[j] = a0; sb_h1[j] = a1; sb_h2[j] = a2;
        }
        GroupMemoryBarrierWithGroupSync();

        // 3. vertical pass + combine with img1/img2 at the center pixel
        if (gx < W && gy < H) {
            float s0 = 0.0, s1 = 0.0, s2 = 0.0;
            for (int m = 0; m < taps; ++m) {
                float w = gauss[m];
                int pidx = (ly + m) * SSIMB_TILE + lx;
                s0 += w * sb_h0[pidx];
                s1 += w * sb_h1[pidx];
                s2 += w * sb_h2[pidx];
            }
            int o = (gy * W + gx) * C + c;
            float p1 = img1[o];
            float p2 = img2[o];
            dL_dimg1[o] = s0 + 2.0 * p1 * s1 + p2 * s2
                        + lossParams[0] * sign(p1 - p2) * lossParams[1];
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
