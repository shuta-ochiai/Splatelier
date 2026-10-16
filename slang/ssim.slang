// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Fused SSIM forward ([H, W, C], range [0, 1]). Writes the SSIM map and 3 partial-derivative maps
// for backward. The math and the partial-derivative caching scheme follow rahul-goel/fused-ssim
// (MIT, Copyright (c) 2024 Rahul Goel). img1 is differentiated (the render); img2 is the fixed
// target.

// Autodiffing SSIM as a function of (mu1, E[x²], E[xy]) folds the mu1 dependence of the sigma terms
// into d/d(mu1), so backward needs only convolutions of the caches.
[Differentiable]
float ssim_pixel(
    float mu1, float m1x2, float m1xy,
    no_diff float mu2, no_diff float m2x2,
    no_diff float c1, no_diff float c2
) {
    float mu1sq = mu1 * mu1;
    float mu2sq = mu2 * mu2;
    float sigma1_sq = m1x2 - mu1sq;
    float sigma2_sq = m2x2 - mu2sq;
    float sigma12   = m1xy - mu1 * mu2;

    float A  = mu1sq + mu2sq + c1;
    float B  = sigma1_sq + sigma2_sq + c2;
    float Cn = 2.0 * mu1 * mu2 + c1;
    float Dn = 2.0 * sigma12 + c2;
    return (Cn * Dn) / (A * B);
}

struct SsimGrad { float d_mu1; float d_m1x2; float d_m1xy; };

SsimGrad ssim_pixel_grad(
    float mu1, float m1x2, float m1xy,
    float mu2, float m2x2,
    float c1, float c2
) {
    DifferentialPair<float> dpMu1  = diffPair(mu1, 0.0);
    DifferentialPair<float> dpM1x2 = diffPair(m1x2, 0.0);
    DifferentialPair<float> dpM1xy = diffPair(m1xy, 0.0);
    bwd_diff(ssim_pixel)(dpMu1, dpM1x2, dpM1xy, mu2, m2x2, c1, c2, 1.0);
    SsimGrad g;
    g.d_mu1  = dpMu1.d;
    g.d_m1x2 = dpM1x2.d;
    g.d_m1xy = dpM1xy.d;
    return g;
}

// C1 = 0.01², C2 = 0.03² (Wang 2004 / gsplat)
static const float kC1 = 0.0001;
static const float kC2 = 0.0009;

// ---- Forward ------------------------------------------------------------------

StructuredBuffer<float> img1;           // [H*W*C] HWC
StructuredBuffer<float> img2;           // [H*W*C]
StructuredBuffer<float> gauss;          // [2*rad + 1] normalized 1D Gaussian weights
StructuredBuffer<int>   dims;           // [H, W, C, rad]
// Compute in fp32, store as half only (ssim_bwd reads them back as float).
RWStructuredBuffer<half> ssim_map;      // [H*W*C]
RWStructuredBuffer<half> dm_dmu1;       // [H*W*C] d(map)/d(mu1) (total derivative)
RWStructuredBuffer<half> dm_dsigma1_sq; // [H*W*C]
RWStructuredBuffer<half> dm_dsigma12;   // [H*W*C]
// Per-tile loss partial sums written only by ssim_fwd_tiled. Every tile writes exactly once, so no
// init needed. The trainer sums them and divides L1 by H*W*C and SSIM by (H-2rad)(W-2rad)C.
RWStructuredBuffer<float> tilePartialL1;     // [numTiles]  Σ|img1-img2| (all pixels)
RWStructuredBuffer<float> tilePartialSSIM;   // [numTiles]  Σ ssim (region excluding the rad border)

[shader("compute")]
[numthreads(16, 16, 1)]
void ssim_fwd(uint3 tid : SV_DispatchThreadID)
{
    int H   = dims[0];
    int W   = dims[1];
    int C   = dims[2];
    int rad = dims[3];

    int x = int(tid.x);
    int y = int(tid.y);
    if (x >= W || y >= H) { return; }

    for (int c = 0; c < C; ++c) {
        float sumX = 0.0, sumX2 = 0.0, sumY = 0.0, sumY2 = 0.0, sumXY = 0.0;
        for (int dy = -rad; dy <= rad; ++dy) {
            int yy = y + dy;
            if (yy < 0 || yy >= H) { continue; }
            float wy = gauss[dy + rad];
            for (int dx = -rad; dx <= rad; ++dx) {
                int xx = x + dx;
                if (xx < 0 || xx >= W) { continue; }
                float w = wy * gauss[dx + rad];
                int idx = (yy * W + xx) * C + c;
                float a = img1[idx];
                float b = img2[idx];
                sumX  += w * a;
                sumX2 += w * a * a;
                sumY  += w * b;
                sumY2 += w * b * b;
                sumXY += w * a * b;
            }
        }

        float val = ssim_pixel(sumX, sumX2, sumXY, sumY, sumY2, kC1, kC2);
        SsimGrad g = ssim_pixel_grad(sumX, sumX2, sumXY, sumY, sumY2, kC1, kC2);

        int o = (y * W + x) * C + c;
        ssim_map[o]      = half(val);
        dm_dmu1[o]       = half(g.d_mu1);
        dm_dsigma1_sq[o] = half(g.d_m1x2);
        dm_dsigma12[o]   = half(g.d_m1xy);
    }
}

// ---- Forward (tiled + separable) ----------------------------------------------
// Same math as ssim_fwd, with tile + halo in threadgroup memory, computed in 2 passes (horizontal →
// vertical). Assumes rad <= 5 (FusedSSIM uses this only when window == 11). Outside the image is
// zero-filled (equivalent to skipping taps).

static const int SSIM_TILE = 16;
static const int SSIM_HALO = 26;   // 16 + 2*5

groupshared float s_img1[SSIM_HALO * SSIM_HALO];
groupshared float s_img2[SSIM_HALO * SSIM_HALO];
groupshared float s_hX[SSIM_TILE * SSIM_HALO];   // horizontal-pass partial sum: μ1
groupshared float s_hX2[SSIM_TILE * SSIM_HALO];  // E[x^2]
groupshared float s_hY[SSIM_TILE * SSIM_HALO];   // μ2
groupshared float s_hY2[SSIM_TILE * SSIM_HALO];  // E[y^2]
groupshared float s_hXY[SSIM_TILE * SSIM_HALO];  // E[xy]
// for the tree reduction of in-tile loss partial sums
groupshared float s_redL1[SSIM_TILE * SSIM_TILE];
groupshared float s_redSSIM[SSIM_TILE * SSIM_TILE];

[shader("compute")]
[numthreads(16, 16, 1)]
void ssim_fwd_tiled(
    uint3 gid  : SV_DispatchThreadID,
    uint3 ltid : SV_GroupThreadID,
    uint3 grp  : SV_GroupID)
{
    int H   = dims[0];
    int W   = dims[1];
    int C   = dims[2];
    int rad = dims[3];

    int lx = int(ltid.x), ly = int(ltid.y);
    int lidx = ly * SSIM_TILE + lx;
    int tx0 = int(grp.x) * SSIM_TILE;
    int ty0 = int(grp.y) * SSIM_TILE;
    int gx = int(gid.x), gy = int(gid.y);
    int taps = 2 * rad + 1;

    float l1acc = 0.0;     // Σ_c |img1-img2|
    float ssimacc = 0.0;   // Σ_c ssim (excluding the border)

    for (int c = 0; c < C; ++c) {
        // 1. halo load. halo[col,row] = global (tx0-rad+col, ty0-rad+row)
        for (int i = lidx; i < SSIM_HALO * SSIM_HALO; i += SSIM_TILE * SSIM_TILE) {
            int col = i % SSIM_HALO;
            int row = i / SSIM_HALO;
            int hx = tx0 - rad + col;
            int hy = ty0 - rad + row;
            bool inb = (hx >= 0 && hx < W && hy >= 0 && hy < H);
            int gidx = (hy * W + hx) * C + c;
            s_img1[i] = inb ? img1[gidx] : 0.0;
            s_img2[i] = inb ? img2[gidx] : 0.0;
        }
        GroupMemoryBarrierWithGroupSync();

        // 2. horizontal pass (output column c0 × halo row r)
        for (int j = lidx; j < SSIM_TILE * SSIM_HALO; j += SSIM_TILE * SSIM_TILE) {
            int c0 = j % SSIM_TILE;
            int r  = j / SSIM_TILE;
            int base = r * SSIM_HALO + c0;
            float aX = 0.0, aX2 = 0.0, aY = 0.0, aY2 = 0.0, aXY = 0.0;
            for (int k = 0; k < taps; ++k) {
                float w = gauss[k];
                float a = s_img1[base + k];
                float b = s_img2[base + k];
                aX  += w * a;
                aX2 += w * a * a;
                aY  += w * b;
                aY2 += w * b * b;
                aXY += w * a * b;
            }
            s_hX[j] = aX; s_hX2[j] = aX2; s_hY[j] = aY; s_hY2[j] = aY2; s_hXY[j] = aXY;
        }
        GroupMemoryBarrierWithGroupSync();

        // 3. vertical pass (1 thread = 1 output pixel)
        if (gx < W && gy < H) {
            float sX = 0.0, sX2 = 0.0, sY = 0.0, sY2 = 0.0, sXY = 0.0;
            for (int m = 0; m < taps; ++m) {
                float w = gauss[m];
                int pidx = (ly + m) * SSIM_TILE + lx;
                sX  += w * s_hX[pidx];
                sX2 += w * s_hX2[pidx];
                sY  += w * s_hY[pidx];
                sY2 += w * s_hY2[pidx];
                sXY += w * s_hXY[pidx];
            }
            float val = ssim_pixel(sX, sX2, sXY, sY, sY2, kC1, kC2);
            SsimGrad g = ssim_pixel_grad(sX, sX2, sXY, sY, sY2, kC1, kC2);
            int o = (gy * W + gx) * C + c;
            ssim_map[o]      = half(val);
            dm_dmu1[o]       = half(g.d_mu1);
            dm_dsigma1_sq[o] = half(g.d_m1x2);
            dm_dsigma12[o]   = half(g.d_m1xy);

            // The center pixel is at (lx+rad, ly+rad) in the halo
            float a0 = s_img1[(ly + rad) * SSIM_HALO + (lx + rad)];
            float b0 = s_img2[(ly + rad) * SSIM_HALO + (lx + rad)];
            l1acc += abs(a0 - b0);
            // Exclude the rad border from SSIM, same as the trainer
            if (gx >= rad && gx < W - rad && gy >= rad && gy < H - rad) {
                ssimacc += val;
            }
        }
        GroupMemoryBarrierWithGroupSync();   // before the next c reuses shared memory
    }

    // In-tile tree reduction (out-of-image threads hold 0)
    s_redL1[lidx]   = l1acc;
    s_redSSIM[lidx] = ssimacc;
    GroupMemoryBarrierWithGroupSync();
    for (int s = (SSIM_TILE * SSIM_TILE) / 2; s > 0; s >>= 1) {
        if (lidx < s) {
            s_redL1[lidx]   += s_redL1[lidx + s];
            s_redSSIM[lidx] += s_redSSIM[lidx + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lidx == 0) {
        int numTilesX = (W + SSIM_TILE - 1) / SSIM_TILE;
        int tileIdx = int(grp.y) * numTilesX + int(grp.x);
        tilePartialL1[tileIdx]   = s_redL1[0];
        tilePartialSSIM[tileIdx] = s_redSSIM[0];
    }
}
