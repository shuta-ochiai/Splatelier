// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// `plane_sweep_cost_median` with hypotheses per pixel: pixel p's k-th depth is
// center_depth[p] · factors[k], so a finer pass can search a narrow band around a coarser pass's
// depth with few hypotheses. A pixel with center_depth <= 0 gets the maximum cost at every k.

StructuredBuffer<int>   target_census;       // [H * W] int32
StructuredBuffer<int>   ref_census;          // [M * H_ref * W_ref] int32 (concatenated per reference)
StructuredBuffer<float> intrinsics_target;   // [4] = (fx, fy, cx, cy)
StructuredBuffer<float> intrinsics_ref;      // [M * 4]
StructuredBuffer<float> rt;                  // [M * 12] = per reference, row-major R_rt (9) + t_rt (3)
StructuredBuffer<float> factors;             // [K] depth factors around the center
StructuredBuffer<float> center_depth;        // [H * W] center depth per pixel
StructuredBuffer<int>   dims;                // [W, H, W_ref, H_ref, K, M]

RWStructuredBuffer<int> cost_volume;         // [H * W * K] int32, median of the M costs

static const int CENSUS_MAX_BITS = 25;        // same as plane_sweep_cost.slang
static const int CENSUS_BIT_MASK = 0x1FFFFFF;
static const int MEDIAN_M_MAX = 8;

[shader("compute")]
[numthreads(32, 8, 1)]
void plane_sweep_cost_median_band(uint3 tid : SV_DispatchThreadID)
{
    int W = dims[0];
    int H = dims[1];
    int W_ref = dims[2];
    int H_ref = dims[3];
    int K = dims[4];
    int M = min(dims[5], MEDIAN_M_MAX);

    int xt = int(tid.x);
    int yt = int(tid.y);
    if (xt >= W || yt >= H) { return; }

    int pix_t = yt * W + xt;
    int target_sig = target_census[pix_t];

    float fx_t = intrinsics_target[0];
    float fy_t = intrinsics_target[1];
    float cx_t = intrinsics_target[2];
    float cy_t = intrinsics_target[3];
    float px = (float(xt) + 0.5 - cx_t) / fx_t;
    float py = (float(yt) + 0.5 - cy_t) / fy_t;

    float center = center_depth[pix_t];
    if (!(center > 0.0)) {
        for (int k = 0; k < K; k++) { cost_volume[pix_t * K + k] = CENSUS_MAX_BITS; }
        return;
    }
    for (int k = 0; k < K; k++) {
        float d = center * factors[k];
        float Xt = px * d;
        float Yt = py * d;
        float Zt = d;

        int c[MEDIAN_M_MAX];
        for (int m = 0; m < M; m++) {
            float fx_r = intrinsics_ref[m * 4 + 0];
            float fy_r = intrinsics_ref[m * 4 + 1];
            float cx_r = intrinsics_ref[m * 4 + 2];
            float cy_r = intrinsics_ref[m * 4 + 3];
            int o = m * 12;
            float Xr = rt[o + 0] * Xt + rt[o + 1] * Yt + rt[o + 2] * Zt + rt[o + 9];
            float Yr = rt[o + 3] * Xt + rt[o + 4] * Yt + rt[o + 5] * Zt + rt[o + 10];
            float Zr = rt[o + 6] * Xt + rt[o + 7] * Yt + rt[o + 8] * Zt + rt[o + 11];
            int cost = CENSUS_MAX_BITS;
            if (Zr > 1e-6) {
                float ur = fx_r * Xr / Zr + cx_r;
                float vr = fy_r * Yr / Zr + cy_r;
                int ix = int(floor(ur));
                int iy = int(floor(vr));
                if (ix >= 0 && ix < W_ref && iy >= 0 && iy < H_ref) {
                    int ref_sig = ref_census[m * H_ref * W_ref + iy * W_ref + ix];
                    cost = countbits((target_sig ^ ref_sig) & CENSUS_BIT_MASK);
                }
            }
            c[m] = cost;
        }

        // odd-even transposition sort (always sorted after M passes) → index M/2 is the median.
        for (int round = 0; round < M; round++) {
            for (int i = round % 2; i < M - 1; i += 2) {
                int lo = min(c[i], c[i + 1]);
                int hi = max(c[i], c[i + 1]);
                c[i] = lo;
                c[i + 1] = hi;
            }
        }
        cost_volume[pix_t * K + k] = c[M / 2];
    }
}
