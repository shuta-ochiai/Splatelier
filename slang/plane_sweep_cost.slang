// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Plane-sweep cost volume (Collins 1996, PocketGS Sec. III-A 3). Projects each target pixel into
// ref at each depth hypothesis and uses the Census Hamming distance as the cost. Hypotheses landing
// outside the ref image or behind it get the maximum cost (CENSUS_MAX_BITS = 25).

StructuredBuffer<int>   target_census;       // [H * W] lower 25 bits valid
StructuredBuffer<int>   ref_census;          // [H_ref * W_ref]
StructuredBuffer<float> intrinsics_target;   // [4] = (fx, fy, cx, cy)
StructuredBuffer<float> intrinsics_ref;      // [4] = (fx, fy, cx, cy)
StructuredBuffer<float> rt;                  // [12] = row-major R_rt (9) + t_rt (3)
StructuredBuffer<float> depths;              // [K] hypothesis depths
StructuredBuffer<int>   dims;                // [W, H, W_ref, H_ref, K]

RWStructuredBuffer<int> cost_volume;         // [H * W * K] Hamming distance 0..25

// CMCT is 25 bits and classic census is 24 bits (bit 24 always 0), so a 25-bit mask handles both.
static const int CENSUS_MAX_BITS = 25;
static const int CENSUS_BIT_MASK = 0x1FFFFFF;

// 32 threads in x = the Apple GPU SIMD width
[shader("compute")]
[numthreads(32, 8, 1)]
void plane_sweep_cost(uint3 tid : SV_DispatchThreadID)
{
    int W = dims[0];
    int H = dims[1];
    int W_ref = dims[2];
    int H_ref = dims[3];
    int K = dims[4];

    int xt = int(tid.x);
    int yt = int(tid.y);
    if (xt >= W || yt >= H) { return; }

    int pix_t = yt * W + xt;
    int target_sig = target_census[pix_t];

    float fx_t = intrinsics_target[0];
    float fy_t = intrinsics_target[1];
    float cx_t = intrinsics_target[2];
    float cy_t = intrinsics_target[3];
    float fx_r = intrinsics_ref[0];
    float fy_r = intrinsics_ref[1];
    float cx_r = intrinsics_ref[2];
    float cy_r = intrinsics_ref[3];

    float r00 = rt[0]; float r01 = rt[1]; float r02 = rt[2];
    float r10 = rt[3]; float r11 = rt[4]; float r12 = rt[5];
    float r20 = rt[6]; float r21 = rt[7]; float r22 = rt[8];
    float tx_ = rt[9];  float ty_ = rt[10]; float tz_ = rt[11];

    // Depth-1 ray through the pixel center
    float px = (float(xt) + 0.5 - cx_t) / fx_t;
    float py = (float(yt) + 0.5 - cy_t) / fy_t;

    for (int k = 0; k < K; k++) {
        float d = depths[k];

        float Xt = px * d;
        float Yt = py * d;
        float Zt = d;
        // P_r = R_rt P_t + t_rt
        float Xr = r00 * Xt + r01 * Yt + r02 * Zt + tx_;
        float Yr = r10 * Xt + r11 * Yt + r12 * Zt + ty_;
        float Zr = r20 * Xt + r21 * Yt + r22 * Zt + tz_;

        int cost = CENSUS_MAX_BITS;
        if (Zr > 1e-6) {
            float ur = fx_r * Xr / Zr + cx_r;
            float vr = fy_r * Yr / Zr + cy_r;
            // Nearest neighbor. Pixel i's center is i + 0.5, so index = floor(u) (floor(u + 0.5) is
            // off by half a pixel)
            int ix = int(floor(ur));
            int iy = int(floor(vr));
            if (ix >= 0 && ix < W_ref && iy >= 0 && iy < H_ref) {
                int ref_sig = ref_census[iy * W_ref + ix];
                int xor_bits = (target_sig ^ ref_sig) & CENSUS_BIT_MASK;
                cost = countbits(xor_bits);
            }
        }
        cost_volume[pix_t * K + k] = cost;
    }
}
