// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// SGM aggregation along one path (Hirschmüller 2005/2008, PocketGS Sec. III-A 3).
//   L_r(p, d) = C(p, d) + min(L_r(p−r, d), L_r(p−r, d±1) + P1, min_i L_r(p−r, i) + P2) − min_i
//   L_r(p−r, i)
// The Swift side dispatches 8 times for directions 0..7 and sums them. 1 thread handles 1 scanline.

// Arrays need a compile-time size. Must match Swift SGMAggregation.kMax.
static const int SGM_K_MAX = 64;

// Effectively infinite value for d±1 at the edges (well above the maximum total cost over 8 paths)
static const int SGM_INF = 1000000;

StructuredBuffer<int> cost_volume;   // [H * W * K] Hamming distance
StructuredBuffer<int> dims;          // [W, H, K, P1, P2, direction]
                                     // direction: 0=L→R, 1=R→L, 2=T→B, 3=B→T,
                                     //            4=TL→BR, 5=BR→TL, 6=TR→BL, 7=BL→TR

RWStructuredBuffer<int> L_out;       // [H * W * K] aggregated cost for this path

[shader("compute")]
[numthreads(64, 1, 1)]
void sgm_aggregate(uint3 tid : SV_DispatchThreadID)
{
    int W = dims[0];
    int H = dims[1];
    int K = dims[2];
    int P1 = dims[3];
    int P2 = dims[4];
    int dir = dims[5];

    int t = int(tid.x);

    int x0; int y0; int dx; int dy; int length;
    if (dir == 0) {
        if (t >= H) { return; }
        x0 = 0; y0 = t; dx = 1; dy = 0; length = W;
    } else if (dir == 1) {
        if (t >= H) { return; }
        x0 = W - 1; y0 = t; dx = -1; dy = 0; length = W;
    } else if (dir == 2) {
        if (t >= W) { return; }
        x0 = t; y0 = 0; dx = 0; dy = 1; length = H;
    } else if (dir == 3) {
        if (t >= W) { return; }
        x0 = t; y0 = H - 1; dx = 0; dy = -1; length = H;
    } else if (dir == 4 || dir == 5) {
        // Diagonal "\" (x - y = t - (H-1) is constant)
        if (t >= W + H - 1) { return; }
        int diag = t - (H - 1);
        int xStart; int yStart; int len;
        if (diag >= 0) {
            xStart = diag; yStart = 0;
            len = min(W - diag, H);
        } else {
            xStart = 0; yStart = -diag;
            len = min(W, H + diag);
        }
        length = len;
        if (dir == 4) {
            x0 = xStart; y0 = yStart; dx = 1; dy = 1;
        } else {
            x0 = xStart + (len - 1); y0 = yStart + (len - 1); dx = -1; dy = -1;
        }
    } else {
        // Diagonal "/" (x + y = t is constant)
        if (t >= W + H - 1) { return; }
        int sum = t;
        int xStartTR; int yStartTR; int len;
        // top-right edge
        if (sum < W) {
            xStartTR = sum; yStartTR = 0;
        } else {
            xStartTR = W - 1; yStartTR = sum - (W - 1);
        }
        len = min(min(sum + 1, W + H - 1 - sum), min(W, H));
        length = len;
        if (dir == 6) {
            x0 = xStartTR; y0 = yStartTR; dx = -1; dy = 1;
        } else {
            x0 = xStartTR - (len - 1); y0 = yStartTR + (len - 1); dx = 1; dy = -1;
        }
    }

    // Defensive; assumes the caller guarantees K <= SGM_K_MAX
    int kRun = min(K, SGM_K_MAX);

    int L_prev[SGM_K_MAX];
    int L_curr[SGM_K_MAX];

    // At the start, L_r = C
    int pix0 = y0 * W + x0;
    [unroll]
    for (int k0 = 0; k0 < SGM_K_MAX; k0++) {
        if (k0 < kRun) {
            int c0 = cost_volume[pix0 * K + k0];
            L_prev[k0] = c0;
            L_out[pix0 * K + k0] = c0;
        }
    }

    int x = x0;
    int y = y0;
    for (int step = 1; step < length; step++) {
        x = x + dx;
        y = y + dy;
        int pix = y * W + x;

        int minLprev = L_prev[0];
        for (int k = 1; k < kRun; k++) {
            if (L_prev[k] < minLprev) { minLprev = L_prev[k]; }
        }

        for (int k = 0; k < kRun; k++) {
            int c = cost_volume[pix * K + k];
            int term1 = L_prev[k];
            int term2 = (k > 0)         ? (L_prev[k - 1] + P1) : SGM_INF;
            int term3 = (k < kRun - 1)  ? (L_prev[k + 1] + P1) : SGM_INF;
            int term4 = minLprev + P2;
            int mt = min(min(term1, term2), min(term3, term4));
            int lv = c + mt - minLprev;
            L_curr[k] = lv;
            L_out[pix * K + k] = lv;
        }

        for (int k = 0; k < kRun; k++) {
            L_prev[k] = L_curr[k];
        }
    }
}
