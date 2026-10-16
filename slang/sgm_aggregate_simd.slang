// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// SIMD-group parallel version of SGM aggregation (1 path per dispatch). The recurrence, boundary
// conditions and output are exactly the same as `sgm_aggregate.slang` (the sequential 1 thread /
// scanline version), and results match.
//
// The parallelization follows the path aggregation of libSGM (fixstars, Apache-2.0,
// https://github.com/fixstars/libSGM): one scanline is handled by one warp (on Metal, one SIMD
// group = 32 lanes), with disparities (here depth indices) distributed across lanes. The
// neighboring disparities L_r(p − r, d ± 1) come from cross-lane shuffles, and min_i L_r(p − r, i)
// from a SIMD-group min reduction. The sequential version has only as many threads as scanlines (H
// / W / W+H−1 = a few hundred) and left most of the GPU idle; this version has scanlines × 32
// threads.
//
// Lane layout: assuming K ≤ 64, lane l handles depth indices d0 = l and d1 = l + 32. Slots with d ≥
// K hold INF and drop out of the min and neighbor terms naturally (equivalent to the sequential
// version's boundary handling of "setting the neighbor term to INF at d = 0 / d = K−1").
//
// Assumes SIMD-group width = 32 (Apple GPU). So that the 32 lanes of a scanline share one SIMD
// group, each run of 32 thread indices gets one scanline (threadgroup size is a multiple of 32).
// Shuffles / reductions must be executed uniformly by all lanes of the SIMD group, so out-of-range
// scanlines return together per SIMD group.

static const int SGM_INF = 1000000;

StructuredBuffer<int> cost_volume;   // [H * W * K] int32, Hamming distance
StructuredBuffer<int> dims;          // [W, H, K, P1, P2, direction] (same as the sequential version)

RWStructuredBuffer<int> L_out;       // [H * W * K] aggregated 1-path cost

[shader("compute")]
[numthreads(64, 1, 1)]
void sgm_aggregate_simd(uint3 tid : SV_DispatchThreadID)
{
    int W = dims[0];
    int H = dims[1];
    int K = dims[2];
    int P1 = dims[3];
    int P2 = dims[4];
    int dir = dims[5];

    int t = int(tid.x) / 32;            // scanline index (uniform within the SIMD group)
    int lane = int(WaveGetLaneIndex());

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
        // Diagonal "\" (x − y = const). Same scan start and length as the sequential version.
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
        // Diagonal "/" (x + y = const). Same scan start and length as the sequential version.
        if (t >= W + H - 1) { return; }
        int sum = t;
        int xStartTR; int yStartTR; int len;
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

    int d0 = lane;
    int d1 = lane + 32;
    bool v0 = d0 < K;
    bool v1 = d1 < K;

    // Start pixel: L_r = C (initial boundary condition)
    int pix0 = y0 * W + x0;
    int Lp0 = SGM_INF;
    int Lp1 = SGM_INF;
    if (v0) { Lp0 = cost_volume[pix0 * K + d0]; L_out[pix0 * K + d0] = Lp0; }
    if (v1) { Lp1 = cost_volume[pix0 * K + d1]; L_out[pix0 * K + d1] = Lp1; }

    int x = x0;
    int y = y0;
    // length is uniform per scanline (all lanes in the SIMD group run the same number of
    // iterations).
    for (int step = 1; step < length; step++) {
        x = x + dx;
        y = y + dy;
        int pix = y * W + x;

        // min_i L_r(p − r, i): min of the 2 slots within a lane, then SIMD-group reduction.
        int minPrev = WaveActiveMin(min(Lp0, Lp1));

        // Fetch L_prev of neighboring depths by shuffle (executed uniformly on all lanes, then
        // selected).
        int laneM = max(lane - 1, 0);
        int laneP = min(lane + 1, 31);
        int sh0m = WaveReadLaneAt(Lp0, laneM);   // L_prev[d0 − 1] (lane > 0)
        int sh1m = WaveReadLaneAt(Lp1, laneM);   // L_prev[d1 − 1] (lane > 0)
        int sh0p = WaveReadLaneAt(Lp0, laneP);   // L_prev[d0 + 1] (lane < 31)
        int sh1p = WaveReadLaneAt(Lp1, laneP);   // L_prev[d1 + 1] (lane < 31)
        int last0 = WaveReadLaneAt(Lp0, 31);     // L_prev[31]: d − 1 for d1 = 32
        int first1 = WaveReadLaneAt(Lp1, 0);     // L_prev[32]: d + 1 for d0 = 31

        int left0 = (lane > 0) ? sh0m : SGM_INF;
        int left1 = (lane > 0) ? sh1m : last0;
        int right0 = (lane < 31) ? sh0p : first1;
        int right1 = (lane < 31) ? sh1p : SGM_INF;

        int t4 = minPrev + P2;
        int Lc0 = SGM_INF;
        int Lc1 = SGM_INF;
        if (v0) {
            int c = cost_volume[pix * K + d0];
            int mt = min(min(Lp0, left0 + P1), min(right0 + P1, t4));
            Lc0 = c + mt - minPrev;
            L_out[pix * K + d0] = Lc0;
        }
        if (v1) {
            int c = cost_volume[pix * K + d1];
            int mt = min(min(Lp1, left1 + P1), min(right1 + P1, t4));
            Lc1 = c + mt - minPrev;
            L_out[pix * K + d1] = Lc1;
        }
        Lp0 = Lc0;
        Lp1 = Lc1;
    }
}
