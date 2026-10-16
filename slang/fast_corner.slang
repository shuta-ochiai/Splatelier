// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// FAST-9 corner detection (Rosten & Drummond 2006, PocketGS Appendix A.b). 1 thread / pixel. A
// corner if 9 or more contiguous points of the 16 on the radius-3 circle are all brighter than I_p
// + t (or all darker than I_p - t). Outputs a score (0 for non-corners). Non-max suppression
// happens later.

StructuredBuffer<float> luma;          // [H * W] luma [0, 1]
StructuredBuffer<int>   dims;          // [3] (W, H, threshold_q8). threshold_q8 = int(thr * 256)

RWStructuredBuffer<int> score_map;     // [H * W] 0 = not a corner, > 0 = score (q8)

// The 16 points of the Bresenham radius-3 circle, clockwise from 12 o'clock (0,-3).
static const int CIRCLE_DX[16] = {  0,  1,  2,  3,  3,  3,  2,  1,  0, -1, -2, -3, -3, -3, -2, -1 };
static const int CIRCLE_DY[16] = { -3, -3, -2, -1,  0,  1,  2,  3,  3,  3,  2,  1,  0, -1, -2, -3 };

static const int N_CORNER = 9;

[shader("compute")]
[numthreads(16, 16, 1)]
void fast_corner(uint3 tid : SV_DispatchThreadID)
{
    int W = dims[0];
    int H = dims[1];
    int threshold_q8 = dims[2];
    int x = int(tid.x);
    int y = int(tid.y);
    if (x < 3 || x >= W - 3 || y < 3 || y >= H - 3) {
        if (x < W && y < H) { score_map[y * W + x] = 0; }
        return;
    }

    int pix = y * W + x;
    // Compare in q8 fixed point
    int ip = int(luma[pix] * 256.0);
    int thr = threshold_q8;
    int hi = ip + thr;
    int lo = ip - thr;

    int v[16];
    [unroll]
    for (int k = 0; k < 16; k++) {
        int nx = x + CIRCLE_DX[k];
        int ny = y + CIRCLE_DY[k];
        v[k] = int(luma[ny * W + nx] * 256.0);
    }

    // Fast pre-test (Rosten): reject unless at least 3 of the 4 points at 12/3/6/9 o'clock exceed
    // the threshold in the same direction
    int hiCount = (v[0] > hi ? 1 : 0) + (v[4] > hi ? 1 : 0) + (v[8] > hi ? 1 : 0) + (v[12] > hi ? 1 : 0);
    int loCount = (v[0] < lo ? 1 : 0) + (v[4] < lo ? 1 : 0) + (v[8] < lo ? 1 : 0) + (v[12] < lo ? 1 : 0);
    if (hiCount < 3 && loCount < 3) {
        score_map[pix] = 0;
        return;
    }

    // The circle wraps, so scan 2 rounds (32 steps) to find the longest bright and dark runs
    int maxBrightRun = 0;
    int maxDarkRun = 0;
    int curBright = 0;
    int curDark = 0;
    [unroll]
    for (int j = 0; j < 32; j++) {
        int idx = j & 15;
        int vj = v[idx];
        if (vj > hi) {
            curBright++;
            if (curBright > maxBrightRun) { maxBrightRun = curBright; }
            curDark = 0;
        } else if (vj < lo) {
            curDark++;
            if (curDark > maxDarkRun) { maxDarkRun = curDark; }
            curBright = 0;
        } else {
            curBright = 0;
            curDark = 0;
        }
    }

    if (maxBrightRun < N_CORNER && maxDarkRun < N_CORNER) {
        score_map[pix] = 0;
        return;
    }

    // score = sum of |v - ip| over all circle points beyond the threshold on that side (simplified,
    // not limited to the run; for NMS comparison)
    int score = 0;
    if (maxBrightRun >= N_CORNER) {
        [unroll]
        for (int k = 0; k < 16; k++) {
            if (v[k] > hi) { score += v[k] - ip; }
        }
    } else {
        [unroll]
        for (int k = 0; k < 16; k++) {
            if (v[k] < lo) { score += ip - v[k]; }
        }
    }
    score_map[pix] = score;
}
