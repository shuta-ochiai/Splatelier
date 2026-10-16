// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// steered BRIEF-256 descriptor (Calonder et al. 2010 / ORB, Rublee et al. 2011 §3.2, §4.1). Rotates
// the 256 pairs of a 31×31 patch by the intensity-centroid orientation θ and compares them: bit i =
// I(a_i) > I(b_i). Implemented from the papers; the pattern is generated with a fixed seed.
// Rotation invariant, so it also matches non-consecutive frame pairs for loop closure.

StructuredBuffer<float> luma;                  // [H * W] luma [0, 1] (BT.601)
StructuredBuffer<int>   kp_xy;                 // [N_kp * 2] integer keypoint coordinates (x, y)
StructuredBuffer<int>   pattern;               // [256 * 4] (xa, ya, xb, yb), each component in [-15, 15]
StructuredBuffer<int>   dims;                  // [3] (W, H, N_kp)

RWStructuredBuffer<int> descriptors;           // [N_kp * 8] int32 (256 bits, LSB-first)

[shader("compute")]
[numthreads(64, 1, 1)]
void brief_descriptor(uint3 tid : SV_DispatchThreadID)
{
    int idx = int(tid.x);
    int W = dims[0];
    int H = dims[1];
    int N = dims[2];
    if (idx >= N) { return; }

    int kx = kp_xy[idx * 2 + 0];
    int ky = kp_xy[idx * 2 + 1];

    // Keypoints whose 31×31 patch leaves the image get a zero descriptor (defensive; upstream
    // should already exclude them)
    if (kx < 15 || kx >= W - 15 || ky < 15 || ky >= H - 15) {
        [unroll]
        for (int w = 0; w < 8; w++) { descriptors[idx * 8 + w] = 0; }
        return;
    }

    // Orientation θ = atan2(m01, m10), first moments of a circular patch of radius 15 (ORB §3.2)
    const int R = 15;
    float m10 = 0.0;
    float m01 = 0.0;
    for (int dy = -R; dy <= R; dy++) {
        for (int dx = -R; dx <= R; dx++) {
            if (dx * dx + dy * dy > R * R) { continue; }
            float v = luma[(ky + dy) * W + (kx + dx)];
            m10 += float(dx) * v;
            m01 += float(dy) * v;
        }
    }
    float theta = atan2(m01, m10);
    float ct = cos(theta);
    float st = sin(theta);

    int outBase = idx * 8;
    [unroll]
    for (int w = 0; w < 8; w++) {
        int bits = 0;
        [unroll]
        for (int b = 0; b < 32; b++) {
            int p = w * 32 + b;
            float pxa = float(pattern[p * 4 + 0]);
            float pya = float(pattern[p * 4 + 1]);
            float pxb = float(pattern[p * 4 + 2]);
            float pyb = float(pattern[p * 4 + 3]);
            // Rotated offsets reach ~21px, exceeding the 15px margin, so clamp
            int xa = clamp(kx + int(round(pxa * ct - pya * st)), 0, W - 1);
            int ya = clamp(ky + int(round(pxa * st + pya * ct)), 0, H - 1);
            int xb = clamp(kx + int(round(pxb * ct - pyb * st)), 0, W - 1);
            int yb = clamp(ky + int(round(pxb * st + pyb * ct)), 0, H - 1);
            float la = luma[ya * W + xa];
            float lb = luma[yb * W + xb];
            if (la > lb) { bits = bits | (1 << b); }
        }
        descriptors[outBase + w] = bits;
    }
}
