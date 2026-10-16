// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Brute-force Hamming matcher for BRIEF-256 (PocketGS Appendix A.b). 1 thread / query. Ambiguous
// matches are -1 via the Lowe ratio test (Lowe 2004). Mutual-match check is done by the host with 2
// dispatches. If bandPx > 0, train points farther than bandPx from the normalized epipolar line are
// excluded before the Hamming computation (the host passes a wide band that absorbs ARKit pose
// noise).

StructuredBuffer<int> query_desc;      // [N_q * 8]
StructuredBuffer<int> train_desc;      // [N_t * 8]
StructuredBuffer<float> train_xy;      // [N_t * 2]  train keypoint pixel coordinates
StructuredBuffer<float> query_lines;   // [N_q * 3]  epipolar line F·[x,y,1] normalized by ‖(l0,l1)‖
StructuredBuffer<int> dims;            // [4] (N_q, N_t, ratio_q8, bandPx)
                                        // ratio_q8 = int(ratio * 256). 0.8 → 204

RWStructuredBuffer<int> best_idx;      // [N_q] (-1 if rejected by Lowe)
RWStructuredBuffer<int> best_dist;     // [N_q] (Hamming distance 0..256)

[shader("compute")]
[numthreads(64, 1, 1)]
void brief_match(uint3 tid : SV_DispatchThreadID)
{
    int q = int(tid.x);
    int Nq = dims[0];
    int Nt = dims[1];
    int ratioQ8 = dims[2];
    int bandPx = dims[3];
    if (q >= Nq) { return; }

    int q0 = query_desc[q * 8 + 0];
    int q1 = query_desc[q * 8 + 1];
    int q2 = query_desc[q * 8 + 2];
    int q3 = query_desc[q * 8 + 3];
    int q4 = query_desc[q * 8 + 4];
    int q5 = query_desc[q * 8 + 5];
    int q6 = query_desc[q * 8 + 6];
    int q7 = query_desc[q * 8 + 7];

    float ln0 = query_lines[q * 3 + 0];
    float ln1 = query_lines[q * 3 + 1];
    float ln2 = query_lines[q * 3 + 2];
    float band = float(bandPx);

    int d1 = 1000;    // minimum distance
    int d2 = 1000;    // second best
    int i1 = -1;
    for (int t = 0; t < Nt; t++) {
        if (bandPx > 0) {
            float tx = train_xy[t * 2 + 0];
            float ty = train_xy[t * 2 + 1];
            float dist = abs(ln0 * tx + ln1 * ty + ln2);
            if (dist > band) { continue; }
        }
        int t0 = train_desc[t * 8 + 0];
        int t1 = train_desc[t * 8 + 1];
        int t2 = train_desc[t * 8 + 2];
        int t3 = train_desc[t * 8 + 3];
        int t4 = train_desc[t * 8 + 4];
        int t5 = train_desc[t * 8 + 5];
        int t6 = train_desc[t * 8 + 6];
        int t7 = train_desc[t * 8 + 7];
        int d = countbits(q0 ^ t0) + countbits(q1 ^ t1)
              + countbits(q2 ^ t2) + countbits(q3 ^ t3)
              + countbits(q4 ^ t4) + countbits(q5 ^ t5)
              + countbits(q6 ^ t6) + countbits(q7 ^ t7);
        if (d < d1) {
            d2 = d1;
            d1 = d;
            i1 = t;
        } else if (d < d2) {
            d2 = d;
        }
    }
    // Lowe ratio test: evaluate d1 / d2 < ratioQ8 / 256 in integers
    if (d1 * 256 < d2 * ratioQ8) {
        best_idx[q] = i1;
        best_dist[q] = d1;
    } else {
        best_idx[q] = -1;
        best_dist[q] = d1;
    }
}
