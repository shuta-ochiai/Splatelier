// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Depth filter using WTA depth + peak-ratio confidence (PocketGS Sec. III-A 3).
// confidence = 1 - best / second (PKRN, Hu & Mordohai 2012). Below threshold → -1.
// LR-consistency is not handled in this kernel.

StructuredBuffer<int>   agg_cost;        // [H * W * K] aggregated cost (SGM output)
StructuredBuffer<float> depths;          // [K] hypothesis depths
StructuredBuffer<int>   dims;            // [W, H, K]
StructuredBuffer<float> threshold_buf;   // [1] confidence threshold (paper uses 0.4)
StructuredBuffer<int>   exclude_radius_buf; // [1] PKRN second-min exclusion radius (in bins)

RWStructuredBuffer<float> depth_map;     // [H * W] accepted depth (-1 if filtered out)
RWStructuredBuffer<float> confidence_map;// [H * W] confidence (raw value before filtering)

[shader("compute")]
[numthreads(16, 16, 1)]
void depth_filter(uint3 tid : SV_DispatchThreadID)
{
    int W = dims[0];
    int H = dims[1];
    int K = dims[2];

    int x = int(tid.x);
    int y = int(tid.y);
    if (x >= W || y >= H) { return; }
    int pix = y * W + x;

    if (K <= 0) {
        depth_map[pix] = -1.0;
        confidence_map[pix] = 0.0;
        return;
    }

    // Pass 1: WTA
    int bestCost = agg_cost[pix * K + 0];
    int bestIdx = 0;
    for (int k = 1; k < K; k++) {
        int c = agg_cost[pix * K + k];
        if (c < bestCost) {
            bestCost = c;
            bestIdx = k;
        }
    }

    // Pass 2: second is the minimum excluding bestIdx ± excludeRadius. SGM turns the minimum into
    // a smooth valley, so without excluding adjacent bins the confidence collapses to 0.
    int excludeRadius = exclude_radius_buf[0];
    int secondCost = 2147483647;   // INT32_MAX
    for (int k = 0; k < K; k++) {
        int d = k - bestIdx;
        if (d < 0) { d = -d; }
        if (d <= excludeRadius) { continue; }
        int c = agg_cost[pix * K + k];
        if (c < secondCost) { secondCost = c; }
    }

    // confidence 0 if every bin is in the excluded range
    float conf = 0.0;
    if (secondCost < 2147483647 && secondCost > 0) {
        conf = 1.0 - float(bestCost) / float(secondCost);
    }
    confidence_map[pix] = conf;

    float thresh = threshold_buf[0];
    if (conf >= thresh) {
        // Sub-pixel refinement by parabola fit (Hirschmüller 2008 Sec. II-C, an addition not in the
        // paper). Hypotheses are evenly spaced in inverse depth, so interpolate in inverse depth
        // too. No refinement for winners at the search edge or denom <= 0 (integer ties /
        // non-convex).
        float chosenDepth = depths[bestIdx];
        if (K >= 3 && bestIdx > 0 && bestIdx < K - 1) {
            float cm = float(agg_cost[pix * K + bestIdx - 1]);
            float c0 = float(bestCost);
            float cp = float(agg_cost[pix * K + bestIdx + 1]);
            float denom = cm - 2.0 * c0 + cp;
            if (denom > 0.0) {
                float delta = (cm - cp) / (2.0 * denom);
                delta = clamp(delta, -0.5, 0.5);
                float wBest = 1.0 / depths[bestIdx];
                float wNext = 1.0 / depths[bestIdx + 1];
                float wSub = wBest + delta * (wNext - wBest);
                chosenDepth = 1.0 / max(wSub, 1e-6);
            }
        }
        depth_map[pix] = chosenDepth;
    } else {
        // -1 = reject (valid depths are always positive)
        depth_map[pix] = -1.0;
    }
}
