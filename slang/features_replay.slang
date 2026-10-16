// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Composites a per-Gaussian 3-vector (e.g. a normal) at every `stride`-th pixel from the replay cache
// of a forward pass (rasterize_fwd_replay: the composited fragments of each pixel, front to back),
// with alpha recomputed exactly as that forward did, over a zero background. Writes only the sampled
// pixels (i s + s/2, j s + s/2) of a full-size [H * W * 3] output (the rest keeps its initial value),
// so `rasterize_bwd_replay` can run on it with cotangents set only there.

static const int REPLAY_CACHE_COUNT_MASK = 0x3FFFFFFF;

StructuredBuffer<float> means2d;      // [N, 2]
StructuredBuffer<float> conics;       // [N, 3]
StructuredBuffer<half>  features;     // [N, 3] (fp16, as rasterize_bwd_replay reads them)
StructuredBuffer<float> opacities;    // [N]
StructuredBuffer<int>   dims;         // [W, H, stride]
StructuredBuffer<int>   cache_id;     // [K_max, H, W]
StructuredBuffer<int>   cache_count;  // [H, W]

RWStructuredBuffer<float> rendered;   // [H * W * 3]

[shader("compute")]
[numthreads(16, 16, 1)]
void features_replay(uint3 tid : SV_DispatchThreadID)
{
    int imageWidth  = dims[0];
    int imageHeight = dims[1];
    int stride      = dims[2];
    if (int(tid.x) >= imageWidth / stride || int(tid.y) >= imageHeight / stride) { return; }
    int pixel_x = int(tid.x) * stride + stride / 2;
    int pixel_y = int(tid.y) * stride + stride / 2;
    int pix_idx = pixel_y * imageWidth + pixel_x;
    int plane = imageWidth * imageHeight;

    float px = float(pixel_x) + 0.5;
    float py = float(pixel_y) + 0.5;
    int K = cache_count[pix_idx] & REPLAY_CACHE_COUNT_MASK;
    float T = 1.0;
    float3 accum = float3(0.0, 0.0, 0.0);
    for (int k = 0; k < K; k++) {
        int gi = cache_id[k * plane + pix_idx];
        float dx = px - means2d[gi * 2 + 0];
        float dy = py - means2d[gi * 2 + 1];
        float sigma = 0.5 * (conics[gi * 3 + 0] * dx * dx + conics[gi * 3 + 2] * dy * dy)
            + conics[gi * 3 + 1] * dx * dy;
        float alpha = min(0.99, opacities[gi] * exp(-sigma));
        accum += T * alpha * float3(float(features[gi * 3 + 0]), float(features[gi * 3 + 1]),
                                    float(features[gi * 3 + 2]));
        T *= 1.0 - alpha;
    }
    rendered[pix_idx * 3 + 0] = accum.x;
    rendered[pix_idx * 3 + 1] = accum.y;
    rendered[pix_idx * 3 + 2] = accum.z;
}
