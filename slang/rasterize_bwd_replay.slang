// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Backward rasterize using the replay cache (PocketGS Operator T, Sec. III-C eq. 8). The cache
// holds only cache_id in front-to-back (FTF) order; the accumulated color accum and α are
// reconstructed here. α must be computed with exactly the same formula and operation order as
// forward (so accum and the cull test match bit for bit). The walk goes front to back (k = 0 →
// K-1), with v_R *= (1-α) after each iteration. A reverse walk is wrong.
//   R_{k+1} = (render_color - accum_after_k) / T_{k+1}
//  v_alpha_k = v_R · (c_k - R_{k+1}) + T_final · ra · v_render_alpha, v_color_k = α_k · v_R
// The background contribution is absorbed into R_{k+1}, so there is no background input
// (algebraically equivalent to rasterize_bwd).

// cache_count: lower 30 bits are K, bit 30 is the overflow flag. Must match
// rasterize_fwd_replay.slang and Swift GaussianRenderer.replayCacheCountMask /
// replayCacheOverflowBit. Forgetting the mask makes K huge, reading the cache out of range and
// crashing the GPU.
static const int REPLAY_CACHE_COUNT_MASK   = 0x3FFFFFFF;
static const int REPLAY_CACHE_OVERFLOW_BIT = 0x40000000;

StructuredBuffer<float> means2d;          // [N, 2]
StructuredBuffer<float> conics;           // [N, 3]
StructuredBuffer<half> colors;           // [N, 3]
StructuredBuffer<float> opacities;        // [N]
StructuredBuffer<int>   dims;             // [W, H, tileSize, numTilesX, K_max]
StructuredBuffer<float> render_color;     // [H * W * 3]  forward output (including background)
StructuredBuffer<float> render_alpha;     // [H * W]
StructuredBuffer<int>   cache_id;         // [K_max, H, W] Gaussian indices in FTF order
StructuredBuffer<int>   cache_count;      // [H, W]
StructuredBuffer<float> v_render_color;   // dL/d(render_color)
StructuredBuffer<float> v_render_alpha;   // dL/d(render_alpha)

RWStructuredBuffer<float> v_means2d;      // [N, 2]
RWStructuredBuffer<float> v_conics;       // [N, 3]
RWStructuredBuffer<float> v_colors;       // [N, 3]
RWStructuredBuffer<float> v_opacities;    // [N]

[shader("compute")]
[numthreads(16, 16, 1)]
void rasterize_bwd_replay(uint3 tid : SV_DispatchThreadID)
{
    int imageWidth  = dims[0];
    int imageHeight = dims[1];
    // tileSize / numTilesX are unused (the cache is indexed by pixel)
    int K_max       = dims[4];

    int pixel_x = int(tid.x);
    int pixel_y = int(tid.y);
    if (pixel_x >= imageWidth || pixel_y >= imageHeight) { return; }

    int pix_idx = pixel_y * imageWidth + pixel_x;
    int pixel_plane_size = imageWidth * imageHeight;

    // The overflow flag is for measurement only, so drop it
    int K = cache_count[pix_idx] & REPLAY_CACHE_COUNT_MASK;
    if (K <= 0) { return; }

    float px = float(pixel_x) + 0.5;
    float py = float(pixel_y) + 0.5;

    float T_final = 1.0 - render_alpha[pix_idx];
    float vra_pixel = v_render_alpha[pix_idx];
    // Nothing to propagate (losses defined on a subset of the pixels, e.g. the geometric
    // regularization's samples).
    if (vra_pixel == 0.0 && v_render_color[pix_idx * 3 + 0] == 0.0 && v_render_color[pix_idx * 3 + 1] == 0.0
        && v_render_color[pix_idx * 3 + 2] == 0.0) { return; }

    float render_r = render_color[pix_idx * 3 + 0];
    float render_g = render_color[pix_idx * 3 + 1];
    float render_b = render_color[pix_idx * 3 + 2];

    // v_R = T_k · v_render (T_0 = 1)
    float v_R_r = v_render_color[pix_idx * 3 + 0];
    float v_R_g = v_render_color[pix_idx * 3 + 1];
    float v_R_b = v_render_color[pix_idx * 3 + 2];

    float T = 1.0;
    // Accumulated color up to the previous step
    float accum_r = 0.0;
    float accum_g = 0.0;
    float accum_b = 0.0;

    for (int k = 0; k < K; k++) {
        int slot = k * pixel_plane_size + pix_idx;
        int gi = cache_id[slot];

        float2 mean2d = float2(means2d[gi * 2 + 0], means2d[gi * 2 + 1]);
        float3 cnc = float3(conics[gi * 3 + 0], conics[gi * 3 + 1], conics[gi * 3 + 2]);
        float ca = cnc.x;
        float cb = cnc.y;
        float cc = cnc.z;
        float opac = opacities[gi];
        float3 g_color = float3(colors[gi * 3 + 0], colors[gi * 3 + 1], colors[gi * 3 + 2]);
        float gr = g_color.x;
        float gg = g_color.y;
        float gb = g_color.z;

        float dx = px - mean2d.x;
        float dy = py - mean2d.y;
        float sigma = 0.5 * (ca * dx * dx + cc * dy * dy) + cb * dx * dy;
        float vis = exp(-sigma);
        float raw_alpha = opac * vis;
        // Same formula as forward. Anything cached already passed the sigma / 1/255 cull, so no
        // retest
        float alpha = min(0.99, raw_alpha);

        float weight = T * alpha;
        float accum_after_r = accum_r + weight * gr;
        float accum_after_g = accum_g + weight * gg;
        float accum_after_b = accum_b + weight * gb;
        float T_next = T * (1.0 - alpha);

        // R_{k+1} (the paper's BTF "C_in"). Forward stops at T < 1e-4, so the floor is only
        // defensive
        float T_next_safe = max(T_next, 1e-6);
        float R_next_r = (render_r - accum_after_r) / T_next_safe;
        float R_next_g = (render_g - accum_after_g) / T_next_safe;
        float R_next_b = (render_b - accum_after_b) / T_next_safe;

        float ra = 1.0 / max(1e-6, 1.0 - alpha);
        float v_alpha =
            v_R_r * (gr - R_next_r) +
            v_R_g * (gg - R_next_g) +
            v_R_b * (gb - R_next_b) +
            T_final * ra * vra_pixel;

        float v_color_r = alpha * v_R_r;
        float v_color_g = alpha * v_R_g;
        float v_color_b = alpha * v_R_b;
        InterlockedAdd(v_colors[gi * 3 + 0], v_color_r);
        InterlockedAdd(v_colors[gi * 3 + 1], v_color_g);
        InterlockedAdd(v_colors[gi * 3 + 2], v_color_b);

        // Gradient is 0 while clamped (raw_alpha > 0.99)
        if (raw_alpha <= 0.99) {
            float v_sigma = -opac * vis * v_alpha;
            InterlockedAdd(v_opacities[gi], vis * v_alpha);
            InterlockedAdd(v_conics[gi * 3 + 0], 0.5 * v_sigma * dx * dx);
            InterlockedAdd(v_conics[gi * 3 + 1], v_sigma * dx * dy);
            InterlockedAdd(v_conics[gi * 3 + 2], 0.5 * v_sigma * dy * dy);
            InterlockedAdd(v_means2d[gi * 2 + 0], -v_sigma * (ca * dx + cb * dy));
            InterlockedAdd(v_means2d[gi * 2 + 1], -v_sigma * (cb * dx + cc * dy));
        }

        T = T_next;
        accum_r = accum_after_r;
        accum_g = accum_after_g;
        accum_b = accum_after_b;
        v_R_r *= (1.0 - alpha);
        v_R_g *= (1.0 - alpha);
        v_R_b *= (1.0 - alpha);
    }
}
