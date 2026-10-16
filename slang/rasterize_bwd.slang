// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Per-tile backward rasterize (1 thread / pixel). Walks the same list as forward in reverse from
// last_ids, recovering transmittance with T_prev = T_next / (1 - alpha). Gradients are atomically
// added per Gaussian. Hand-written gradients following gsplat RasterizeToPixels3DGSBwd.cu
// (Apache-2.0).

StructuredBuffer<float> means2d;          // [N, 2]
StructuredBuffer<float> conics;           // [N, 3]
StructuredBuffer<half> colors;           // [N, 3]
StructuredBuffer<float> opacities;        // [N]
StructuredBuffer<int>   flatten_ids;
StructuredBuffer<int>   tile_offsets;
StructuredBuffer<int>   dims;             // [W, H, tileSize, numTilesX]
StructuredBuffer<float> background;       // [3]
StructuredBuffer<float> render_alpha;     // [H * W] forward output
StructuredBuffer<int>   last_ids;         // [H * W] forward output
StructuredBuffer<float> v_render_color;   // [H*W*3] dL/d(render_color)
StructuredBuffer<float> v_render_alpha;   // [H*W]   dL/d(render_alpha)

RWStructuredBuffer<float> v_means2d;      // [N, 2]
RWStructuredBuffer<float> v_conics;       // [N, 3]
RWStructuredBuffer<float> v_colors;       // [N, 3]
RWStructuredBuffer<float> v_opacities;    // [N]

[shader("compute")]
[numthreads(16, 16, 1)]
void rasterize_bwd(uint3 tid : SV_DispatchThreadID)
{
    int imageWidth  = dims[0];
    int imageHeight = dims[1];
    int tileSize    = dims[2];
    int numTilesX   = dims[3];

    int pixel_x = int(tid.x);
    int pixel_y = int(tid.y);
    if (pixel_x >= imageWidth || pixel_y >= imageHeight) { return; }

    int tile_x = pixel_x / tileSize;
    int tile_y = pixel_y / tileSize;
    int tile_id = tile_y * numTilesX + tile_x;
    int isect_start = tile_offsets[tile_id];
    int isect_end   = tile_offsets[tile_id + 1];
    int pix_idx = pixel_y * imageWidth + pixel_x;
    int last_local_idx = last_ids[pix_idx];

    float px = float(pixel_x) + 0.5;
    float py = float(pixel_y) + 0.5;

    float T_final = 1.0 - render_alpha[pix_idx];
    float T = T_final;
    float v_render_r = v_render_color[pix_idx * 3 + 0];
    float v_render_g = v_render_color[pix_idx * 3 + 1];
    float v_render_b = v_render_color[pix_idx * 3 + 2];
    float v_render_a = v_render_alpha[pix_idx];
    float bg_dot =
        background[0] * v_render_r +
        background[1] * v_render_g +
        background[2] * v_render_b;

    // Accumulated color contribution of Gaussians behind the current one (already passed in the
    // reverse walk)
    float buf_r = 0.0;
    float buf_g = 0.0;
    float buf_b = 0.0;

    // -1 = no contribution in forward
    if (last_local_idx < 0 || isect_end <= isect_start) { return; }

    for (int k = last_local_idx; k >= 0; k--) {
        int gi = flatten_ids[isect_start + k];

        float mx = means2d[gi * 2 + 0];
        float my = means2d[gi * 2 + 1];
        float ca = conics[gi * 3 + 0];
        float cb = conics[gi * 3 + 1];
        float cc = conics[gi * 3 + 2];
        float opac = opacities[gi];

        float dx = px - mx;
        float dy = py - my;
        float sigma = 0.5 * (ca * dx * dx + cc * dy * dy) + cb * dx * dy;
        if (sigma < 0.0) { continue; }

        float vis = exp(-sigma);
        float raw_alpha = opac * vis;
        float alpha = min(0.99, raw_alpha);
        if (alpha < 0.00392157) { continue; }

        float ra = 1.0 / max(1e-6, 1.0 - alpha);
        T *= ra;                                  // T just before this Gaussian
        float weight = T * alpha;

        float3 g_color = float3(colors[gi * 3 + 0], colors[gi * 3 + 1], colors[gi * 3 + 2]);
        float gr = g_color.x;
        float gg = g_color.y;
        float gb = g_color.z;

        float vc_r = weight * v_render_r;
        float vc_g = weight * v_render_g;
        float vc_b = weight * v_render_b;
        InterlockedAdd(v_colors[gi * 3 + 0], vc_r);
        InterlockedAdd(v_colors[gi * 3 + 1], vc_g);
        InterlockedAdd(v_colors[gi * 3 + 2], vc_b);

        // dL/d(alpha): 3 terms: own color T·c, attenuation of what is behind -buf·ra, and via
        // T_final (alpha and background)
        float v_alpha =
            (gr * T - buf_r * ra) * v_render_r +
            (gg * T - buf_g * ra) * v_render_g +
            (gb * T - buf_b * ra) * v_render_b +
            T_final * ra * v_render_a -
            T_final * ra * bg_dot;

        // alpha = min(0.99, opacity·exp(-sigma)). Gradient is 0 while clamped
        if (raw_alpha <= 0.99) {
            float v_sigma = -opac * vis * v_alpha;
            float v_opacity_g = vis * v_alpha;
            InterlockedAdd(v_opacities[gi], v_opacity_g);

            InterlockedAdd(v_conics[gi * 3 + 0], 0.5 * v_sigma * dx * dx);
            InterlockedAdd(v_conics[gi * 3 + 1], v_sigma * dx * dy);
            InterlockedAdd(v_conics[gi * 3 + 2], 0.5 * v_sigma * dy * dy);
            InterlockedAdd(v_means2d[gi * 2 + 0], -v_sigma * (ca * dx + cb * dy));
            InterlockedAdd(v_means2d[gi * 2 + 1], -v_sigma * (cb * dx + cc * dy));
        }

        buf_r += weight * gr;
        buf_g += weight * gg;
        buf_b += weight * gb;
    }
}
