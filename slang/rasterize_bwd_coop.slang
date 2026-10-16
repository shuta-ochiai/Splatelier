// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// SIMD-group cooperative backward rasterize. Same math as rasterize_bwd.slang (follows gsplat). All
// lanes of a SIMD-group process the same k (= the same Gaussian) together, sum gradients with
// WaveActiveSum, and only lane 0 does the atomic add (reduces atomic contention). WaveActiveSum
// needs every lane to reach it, so per-lane continue is forbidden. Skipping is only via a uniform
// WaveActiveAnyTrue test. Inactive lanes contribute 0.

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
void rasterize_bwd_coop(uint3 tid : SV_DispatchThreadID)
{
    int imageWidth  = dims[0];
    int imageHeight = dims[1];
    int tileSize    = dims[2];
    int numTilesX   = dims[3];

    int pixel_x = int(tid.x);
    int pixel_y = int(tid.y);
    // Do not return, so out-of-image lanes still take part in the Wave ops
    bool inside = (pixel_x < imageWidth) && (pixel_y < imageHeight);

    // Assumes tileSize = 16 (threadgroup width): tile_id is uniform within the threadgroup
    int tile_x = pixel_x / tileSize;
    int tile_y = pixel_y / tileSize;
    int tile_id = tile_y * numTilesX + tile_x;
    int isect_start = tile_offsets[tile_id];
    int isect_end   = tile_offsets[tile_id + 1];
    int isect_count = isect_end - isect_start;

    int pix_idx = inside ? (pixel_y * imageWidth + pixel_x) : 0;
    int last_local_idx = inside ? last_ids[pix_idx] : -1;
    if (isect_count <= 0) { last_local_idx = -1; }

    float px = float(pixel_x) + 0.5;
    float py = float(pixel_y) + 0.5;

    float T_final = inside ? (1.0 - render_alpha[pix_idx]) : 1.0;
    float T = T_final;
    float v_render_r = inside ? v_render_color[pix_idx * 3 + 0] : 0.0;
    float v_render_g = inside ? v_render_color[pix_idx * 3 + 1] : 0.0;
    float v_render_b = inside ? v_render_color[pix_idx * 3 + 2] : 0.0;
    float v_render_a = inside ? v_render_alpha[pix_idx] : 0.0;
    float bg_dot =
        background[0] * v_render_r +
        background[1] * v_render_g +
        background[2] * v_render_b;
    float buf_r = 0.0, buf_g = 0.0, buf_b = 0.0;

    // Loop bound is the max last_local within the SIMD-group (uniform)
    int warp_last = WaveActiveMax(last_local_idx);

    for (int k = warp_last; k >= 0; k--) {
        // gi is uniform within the SIMD-group
        bool inRange = (k < isect_count);
        int gi = inRange ? flatten_ids[isect_start + k] : 0;

        bool valid = inside && inRange && (k <= last_local_idx);
        float dx = 0.0, dy = 0.0, vis = 0.0, alpha = 0.0, opac = 0.0;
        float ca = 0.0, cb = 0.0, cc = 0.0;
        float gr = 0.0, gg = 0.0, gb = 0.0;
        float raw_alpha = 0.0;
        if (valid) {
            float2 mean2d = float2(means2d[gi * 2 + 0], means2d[gi * 2 + 1]);
            float3 cnc = float3(conics[gi * 3 + 0], conics[gi * 3 + 1], conics[gi * 3 + 2]);
            ca = cnc.x; cb = cnc.y; cc = cnc.z;
            opac = opacities[gi];
            dx = px - mean2d.x;
            dy = py - mean2d.y;
            float sigma = 0.5 * (ca * dx * dx + cc * dy * dy) + cb * dx * dy;
            if (sigma < 0.0) {
                valid = false;
            } else {
                vis = exp(-sigma);
                raw_alpha = opac * vis;
                alpha = min(0.99, raw_alpha);
                if (alpha < 0.00392157) { valid = false; }
            }
        }

        if (!WaveActiveAnyTrue(valid)) { continue; }

        float vc_r = 0.0, vc_g = 0.0, vc_b = 0.0;
        float vcon_x = 0.0, vcon_y = 0.0, vcon_z = 0.0;
        float vxy_x = 0.0, vxy_y = 0.0;
        float vop = 0.0;
        if (valid) {
            gr = colors[gi * 3 + 0]; gg = colors[gi * 3 + 1]; gb = colors[gi * 3 + 2];
            float ra = 1.0 / max(1e-6, 1.0 - alpha);
            T *= ra;
            float weight = T * alpha;

            vc_r = weight * v_render_r;
            vc_g = weight * v_render_g;
            vc_b = weight * v_render_b;

            float v_alpha =
                (gr * T - buf_r * ra) * v_render_r +
                (gg * T - buf_g * ra) * v_render_g +
                (gb * T - buf_b * ra) * v_render_b +
                T_final * ra * v_render_a -
                T_final * ra * bg_dot;

            if (raw_alpha <= 0.99) {
                float v_sigma = -opac * vis * v_alpha;
                vop = vis * v_alpha;
                vcon_x = 0.5 * v_sigma * dx * dx;
                vcon_y = v_sigma * dx * dy;
                vcon_z = 0.5 * v_sigma * dy * dy;
                vxy_x = -v_sigma * (ca * dx + cb * dy);
                vxy_y = -v_sigma * (cb * dx + cc * dy);
            }

            buf_r += weight * gr;
            buf_g += weight * gg;
            buf_b += weight * gb;
        }

        vc_r = WaveActiveSum(vc_r);
        vc_g = WaveActiveSum(vc_g);
        vc_b = WaveActiveSum(vc_b);
        vcon_x = WaveActiveSum(vcon_x);
        vcon_y = WaveActiveSum(vcon_y);
        vcon_z = WaveActiveSum(vcon_z);
        vxy_x = WaveActiveSum(vxy_x);
        vxy_y = WaveActiveSum(vxy_y);
        vop = WaveActiveSum(vop);

        if (WaveGetLaneIndex() == 0) {
            InterlockedAdd(v_colors[gi * 3 + 0], vc_r);
            InterlockedAdd(v_colors[gi * 3 + 1], vc_g);
            InterlockedAdd(v_colors[gi * 3 + 2], vc_b);
            InterlockedAdd(v_conics[gi * 3 + 0], vcon_x);
            InterlockedAdd(v_conics[gi * 3 + 1], vcon_y);
            InterlockedAdd(v_conics[gi * 3 + 2], vcon_z);
            InterlockedAdd(v_means2d[gi * 2 + 0], vxy_x);
            InterlockedAdd(v_means2d[gi * 2 + 1], vxy_y);
            InterlockedAdd(v_opacities[gi], vop);
        }
    }
}
