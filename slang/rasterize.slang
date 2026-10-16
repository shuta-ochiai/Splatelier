// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Per-tile forward rasterize (1 thread / pixel). Alpha-blends the tile's Gaussian list front to
// back and stops once transmittance drops below 1e-4. Follows gsplat RasterizeToPixels3DGSFwd.cu
// (Apache-2.0). The thresholds 0.99 / 1/255 / 1e-4 are also gsplat's. Backward reconstructs
// gradients by hand from last_ids and T_final.

StructuredBuffer<float> means2d;          // [N, 2]
StructuredBuffer<float> conics;           // [N, 3]  (a, b, c) of Σ_2d⁻¹
StructuredBuffer<half> colors;           // [N, 3]
StructuredBuffer<float> opacities;        // [N]
StructuredBuffer<int>   flatten_ids;      // [n_isects]  Gaussian index per (tile, entry)
StructuredBuffer<int>   tile_offsets;     // [num_tiles + 1]  start offset of each tile
StructuredBuffer<int>   dims;             // [imageWidth, imageHeight, tileSize, numTilesX]
StructuredBuffer<float> background;       // [3]

RWStructuredBuffer<float> render_color;   // [H * W * 3] background already composited
RWStructuredBuffer<float> render_alpha;   // [H * W] 1 - T_final
RWStructuredBuffer<int>   last_ids;       // [H * W] in-tile index of the last contributing Gaussian

[shader("compute")]
[numthreads(16, 16, 1)]
void rasterize_fwd(uint3 tid : SV_DispatchThreadID)
{
    int imageWidth  = dims[0];
    int imageHeight = dims[1];
    int tileSize    = dims[2];
    int numTilesX   = dims[3];

    int pixel_x = int(tid.x);
    int pixel_y = int(tid.y);
    if (pixel_x >= imageWidth || pixel_y >= imageHeight) {
        return;
    }

    int tile_x = pixel_x / tileSize;
    int tile_y = pixel_y / tileSize;
    int tile_id = tile_y * numTilesX + tile_x;
    int isect_start = tile_offsets[tile_id];
    int isect_end   = tile_offsets[tile_id + 1];

    float px = float(pixel_x) + 0.5;
    float py = float(pixel_y) + 0.5;

    float T = 1.0;
    float accum_r = 0.0, accum_g = 0.0, accum_b = 0.0;
    // -1 = no contribution (backward skips the whole loop)
    int last_local_idx = -1;
    bool done = false;

    for (int k = isect_start; k < isect_end; k++) {
        if (done) { break; }

        int gi = flatten_ids[k];
        float mx = means2d[gi * 2 + 0];
        float my = means2d[gi * 2 + 1];
        float ca = conics[gi * 3 + 0];
        float cb = conics[gi * 3 + 1];
        float cc = conics[gi * 3 + 2];

        float dx = px - mx;
        float dy = py - my;
        float sigma = 0.5 * (ca * dx * dx + cc * dy * dy) + cb * dx * dy;
        if (sigma < 0.0) { continue; }

        float opac = opacities[gi];
        float alpha = min(0.99, opac * exp(-sigma));
        if (alpha < 0.00392157) { continue; }  // 1/255

        float next_T = T * (1.0 - alpha);
        if (next_T < 1e-4) {
            done = true;
            break;
        }

        float weight = T * alpha;
        float3 c = float3(colors[gi * 3 + 0], colors[gi * 3 + 1], colors[gi * 3 + 2]);
        accum_r += weight * c.x;
        accum_g += weight * c.y;
        accum_b += weight * c.z;
        last_local_idx = k - isect_start;
        T = next_T;
    }

    int pix_idx = pixel_y * imageWidth + pixel_x;
    render_color[pix_idx * 3 + 0] = accum_r + T * background[0];
    render_color[pix_idx * 3 + 1] = accum_g + T * background[1];
    render_color[pix_idx * 3 + 2] = accum_b + T * background[2];
    render_alpha[pix_idx] = 1.0 - T;
    last_ids[pix_idx] = last_local_idx;
}
