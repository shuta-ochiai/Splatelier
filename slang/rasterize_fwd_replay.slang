// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Forward rasterize that writes the replay cache (PocketGS Operator T, Sec. III-C). Composites the
// same way as rasterize.slang, while writing only the indices of contributing Gaussians to cache_id
// in FTF order (backward reconstructs accumulated color and α). The cache is [K_max, H, W]:
// adjacent pixels are contiguous, so writes and reads coalesce. Stops at K_max (skipping instead
// would make render and cache disagree and break backward's T reconstruction).

// cache_count: lower 30 bits are K, bit 30 is the overflow flag. Must match
// rasterize_bwd_replay.slang and Swift GaussianRenderer.replayCacheCountMask /
// replayCacheOverflowBit. The flag is not an atomic in a separate buffer because zero-initializing
// it (MLX initValue) would also zero-fill the other huge output buffers.
static const int REPLAY_CACHE_COUNT_MASK   = 0x3FFFFFFF;
static const int REPLAY_CACHE_OVERFLOW_BIT = 0x40000000;

StructuredBuffer<float> means2d;          // [N, 2]
StructuredBuffer<float> conics;           // [N, 3]
StructuredBuffer<half> colors;           // [N, 3]
StructuredBuffer<float> opacities;        // [N]
StructuredBuffer<int>   flatten_ids;      // [n_isects]
StructuredBuffer<int>   tile_offsets;     // [num_tiles + 1]
StructuredBuffer<int>   dims;             // [W, H, tileSize, numTilesX, K_max]
StructuredBuffer<float> background;       // [3]

RWStructuredBuffer<float> render_color;   // [H * W * 3]
RWStructuredBuffer<float> render_alpha;   // [H * W]
RWStructuredBuffer<int>   cache_id;       // [K_max, H, W]
RWStructuredBuffer<int>   cache_count;    // [H, W]

[shader("compute")]
[numthreads(16, 16, 1)]
void rasterize_fwd_replay(uint3 tid : SV_DispatchThreadID)
{
    int imageWidth  = dims[0];
    int imageHeight = dims[1];
    int tileSize    = dims[2];
    int numTilesX   = dims[3];
    int K_max       = dims[4];

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
    int pix_idx = pixel_y * imageWidth + pixel_x;
    int pixel_plane_size = imageWidth * imageHeight;

    float px = float(pixel_x) + 0.5;
    float py = float(pixel_y) + 0.5;

    float T = 1.0;
    float accum_r = 0.0, accum_g = 0.0, accum_b = 0.0;
    int local_count = 0;
    bool done = false;
    bool overflowed = false;

    for (int k = isect_start; k < isect_end; k++) {
        if (done) { break; }

        int gi = flatten_ids[k];
        float2 mean2d = float2(means2d[gi * 2 + 0], means2d[gi * 2 + 1]);
        float3 cnc = float3(conics[gi * 3 + 0], conics[gi * 3 + 1], conics[gi * 3 + 2]);
        float ca = cnc.x;
        float cb = cnc.y;
        float cc = cnc.z;

        float dx = px - mean2d.x;
        float dy = py - mean2d.y;
        float sigma = 0.5 * (ca * dx * dx + cc * dy * dy) + cb * dx * dy;
        if (sigma < 0.0) { continue; }

        float opac = opacities[gi];
        float alpha = min(0.99, opac * exp(-sigma));
        if (alpha < 0.00392157) { continue; }  // 1/255

        float next_T = T * (1.0 - alpha);
        if (next_T < 1e-4) {
            // Same cutoff as rasterize.slang. This fragment is not composited
            done = true;
            break;
        }

        if (local_count >= K_max) {
            overflowed = true;
            done = true;
            break;
        }

        int slot = local_count * pixel_plane_size + pix_idx;
        cache_id[slot] = gi;
        local_count++;

        float weight = T * alpha;
        float3 c = float3(colors[gi * 3 + 0], colors[gi * 3 + 1], colors[gi * 3 + 2]);
        accum_r += weight * c.x;
        accum_g += weight * c.y;
        accum_b += weight * c.z;
        T = next_T;
    }

    render_color[pix_idx * 3 + 0] = accum_r + T * background[0];
    render_color[pix_idx * 3 + 1] = accum_g + T * background[1];
    render_color[pix_idx * 3 + 2] = accum_b + T * background[2];
    render_alpha[pix_idx] = 1.0 - T;
    cache_count[pix_idx] = local_count | (overflowed ? REPLAY_CACHE_OVERFLOW_BIT : 0);
}
