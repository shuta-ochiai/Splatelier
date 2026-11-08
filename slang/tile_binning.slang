// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// GPU tile binning: tile_aabb (AABB and intersection count) → exclusive cumsum in MLX →
// tile_emit_pairs (emit a key per intersection) → sort in MLX → tile_lower_bound (start offset per
// tile). In scanline mode (dims[4] = 1), tiles inside the AABB are scanned row by row (or column by
// column), counting only the tiles the ellipse covers within each band (same idea as
// count_ellipse_grid_overlaps in vksplat utils.slang). The more diagonal and elongated the
// Gaussian, the more empty tiles at the AABB corners are dropped, reducing sort / rasterize work.
// The sort key is 64-bit (tile_id, depth_bits), but Slang's int64 is unreliable, so it is emitted
// as 2 × 32 bits and combined on the Swift/MLX side. Integer comparison of depth_bits matches depth
// order only when z > 0 (sign bit = 0).

// ---- Scanline (ellipse × band) ------------------------------------------------
// rasterize discards opac·exp(-Q/2) < 4/255 for Q = a dx² + 2b dx dy + c dy² (conic), so
// contributing pixels lie inside the ellipse Q <= rr = 2·log(255/4·opac). Capped at 3.33² like
// radii.
// Pixel centers are 0.5 px inside tile boundaries, so a range error under 0.5 px never drops a
// contributing tile.
float ellipse_rr(float opac)
{
    float extend = min(3.33, sqrt(2.0 * log(opac * 63.75)));
    return extend * extend;
}

// In coordinates relative to the center, returns the x range (xl, xr) of the intersection of
// ellipse Q <= rr and band y0 <= y <= y1. Empty if xl > xr. For elongated ellipses det = ac - b²
// suffers cancellation and the endpoints shift a lot, so in that case bound conservatively by "the
// range of the chord midpoint x = -b/a·y over the band ± the chord half-length upper bound
// sqrt(rr/a)" (since Q = a(x + b/a·y)² + det/a·y², |x + b/a·y| <= sqrt(rr/a) inside the ellipse).
float2 ellipse_x_range(float a, float b, float c, float rr, float y0, float y1)
{
    float det = a * c - b * b;
    float slope = -b / a;
    float hwMax = sqrt(rr / a);
    if (det > 1e-3 * a * c) {
        float ymax = sqrt(rr * a / det);
        if (y0 > ymax || y1 < -ymax) { return float2(1.0, -1.0); }
        // y of the point where x is maximal (same as vksplat ellipse_range_bound). The left end is
        // -ym by symmetry.
        float ym = -b / c * sqrt(rr * c / det);
        float v0 = clamp(-ym, y0, y1);
        float v1 = clamp(ym, y0, y1);
        float xl = slope * v0 - sqrt(max(0.0, a * rr - det * v0 * v0)) / a;
        float xr = slope * v1 + sqrt(max(0.0, a * rr - det * v1 * v1)) / a;
        // Absorbs sqrt rounding near the tangent (relative ~sqrt(eps)).
        float pad = 1e-3 * hwMax;
        return float2(xl - pad, xr + pad);
    }
    float m0 = slope * y0;
    float m1 = slope * y1;
    return float2(min(m0, m1) - hwMax, max(m0, m1) + hwMax);
}

// Range [lo, hi] of tiles the ellipse covers on one scanline (clamped to the AABB; hi < lo if
// empty). If rowMajor, scans tile row `line` and returns the x range; otherwise scans tile column
// `line` and returns the y range (a column is treated as a row of the ellipse with x and y
// swapped). The band is trimmed to within the radius (ry or rx).
int2 scanline_tile_range(
    int line, bool rowMajor, float px, float py, float ca, float cb, float cc, float rr,
    int rx, int ry, int tileSize, int xMin, int yMin, int xMax, int yMax)
{
    float u    = rowMajor ? px : py;
    float w    = rowMajor ? py : px;
    float qa   = rowMajor ? ca : cc;
    float qc   = rowMajor ? cc : ca;
    float wLim = float(rowMajor ? ry : rx);
    int lo = rowMajor ? xMin : yMin;
    int hi = rowMajor ? xMax : yMax;

    float t = float(tileSize);
    float w0 = max(float(line) * t - w, -wLim);
    float w1 = min(float(line) * t + t - w, wLim);
    float2 r = ellipse_x_range(qa, cb, qc, rr, w0, w1);
    if (!(r.x <= r.y)) { return int2(1, 0); }
    int t0 = int(floor((r.x + u) / t));
    int t1 = int(floor((r.y + u) / t));
    return int2(max(t0, lo), min(t1, hi));
}

// ---- 1. AABB + intersection count ---------------------------------------------

StructuredBuffer<float> means2d;        // [N, 2]  (px, py)
StructuredBuffer<int>   radii;          // [N, 2]  (rx, ry)
StructuredBuffer<float> depths;         // [N]
StructuredBuffer<float> conics;         // [N, 3]  (a, b, c), read only in scanline mode
StructuredBuffer<float> opacities;      // [N]     after sigmoid, read only in scanline mode
StructuredBuffer<int>   aabb_dims;      // [5] : numTilesX, numTilesY, tileSize, N, scanline mode (0/1)

RWStructuredBuffer<int> aabb;           // [N, 4] : xMin, yMin, xMax, yMax (invalid if xMax < xMin)
RWStructuredBuffer<int> aabb_count;     // [N]    : number of intersected tiles, 0 if invalid

[shader("compute")]
[numthreads(256, 1, 1)]
void tile_aabb(uint3 tid : SV_DispatchThreadID)
{
    int i = int(tid.x);
    int N = aabb_dims[3];
    if (i >= N) { return; }

    int numTilesX = aabb_dims[0];
    int numTilesY = aabb_dims[1];
    int tileSize  = aabb_dims[2];

    int rx = radii[i * 2 + 0];
    int ry = radii[i * 2 + 1];
    float d = depths[i];

    if (rx <= 0 || ry <= 0 || d <= 0.0) {
        aabb[i * 4 + 0] =  1;
        aabb[i * 4 + 1] =  1;
        aabb[i * 4 + 2] = -1;
        aabb[i * 4 + 3] = -1;
        aabb_count[i]   =  0;
        return;
    }

    float px = means2d[i * 2 + 0];
    float py = means2d[i * 2 + 1];
    float bxMin = px - float(rx);
    float bxMax = px + float(rx);
    float byMin = py - float(ry);
    float byMax = py + float(ry);

    int tileXMin = int(floor(bxMin / float(tileSize)));
    int tileXMax = int(floor(bxMax / float(tileSize)));
    int tileYMin = int(floor(byMin / float(tileSize)));
    int tileYMax = int(floor(byMax / float(tileSize)));

    tileXMin = max(0, tileXMin);
    tileXMax = min(numTilesX - 1, tileXMax);
    tileYMin = max(0, tileYMin);
    tileYMax = min(numTilesY - 1, tileYMax);

    if (tileXMax < tileXMin || tileYMax < tileYMin) {
        aabb[i * 4 + 0] =  1;
        aabb[i * 4 + 1] =  1;
        aabb[i * 4 + 2] = -1;
        aabb[i * 4 + 3] = -1;
        aabb_count[i]   =  0;
        return;
    }

    aabb[i * 4 + 0] = tileXMin;
    aabb[i * 4 + 1] = tileYMin;
    aabb[i * 4 + 2] = tileXMax;
    aabb[i * 4 + 3] = tileYMax;

    if (aabb_dims[4] == 0) {
        aabb_count[i] = (tileXMax - tileXMin + 1) * (tileYMax - tileYMin + 1);
        return;
    }
    // Scan along the shorter side (fewer loop iterations).
    float ca = conics[i * 3 + 0];
    float cb = conics[i * 3 + 1];
    float cc = conics[i * 3 + 2];
    float rr = ellipse_rr(opacities[i]);
    bool rowMajor = (tileYMax - tileYMin) <= (tileXMax - tileXMin);
    int first = rowMajor ? tileYMin : tileXMin;
    int last  = rowMajor ? tileYMax : tileXMax;
    int count = 0;
    for (int line = first; line <= last; line++) {
        int2 r = scanline_tile_range(line, rowMajor, px, py, ca, cb, cc, rr, rx, ry, tileSize,
                                     tileXMin, tileYMin, tileXMax, tileYMax);
        count += max(0, r.y - r.x + 1);
    }
    aabb_count[i] = count;
}

// ---- 2. Emit (tile_id, sort_key, gauss_idx) -----------------------------------
// sort_key = (tile_id << depthBits) | quantized depth, packed into a non-negative int32 within 31
// bits (the host picks depthBits so that tileBits + depthBits = 31). Sorting a single int32 halves
// the sort bandwidth compared to a 64-bit key. The quantized depth is the bit pattern of a positive
// float (integer order = value order) counted from near, right-shifted by depthShift. Relative
// precision stays constant regardless of depth units.

StructuredBuffer<int>   aabb_e;         // [N, 4] output of tile_aabb
StructuredBuffer<int>   aabb_count_e;   // [N]
StructuredBuffer<int>   starts;         // [N] exclusive prefix sum of aabb_count
StructuredBuffer<float> depths_e;       // [N]
StructuredBuffer<int>   emit_dims;      // [6] : numTilesX, N, depthBits, depthShift, asint(near), asint(far)
// Inputs for recounting the same ranges as tile_aabb in scanline mode (aabb_dims_e is the same
// array as tile_aabb's aabb_dims).
StructuredBuffer<float> means2d_e;      // [N, 2]
StructuredBuffer<int>   radii_e;        // [N, 2]
StructuredBuffer<float> conics_e;       // [N, 3]
StructuredBuffer<float> opacities_e;    // [N]
StructuredBuffer<int>   aabb_dims_e;    // [5]

RWStructuredBuffer<int> tile_ids;       // [n_isects]
RWStructuredBuffer<int> sort_keys;      // [n_isects] : (tile_id << depthBits) | quantized depth
RWStructuredBuffer<int> gauss_indices;  // [n_isects]

[shader("compute")]
[numthreads(256, 1, 1)]
void tile_emit_pairs(uint3 tid : SV_DispatchThreadID)
{
    int i = int(tid.x);
    int N = emit_dims[1];
    if (i >= N) { return; }

    int count = aabb_count_e[i];
    if (count == 0) { return; }

    int numTilesX = emit_dims[0];
    int xMin = aabb_e[i * 4 + 0];
    int yMin = aabb_e[i * 4 + 1];
    int xMax = aabb_e[i * 4 + 2];
    int yMax = aabb_e[i * 4 + 3];
    int base = starts[i];
    int depthBits = emit_dims[2];
    int depthShift = emit_dims[3];
    int nearBits = emit_dims[4];
    int farBits = emit_dims[5];
    // Projection already culled near/far so this should be in range, but clamp to respect the key's
    // bit width.
    int dBits = clamp(asint(depths_e[i]), nearBits, farBits);
    int dq = (dBits - nearBits) >> depthShift;
    int k = 0;
    if (aabb_dims_e[4] == 0) {
        for (int ty = yMin; ty <= yMax; ty++) {
            for (int tx = xMin; tx <= xMax; tx++) {
                int slot = base + k;
                int tileId = ty * numTilesX + tx;
                tile_ids[slot]      = tileId;
                sort_keys[slot]     = (tileId << depthBits) | dq;
                gauss_indices[slot] = i;
                k++;
            }
        }
        return;
    }

    // Recount the range with the same functions as tile_aabb. Writes are capped at count entries.
    // (Helper functions that touch global buffers cannot be converted by SlangCompiledKernel, so
    // the writes are inlined)
    int tileSize = aabb_dims_e[2];
    int rx = radii_e[i * 2 + 0];
    int ry = radii_e[i * 2 + 1];
    float px = means2d_e[i * 2 + 0];
    float py = means2d_e[i * 2 + 1];
    float ca = conics_e[i * 3 + 0];
    float cb = conics_e[i * 3 + 1];
    float cc = conics_e[i * 3 + 2];
    float rr = ellipse_rr(opacities_e[i]);
    bool rowMajor = (yMax - yMin) <= (xMax - xMin);
    int first = rowMajor ? yMin : xMin;
    int last  = rowMajor ? yMax : xMax;
    for (int line = first; line <= last; line++) {
        int2 r = scanline_tile_range(line, rowMajor, px, py, ca, cb, cc, rr, rx, ry, tileSize,
                                     xMin, yMin, xMax, yMax);
        for (int j = r.x; j <= r.y && k < count; j++) {
            int slot = base + k;
            int tileId = rowMajor ? line * numTilesX + j : j * numTilesX + line;
            tile_ids[slot]      = tileId;
            sort_keys[slot]     = (tileId << depthBits) | dq;
            gauss_indices[slot] = i;
            k++;
        }
    }
    // Safety net for when float rounding differs between the 2 kernels and the recount falls short
    // of count. To leave no uninitialized slots, fill with out-of-range tiles inside the AABB
    // (extra tiles do not change the render).
    int lo = rowMajor ? xMin : yMin;
    int hi = rowMajor ? xMax : yMax;
    for (int line = first; line <= last && k < count; line++) {
        int2 r = scanline_tile_range(line, rowMajor, px, py, ca, cb, cc, rr, rx, ry, tileSize,
                                     xMin, yMin, xMax, yMax);
        for (int j = lo; j <= hi && k < count; j++) {
            if (j >= r.x && j <= r.y) { continue; }
            int slot = base + k;
            int tileId = rowMajor ? line * numTilesX + j : j * numTilesX + line;
            tile_ids[slot]      = tileId;
            sort_keys[slot]     = (tileId << depthBits) | dq;
            gauss_indices[slot] = i;
            k++;
        }
    }
}

// ---- 3. tile lower_bound -------------------------------------------------------
// Binary-search the lower_bound of tile t in the sorted tile_ids and write it to tile_offsets[t].
// This kernel also writes tile_offsets[numTiles] = n_isects.

StructuredBuffer<int> sorted_tile_ids;   // [n_isects]
StructuredBuffer<int> lb_dims;           // [2] : n_isects, numTiles

RWStructuredBuffer<int> tile_offsets;    // [numTiles + 1]

[shader("compute")]
[numthreads(128, 1, 1)]
void tile_lower_bound(uint3 tid : SV_DispatchThreadID)
{
    int t = int(tid.x);
    int numTiles = lb_dims[1];
    if (t > numTiles) { return; }

    int nIsects = lb_dims[0];
    if (t == numTiles) {
        tile_offsets[t] = nIsects;
        return;
    }

    int lo = 0;
    int hi = nIsects;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (sorted_tile_ids[mid] < t) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    tile_offsets[t] = lo;
}

// ---- 4. Split 16-px tiles into 8-px tiles --------------------------------------
// Sorting the pairs is the costly part of binning, and 8-px tiles have about 2.3× the pairs of
// 16-px ones; 8-px tiles rasterize at about 60% of the cost, since a small Gaussian's tile then
// costs 64 pixels instead of 256. So the pairs are sorted for 16-px tiles and each tile's list is
// split into its four 8-px quadrants, in order (the depth order carries over). One threadgroup
// takes one 16-px tile: tile_split_count counts the pairs of each quadrant, MLX turns the counts
// into list offsets (a cumsum over the 8-px tiles), and tile_split_write walks the list again in
// chunks of 256 and writes each pair to the quadrants it covers, ranked by a prefix sum within the
// chunk. A pair covers a quadrant when the ellipse reaches one of its 8-px rows' tile columns (the
// same ellipse-and-band test as the scanline count).

static const int SPLIT_GROUP = 256;

// 4-bit mask of the quadrants (bit qy·2 + qx) of 16-px tile (tx, ty) that a Gaussian covers.
int quadrant_mask(float px, float py, int rx, int ry, float ca, float cb, float cc, float opac,
                  int tx, int ty, int numTilesX8, int numTilesY8)
{
    float rr = ellipse_rr(opac);
    int xMin = max(0, int(floor((px - float(rx)) / 8.0)));
    int xMax = min(numTilesX8 - 1, int(floor((px + float(rx)) / 8.0)));
    int yMin = max(0, int(floor((py - float(ry)) / 8.0)));
    int yMax = min(numTilesY8 - 1, int(floor((py + float(ry)) / 8.0)));
    int mask = 0;
    [unroll] for (int qy = 0; qy < 2; ++qy) {
        int row = ty * 2 + qy;
        if (row < yMin || row > yMax) { continue; }
        int2 r = scanline_tile_range(row, true, px, py, ca, cb, cc, rr, rx, ry, 8, xMin, yMin, xMax, yMax);
        [unroll] for (int qx = 0; qx < 2; ++qx) {
            int col = tx * 2 + qx;
            if (col >= r.x && col <= r.y) { mask |= 1 << (qy * 2 + qx); }
        }
    }
    return mask;
}

StructuredBuffer<int>   sc_flatten;      // [n16] Gaussian index per pair, sorted
StructuredBuffer<int>   sc_offsets;      // [numTiles16 + 1]
StructuredBuffer<float> sc_means2d;      // [N, 2]
StructuredBuffer<int>   sc_radii;        // [N, 2]
StructuredBuffer<float> sc_conics;       // [N, 3]
StructuredBuffer<float> sc_opacities;    // [N]
StructuredBuffer<int>   sc_dims;         // [3] : numTilesX16, numTilesX8, numTilesY8

RWStructuredBuffer<int> sc_counts;       // [numTiles16, 4] pairs per quadrant

groupshared int sc_partial[(SPLIT_GROUP / 32) * 4];

[shader("compute")]
[numthreads(256, 1, 1)]
void tile_split_count(uint3 gid : SV_GroupID, uint lidx : SV_GroupIndex)
{
    int t = int(gid.x);
    int numTilesX16 = sc_dims[0];
    int tx = t % numTilesX16, ty = t / numTilesX16;
    int start = sc_offsets[t], end = sc_offsets[t + 1];
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    for (int k = start + int(lidx); k < end; k += SPLIT_GROUP) {
        int gi = sc_flatten[k];
        int m = quadrant_mask(sc_means2d[gi * 2], sc_means2d[gi * 2 + 1], sc_radii[gi * 2], sc_radii[gi * 2 + 1],
                              sc_conics[gi * 3], sc_conics[gi * 3 + 1], sc_conics[gi * 3 + 2], sc_opacities[gi],
                              tx, ty, sc_dims[1], sc_dims[2]);
        c0 += m & 1; c1 += (m >> 1) & 1; c2 += (m >> 2) & 1; c3 += (m >> 3) & 1;
    }
    c0 = WaveActiveSum(c0); c1 = WaveActiveSum(c1); c2 = WaveActiveSum(c2); c3 = WaveActiveSum(c3);
    uint simd = lidx / 32u;
    if (WaveGetLaneIndex() == 0u) {
        sc_partial[simd * 4 + 0] = c0; sc_partial[simd * 4 + 1] = c1;
        sc_partial[simd * 4 + 2] = c2; sc_partial[simd * 4 + 3] = c3;
    }
    GroupMemoryBarrierWithGroupSync();
    if (lidx < 4u) {
        int sum = 0;
        for (int s = 0; s < SPLIT_GROUP / 32; ++s) { sum += sc_partial[s * 4 + int(lidx)]; }
        sc_counts[t * 4 + int(lidx)] = sum;
    }
}

StructuredBuffer<int>   sw_flatten;      // [n16]
StructuredBuffer<int>   sw_offsets;      // [numTiles16 + 1]
StructuredBuffer<float> sw_means2d;
StructuredBuffer<int>   sw_radii;
StructuredBuffer<float> sw_conics;
StructuredBuffer<float> sw_opacities;
StructuredBuffer<int>   sw_offsets8;     // [numTiles8 + 1] start of each 8-px tile's list
StructuredBuffer<int>   sw_dims;         // [3] : numTilesX16, numTilesX8, numTilesY8

RWStructuredBuffer<int> sw_flatten8;     // [4 · n16] (upper bound; sw_offsets8 tells the used part)

groupshared int sw_simd_total[(SPLIT_GROUP / 32) * 4];

[shader("compute")]
[numthreads(256, 1, 1)]
void tile_split_write(uint3 gid : SV_GroupID, uint lidx : SV_GroupIndex)
{
    int t = int(gid.x);
    int numTilesX16 = sw_dims[0], numTilesX8 = sw_dims[1];
    int tx = t % numTilesX16, ty = t / numTilesX16;
    int start = sw_offsets[t], end = sw_offsets[t + 1];
    // Write position of the next pair of each quadrant.
    int base[4];
    [unroll] for (int q = 0; q < 4; ++q) {
        base[q] = sw_offsets8[(ty * 2 + q / 2) * numTilesX8 + tx * 2 + q % 2];
    }
    uint simd = lidx / 32u;
    for (int chunk = start; chunk < end; chunk += SPLIT_GROUP) {
        int k = chunk + int(lidx);
        int gi = -1, m = 0;
        if (k < end) {
            gi = sw_flatten[k];
            m = quadrant_mask(sw_means2d[gi * 2], sw_means2d[gi * 2 + 1], sw_radii[gi * 2], sw_radii[gi * 2 + 1],
                              sw_conics[gi * 3], sw_conics[gi * 3 + 1], sw_conics[gi * 3 + 2], sw_opacities[gi],
                              tx, ty, sw_dims[1], sw_dims[2]);
        }
        int rank[4];
        [unroll] for (int q = 0; q < 4; ++q) {
            int bit = (m >> q) & 1;
            rank[q] = WavePrefixSum(bit);
            int total = WaveActiveSum(bit);
            if (WaveGetLaneIndex() == 0u) { sw_simd_total[simd * 4 + q] = total; }
        }
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (int q = 0; q < 4; ++q) {
            int before = 0, all = 0;
            for (int s = 0; s < SPLIT_GROUP / 32; ++s) {
                int v = sw_simd_total[s * 4 + q];
                if (s < int(simd)) { before += v; }
                all += v;
            }
            if (((m >> q) & 1) != 0) { sw_flatten8[base[q] + before + rank[q]] = gi; }
            base[q] += all;
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
