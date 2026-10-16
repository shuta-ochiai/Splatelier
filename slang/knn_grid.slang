// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// K nearest neighbors via a grid (same result as the brute force in knn_topk.slang). The host sorts
// points by grid cell of side h, and each query visits cells in shells of Chebyshev distance r = 0,
// 1, 2, ... from its own cell. Once shell r is done, if the (K+1)-th distance is below r·h, points
// in cells farther out cannot qualify, so the result is final. Isolated points not finalized by
// r_max fall back to brute force over all points. Ranking is ascending by (distance², original
// index), which gives the same order as brute force (scanning by ascending index, keeping the
// earlier index first on ties). Slot 0 is self (distance 0); slots 1..K are output.

static const int MAX_K = 16;   // same as knn_topk.slang

StructuredBuffer<float> positions;     // [N*3] original order
StructuredBuffer<int>   query_cell;    // [N*3] cell coordinates of each point, original order
StructuredBuffer<float> sorted_pos;    // [N*3] points sorted by cell
StructuredBuffer<int>   sorted_orig;   // [N] original index of each point in cell order
StructuredBuffer<int>   cell_coords;   // [C*3] coordinates of non-empty cells (lexicographically ascending)
StructuredBuffer<int>   cell_start;    // [C+1] start offset of each cell in sorted_pos
StructuredBuffer<int>   grid_dims;     // [4] = (N, K, C, r_max)
StructuredBuffer<float> grid_h;        // [1] cell side length

RWStructuredBuffer<int>   outIndices;   // [N * K]
RWStructuredBuffer<float> outDistances; // [N * K] distance (d, not d²)

[shader("compute")]
[numthreads(64, 1, 1)]
void knn_grid(uint3 tid : SV_DispatchThreadID)
{
    int i = int(tid.x);
    int N = grid_dims[0];
    int K = grid_dims[1];
    int C = grid_dims[2];
    int rMax = grid_dims[3];
    if (i >= N) { return; }
    if (K > MAX_K) { K = MAX_K; }
    float h = grid_h[0];

    float pix = positions[i * 3 + 0];
    float piy = positions[i * 3 + 1];
    float piz = positions[i * 3 + 2];
    int qx = query_cell[i * 3 + 0];
    int qy = query_cell[i * 3 + 1];
    int qz = query_cell[i * 3 + 2];

    float bestD2[MAX_K + 1];
    int   bestIdx[MAX_K + 1];
    [unroll]
    for (int s = 0; s < MAX_K + 1; ++s) {
        bestD2[s]  = 3.402823e+38;
        bestIdx[s] = -1;
    }

    bool done = false;
    for (int r = 0; r <= rMax && !done; ++r) {
        for (int dx = -r; dx <= r; ++dx) {
            for (int dy = -r; dy <= r; ++dy) {
                for (int dz = -r; dz <= r; ++dz) {
                    if (max(abs(dx), max(abs(dy), abs(dz))) != r) { continue; }
                    // Find cell (x, y, z) by lexicographic binary search (inlined into the body
                    // because helper functions that read global buffers cannot be handled by the
                    // MLX conversion)
                    int x = qx + dx, y = qy + dy, z = qz + dz;
                    int lo = 0, hi = C - 1, c = -1;
                    while (lo <= hi) {
                        int mid = (lo + hi) >> 1;
                        int mx = cell_coords[mid * 3 + 0];
                        int my = cell_coords[mid * 3 + 1];
                        int mz = cell_coords[mid * 3 + 2];
                        int cmp = (mx != x) ? (mx < x ? -1 : 1)
                                : ((my != y) ? (my < y ? -1 : 1) : ((mz != z) ? (mz < z ? -1 : 1) : 0));
                        if (cmp == 0) { c = mid; break; }
                        if (cmp < 0) { lo = mid + 1; } else { hi = mid - 1; }
                    }
                    if (c < 0) { continue; }
                    for (int s = cell_start[c]; s < cell_start[c + 1]; ++s) {
                        float ex = sorted_pos[s * 3 + 0] - pix;
                        float ey = sorted_pos[s * 3 + 1] - piy;
                        float ez = sorted_pos[s * 3 + 2] - piz;
                        float d2 = ex * ex + ey * ey + ez * ez;
                        int j = sorted_orig[s];
                        if (d2 > bestD2[K] || (d2 == bestD2[K] && j > bestIdx[K])) { continue; }
                        int slot = K;
                        [unroll]
                        for (int hop = 0; hop < MAX_K; ++hop) {
                            if (slot <= 0) { break; }
                            if (bestD2[slot - 1] < d2 || (bestD2[slot - 1] == d2 && bestIdx[slot - 1] < j)) { break; }
                            bestD2[slot]  = bestD2[slot - 1];
                            bestIdx[slot] = bestIdx[slot - 1];
                            slot--;
                        }
                        bestD2[slot]  = d2;
                        bestIdx[slot] = j;
                    }
                }
            }
        }
        // Points outside shell r are at least r·h away (with a little slack for rounding in cell
        // assignment).
        float bound = float(r) * h * (1.0 - 1e-5);
        if (bestIdx[K] >= 0 && bestD2[K] < bound * bound) { done = true; }
    }

    if (!done) {
        // Isolated point: brute force (scan by ascending original index, same as knn_topk.slang)
        [unroll]
        for (int s = 0; s < MAX_K + 1; ++s) {
            bestD2[s]  = 3.402823e+38;
            bestIdx[s] = -1;
        }
        for (int j = 0; j < N; ++j) {
            float ex = positions[j * 3 + 0] - pix;
            float ey = positions[j * 3 + 1] - piy;
            float ez = positions[j * 3 + 2] - piz;
            float d2 = ex * ex + ey * ey + ez * ez;
            if (d2 >= bestD2[K]) { continue; }
            int slot = K;
            [unroll]
            for (int hop = 0; hop < MAX_K; ++hop) {
                if (slot <= 0) { break; }
                if (bestD2[slot - 1] <= d2) { break; }
                bestD2[slot]  = bestD2[slot - 1];
                bestIdx[slot] = bestIdx[slot - 1];
                slot--;
            }
            bestD2[slot]  = d2;
            bestIdx[slot] = j;
        }
    }

    for (int k = 0; k < K; ++k) {
        outIndices[i * K + k]   = bestIdx[k + 1];
        outDistances[i * K + k] = sqrt(max(0.0, bestD2[k + 1]));
    }
}
