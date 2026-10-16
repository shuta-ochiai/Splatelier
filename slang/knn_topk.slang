// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// K nearest neighbors (PocketGS Operator I, Sec. III-B 1). 1 thread / query scans all points
// linearly and keeps an ascending array (K+1 slots) in registers via insertion sort (fewer branches
// than a heap on SIMT). Slot 0 is treated as self (d² = 0); slots 1..K are output (same convention
// as the host's knnIndicesAndDistances).

// Must match Swift GaussianModel.knnMaxK. A compile-time constant, so the array stays in registers.
static const int MAX_K = 16;

StructuredBuffer<float> positions;      // [N * 3] row-major xyz
StructuredBuffer<int>   knn_dims;       // [2] = (N, K), K <= MAX_K

RWStructuredBuffer<int>   outIndices;   // [N * K] excluding self, ascending distance
RWStructuredBuffer<float> outDistances; // [N * K] distance (d, not d²)

[shader("compute")]
[numthreads(64, 1, 1)]
void knn_topk(uint3 tid : SV_DispatchThreadID)
{
    int i = int(tid.x);
    int N = knn_dims[0];
    int K = knn_dims[1];
    if (i >= N) { return; }
    // Defensive; the caller is expected to guarantee this
    if (K > MAX_K) { K = MAX_K; }

    float pix = positions[i * 3 + 0];
    float piy = positions[i * 3 + 1];
    float piz = positions[i * 3 + 2];

    float bestD2[MAX_K + 1];
    int   bestIdx[MAX_K + 1];
    [unroll]
    for (int s = 0; s < MAX_K + 1; ++s) {
        bestD2[s]  = 3.402823e+38; // float max
        bestIdx[s] = -1;
    }

    for (int j = 0; j < N; ++j) {
        float dx = positions[j * 3 + 0] - pix;
        float dy = positions[j * 3 + 1] - piy;
        float dz = positions[j * 3 + 2] - piz;
        float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 >= bestD2[K]) { continue; }
        // Shift only on strict `>`: ties keep the smaller j first, matching the host's stable
        // argsort
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

    for (int k = 0; k < K; ++k) {
        outIndices[i * K + k]   = bestIdx[k + 1];
        outDistances[i * K + k] = sqrt(max(0.0, bestD2[k + 1]));
    }
}
