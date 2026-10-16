// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Multi-view depth consistency check (the per-pixel test of CostVolumeMVS.fuseWithConsistency).
// Lifts each valid target pixel to world, reprojects it into neighbor frames' depth maps, and
// accepts it if at least vMin neighbors agree within relative error < rel_tol. Output is a
// per-pixel depth (-1 if rejected/invalid; with averageDepth, the mean of the agreeing neighbors'
// depths converted to target depth). The caller builds point positions from this depth in pixel
// order (so the output order matches the CPU version).
//
// Frames are referenced by "slot": depths is the concatenation of per-slot depth maps; frame_info /
// frame_dims are per slot. grid is (max width, max height, number of targets).

StructuredBuffer<float> depths;         // Depth maps of all slots concatenated (slot s has w*h values starting at frame_dims[s*3+2])
StructuredBuffer<float> frame_info;     // [S*16] per slot: top 3 rows of viewmat (row-major 12) + (fx, fy, cx, cy)
StructuredBuffer<int>   frame_dims;     // [S*3] per slot: (w, h, start offset in depths)
StructuredBuffer<int>   targets;        // [T*2] per target: (slot, output start offset)
StructuredBuffer<int>   neighbors;      // [T*K] neighbor slots per target (-1 = none)
StructuredBuffer<int>   params;         // [3] (K, vMin, averageDepth)
StructuredBuffer<float> rel_tol_buf;    // [1]

RWStructuredBuffer<float> out_depth;    // [Σ target w*h]

[shader("compute")]
[numthreads(16, 16, 1)]
void mvs_fusion_consistency(uint3 tid : SV_DispatchThreadID)
{
    int t = int(tid.z);
    int ts = targets[t * 2 + 0];
    int outBase = targets[t * 2 + 1];
    int W = frame_dims[ts * 3 + 0];
    int H = frame_dims[ts * 3 + 1];
    int x = int(tid.x);
    int y = int(tid.y);
    if (x >= W || y >= H) { return; }
    int pix = y * W + x;

    float d = depths[frame_dims[ts * 3 + 2] + pix];
    if (d <= 0.0) {
        out_depth[outBase + pix] = -1.0;
        return;
    }

    int K = params[0];
    int vMin = params[1];
    bool averageDepth = params[2] != 0;
    float relTol = rel_tol_buf[0];

    // Target pixel to world: world = Rᵀ(cam - t)
    int ti = ts * 16;
    float fx = frame_info[ti + 12];
    float fy = frame_info[ti + 13];
    float cx = frame_info[ti + 14];
    float cy = frame_info[ti + 15];
    float xc = (float(x) + 0.5 - cx) / fx * d;
    float yc = (float(y) + 0.5 - cy) / fy * d;
    float dx = xc - frame_info[ti + 3];
    float dy = yc - frame_info[ti + 7];
    float dz = d - frame_info[ti + 11];
    float wx = frame_info[ti + 0] * dx + frame_info[ti + 4] * dy + frame_info[ti + 8] * dz;
    float wy = frame_info[ti + 1] * dx + frame_info[ti + 5] * dy + frame_info[ti + 9] * dz;
    float wz = frame_info[ti + 2] * dx + frame_info[ti + 6] * dy + frame_info[ti + 10] * dz;

    int conf = 0;
    float depthSum = d;
    for (int k = 0; k < K; k++) {
        int ns = neighbors[t * K + k];
        if (ns < 0) { continue; }
        int ni = ns * 16;
        float cz = frame_info[ni + 8] * wx + frame_info[ni + 9] * wy + frame_info[ni + 10] * wz + frame_info[ni + 11];
        if (cz <= 1e-6) { continue; }
        float cxx = frame_info[ni + 0] * wx + frame_info[ni + 1] * wy + frame_info[ni + 2] * wz + frame_info[ni + 3];
        float cyy = frame_info[ni + 4] * wx + frame_info[ni + 5] * wy + frame_info[ni + 6] * wz + frame_info[ni + 7];
        float nfx = frame_info[ni + 12];
        float nfy = frame_info[ni + 13];
        float ncx = frame_info[ni + 14];
        float ncy = frame_info[ni + 15];
        // Pixel center = i + 0.5 convention, so index is floor. Test "in range" so NaN is rejected
        // too.
        float fu = floor(nfx * cxx / cz + ncx);
        float fv = floor(nfy * cyy / cz + ncy);
        int nw = frame_dims[ns * 3 + 0];
        int nh = frame_dims[ns * 3 + 1];
        if (!(fu >= 0.0 && fu < float(nw) && fv >= 0.0 && fv < float(nh))) { continue; }
        int ux = int(fu);
        int vy = int(fv);
        float dj = depths[frame_dims[ns * 3 + 2] + vy * nw + ux];
        if (dj <= 0.0) { continue; }
        if (abs(cz - dj) / cz < relTol) {
            conf += 1;
            if (averageDepth) {
                // Lift the neighbor pixel (ux, vy) to world with depth dj and convert to target
                // depth z
                float ex = (float(ux) + 0.5 - ncx) / nfx * dj - frame_info[ni + 3];
                float ey = (float(vy) + 0.5 - ncy) / nfy * dj - frame_info[ni + 7];
                float ez = dj - frame_info[ni + 11];
                float px = frame_info[ni + 0] * ex + frame_info[ni + 4] * ey + frame_info[ni + 8] * ez;
                float py = frame_info[ni + 1] * ex + frame_info[ni + 5] * ey + frame_info[ni + 9] * ez;
                float pz = frame_info[ni + 2] * ex + frame_info[ni + 6] * ey + frame_info[ni + 10] * ez;
                depthSum += frame_info[ti + 8] * px + frame_info[ti + 9] * py + frame_info[ti + 10] * pz + frame_info[ti + 11];
            }
        }
    }
    float result = -1.0;
    if (conf >= vMin) {
        result = averageDepth ? depthSum / float(conf + 1) : d;
    }
    out_depth[outBase + pix] = result;
}
