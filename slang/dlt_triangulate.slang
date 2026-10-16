// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// DLT triangulation (1 thread / point). For each observation (u, v), accumulate A = [u·p3 - p1;
// v·p3 - p2] into Σ AᵀA, find the eigenvector of the smallest eigenvalue by the Jacobi method, and
// dehomogenize. With < 2 observations or degeneracy (w ≈ 0), outputs the input point unchanged
// (same as the CPU version). The CPU version accumulates in double, but float is enough given the
// few observations per point.

StructuredBuffer<float> viewmats;        // [num_cameras * 16] row-major 4x4
StructuredBuffer<float> intrinsics_buf;  // [num_cameras * 4] (fx, fy, cx, cy)
StructuredBuffer<int>   obs_cam_idx;     // [num_obs]
StructuredBuffer<float> obs_pixels;      // [num_obs * 2] (x, y)
StructuredBuffer<int>   obs_starts;      // [num_points] observations are sorted by point
StructuredBuffer<int>   obs_counts;      // [num_points]
StructuredBuffer<float> points_in;       // [num_points * 3] for fallback
StructuredBuffer<int>   dlt_dims;        // [1] : num_points

RWStructuredBuffer<float> points_out;    // [num_points * 3]

[shader("compute")]
[numthreads(64, 1, 1)]
void dlt_triangulate(uint3 tid : SV_DispatchThreadID)
{
    int j = int(tid.x);
    int N = dlt_dims[0];
    if (j >= N) { return; }

    points_out[j * 3 + 0] = points_in[j * 3 + 0];
    points_out[j * 3 + 1] = points_in[j * 3 + 1];
    points_out[j * 3 + 2] = points_in[j * 3 + 2];

    int count = obs_counts[j];
    if (count < 2) { return; }
    int start = obs_starts[j];

    // 4×4 symmetric AᵀA (row-major)
    float ATA[16];
    [unroll]
    for (int i = 0; i < 16; i++) { ATA[i] = 0.0; }

    for (int k = 0; k < count; k++) {
        int obs_idx = start + k;
        int cam = obs_cam_idx[obs_idx];
        float u = obs_pixels[obs_idx * 2 + 0];
        float v = obs_pixels[obs_idx * 2 + 1];

        int vmBase = cam * 16;
        int inBase = cam * 4;
        float fx = intrinsics_buf[inBase + 0];
        float fy = intrinsics_buf[inBase + 1];
        float cx = intrinsics_buf[inBase + 2];
        float cy = intrinsics_buf[inBase + 3];

        // The 3 rows p1, p2, p3 of K [R | t]
        float p1_0 = fx * viewmats[vmBase + 0] + cx * viewmats[vmBase + 8];
        float p1_1 = fx * viewmats[vmBase + 1] + cx * viewmats[vmBase + 9];
        float p1_2 = fx * viewmats[vmBase + 2] + cx * viewmats[vmBase + 10];
        float p1_3 = fx * viewmats[vmBase + 3] + cx * viewmats[vmBase + 11];
        float p2_0 = fy * viewmats[vmBase + 4] + cy * viewmats[vmBase + 8];
        float p2_1 = fy * viewmats[vmBase + 5] + cy * viewmats[vmBase + 9];
        float p2_2 = fy * viewmats[vmBase + 6] + cy * viewmats[vmBase + 10];
        float p2_3 = fy * viewmats[vmBase + 7] + cy * viewmats[vmBase + 11];
        float p3_0 = viewmats[vmBase + 8];
        float p3_1 = viewmats[vmBase + 9];
        float p3_2 = viewmats[vmBase + 10];
        float p3_3 = viewmats[vmBase + 11];

        float row0_0 = u * p3_0 - p1_0;
        float row0_1 = u * p3_1 - p1_1;
        float row0_2 = u * p3_2 - p1_2;
        float row0_3 = u * p3_3 - p1_3;
        float row1_0 = v * p3_0 - p2_0;
        float row1_1 = v * p3_1 - p2_1;
        float row1_2 = v * p3_2 - p2_2;
        float row1_3 = v * p3_3 - p2_3;

        // Jacobi reads and writes both halves, so fill all 16 elements even though it is symmetric
        ATA[0]  += row0_0 * row0_0 + row1_0 * row1_0;
        ATA[1]  += row0_0 * row0_1 + row1_0 * row1_1;
        ATA[2]  += row0_0 * row0_2 + row1_0 * row1_2;
        ATA[3]  += row0_0 * row0_3 + row1_0 * row1_3;
        ATA[4]  += row0_1 * row0_0 + row1_1 * row1_0;
        ATA[5]  += row0_1 * row0_1 + row1_1 * row1_1;
        ATA[6]  += row0_1 * row0_2 + row1_1 * row1_2;
        ATA[7]  += row0_1 * row0_3 + row1_1 * row1_3;
        ATA[8]  += row0_2 * row0_0 + row1_2 * row1_0;
        ATA[9]  += row0_2 * row0_1 + row1_2 * row1_1;
        ATA[10] += row0_2 * row0_2 + row1_2 * row1_2;
        ATA[11] += row0_2 * row0_3 + row1_2 * row1_3;
        ATA[12] += row0_3 * row0_0 + row1_3 * row1_0;
        ATA[13] += row0_3 * row0_1 + row1_3 * row1_1;
        ATA[14] += row0_3 * row0_2 + row1_3 * row1_2;
        ATA[15] += row0_3 * row0_3 + row1_3 * row1_3;
    }

    // Diagonalize with the Jacobi method (at most 60 sweeps, same as the CPU version)
    float V[16];
    [unroll]
    for (int i = 0; i < 16; i++) { V[i] = 0.0; }
    V[0]  = 1.0;
    V[5]  = 1.0;
    V[10] = 1.0;
    V[15] = 1.0;

    for (int sweep = 0; sweep < 60; sweep++) {
        // Pick the off-diagonal element with the largest absolute value
        float maxAbs = 0.0;
        int p = 0;
        int q = 1;

        float a01 = abs(ATA[1]);
        if (a01 > maxAbs) { maxAbs = a01; p = 0; q = 1; }
        float a02 = abs(ATA[2]);
        if (a02 > maxAbs) { maxAbs = a02; p = 0; q = 2; }
        float a03 = abs(ATA[3]);
        if (a03 > maxAbs) { maxAbs = a03; p = 0; q = 3; }
        float a12 = abs(ATA[6]);
        if (a12 > maxAbs) { maxAbs = a12; p = 1; q = 2; }
        float a13 = abs(ATA[7]);
        if (a13 > maxAbs) { maxAbs = a13; p = 1; q = 3; }
        float a23 = abs(ATA[11]);
        if (a23 > maxAbs) { maxAbs = a23; p = 2; q = 3; }

        if (maxAbs < 1e-14) { break; }

        float app = ATA[p * 4 + p];
        float aqq = ATA[q * 4 + q];
        float apq = ATA[p * 4 + q];
        float theta = (aqq - app) / (2.0 * apq);
        float t;
        if (abs(theta) < 1e10) {
            float signTheta = theta >= 0.0 ? 1.0 : -1.0;
            t = signTheta / (abs(theta) + sqrt(theta * theta + 1.0));
        } else {
            t = 1.0 / (2.0 * theta);
        }
        float c = 1.0 / sqrt(t * t + 1.0);
        float s = t * c;

        for (int i = 0; i < 4; i++) {
            if (i == p || i == q) { continue; }
            float aip = ATA[i * 4 + p];
            float aiq = ATA[i * 4 + q];
            float newAip = c * aip - s * aiq;
            float newAiq = s * aip + c * aiq;
            ATA[i * 4 + p] = newAip;
            ATA[p * 4 + i] = newAip;
            ATA[i * 4 + q] = newAiq;
            ATA[q * 4 + i] = newAiq;
        }
        ATA[p * 4 + p] = app - t * apq;
        ATA[q * 4 + q] = aqq + t * apq;
        ATA[p * 4 + q] = 0.0;
        ATA[q * 4 + p] = 0.0;

        // V ← V · R
        for (int i = 0; i < 4; i++) {
            float vip = V[i * 4 + p];
            float viq = V[i * 4 + q];
            V[i * 4 + p] = c * vip - s * viq;
            V[i * 4 + q] = s * vip + c * viq;
        }
    }

    int minIdx = 0;
    float minVal = ATA[0];
    if (ATA[5] < minVal)  { minVal = ATA[5];  minIdx = 1; }
    if (ATA[10] < minVal) { minVal = ATA[10]; minIdx = 2; }
    if (ATA[15] < minVal) { minVal = ATA[15]; minIdx = 3; }

    // w ≈ 0 is degenerate, so keep the input point
    float w = V[3 * 4 + minIdx];
    if (abs(w) < 1e-9) { return; }

    points_out[j * 3 + 0] = V[0 * 4 + minIdx] / w;
    points_out[j * 3 + 1] = V[1 * 4 + minIdx] / w;
    points_out[j * 3 + 2] = V[2 * 4 + minIdx] / w;
}
