// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// PCA normals from K nearest neighbors (PocketGS Operator I, Sec. III-B 1). 1 thread / point. Finds
// the smallest eigenvalue of the neighbor covariance in closed form (Smith 1961) and uses its
// eigenvector as the normal. Stays (0, 0, 1) when degenerate.

StructuredBuffer<float> positions;      // [N * 3] row-major
StructuredBuffer<int>   knIndices;      // [N * K] K_n neighbor indices
StructuredBuffer<int>   pca_dims;       // [2] = (N, K)

RWStructuredBuffer<float> normals;      // [N * 3]

[shader("compute")]
[numthreads(64, 1, 1)]
void pca_normals_3x3(uint3 tid : SV_DispatchThreadID)
{
    int i = int(tid.x);
    int N = pca_dims[0];
    int K = pca_dims[1];
    if (i >= N) { return; }

    // Default for degenerate cases. quatFromZToNormals turns it into the identity quaternion, so
    // harmless
    normals[i * 3 + 0] = 0.0;
    normals[i * 3 + 1] = 0.0;
    normals[i * 3 + 2] = 1.0;

    if (K < 2) { return; }

    // Centroid
    float cx = 0.0, cy = 0.0, cz = 0.0;
    for (int k = 0; k < K; k++) {
        int idx = knIndices[i * K + k];
        cx += positions[idx * 3 + 0];
        cy += positions[idx * 3 + 1];
        cz += positions[idx * 3 + 2];
    }
    float invK = 1.0 / float(K);
    cx *= invK; cy *= invK; cz *= invK;

    // Covariance (6 upper-triangular elements)
    float a00 = 0.0, a01 = 0.0, a02 = 0.0;
    float a11 = 0.0, a12 = 0.0, a22 = 0.0;
    for (int k = 0; k < K; k++) {
        int idx = knIndices[i * K + k];
        float dx = positions[idx * 3 + 0] - cx;
        float dy = positions[idx * 3 + 1] - cy;
        float dz = positions[idx * 3 + 2] - cz;
        a00 += dx * dx;
        a01 += dx * dy;
        a02 += dx * dz;
        a11 += dy * dy;
        a12 += dy * dz;
        a22 += dz * dz;
    }
    a00 *= invK; a01 *= invK; a02 *= invK;
    a11 *= invK; a12 *= invK; a22 *= invK;

    // Smallest eigenvalue (Smith 1961). Diagonal matrix if p1 ≈ 0
    float p1 = a01 * a01 + a02 * a02 + a12 * a12;

    float lambdaMin;
    if (p1 < 1e-20) {
        lambdaMin = min(a00, min(a11, a22));
    } else {
        float q = (a00 + a11 + a22) / 3.0;
        float dq0 = a00 - q;
        float dq1 = a11 - q;
        float dq2 = a22 - q;
        float p2 = dq0 * dq0 + dq1 * dq1 + dq2 * dq2 + 2.0 * p1;
        float p = sqrt(p2 / 6.0);

        // B = (A - q I) / p
        float b00 = dq0 / p, b11 = dq1 / p, b22 = dq2 / p;
        float b01 = a01 / p, b02 = a02 / p, b12 = a12 / p;
        float detB = b00 * (b11 * b22 - b12 * b12)
                   - b01 * (b01 * b22 - b12 * b02)
                   + b02 * (b01 * b12 - b11 * b02);
        // Rounding error can push it outside [-1, 1], so clamp
        float r = clamp(detB * 0.5, -1.0, 1.0);
        float phi = acos(r) / 3.0;
        // λ_min = q + 2p cos(φ + 2π/3)
        const float TWO_PI_3 = 2.0943951; // 2π/3
        lambdaMin = q + 2.0 * p * cos(phi + TWO_PI_3);
    }

    // Null space of (A - λ_min I) = normal. Use the largest-norm cross product of row pairs to
    // avoid dependent rows
    float m00 = a00 - lambdaMin;
    float m11 = a11 - lambdaMin;
    float m22 = a22 - lambdaMin;
    float3 r0 = float3(m00, a01, a02);
    float3 r1 = float3(a01, m11, a12);
    float3 r2 = float3(a02, a12, m22);

    float3 c01 = cross(r0, r1);
    float3 c02 = cross(r0, r2);
    float3 c12 = cross(r1, r2);

    float l01 = dot(c01, c01);
    float l02 = dot(c02, c02);
    float l12 = dot(c12, c12);

    float3 n;
    if (l01 >= l02 && l01 >= l12) { n = c01; }
    else if (l02 >= l12)          { n = c02; }
    else                          { n = c12; }

    float nLen2 = dot(n, n);
    if (nLen2 > 1e-18) {
        float invN = 1.0 / sqrt(nLen2);
        normals[i * 3 + 0] = n.x * invN;
        normals[i * 3 + 1] = n.y * invN;
        normals[i * 3 + 2] = n.z * invN;
    }
}
