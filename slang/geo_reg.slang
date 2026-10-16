// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Per-Gaussian parts of the geometric regularization (GeometryRegularization.swift), one thread per
// Gaussian:
//   geo_normals_fwd / geo_normals_bwd: the camera-space normal of the plane a Gaussian's ray-wise
//     peaks lie on under the local affine approximation (RaDe-GS), n ∝ Σ⁻¹ d = R S⁻² Rᵀ d with
//     d = μ - campos, turned toward the camera (the turn is a constant of the step).
//   gof_frame_bwd: the gradient of the ray-space frame of `gof_prepare` (M = S⁻¹Rᵀ,
//     o_g = M (campos - μ)) chained to means, quats and scales.
// Gradients reach the quaternion through `quat_to_rotmat_grad` (normalization included).

import gs_math;

StructuredBuffer<float> means;      // [N, 3]
StructuredBuffer<float> quats;      // [N, 4] (w, x, y, z), unnormalized
StructuredBuffer<float> scales;     // [N, 3] activated
StructuredBuffer<float> viewmat;    // [16] row-major world→camera
StructuredBuffer<int>   count;      // [1] N
StructuredBuffer<float> v_normals;  // [N, 3] dL/d(camera-space normal)
StructuredBuffer<float> v_frames;   // [N, 12] dL/d(M rows, o_g)

RWStructuredBuffer<float> normals;  // [N, 3]
RWStructuredBuffer<float> d_means;  // [N, 3]
RWStructuredBuffer<float> d_quats;  // [N, 4]
RWStructuredBuffer<float> d_scales; // [N, 3]

// Camera center -Rcᵀ t.
float3 cameraCenter(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7,
                    float v8, float v9, float v10, float v11)
{
    return -float3(v0 * v3 + v4 * v7 + v8 * v11, v1 * v3 + v5 * v7 + v9 * v11, v2 * v3 + v6 * v7 + v10 * v11);
}

float3 rotate(Mat3 R, float3 x)
{
    return float3(R.r00 * x.x + R.r01 * x.y + R.r02 * x.z,
                  R.r10 * x.x + R.r11 * x.y + R.r12 * x.z,
                  R.r20 * x.x + R.r21 * x.y + R.r22 * x.z);
}

float3 rotateT(Mat3 R, float3 x)
{
    return float3(R.r00 * x.x + R.r10 * x.y + R.r20 * x.z,
                  R.r01 * x.x + R.r11 * x.y + R.r21 * x.z,
                  R.r02 * x.x + R.r12 * x.y + R.r22 * x.z);
}

// R += u vᵀ.
Mat3 addOuter(Mat3 R, float3 u, float3 v)
{
    R.r00 += u.x * v.x; R.r01 += u.x * v.y; R.r02 += u.x * v.z;
    R.r10 += u.y * v.x; R.r11 += u.y * v.y; R.r12 += u.y * v.z;
    R.r20 += u.z * v.x; R.r21 += u.z * v.y; R.r22 += u.z * v.z;
    return R;
}

Mat3 zeroMat3()
{
    Mat3 Z;
    Z.r00 = 0.0; Z.r01 = 0.0; Z.r02 = 0.0;
    Z.r10 = 0.0; Z.r11 = 0.0; Z.r12 = 0.0;
    Z.r20 = 0.0; Z.r21 = 0.0; Z.r22 = 0.0;
    return Z;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void geo_normals_fwd(uint3 tid : SV_DispatchThreadID)
{
    int i = int(tid.x);
    if (i >= count[0]) { return; }
    Mat3 R = quat_to_rotmat(quats[i * 4 + 0], quats[i * 4 + 1], quats[i * 4 + 2], quats[i * 4 + 3]);
    float3 s = float3(scales[i * 3 + 0], scales[i * 3 + 1], scales[i * 3 + 2]);
    float3 c = cameraCenter(viewmat[0], viewmat[1], viewmat[2], viewmat[3], viewmat[4], viewmat[5],
                            viewmat[6], viewmat[7], viewmat[8], viewmat[9], viewmat[10], viewmat[11]);
    float3 d = float3(means[i * 3 + 0], means[i * 3 + 1], means[i * 3 + 2]) - c;
    float3 b = rotateT(R, d) / max(s * s, float3(1e-20));
    float3 n = rotate(R, b);
    float len = sqrt(max(dot(n, n), 1e-30));
    float flip = dot(n, d) > 0.0 ? -1.0 : 1.0;
    float3 nw = n * (flip / len);
    normals[i * 3 + 0] = viewmat[0] * nw.x + viewmat[1] * nw.y + viewmat[2] * nw.z;
    normals[i * 3 + 1] = viewmat[4] * nw.x + viewmat[5] * nw.y + viewmat[6] * nw.z;
    normals[i * 3 + 2] = viewmat[8] * nw.x + viewmat[9] * nw.y + viewmat[10] * nw.z;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void geo_normals_bwd(uint3 tid : SV_DispatchThreadID)
{
    int i = int(tid.x);
    if (i >= count[0]) { return; }
    float qw = quats[i * 4 + 0], qx = quats[i * 4 + 1], qy = quats[i * 4 + 2], qz = quats[i * 4 + 3];
    Mat3 R = quat_to_rotmat(qw, qx, qy, qz);
    float3 s = float3(scales[i * 3 + 0], scales[i * 3 + 1], scales[i * 3 + 2]);
    float3 c = cameraCenter(viewmat[0], viewmat[1], viewmat[2], viewmat[3], viewmat[4], viewmat[5],
                            viewmat[6], viewmat[7], viewmat[8], viewmat[9], viewmat[10], viewmat[11]);
    float3 d = float3(means[i * 3 + 0], means[i * 3 + 1], means[i * 3 + 2]) - c;
    float3 a = rotateT(R, d);
    float3 s2 = max(s * s, float3(1e-20));
    float3 b = a / s2;
    float3 n = rotate(R, b);
    float len = sqrt(max(dot(n, n), 1e-30));
    float flip = dot(n, d) > 0.0 ? -1.0 : 1.0;
    float3 nhat = n / len;

    // Camera → world: the view rotation transposed.
    float3 vc = float3(v_normals[i * 3 + 0], v_normals[i * 3 + 1], v_normals[i * 3 + 2]);
    float3 vw = float3(viewmat[0] * vc.x + viewmat[4] * vc.y + viewmat[8] * vc.z,
                       viewmat[1] * vc.x + viewmat[5] * vc.y + viewmat[9] * vc.z,
                       viewmat[2] * vc.x + viewmat[6] * vc.y + viewmat[10] * vc.z) * flip;
    // Normalization.
    float3 vn = (vw - nhat * dot(nhat, vw)) / len;
    // n = R b, b = a / s², a = Rᵀ d.
    Mat3 dR = addOuter(zeroMat3(), vn, b);
    float3 vb = rotateT(R, vn);
    float3 va = vb / s2;
    float3 ds = -2.0 * vb * a / (s2 * max(s, float3(1e-10)));
    dR = addOuter(dR, d, va);   // ∂(Rᵀ d)_k / ∂R_ik = d_i
    float3 vd = rotate(R, va);
    QuatGrad gq = quat_to_rotmat_grad(qw, qx, qy, qz, dR);
    d_means[i * 3 + 0] = vd.x; d_means[i * 3 + 1] = vd.y; d_means[i * 3 + 2] = vd.z;
    d_quats[i * 4 + 0] = gq.w; d_quats[i * 4 + 1] = gq.x; d_quats[i * 4 + 2] = gq.y; d_quats[i * 4 + 3] = gq.z;
    d_scales[i * 3 + 0] = ds.x; d_scales[i * 3 + 1] = ds.y; d_scales[i * 3 + 2] = ds.z;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void gof_frame_bwd(uint3 tid : SV_DispatchThreadID)
{
    int i = int(tid.x);
    if (i >= count[0]) { return; }
    float qw = quats[i * 4 + 0], qx = quats[i * 4 + 1], qy = quats[i * 4 + 2], qz = quats[i * 4 + 3];
    Mat3 R = quat_to_rotmat(qw, qx, qy, qz);
    float3 inv = 1.0 / max(float3(scales[i * 3 + 0], scales[i * 3 + 1], scales[i * 3 + 2]), float3(1e-9));
    float3 c = cameraCenter(viewmat[0], viewmat[1], viewmat[2], viewmat[3], viewmat[4], viewmat[5],
                            viewmat[6], viewmat[7], viewmat[8], viewmat[9], viewmat[10], viewmat[11]);
    float3 p = c - float3(means[i * 3 + 0], means[i * 3 + 1], means[i * 3 + 2]);
    int o = i * 12;
    // Rows of M = S⁻¹Rᵀ: row r is column r of R over s_r.
    float3 m0 = float3(R.r00, R.r10, R.r20) * inv.x;
    float3 m1 = float3(R.r01, R.r11, R.r21) * inv.y;
    float3 m2 = float3(R.r02, R.r12, R.r22) * inv.z;
    float3 vog = float3(v_frames[o + 9], v_frames[o + 10], v_frames[o + 11]);
    // o_g = M p adds vog ⊗ p to dL/dM; dL/dp = Mᵀ vog, p = campos - μ.
    float3 v0 = float3(v_frames[o + 0], v_frames[o + 1], v_frames[o + 2]) + vog.x * p;
    float3 v1 = float3(v_frames[o + 3], v_frames[o + 4], v_frames[o + 5]) + vog.y * p;
    float3 v2 = float3(v_frames[o + 6], v_frames[o + 7], v_frames[o + 8]) + vog.z * p;
    float3 vp = vog.x * m0 + vog.y * m1 + vog.z * m2;
    // M[r][c] = R[c][r] / s_r.
    Mat3 dR;
    dR.r00 = v0.x * inv.x; dR.r10 = v0.y * inv.x; dR.r20 = v0.z * inv.x;
    dR.r01 = v1.x * inv.y; dR.r11 = v1.y * inv.y; dR.r21 = v1.z * inv.y;
    dR.r02 = v2.x * inv.z; dR.r12 = v2.y * inv.z; dR.r22 = v2.z * inv.z;
    float3 ds = -float3(dot(v0, m0) * inv.x, dot(v1, m1) * inv.y, dot(v2, m2) * inv.z);
    QuatGrad gq = quat_to_rotmat_grad(qw, qx, qy, qz, dR);
    d_means[i * 3 + 0] = -vp.x; d_means[i * 3 + 1] = -vp.y; d_means[i * 3 + 2] = -vp.z;
    d_quats[i * 4 + 0] = gq.w; d_quats[i * 4 + 1] = gq.x; d_quats[i * 4 + 2] = gq.y; d_quats[i * 4 + 3] = gq.z;
    d_scales[i * 3 + 0] = ds.x; d_scales[i * 3 + 1] = ds.y; d_scales[i * 3 + 2] = ds.z;
}
