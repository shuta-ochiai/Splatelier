// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Writes per-observation reprojection residuals (pred - obs) and camera-space z (for the BA in
// PocketGS Sec. III-A 2).

StructuredBuffer<float> viewmats;        // [num_cameras * 16] row-major world→camera
StructuredBuffer<float> intrinsics_buf;  // [num_cameras * 4] (fx, fy, cx, cy)
StructuredBuffer<float> points;          // [num_points * 3]
StructuredBuffer<int>   obs_cam_idx;     // [num_obs]
StructuredBuffer<int>   obs_pt_idx;      // [num_obs]
StructuredBuffer<float> obs_pixels;      // [num_obs * 2] (u, v)
StructuredBuffer<int>   res_dims;        // [1] num_obs

RWStructuredBuffer<float> residuals;     // [num_obs * 2], 0 if zc <= 1e-3
RWStructuredBuffer<float> z_cam;         // [num_obs], used by the caller to decide validity

[shader("compute")]
[numthreads(256, 1, 1)]
void ba_residual(uint3 tid : SV_DispatchThreadID)
{
    int i = int(tid.x);
    int N = res_dims[0];
    if (i >= N) { return; }

    int cam = obs_cam_idx[i];
    int pt = obs_pt_idx[i];
    int vmBase = cam * 16;
    int inBase = cam * 4;
    int ptBase = pt * 3;

    float fx = intrinsics_buf[inBase + 0];
    float fy = intrinsics_buf[inBase + 1];
    float cx = intrinsics_buf[inBase + 2];
    float cy = intrinsics_buf[inBase + 3];

    float Px = points[ptBase + 0];
    float Py = points[ptBase + 1];
    float Pz = points[ptBase + 2];

    // P_c = R · P + t  (row-major viewmat)
    float xc = viewmats[vmBase + 0] * Px + viewmats[vmBase + 1] * Py + viewmats[vmBase + 2]  * Pz + viewmats[vmBase + 3];
    float yc = viewmats[vmBase + 4] * Px + viewmats[vmBase + 5] * Py + viewmats[vmBase + 6]  * Pz + viewmats[vmBase + 7];
    float zc = viewmats[vmBase + 8] * Px + viewmats[vmBase + 9] * Py + viewmats[vmBase + 10] * Pz + viewmats[vmBase + 11];

    z_cam[i] = zc;
    if (zc <= 1e-3) {
        residuals[i * 2 + 0] = 0.0;
        residuals[i * 2 + 1] = 0.0;
        return;
    }

    float invZ = 1.0 / zc;
    float predU = fx * xc * invZ + cx;
    float predV = fy * yc * invZ + cy;
    residuals[i * 2 + 0] = predU - obs_pixels[i * 2 + 0];
    residuals[i * 2 + 1] = predV - obs_pixels[i * 2 + 1];
}
