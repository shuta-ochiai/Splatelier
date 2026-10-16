// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Removes radial lens distortion (`LensUndistortion.undistortImage` in LensDistortion.swift). Maps
// each pixel center of the output (undistorted, intrinsics `dst`) back to normalized coordinates,
// distorts it with x_d = x_u·(1 + k1 r² + k2 r⁴) to find the position in the input (distorted,
// intrinsics `src`), and samples 3 channels with Catmull-Rom bicubic (a = −0.5). Pixel centers are
// at integer + 0.5 in continuous coordinates. Out-of-range taps use the edge pixel (clamp).

StructuredBuffer<float> image;      // [H * W * 3] input (distorted)
StructuredBuffer<float> params;     // [src fx, fy, cx, cy, dst fx, fy, cx, cy, k1, k2]
StructuredBuffer<int>   dims;       // [W, H]

RWStructuredBuffer<float> undistorted;  // [H * W * 3]

float4 catmullRomWeights(float t)
{
    float t2 = t * t;
    float t3 = t2 * t;
    return float4(0.5 * (-t3 + 2.0 * t2 - t),
                  0.5 * (3.0 * t3 - 5.0 * t2 + 2.0),
                  0.5 * (-3.0 * t3 + 4.0 * t2 + t),
                  0.5 * (t3 - t2));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void lens_undistort(uint3 tid : SV_DispatchThreadID)
{
    int imageWidth  = dims[0];
    int imageHeight = dims[1];
    int x = int(tid.x);
    int y = int(tid.y);
    if (x >= imageWidth || y >= imageHeight) { return; }

    float xu = (float(x) + 0.5 - params[6]) / params[4];
    float yu = (float(y) + 0.5 - params[7]) / params[5];
    float r2 = xu * xu + yu * yu;
    float f = 1.0 + params[8] * r2 + params[9] * r2 * r2;
    // Input index space (pixel centers at integers).
    float ui = params[0] * xu * f + params[2] - 0.5;
    float vi = params[1] * yu * f + params[3] - 0.5;
    float x0 = floor(ui);
    float y0 = floor(vi);
    float4 wx = catmullRomWeights(ui - x0);
    float4 wy = catmullRomWeights(vi - y0);

    float3 acc = float3(0.0, 0.0, 0.0);
    [unroll]
    for (int j = 0; j < 4; j++) {
        int sy = clamp(int(y0) + j - 1, 0, imageHeight - 1);
        float3 row = float3(0.0, 0.0, 0.0);
        [unroll]
        for (int i = 0; i < 4; i++) {
            int sx = clamp(int(x0) + i - 1, 0, imageWidth - 1);
            int p = (sy * imageWidth + sx) * 3;
            row += wx[i] * float3(image[p], image[p + 1], image[p + 2]);
        }
        acc += wy[j] * row;
    }
    int o = (y * imageWidth + x) * 3;
    undistorted[o]     = acc.x;
    undistorted[o + 1] = acc.y;
    undistorted[o + 2] = acc.z;
}
