// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Surface depth of the Gaussian opacity field along each pixel ray. Implemented from the papers only
// (GOF: Yu et al., "Gaussian Opacity Fields", TOG 2024, Sec. 3.1-3.2; SOF: Radl et al., "Sorted
// Opacity Fields", SIGGRAPH Asia 2025, Sec. 3.2; G³S: Zhang et al., "Geometry-Grounded Gaussian
// Splatting", arXiv 2601.17835, Sec. 4.1-4.2 and App. A.1, H); no code from their repositories.
//
// Each Gaussian is evaluated in 3D along the ray (GOF eq. 2-7): with o_g = S⁻¹Rᵀ(o - μ) and
// d_g = S⁻¹Rᵀ d, the ray x = o + t·d sees the 1D Gaussian G(t) = exp(-½(A t² + 2B t + C)) with
// A = d_g·d_g, B = o_g·d_g, C = o_g·o_g, peaking at t* = -B / A. With g(t) = α G(t) (α the opacity,
// so g(t*) is the rasterizer's alpha), the ray transmittance is the product of one factor per
// Gaussian, independent of the blending order, and non-increasing in t:
//   profile 0 (GOF Sec. 3.2):  T_k(t) = 1 - g_k(min(t, t*_k)) (all attenuation before the peak)
//   profile 1 (G³S eq. 12):    T_k(t) = v_k(t) for t <= t*_k, v_k(t*_k)² / v_k(t) after it, with the
//                              stochastic-solid vacancy v = √(1 - g) (attenuation split around the
//                              peak, smooth in t)
// Both end at 1 - g_k(t*_k), so the color of 3DGS is unchanged. The depth is the 0.5 crossing of
// T(t), searched as in G³S App. A.1: each pass over the tile list measures T at 7 points splitting the
// interval into 8 (3 bisection steps per pass). The first pass spans ±3σ around the Gaussian that
// takes the product of final factors below 0.5, instead of a fixed interval. The view's opacity field is ≥ 0.5 exactly behind that
// depth, which is what `gof_carve` compares against.
//
// d is chosen with camera-space z = 1, so t is the camera-space depth.
//
// `gof_depth_bwd` differentiates that depth (G³S eq. 13, App. A.2): T(t_med; θ) = 0.5 gives
// dt_med/dθ_i = -(∂ log T_i/∂θ_i) / Σ_k ∂ log T_k/∂t, so one pass over the Gaussians at t_med sums
// the denominator and a second scatters every Gaussian's share. The gradient reaches each Gaussian
// as its ray-space frame (M, o_g) and opacity; the caller chains those to the parameters.

import gs_math;

// ---- gof_prepare: per Gaussian, per view -------------------------------------------------------

StructuredBuffer<float> means;        // [N, 3]
StructuredBuffer<float> quats;        // [N, 4] (w, x, y, z), unnormalized
StructuredBuffer<float> scales;       // [N, 3] activated
StructuredBuffer<float> viewmat;      // [16] row-major world→camera
StructuredBuffer<int>   prep_n;       // [1] N

RWStructuredBuffer<float> gauss_frame; // [N, 12] rows of M = S⁻¹Rᵀ (9), then o_g = M (o - μ) (3)
RWStructuredBuffer<float> z_extent;    // [N] 3σ of the Gaussian along the camera z axis

[shader("compute")]
[numthreads(64, 1, 1)]
void gof_prepare(uint3 tid : SV_DispatchThreadID)
{
    int i = int(tid.x);
    if (i >= prep_n[0]) { return; }

    Mat3 R = quat_to_rotmat(quats[i * 4 + 0], quats[i * 4 + 1], quats[i * 4 + 2], quats[i * 4 + 3]);
    float isx = 1.0 / max(scales[i * 3 + 0], 1e-9);
    float isy = 1.0 / max(scales[i * 3 + 1], 1e-9);
    float isz = 1.0 / max(scales[i * 3 + 2], 1e-9);
    // M = S⁻¹ Rᵀ: row r of M is column r of R scaled by 1/s_r.
    float m00 = R.r00 * isx; float m01 = R.r10 * isx; float m02 = R.r20 * isx;
    float m10 = R.r01 * isy; float m11 = R.r11 * isy; float m12 = R.r21 * isy;
    float m20 = R.r02 * isz; float m21 = R.r12 * isz; float m22 = R.r22 * isz;

    // Camera center o = -Rcᵀ t.
    float ox = -(viewmat[0] * viewmat[3] + viewmat[4] * viewmat[7] + viewmat[8] * viewmat[11]);
    float oy = -(viewmat[1] * viewmat[3] + viewmat[5] * viewmat[7] + viewmat[9] * viewmat[11]);
    float oz = -(viewmat[2] * viewmat[3] + viewmat[6] * viewmat[7] + viewmat[10] * viewmat[11]);
    float px = ox - means[i * 3 + 0];
    float py = oy - means[i * 3 + 1];
    float pz = oz - means[i * 3 + 2];

    int o = i * 12;
    gauss_frame[o + 0] = m00; gauss_frame[o + 1] = m01; gauss_frame[o + 2] = m02;
    gauss_frame[o + 3] = m10; gauss_frame[o + 4] = m11; gauss_frame[o + 5] = m12;
    gauss_frame[o + 6] = m20; gauss_frame[o + 7] = m21; gauss_frame[o + 8] = m22;
    gauss_frame[o + 9]  = m00 * px + m01 * py + m02 * pz;
    gauss_frame[o + 10] = m10 * px + m11 * py + m12 * pz;
    gauss_frame[o + 11] = m20 * px + m21 * py + m22 * pz;

    // σ_z² = r3ᵀ Σ r3 = |S Rᵀ r3|² with r3 the camera z axis (third row of the view rotation).
    float vx = R.r00 * viewmat[8] + R.r10 * viewmat[9] + R.r20 * viewmat[10];
    float vy = R.r01 * viewmat[8] + R.r11 * viewmat[9] + R.r21 * viewmat[10];
    float vz = R.r02 * viewmat[8] + R.r12 * viewmat[9] + R.r22 * viewmat[10];
    float sx = scales[i * 3 + 0] * vx;
    float sy = scales[i * 3 + 1] * vy;
    float sz = scales[i * 3 + 2] * vz;
    z_extent[i] = 3.0 * sqrt(sx * sx + sy * sy + sz * sz);
}

// ---- gof_depth: per pixel ----------------------------------------------------------------------

StructuredBuffer<float> frames;        // [N, 12] from gof_prepare
StructuredBuffer<float> extents;       // [N] from gof_prepare
StructuredBuffer<float> center_depths; // [N] camera-space depth of μ (projection output)
StructuredBuffer<float> opacities;     // [N]
StructuredBuffer<float> means2d;       // [N, 2]
StructuredBuffer<float> conics;        // [N, 3]
StructuredBuffer<int>   flatten_ids;   // [n_isects] tile lists, sorted by center depth
StructuredBuffer<int>   tile_offsets;  // [num_tiles + 1]
StructuredBuffer<int>   dims;          // [imageWidth, imageHeight, tileSize, numTilesX, bisectionIterations, stride]
StructuredBuffer<float> camera;        // [20] viewmat (16) then fx, fy, cx, cy
StructuredBuffer<float> depth_params;  // [3] (nearPlane, maxExtent: max of `extents` over the view, profile)

// With stride s, only the pixels (i s + s/2, j s + s/2) are evaluated, into [(H / s) * (W / s)].
RWStructuredBuffer<float> surface_depth; // 0.5-crossing depth, -1 if the ray never reaches 0.5
RWStructuredBuffer<float> surface_slope; // d log T / dt there from the final interval, 0 if unknown

// gof_depth_bwd only.
StructuredBuffer<float> fwd_depth;       // gof_depth output
StructuredBuffer<float> fwd_slope;       // gof_depth output (0: recomputed here)
StructuredBuffer<float> v_depth;         // dL/d(depth), same layout
RWStructuredBuffer<float> v_frames;      // [N, 12] dL/d(M rows, o_g)
RWStructuredBuffer<float> v_frame_opacities; // [N] dL/d(opacity)

static const float kMinAlpha = 0.00392157;  // 1/255, as the rasterizer
static const float kMaxAlpha = 0.99;
static const int kSplit = 8;                // search segments per pass (kSplit - 1 interior points)
static const int kMaxPoints = kSplit + 1;   // the first pass also measures its interval's ends

// Coefficients of the 1D Gaussian seen along the ray (GOF eq. 5): G(t) = exp(-½(A t² + 2B t + C)).
// The helpers below take values only (no buffer access), as the kernel loader requires.
struct RayGaussian
{
    float A;
    float B;
    float C;
};

RayGaussian rayGaussian(float3 m0, float3 m1, float3 m2, float3 og, float3 d)
{
    float3 dg = float3(dot(m0, d), dot(m1, d), dot(m2, d));
    RayGaussian g;
    g.A = dot(dg, dg);
    g.B = dot(og, dg);
    g.C = dot(og, og);
    return g;
}

// Peak α G(t*) and t* (GOF eq. 6-7), 0 if negligible or if the peak lies behind the near plane (the
// Gaussian is then mostly behind the camera along this ray).
float peakAlpha(RayGaussian g, float opacity, float nearPlane, out float tStar)
{
    tStar = 0.0;
    if (g.A <= 1e-20) { return 0.0; }
    tStar = -g.B / g.A;
    if (tStar < nearPlane) { return 0.0; }
    float a = min(kMaxAlpha, opacity * exp(-0.5 * max(g.C - g.B * g.B / g.A, 0.0)));
    return a < kMinAlpha ? 0.0 : a;
}

// One Gaussian's transmittance up to depth t (see the header), from its peak alpha, t* and A
// (g(t) = peak · exp(-½ A (t - t*)²)).
float transmittanceAt(float peak, float tStar, float A, float t, int profile)
{
    float dt = t - tStar;
    float gt = peak * exp(-0.5 * A * dt * dt);
    if (profile == 0) { return dt >= 0.0 ? 1.0 - peak : 1.0 - gt; }
    float v = sqrt(1.0 - gt);
    return dt <= 0.0 ? v : (1.0 - peak) / v;
}

// ∂ log T_k / ∂(t, A, B, C, opacity) at depth t, for the factor of `transmittanceAt`. With
// u = t - t*, e = exp(-½ A u²), g = p e and p = α exp(-½(C - B²/A)) (constant where clamped):
//   before the peak  log T = k log(1 - g)                      (k = 1 GOF, ½ stochastic solid)
//   after it         log T = log(1 - p) - k' log(1 - g)       (k' = 0 GOF, ½ stochastic solid)
struct LogTransmittanceGrad
{
    float dt;
    float dA;
    float dB;
    float dC;
    float dO;
};

LogTransmittanceGrad logTransmittanceGrad(
    float A, float B, float C, float opacity, float peak, bool clamped, float tStar, float t, int profile)
{
    float u = t - tStar;
    float e = exp(-0.5 * A * u * u);
    float gt = peak * e;
    float pA = 0.0, pB = 0.0, pC = 0.0, pO = 0.0;
    if (!clamped) {
        pA = -0.5 * peak * B * B / (A * A);
        pB = peak * B / A;
        pC = -0.5 * peak;
        pO = peak / max(opacity, 1e-12);
    }
    // ∂ log e: ∂u/∂A = -B/A², ∂u/∂B = 1/A (t* = -B/A).
    float leA = -0.5 * u * u + u * B / A;
    float leB = -u;
    float gA = e * pA + gt * leA;
    float gB = e * pB + gt * leB;
    float gC = e * pC;
    float gO = e * pO;
    float gT = -gt * A * u;
    float inv1g = 1.0 / max(1.0 - gt, 1e-6);
    LogTransmittanceGrad r;
    if (u < 0.0) {
        float k = profile == 0 ? 1.0 : 0.5;
        r.dt = -k * gT * inv1g;
        r.dA = -k * gA * inv1g;
        r.dB = -k * gB * inv1g;
        r.dC = -k * gC * inv1g;
        r.dO = -k * gO * inv1g;
    } else {
        float k = profile == 0 ? 0.0 : 0.5;
        float inv1p = 1.0 / max(1.0 - peak, 1e-6);
        r.dt = k * gT * inv1g;
        r.dA = -pA * inv1p + k * gA * inv1g;
        r.dB = -pB * inv1p + k * gB * inv1g;
        r.dC = -pC * inv1p + k * gC * inv1g;
        r.dO = -pO * inv1p + k * gO * inv1g;
    }
    return r;
}

// A Gaussian whose peak still lies ahead (u = t - t* < 0) with g(t) below 1e-4 changes neither T
// nor its derivatives noticeably at t; the backward skips it.
bool negligibleAhead(float peak, float A, float u)
{
    return u < 0.0 && peak * exp(-0.5 * A * u * u) < 1e-4;
}

// Coarse screen-space reject: beyond ~5σ of the projected footprint.
bool farFromFootprint(float dx, float dy, float ca, float cb, float cc)
{
    return 0.5 * (ca * dx * dx + cc * dy * dy) + cb * dx * dy > 12.5;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void gof_depth(uint3 tid : SV_DispatchThreadID)
{
    int imageWidth  = dims[0];
    int imageHeight = dims[1];
    int tileSize    = dims[2];
    int numTilesX   = dims[3];
    int iterations  = dims[4];
    int stride      = dims[5];

    if (int(tid.x) >= imageWidth / stride || int(tid.y) >= imageHeight / stride) { return; }
    int pixel_x = int(tid.x) * stride + stride / 2;
    int pixel_y = int(tid.y) * stride + stride / 2;

    int tile_id = (pixel_y / tileSize) * numTilesX + (pixel_x / tileSize);
    int isect_start = tile_offsets[tile_id];
    int isect_end   = tile_offsets[tile_id + 1];
    int pix_idx = int(tid.y) * (imageWidth / stride) + int(tid.x);

    float px = float(pixel_x) + 0.5;
    float py = float(pixel_y) + 0.5;
    float nearPlane = depth_params[0];
    float maxExtent = depth_params[1];
    int profile = int(depth_params[2]);

    // World ray direction with camera z = 1: d = Rcᵀ (x_n, y_n, 1).
    float xn = (px - camera[18]) / camera[16];
    float yn = (py - camera[19]) / camera[17];
    float3 d = float3(
        camera[0] * xn + camera[4] * yn + camera[8],
        camera[1] * xn + camera[5] * yn + camera[9],
        camera[2] * xn + camera[6] * yn + camera[10]);

    // Bracket: accumulate the final factors 1 - g(t*). Once their product drops below 0.5, T is
    // below 0.5 wherever every Gaussian taken so far has (nearly) reached its final factor: from
    // max t* (profile 0), or max t* + 4σ along the ray (profile 1, whose factors keep falling behind
    // the peak; at 4σ they are within 2e-4 of the final value).
    float T = 1.0;
    float tHi = nearPlane;
    // Peak and ray-wise σ of the Gaussian whose final factor took the product below 0.5: the
    // crossing is usually within a few σ of it.
    float tCross = nearPlane;
    float sCross = 0.0;
    bool reached = false;
    for (int k = isect_start; k < isect_end; k++) {
        int gi = flatten_ids[k];
        if (farFromFootprint(px - means2d[gi * 2 + 0], py - means2d[gi * 2 + 1],
                             conics[gi * 3 + 0], conics[gi * 3 + 1], conics[gi * 3 + 2])) { continue; }
        int o = gi * 12;
        RayGaussian g = rayGaussian(
            float3(frames[o + 0], frames[o + 1], frames[o + 2]),
            float3(frames[o + 3], frames[o + 4], frames[o + 5]),
            float3(frames[o + 6], frames[o + 7], frames[o + 8]),
            float3(frames[o + 9], frames[o + 10], frames[o + 11]), d);
        float tStar;
        float a = peakAlpha(g, opacities[gi], nearPlane, tStar);
        if (a <= 0.0) { continue; }
        T *= (1.0 - a);
        tHi = max(tHi, profile == 0 ? tStar : tStar + 4.0 / sqrt(g.A));
        if (T < 0.5) {
            reached = true;
            tCross = tStar;
            sCross = 1.0 / sqrt(g.A);
            break;
        }
    }
    if (!reached) {
        surface_depth[pix_idx] = -1.0;
        surface_slope[pix_idx] = 0.0;
        return;
    }

    // Search the crossing of the order-free T(t) over the whole tile list: each pass measures T at a
    // few points of [lo, hi] and keeps the segment where it crosses 0.5. The first pass spans
    // tCross ± 3σ (both ends included, so a crossing outside it falls back to [lo, end] or [end,
    // hi]); the others measure the kSplit - 1 interior points.
    float lo = nearPlane;
    float hi = tHi;
    // T at the ends of [lo, hi] when measured exactly (T(near) = 1; tHi's is only bounded).
    float Tlo = 1.0;
    float Thi = 0.0;
    bool loExact = true;
    bool hiExact = false;
    int passes = (iterations + 2) / 3;
    for (int pass = 0; pass < passes; pass++) {
        float pts[kMaxPoints];
        int count;
        float a0 = max(lo, tCross - 3.0 * sCross);
        float b0 = min(hi, tCross + 3.0 * sCross);
        if (pass == 0 && a0 < b0) {
            count = kMaxPoints;
            for (int j = 0; j < kMaxPoints; j++) { pts[j] = a0 + (b0 - a0) * float(j) / float(kSplit); }
        } else {
            count = kSplit - 1;
            float step = (hi - lo) / float(kSplit);
            for (int j = 0; j < kSplit - 1; j++) { pts[j] = lo + float(j + 1) * step; }
        }
        float tLast = pts[count - 1];
        float Ts[kMaxPoints];
        for (int j = 0; j < kMaxPoints; j++) { Ts[j] = 1.0; }
        for (int k = isect_start; k < isect_end; k++) {
            int gi = flatten_ids[k];
            // The list is sorted by center depth, so every later Gaussian starts beyond `tLast`.
            if (center_depths[gi] - maxExtent > tLast) { break; }
            if (center_depths[gi] - extents[gi] > tLast) { continue; }
            if (farFromFootprint(px - means2d[gi * 2 + 0], py - means2d[gi * 2 + 1],
                                 conics[gi * 3 + 0], conics[gi * 3 + 1], conics[gi * 3 + 2])) { continue; }
            int o = gi * 12;
            RayGaussian g = rayGaussian(
                float3(frames[o + 0], frames[o + 1], frames[o + 2]),
                float3(frames[o + 3], frames[o + 4], frames[o + 5]),
                float3(frames[o + 6], frames[o + 7], frames[o + 8]),
                float3(frames[o + 9], frames[o + 10], frames[o + 11]), d);
            float tStar;
            float a = peakAlpha(g, opacities[gi], nearPlane, tStar);
            if (a <= 0.0) { continue; }
            // Most Gaussians are far from every point once the interval is narrow: wholly passed
            // (all points beyond t* + 4σ: the factor is its final 1 - α, within 2e-4 for the stochastic
            // solid) or wholly ahead (all points before t* - 4σ: factor 1). Only the rest needs the
            // per-point evaluation.
            float sigma4 = 4.0 / sqrt(g.A);
            if (pts[0] - tStar > sigma4) {
                for (int j = 0; j < count; j++) { Ts[j] *= 1.0 - a; }
            } else if (pts[count - 1] - tStar >= -sigma4) {
                for (int j = 0; j < count; j++) {
                    Ts[j] *= transmittanceAt(a, tStar, g.A, pts[j], profile);
                }
            }
            // T is non-increasing in t, so all points are below 0.5 once the first is.
            if (Ts[0] < 0.5) { break; }
        }
        // The scan stops early only once Ts[0] < 0.5 (then seg = 0 and Ts[0] is not final);
        // otherwise every Ts is exact.
        int seg = count;
        for (int j = 0; j < count; j++) {
            if (Ts[j] < 0.5) { seg = j; break; }
        }
        if (seg > 0) { Tlo = Ts[seg - 1]; loExact = true; }
        if (seg == 0) { hiExact = false; }
        else if (seg < count) { Thi = Ts[seg]; hiExact = true; }
        float newLo = seg == 0 ? lo : pts[seg - 1];
        float newHi = seg == count ? hi : pts[seg];
        lo = newLo;
        hi = newHi;
    }
    // log T is smooth, so within the final interval its 0.5 crossing and slope follow from the
    // ends (linear in log T); the midpoint where an end is not exact.
    if (loExact && hiExact && Tlo > 0.5 && Thi < 0.5 && hi > lo) {
        float llo = log(Tlo), lhi = log(max(Thi, 1e-30));
        surface_depth[pix_idx] = lo + (hi - lo) * (llo - log(0.5)) / (llo - lhi);
        // Across a very narrow interval the difference of log T is lost to rounding: leave the
        // slope to the backward pass then.
        surface_slope[pix_idx] = llo - lhi > 1e-3 ? (lhi - llo) / (hi - lo) : 0.0;
    } else {
        surface_depth[pix_idx] = 0.5 * (lo + hi);
        surface_slope[pix_idx] = 0.0;
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void gof_depth_bwd(uint3 tid : SV_DispatchThreadID)
{
    int imageWidth  = dims[0];
    int imageHeight = dims[1];
    int tileSize    = dims[2];
    int numTilesX   = dims[3];
    int stride      = dims[5];

    if (int(tid.x) >= imageWidth / stride || int(tid.y) >= imageHeight / stride) { return; }
    int pixel_x = int(tid.x) * stride + stride / 2;
    int pixel_y = int(tid.y) * stride + stride / 2;
    int pix_idx = int(tid.y) * (imageWidth / stride) + int(tid.x);
    float t = fwd_depth[pix_idx];
    float vt = v_depth[pix_idx];
    if (t <= 0.0 || vt == 0.0) { return; }

    int tile_id = (pixel_y / tileSize) * numTilesX + (pixel_x / tileSize);
    int isect_start = tile_offsets[tile_id];
    int isect_end   = tile_offsets[tile_id + 1];

    float px = float(pixel_x) + 0.5;
    float py = float(pixel_y) + 0.5;
    float nearPlane = depth_params[0];
    float maxExtent = depth_params[1];
    int profile = int(depth_params[2]);

    float xn = (px - camera[18]) / camera[16];
    float yn = (py - camera[19]) / camera[17];
    float3 d = float3(
        camera[0] * xn + camera[4] * yn + camera[8],
        camera[1] * xn + camera[5] * yn + camera[9],
        camera[2] * xn + camera[6] * yn + camera[10]);

    // Pass 1: Σ_k ∂ log T_k / ∂t at t (< 0 where the ray crosses 0.5), unless the forward measured
    // it over its final interval.
    float slope = fwd_slope[pix_idx];
    bool measure = slope == 0.0;
    for (int k = isect_start; measure && k < isect_end; k++) {
        int gi = flatten_ids[k];
        if (center_depths[gi] - maxExtent > t) { break; }
        if (center_depths[gi] - extents[gi] > t) { continue; }
        if (farFromFootprint(px - means2d[gi * 2 + 0], py - means2d[gi * 2 + 1],
                             conics[gi * 3 + 0], conics[gi * 3 + 1], conics[gi * 3 + 2])) { continue; }
        int o = gi * 12;
        RayGaussian g = rayGaussian(
            float3(frames[o + 0], frames[o + 1], frames[o + 2]),
            float3(frames[o + 3], frames[o + 4], frames[o + 5]),
            float3(frames[o + 6], frames[o + 7], frames[o + 8]),
            float3(frames[o + 9], frames[o + 10], frames[o + 11]), d);
        float tStar;
        float a = peakAlpha(g, opacities[gi], nearPlane, tStar);
        if (a <= 0.0 || negligibleAhead(a, g.A, t - tStar)) { continue; }
        bool clamped = a >= kMaxAlpha;
        slope += logTransmittanceGrad(g.A, g.B, g.C, opacities[gi], a, clamped, tStar, t, profile).dt;
    }
    if (slope > -1e-8) { return; }
    float scale = -vt / slope;

    // Pass 2: dL/dθ_i = -vt (∂ log T_i/∂θ_i) / slope, through A = |M d|², B = o_g·(M d), C = |o_g|².
    for (int k = isect_start; k < isect_end; k++) {
        int gi = flatten_ids[k];
        if (center_depths[gi] - maxExtent > t) { break; }
        if (center_depths[gi] - extents[gi] > t) { continue; }
        if (farFromFootprint(px - means2d[gi * 2 + 0], py - means2d[gi * 2 + 1],
                             conics[gi * 3 + 0], conics[gi * 3 + 1], conics[gi * 3 + 2])) { continue; }
        int o = gi * 12;
        float3 m0 = float3(frames[o + 0], frames[o + 1], frames[o + 2]);
        float3 m1 = float3(frames[o + 3], frames[o + 4], frames[o + 5]);
        float3 m2 = float3(frames[o + 6], frames[o + 7], frames[o + 8]);
        float3 og = float3(frames[o + 9], frames[o + 10], frames[o + 11]);
        RayGaussian g = rayGaussian(m0, m1, m2, og, d);
        float tStar;
        float a = peakAlpha(g, opacities[gi], nearPlane, tStar);
        if (a <= 0.0 || negligibleAhead(a, g.A, t - tStar)) { continue; }
        bool clamped = a >= kMaxAlpha;
        LogTransmittanceGrad gr = logTransmittanceGrad(
            g.A, g.B, g.C, opacities[gi], a, clamped, tStar, t, profile);
        float lA = scale * gr.dA;
        float lB = scale * gr.dB;
        float lC = scale * gr.dC;
        float3 dg = float3(dot(m0, d), dot(m1, d), dot(m2, d));
        // ∂/∂M[r][c] = (2 lA dg_r + lB og_r) d_c, ∂/∂o_g = lB dg + 2 lC o_g.
        float3 w = 2.0 * lA * dg + lB * og;
        InterlockedAdd(v_frames[o + 0], w.x * d.x);
        InterlockedAdd(v_frames[o + 1], w.x * d.y);
        InterlockedAdd(v_frames[o + 2], w.x * d.z);
        InterlockedAdd(v_frames[o + 3], w.y * d.x);
        InterlockedAdd(v_frames[o + 4], w.y * d.y);
        InterlockedAdd(v_frames[o + 5], w.y * d.z);
        InterlockedAdd(v_frames[o + 6], w.z * d.x);
        InterlockedAdd(v_frames[o + 7], w.z * d.y);
        InterlockedAdd(v_frames[o + 8], w.z * d.z);
        float3 vog = lB * dg + 2.0 * lC * og;
        InterlockedAdd(v_frames[o + 9], vog.x);
        InterlockedAdd(v_frames[o + 10], vog.y);
        InterlockedAdd(v_frames[o + 11], vog.z);
        InterlockedAdd(v_frame_opacities[gi], scale * gr.dO);
    }
}

// ---- Cooperative versions: one SIMD group (32 lanes) per evaluated pixel -------------------------
// The same depth and gradient as gof_depth / gof_depth_bwd, for few pixels (the sparse samples of
// the training regularization), where one thread per pixel would leave the GPU mostly idle. The
// lanes take every 32nd entry of the tile list: the order-free products become sums of log factors
// (WaveActiveSum), and the bracket's ordered product a prefix sum. Dispatch (W/s * 32, H/s) in
// groups of (32, 4): each row of 32 threads is one SIMD group and one pixel.

static const int kLanes = 32;

[shader("compute")]
[numthreads(32, 4, 1)]
void gof_depth_coop(uint3 tid : SV_DispatchThreadID)
{
    int imageWidth  = dims[0];
    int imageHeight = dims[1];
    int tileSize    = dims[2];
    int numTilesX   = dims[3];
    int iterations  = dims[4];
    int stride      = dims[5];

    int lane = int(WaveGetLaneIndex());
    int sx = int(tid.x) / kLanes;
    int sy = int(tid.y);
    // Uniform over the SIMD group.
    if (sx >= imageWidth / stride || sy >= imageHeight / stride) { return; }
    int pixel_x = sx * stride + stride / 2;
    int pixel_y = sy * stride + stride / 2;
    int tile_id = (pixel_y / tileSize) * numTilesX + (pixel_x / tileSize);
    int isect_start = tile_offsets[tile_id];
    int isect_end   = tile_offsets[tile_id + 1];
    int pix_idx = sy * (imageWidth / stride) + sx;

    float px = float(pixel_x) + 0.5;
    float py = float(pixel_y) + 0.5;
    float nearPlane = depth_params[0];
    float maxExtent = depth_params[1];
    int profile = int(depth_params[2]);
    float xn = (px - camera[18]) / camera[16];
    float yn = (py - camera[19]) / camera[17];
    float3 d = float3(
        camera[0] * xn + camera[4] * yn + camera[8],
        camera[1] * xn + camera[5] * yn + camera[9],
        camera[2] * xn + camera[6] * yn + camera[10]);

    // Bracket (see gof_depth): the first entry whose final factor takes the running product below
    // 0.5, found per chunk of 32 from the inclusive prefix sum of log(1 - α).
    float logT = 0.0;
    float tHi = nearPlane;
    float tCross = nearPlane;
    float sCross = 0.0;
    bool reached = false;
    for (int base = isect_start; base < isect_end && !reached; base += kLanes) {
        int k = base + lane;
        float lf = 0.0;
        float tS = nearPlane;
        float sg = 0.0;
        float ext = nearPlane;
        if (k < isect_end) {
            int gi = flatten_ids[k];
            if (!farFromFootprint(px - means2d[gi * 2 + 0], py - means2d[gi * 2 + 1],
                                  conics[gi * 3 + 0], conics[gi * 3 + 1], conics[gi * 3 + 2])) {
                int o = gi * 12;
                RayGaussian g = rayGaussian(
                    float3(frames[o + 0], frames[o + 1], frames[o + 2]),
                    float3(frames[o + 3], frames[o + 4], frames[o + 5]),
                    float3(frames[o + 6], frames[o + 7], frames[o + 8]),
                    float3(frames[o + 9], frames[o + 10], frames[o + 11]), d);
                float tStar;
                float a = peakAlpha(g, opacities[gi], nearPlane, tStar);
                if (a > 0.0) {
                    lf = log(1.0 - a);
                    tS = tStar;
                    sg = 1.0 / sqrt(g.A);
                    ext = profile == 0 ? tStar : tStar + 4.0 * sg;
                }
            }
        }
        float incl = WavePrefixSum(lf) + lf;
        bool crossed = lf < 0.0 && logT + incl < log(0.5);
        if (WaveActiveAnyTrue(crossed)) {
            int first = WaveActiveMin(crossed ? lane : kLanes);
            tCross = WaveReadLaneAt(tS, first);
            sCross = WaveReadLaneAt(sg, first);
            tHi = max(tHi, WaveActiveMax(lane <= first ? ext : nearPlane));
            reached = true;
        } else {
            logT += WaveActiveSum(lf);
            tHi = max(tHi, WaveActiveMax(ext));
        }
    }
    if (!reached) {
        if (lane == 0) {
            surface_depth[pix_idx] = -1.0;
            surface_slope[pix_idx] = 0.0;
        }
        return;
    }

    // Search (see gof_depth). Without an early stop every measured T is exact.
    float lo = nearPlane;
    float hi = tHi;
    float Tlo = 1.0;
    float Thi = 0.0;
    bool hiExact = false;
    int passes = (iterations + 2) / 3;
    for (int pass = 0; pass < passes; pass++) {
        float pts[kMaxPoints];
        int count;
        float a0 = max(lo, tCross - 3.0 * sCross);
        float b0 = min(hi, tCross + 3.0 * sCross);
        if (pass == 0 && a0 < b0) {
            count = kMaxPoints;
            for (int j = 0; j < kMaxPoints; j++) { pts[j] = a0 + (b0 - a0) * float(j) / float(kSplit); }
        } else {
            count = kSplit - 1;
            float step = (hi - lo) / float(kSplit);
            for (int j = 0; j < kSplit - 1; j++) { pts[j] = lo + float(j + 1) * step; }
        }
        float tLast = pts[count - 1];
        float ls[kMaxPoints];
        for (int j = 0; j < kMaxPoints; j++) { ls[j] = 0.0; }
        for (int base = isect_start; base < isect_end; base += kLanes) {
            int k = base + lane;
            bool beyond = true;
            if (k < isect_end) {
                int gi = flatten_ids[k];
                beyond = center_depths[gi] - maxExtent > tLast;
                if (!beyond && center_depths[gi] - extents[gi] <= tLast
                    && !farFromFootprint(px - means2d[gi * 2 + 0], py - means2d[gi * 2 + 1],
                                         conics[gi * 3 + 0], conics[gi * 3 + 1], conics[gi * 3 + 2])) {
                    int o = gi * 12;
                    RayGaussian g = rayGaussian(
                        float3(frames[o + 0], frames[o + 1], frames[o + 2]),
                        float3(frames[o + 3], frames[o + 4], frames[o + 5]),
                        float3(frames[o + 6], frames[o + 7], frames[o + 8]),
                        float3(frames[o + 9], frames[o + 10], frames[o + 11]), d);
                    float tStar;
                    float a = peakAlpha(g, opacities[gi], nearPlane, tStar);
                    if (a > 0.0) {
                        float sigma4 = 4.0 / sqrt(g.A);
                        if (pts[0] - tStar > sigma4) {
                            float l = log(1.0 - a);
                            for (int j = 0; j < count; j++) { ls[j] += l; }
                        } else if (pts[count - 1] - tStar >= -sigma4) {
                            for (int j = 0; j < count; j++) {
                                ls[j] += log(transmittanceAt(a, tStar, g.A, pts[j], profile));
                            }
                        }
                    }
                }
            }
            // The list is sorted by center depth: once every lane is beyond, so is the rest.
            if (WaveActiveAllTrue(beyond)) { break; }
        }
        float Ts[kMaxPoints];
        for (int j = 0; j < kMaxPoints; j++) { Ts[j] = exp(WaveActiveSum(ls[j])); }
        int seg = count;
        for (int j = 0; j < count; j++) {
            if (Ts[j] < 0.5) { seg = j; break; }
        }
        if (seg > 0) { Tlo = Ts[seg - 1]; }
        if (seg < count) { Thi = Ts[seg]; hiExact = true; }
        float newLo = seg == 0 ? lo : pts[seg - 1];
        float newHi = seg == count ? hi : pts[seg];
        lo = newLo;
        hi = newHi;
    }
    if (lane != 0) { return; }
    if (hiExact && Tlo > 0.5 && Thi < 0.5 && hi > lo) {
        float llo = log(Tlo), lhi = log(max(Thi, 1e-30));
        surface_depth[pix_idx] = lo + (hi - lo) * (llo - log(0.5)) / (llo - lhi);
        surface_slope[pix_idx] = llo - lhi > 1e-3 ? (lhi - llo) / (hi - lo) : 0.0;
    } else {
        surface_depth[pix_idx] = 0.5 * (lo + hi);
        surface_slope[pix_idx] = 0.0;
    }
}

[shader("compute")]
[numthreads(32, 4, 1)]
void gof_depth_bwd_coop(uint3 tid : SV_DispatchThreadID)
{
    int imageWidth  = dims[0];
    int imageHeight = dims[1];
    int tileSize    = dims[2];
    int numTilesX   = dims[3];
    int stride      = dims[5];

    int lane = int(WaveGetLaneIndex());
    int sx = int(tid.x) / kLanes;
    int sy = int(tid.y);
    if (sx >= imageWidth / stride || sy >= imageHeight / stride) { return; }
    int pix_idx = sy * (imageWidth / stride) + sx;
    float t = fwd_depth[pix_idx];
    float vt = v_depth[pix_idx];
    if (t <= 0.0 || vt == 0.0) { return; }
    int pixel_x = sx * stride + stride / 2;
    int pixel_y = sy * stride + stride / 2;
    int tile_id = (pixel_y / tileSize) * numTilesX + (pixel_x / tileSize);
    int isect_start = tile_offsets[tile_id];
    int isect_end   = tile_offsets[tile_id + 1];

    float px = float(pixel_x) + 0.5;
    float py = float(pixel_y) + 0.5;
    float nearPlane = depth_params[0];
    float maxExtent = depth_params[1];
    int profile = int(depth_params[2]);
    float xn = (px - camera[18]) / camera[16];
    float yn = (py - camera[19]) / camera[17];
    float3 d = float3(
        camera[0] * xn + camera[4] * yn + camera[8],
        camera[1] * xn + camera[5] * yn + camera[9],
        camera[2] * xn + camera[6] * yn + camera[10]);

    // Pass 1 (only where the forward did not measure the slope).
    float slope = fwd_slope[pix_idx];
    if (slope == 0.0) {
        float part = 0.0;
        for (int base = isect_start; base < isect_end; base += kLanes) {
            int k = base + lane;
            bool beyond = true;
            if (k < isect_end) {
                int gi = flatten_ids[k];
                beyond = center_depths[gi] - maxExtent > t;
                if (!beyond && center_depths[gi] - extents[gi] <= t
                    && !farFromFootprint(px - means2d[gi * 2 + 0], py - means2d[gi * 2 + 1],
                                         conics[gi * 3 + 0], conics[gi * 3 + 1], conics[gi * 3 + 2])) {
                    int o = gi * 12;
                    RayGaussian g = rayGaussian(
                        float3(frames[o + 0], frames[o + 1], frames[o + 2]),
                        float3(frames[o + 3], frames[o + 4], frames[o + 5]),
                        float3(frames[o + 6], frames[o + 7], frames[o + 8]),
                        float3(frames[o + 9], frames[o + 10], frames[o + 11]), d);
                    float tStar;
                    float a = peakAlpha(g, opacities[gi], nearPlane, tStar);
                    if (a > 0.0 && !negligibleAhead(a, g.A, t - tStar)) {
                        part += logTransmittanceGrad(
                            g.A, g.B, g.C, opacities[gi], a, a >= kMaxAlpha, tStar, t, profile).dt;
                    }
                }
            }
            if (WaveActiveAllTrue(beyond)) { break; }
        }
        slope = WaveActiveSum(part);
    }
    if (slope > -1e-8) { return; }
    float scale = -vt / slope;

    // Pass 2: each lane scatters the share of its entries (see gof_depth_bwd).
    for (int base = isect_start; base < isect_end; base += kLanes) {
        int k = base + lane;
        bool beyond = true;
        if (k < isect_end) {
            int gi = flatten_ids[k];
            beyond = center_depths[gi] - maxExtent > t;
            if (!beyond && center_depths[gi] - extents[gi] <= t
                && !farFromFootprint(px - means2d[gi * 2 + 0], py - means2d[gi * 2 + 1],
                                     conics[gi * 3 + 0], conics[gi * 3 + 1], conics[gi * 3 + 2])) {
                int o = gi * 12;
                float3 m0 = float3(frames[o + 0], frames[o + 1], frames[o + 2]);
                float3 m1 = float3(frames[o + 3], frames[o + 4], frames[o + 5]);
                float3 m2 = float3(frames[o + 6], frames[o + 7], frames[o + 8]);
                float3 og = float3(frames[o + 9], frames[o + 10], frames[o + 11]);
                RayGaussian g = rayGaussian(m0, m1, m2, og, d);
                float tStar;
                float a = peakAlpha(g, opacities[gi], nearPlane, tStar);
                if (a > 0.0 && !negligibleAhead(a, g.A, t - tStar)) {
                    LogTransmittanceGrad gr = logTransmittanceGrad(
                        g.A, g.B, g.C, opacities[gi], a, a >= kMaxAlpha, tStar, t, profile);
                    float lA = scale * gr.dA;
                    float lB = scale * gr.dB;
                    float lC = scale * gr.dC;
                    float3 dg = float3(dot(m0, d), dot(m1, d), dot(m2, d));
                    float3 w = 2.0 * lA * dg + lB * og;
                    InterlockedAdd(v_frames[o + 0], w.x * d.x);
                    InterlockedAdd(v_frames[o + 1], w.x * d.y);
                    InterlockedAdd(v_frames[o + 2], w.x * d.z);
                    InterlockedAdd(v_frames[o + 3], w.y * d.x);
                    InterlockedAdd(v_frames[o + 4], w.y * d.y);
                    InterlockedAdd(v_frames[o + 5], w.y * d.z);
                    InterlockedAdd(v_frames[o + 6], w.z * d.x);
                    InterlockedAdd(v_frames[o + 7], w.z * d.y);
                    InterlockedAdd(v_frames[o + 8], w.z * d.z);
                    float3 vog = lB * dg + 2.0 * lC * og;
                    InterlockedAdd(v_frames[o + 9], vog.x);
                    InterlockedAdd(v_frames[o + 10], vog.y);
                    InterlockedAdd(v_frames[o + 11], vog.z);
                    InterlockedAdd(v_frame_opacities[gi], scale * gr.dO);
                }
            }
        }
        if (WaveActiveAllTrue(beyond)) { break; }
    }
}
