// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Shared [Differentiable] math functions used by each kernel via `import gs_math;`.
// Conventions: quaternions are (w, x, y, z); rotations and viewmats are row-major.
// The math follows gsplat (Apache-2.0) Utils.cuh / SphericalHarmonicsCUDA.cu.
// SH constants are the standard values (Sloan 2008).

module gs_math;

// ---- Types --------------------------------------------------------------------

public struct Mat3 : IDifferentiable
{
    public float r00; public float r01; public float r02;
    public float r10; public float r11; public float r12;
    public float r20; public float r21; public float r22;
};

// Symmetric 3x3 covariance (6 upper-triangular elements)
public struct SymMat3 : IDifferentiable
{
    public float c00; public float c01; public float c02;
    public float c11; public float c12; public float c22;
};

// Symmetric 2x2 covariance
public struct SymMat2 : IDifferentiable
{
    public float c00; public float c01; public float c11;
};

// Output of world_to_cam (camera space)
public struct CamPose : IDifferentiable
{
    public float mx; public float my; public float mz;
    public SymMat3 cov;
}

// Output of persp_proj (image plane)
public struct ProjResult : IDifferentiable
{
    public float px; public float py;
    public SymMat2 cov2d;
}

// RGB SH coefficients, 16 bands (degree 0..3)
public struct SHBands16 : IDifferentiable
{
    public float3 b0;  public float3 b1;  public float3 b2;  public float3 b3;
    public float3 b4;  public float3 b5;  public float3 b6;  public float3 b7;
    public float3 b8;  public float3 b9;  public float3 b10; public float3 b11;
    public float3 b12; public float3 b13; public float3 b14; public float3 b15;
};

// ---- Functions ----------------------------------------------------------------

[Differentiable]
public Mat3 quat_to_rotmat(float qw, float qx, float qy, float qz)
{
    float inv_norm = rsqrt(qx * qx + qy * qy + qz * qz + qw * qw);
    float w = qw * inv_norm;
    float x = qx * inv_norm;
    float y = qy * inv_norm;
    float z = qz * inv_norm;

    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;

    // Standard active rotation matrix (row-major). Matches gsplat / simd / Eigen so that
    // exported quaternions can be read as-is by other tools
    Mat3 R;
    R.r00 = 1.0 - 2.0 * (y2 + z2);
    R.r01 = 2.0 * (xy - wz);
    R.r02 = 2.0 * (xz + wy);
    R.r10 = 2.0 * (xy + wz);
    R.r11 = 1.0 - 2.0 * (x2 + z2);
    R.r12 = 2.0 * (yz - wx);
    R.r20 = 2.0 * (xz - wy);
    R.r21 = 2.0 * (yz + wx);
    R.r22 = 1.0 - 2.0 * (x2 + y2);
    return R;
}

[Differentiable]
public SymMat3 quat_scale_to_cov3d(
    float qw, float qx, float qy, float qz,
    float sx, float sy, float sz
)
{
    Mat3 R = quat_to_rotmat(qw, qx, qy, qz);

    float M00 = R.r00 * sx; float M01 = R.r01 * sy; float M02 = R.r02 * sz;
    float M10 = R.r10 * sx; float M11 = R.r11 * sy; float M12 = R.r12 * sz;
    float M20 = R.r20 * sx; float M21 = R.r21 * sy; float M22 = R.r22 * sz;

    SymMat3 C;
    C.c00 = M00 * M00 + M01 * M01 + M02 * M02;
    C.c01 = M00 * M10 + M01 * M11 + M02 * M12;
    C.c02 = M00 * M20 + M01 * M21 + M02 * M22;
    C.c11 = M10 * M10 + M11 * M11 + M12 * M12;
    C.c12 = M10 * M20 + M11 * M21 + M12 * M22;
    C.c22 = M20 * M20 + M21 * M21 + M22 * M22;
    return C;
}

[Differentiable]
public CamPose world_to_cam(
    float wx, float wy, float wz,
    SymMat3 covW,
    no_diff float r00, no_diff float r01, no_diff float r02,
    no_diff float r10, no_diff float r11, no_diff float r12,
    no_diff float r20, no_diff float r21, no_diff float r22,
    no_diff float tx, no_diff float ty, no_diff float tz
)
{
    CamPose out;
    out.mx = r00 * wx + r01 * wy + r02 * wz + tx;
    out.my = r10 * wx + r11 * wy + r12 * wz + ty;
    out.mz = r20 * wx + r21 * wy + r22 * wz + tz;

    float a00 = covW.c00, a01 = covW.c01, a02 = covW.c02;
    float a11 = covW.c11, a12 = covW.c12, a22 = covW.c22;

    float m00 = r00 * a00 + r01 * a01 + r02 * a02;
    float m01 = r00 * a01 + r01 * a11 + r02 * a12;
    float m02 = r00 * a02 + r01 * a12 + r02 * a22;
    float m10 = r10 * a00 + r11 * a01 + r12 * a02;
    float m11 = r10 * a01 + r11 * a11 + r12 * a12;
    float m12 = r10 * a02 + r11 * a12 + r12 * a22;
    float m20 = r20 * a00 + r21 * a01 + r22 * a02;
    float m21 = r20 * a01 + r21 * a11 + r22 * a12;
    float m22 = r20 * a02 + r21 * a12 + r22 * a22;

    out.cov.c00 = m00 * r00 + m01 * r01 + m02 * r02;
    out.cov.c01 = m00 * r10 + m01 * r11 + m02 * r12;
    out.cov.c02 = m00 * r20 + m01 * r21 + m02 * r22;
    out.cov.c11 = m10 * r10 + m11 * r11 + m12 * r12;
    out.cov.c12 = m10 * r20 + m11 * r21 + m12 * r22;
    out.cov.c22 = m20 * r20 + m21 * r21 + m22 * r22;
    return out;
}

[Differentiable]
public ProjResult persp_proj(
    float mx, float my, float mz,
    SymMat3 cov3d,
    no_diff float fx, no_diff float fy,
    no_diff float cx, no_diff float cy,
    no_diff float imageWidth, no_diff float imageHeight
)
{
    float rz = 1.0 / mz;
    float rz2 = rz * rz;

    // For the EWA Jacobian, clamp x/z, y/z to the frustum + 0.3·tan_fov (same as gsplat Utils.cuh).
    // Prevents the 2D covariance of off-screen Gaussians from diverging into streaks. The projected
    // mean is not clamped.
    float tan_fovx = 0.5 * imageWidth / fx;
    float tan_fovy = 0.5 * imageHeight / fy;
    float lim_x_pos = (imageWidth - cx) / fx + 0.3 * tan_fovx;
    float lim_x_neg = cx / fx + 0.3 * tan_fovx;
    float lim_y_pos = (imageHeight - cy) / fy + 0.3 * tan_fovy;
    float lim_y_neg = cy / fy + 0.3 * tan_fovy;
    float tx = mz * min(lim_x_pos, max(-lim_x_neg, mx * rz));
    float ty = mz * min(lim_y_pos, max(-lim_y_neg, my * rz));

    float j00 = fx * rz;
    float j02 = -fx * tx * rz2;
    float j11 = fy * rz;
    float j12 = -fy * ty * rz2;

    float a00 = cov3d.c00, a01 = cov3d.c01, a02 = cov3d.c02;
    float a11 = cov3d.c11, a12 = cov3d.c12, a22 = cov3d.c22;

    float t00 = j00 * a00 + j02 * a02;
    float t01 = j00 * a01 + j02 * a12;
    float t02 = j00 * a02 + j02 * a22;
    float t10 = j11 * a01 + j12 * a02;
    float t11 = j11 * a11 + j12 * a12;
    float t12 = j11 * a12 + j12 * a22;

    SymMat2 cov2d;
    cov2d.c00 = t00 * j00 + t02 * j02;
    cov2d.c01 = t01 * j11 + t02 * j12;
    cov2d.c11 = t11 * j11 + t12 * j12;

    ProjResult r;
    r.px = fx * mx * rz + cx;
    r.py = fy * my * rz + cy;
    r.cov2d = cov2d;
    return r;
}

// Output of project_to_screen. cov2d_* is the 2D covariance after dilation (for computing radii).
// The cotangents of depth and cov2d_* are zeroed in the backward kernel.
public struct ProjectedScreen : IDifferentiable
{
    public float px;
    public float py;
    public float conic_a; public float conic_b; public float conic_c;
    public float depth;
    public float cov2d_a; public float cov2d_b; public float cov2d_c;
}

// Composition of quat_scale_to_cov3d → world_to_cam → persp_proj → 2x2 inverse (conic).
[Differentiable]
public ProjectedScreen project_to_screen(
    float wx, float wy, float wz,
    float qw, float qx, float qy, float qz,
    float sx, float sy, float sz,
    no_diff float r00, no_diff float r01, no_diff float r02,
    no_diff float r10, no_diff float r11, no_diff float r12,
    no_diff float r20, no_diff float r21, no_diff float r22,
    no_diff float tx, no_diff float ty, no_diff float tz,
    no_diff float fx, no_diff float fy, no_diff float cx, no_diff float cy,
    no_diff float imageWidth, no_diff float imageHeight
)
{
    SymMat3 covW = quat_scale_to_cov3d(qw, qx, qy, qz, sx, sy, sz);
    CamPose cam = world_to_cam(
        wx, wy, wz, covW,
        r00, r01, r02, r10, r11, r12, r20, r21, r22, tx, ty, tz
    );
    // near/far cull (gsplat ProjectionEWA3DGSFused.cu). Backward runs over every Gaussian, so a
    // diverging 1/z gives inf*0 = NaN that pollutes the Adam state. Clamp the z used for division
    // at near, and set cov2d = 0 below for out-of-range ones to drop them from binning. Metal's max
    // returns the non-NaN operand, so NaN depths also end up culled.
    float nearPlane = 0.01;
    float farPlane = 1e10;
    float mzSafe = max(cam.mz, nearPlane);
    ProjResult proj = persp_proj(
        cam.mx, cam.my, mzSafe, cam.cov,
        fx, fy, cx, cy, imageWidth, imageHeight
    );

    // Screen-space dilation (gsplat eps2d = 0.3 px²). The viewer applies the same dilation, so
    // without it tiny bright Gaussians accumulate and show up as white speckles in the viewer.
    float a = proj.cov2d.c00 + 0.3;
    float b = proj.cov2d.c01;
    float c = proj.cov2d.c11 + 0.3;
    float det = a * c - b * b;
    // Do not divide when det <= 0 (degenerate Gaussians are culled right after)
    float inv_det = 1.0 / max(det, 1e-9);

    bool culled = !(cam.mz >= nearPlane && cam.mz <= farPlane) || !(det > 0.0);

    ProjectedScreen out;
    out.px = proj.px;
    out.py = proj.py;
    out.conic_a = c * inv_det;
    out.conic_b = -b * inv_det;
    out.conic_c = a * inv_det;
    out.depth = cam.mz;
    // Culled ones get cov2d = 0 → radius 0 → skipped by TileBinning
    out.cov2d_a = culled ? 0.0 : a;
    out.cov2d_b = culled ? 0.0 : b;
    out.cov2d_c = culled ? 0.0 : c;
    return out;
}

// ---- Backward wrappers --------------------------------------------------------
// Slang cannot apply bwd_diff to functions imported from another module, so provide *_grad
// wrappers in this module that call bwd_diff.

public struct QuatGrad
{
    public float w; public float x; public float y; public float z;
};

public QuatGrad quat_to_rotmat_grad(
    float qw, float qx, float qy, float qz,
    Mat3 cot
)
{
    DifferentialPair<float> dpW = diffPair(qw, 0.0);
    DifferentialPair<float> dpX = diffPair(qx, 0.0);
    DifferentialPair<float> dpY = diffPair(qy, 0.0);
    DifferentialPair<float> dpZ = diffPair(qz, 0.0);
    bwd_diff(quat_to_rotmat)(dpW, dpX, dpY, dpZ, cot);
    QuatGrad g;
    g.w = dpW.d; g.x = dpX.d; g.y = dpY.d; g.z = dpZ.d;
    return g;
}

public struct QuatScaleGrad
{
    public float qw; public float qx; public float qy; public float qz;
    public float sx; public float sy; public float sz;
};

public QuatScaleGrad quat_scale_to_cov3d_grad(
    float qw, float qx, float qy, float qz,
    float sx, float sy, float sz,
    SymMat3 cot
)
{
    DifferentialPair<float> dpW  = diffPair(qw, 0.0);
    DifferentialPair<float> dpX  = diffPair(qx, 0.0);
    DifferentialPair<float> dpY  = diffPair(qy, 0.0);
    DifferentialPair<float> dpZ  = diffPair(qz, 0.0);
    DifferentialPair<float> dpSx = diffPair(sx, 0.0);
    DifferentialPair<float> dpSy = diffPair(sy, 0.0);
    DifferentialPair<float> dpSz = diffPair(sz, 0.0);
    bwd_diff(quat_scale_to_cov3d)(dpW, dpX, dpY, dpZ, dpSx, dpSy, dpSz, cot);
    QuatScaleGrad g;
    g.qw = dpW.d; g.qx = dpX.d; g.qy = dpY.d; g.qz = dpZ.d;
    g.sx = dpSx.d; g.sy = dpSy.d; g.sz = dpSz.d;
    return g;
}

public struct WorldToCamGrad
{
    public float wx; public float wy; public float wz;
    public SymMat3 covW;
};

public WorldToCamGrad world_to_cam_grad(
    float wx, float wy, float wz,
    SymMat3 covW,
    float r00, float r01, float r02,
    float r10, float r11, float r12,
    float r20, float r21, float r22,
    float tx, float ty, float tz,
    CamPose cot
)
{
    DifferentialPair<float> dpWx = diffPair(wx, 0.0);
    DifferentialPair<float> dpWy = diffPair(wy, 0.0);
    DifferentialPair<float> dpWz = diffPair(wz, 0.0);
    SymMat3 zero;
    zero.c00 = 0.0; zero.c01 = 0.0; zero.c02 = 0.0;
    zero.c11 = 0.0; zero.c12 = 0.0; zero.c22 = 0.0;
    DifferentialPair<SymMat3> dpCov = diffPair(covW, zero);
    bwd_diff(world_to_cam)(
        dpWx, dpWy, dpWz, dpCov,
        r00, r01, r02, r10, r11, r12, r20, r21, r22, tx, ty, tz,
        cot
    );
    WorldToCamGrad g;
    g.wx = dpWx.d; g.wy = dpWy.d; g.wz = dpWz.d;
    g.covW = dpCov.d;
    return g;
}

public struct PerspProjGrad
{
    public float mx; public float my; public float mz;
    public SymMat3 cov3d;
};

public PerspProjGrad persp_proj_grad(
    float mx, float my, float mz,
    SymMat3 cov3d,
    float fx, float fy, float cx, float cy,
    float imageWidth, float imageHeight,
    ProjResult cot
)
{
    DifferentialPair<float> dpMx = diffPair(mx, 0.0);
    DifferentialPair<float> dpMy = diffPair(my, 0.0);
    DifferentialPair<float> dpMz = diffPair(mz, 0.0);
    SymMat3 zero;
    zero.c00 = 0.0; zero.c01 = 0.0; zero.c02 = 0.0;
    zero.c11 = 0.0; zero.c12 = 0.0; zero.c22 = 0.0;
    DifferentialPair<SymMat3> dpCov = diffPair(cov3d, zero);
    bwd_diff(persp_proj)(
        dpMx, dpMy, dpMz, dpCov,
        fx, fy, cx, cy, imageWidth, imageHeight,
        cot
    );
    PerspProjGrad g;
    g.mx = dpMx.d; g.my = dpMy.d; g.mz = dpMz.d;
    g.cov3d = dpCov.d;
    return g;
}

public struct ShEvalGrad
{
    public float3 dir;
    public SHBands16 sh;
};

public struct ProjectScreenGrad
{
    public float wx; public float wy; public float wz;
    public float qw; public float qx; public float qy; public float qz;
    public float sx; public float sy; public float sz;
};

public ProjectScreenGrad project_to_screen_grad(
    float wx, float wy, float wz,
    float qw, float qx, float qy, float qz,
    float sx, float sy, float sz,
    float r00, float r01, float r02,
    float r10, float r11, float r12,
    float r20, float r21, float r22,
    float tx, float ty, float tz,
    float fx, float fy, float cx, float cy,
    float imageWidth, float imageHeight,
    ProjectedScreen cot
)
{
    DifferentialPair<float> dpWx = diffPair(wx, 0.0);
    DifferentialPair<float> dpWy = diffPair(wy, 0.0);
    DifferentialPair<float> dpWz = diffPair(wz, 0.0);
    DifferentialPair<float> dpQw = diffPair(qw, 0.0);
    DifferentialPair<float> dpQx = diffPair(qx, 0.0);
    DifferentialPair<float> dpQy = diffPair(qy, 0.0);
    DifferentialPair<float> dpQz = diffPair(qz, 0.0);
    DifferentialPair<float> dpSx = diffPair(sx, 0.0);
    DifferentialPair<float> dpSy = diffPair(sy, 0.0);
    DifferentialPair<float> dpSz = diffPair(sz, 0.0);
    bwd_diff(project_to_screen)(
        dpWx, dpWy, dpWz,
        dpQw, dpQx, dpQy, dpQz,
        dpSx, dpSy, dpSz,
        r00, r01, r02, r10, r11, r12, r20, r21, r22, tx, ty, tz,
        fx, fy, cx, cy, imageWidth, imageHeight,
        cot
    );
    ProjectScreenGrad g;
    g.wx = dpWx.d; g.wy = dpWy.d; g.wz = dpWz.d;
    g.qw = dpQw.d; g.qx = dpQx.d; g.qy = dpQy.d; g.qz = dpQz.d;
    g.sx = dpSx.d; g.sy = dpSy.d; g.sz = dpSz.d;
    return g;
}

public ShEvalGrad sh_eval_deg3_grad(
    int degree, float3 dir, SHBands16 sh,
    float3 cot
)
{
    DifferentialPair<float3> dpDir = diffPair(dir, float3(0.0, 0.0, 0.0));
    SHBands16 zero;
    zero.b0 = float3(0.0); zero.b1 = float3(0.0); zero.b2 = float3(0.0); zero.b3 = float3(0.0);
    zero.b4 = float3(0.0); zero.b5 = float3(0.0); zero.b6 = float3(0.0); zero.b7 = float3(0.0);
    zero.b8 = float3(0.0); zero.b9 = float3(0.0); zero.b10 = float3(0.0); zero.b11 = float3(0.0);
    zero.b12 = float3(0.0); zero.b13 = float3(0.0); zero.b14 = float3(0.0); zero.b15 = float3(0.0);
    DifferentialPair<SHBands16> dpSh = diffPair(sh, zero);
    bwd_diff(sh_eval_deg3)(degree, dpDir, dpSh, cot);
    ShEvalGrad g;
    g.dir = dpDir.d;
    g.sh = dpSh.d;
    return g;
}

[Differentiable]
public float3 sh_eval_deg3(no_diff int degree, float3 dir, SHBands16 sh)
{
    const float C0    = 0.28209479177387814;
    const float C1    = 0.4886025119029199;
    const float C2_0  = 1.0925484305920792;
    const float C2_1  = -1.0925484305920792;
    const float C2_2  = 0.31539156525252005;
    const float C2_3  = -1.0925484305920792;
    const float C2_4  = 0.5462742152960396;
    const float C3_0  = -0.5900435899266435;
    const float C3_1  = 2.890611442640554;
    const float C3_2  = -0.4570457994644658;
    const float C3_3  = 0.3731763325901154;
    const float C3_4  = -0.4570457994644658;
    const float C3_5  = 1.445305721320277;
    const float C3_6  = -0.5900435899266435;

    float invLen = rsqrt(dot(dir, dir));
    float x = dir.x * invLen;
    float y = dir.y * invLen;
    float z = dir.z * invLen;

    float3 result = C0 * sh.b0;

    if (degree >= 1) {
        result = result
            + (-C1 * y) * sh.b1
            + ( C1 * z) * sh.b2
            + (-C1 * x) * sh.b3;
    }
    if (degree >= 2) {
        float xx = x * x, yy = y * y, zz = z * z;
        float xy = x * y, yz = y * z, xz = x * z;
        result = result
            + (C2_0 * xy)             * sh.b4
            + (C2_1 * yz)             * sh.b5
            + (C2_2 * (2.0 * zz - xx - yy)) * sh.b6
            + (C2_3 * xz)             * sh.b7
            + (C2_4 * (xx - yy))      * sh.b8;
    }
    if (degree >= 3) {
        float xx = x * x, yy = y * y, zz = z * z;
        float xy = x * y, yz = y * z, xz = x * z;
        result = result
            + (C3_0 * y * (3.0 * xx - yy))              * sh.b9
            + (C3_1 * xy * z)                           * sh.b10
            + (C3_2 * y * (4.0 * zz - xx - yy))         * sh.b11
            + (C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)) * sh.b12
            + (C3_4 * x * (4.0 * zz - xx - yy))         * sh.b13
            + (C3_5 * z * (xx - yy))                    * sh.b14
            + (C3_6 * x * (xx - 3.0 * yy))              * sh.b15;
    }

    result = result + float3(0.5, 0.5, 0.5);
    return max(result, float3(0.0, 0.0, 0.0));
}
