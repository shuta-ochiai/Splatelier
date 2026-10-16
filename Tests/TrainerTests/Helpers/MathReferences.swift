// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// MLX-Swift reference implementations of the math primitives shipped as Slang kernels. Tests use
// them as ground truth for Slang output.

import Foundation
import MLX

/// quaternion (w, x, y, z) → 3x3 rotation matrix. [..., 4] → [..., 3, 3].
/// Same formula as slang/quat_to_rotmat.slang (follows gsplat cuda/include/Utils.cuh, Apache-2.0).
func referenceQuatToRotmat(_ quats: MLXArray) -> MLXArray {
    let invNorm = MLX.rsqrt((quats * quats).sum(axis: -1, keepDims: true))
    let q = quats * invNorm
    let w = q[.ellipsis, 0]
    let x = q[.ellipsis, 1]
    let y = q[.ellipsis, 2]
    let z = q[.ellipsis, 3]
    let x2 = x * x, y2 = y * y, z2 = z * z
    let xy = x * y, xz = x * z, yz = y * z
    let wx = w * x, wy = w * y, wz = w * z

    // Standard active rotation (row-major). Matches slang/gs_math.slang.
    let row0 = MLX.stacked(
        [1.0 - 2.0 * (y2 + z2), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        axis: -1
    )
    let row1 = MLX.stacked(
        [2.0 * (xy + wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz - wx)],
        axis: -1
    )
    let row2 = MLX.stacked(
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (x2 + y2)],
        axis: -1
    )
    return MLX.stacked([row0, row1, row2], axis: -2)
}

/// (quat, scale) → upper triangle of 3x3 covariance. quats [N, 4], scales [N, 3] → [N, 6] (c00,
/// c01, c02, c11, c12, c22). Same formula as slang/quat_scale_to_cov3d.slang.
func referenceQuatScaleToCov3d(_ quats: MLXArray, _ scales: MLXArray) -> MLXArray {
    let r = referenceQuatToRotmat(quats)                       // [N, 3, 3]
    // M[i, j] = R[i, j] * s[j] (broadcast scales over row axis -2).
    let s = scales.expandedDimensions(axis: -2)                // [N, 1, 3]
    let m = r * s                                              // [N, 3, 3]
    // C = M·Mᵀ as a multiply and sum (a float32 matmul runs in TF32 on GPUs with matrix units).
    let c = (m.expandedDimensions(axis: -2) * m.expandedDimensions(axis: -3))
        .sum(axis: -1)                                         // [N, 3, 3]
    let c00 = c[.ellipsis, 0, 0]
    let c01 = c[.ellipsis, 0, 1]
    let c02 = c[.ellipsis, 0, 2]
    let c11 = c[.ellipsis, 1, 1]
    let c12 = c[.ellipsis, 1, 2]
    let c22 = c[.ellipsis, 2, 2]
    return MLX.stacked([c00, c01, c02, c11, c12, c22], axis: -1)
}

/// Random row-major 4x4 viewmat ([16]). Returns the MLXArray and a host copy for finite
/// differences.
func makeViewMatrix(seed: UInt64) -> (mlx: MLXArray, host: [Float]) {
    MLXRandom.seed(seed)
    let q = MLXRandom.normal([4]) * 0.8 + 0.3
    let invNorm = MLX.rsqrt((q * q).sum())
    let qn = q * invNorm
    let w = qn[0].item(Float.self)
    let x = qn[1].item(Float.self)
    let y = qn[2].item(Float.self)
    let z = qn[3].item(Float.self)
    let r00: Float = 1 - 2 * (y * y + z * z)
    let r01: Float = 2 * (x * y + w * z)
    let r02: Float = 2 * (x * z - w * y)
    let r10: Float = 2 * (x * y - w * z)
    let r11: Float = 1 - 2 * (x * x + z * z)
    let r12: Float = 2 * (y * z + w * x)
    let r20: Float = 2 * (x * z + w * y)
    let r21: Float = 2 * (y * z - w * x)
    let r22: Float = 1 - 2 * (x * x + y * y)
    let tx = MLXRandom.normal([1])[0].item(Float.self) * 0.3
    let ty = MLXRandom.normal([1])[0].item(Float.self) * 0.3
    let tz = MLXRandom.normal([1])[0].item(Float.self) * 0.3 + 4.0   // pull the camera back
    let host: [Float] = [
        r00, r01, r02, tx,
        r10, r11, r12, ty,
        r20, r21, r22, tz,
        0,   0,   0,   1
    ]
    return (MLXArray(host, [16]), host)
}

/// world → camera transform (same formula as slang/world_to_cam.slang).
/// means3dWorld [N, 3], cov3dsWorld [N, 6], viewmat [16] → means3dCam [N, 3], cov3dsCam [N, 6].
func referenceWorldToCam(
    means3dWorld: MLXArray, cov3dsWorld: MLXArray, viewmat: [Float]
) -> (means3dCam: MLXArray, cov3dsCam: MLXArray) {
    let r00 = viewmat[0], r01 = viewmat[1], r02 = viewmat[2], tx = viewmat[3]
    let r10 = viewmat[4], r11 = viewmat[5], r12 = viewmat[6], ty = viewmat[7]
    let r20 = viewmat[8], r21 = viewmat[9], r22 = viewmat[10], tz = viewmat[11]

    let wx = means3dWorld[.ellipsis, 0]
    let wy = means3dWorld[.ellipsis, 1]
    let wz = means3dWorld[.ellipsis, 2]

    let mxCam = r00 * wx + r01 * wy + r02 * wz + tx
    let myCam = r10 * wx + r11 * wy + r12 * wz + ty
    let mzCam = r20 * wx + r21 * wy + r22 * wz + tz
    let means3dCam = MLX.stacked([mxCam, myCam, mzCam], axis: -1)

    let a00 = cov3dsWorld[.ellipsis, 0]
    let a01 = cov3dsWorld[.ellipsis, 1]
    let a02 = cov3dsWorld[.ellipsis, 2]
    let a11 = cov3dsWorld[.ellipsis, 3]
    let a12 = cov3dsWorld[.ellipsis, 4]
    let a22 = cov3dsWorld[.ellipsis, 5]

    // M = R * cov_world (3x3). Only the 6 upper-triangle elements of M * R^T are needed.
    let m00 = r00 * a00 + r01 * a01 + r02 * a02
    let m01 = r00 * a01 + r01 * a11 + r02 * a12
    let m02 = r00 * a02 + r01 * a12 + r02 * a22
    let m10 = r10 * a00 + r11 * a01 + r12 * a02
    let m11 = r10 * a01 + r11 * a11 + r12 * a12
    let m12 = r10 * a02 + r11 * a12 + r12 * a22
    let m20 = r20 * a00 + r21 * a01 + r22 * a02
    let m21 = r20 * a01 + r21 * a11 + r22 * a12
    let m22 = r20 * a02 + r21 * a12 + r22 * a22

    let c00 = m00 * r00 + m01 * r01 + m02 * r02
    let c01 = m00 * r10 + m01 * r11 + m02 * r12
    let c02 = m00 * r20 + m01 * r21 + m02 * r22
    let c11 = m10 * r10 + m11 * r11 + m12 * r12
    let c12 = m10 * r20 + m11 * r21 + m12 * r22
    let c22 = m20 * r20 + m21 * r21 + m22 * r22
    let cov3dsCam = MLX.stacked([c00, c01, c02, c11, c12, c22], axis: -1)

    return (means3dCam, cov3dsCam)
}

/// Same composite projection as slang/project_to_screen.slang (cov3d → world_to_cam → persp_proj →
/// 2x2 inverse). With `opacities` nil: no opacity cull and extend = GAUSSIAN_EXTEND; when given,
/// radius tightening and cull are included. Output: means2d [N, 2], conics [N, 3] (upper triangle
/// a, b, c of Σ_2d⁻¹), depths [N], radii [N, 2] (rx, ry; 0 if opacity < 1/255 or covariance ≤ 0).
func referenceProjectToScreen(
    means3dWorld: MLXArray, quats: MLXArray, scales: MLXArray,
    viewmat: [Float],
    fx: Float, fy: Float, cx: Float, cy: Float,
    imageWidth: Float, imageHeight: Float,
    opacities: MLXArray? = nil
) -> (means2d: MLXArray, conics: MLXArray, depths: MLXArray, radii: MLXArray) {
    let cov3dW = referenceQuatScaleToCov3d(quats, scales)
    let (means3dCam, cov3dsCam) = referenceWorldToCam(
        means3dWorld: means3dWorld, cov3dsWorld: cov3dW, viewmat: viewmat
    )
    let (means2d, cov2ds) = referencePerspProj(
        means3dCam: means3dCam, cov3dsCam: cov3dsCam,
        fx: fx, fy: fy, cx: cx, cy: cy,
        imageWidth: imageWidth, imageHeight: imageHeight
    )

    let a = cov2ds[.ellipsis, 0]
    let b = cov2ds[.ellipsis, 1]
    let c = cov2ds[.ellipsis, 2]
    let det = a * c - b * b
    let invDet = 1.0 / det
    let conicA = c * invDet
    let conicB = -b * invDet
    let conicC = a * invDet
    let conics = MLX.stacked([conicA, conicB, conicC], axis: -1)

    let depths = means3dCam[.ellipsis, 2]

    // Per-axis radius (gsplat ProjectionEWA3DGSFused.cu:186-206).
    let ALPHA_THRESHOLD: Float = 1.0 / 255.0
    let GAUSSIAN_EXTEND: Float = 3.33
    let aPos = MLX.maximum(a, MLXArray(0.0 as Float))
    let cPos = MLX.maximum(c, MLXArray(0.0 as Float))
    let radiiX: MLXArray
    let radiiY: MLXArray
    if let opacities {
        let extend = MLX.minimum(
            MLXArray(GAUSSIAN_EXTEND),
            MLX.sqrt(2.0 * MLX.log(opacities / ALPHA_THRESHOLD))
        )
        let rx = MLX.ceil(extend * MLX.sqrt(aPos))
        let ry = MLX.ceil(extend * MLX.sqrt(cPos))
        let live = opacities .>= MLXArray(ALPHA_THRESHOLD)
        radiiX = MLX.where(live, rx, MLXArray(0.0 as Float))
        radiiY = MLX.where(live, ry, MLXArray(0.0 as Float))
    } else {
        radiiX = MLX.ceil(GAUSSIAN_EXTEND * MLX.sqrt(aPos))
        radiiY = MLX.ceil(GAUSSIAN_EXTEND * MLX.sqrt(cPos))
    }
    let radii = MLX.stacked([radiiX, radiiY], axis: -1).asType(.int32)

    return (means2d, conics, depths, radii)
}

/// SH evaluation (degree 0..3, up to 16 bands, same formula as slang/sh_eval_deg3.slang). Returns
/// max(0, color + 0.5). dirs [N, 3] are normalized internally, coeffs is [N, 16, 3] or [N, 48],
/// degree is shared by all points.
func referenceShEvalDeg3(degree: Int, dirs: MLXArray, coeffs: MLXArray) -> MLXArray {
    let n = dirs.shape[0]
    let c = coeffs.reshaped([n, 16, 3])

    let invLen = MLX.rsqrt((dirs * dirs).sum(axis: -1, keepDims: true))
    let dn = dirs * invLen
    let x = dn[.ellipsis, 0]                    // [N]
    let y = dn[.ellipsis, 1]
    let z = dn[.ellipsis, 2]

    let C0: Float    = 0.28209479177387814
    let C1: Float    = 0.4886025119029199
    let C2_0: Float  =  1.0925484305920792
    let C2_1: Float  = -1.0925484305920792
    let C2_2: Float  =  0.31539156525252005
    let C2_3: Float  = -1.0925484305920792
    let C2_4: Float  =  0.5462742152960396
    let C3_0: Float  = -0.5900435899266435
    let C3_1: Float  =  2.890611442640554
    let C3_2: Float  = -0.4570457994644658
    let C3_3: Float  =  0.3731763325901154
    let C3_4: Float  = -0.4570457994644658
    let C3_5: Float  =  1.445305721320277
    let C3_6: Float  = -0.5900435899266435

    let xx = x * x, yy = y * y, zz = z * z
    let xy = x * y, yz = y * z, xz = x * z

    // 16 basis values per Gaussian. Bands outside `degree` are set to 0.
    let b0  = MLXArray(Array(repeating: C0, count: n), [n])
    let b1  = -C1 * y
    let b2  = C1 * z
    let b3  = -C1 * x
    let b4  = C2_0 * xy
    let b5  = C2_1 * yz
    let b6  = C2_2 * (2.0 * zz - xx - yy)
    let b7  = C2_3 * xz
    let b8  = C2_4 * (xx - yy)
    let b9  = C3_0 * y * (3.0 * xx - yy)
    let b10 = C3_1 * xy * z
    let b11 = C3_2 * y * (4.0 * zz - xx - yy)
    let b12 = C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)
    let b13 = C3_4 * x * (4.0 * zz - xx - yy)
    let b14 = C3_5 * z * (xx - yy)
    let b15 = C3_6 * x * (xx - 3.0 * yy)

    let basis = MLX.stacked(
        [b0, b1, b2, b3, b4, b5, b6, b7,
         b8, b9, b10, b11, b12, b13, b14, b15],
        axis: -1
    ) // [N, 16]

    let totalK = (degree + 1) * (degree + 1)
    let maskVals: [Float] = (0..<16).map { $0 < totalK ? 1 : 0 }
    let mask = MLXArray(maskVals, [16])
    let maskedBasis = basis * mask                // [N, 16]

    // color[i, c] = sum_k basis[i, k] * coeffs[i, k, c]
    let b3d = maskedBasis.expandedDimensions(axis: 1)   // [N, 1, 16]
    let color = MLX.matmul(b3d, c).squeezed(axis: 1)    // [N, 3]

    return MLX.maximum(color + 0.5, MLXArray(0.0 as Float))
}

/// EWA perspective projection (same formula as slang/persp_proj.slang). means3dCam [N, 3],
/// cov3dsCam [N, 6] → means2d [N, 2], cov2ds [N, 3] (c00, c01, c11). imageWidth / imageHeight are
/// used to clamp the EWA Jacobian's x/z, y/z to the frustum (+0.3·tan_fov) (same as gsplat
/// persp_proj; keeps the 2D footprint finite even outside the frustum).
func referencePerspProj(
    means3dCam: MLXArray, cov3dsCam: MLXArray,
    fx: Float, fy: Float, cx: Float, cy: Float,
    imageWidth: Float, imageHeight: Float
) -> (means2d: MLXArray, cov2ds: MLXArray) {
    let mx = means3dCam[.ellipsis, 0]
    let my = means3dCam[.ellipsis, 1]
    let mz = means3dCam[.ellipsis, 2]
    let rz = 1.0 / mz
    let rz2 = rz * rz

    // gsplat Utils.cuh:558-571: clamp x/z, y/z to the frustum (+0.3·tan_fov) before the Jacobian.
    // Without it, j02/j12 blow up for off-screen Gaussians, the 2D covariance explodes and causes
    // streak artifacts.
    let tanFovx = 0.5 * imageWidth / fx
    let tanFovy = 0.5 * imageHeight / fy
    let limXPos = (imageWidth - cx) / fx + 0.3 * tanFovx
    let limXNeg = cx / fx + 0.3 * tanFovx
    let limYPos = (imageHeight - cy) / fy + 0.3 * tanFovy
    let limYNeg = cy / fy + 0.3 * tanFovy
    let tx = mz * MLX.minimum(MLXArray(limXPos), MLX.maximum(MLXArray(-limXNeg), mx * rz))
    let ty = mz * MLX.minimum(MLXArray(limYPos), MLX.maximum(MLXArray(-limYNeg), my * rz))

    let j00 = fx * rz
    let j02 = -fx * tx * rz2
    let j11 = fy * rz
    let j12 = -fy * ty * rz2

    let a00 = cov3dsCam[.ellipsis, 0]
    let a01 = cov3dsCam[.ellipsis, 1]
    let a02 = cov3dsCam[.ellipsis, 2]
    let a11 = cov3dsCam[.ellipsis, 3]
    let a12 = cov3dsCam[.ellipsis, 4]
    let a22 = cov3dsCam[.ellipsis, 5]

    // T = J * cov3d (2x3): rows i = 0, 1
    let t00 = j00 * a00 + j02 * a02
    let t01 = j00 * a01 + j02 * a12
    let t02 = j00 * a02 + j02 * a22
    let t11 = j11 * a11 + j12 * a12
    let t12 = j11 * a12 + j12 * a22

    // cov2d = T * J^T (2x2 symmetric)
    let c00 = t00 * j00 + t02 * j02
    let c01 = t01 * j11 + t02 * j12
    let c11 = t11 * j11 + t12 * j12

    let px = fx * mx * rz + cx
    let py = fy * my * rz + cy

    let means2d = MLX.stacked([px, py], axis: -1)
    let cov2ds  = MLX.stacked([c00, c01, c11], axis: -1)
    return (means2d, cov2ds)
}
