// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXFast

// Small matrix ops for BA (3×3 / 6×3 blocks, SE(3) left perturbation, Rodrigues).
extension BundleAdjustment {

    /// Inverse of a 3×3 (row-major) matrix (via cofactors). Returns identity if |det| < 1e-12.
    static func invert3x3(a: [Float], into: inout [Float], offset: Int) {
        let m00 = a[0], m01 = a[1], m02 = a[2]
        let m10 = a[3], m11 = a[4], m12 = a[5]
        let m20 = a[6], m21 = a[7], m22 = a[8]
        let c00 = m11 * m22 - m12 * m21
        let c01 = -(m10 * m22 - m12 * m20)
        let c02 = m10 * m21 - m11 * m20
        let det = m00 * c00 + m01 * c01 + m02 * c02
        if abs(det) < 1e-12 {
            into[offset + 0] = 1; into[offset + 1] = 0; into[offset + 2] = 0
            into[offset + 3] = 0; into[offset + 4] = 1; into[offset + 5] = 0
            into[offset + 6] = 0; into[offset + 7] = 0; into[offset + 8] = 1
            return
        }
        let invDet = 1.0 / det
        into[offset + 0] = c00 * invDet
        into[offset + 1] = -(m01 * m22 - m02 * m21) * invDet
        into[offset + 2] =  (m01 * m12 - m02 * m11) * invDet
        into[offset + 3] = c01 * invDet
        into[offset + 4] =  (m00 * m22 - m02 * m20) * invDet
        into[offset + 5] = -(m00 * m12 - m02 * m10) * invDet
        into[offset + 6] = c02 * invDet
        into[offset + 7] = -(m00 * m21 - m01 * m20) * invDet
        into[offset + 8] =  (m00 * m11 - m01 * m10) * invDet
    }

    /// 3×3 (row-major) at `m[matrixOffset...]` · v.
    static func matVec3x3(matrixOffset: Int, m: [Float], v: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3<Float>(
            m[matrixOffset + 0] * v.x + m[matrixOffset + 1] * v.y + m[matrixOffset + 2] * v.z,
            m[matrixOffset + 3] * v.x + m[matrixOffset + 4] * v.y + m[matrixOffset + 5] * v.z,
            m[matrixOffset + 6] * v.x + m[matrixOffset + 7] * v.y + m[matrixOffset + 8] * v.z
        )
    }

    /// Stand-in because Swift has no SIMD6.
    struct SIMD6Floats {
        let v0, v1, v2, v3, v4, v5: Float
        init(_ a: Float, _ b: Float, _ c: Float, _ d: Float, _ e: Float, _ f: Float) {
            v0 = a; v1 = b; v2 = c; v3 = d; v4 = e; v5 = f
        }
    }

    /// (6×3) · v → 6 elements.
    static func matVec6x3(m6x3: [Float], v: SIMD3<Float>) -> [Float] {
        var out = [Float](repeating: 0, count: 6)
        for r in 0..<6 {
            out[r] = m6x3[r * 3 + 0] * v.x + m6x3[r * 3 + 1] * v.y + m6x3[r * 3 + 2] * v.z
        }
        return out
    }

    /// H_pp⁻¹ (3×3) · H_tpᵀ (transpose of 6×3) → 3×6 row-major.
    static func matMul3x3_3x6T(
        matrixOffset: Int, m: [Float], n6x3: [Float]
    ) -> [Float] {
        var out = [Float](repeating: 0, count: 18)   // 3×6
        for r in 0..<3 {
            for c in 0..<6 {
                var s: Float = 0
                for k in 0..<3 {
                    s += m[matrixOffset + r * 3 + k] * n6x3[c * 3 + k]
                }
                out[r * 6 + c] = s
            }
        }
        return out
    }

    /// (6×3) · (3×6) → 6×6 row-major.
    static func matMul6x3_3x6(a6x3: [Float], b3x6: [Float]) -> [Float] {
        var out = [Float](repeating: 0, count: 36)
        for r in 0..<6 {
            for c in 0..<6 {
                var s: Float = 0
                for k in 0..<3 {
                    s += a6x3[r * 3 + k] * b3x6[k * 6 + c]
                }
                out[r * 6 + c] = s
            }
        }
        return out
    }

    /// (6×3)ᵀ · v6 → 3 elements.
    static func matVec3x6Transpose(m6x3: [Float], v6: SIMD6Floats) -> SIMD3<Float> {
        let vec = [v6.v0, v6.v1, v6.v2, v6.v3, v6.v4, v6.v5]
        var out = SIMD3<Float>(0, 0, 0)
        for c in 0..<3 {
            var s: Float = 0
            for r in 0..<6 {
                s += m6x3[r * 3 + c] * vec[r]
            }
            switch c {
            case 0: out.x = s
            case 1: out.y = s
            default: out.z = s
            }
        }
        return out
    }

    /// Left perturbation T_new = exp_se3([ω, v]) · T_old (row-major 4×4). Assumes ω, v are small.
    static func applyLeftSE3Perturbation(
        viewmat: [Float], omega: SIMD3<Float>, v: SIMD3<Float>
    ) -> [Float] {
        // Approximate the translational left Jacobian V as V ≈ I, so t_new = R_pert · t_old + v.
        let R_pert = rodriguesRotation(omega: omega)
        // R_new = R_pert · R_old
        let R00 = R_pert[0] * viewmat[0] + R_pert[1] * viewmat[4] + R_pert[2] * viewmat[8]
        let R01 = R_pert[0] * viewmat[1] + R_pert[1] * viewmat[5] + R_pert[2] * viewmat[9]
        let R02 = R_pert[0] * viewmat[2] + R_pert[1] * viewmat[6] + R_pert[2] * viewmat[10]
        let R10 = R_pert[3] * viewmat[0] + R_pert[4] * viewmat[4] + R_pert[5] * viewmat[8]
        let R11 = R_pert[3] * viewmat[1] + R_pert[4] * viewmat[5] + R_pert[5] * viewmat[9]
        let R12 = R_pert[3] * viewmat[2] + R_pert[4] * viewmat[6] + R_pert[5] * viewmat[10]
        let R20 = R_pert[6] * viewmat[0] + R_pert[7] * viewmat[4] + R_pert[8] * viewmat[8]
        let R21 = R_pert[6] * viewmat[1] + R_pert[7] * viewmat[5] + R_pert[8] * viewmat[9]
        let R22 = R_pert[6] * viewmat[2] + R_pert[7] * viewmat[6] + R_pert[8] * viewmat[10]
        let t0 = R_pert[0] * viewmat[3] + R_pert[1] * viewmat[7] + R_pert[2] * viewmat[11] + v.x
        let t1 = R_pert[3] * viewmat[3] + R_pert[4] * viewmat[7] + R_pert[5] * viewmat[11] + v.y
        let t2 = R_pert[6] * viewmat[3] + R_pert[7] * viewmat[7] + R_pert[8] * viewmat[11] + v.z
        return [
            R00, R01, R02, t0,
            R10, R11, R12, t1,
            R20, R21, R22, t2,
            0, 0, 0, 1,
        ]
    }

    /// Rodrigues: axis-angle ω → 3×3 (row-major). First-order approximation for θ < 1e-6.
    static func rodriguesRotation(omega: SIMD3<Float>) -> [Float] {
        let theta2 = omega.x * omega.x + omega.y * omega.y + omega.z * omega.z
        let theta = theta2.squareRoot()
        if theta < 1e-6 {
            // R ≈ I + [ω]_×
            return [
                1, -omega.z, omega.y,
                omega.z, 1, -omega.x,
                -omega.y, omega.x, 1,
            ]
        }
        let kx = omega.x / theta
        let ky = omega.y / theta
        let kz = omega.z / theta
        let s = sin(theta)
        let c = cos(theta)
        let oneMinusC = 1 - c
        // R = I + sin(θ) K + (1 - cos(θ)) K²
        let r00 = c + kx * kx * oneMinusC
        let r01 = kx * ky * oneMinusC - kz * s
        let r02 = kx * kz * oneMinusC + ky * s
        let r10 = ky * kx * oneMinusC + kz * s
        let r11 = c + ky * ky * oneMinusC
        let r12 = ky * kz * oneMinusC - kx * s
        let r20 = kz * kx * oneMinusC - ky * s
        let r21 = kz * ky * oneMinusC + kx * s
        let r22 = c + kz * kz * oneMinusC
        return [r00, r01, r02, r10, r11, r12, r20, r21, r22]
    }
}
