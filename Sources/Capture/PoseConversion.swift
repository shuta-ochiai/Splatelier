// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Coordinate convention conversion ARKit ↔ trainer (simd only, testable without ARKit). ARKit:
// column-major c2w, camera +Y up / −Z forward (OpenGL). trainer: row-major [Float] 16 w2c, camera
// +Y down / +Z forward (OpenCV / gsplat). Without the Y/Z flip every forward point gets mz ≤ 0, the
// near-plane cull removes them, and training is a no-op. Loop closure corrections are not reflected
// in ARCamera.transform; only ARAnchor.transform is updated.

import Foundation
import simd
import Trainer

public enum PoseConversion {

    /// Negates Y/Z of the camera frame (OpenGL → OpenCV). Self-inverse, so the same matrix serves
    /// as the inverse.
    private static let arkitToCV = simd_float4x4(diagonal: SIMD4<Float>(1, -1, -1, 1))

    /// Converts an ARKit c2w to the trainer's row-major w2c (OpenCV camera).
    public static func viewmat(fromCameraToWorld c2w: simd_float4x4) -> [Float] {
        let w2cARKit = c2w.inverse
        let w2cTrainer = arkitToCV * w2cARKit
        return rowMajor16(w2cTrainer)
    }

    /// Directly converts a w2c such as `ARCamera.viewMatrix(for:)` (no inversion needed).
    public static func viewmat(fromWorldToCamera w2c: simd_float4x4) -> [Float] {
        return rowMajor16(arkitToCV * w2c)
    }

    /// Inverse transform from viewmat (row-major, OpenCV w2c) back to ARKit c2w.
    public static func cameraToWorld(fromViewmat viewmat: [Float]) -> simd_float4x4 {
        precondition(viewmat.count == 16, "viewmat must be a 16-element row-major matrix")
        let w2cTrainer = simd_float4x4(
            SIMD4<Float>(viewmat[0],  viewmat[4],  viewmat[8],  viewmat[12]),   // col 0
            SIMD4<Float>(viewmat[1],  viewmat[5],  viewmat[9],  viewmat[13]),   // col 1
            SIMD4<Float>(viewmat[2],  viewmat[6],  viewmat[10], viewmat[14]),   // col 2
            SIMD4<Float>(viewmat[3],  viewmat[7],  viewmat[11], viewmat[15])    // col 3
        )
        return w2cTrainer.inverse * arkitToCV
    }

    /// Scales `ARCamera.intrinsics` (relative to imageResolution) to the downscaled resolution.
    public static func scaledIntrinsics(
        fx: Float, fy: Float, cx: Float, cy: Float,
        originalWidth: Int, originalHeight: Int,
        scaledWidth: Int, scaledHeight: Int
    ) -> Intrinsics {
        precondition(originalWidth > 0 && originalHeight > 0)
        precondition(scaledWidth > 0 && scaledHeight > 0)
        let sx = Float(scaledWidth) / Float(originalWidth)
        let sy = Float(scaledHeight) / Float(originalHeight)
        return (fx * sx, fy * sy, cx * sx, cy * sy)
    }

    // MARK: - private

    private static func rowMajor16(_ m: simd_float4x4) -> [Float] {
        return [
            m.columns.0.x, m.columns.1.x, m.columns.2.x, m.columns.3.x,   // row 0
            m.columns.0.y, m.columns.1.y, m.columns.2.y, m.columns.3.y,   // row 1
            m.columns.0.z, m.columns.1.z, m.columns.2.z, m.columns.3.z,   // row 2
            m.columns.0.w, m.columns.1.w, m.columns.2.w, m.columns.3.w,   // row 3
        ]
    }
}
