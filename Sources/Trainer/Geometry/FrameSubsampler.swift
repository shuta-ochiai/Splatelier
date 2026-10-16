// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Information-Gated Frame Subsampling (PocketGS Operator G 3a, Sec. III-A 1).
// 1. displacement gate: keep only frames with ‖ΔC‖ ≥ τ_d from the last accepted one (drops
//    low-parallax frames). MVS uses this alone and keeps every frame that passes it.
// 2. sharpness: measure blur by gradient energy on a sparse luma grid (eq. 1). Used by the
//    capture-side frame filters.

import Foundation
import MLX

/// Frame selection helpers: the displacement gate and the sharpness score.
public enum FrameSubsampler {

    /// Camera center C = -Rᵀ t (world coordinates) from a row-major world→camera matrix.
    public static func cameraCenter(viewmat: [Float]) -> SIMD3<Float> {
        precondition(viewmat.count == 16, "viewmat must be a 16-element row-major matrix")
        return SIMD3<Float>(
            -(viewmat[0] * viewmat[3] + viewmat[4] * viewmat[7] + viewmat[8]  * viewmat[11]),
            -(viewmat[1] * viewmat[3] + viewmat[5] * viewmat[7] + viewmat[9]  * viewmat[11]),
            -(viewmat[2] * viewmat[3] + viewmat[6] * viewmat[7] + viewmat[10] * viewmat[11])
        )
    }

    /// true only for frames at least `tauD` from the previous accepted frame. The first is always
    /// accepted.
    public static func displacementGate(
        centers: [SIMD3<Float>], tauD: Float
    ) -> [Bool] {
        var kept = [Bool](repeating: false, count: centers.count)
        guard !centers.isEmpty else { return kept }
        kept[0] = true
        var last = centers[0]
        let tau2 = tauD * tauD
        for i in 1..<centers.count {
            let d = centers[i] - last
            let d2 = d.x * d.x + d.y * d.y + d.z * d.z
            if d2 >= tau2 {
                kept[i] = true
                last = centers[i]
            }
        }
        return kept
    }

    /// Gradient energy on a sparse luma grid (PocketGS eq. 1). Larger is sharper.
    /// `rgb` is `[H, W, 3]` float.
    public static func sharpnessScore(rgb: MLXArray, stride: Int) -> Float {
        precondition(rgb.shape.count == 3 && rgb.shape[2] == 3,
                     "rgb must be [H, W, 3]")
        precondition(stride >= 1, "stride must be >= 1")
        let h = rgb.shape[0]
        let w = rgb.shape[1]
        if h <= stride || w <= stride { return 0 }
        let r = rgb[0..., 0..., 0]
        let g = rgb[0..., 0..., 1]
        let b = rgb[0..., 0..., 2]
        let luma = 0.299 * r + 0.587 * g + 0.114 * b                  // [H, W]

        // Take differences Δ apart over all pixels, then subsample on the stride-Δ grid Ω.
        let diffX = MLX.abs(
            luma[stride..<h, 0...] - luma[0..<(h - stride), 0...]
        )                                                              // [H-Δ, W]
        let diffY = MLX.abs(
            luma[0..., stride..<w] - luma[0..., 0..<(w - stride)]
        )                                                              // [H, W-Δ]

        let dxStrided = diffX[.stride(from: 0, to: nil, by: stride), 0...]
        let dyStrided = diffY[0..., .stride(from: 0, to: nil, by: stride)]
        let scoreArr = dxStrided.mean() + dyStrided.mean()
        return scoreArr.item(Float.self)
    }

    /// Runs the displacement gate and returns the original indices of the accepted frames, in
    /// order. `tauD` is the gate threshold (m; paper value 0.05), which depends on scene scale.
    public static func selectFrames(viewmats: [[Float]], tauD: Float = 0.05) -> [Int] {
        precondition(tauD >= 0, "tauD must be non-negative")
        let kept = displacementGate(centers: viewmats.map { cameraCenter(viewmat: $0) }, tauD: tauD)
        return kept.indices.filter { kept[$0] }
    }
}
