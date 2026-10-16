// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import Synchronization

// Coarse-to-fine resolution conversion (internal, since tests use it).
extension TrainingDriver {

    /// Shrinks the camera by 1/factor (viewmat unchanged, intrinsics and resolution × 1/factor).
    /// Projection is linear in resolution, so gradient directions are preserved.
    static func coarseCamera(_ cam: Camera, factor f: Int) -> Camera {
        Camera(
            viewmat: cam.viewmat, intrinsics: cam.intrinsics * (1.0 / Float(f)),
            width: cam.width / f, height: cam.height / f,
            tileSize: cam.tileSize, background: cam.background
        )
    }

    /// Downscales target [H, W, 3] by a factor×factor box average (edges that don't divide evenly
    /// are cropped). Takes uint8 directly (never builds a full-res float32); the result is float
    /// ([0,1] for uint8). The box average is the area resampling 1/factor.
    static func coarseDownsample(_ img: MLXArray, factor f: Int) -> MLXArray {
        resampleRational(img, num: 1, den: f)
    }

    /// Area resampling by the rational ratio num/den (output dim = srcDim/den*num, edges cropped).
    /// Accepts uint8 input (result /255). num>=den is identity scale.
    ///
    /// Each output pixel is a weighted sum of the phases of its den-pixel block (strided views), so
    /// the whole resample is elementwise work that `compile` fuses into one kernel, with no
    /// transposed copies and no full-res float32. It runs every step (the step target), and the
    /// earlier matmul formulation cost about 4× the GPU energy (2026-10-13 Mac: fine 2/3 8.4 → 2.4
    /// mJ, coarse 1/3 6.1 → 1.2 mJ) while being accurate only to ~1e-3 (float32 matmul on the GPU).
    static func resampleRational(_ img: MLXArray, num: Int, den: Int) -> MLXArray {
        if num >= den { return img.dtype == .uint8 ? img.asType(.float32) / 255.0 : img }
        let h = img.shape[0], w = img.shape[1]
        let hC = (h / den) * den, wC = (w / den) * den
        let cropped = (hC == h && wC == w) ? img : img[0..<hC, 0..<wC, 0...]
        return areaResampler(num: num, den: den)([cropped])[0]
    }

    /// Compiled `resampleRational` per (num, den); `compile` retraces per input shape and dtype.
    private static let areaResamplers = Mutex<[Int: @Sendable ([MLXArray]) -> [MLXArray]]>([:])

    private static func areaResampler(num: Int, den: Int) -> @Sendable ([MLXArray]) -> [MLXArray] {
        areaResamplers.withLock { cache in
            let key = num << 16 | den
            if let f = cache[key] { return f }
            let weights = areaWeight(den: den, num: num).asArray(Float.self)   // [den, num]
            // out_j = Σ_i weights[i, j] · phase_i, skipping zero weights.
            @Sendable func blend(_ phases: [MLXArray], _ j: Int) -> MLXArray {
                var out: MLXArray?
                for (i, phase) in phases.enumerated() where weights[i * num + j] != 0 {
                    let term = phase * weights[i * num + j]
                    out = out.map { $0 + term } ?? term
                }
                return out!
            }
            let f = compile { (x: [MLXArray]) -> [MLXArray] in
                let img = x[0]
                let hC = img.shape[0], wC = img.shape[1], c = img.shape[2]
                let kH = hC / den, kW = wC / den
                let src = img.asType(.float32)
                // W axis: [H, kW, den, C] → [H, kW, num, C]
                let byCol = src.reshaped([hC, kW, den, c])
                let colPhases = (0..<den).map { byCol[0..., 0..., $0, 0...] }
                let wide = MLX.stacked((0..<num).map { blend(colPhases, $0) }, axis: 2)
                    .reshaped([hC, kW * num, c])
                // H axis: [kH, den, Wout, C] → [kH, num, Wout, C]
                let byRow = wide.reshaped([kH, den, kW * num, c])
                let rowPhases = (0..<den).map { byRow[0..., $0] }
                let out = MLX.stacked((0..<num).map { blend(rowPhases, $0) }, axis: 1)
                    .reshaped([kH * num, kW * num, c])
                return [img.dtype == .uint8 ? out / 255.0 : out]
            }
            cache[key] = f
            return f
        }
    }

    /// 1D area-resampling weights [den, num] (rows = input pixels, columns = output pixels, columns
    /// sum to 1). Weight = overlap length / (den/num).
    static func areaWeight(den: Int, num: Int) -> MLXArray {
        let cell = Float(den) / Float(num)
        var w = [Float](repeating: 0, count: den * num)
        for j in 0..<num {
            let lo = Float(j) * cell, hi = Float(j + 1) * cell
            for i in 0..<den {
                let ov = Swift.max(0, Swift.min(hi, Float(i + 1)) - Swift.max(lo, Float(i)))
                w[i * num + j] = ov / cell
            }
        }
        return MLXArray(w, [den, num])
    }

    /// Rational-ratio version of coarseCamera (intrinsics and resolution × num/den). Resolution
    /// must follow the same crop rule as resampleRational (width/den*num) so they match.
    static func coarseCameraScaled(_ cam: Camera, num: Int, den: Int) -> Camera {
        Camera(
            viewmat: cam.viewmat, intrinsics: cam.intrinsics * (Float(num) / Float(den)),
            width: cam.width / den * num, height: cam.height / den * num,
            tileSize: cam.tileSize, background: cam.background
        )
    }
}
