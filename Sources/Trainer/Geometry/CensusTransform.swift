// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Census Transform (PocketGS Operator G 3b-1, Sec. III-A 3). Input to plane sweep / SGM. Reference:
// Fife & Archibald 2013, "Improved Census Transforms for Resource-Optimized Stereo Vision".

import Foundation
import MLX
import MLXFast

/// Computes a 5×5 Census in a single Slang kernel dispatch.
public enum CensusTransform {

    private static let kernel = SlangCompiledKernel.required("census_transform")

    private static let improvedKernel = SlangCompiledKernel.required("improved_census_transform")

    /// Two of the variants from Fife & Archibald 2013 [40].
    public enum Variant: Sendable {
        /// Compare the center pixel with its 24 neighbors (24 bits).
        case classic
        /// CMCT: compare the 5×5 mean with all 25 pixels (25 bits). Robust to center-pixel noise;
        /// better in low-texture and dark regions.
        case improved
    }

    /// BT.601 luma. `[H, W, 3]` → `[H, W]` float32.
    public static func rgbToLuma(_ rgb: MLXArray) -> MLXArray {
        precondition(rgb.shape.count == 3 && rgb.shape[2] == 3,
                     "rgb must be [H, W, 3]")
        let r = rgb[0..., 0..., 0]
        let g = rgb[0..., 0..., 1]
        let b = rgb[0..., 0..., 2]
        return 0.299 * r + 0.587 * g + 0.114 * b
    }

    /// 5×5 Census. `luma` `[H, W]` float32 → `[H, W]` int32 signature.
    /// See `slang/census_transform.slang` for the bit layout.
    public static func compute(luma: MLXArray, variant: Variant = .classic) -> MLXArray {
        precondition(luma.shape.count == 2, "luma must be [H, W]")
        let h = luma.shape[0]
        let w = luma.shape[1]
        let dims = MLXArray([Int32(w), Int32(h)], [2])
        let lumaFlat = luma.reshaped([h * w])
        let chosen: MLXFast.MLXFastKernel = (variant == .improved) ? improvedKernel : kernel
        let outputs = chosen(
            [lumaFlat, dims],
            grid: (w, h, 1),
            threadGroup: (16, 16, 1),
            outputShapes: [[h * w]],
            outputDTypes: [.int32]
        )
        return outputs[0].reshaped([h, w])
    }

    /// CPU reference Hamming distance for tests.
    public static func hammingDistance(_ a: Int32, _ b: Int32) -> Int {
        var x = UInt32(bitPattern: a ^ b)
        var count = 0
        while x != 0 {
            count += Int(x & 1)
            x >>= 1
        }
        return count
    }
}
