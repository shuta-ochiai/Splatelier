// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXFast

/// Fused SSIM via two Slang kernels (`ssim_fwd` / `ssim_bwd`); the GPU version of the pure-MLX
/// ``SSIM``. Forward keeps the SSIM map plus 3 partial-derivative maps; backward uses them and
/// `dL/d(map)` to produce `dL/d(img1)`. The formulas and partial-derivative caching follow
/// rahul-goel/fused-ssim (MIT). Only `img1` (render) is differentiated.
public final class FusedSSIM {

    /// Per-pixel partial derivatives kept by forward and used by backward (flat [H*W*C]).
    public struct Cache {
        let dMu1: MLXArray
        let dSigma1Sq: MLXArray
        let dSigma12: MLXArray
        let shape: [Int]   // [H, W, C]
    }

    /// Whether to use the tiled + separable kernels (`ssim_{fwd,bwd}_tiled`). Same math as naive,
    /// sped up with shared-memory tiling and 1D separation. Tiled only supports rad<=5
    /// (window<=11); anything else falls back to naive.
    public nonisolated(unsafe) static var useTiled = true

    private let fwdKernel: MLXFast.MLXFastKernel
    private let bwdKernel: MLXFast.MLXFastKernel
    private let gauss: MLXArray
    private let windowSize: Int
    /// Whether the tiled kernel is in use. If true, the grid is padded to a multiple of 16 and edge
    /// threadgroups also get the full 256 threads, so the cooperative load misses nothing.
    private let tiled: Bool

    public init(windowSize: Int = 11, sigma: Float = 1.5) throws {
        // Tiled has a fixed HALO=26 (=16+2*5), so it only supports rad<=5 (window<=11).
        let useTiled = Self.useTiled && windowSize <= 11
        self.tiled = useTiled
        self.fwdKernel = try SlangCompiledKernel.load(name: useTiled ? "ssim_fwd_tiled" : "ssim_fwd")
        self.bwdKernel = try SlangCompiledKernel.load(name: useTiled ? "ssim_bwd_tiled" : "ssim_bwd")
        self.windowSize = windowSize
        self.gauss = MLXArray(SSIM.gaussian1D(size: windowSize, sigma: sigma), [windowSize])
    }

    /// Dispatch grid (padded to a multiple of 16 when tiled).
    private func dispatchGrid(w: Int, h: Int) -> (Int, Int, Int) {
        guard tiled else { return (w, h, 1) }
        return (((w + 15) / 16) * 16, ((h + 15) / 16) * 16, 1)
    }

    /// Partial loss sums returned by the tiled forward. The caller divides `l1Sum` by `H*W*C` and
    /// `ssimSum` by `(H-2rad)(W-2rad)*C` to form `(1-λ)·L1 + λ·(1-SSIM)` (skipping the full-image
    /// mean).
    public struct LossPartials {
        public let l1Sum: MLXArray     // scalar: Σ|a-b| over all of H*W*C
        public let ssimSum: MLXArray   // scalar: Σ ssim over the valid region excluding the rad px border
    }

    /// SSIM forward. Returns the per-pixel map `[H, W, C]` and the cache for backward. `loss` is
    /// non-nil only on the tiled path. The map is fp16, so upcast to float32 before taking the mean
    /// (otherwise the sum overflows to inf).
    public func forward(_ a: MLXArray, _ b: MLXArray)
        -> (map: MLXArray, cache: Cache, loss: LossPartials?)
    {
        let h = a.shape[0], w = a.shape[1], c = a.shape[2]
        let total = h * w * c
        let dims = MLXArray(
            [Int32(h), Int32(w), Int32(c), Int32(windowSize / 2)], [4]
        )
        let inputs = [a.reshaped([total]), b.reshaped([total]), gauss, dims]
        if tiled {
            let numTiles = ((w + 15) / 16) * ((h + 15) / 16)
            let out = fwdKernel(
                inputs,
                grid: dispatchGrid(w: w, h: h),
                threadGroup: (16, 16, 1),
                outputShapes: [[total], [total], [total], [total], [numTiles], [numTiles]],
                // The 4 per-pixel maps (ssim_map + dm_d*) are fp16; the tile partial sums (out[4],
                // out[5]) are fp32.
                outputDTypes: [.float16, .float16, .float16, .float16, .float32, .float32]
            )
            let cache = Cache(dMu1: out[1], dSigma1Sq: out[2], dSigma12: out[3], shape: [h, w, c])
            let loss = LossPartials(l1Sum: out[4].sum(), ssimSum: out[5].sum())
            return (out[0].reshaped([h, w, c]), cache, loss)
        } else {
            let out = fwdKernel(
                inputs,
                grid: dispatchGrid(w: w, h: h),
                threadGroup: (16, 16, 1),
                outputShapes: [[total], [total], [total], [total]],
                outputDTypes: [.float16, .float16, .float16, .float16]  // maps are fp16
            )
            let cache = Cache(dMu1: out[1], dSigma1Sq: out[2], dSigma12: out[3], shape: [h, w, c])
            return (out[0].reshaped([h, w, c]), cache, nil)
        }
    }

    /// SSIM backward: returns `dL/d(a)` from `dL/d(map)` (any shape that flattens to `[H*W*C]`) and
    /// `cache`. If `oneMinusLambda > 0`, the L1-term cotangent `(1-λ)·sign(a-b)·invN` is added in
    /// the same dispatch. Default (0, 0) is SSIM only.
    public func backward(
        _ a: MLXArray, _ b: MLXArray, dLdMap: MLXArray, cache: Cache,
        oneMinusLambda: Float = 0, invN: Float = 0
    ) -> MLXArray {
        let h = cache.shape[0], w = cache.shape[1], c = cache.shape[2]
        let total = h * w * c
        let dims = MLXArray(
            [Int32(h), Int32(w), Int32(c), Int32(windowSize / 2)], [4]
        )
        let lossParams = MLXArray([oneMinusLambda, invN], [2])
        let out = bwdKernel(
            [
                a.reshaped([total]), b.reshaped([total]), dLdMap.reshaped([total]),
                cache.dMu1, cache.dSigma1Sq, cache.dSigma12,
                gauss, dims, lossParams,
            ],
            grid: dispatchGrid(w: w, h: h),
            threadGroup: (16, 16, 1),
            outputShapes: [[total]],
            outputDTypes: [.float32]
        )
        return out[0].reshaped([h, w, c])
    }
}
