// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX

/// Gaussian-window SSIM of `[H, W, C]` images (Wang et al. 2004, same formulation as gsplat
/// (Apache-2.0)). Pure MLX, differentiable via autograd. Training uses ``FusedSSIM``; this is its
/// reference (the test oracle).
public enum SSIM {
    /// 1D Gaussian weights of length `size` and standard deviation `sigma`, normalized about the
    /// window center.
    static func gaussian1D(size: Int, sigma: Float) -> [Float] {
        let center = Float(size - 1) / 2
        var w = (0..<size).map { i -> Float in
            let d = Float(i) - center
            return exp(-(d * d) / (2 * sigma * sigma))
        }
        let sum = w.reduce(0, +)
        for i in w.indices { w[i] /= sum }
        return w
    }

    /// Depthwise 1D Gaussian conv weights. `[channels, 1, size, 1]` (along width) if `horizontal`,
    /// otherwise `[channels, size, 1, 1]`. Layout is `conv2d`'s `[C_out, KH, KW, C_in/groups]`
    /// (groups == channels).
    static func gaussianKernel1D(
        size: Int, sigma: Float, channels: Int, horizontal: Bool
    ) -> MLXArray {
        let g = gaussian1D(size: size, sigma: sigma)
        var data = [Float]()
        data.reserveCapacity(channels * size)
        for _ in 0..<channels { data.append(contentsOf: g) }
        let shape = horizontal ? [channels, 1, size, 1] : [channels, size, 1, 1]
        return MLXArray(data, shape)
    }

    /// Per-pixel SSIM map of two [0, 1] images (zero-pad "same" convolution, same resolution as the
    /// input). The windowSize/2 px border has μ/σ distorted by the zero padding (the training loss
    /// excludes it).
    public static func ssimMap(
        _ a: MLXArray, _ b: MLXArray, windowSize: Int = 11, sigma: Float = 1.5
    ) -> MLXArray {
        let c1: Float = 0.01 * 0.01
        let c2: Float = 0.03 * 0.03
        let channels = a.shape[a.ndim - 1]
        let pad = windowSize / 2
        let hK = gaussianKernel1D(size: windowSize, sigma: sigma, channels: channels, horizontal: true)
        let vK = gaussianKernel1D(size: windowSize, sigma: sigma, channels: channels, horizontal: false)

        // To [1, H, W, C] to match conv2d's NHWC.
        let x = a.reshaped([1, a.shape[0], a.shape[1], channels])
        let y = b.reshaped([1, b.shape[0], b.shape[1], channels])

        // Separable Gaussian: 1×K along width → K×1 along height (same result as K×K with fewer
        // MACs).
        func blur(_ t: MLXArray) -> MLXArray {
            let h = conv2d(t, hK, stride: 1, padding: IntOrPair((0, pad)), groups: channels)
            return conv2d(h, vK, stride: 1, padding: IntOrPair((pad, 0)), groups: channels)
        }

        let muX = blur(x)
        let muY = blur(y)
        let muX2 = muX * muX
        let muY2 = muY * muY
        let muXY = muX * muY
        let sigmaX2 = blur(x * x) - muX2
        let sigmaY2 = blur(y * y) - muY2
        let sigmaXY = blur(x * y) - muXY

        let numerator = (2 * muXY + c1) * (2 * sigmaXY + c2)
        let denominator = (muX2 + muY2 + c1) * (sigmaX2 + sigmaY2 + c2)
        return numerator / denominator
    }

    /// Mean SSIM (scalar, [-1, 1], 1 means identical).
    public static func ssim(
        _ a: MLXArray, _ b: MLXArray, windowSize: Int = 11, sigma: Float = 1.5
    ) -> MLXArray {
        ssimMap(a, b, windowSize: windowSize, sigma: sigma).mean()
    }
}
