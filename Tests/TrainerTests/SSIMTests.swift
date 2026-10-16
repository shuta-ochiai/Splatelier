// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXRandom
import Testing
@testable import Trainer

// Known SSIM values and monotonicity, and the image-space gradient passed to the renderer (MLX
// autograd) matches central differences.

@Test
func ssimOfIdenticalImageIsOne() throws {
    MLXRandom.seed(0)
    let x = MLXRandom.normal([20, 20, 3]) * 0.2 + 0.5
    eval(x)
    let s = SSIM.ssim(x, x).item(Float.self)
    #expect(abs(s - 1) < 1e-4, "SSIM(x, x) should be 1, got \(s)")
}

@Test
func ssimOfConstantImagesMatchesLuminanceTerm() throws {
    // Between two constant images the local variance and covariance are 0, so the
    // contrast/structure terms are 1 and SSIM equals the luminance term (2·c1·c2 + C1) / (c1² + c2²
    // + C1).
    let c1v: Float = 0.5
    let c2v: Float = 0.6
    let h = 24, w = 24, c = 3
    let a = MLXArray([Float](repeating: c1v, count: h * w * c), [h, w, c])
    let b = MLXArray([Float](repeating: c2v, count: h * w * c), [h, w, c])

    let mapHost = SSIM.ssimMap(a, b, windowSize: 11, sigma: 1.5).asArray(Float.self)
    // The center pixel's window lies inside the image and is unaffected by the zero-pad border.
    let cy = h / 2, cx = w / 2
    let center = mapHost[(cy * w + cx) * c + 0]

    let bigC1: Float = 0.01 * 0.01
    let expected = (2 * c1v * c2v + bigC1) / (c1v * c1v + c2v * c2v + bigC1)
    // Tolerance 1e-3: in float32 the near-zero local variance suffers cancellation between ~c²
    // terms and is divided by a small C2, so it is off by ~2e-4.
    #expect(abs(center - expected) < 1e-3,
            "constant-image SSIM \(center) != luminance term \(expected)")
}

@Test
func ssimDecreasesWithNoise() throws {
    MLXRandom.seed(3)
    let h = 24, w = 24, c = 3
    let x = MLXRandom.normal([h, w, c]) * 0.2 + 0.5
    let small = x + MLXRandom.normal([h, w, c]) * 0.02
    let large = x + MLXRandom.normal([h, w, c]) * 0.2
    eval(x, small, large)

    let sSmall = SSIM.ssim(x, small).item(Float.self)
    let sLarge = SSIM.ssim(x, large).item(Float.self)
    #expect(sSmall < 1.0, "adding noise should drop SSIM below 1, got \(sSmall)")
    #expect(sLarge < sSmall, "more noise should lower SSIM: \(sLarge) !< \(sSmall)")
}

@Test
func ssimGradientMatchesFiniteDifferences() throws {
    MLXRandom.seed(7)
    let h = 10, w = 10, c = 3
    let win = 5
    let target = MLXRandom.normal([h, w, c]) * 0.2 + 0.5
    let x = MLXRandom.normal([h, w, c]) * 0.2 + 0.5
    eval(target, x)

    // D-SSIM loss differentiated by the trainer = 1 - SSIM.
    let lossOf: (MLXArray) -> MLXArray = { img in
        1 - SSIM.ssim(img, target, windowSize: win)
    }
    let analytical = grad(lossOf)(x)
    eval(analytical)
    let analyticalHost = analytical.asArray(Float.self)

    let epsValue: Float = 1e-3
    let xHost = x.asArray(Float.self)
    var perturbed = xHost
    func dssimHost(_ arr: [Float]) -> Float {
        lossOf(MLXArray(arr, [h, w, c])).item(Float.self)
    }

    var maxAbsDiff: Float = 0
    var maxRelDiff: Float = 0
    for idx in 0..<(h * w * c) {
        perturbed[idx] = xHost[idx] + epsValue
        let lossPlus = dssimHost(perturbed)
        perturbed[idx] = xHost[idx] - epsValue
        let lossMinus = dssimHost(perturbed)
        perturbed[idx] = xHost[idx]

        let numeric = (lossPlus - lossMinus) / (2 * epsValue)
        let a = analyticalHost[idx]
        let absDiff = abs(a - numeric)
        let scale = max(abs(a), abs(numeric), 1e-3)
        maxAbsDiff = max(maxAbsDiff, absDiff)
        maxRelDiff = max(maxRelDiff, absDiff / scale)
    }
    #expect(maxAbsDiff < 2e-3 || maxRelDiff < 3e-2,
            "SSIM gradient diverged from finite differences: maxAbs=\(maxAbsDiff), maxRel=\(maxRelDiff)")
}
