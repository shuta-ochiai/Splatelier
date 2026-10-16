// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXRandom
import Testing
@testable import Trainer

/// `GaussianTrainer.alphaTerms` through `compile` (as training runs it) gives the same bits as the
/// separate ops, at both step resolutions.
@Test func alphaTermsCompiledMatchesSeparateOps() {
    let weight: Float = 0.15
    let compiled = compile { (x: [MLXArray]) -> [MLXArray] in
        let t = GaussianTrainer.alphaTerms(
            vRender: x[0], bg: x[1], target: x[2], renderAlpha: x[3], weight: weight)
        return [t.vAlpha, t.loss]
    }
    for (h, w) in [(270, 360), (540, 720)] {
        MLXRandom.seed(UInt64(h))
        let vRender = MLXRandom.normal([h, w, 3]) * 1e-6
        let bg = MLXRandom.uniform(low: 0, high: 1, [h, w, 3])
        // About 40% background pixels (target equals bg there), so the mask is non-trivial.
        let isBg = MLXRandom.uniform(low: 0, high: 1, [h, w, 1]) .< Float(0.4)
        let target = MLX.which(isBg, bg, MLXRandom.uniform(low: 0, high: 1, [h, w, 3]))
        let alpha = MLXRandom.uniform(low: 0, high: 1, [h * w])
        let separate = GaussianTrainer.alphaTerms(
            vRender: vRender, bg: bg, target: target, renderAlpha: alpha, weight: weight)
        let fused = compiled([vRender, bg, target, alpha])
        eval(separate.vAlpha, separate.loss, fused[0], fused[1])
        #expect(fused[0].shape == [h, w])
        let a = separate.vAlpha.asArray(Float.self), b = fused[0].asArray(Float.self)
        #expect(zip(a, b).allSatisfy { $0.bitPattern == $1.bitPattern }, "vAlpha differs at \(h)x\(w)")
        let la = separate.loss.item(Float.self), lb = fused[1].item(Float.self)
        #expect(la.bitPattern == lb.bitPattern, "loss \(la) vs \(lb) at \(h)x\(w)")
        #expect(la > 0, "the mask should cover some pixels")
    }
}
