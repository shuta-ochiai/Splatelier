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

/// Pixels marked as a measured surface never count as background: with every pixel marked, the
/// alpha loss vanishes even where the target matches the background exactly.
@Test func alphaTermsSkipSurfacePixels() {
    let h = 8, w = 8
    let bg = MLXRandom.uniform(low: 0, high: 1, [h, w, 3])
    let vRender = MLXArray.zeros([h, w, 3])
    let alpha = MLXArray.ones([h * w])
    let open = GaussianTrainer.alphaTerms(vRender: vRender, bg: bg, target: bg, renderAlpha: alpha, weight: 0.15)
    let marked = GaussianTrainer.alphaTerms(vRender: vRender, bg: bg, target: bg, renderAlpha: alpha, weight: 0.15,
                                            surface: MLXArray.ones([h, w]))
    #expect(open.loss.item(Float.self) > 0)
    #expect(marked.loss.item(Float.self) == 0)
    #expect(MLX.abs(marked.vAlpha).max().item(Float.self) == 0)
}

/// On opaque pixels the alpha loss is the mean of (1 - alpha) there, and its gradient pulls alpha
/// up on those pixels only.
@Test func alphaTermsPullOpaquePixelsToOne() {
    let h = 4, w = 4
    let bg = MLXArray.zeros([h, w, 3])
    let target = MLXArray.ones([h, w, 3])   // nowhere background
    let vRender = MLXArray.zeros([h, w, 3])
    let alpha = MLXArray([Float](repeating: 0.25, count: h * w))
    let opaque = MLXArray((0..<(h * w)).map { Float($0 < w ? 1 : 0) }, [h, w])   // the top row
    let t = GaussianTrainer.alphaTerms(vRender: vRender, bg: bg, target: target, renderAlpha: alpha,
                                       weight: 0.15, opaque: opaque, opaqueWeight: 0.2)
    #expect(abs(t.loss.item(Float.self) - 0.2 * 0.75) < 1e-6)
    let v = t.vAlpha.asArray(Float.self)
    #expect(v[0 ..< w].allSatisfy { abs($0 + 0.2 / Float(w)) < 1e-7 })
    #expect(v[w...].allSatisfy { $0 == 0 })
}

/// The nearest-cell upsampling keeps each grid cell's value over its share of the image.
@Test func surfaceMaskUpsamplesByNearestCell() {
    let grid = MLXArray([Float(1), 0, 0, 1], [2, 2])
    let up = GaussianTrainer.upsampleNearest(grid, height: 4, width: 6)
    #expect(up.shape == [4, 6])
    #expect(up.asArray(Float.self) == [1, 1, 1, 0, 0, 0,
                                       1, 1, 1, 0, 0, 0,
                                       0, 0, 0, 1, 1, 1,
                                       0, 0, 0, 1, 1, 1])
}
