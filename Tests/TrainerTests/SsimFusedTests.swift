// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXRandom
import Testing
@testable import Trainer

// Fused SSIM Slang kernel. forward reproduces the MLX reference `SSIM.ssimMap`, and backward
// matches finite differences of the SSIM map loss.

private let kWindow = 11
private let kRadius = kWindow / 2
private let kSigma: Float = 1.5

/// Runs `ssim_fwd` and returns the SSIM map and 3 partial-derivative caches (all [H, W, C]).
private func fusedSsimForward(_ a: MLXArray, _ b: MLXArray)
    throws -> (map: MLXArray, dMu1: MLXArray, dS1: MLXArray, dS12: MLXArray)
{
    let h = a.shape[0], w = a.shape[1], c = a.shape[2]
    let total = h * w * c
    let kernel = try SlangCompiledKernel.load(name: "ssim_fwd")
    let gauss = MLXArray(SSIM.gaussian1D(size: kWindow, sigma: kSigma), [kWindow])
    let dims = MLXArray([Int32(h), Int32(w), Int32(c), Int32(kRadius)], [4])
    let out = kernel(
        [a.reshaped([total]), b.reshaped([total]), gauss, dims],
        grid: (w, h, 1),
        threadGroup: (16, 16, 1),
        outputShapes: [[total], [total], [total], [total]],
        outputDTypes: [.float32, .float32, .float32, .float32]
    )
    eval(out[0], out[1], out[2], out[3])
    return (
        out[0].reshaped([h, w, c]),
        out[1].reshaped([h, w, c]),
        out[2].reshaped([h, w, c]),
        out[3].reshaped([h, w, c])
    )
}

/// Runs `ssim_bwd` and returns dL/d(img1) [H, W, C].
private func fusedSsimBackward(
    _ a: MLXArray, _ b: MLXArray, dLdmap: MLXArray,
    dMu1: MLXArray, dS1: MLXArray, dS12: MLXArray
) throws -> MLXArray {
    let h = a.shape[0], w = a.shape[1], c = a.shape[2]
    let total = h * w * c
    let kernel = try SlangCompiledKernel.load(name: "ssim_bwd")
    let gauss = MLXArray(SSIM.gaussian1D(size: kWindow, sigma: kSigma), [kWindow])
    let dims = MLXArray([Int32(h), Int32(w), Int32(c), Int32(kRadius)], [4])
    // lossParams = [oneMinusLambda, invN] = [0, 0] → pure SSIM cotangent with no L1 term.
    let lossParams = MLXArray([Float(0), Float(0)], [2])
    let out = kernel(
        [
            a.reshaped([total]), b.reshaped([total]), dLdmap.reshaped([total]),
            dMu1.reshaped([total]), dS1.reshaped([total]), dS12.reshaped([total]),
            gauss, dims, lossParams,
        ],
        grid: (w, h, 1),
        threadGroup: (16, 16, 1),
        outputShapes: [[total]],
        outputDTypes: [.float32]
    )
    eval(out[0])
    return out[0].reshaped([h, w, c])
}

@Test
func fusedSsimBackwardMatchesFiniteDifferences() throws {
    MLXRandom.seed(23)
    let h = 12, w = 12, c = 3
    let a = MLXRandom.normal([h, w, c]) * 0.2 + 0.5
    let b = MLXRandom.normal([h, w, c]) * 0.2 + 0.5
    let dLdmap = MLXRandom.normal([h, w, c]) * 0.1
    eval(a, b, dLdmap)

    // Analytic: fwd caches the partial derivatives, bwd turns them into dL/d(a).
    let fwd = try fusedSsimForward(a, b)
    let analytical = try fusedSsimBackward(
        a, b, dLdmap: dLdmap, dMu1: fwd.dMu1, dS1: fwd.dS1, dS12: fwd.dS12
    )
    let analyticalHost = analytical.asArray(Float.self)

    // Numeric: central differences of L(a) = Σ(dLdmap * ssimMap(a, b)) using an independent MLX
    // reference map.
    let dLdmapConst = dLdmap
    let bConst = b
    func loss(_ arr: [Float]) -> Float {
        let img = MLXArray(arr, [h, w, c])
        let m = SSIM.ssimMap(img, bConst, windowSize: kWindow, sigma: kSigma)
        return (dLdmapConst * m).sum().item(Float.self)
    }

    let eps: Float = 1e-3
    let aHost = a.asArray(Float.self)
    var perturbed = aHost
    var maxAbs: Float = 0, maxRel: Float = 0
    var worstIdx = -1
    for idx in 0..<(h * w * c) {
        perturbed[idx] = aHost[idx] + eps
        let lp = loss(perturbed)
        perturbed[idx] = aHost[idx] - eps
        let lm = loss(perturbed)
        perturbed[idx] = aHost[idx]

        let numeric = (lp - lm) / (2 * eps)
        let exact = analyticalHost[idx]
        let absDiff = abs(numeric - exact)
        let scale = max(abs(numeric), abs(exact), 1e-4)
        if absDiff > maxAbs { maxAbs = absDiff; worstIdx = idx }
        maxRel = max(maxRel, absDiff / scale)
    }
    #expect(maxAbs < 5e-3 || maxRel < 5e-2,
            "fused SSIM bwd diverged from FD: maxAbs=\(maxAbs) maxRel=\(maxRel) at idx=\(worstIdx)")
}

@Test
func fusedSsimForwardMatchesReference() throws {
    MLXRandom.seed(11)
    // Use a size that is not a multiple of 16x16 to exercise partial threadgroups and the zero-pad
    // border.
    let h = 20, w = 24, c = 3
    let a = MLXRandom.normal([h, w, c]) * 0.2 + 0.5
    let b = MLXRandom.normal([h, w, c]) * 0.2 + 0.5
    eval(a, b)

    let fused = try fusedSsimForward(a, b)
    let reference = SSIM.ssimMap(a, b, windowSize: kWindow, sigma: kSigma)
    eval(reference)

    let fusedHost = fused.map.asArray(Float.self)
    let refHost = reference.asArray(Float.self)

    var maxAbs: Float = 0
    for i in 0..<(h * w * c) {
        maxAbs = max(maxAbs, abs(fusedHost[i] - refHost[i]))
    }
    #expect(maxAbs < 1e-3, "fused SSIM map diverged from reference: maxAbs=\(maxAbs)")
}

// MARK: - loss_eval (SSIM) breakdown bench

/// Bench that splits the trainer's loss_eval into ssim_fwd / ssim_bwd / MLX glue and measures
/// median ms. Run: `BENCH_SSIM=1 swift test --filter ssimLossEvalBench` (tune with
/// BENCH_W/H/ITERS/WARMUP/LAMBDA/TILED/WIN)
@Test
func ssimLossEvalBench() throws {
    let env = ProcessInfo.processInfo.environment
    guard env["BENCH_SSIM"] != nil else { return }
    func envInt(_ k: String, _ d: Int) -> Int { env[k].flatMap { Int($0) } ?? d }
    let W = envInt("BENCH_W", 1024), H = envInt("BENCH_H", 768)
    let iters = envInt("BENCH_ITERS", 120), warmup = envInt("BENCH_WARMUP", 20)
    let lambda = Float(env["BENCH_LAMBDA"].flatMap { Double($0) } ?? 0.2)

    if let t = env["BENCH_TILED"] { FusedSSIM.useTiled = (t != "0") }
    MLXRandom.seed(1)
    let render = MLXRandom.uniform(low: Float(0), high: Float(1), [H, W, 3])
    let target = MLXRandom.uniform(low: Float(0), high: Float(1), [H, W, 3])
    eval(render, target)
    let win = envInt("BENCH_WIN", 11); let ssim = try FusedSSIM(windowSize: win)
    print("(useTiled=\(FusedSSIM.useTiled))")
    let n = Float(H * W * 3)
    let pad = win / 2
    let validH = H - 2 * pad, validW = W - 2 * pad
    let nValid = Float(validH * validW * 3)

    func median(_ body: () -> Void) -> Double {
        for _ in 0..<warmup { body() }
        var t: [Double] = []
        for _ in 0..<iters { let s = Date(); body(); t.append(Date().timeIntervalSince(s) * 1000) }
        t.sort(); return t[t.count / 2]
    }

    // (a) ssim_fwd only (eval map + 3 partials)
    let fwdMs = median {
        let (m, c, _) = ssim.forward(render, target)
        eval(m, c.dMu1, c.dSigma1Sq, c.dSigma12)
    }
    // (b) ssim_bwd only (cache pre-evaluated, eval the backward)
    let (_, cacheB, _) = ssim.forward(render, target)
    eval(cacheB.dMu1, cacheB.dSigma1Sq, cacheB.dSigma12)
    let dLdMapFull = MLX.padded(
        (-lambda / nValid) * MLXArray.ones([validH, validW, 3]),
        widths: [.init((pad, pad)), .init((pad, pad)), .init((0, 0))],
        value: MLXArray(0.0 as Float))
    eval(dLdMapFull)
    let bwdMs = median {
        let g = ssim.backward(render, target, dLdMap: dLdMapFull, cache: cacheB)
        eval(g)
    }
    // (c) L1 + cotangent glue only (independent of ssim)
    let l1Ms = median {
        let diff = render - target
        let l1 = MLX.abs(diff).mean()
        let vR = (1 - lambda) * MLX.sign(diff) / n
        eval(l1, vR)
    }
    // (d) Reproduce the trainer's entire loss_eval
    let fullMs = median {
        let diff = render - target
        let l1 = MLX.abs(diff).mean()
        var vRender = (1 - lambda) * MLX.sign(diff) / n
        let (ssimMap, cache, _) = ssim.forward(render, target)
        let validMap = ssimMap[pad..<(H - pad), pad..<(W - pad), 0...]
        let lossArray = (1 - lambda) * l1 + lambda * (1 - validMap.mean())
        let dMap = MLX.padded(
            (-lambda / nValid) * MLXArray.ones([validH, validW, 3]),
            widths: [.init((pad, pad)), .init((pad, pad)), .init((0, 0))],
            value: MLXArray(0.0 as Float))
        vRender = vRender + ssim.backward(render, target, dLdMap: dMap, cache: cache)
        let vAlpha = MLXArray.zeros([H, W])
        eval(lossArray, vRender, vAlpha)
    }
    print("=== ssim loss_eval bench: \(W)x\(H) lambda=\(lambda) ===")
    print(String(format: "ssim_fwd        = %.3f ms", fwdMs))
    print(String(format: "ssim_bwd        = %.3f ms", bwdMs))
    print(String(format: "L1+cotangent    = %.3f ms", l1Ms))
    print(String(format: "FULL loss_eval  = %.3f ms", fullMs))
    print(String(format: "glue (full-fwd-bwd) ≈ %.3f ms", fullMs - fwdMs - bwdMs))
}

// MARK: - forward loss partial sums match the MLX reduction

/// The tiled forward's per-tile partial sums (l1Sum / ssimSum) divided by the denominator match
/// MLX's `abs(diff).mean()` (L1, all pixels) and `validMap.mean()` (SSIM, excluding a 5px border)
/// (differences are reduction-order rounding only).
@Test
func fusedSsimForwardLossPartialsMatchMLXReference() throws {
    for (h, w) in [(48, 40), (33, 27), (64, 50)] {
        let c = 3
        MLXRandom.seed(3)
        let a = MLXRandom.uniform(low: Float(0), high: Float(1), [h, w, c])
        let b = MLXRandom.uniform(low: Float(0), high: Float(1), [h, w, c])
        eval(a, b)
        let ssim = try FusedSSIM()   // window 11 → tiled path
        let (map, _, loss) = ssim.forward(a, b)
        let lp = try #require(loss, "tiled forward should return loss partials")

        let pad = 5
        let n = Float(h * w * c)
        let nValid = Float((h - 2 * pad) * (w - 2 * pad) * c)
        // MLX path reference.
        let l1Ref = MLX.abs(a - b).mean()
        // The map is fp16, so promote to float32 before averaging (otherwise the reference value
        // itself gets rounded).
        let ssimValidRef = map[pad..<(h - pad), pad..<(w - pad), 0...].asType(.float32).mean()
        // Fused version: per-tile partial sums / denominator.
        let l1Fused = lp.l1Sum / n
        let ssimFused = lp.ssimSum / nValid
        eval(l1Ref, ssimValidRef, l1Fused, ssimFused)
        func d(_ x: MLXArray, _ y: MLXArray) -> Float { MLX.abs(x - y).item(Float.self) }
        #expect(d(l1Fused, l1Ref) < 1e-5,
                "\(h)x\(w) L1: fused \(l1Fused.item(Float.self)) vs ref \(l1Ref.item(Float.self))")
        #expect(d(ssimFused, ssimValidRef) < 1e-5,
                "\(h)x\(w) SSIM-valid: fused \(ssimFused.item(Float.self)) vs ref \(ssimValidRef.item(Float.self))")
    }
}

// MARK: - tiled+separable SSIM matches the naive fused kernel

private func fusedSsimForwardTiled(_ a: MLXArray, _ b: MLXArray)
    throws -> (map: MLXArray, dMu1: MLXArray, dS1: MLXArray, dS12: MLXArray)
{
    let h = a.shape[0], w = a.shape[1], c = a.shape[2]
    let total = h * w * c
    let kernel = try SlangCompiledKernel.load(name: "ssim_fwd_tiled")
    let gauss = MLXArray(SSIM.gaussian1D(size: kWindow, sigma: kSigma), [kWindow])
    let dims = MLXArray([Int32(h), Int32(w), Int32(c), Int32(kRadius)], [4])
    // Pad the grid to a multiple of 16 so the cooperative load misses nothing (out-of-range threads
    // no-op via the output guard).
    let gw = ((w + 15) / 16) * 16
    let gh = ((h + 15) / 16) * 16
    // ssim_fwd_tiled also has 2 outputs for per-tile loss partial sums (tilePartialL1/SSIM,
    // [numTiles]); discard them.
    let numTiles = (gw / 16) * (gh / 16)
    let out = kernel(
        [a.reshaped([total]), b.reshaped([total]), gauss, dims],
        grid: (gw, gh, 1),
        threadGroup: (16, 16, 1),
        outputShapes: [[total], [total], [total], [total], [numTiles], [numTiles]],
        outputDTypes: [.float32, .float32, .float32, .float32, .float32, .float32]
    )
    eval(out[0], out[1], out[2], out[3])
    return (out[0].reshaped([h, w, c]), out[1].reshaped([h, w, c]),
            out[2].reshaped([h, w, c]), out[3].reshaped([h, w, c]))
}

private func maxAbsDiffSSIM(_ a: MLXArray, _ b: MLXArray) -> Float {
    MLX.abs(a - b).max().asArray(Float.self)[0]
}

@Test
func ssimFwdTiledMatchesNaive() throws {
    // Multiple tiles + non-multiples of 16 (covers OOB at tile borders / partial warps / halo).
    for (h, w) in [(24, 40), (33, 17), (64, 48)] {
        MLXRandom.seed(7)
        let a = MLXRandom.uniform(low: Float(0), high: Float(1), [h, w, 3])
        let b = MLXRandom.uniform(low: Float(0), high: Float(1), [h, w, 3])
        eval(a, b)
        let naive = try fusedSsimForward(a, b)
        let tiled = try fusedSsimForwardTiled(a, b)
        // The map / dm_d* caches use half storage, so recombination differences carry 1 fp16 ULP
        // (map ≤2⁻¹¹, dm_d* ≤2⁻⁷). Thresholds allow that while still catching real bugs (map ≫1e-3
        // / cache ≫1e-2).
        #expect(maxAbsDiffSSIM(naive.map, tiled.map) < 1e-3, "\(h)x\(w) map mismatch")
        #expect(maxAbsDiffSSIM(naive.dMu1, tiled.dMu1) < 1.5e-2, "\(h)x\(w) dMu1 mismatch")
        #expect(maxAbsDiffSSIM(naive.dS1, tiled.dS1) < 1.5e-2, "\(h)x\(w) dS1 mismatch")
        #expect(maxAbsDiffSSIM(naive.dS12, tiled.dS12) < 1.5e-2, "\(h)x\(w) dS12 mismatch")
    }
}

private func fusedSsimBackwardTiled(
    _ a: MLXArray, _ b: MLXArray, dLdmap: MLXArray,
    dMu1: MLXArray, dS1: MLXArray, dS12: MLXArray
) throws -> MLXArray {
    let h = a.shape[0], w = a.shape[1], c = a.shape[2]
    let total = h * w * c
    let kernel = try SlangCompiledKernel.load(name: "ssim_bwd_tiled")
    let gauss = MLXArray(SSIM.gaussian1D(size: kWindow, sigma: kSigma), [kWindow])
    let dims = MLXArray([Int32(h), Int32(w), Int32(c), Int32(kRadius)], [4])
    let gw = ((w + 15) / 16) * 16
    let gh = ((h + 15) / 16) * 16
    // lossParams = [0, 0] → pure SSIM cotangent.
    let lossParams = MLXArray([Float(0), Float(0)], [2])
    let out = kernel(
        [
            a.reshaped([total]), b.reshaped([total]), dLdmap.reshaped([total]),
            dMu1.reshaped([total]), dS1.reshaped([total]), dS12.reshaped([total]),
            gauss, dims, lossParams,
        ],
        grid: (gw, gh, 1),
        threadGroup: (16, 16, 1),
        outputShapes: [[total]],
        outputDTypes: [.float32]
    )
    eval(out[0])
    return out[0].reshaped([h, w, c])
}

@Test
func ssimBwdTiledMatchesNaive() throws {
    for (h, w) in [(24, 40), (33, 17), (64, 48)] {
        MLXRandom.seed(13)
        let a = MLXRandom.uniform(low: Float(0), high: Float(1), [h, w, 3])
        let b = MLXRandom.uniform(low: Float(0), high: Float(1), [h, w, 3])
        let dLdmap = MLXRandom.normal([h, w, 3]) * 0.1
        eval(a, b, dLdmap)
        let fwd = try fusedSsimForward(a, b)
        let naive = try fusedSsimBackward(a, b, dLdmap: dLdmap,
                                          dMu1: fwd.dMu1, dS1: fwd.dS1, dS12: fwd.dS12)
        let tiled = try fusedSsimBackwardTiled(a, b, dLdmap: dLdmap,
                                               dMu1: fwd.dMu1, dS1: fwd.dS1, dS12: fwd.dS12)
        let d = maxAbsDiffSSIM(naive, tiled)
        #expect(d < 1e-4, "\(h)x\(w) dL/dimg1 mismatch: \(d)")
    }
}

/// The fused backward (oneMinusLambda > 0) matches "pure SSIM cotangent + L1 cotangent
/// (1-λ)·sign(a-b)/N" (catches regressions in the L1 term folded into the kernel in isolation).
@Test
func fusedSsimBackwardFoldsL1Cotangent() throws {
    MLXRandom.seed(7)
    let h = 48, w = 56, c = 3
    let total = h * w * c
    let a = MLXRandom.uniform(low: 0.0, high: 1.0, [h, w, c])
    let b = MLXRandom.uniform(low: 0.0, high: 1.0, [h, w, c])
    let dLdMap = MLXRandom.normal([h, w, c]) * 0.1
    eval(a, b, dLdMap)

    let ssim = try FusedSSIM()
    let (_, cache, _) = ssim.forward(a, b)
    let lambda: Float = 0.2
    let n = Float(total)

    // SSIM-only cotangent (oneMinusLambda defaults to 0 → no L1 term).
    let ssimCot = ssim.backward(a, b, dLdMap: dLdMap, cache: cache)
    let fused = ssim.backward(
        a, b, dLdMap: dLdMap, cache: cache, oneMinusLambda: 1 - lambda, invN: 1 / n)
    let l1Cot = (1 - lambda) * MLX.sign(a - b) / n
    let ref = ssimCot + l1Cot
    eval(fused, ref)

    let fH = fused.asArray(Float.self)
    let rH = ref.asArray(Float.self)
    var maxd: Float = 0
    for i in 0..<total { maxd = max(maxd, abs(fH[i] - rH[i])) }
    #expect(maxd < 1e-6, "fused L1 cotangent diverges from separate SSIM+L1: \(maxd)")
}
