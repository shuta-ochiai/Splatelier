// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX

/// Gaussian trainer that updates one step at a time with `(1-λ)·L1 + λ·(1-SSIM)` on one camera and
/// one image. Holds one Adam per parameter group; lr defaults roughly follow gsplat simple_trainer
/// (the caller sets the means lr from the scene scale).
public final class GaussianTrainer: GaussianParamStore {
    public let model: GaussianModel
    public let renderer: GaussianRenderer

    /// Per-stage timing hook (for the training loop's profiler). Normally nil.
    public nonisolated(unsafe) static var profileHook: ((String, Double) -> Void)?

    /// Fused SSIM used when `lambdaSSIM > 0`.
    private let fusedSSIM: FusedSSIM

    /// Per-resolution cache of constant cotangents (key = h*100000+w). dL/d(SSIM map) is -λ/nValid
    /// inside and 0 on the 5px border; the alpha cotangent is zeros. Materialized once and reused
    /// (read-only) so it is not rebuilt at full resolution every iter.
    private var dLdSsimMapCache: [Int: MLXArray] = [:]
    private var vAlphaZeroCache: [Int: MLXArray] = [:]

    /// Weight λ of the D-SSIM term (`loss = (1-λ)·L1 + λ·(1-SSIM)`). 0 means L1 only.
    public let lambdaSSIM: Float

    /// MCMC regularization `opacityReg·mean(opacity) + scaleReg·mean(scale)` (gsplat uses 0.01
    /// each). Added as a closed-form gradient so unused Gaussians degenerate and get relocated. 0
    /// disables it.
    public let opacityReg: Float
    public let scaleReg: Float
    /// Upper bound on foreground scale (absolute world units; the caller passes e.g.
    /// 0.5×scene_scale). After the Adam step, enforces `exp(logScale) ≤ maxScale` to stop blob
    /// floaters from growing after pruning stops. 0 disables it.
    public let maxScale: Float
    private let logMaxScale: Float

    /// Directional SH environment-map background (Splatfacto-W style; its code is Apache-2.0):
    /// `final = render(foreground, black background) + (1-alpha)·bg(dir)`. An O(pixels) background
    /// that adds no rasterize work. One global set of coefficients.
    public let enableBackgroundSH: Bool
    public let backgroundSHDegree: Int
    /// Weight of the alpha loss (splatfacto-w uses 0.15). Pushes foreground alpha to 0 on pixels
    /// the background alone explains, suppressing floaters/haze.
    public let alphaLossWeight: Float
    /// Weight of the loss pulling foreground alpha to 1 on `TrainView.opaqueMask` pixels (inside
    /// measured surfaces), so the background never stands in for a surface. 0 disables it.
    public let surfaceAlphaWeight: Float
    /// Learned background SH coefficients [bands, 3]. An unused [1, 3] when bg-SH is disabled.
    public var bgCoeffs: MLXArray
    public let optBgCoeffs: AdamOptimizer
    private var stateBgCoeffs: AdamOptimizer.State
    /// Fused bg-SH kernel (forward: dir+SH+compositing, backward: coefficient gradients). nil when
    /// bg-SH is disabled.
    private let bgKernels: BackgroundSHKernels?

    public let optMeans: AdamOptimizer
    public let optQuats: AdamOptimizer
    public let optLogScales: AdamOptimizer
    public let optLogitOpacities: AdamOptimizer
    public let optSh0: AdamOptimizer
    public let optShN: AdamOptimizer

    private var stateMeans: AdamOptimizer.State
    private var stateQuats: AdamOptimizer.State
    private var stateLogScales: AdamOptimizer.State
    private var stateLogitOpacities: AdamOptimizer.State
    private var stateSh0: AdamOptimizer.State
    private var stateShN: AdamOptimizer.State
    /// shN's Adam moments are still all zero (no step has run above SH degree 0).
    private var shNMomentsAreZero = true

    public init(
        model: GaussianModel, renderer: GaussianRenderer,
        meansLR: Float = 1.6e-4,
        quatsLR: Float = 1e-3,
        logScalesLR: Float = 5e-3,
        logitOpacitiesLR: Float = 5e-2,
        sh0LR: Float = 2.5e-3,
        shNLR: Float = 2.5e-3 / 20,
        lambdaSSIM: Float = 0.2,
        opacityReg: Float = 0,
        scaleReg: Float = 0,
        maxScale: Float = 0,
        enableBackgroundSH: Bool = false,
        backgroundSHDegree: Int = 3,
        bgCoeffsLR: Float = 2.5e-3,
        alphaLossWeight: Float = 0.15,
        surfaceAlphaWeight: Float = 0
    ) throws {
        self.model = model
        self.renderer = renderer
        self.fusedSSIM = try FusedSSIM()
        self.lambdaSSIM = lambdaSSIM
        self.opacityReg = opacityReg
        self.scaleReg = scaleReg
        self.maxScale = maxScale
        self.logMaxScale = maxScale > 0 ? Foundation.log(maxScale) : 0
        self.enableBackgroundSH = enableBackgroundSH
        self.backgroundSHDegree = backgroundSHDegree
        self.alphaLossWeight = alphaLossWeight
        self.surfaceAlphaWeight = surfaceAlphaWeight
        self.bgCoeffs = BackgroundSH.initialCoeffs(degree: enableBackgroundSH ? backgroundSHDegree : 0)
        self.optBgCoeffs = AdamOptimizer(lr: bgCoeffsLR)
        self.stateBgCoeffs = optBgCoeffs.newState(like: bgCoeffs)
        self.bgKernels = enableBackgroundSH ? try BackgroundSHKernels() : nil
        self.optMeans = AdamOptimizer(lr: meansLR)
        self.optQuats = AdamOptimizer(lr: quatsLR)
        self.optLogScales = AdamOptimizer(lr: logScalesLR)
        self.optLogitOpacities = AdamOptimizer(lr: logitOpacitiesLR)
        self.optSh0 = AdamOptimizer(lr: sh0LR)
        self.optShN = AdamOptimizer(lr: shNLR)
        self.stateMeans = optMeans.newState(like: model.means)
        self.stateQuats = optQuats.newState(like: model.quats)
        self.stateLogScales = optLogScales.newState(like: model.logScales)
        self.stateLogitOpacities = optLogitOpacities.newState(like: model.logitOpacities)
        self.stateSh0 = optSh0.newState(like: model.sh0)
        self.stateShN = optShN.newState(like: model.shN)
    }

    public var gaussianCount: Int { model.count }

    @inline(__always)
    private func mark(_ stage: String, since t0: Date) {
        guard Self.profileHook != nil || TrainingSignpost.log.signpostsEnabled else { return }
        let ms = Date().timeIntervalSince(t0) * 1000
        Self.profileHook?(stage, ms)
        TrainingSignpost.event(stage, ms: ms)
    }

    /// References the `GaussianParamStore`. Rewrites and realizes the 6 parameter groups together
    /// with their Adam moments. Per-group stepCount is left untouched, so bias correction stays
    /// continuous across densify.
    public func updateParamsWithOptimizer(
        paramFn: (GaussianParamKind, MLXArray) -> MLXArray,
        momentFn: (GaussianParamKind, MLXArray) -> MLXArray
    ) {
        model.means = paramFn(.means, model.means)
        stateMeans.m = momentFn(.means, stateMeans.m)
        stateMeans.v = momentFn(.means, stateMeans.v)

        model.quats = paramFn(.quats, model.quats)
        stateQuats.m = momentFn(.quats, stateQuats.m)
        stateQuats.v = momentFn(.quats, stateQuats.v)

        model.logScales = paramFn(.logScales, model.logScales)
        stateLogScales.m = momentFn(.logScales, stateLogScales.m)
        stateLogScales.v = momentFn(.logScales, stateLogScales.v)

        model.logitOpacities = paramFn(.logitOpacities, model.logitOpacities)
        stateLogitOpacities.m = momentFn(.logitOpacities, stateLogitOpacities.m)
        stateLogitOpacities.v = momentFn(.logitOpacities, stateLogitOpacities.v)

        model.sh0 = paramFn(.sh0, model.sh0)
        stateSh0.m = momentFn(.sh0, stateSh0.m)
        stateSh0.v = momentFn(.sh0, stateSh0.v)

        model.shN = paramFn(.shN, model.shN)
        stateShN.m = momentFn(.shN, stateShN.m)
        stateShN.v = momentFn(.shN, stateShN.v)

        eval(
            model.means, model.quats, model.logScales, model.logitOpacities,
            model.sh0, model.shN,
            stateMeans.m, stateMeans.v, stateQuats.m, stateQuats.v,
            stateLogScales.m, stateLogScales.v,
            stateLogitOpacities.m, stateLogitOpacities.v,
            stateSh0.m, stateSh0.v, stateShN.m, stateShN.v
        )
    }

    public struct StepOutput {
        public let loss: Float
        public let cache: GaussianRenderer.ForwardCache
    }

    /// Losses of the geometric regularization on the last step that ran it (read back on access).
    public var lastGeometryLosses: GeometryRegularization.Losses? { lastGeometryLossesReader?() }
    private var lastGeometryLossesReader: (() -> GeometryRegularization.Losses)?

    /// Runs forward + backward + Adam once and returns the loss before the update. With `geometry`,
    /// its regularization (and `mvsDepth`, this view's MVS depth) joins the gradients.
    @discardableResult
    public func step(
        camera: Camera, targetImage: MLXArray, shDegree: Int = 3,
        geometry: GeometryRegularization? = nil, mvsDepth: DepthTarget? = nil,
        surfaceMask: MLXArray? = nil, opaqueMask: MLXArray? = nil
    ) -> Float {
        stepWithCache(camera: camera, targetImage: targetImage, shDegree: shDegree,
                      geometry: geometry, mvsDepth: mvsDepth, surfaceMask: surfaceMask,
                      opaqueMask: opaqueMask).loss
    }

    /// Renders the current model for evaluation and returns [H,W,3], compositing the background as
    /// in training when bg-SH is on (no state update).
    @discardableResult
    public func renderForEval(camera: Camera, shDegree: Int = 3) -> MLXArray {
        let (render, renderAlpha, _) = renderer.forward(model: model, camera: camera, shDegree: shDegree)
        guard enableBackgroundSH, let bgK = bgKernels else { return render }
        let vm = camera.viewmat.asArray(Float.self)
        let intr = camera.intrinsics.asArray(Float.self)
        let (comp, _) = bgK.forward(
            render: render, renderAlpha: renderAlpha, coeffs: bgCoeffs,
            width: camera.width, height: camera.height, degree: backgroundSHDegree,
            fx: intr[0], fy: intr[1], cx: intr[2], cy: intr[3], viewmatRowMajor: vm)
        return comp
    }

    /// Same update as `step`, but also returns the pre-update forward cache (IGSPlus uses it to map
    /// edge scores back to visible Gaussians).
    public func stepWithCache(
        camera: Camera,
        targetImage: MLXArray,
        shDegree: Int = 3,
        geometry: GeometryRegularization? = nil,
        mvsDepth: DepthTarget? = nil,
        // [gh, gw] 1 where LiDAR or MVS measured a surface (`TrainView.surfaceMask`).
        surfaceMask: MLXArray? = nil,
        // [gh, gw] 1 where alpha is pulled to 1 (`TrainView.opaqueMask`).
        opaqueMask: MLXArray? = nil
    ) -> StepOutput {
        let (render, renderAlpha, cache) = renderer.forward(model: model, camera: camera, shDegree: shDegree)
        let h = camera.height
        let w = camera.width

        // bg-SH compositing. The fused kernel does dir+SH+compositing in one pass without
        // materializing basis [P,bands]. The loss is taken on `composited`; bgImage is reused for
        // vAlpha / the alpha loss in backward.
        let composited: MLXArray
        var bgImage: MLXArray? = nil        // [H,W,3]
        if enableBackgroundSH, let bgK = bgKernels {
            let vm = camera.viewmat.asArray(Float.self)
            let intr = camera.intrinsics.asArray(Float.self)
            let (comp, bg) = bgK.forward(
                render: render, renderAlpha: renderAlpha, coeffs: bgCoeffs,
                width: w, height: h, degree: backgroundSHDegree,
                fx: intr[0], fy: intr[1], cx: intr[2], cy: intr[3], viewmatRowMajor: vm)
            composited = comp
            bgImage = bg
        } else {
            composited = render
        }

        // Image loss and dL/d(composited). L1 is closed-form; D-SSIM uses the fused SSIM kernel.
        let target = targetImage
        let lambda = lambdaSSIM
        let n = Float(h * w * 3)

        // The L1 cotangent is (1-λ)·sign/N, and ∂composited/∂render=1, so it becomes vRender as is.
        // With λ>0 the L1 cotangent is fused into the SSIM backward kernel, and when tiled the loss
        // is also built from per-tile partial sums.
        let lossArray: MLXArray
        let vRender: MLXArray
        if lambda > 0 {
            let (ssimMap, ssimCache, lossPartials) = fusedSSIM.forward(composited, target)
            // As in gsplat (fused-ssim padding="valid"): the 5px border, where zero padding
            // distorts μ/σ, is excluded from the loss mean and dL/d(map) is 0 there too (no
            // spurious gradients to Gaussians touching the border).
            let pad = 5  // radius of the 11x11 window
            let validH = max(0, h - 2 * pad)
            let validW = max(0, w - 2 * pad)
            let nValid = Float(validH * validW * 3)
            if let lp = lossPartials {
                // tiled: divide the per-tile partial sums by H*W*3 for L1 and by the valid pixel
                // count for SSIM (summation order only affects the displayed loss, not gradients).
                lossArray = (1 - lambda) * (lp.l1Sum / n) + lambda * (1 - lp.ssimSum / nValid)
            } else {
                // non-tiled: reduce in MLX.
                let l1 = MLX.abs(composited - target).mean()
                let validMap = ssimMap[pad..<(h - pad), pad..<(w - pad), 0...]
                lossArray = (1 - lambda) * l1 + lambda * (1 - validMap.mean())
            }
            // dL/d(map) = -λ/N_valid in the valid region, 0 outside. Built with MLX.padded
            // (subscript assignment is a no-op in this MLX version). It depends only on resolution
            // and the fixed λ, so it is materialized once and cached.
            let resKey = h &* 100_000 &+ w
            let dLdMapFull: MLXArray
            if let cached = dLdSsimMapCache[resKey] {
                dLdMapFull = cached
            } else {
                let validDLdMap =
                    (-lambda / nValid) * MLXArray.ones([validH, validW, 3])
                let built = MLX.padded(
                    validDLdMap,
                    widths: [.init((pad, pad)), .init((pad, pad)), .init((0, 0))],
                    value: MLXArray(0.0 as Float)
                )
                eval(built)
                dLdSsimMapCache[resKey] = built
                dLdMapFull = built
            }
            // Fused backward: SSIM cotangent + (1-λ)·sign(diff)/N in one dispatch.
            vRender = fusedSSIM.backward(
                composited, target, dLdMap: dLdMapFull, cache: ssimCache,
                oneMinusLambda: 1 - lambda, invN: 1 / n
            )
        } else {
            // λ==0: L1 only (MLX).
            let diff = composited - target
            lossArray = MLX.abs(diff).mean()
            vRender = (1 - lambda) * MLX.sign(diff) / n
        }

        // Alpha cotangent and bg-SH coefficient gradients. With bg-SH on, ∂final/∂alpha = -bg gives
        // dL/dalpha = -Σ_c(vRender·bg), plus the alpha loss. When off, vAlpha = zeros (cached by
        // resolution key).
        let vAlpha: MLXArray
        var dBgCoeffs: MLXArray? = nil
        var lossArrayFinal = lossArray
        if enableBackgroundSH, let bg = bgImage, let bgK = bgKernels {
            if alphaLossWeight > 0 || surfaceAlphaWeight > 0 {
                let surface = surfaceMask.map { Self.upsampleNearest($0, height: h, width: w) }
                    ?? MLXArray.zeros([h, w])
                let opaque = opaqueMask.map { Self.upsampleNearest($0, height: h, width: w) }
                    ?? MLXArray.zeros([h, w])
                let terms = compiledAlphaTerms([vRender, bg, target, renderAlpha, surface, opaque])
                vAlpha = terms[0]
                lossArrayFinal = lossArrayFinal + terms[1]
            } else {
                vAlpha = -(vRender * bg).sum(axis: 2)                  // [H,W]
            }
            // Compute the bg coefficient gradient Σ_px basis(dir_px)·(vRender·(1-alpha)) with the
            // fused kernel.
            let vm = camera.viewmat.asArray(Float.self)
            let intr = camera.intrinsics.asArray(Float.self)
            dBgCoeffs = bgK.backwardCoeffs(
                vRender: vRender, renderAlpha: renderAlpha,
                width: w, height: h, degree: backgroundSHDegree,
                fx: intr[0], fy: intr[1], cx: intr[2], cy: intr[3], viewmatRowMajor: vm)
        } else {
            let vAlphaKey = h &* 100_000 &+ w
            if let cachedVA = vAlphaZeroCache[vAlphaKey] {
                vAlpha = cachedVA
            } else {
                let z = MLXArray.zeros([h, w])
                eval(z)
                vAlphaZeroCache[vAlphaKey] = z
                vAlpha = z
            }
        }
        let tLossEval = Date()
        eval(lossArrayFinal, vRender, vAlpha)
        mark("loss_eval", since: tLossEval)
        let tLossItem = Date()
        let loss = lossArrayFinal.item(Float.self)
        mark("loss_item", since: tLossItem)

        let grads = renderer.backward(
            model: model, camera: camera, cache: cache,
            vRenderColor: vRender, vRenderAlpha: vAlpha
        )

        // Adam step for the background SH coefficients (independent of the model).
        if let dCoeffs = dBgCoeffs {
            optBgCoeffs.step(param: &bgCoeffs, grad: dCoeffs, state: &stateBgCoeffs)
        }

        // Add the MCMC regularization as parameter-space gradients (they do not pass through the
        // renderer):
        //   d/d(logitOpacity) [opacityReg·mean σ(·)] = opacityReg·σ(1-σ)/N
        //   d/d(logScale)     [scaleReg·mean exp(·)] = scaleReg·exp(·)/(N·3)
        // Do not drop the /N. Without it, it interferes with Adam's normalization, opacity
        // collapses to 0, and relocate treats every point as dead.
        var dMeans = grads.dMeans
        var dQuats = grads.dQuats
        var dLogScales = grads.dLogScales
        var dLogitOpacities = grads.dLogitOpacities
        if let geometry {
            let tGeo = Date()
            if let g = geometry.gradients(
                model: model, renderer: renderer, camera: camera, cache: cache, mvsDepth: mvsDepth) {
                dMeans = dMeans + g.dMeans
                dQuats = dQuats + g.dQuats
                dLogScales = dLogScales + g.dLogScales
                dLogitOpacities = dLogitOpacities + g.dLogitOpacities
                lastGeometryLossesReader = g.losses
            }
            mark("geometry_reg", since: tGeo)
        }
        // cache.scales / opacities are the exp / sigmoid at forward time (identical to the current
        // values, since this is before Adam).
        if scaleReg > 0 {
            dLogScales = dLogScales + scaleReg * cache.scales / Float(model.count * 3)
        }
        if opacityReg > 0 {
            let op = cache.opacities
            dLogitOpacities = dLogitOpacities + opacityReg * (op * (1 - op)) / Float(model.count)
        }

        let tAdam = Date()
        optMeans.step(param: &model.means, grad: dMeans, state: &stateMeans)
        optQuats.step(param: &model.quats, grad: dQuats, state: &stateQuats)
        optLogScales.step(param: &model.logScales, grad: dLogScales, state: &stateLogScales)
        optLogitOpacities.step(
            param: &model.logitOpacities, grad: dLogitOpacities, state: &stateLogitOpacities
        )
        optSh0.step(param: &model.sh0, grad: grads.dSh0, state: &stateSh0)
        if shDegree == 0 && shNMomentsAreZero {
            // At degree 0 the backward writes exact zeros for shN, and with zero moments the Adam
            // step is -0, so shN, m and v would stay bit-identical. Skip the kernel (about 8
            // shN-sized buffers of traffic) and only advance the step count for bias correction.
            stateShN.stepCount += 1
        } else {
            optShN.step(param: &model.shN, grad: grads.dShN, state: &stateShN)
            shNMomentsAreZero = false
        }
        // Scale cap after the Adam step.
        if maxScale > 0 { model.logScales = MLX.minimum(model.logScales, MLXArray(logMaxScale)) }
        mark("adam_enqueue", since: tAdam)

        return StepOutput(loss: loss, cache: cache)
    }

    /// With bg-SH: dL/dalpha = -Σ_c(vRender·bg) [H,W] plus the alpha loss (which penalizes alpha on
    /// background pixels, and `opaqueWeight`·(1-alpha) on the `opaque` pixels) and its terms in
    /// vAlpha. As separate MLX ops; `compiledAlphaTerms` runs them through `compile`, which fuses
    /// the ~30 full-image ops and gives the same bits at lower GPU energy per step.
    static func alphaTerms(
        vRender: MLXArray, bg: MLXArray, target: MLXArray, renderAlpha: MLXArray, weight: Float,
        surface: MLXArray? = nil, opaque: MLXArray? = nil, opaqueWeight: Float = 0
    ) -> (vAlpha: MLXArray, loss: MLXArray) {
        let h = vRender.shape[0], w = vRender.shape[1]
        let bgErr = MLX.abs(target - bg).mean(axis: 2)         // [H,W]
        let raw = (bgErr .< Float(0.003)).asType(.float32)     // [H,W] 1 = background pixel
        // As in splatfacto-w (model.py): box-filter the raw mask 3×3 and re-threshold at >0.6 to
        // avoid false penalties on thin foreground edges. The mask is treated as a constant.
        let padded = MLX.padded(
            raw, widths: [.init((1, 1)), .init((1, 1))], value: MLXArray(0.0 as Float))
        var acc = MLXArray.zeros([h, w])
        for dy in 0..<3 { for dx in 0..<3 { acc = acc + padded[dy ..< (dy + h), dx ..< (dx + w)] } }
        var mask = ((acc / 9.0) .> Float(0.6)).asType(.float32)
        // Where a depth sensor measured a surface, the pixel is foreground however well the
        // background color matches it (a plain wall the background SH has learned to imitate).
        if let surface { mask = mask * (1 - surface) }
        let mSum = MLX.maximum(mask.sum(), MLXArray(1.0 as Float))
        let alpha2d = renderAlpha.reshaped([h, w])
        var vAlpha = -(vRender * bg).sum(axis: 2) + weight * mask / mSum
        var loss = weight * (alpha2d * mask).sum() / mSum
        // Not in splatfacto-w: where a measured surface lies, the foreground has to cover it by
        // itself, or hiding the background in the viewer leaves holes where the background had
        // imitated it.
        if let opaque, opaqueWeight > 0 {
            let oSum = MLX.maximum(opaque.sum(), MLXArray(1.0 as Float))
            vAlpha = vAlpha - opaqueWeight * opaque / oSum
            loss = loss + opaqueWeight * ((1 - alpha2d) * opaque).sum() / oSum
        }
        return (vAlpha, loss)
    }

    /// `grid` [gh, gw] resampled to [height, width] by nearest cell (both span the same field of
    /// view).
    public static func upsampleNearest(_ grid: MLXArray, height: Int, width: Int) -> MLXArray {
        let gh = grid.shape[0], gw = grid.shape[1]
        let rows = MLXArray((0..<height).map { Int32($0 * gh / height) })
        let cols = MLXArray((0..<width).map { Int32($0 * gw / width) })
        return grid.take(rows, axis: 0).take(cols, axis: 1)
    }

    /// `alphaTerms` compiled (one trace per step resolution: coarse and fine).
    private lazy var compiledAlphaTerms = { [alphaLossWeight, surfaceAlphaWeight] in
        compile { (x: [MLXArray]) -> [MLXArray] in
            let t = GaussianTrainer.alphaTerms(
                vRender: x[0], bg: x[1], target: x[2], renderAlpha: x[3], weight: alphaLossWeight,
                surface: x[4], opaque: x[5], opaqueWeight: surfaceAlphaWeight)
            return [t.vAlpha, t.loss]
        }
    }()
}
