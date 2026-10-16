// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXRandom

extension TrainingDriver {

    /// The training loop itself. Called by both trainColmap and Capture's `CaptureTraining.train`
    /// (public so Capture can use it).
    public static func runTrainingLoop(
        model: GaussianModel, views: [TrainView], sceneScale: Float,
        iterations: Int, meansLR: Float, logEvery: Int, seed: UInt64,
        strategy: (any DensificationStrategy)?,
        opacityReg: Float, scaleReg: Float, lambdaSSIM: Float = 0.2,
        replayCache: ReplayCacheMode, replayKMax: Int,
        outputPLY: URL?,
        thumbnailPNG: URL? = nil,
        maxScaleFraction: Float = 0.5,
        // When thermal=serious, sleep for step time × factor between steps (1.0 = 50% duty). 0
        // disables it.
        seriousThrottleFactor: Float = 1.0,
        // bg-SH background. The views' camera.background must be black.
        enableBackgroundSH: Bool = false,
        backgroundSHDegree: Int = 3,
        alphaLossWeight: Float = 0.15,
        // Maximum foreground SH degree (lower means less sh_eval bandwidth and heat). Capped by
        // `GaussianModel.maxShDegree`.
        maxShDegree maxShDegreeParam: Int = GaussianModel.maxShDegree,
        // coarse-to-fine: train the first iterations×fraction at 1/factor resolution to cut
        // pixel-proportional cost and heat. factor=1 or fraction=0 disables it.
        coarseToFineFactor: Int = 1,
        coarseToFineFraction: Float = 0,
        // Resolution factor num/den for the fine phase (from coarseUntil on) (default (1,1) = full,
        // e.g. (2,3)).
        fineResample: (num: Int, den: Int) = (1, 1),
        // Bring the SH degree warm-up forward (half the interval), so even short runs learn
        // view-dependent color.
        useEarlySH: Bool = false,
        // Geometric regularization (surface depth + normal consistency + MVS depth) from
        // `startFraction` of the iterations on. nil disables it.
        geometryRegularization: GeometryRegularization? = nil,
        onThermalThrottle: ((ThermalThrottleState) -> Void)? = nil,
        onSnapshot: ((GaussianModel, Int) -> Void)?,
        log: (String) -> Void
    ) throws -> Result {
        let renderer = try GaussianRenderer()
        // Forward replay cache from PocketGS Operator T (PocketGS Sec. III-C, eq. 8).
        // Mathematically equivalent to the T-reconstruction backward; differs only in numerical
        // behavior and memory/speed/heat.
        replayCache.configure(renderer, replayKMax: replayKMax, log: log)
        // The foreground scale cap is set relative to scene_scale (a fixed absolute value breaks on
        // some scenes). The default 0.5× stops blob runaway after pruning stops without clipping
        // the largest healthy scales. 0 disables it.
        let maxScale = maxScaleFraction > 0 ? maxScaleFraction * sceneScale : 0
        if maxScale > 0 {
            log(String(format: "trainer: scale cap = %.4f (%.2f × scene_scale %.4f)",
                       maxScale, maxScaleFraction, sceneScale))
        }
        let trainer = try GaussianTrainer(
            model: model, renderer: renderer, meansLR: meansLR,
            lambdaSSIM: lambdaSSIM,
            opacityReg: opacityReg, scaleReg: scaleReg,
            maxScale: maxScale,
            enableBackgroundSH: enableBackgroundSH,
            backgroundSHDegree: backgroundSHDegree,
            alphaLossWeight: alphaLossWeight
        )
        if enableBackgroundSH {
            log("rasterizer: background SH env-map enabled (degree=\(backgroundSHDegree), alphaLoss=\(alphaLossWeight))")
        }
        // gsplat's means lr: exponential decay from scene-scale times to 1/100. MCMC's Langevin
        // noise (lr·noiseLR) also follows this decay, so it does not disturb and diverge an already
        // converged model late in training.
        let baseMeansLR = meansLR * sceneScale
        let lrDecay = 0.01
        log("scene scale \(sceneScale); default means LR \(baseMeansLR) → \(baseMeansLR * Float(lrDecay)) over \(iterations) iters")
        if let schedule = (strategy as? MCMCStrategy)?.scaleLRSchedule {
            log("MCMC scale LR \(schedule.start) → \(schedule.end) over \(iterations) iters")
        }
        if let igs = strategy as? IGSPlusStrategy {
            let igsMeansStart = igs.meansLearningRate(step: 0, iterations: iterations, sceneScale: sceneScale)
            let igsMeansEnd = igs.meansLearningRate(
                step: Swift.max(0, iterations - 1), iterations: iterations, sceneScale: sceneScale
            )
            log("IGSPlus means LR \(igsMeansStart) → \(igsMeansEnd) over \(iterations) iters")
            log("IGSPlus scale LR \(igs.scaleLRStart) → \(igs.scaleLREnd) over \(iterations) iters")
            log("IGSPlus edge-score importance enabled (5x5 blur, Sobel, NMS, median-normalized)")
        }

        MLXRandom.seed(seed)
        var rng = SystemRandomNumberGenerator()
        var strat = strategy
        var history: [Float] = []
        history.reserveCapacity(iterations)
        var edgeMapCache: [String: Densification.EdgeImportanceMap] = [:]

        // Aggregation window (iters) for per-stage profiling.
        let profileEvery = 100
        // Interval (iters) for diagnostic logs of gradient / radii / opacity distributions. Kept to
        // about 5 per run + the final iter.
        let diagEvery = Swift.max(profileEvery, iterations / 5 / profileEvery * profileEvery)
        // As in gsplat, raise the active SH degree by 1 every shDegreeInterval so the DC color is
        // fit first. The interval is iterations/(maxShDegree+1), so the max degree is reached for
        // any iteration count.
        let maxShDegree = Swift.max(0, Swift.min(GaussianModel.maxShDegree, maxShDegreeParam))
        let shDegreeDivisor = (maxShDegree + 1) * (useEarlySH ? 2 : 1)
        let shDegreeInterval = Swift.max(1, iterations / shDegreeDivisor)

        let profiler = TrainingProfiler()
        profiler.install()
        defer { profiler.uninstall() }

        let viewSampler = ViewSampler(views: views)
        if let cdf = viewSampler.cdf {
            let minW = views.map(\.weight).min() ?? 1
            let maxW = views.map(\.weight).max() ?? 1
            log(String(format:
                "[ViewSampling] weighted draw over %d views (weight min=%.2f max=%.2f total=%.1f)",
                views.count, minW, maxW, cdf.last ?? 0))
        }

        // Time the log callback blocks the training thread (the App's handler does a synchronous
        // write+fsync on this thread). Reported at the end as [UICallback].
        var uiCallbackMs = 0.0
        var uiCallbackCount = 0
        func emitLog(_ s: String) {
            let t0 = Date()
            log(s)
            uiCallbackMs += Date().timeIntervalSince(t0) * 1000
            uiCallbackCount += 1
        }
        // coarse-to-fine: train [0, coarseUntil) at 1/factor resolution.
        let coarseUntil = (coarseToFineFactor > 1 && coarseToFineFraction > 0)
            ? Int(Float(iterations) * Swift.min(1, Swift.max(0, coarseToFineFraction)))
            : 0
        if coarseUntil > 0 {
            log("coarse-to-fine: factor \(coarseToFineFactor) for iter [0..<\(coarseUntil)] "
                + "(\(iterations - coarseUntil) iters at "
                + (fineResample.num < fineResample.den ? "fine-resample" : "full res") + ")")
        }
        if fineResample.num < fineResample.den, let v0 = views.first {
            let fw = v0.camera.width / fineResample.den * fineResample.num
            let fh = v0.camera.height / fineResample.den * fineResample.num
            log("fine-resample: \(fineResample.num)/\(fineResample.den) → fine phase "
                + "\(v0.camera.width)x\(v0.camera.height) → \(fw)x\(fh) (pixel "
                + String(format: "%.0f%%", 100 * Double(fw * fh) / Double(v0.camera.width * v0.camera.height)) + ")")
        }

        // Print the build configuration every run. Performance numbers are only valid in Release
        // (in Debug the CPU glue is slow and the bottleneck stage changes).
        #if DEBUG
        log("[build] DEBUG — ⚠️ performance numbers are invalid (Swift/MLX CPU ~38x slow). Measure again in Release")
        #else
        log("[build] RELEASE")
        #endif

        let geometryStart = geometryRegularization.map { Int(Float(iterations) * Swift.min(1, $0.startFraction)) }
        if let g = geometryRegularization, let start = geometryStart {
            let mvsViews = views.filter { $0.mvsDepth != nil }.count
            log(String(format: "geometry regularization from iter %d every %d: normal %.3f, MVS depth %.3f "
                       + "(%d/%d views with MVS depth), depth profile %d",
                       start, g.every, g.normalWeight, g.mvsDepthWeight, mvsViews, views.count,
                       g.depth.profile.rawValue))
        }

        let runStart = Date()

        // Wall time of the previous step (used for the sleep amount when serious).
        var lastStepSeconds = 0.0
        var thermal = ThermalGovernor(seriousThrottleFactor: seriousThrottleFactor, onChange: onThermalThrottle)
        for it in 0..<iterations {
            thermal.beforeStep(iteration: it, lastStepSeconds: lastStepSeconds, log: log)
            // Enable diagnostics only on steps that log (every step would add sync barriers).
            let isDiagIter = it % diagEvery == 0 || it == iterations - 1
            GaussianRenderer.diagnosticEnabled = isDiagIter
            if let igs = strat as? IGSPlusStrategy {
                let lr = igs.meansLearningRate(step: it, iterations: iterations, sceneScale: sceneScale)
                trainer.optMeans.lr = lr
                trainer.optLogScales.lr = igs.scaleLearningRate(step: it, iterations: iterations)
            } else {
                let lr = baseMeansLR * Float(pow(lrDecay, Double(it) / Double(max(1, iterations))))
                trainer.optMeans.lr = lr
                if let scaleLR = (strat as? MCMCStrategy)?.scaleLearningRate(
                    step: it, iterations: iterations)
                {
                    trainer.optLogScales.lr = scaleLR
                }
            }
            let lr = trainer.optMeans.lr
            let shDegree = Swift.min(it / shDegreeInterval, maxShDegree)
            let v = views[viewSampler.sample(using: &rng)]
            let cf = it < coarseUntil ? coarseToFineFactor : 1
            // The coarse phase box-averages the uint8 v.image directly (no full-res float32); the
            // fine phase uses fineResample.
            let stepCamera: Camera
            let stepTarget: MLXArray
            // Scale from step pixels to full-res pixels (the IGSPlus edge map is full res).
            let stepToFullRes: Float
            if cf > 1 {
                stepCamera = coarseCamera(v.camera, factor: cf)
                stepTarget = coarseDownsample(v.image, factor: cf)
                stepToFullRes = Float(cf)
            } else if fineResample.num < fineResample.den {
                stepCamera = coarseCameraScaled(v.camera, num: fineResample.num, den: fineResample.den)
                stepTarget = resampleRational(v.image, num: fineResample.num, den: fineResample.den)
                stepToFullRes = Float(fineResample.den) / Float(fineResample.num)
            } else {
                stepCamera = v.camera
                stepTarget = v.imageFloat   // full-res (uint8→float[0,1])
                stepToFullRes = 1
            }
            let _pfT0 = Date()
            // Drain an autoreleasepool every step. In a loop without a run loop, autorelease
            // objects such as Metal command buffers keep piling up and long runs OOM.
            // Keep the forward cache (including the replay pixel buffer, ~150MB in the fine phase)
            // only on the steps where IGSPlus builds edge scores.
            let needsCache = (strat as? IGSPlusStrategy)?.shouldPrepareImportance(
                step: it, gaussianCount: trainer.gaussianCount) ?? false
            let geometry: GeometryRegularization? = {
                guard let g = geometryRegularization, let start = geometryStart, it >= start,
                      (it - start) % g.every == 0 else { return nil }
                return g
            }()
            let (loss, stepCache): (Float, GaussianRenderer.ForwardCache?)
                = autoreleasepool {
                if needsCache {
                    let output = trainer.stepWithCache(
                        camera: stepCamera, targetImage: stepTarget, shDegree: shDegree,
                        geometry: geometry, mvsDepth: v.mvsDepth
                    )
                    return (output.loss, output.cache)
                } else {
                    return (
                        trainer.step(
                            camera: stepCamera, targetImage: stepTarget, shDegree: shDegree,
                            geometry: geometry, mvsDepth: v.mvsDepth
                        ),
                        nil
                    )
                }
            }
            let _pfT1 = Date()
            history.append(loss)
            // densify / prune / perturb (no-op without a strategy).
            if var igs = strat as? IGSPlusStrategy {
                if let stepCache {
                    let key = "\(v.name)#\(v.camera.width)x\(v.camera.height)"
                    let edgeMap: Densification.EdgeImportanceMap
                    if let cached = edgeMapCache[key] {
                        edgeMap = cached
                    } else {
                        let computed = Densification.edgeImportanceMap(image: v.imageFloat)
                        edgeMapCache[key] = computed
                        edgeMap = computed
                    }
                    igs.setImportanceScores(
                        Densification.edgeImportanceScores(
                            edgeMap: edgeMap, cache: stepCache, stepToMapScale: stepToFullRes)
                    )
                }
                igs.postStep(store: trainer, step: it, lr: lr)
                strat = igs
            } else {
                strat?.postStep(store: trainer, step: it, lr: lr)
            }
            let _pfT2 = Date()
            // The serious-state sleep amount is measured including postStep. step only waits up to
            // the loss; backward and Adam flow to the GPU with MCMC's position noise (the postStep
            // eval), so measuring step alone would make the sleep a fraction of what it should be.
            lastStepSeconds = _pfT2.timeIntervalSince(_pfT0)
            // The strategy's diagnostic is kept as a string and pulled by the caller (to avoid
            // @escaping propagation). It is not printed on every fire; the profile window report
            // carries the latest one.
            profiler.record(
                stepMs: _pfT1.timeIntervalSince(_pfT0) * 1000,
                postStepMs: _pfT2.timeIntervalSince(_pfT1) * 1000,
                densifyDiagnostic: (strat as? MCMCStrategy)?.lastDiagnostic
                    ?? (strat as? IGSPlusStrategy)?.lastDiagnostic)
            // Pass the post-step model to the live viewer (between steps, so no concurrent
            // mutation).
            onSnapshot?(model, it)
            // Always print the final iter too (so it does not look like it stopped early).
            if logEvery > 0 && (it % logEvery == 0 || it == iterations - 1) {
                // 1 iter = 1 view, so the raw loss swings a lot per view. Use a window of the view
                // count (minimum 20) and print a rolling mean that covers each view about once (the
                // App's smoothed loss assumes this window too).
                let window = Swift.max(20, views.count)
                let lo = Swift.max(0, history.count - window)
                let recent = history[lo..<history.count]
                let smoothed = recent.reduce(0, +) / Float(recent.count)
                // The App parses the "iter N: loss X" format, so do not change it.
                emitLog(String(format: "iter %d: loss %.5f (smooth %.5f, w=%d)  (%d gaussians)  (%@)",
                           it, loss, smoothed, recent.count,
                           model.count, v.name as NSString))
                if let start = geometryStart, it >= start, let g = trainer.lastGeometryLosses {
                    emitLog(String(format: "  geometry: normal %.4f, MVS depth %.4f, depth coverage %.1f%%",
                                   g.normal, g.mvsDepth, 100 * g.depthCoverage))
                }
            }
            if isDiagIter, let d = GaussianRenderer.lastDiagnostic {
                emitLog(Self.diagnosticReport(
                    d, replayOverflow: renderer.useReplayCache
                        ? (kMax: replayKMax, pixels: GaussianRenderer.lastReplayCacheOverflowPixels) : nil))
            }
            if (it + 1) % profileEvery == 0 || it == iterations - 1, let report = profiler.closeWindow(at: it) {
                emitLog(report)
            }
            // Flush the MLX buffer pool only every 100 iters. Flushing every iter prevents reusing
            // Adam's output buffers from the pool and runs makeBuffer every time.
            if it % 100 == 0 {
                MLX.Memory.clearCache()
            }
        }
        GaussianRenderer.diagnosticEnabled = false
        if let energy = profiler.energySummary() {
            log(energy)
        }

        // Cost the log callback imposed on the training thread (if large, move fsync / file I/O off
        // the training thread).
        let runMs = Date().timeIntervalSince(runStart) * 1000
        log(String(format:
            "[UICallback] log-callback training-thread cost: %.0fms over %d calls (%.2f%% of %.1fs run; %.2fms/call avg)",
            uiCallbackMs, uiCallbackCount,
            runMs > 0 ? uiCallbackMs / runMs * 100 : 0,
            runMs / 1000,
            uiCallbackCount > 0 ? uiCallbackMs / Double(uiCallbackCount) : 0))

        // Write PLY / thumbnail (bg-SH is baked into a copy; the training model is left unchanged).
        try exportTrained(
            model: model, renderer: renderer, trainer: trainer, sceneScale: sceneScale,
            enableBackgroundSH: enableBackgroundSH, backgroundSHDegree: backgroundSHDegree,
            outputPLY: outputPLY, thumbnailPNG: thumbnailPNG, views: views, log: log)

        // Keep the trainer (model + Adam momentum) and views for continue training.
        let session = ResumableTrainingSession(
            trainer: trainer, views: views, viewSampler: viewSampler,
            sceneScale: sceneScale, maxShDegree: maxShDegree, geometryRegularization: geometryRegularization,
            totalIterations: iterations)

        return Result(
            gaussianCount: model.count, viewCount: views.count,
            lossHistory: history, session: session
        )
    }

    /// Log for diagnostic iters (radii / scale / opacity / gradients / replay cache overflow).
    private static func diagnosticReport(
        _ d: GaussianRenderer.Diagnostic, replayOverflow: (kMax: Int, pixels: Int)?
    ) -> String {
        func pct(_ c: Int) -> Double { 100.0 * Double(c) / Double(max(1, d.n)) }
        let replay = replayOverflow.map { " | replay K=\($0.kMax) overflowPixels=\($0.pixels)" } ?? ""
        let lines = [
            String(format: "  diag: radii_x_max=%d radii_y_max=%d culled=%d/%d (%.1f%%)%@",
                   d.radiiMaxX, d.radiiMaxY, d.radiiCulled, d.n, pct(d.radiiCulled), replay as NSString),
            String(format:
                "       scale|med=%.4f max=%.3f >1:%d| bloat >0.05/0.10/0.20:%d/%d/%d (%.1f%%/%.1f%%/%.1f%%)| "
                + "opa|med=%.4f max=%.4f <1/255:%d (%.1f%%)|",
                d.scaleMedian, d.scaleMax, d.scaleAbove1Count,
                d.scaleAbove005Count, d.scaleAbove010Count, d.scaleAbove020Count,
                pct(d.scaleAbove005Count), pct(d.scaleAbove010Count), pct(d.scaleAbove020Count),
                d.opacityMedian, d.opacityMax, d.opacityBelowThreshCount, pct(d.opacityBelowThreshCount)),
            String(format:
                "       dMeans|%.3g/%.3g| gradDirs|%.3g/%.3g| vColors|%.3g/%.3g| "
                + "maxRadiiGauss[idx=%d]: scale=%.4f opa=%.4f z=%.3f",
                d.dMeansAbsMean, d.dMeansAbsMax, d.gradDirsAbsMean, d.gradDirsAbsMax,
                d.vColorsAbsMean, d.vColorsAbsMax,
                d.maxRadiiGaussIndex, d.maxRadiiGaussScale, d.maxRadiiGaussOpac, d.maxRadiiGaussDepth),
        ]
        return lines.joined(separator: "\n")
    }

    /// Writes the trained model to PLY / thumbnail (called from `runTrainingLoop` and
    /// `continueTraining`).
    static func exportTrained(
        model: GaussianModel, renderer: GaussianRenderer, trainer: GaussianTrainer,
        sceneScale: Float, enableBackgroundSH: Bool, backgroundSHDegree: Int,
        outputPLY: URL?, thumbnailPNG: URL?, views: [TrainView],
        cull: PLYWriter.CullOptions = PLYWriter.CullOptions(),
        log: (String) -> Void
    ) throws {
        // bg-SH is baked into environment-sphere Gaussians for viewers. Appending to the training
        // model would duplicate the bake on continued training and mismatch the optimizer state and
        // shapes, so it is baked into a shallowCopy (bake-on-copy).
        let exportModel: GaussianModel
        if enableBackgroundSH {
            let centroidArr = model.means.mean(axis: 0)
            eval(centroidArr)
            let cc = centroidArr.asArray(Float.self)
            let bakedCount = 6000
            let bakedRadius = 24.0 * sceneScale
            let copy = model.shallowCopy()
            copy.appendBakedBackgroundSphere(
                coeffs: trainer.bgCoeffs, degree: backgroundSHDegree,
                center: SIMD3<Float>(cc[0], cc[1], cc[2]),
                radius: bakedRadius, count: bakedCount, opacity: 0.9)
            log(String(format: "baked bg-SH → %d env gaussians for viewer (R=%.3f, %.1f×scene)",
                       bakedCount, bakedRadius, 24.0))
            exportModel = copy
        } else {
            exportModel = model
        }

        if let outputPLY {
            let stats = try PLYWriter.writeGaussianPLY(model: exportModel, to: outputPLY, cull: cull)
            log("wrote \(stats.kept) gaussians to \(outputPLY.path)")
            do {
                try PLYWriter.writeBackgroundSidecar(splatCount: stats.keptEnvironment, siblingOf: outputPLY)
            } catch {
                log("background sidecar write failed: \(error)")
            }
            if stats.kept < stats.total {
                log("  PLY cull: \(stats.total) → \(stats.kept) kept (NaN/Inf \(stats.droppedNonFinite), "
                    + "low-opac \(stats.droppedLowOpacity), giant-scale \(stats.droppedGiantScale), "
                    + "outlier-pos \(stats.droppedOutlierPosition))")
            }
        }

        if let thumbnailPNG, !views.isEmpty {
            // Render up to 8 views and use the one with the largest mean alpha (how much of the
            // frame the subject fills) as the thumbnail.
            let step = Swift.max(1, views.count / 8)
            var bestColor: MLXArray?
            var bestView: TrainView?
            var bestCoverage: Float = -1
            for idx in Swift.stride(from: 0, to: views.count, by: step) {
                let out = renderer.forward(model: exportModel, camera: views[idx].camera, shDegree: 3)
                let cov = out.renderAlpha.mean()
                eval(cov)
                let covVal = cov.item(Float.self)
                if covVal > bestCoverage {
                    eval(out.renderColor)
                    bestCoverage = covVal
                    bestColor = out.renderColor
                    bestView = views[idx]
                }
            }
            if let bestColor {
                do {
                    try ImagePNG.write(bestColor, to: thumbnailPNG, maxDim: 512)
                    if let bestView {
                        try writePreviewCameraMetadata(
                            view: bestView,
                            coverage: bestCoverage,
                            to: thumbnailPNG.deletingPathExtension()
                                .appendingPathExtension("preview.json")
                        )
                    }
                    log(String(format: "wrote thumbnail (coverage %.3f) to %@",
                               bestCoverage, thumbnailPNG.path))
                } catch {
                    log("thumbnail render failed: \(error)")
                }
            }
        }
    }

    /// Fine-tunes a finished session for N more iters (no densification, full resolution, max SH
    /// degree, lr fixed at the current value). Adam momentum is still in the trainer, so it resumes
    /// continuously. Afterwards overwrites the PLY and updates totalIterations.
    public static func continueTraining(
        session: ResumableTrainingSession,
        additionalIterations: Int,
        outputPLY: URL?,
        thumbnailPNG: URL? = nil,
        logEvery: Int = 50,
        onProgress: ((Int, Float) -> Void)? = nil,
        onThermalThrottle: ((ThermalThrottleState) -> Void)? = nil,
        log: (String) -> Void
    ) throws -> Result {
        let trainer = session.trainer
        let views = session.views
        guard !views.isEmpty, additionalIterations > 0 else {
            return Result(gaussianCount: trainer.model.count, viewCount: views.count,
                          lossHistory: [], session: session)
        }
        var rng = SystemRandomNumberGenerator()
        let shDegree = session.maxShDegree
        let startIter = session.totalIterations
        log(String(format:
            "Train more: fine-tuning +%d iter (no densification / full resolution / SH degree %d / fixed LR=%.3g). Total %d → %d.",
            additionalIterations, shDegree, trainer.optMeans.lr, startIter, startIter + additionalIterations))
        if let g = session.geometryRegularization {
            log("geometry regularization continues every \(g.every) iter")
        }
        GaussianRenderer.diagnosticEnabled = false
        var history: [Float] = []
        history.reserveCapacity(additionalIterations)
        // Same thermal guard as the initial training (prevents critical / an OS thermal kill during
        // continuation too).
        var thermal = ThermalGovernor(seriousThrottleFactor: 1.0, onChange: onThermalThrottle)
        var lastStepSeconds = 0.0
        for i in 0..<additionalIterations {
            thermal.beforeStep(iteration: startIter + i, lastStepSeconds: lastStepSeconds, log: log)
            let t0 = Date()
            // The run's geometric regularization continues at its interval, so fine-tuning keeps the
            // surfaces it flattened.
            let geometry = session.geometryRegularization.flatMap { i % $0.every == 0 ? $0 : nil }
            let loss: Float = autoreleasepool {
                let v = views[session.viewSampler.sample(using: &rng)]
                // The scale cap is applied inside trainer.step.
                return trainer.step(camera: v.camera, targetImage: v.imageFloat, shDegree: shDegree,
                                    geometry: geometry, mvsDepth: v.mvsDepth)
            }
            lastStepSeconds = Date().timeIntervalSince(t0)
            history.append(loss)
            let cumulative = startIter + i + 1
            if i % logEvery == 0 || i == additionalIterations - 1 {
                // The App's parseProgress matches the "iter " prefix, so do not change the "extra
                // iter" (continue-training iter) prefix (progress goes through onProgress).
                log("extra iter \(cumulative): loss \(String(format: "%.4f", loss)) (\(trainer.model.count) gaussians)")
            }
            onProgress?(i + 1, loss)
        }
        session.totalIterations += additionalIterations

        // Disable the opacity cull during continue training (minOpacity=0), so points below 1/255
        // do not disappear on every export and the count keeps shrinking. The scale / bbox / NaN
        // culls stay.
        try exportTrained(
            model: trainer.model, renderer: trainer.renderer, trainer: trainer,
            sceneScale: session.sceneScale,
            enableBackgroundSH: trainer.enableBackgroundSH,
            backgroundSHDegree: trainer.backgroundSHDegree,
            outputPLY: outputPLY, thumbnailPNG: thumbnailPNG, views: views,
            cull: PLYWriter.CullOptions(minOpacity: 0), log: log)

        return Result(
            gaussianCount: trainer.model.count, viewCount: views.count,
            lossHistory: history, session: session
        )
    }

    private struct PreviewCameraMetadata: Codable {
        let version: Int
        let sourceViewName: String
        let coverage: Float
        let width: Int
        let height: Int
        let viewmat: [Float]
        let intrinsics: [Float]
    }

    private static func writePreviewCameraMetadata(
        view: TrainView,
        coverage: Float,
        to url: URL
    ) throws {
        eval(view.camera.viewmat, view.camera.intrinsics)
        let metadata = PreviewCameraMetadata(
            version: 1,
            sourceViewName: view.name,
            coverage: coverage,
            width: view.camera.width,
            height: view.camera.height,
            viewmat: view.camera.viewmat.asArray(Float.self),
            intrinsics: view.camera.intrinsics.asArray(Float.self)
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(metadata).write(to: url, options: .atomic)
    }
}
