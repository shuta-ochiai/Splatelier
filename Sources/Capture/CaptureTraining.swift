// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Bridge that builds TrainView[] from an on-device capture's `CapturedScene` and trains
// (counterpart of `ColmapTraining`). CapturedFrame intrinsics are already scaled to
// PixelBufferToBytes' output resolution (by convention ARKitCaptureSession applies
// `PoseConversion.scaledIntrinsics` before storing), so no rescaling is needed.

import Foundation
import MLX
import Trainer
import simd

public enum CaptureTraining {

    /// Recommended training iterations proportional to the capture frame count (1 iter = 1 view),
    /// clamped to [minIters, maxIters] and rounded to the stepper granularity of 50. ~8 iter/frame
    /// in [500, 1350], favoring heat/time. MVS mode uses 4 iter/frame in [500, 2400]: with the
    /// dense MVS seed, the 2026-10 sweep found 4×frames the knee of quality vs energy, and the cap
    /// lets a 600-frame (about 2-minute) capture keep 4 iter/frame.
    public static func recommendedIterations(frameCount: Int, mvs: Bool = false) -> Int {
        let itersPerFrame = mvs ? 4 : 8
        let minIters = 500
        let maxIters = mvs ? 2400 : 1350
        let raw = Swift.max(0, frameCount) * itersPerFrame
        let rounded = ((raw + 25) / 50) * 50
        return Swift.min(maxIters, Swift.max(minIters, rounded))
    }

    /// Recommended cap on training views: 0.4 view/frame, at least 120 (300 frames → 120, 600 →
    /// 240), so a longer capture keeps the share of views of the 300-frame captures the 2026-10
    /// sweep ran on. Views beyond the cap are thinned by viewpoint coverage (FPS).
    public static func recommendedTrainingViewCap(frameCount: Int) -> Int {
        Swift.max(120, Swift.max(0, frameCount) * 2 / 5)
    }

    /// Builds TrainView[] from a set of CapturedFrames. `background` is the background color [3]
    /// shared by all views.
    public static func makeViews(
        scene: CapturedScene, tileSize: Int = 16, background: MLXArray,
        weightByTimestamp: [TimeInterval: Float]? = nil
    ) -> [TrainView] {
        makeViews(frames: scene.frames, tileSize: tileSize, background: background,
                  weightByTimestamp: weightByTimestamp)
    }

    /// Variant with explicit frames (builds training views from frames thinned separately from the
    /// seed).
    public static func makeViews(
        frames: [CapturedFrame], tileSize: Int = 16, background: MLXArray,
        weightByTimestamp: [TimeInterval: Float]? = nil
    ) -> [TrainView] {
        var views: [TrainView] = []
        views.reserveCapacity(frames.count)
        for (idx, frame) in frames.enumerated() {
            // Frames marked for undistortion rebuild `image` on every read, so read it only once.
            let image = frame.image
            precondition(
                image.shape.count == 3
                    && image.shape[0] == frame.height
                    && image.shape[1] == frame.width
                    && image.shape[2] == 3,
                "frame.image shape mismatch: expected [\(frame.height), \(frame.width), 3], got \(image.shape)"
            )
            let camera = Camera(
                viewmat: MLXArray(frame.viewmat, [16]),
                intrinsics: MLXArray(
                    [frame.intrinsics.fx, frame.intrinsics.fy,
                     frame.intrinsics.cx, frame.intrinsics.cy], [4]
                ),
                width: frame.width, height: frame.height, tileSize: tileSize,
                background: background
            )
            // Keep images resident as uint8 (callers convert to float on demand; no caching).
            eval(camera.viewmat, camera.intrinsics, image)
            let weight = weightByTimestamp?[frame.timestamp] ?? 1.0
            views.append(TrainView(
                camera: camera, image: image,
                name: String(format: "capture_%06d.png", idx),
                weight: weight
            ))
        }
        return views
    }

    /// Trains an on-device capture from its geometry (`prepareGeometry`, shared with meshing;
    /// non-COLMAP version of `TrainingDriver.trainColmap`). The seed fuses LiDAR and MVS dense
    /// points with Operator G; with none of them, a spherical random seed. The defaults are the
    /// app's configuration (`AppPreset`); the strategy and the initial-count cap are the caller's
    /// (`AppPreset.strategy`).
    public static func train(
        geometry: CaptureGeometry,
        iterations: Int = 500,
        tileSize: Int = 16,
        meansLR: Float = 1.6e-4,
        logEvery: Int = 50,
        seed: UInt64 = 0,
        strategy: (any DensificationStrategy)? = nil,
        opacityReg: Float = AppPreset.regularization,
        scaleReg: Float = AppPreset.regularization,
        lambdaSSIM: Float = 0.2,
        // If >0, hold out every Nth training view and log PSNR/SSIM after training. BA / seed use
        // all frames.
        evalHoldoutEvery: Int = 0,
        // Take the held-out views from all frames before thinning to trainingViewCap (every Nth
        // frame), instead of from the thinned training views. Keeps the held-out set the same
        // across different trainingViewCap values so they can be compared.
        evalHoldoutBeforeViewSelection: Bool = false,
        // Also evaluate held-out views against the images as captured (not resampled by
        // undistortion). For comparing with/without undistortion.
        evalAgainstCapturedImages: Bool = false,
        // Fragment slots per pixel of the replay cache (always on, see `AppPreset.replayKMax`).
        replayKMax: Int = AppPreset.replayKMax,
        syntheticSparseCount: Int = 5_000,
        // Cap on training views. Beyond it, thin with viewpoint-coverage priority (FPS) (seed uses
        // all frames). 0 disables; nil is the app's `recommendedTrainingViewCap` for the capture.
        trainingViewCap: Int? = nil,
        maxInitGaussians: Int? = nil,
        // The background is always bg-SH: an O(pixels) background that renders the foreground on
        // black and composites `render + (1-alpha)·bg(dir)`. It is baked into far-shell Gaussians
        // at export, which the viewer can hide. Degree 4 (25 bands) absorbs more of the
        // background's directional color and suppresses foreground floaters.
        backgroundSHDegree: Int = 4,
        alphaLossWeight: Float = 0.15,
        // coarse-to-fine: train the first fraction of iters at 1/factor resolution to reduce heat
        // (factor=1 disables). Avoid factor 4+, which makes tiny Gaussians sub-pixel.
        coarseToFineFactor: Int = AppPreset.coarseToFineFactor,
        coarseToFineFraction: Float = AppPreset.coarseToFineFraction,
        // Resolution scale num/den of the fine phase ((1,1) = full). For reducing heat in the fine
        // phase.
        fineResample: (num: Int, den: Int) = AppPreset.fineResample,
        outputPLY: URL? = nil,
        thumbnailPNG: URL? = nil,
        // Write the pre-training seed point cloud + camera poses to a PLY for inspection. nil is a
        // no-op.
        seedInspectionPLY: URL? = nil,
        // Geometric regularization (surface depth + normal consistency + MVS depth from the
        // geometry's depth maps). nil disables it.
        geometryRegularization: GeometryRegularization? = AppPreset.geometryRegularization,
        onThermalThrottle: ((ThermalThrottleState) -> Void)? = nil,
        onSnapshot: ((GaussianModel, Int) -> Void)? = nil,
        log: (String) -> Void = { print($0) }
    ) throws -> TrainingDriver.Result {
        // [timing] Diagnostic log of per-phase wall-clock before training.
        let _tPre0 = Date()
        var _tMark = _tPre0
        func phaseMs(_ name: String) {
            let now = Date()
            log(String(format: "[timing] %@ = %.2fs", name, now.timeIntervalSince(_tMark)))
            _tMark = now
        }
        let workingScene = geometry.scene
        let background = MLXArray([Float(0), 0, 0], [3])
        let sceneScale = geometry.sceneScale
        let (trainingFrames, heldOutFrames) = selectTrainingFrames(
            workingScene.frames,
            cap: trainingViewCap ?? recommendedTrainingViewCap(frameCount: geometry.capturedFrameCount),
            sceneScale: sceneScale,
            holdoutEvery: evalHoldoutEvery, holdoutBeforeSelection: evalHoldoutBeforeViewSelection,
            log: log)

        // seed = confidence-weighted fusion of LiDAR + MVS dense. Colors are initialized from the
        // seed and converge via sh0 gradients. The seed point cloud (millions of points with MVS) is
        // only used for init, so take it inside the closure and release it before training (kept
        // during training it stays on the GPU: 2026-10-09 4.8M MVS points, +90MB).
        let model: GaussianModel = {
            let mvsCloud = geometry.takeSeedCloud()
            // MVS is already deduplicated at voxelDedupSize (3mm). If the fusion voxel (1cm) is
            // coarser, fusing with other points regroups MVS points onto the 1cm grid and thins
            // them to about 1/9 (2026-10-08 on device: 328k → 36k), so match the fusion voxel to
            // the MVS dedup width.
            let fusionVoxelSize = Swift.min(
                seedFusionVoxelSize, geometry.mvsVoxelSize ?? seedFusionVoxelSize)
            let (positions, colors) = captureLiDAROrSyntheticSeed(
                scene: workingScene, sceneScale: sceneScale,
                syntheticSparseCount: syntheticSparseCount, seed: seed,
                extraSeedClouds: mvsCloud.map { [$0] } ?? [],
                fusionVoxelSize: fusionVoxelSize,
                log: log
            )
            eval(positions, colors)
            phaseMs("seed-build(MVS/LiDAR fusion)")

            return initialModel(
                positions: positions, colors: colors, maxInitGaussians: maxInitGaussians, log: log)
        }()
        if let seedInspectionPLY {
            do {
                let n = try writeSeedInspectionPLY(
                    model: model, frames: workingScene.frames,
                    sceneScale: sceneScale, to: seedInspectionPLY)
                log("wrote seed inspection PLY (\(n) gaussians: seed=initial colors/cameras=red) "
                    + "to \(seedInspectionPLY.lastPathComponent)")
            } catch {
                log("[warn] seed inspection PLY write failed: \(error)")
            }
        }
        log(memorySnapshot("post-init"))
        phaseMs("init(OperatorI disc + model)")
        // Built last so the view copies are not resident alongside the seed build.
        var views = makeViews(
            frames: trainingFrames, tileSize: tileSize, background: background,
            weightByTimestamp: geometry.weightByTimestamp
        )
        phaseMs("makeViews(decode+upload \(trainingFrames.count) frames)")
        if geometryRegularization != nil {
            // Each training view's MVS depth map (same frame, so the same pose).
            do {
                var byPose: [[Float]: CostVolumeMVS.DepthView] = [:]
                for map in try geometry.depthViews() { byPose[map.viewmat] = map }
                for i in views.indices {
                    guard let map = byPose[trainingFrames[i].viewmat] else { continue }
                    views[i].mvsDepth = DepthTarget(
                        width: map.width, height: map.height, depth: map.depth, intrinsics: map.intrinsics)
                }
            } catch {
                log("[warn] geometry regularization without MVS depth: \(error)")
            }
        }
        log(String(format: "[timing] === pre-training total = %.2fs (training starts) ===",
                   Date().timeIntervalSince(_tPre0)))
        MLX.Memory.clearCache()
        log(memorySnapshot("pre-training"))
        let result = try TrainingDriver.runTrainingLoop(
            model: model, views: views, sceneScale: sceneScale,
            iterations: iterations, meansLR: meansLR, logEvery: logEvery,
            seed: seed, strategy: strategy,
            opacityReg: opacityReg, scaleReg: scaleReg, lambdaSSIM: lambdaSSIM,
            replayCache: .on, replayKMax: replayKMax,
            outputPLY: outputPLY, thumbnailPNG: thumbnailPNG,
            enableBackgroundSH: true,
            backgroundSHDegree: backgroundSHDegree,
            alphaLossWeight: alphaLossWeight,
            coarseToFineFactor: coarseToFineFactor,
            coarseToFineFraction: coarseToFineFraction,
            fineResample: fineResample,
            // Bring the SH degree warm-up forward so degree 1 takes effect from about 25% of
            // training.
            useEarlySH: true,
            geometryRegularization: geometryRegularization,
            onThermalThrottle: onThermalThrottle,
            onSnapshot: onSnapshot, log: log
        )
        if !heldOutFrames.isEmpty, let trainer = result.session?.trainer {
            let heldOutViews = makeViews(frames: heldOutFrames, tileSize: tileSize, background: background)
            log("[eval-holdout] held-out views:")
            _ = try TrainingDriver.evaluate(views: heldOutViews, log: log) {
                trainer.renderForEval(camera: $0.camera)
            }
            log("[eval-holdout] training views:")
            _ = try TrainingDriver.evaluate(views: views, log: log) {
                trainer.renderForEval(camera: $0.camera)
            }
            if evalAgainstCapturedImages {
                log("[eval-holdout] held-out views vs captured images (2× render → capture pixel grid, margin 8px):")
                try evaluateAgainstCapturedImages(
                    frames: heldOutFrames, trainer: trainer, tileSize: tileSize, background: background, log: log)
            }
        }
        return result
    }

    /// MVS dense seed of the app's configuration (the app always trains in this mode; the rest is
    /// `AppPreset`). Agreement checks reduce points, so all frames + 3mm voxel keep the point count
    /// up.
    public enum MVSMode {
        /// cost mode: multi-view cost (m=5) + agreement in 3+ views + depth averaged over agreeing
        /// views, 3mm voxel. Agreeing views 2→3 (2026-10-08): on poster, initial-point plane offset
        /// 8.9→7.3mm, >10cm outliers 0.5→0.1%, floating opaque Gaussians (>10cm) after training
        /// 29→16, user visual check "low noise, very good". In exchange desk held-out PSNR went
        /// 21.43→20.78 dB (3 runs each).
        /// References prefer covisibility ≥ 0.6 (2026-10-09): selecting by camera position alone
        /// mixes in refs that do not see the pixels, splitting walls with gaps between them
        /// (Jonathan: the wall split into 3 in both the capture guide and the final result).
        public static var mvsParameters: CostVolumeMVS.Parameters {
            .mobile3DRecon(
                m: 5, voxelDedupSize: 0.003,
                fusionConsistency: CostVolumeMVS.FusionConsistency(vMin: 3),
                refSelection: ReferenceSelection.Parameters(minCovisibility: 0.6))
        }
    }

    /// Target long side (px) of the MVS input. With `mvsDownscale == nil`, box-downscale by the
    /// smallest integer factor making the long side ≤ this (1080px → 1/4). 1/4 beats full size and
    /// 1/2 in both plane error and multi-view agreement rate.
    public static let mvsTargetLongSide = 320
}
