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
    /// dense MVS seed, 4×frames is the knee of quality vs energy, and the cap
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
    /// 240), so a longer capture keeps the same share of views as a 300-frame capture. Views beyond
    /// the cap are thinned by viewpoint coverage (FPS).
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
        weightByTimestamp: [TimeInterval: Float]? = nil,
        surfaceMask: Bool = false,
        // Each frame's measured cells (`mvsSurfaceCells` / `sparseSurfaceCells`, combined) by
        // viewmat (LiDAR alone for the frames without).
        surfaceCells: [[Float]: [Float]] = [:],
        // Count the LiDAR samples as measured surface besides MVS.
        lidarSurface: Bool = true
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
            var view = TrainView(
                camera: camera, image: image,
                name: String(format: "capture_%06d.png", idx),
                weight: weight
            )
            if surfaceMask, let masks = surfaceMasks(frame, cells: surfaceCells[frame.viewmat], lidar: lidarSurface) {
                view.surfaceMask = masks.surface
                view.opaqueMask = masks.opaque
            }
            views.append(view)
        }
        return views
    }

    /// Cells of the surface mask across the image: the grid of the recorded LiDAR samples (every
    /// 8th of the 256×192 depth pixels), so each sample marks its own cell.
    static let surfaceMaskGrid = (rows: 24, cols: 32)

    /// [rows, cols] grids of the frame's measured surface: `surface` has 1 in the cells its LiDAR
    /// samples project to or `cells` marks (MVS / sparse), `opaque` in those of them whose 8
    /// neighbors are marked too (the image border counts as marked). nil with neither. LiDAR
    /// samples only exist where the sensor returned a surface within range, MVS depth where views
    /// agreed on one and sparse points where features were tracked, so the cells left 0 are where
    /// the background may show (sky, far walls, windows). Leaving out
    /// the edge cells keeps a silhouette from being pulled over the background beside it, which a
    /// cell this coarse (1/32 of the width) would partly cover.
    static func surfaceMasks(
        _ frame: CapturedFrame, cells: [Float]?, lidar: Bool = true
    ) -> (surface: MLXArray, opaque: MLXArray)? {
        let lidarSamples = lidar ? frame.lidarSamples : []
        guard !lidarSamples.isEmpty || cells != nil else { return nil }
        let (rows, cols) = surfaceMaskGrid
        var grid = cells ?? [Float](repeating: 0, count: rows * cols)
        let k = frame.intrinsics
        for s in lidarSamples {
            let p = s.cameraPoint
            guard p.z > 0 else { continue }
            let u = (k.fx * p.x / p.z + k.cx) / Float(frame.width)
            let v = (k.fy * p.y / p.z + k.cy) / Float(frame.height)
            guard u >= 0, u < 1, v >= 0, v < 1 else { continue }
            grid[Int(v * Float(rows)) * cols + Int(u * Float(cols))] = 1
        }
        var opaque = grid
        for r in 0..<rows {
            for c in 0..<cols where grid[r * cols + c] > 0 {
                for nr in Swift.max(0, r - 1)...Swift.min(rows - 1, r + 1) {
                    for nc in Swift.max(0, c - 1)...Swift.min(cols - 1, c + 1) where grid[nr * cols + nc] == 0 {
                        opaque[r * cols + c] = 0
                    }
                }
            }
        }
        return (MLXArray(grid, [rows, cols]), MLXArray(opaque, [rows, cols]))
    }

    /// [rows·cols] cells over `frame` (`surfaceMaskGrid`) with 1 where at least `minPoints` of
    /// `points` (world) land in front of the camera. A point anywhere along a cell's rays means a
    /// surface at or before it, so occlusion does not matter. The count keeps a stray point in the
    /// sky from marking its cell.
    static func sparseSurfaceCells(_ points: [SIMD3<Float>], frame: CapturedFrame, minPoints: Int = 3) -> [Float] {
        let (rows, cols) = surfaceMaskGrid
        let k = frame.intrinsics, vm = frame.viewmat
        let r0 = SIMD4(vm[0], vm[1], vm[2], vm[3])
        let r1 = SIMD4(vm[4], vm[5], vm[6], vm[7])
        let r2 = SIMD4(vm[8], vm[9], vm[10], vm[11])
        // Intrinsics in cells rather than pixels.
        let sx = Float(cols) / Float(frame.width), sy = Float(rows) / Float(frame.height)
        let fx = k.fx * sx, cx = k.cx * sx, fy = k.fy * sy, cy = k.cy * sy
        let colsF = Float(cols), rowsF = Float(rows)
        var counts = [Int32](repeating: 0, count: rows * cols)
        counts.withUnsafeMutableBufferPointer { counts in
            for p in points {
                let h = SIMD4(p, 1)
                let z = simd_dot(r2, h)
                guard z > 0 else { continue }
                let u = fx * simd_dot(r0, h) / z + cx
                let v = fy * simd_dot(r1, h) / z + cy
                guard u >= 0, u < colsF, v >= 0, v < rowsF else { continue }
                counts[Int(v) * cols + Int(u)] += 1
            }
        }
        return counts.map { $0 >= minPoints ? 1 : 0 }
    }

    /// [rows·cols] cells over `frame` (`surfaceMaskGrid`) with 1 where at least half of a 4×4
    /// sampling of its MVS depth map (same pose, own resolution and intrinsics) holds a depth.
    static func mvsSurfaceCells(_ map: CostVolumeMVS.DepthView, frame: CapturedFrame) -> [Float] {
        let (rows, cols) = surfaceMaskGrid
        let k = frame.intrinsics, ti = map.intrinsics
        let sub = 4
        var cells = [Float](repeating: 0, count: rows * cols)
        for r in 0..<rows {
            for c in 0..<cols {
                var hits = 0
                for i in 0..<sub {
                    let py = (Float(r) + (Float(i) + 0.5) / Float(sub)) / Float(rows) * Float(frame.height)
                    let v = ((py - k.cy) / k.fy * ti.fy + ti.cy).rounded(.down)
                    guard v >= 0, v < Float(map.height) else { continue }
                    for j in 0..<sub {
                        let px = (Float(c) + (Float(j) + 0.5) / Float(sub)) / Float(cols) * Float(frame.width)
                        let u = ((px - k.cx) / k.fx * ti.fx + ti.cx).rounded(.down)
                        guard u >= 0, u < Float(map.width) else { continue }
                        if map.depth[Int(v) * map.width + Int(u)] > 0 { hits += 1 }
                    }
                }
                if hits * 2 >= sub * sub { cells[r * cols + c] = 1 }
            }
        }
        return cells
    }

    /// Mean rendered foreground alpha over the pixels of a measured surface (`surfaceMask`), and the
    /// share of them below 0.5 (holes once the viewer hides the background): a wall taken over by
    /// the background shows as alpha near 0 there although the image still matches.
    static func logSurfaceAlpha(views: [TrainView], trainer: GaussianTrainer, label: String, log: (String) -> Void) {
        var sum: Float = 0, holes: Float = 0, count: Float = 0
        for v in views {
            guard let mask = v.surfaceMask else { continue }
            let alpha = trainer.renderer.forward(model: trainer.model, camera: v.camera, shDegree: 3).renderAlpha
                .reshaped([v.camera.height, v.camera.width])
            let m = GaussianTrainer.upsampleNearest(mask, height: v.camera.height, width: v.camera.width)
            sum += (alpha * m).sum().item(Float.self)
            holes += ((alpha .< Float(0.5)).asType(.float32) * m).sum().item(Float.self)
            count += m.sum().item(Float.self)
        }
        guard count > 0 else { return }
        log(String(format: "[eval-holdout] %@: on measured surface pixels mean foreground alpha = %.3f, alpha < 0.5 = %.1f%%",
                   label, sum / count, 100 * holes / count))
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
        // How MVS and LiDAR share the initial Gaussians (`SeedAllocation`).
        seedAllocation: SeedAllocation = .even,
        // Color the LiDAR seed points from the frame images (mid-grey otherwise). Worth +0.8 to
        // +1.3 dB held-out where LiDAR is a large share of the seed.
        lidarColors: Bool = true,
        // Keep the pixels where LiDAR or MVS measured a surface out of the bg-SH background mask,
        // so a plain wall the background color can imitate stays foreground (LiDAR: +0.5 dB
        // held-out on a LiDAR capture; MVS alone for the same purpose gained nothing).
        surfaceMask: Bool = true,
        // Count MVS depth as a measured surface besides LiDAR (`surfaceMask`).
        mvsSurfaceMask: Bool = true,
        // Count LiDAR samples as a measured surface besides MVS (`surfaceMask`).
        lidarSurfaceMask: Bool = true,
        // Count the sparse points (BA and ARKit features) as a measured surface besides MVS
        // (`surfaceMask`): they also land on repeating or plain-ish texture (brick, floor tiles)
        // where the MVS views do not agree.
        sparseSurfaceMask: Bool = true,
        // Seed the BA points where MVS / LiDAR left a gap, so the foreground has Gaussians on the
        // surfaces the sparse cells mark (`sparseSurfaceMask`) (+0.45 dB held-out on one
        // MVS-only capture with the surface pull, the same on another).
        sparseSeed: Bool = true,
        // Weight of the loss pulling foreground alpha to 1 inside the measured surface (away from
        // its edges), so hiding the background in the viewer leaves no holes there. 0 disables it.
        surfaceAlphaWeight: Float = 0.15,
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
        onPreview: ((TrainingPreviewFrame) -> Void)? = nil,
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
        // during training it stays on the GPU).
        let model: GaussianModel = {
            let mvsCloud = geometry.takeSeedCloud()
            // MVS is already deduplicated at voxelDedupSize (3mm). If the fusion voxel (1cm) is
            // coarser, fusing with other points regroups MVS points onto the 1cm grid and thins
            // them out heavily, so match the fusion voxel to the MVS dedup width.
            let fusionVoxelSize = Swift.min(
                seedFusionVoxelSize, geometry.mvsVoxelSize ?? seedFusionVoxelSize)
            let (positions, colors) = captureLiDAROrSyntheticSeed(
                scene: workingScene, sceneScale: sceneScale,
                syntheticSparseCount: syntheticSparseCount, seed: seed,
                extraSeedClouds: mvsCloud.map { [$0] } ?? [],
                fusionVoxelSize: fusionVoxelSize,
                allocation: seedAllocation, maxPoints: maxInitGaussians,
                lidarColors: lidarColors,
                sparseGapFill: sparseSeed,
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
        // Each frame's MVS depth map (same frame, so the same pose), for the geometric
        // regularization and the surface masks. The maps are let go before training; the
        // held-out views keep only their mask cells.
        var surfaceCells: [[Float]: [Float]] = [:]
        var views: [TrainView]
        do {
            var byPose: [[Float]: CostVolumeMVS.DepthView] = [:]
            if geometryRegularization != nil || surfaceMask || !heldOutFrames.isEmpty {
                do {
                    for map in try geometry.depthViews() { byPose[map.viewmat] = map }
                } catch {
                    log("[warn] no MVS depth for the geometric regularization / surface mask: \(error)")
                }
            }
            let sparsePoints = farClipped(workingScene.sparsePoints, frames: workingScene.frames, sceneScale: sceneScale)
            // The held-out views always get every source, so their surface alpha compares across
            // settings.
            let maskFrames = trainingFrames.map { ($0, true) } + heldOutFrames.map { ($0, false) }
            for (frame, training) in maskFrames {
                var cells: [Float]?
                if let map = byPose[frame.viewmat], !training || (surfaceMask && mvsSurfaceMask) {
                    cells = mvsSurfaceCells(map, frame: frame)
                }
                if !sparsePoints.isEmpty, !training || (surfaceMask && sparseSurfaceMask) {
                    let sparse = sparseSurfaceCells(sparsePoints, frame: frame)
                    cells = cells.map { zip($0, sparse).map(Swift.max) } ?? sparse
                }
                surfaceCells[frame.viewmat] = cells
            }
            // Built last so the view copies are not resident alongside the seed build.
            views = makeViews(
                frames: trainingFrames, tileSize: tileSize, background: background,
                weightByTimestamp: geometry.weightByTimestamp, surfaceMask: surfaceMask,
                surfaceCells: surfaceCells, lidarSurface: lidarSurfaceMask
            )
            phaseMs("makeViews(decode+upload \(trainingFrames.count) frames)")
            if geometryRegularization != nil {
                for i in views.indices {
                    guard let map = byPose[trainingFrames[i].viewmat] else { continue }
                    views[i].mvsDepth = DepthTarget(
                        width: map.width, height: map.height, depth: map.depth, intrinsics: map.intrinsics)
                }
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
            surfaceAlphaWeight: surfaceMask ? surfaceAlphaWeight : 0,
            coarseToFineFactor: coarseToFineFactor,
            coarseToFineFraction: coarseToFineFraction,
            fineResample: fineResample,
            // Bring the SH degree warm-up forward so degree 1 takes effect from about 25% of
            // training.
            useEarlySH: true,
            geometryRegularization: geometryRegularization,
            onThermalThrottle: onThermalThrottle,
            onPreview: onPreview,
            onSnapshot: onSnapshot, log: log
        )
        if !heldOutFrames.isEmpty, let trainer = result.session?.trainer {
            let heldOutViews = makeViews(frames: heldOutFrames, tileSize: tileSize, background: background,
                                         surfaceMask: true, surfaceCells: surfaceCells)
            log("[eval-holdout] held-out views:")
            _ = try TrainingDriver.evaluate(views: heldOutViews, log: log) {
                trainer.renderForEval(camera: $0.camera)
            }
            logSurfaceAlpha(views: heldOutViews, trainer: trainer, label: "held-out views", log: log)
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
        /// views, 3mm voxel. Requiring 3 agreeing views instead of 2 reduces the initial points'
        /// offset from planes, far outliers and floating opaque Gaussians after training, at a
        /// small cost in held-out PSNR.
        /// References prefer covisibility ≥ 0.6: selecting by camera position alone mixes in refs
        /// that do not see the pixels, splitting walls with gaps between them.
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
