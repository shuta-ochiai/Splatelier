// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX

/// End-to-end training driver. Keeps MLX usage inside the library; the CLI only parses arguments.
/// Uses the GPU, so it runs only in builds with MLX's metallib (xcodebuild), not under `swift run`.
public enum TrainingDriver {

    /// Reader returning cumulative GPU energy (J) since launch. When set, the training profiler
    /// reports GPU energy per iter and average power for each window (heat = energy, so thermal
    /// measures are compared on this, not on speed). Set by the Mac CLI via IOReport. Unreadable on
    /// iOS, so it stays nil there.
    public nonisolated(unsafe) static var gpuEnergyReader: (() -> Double?)?

    /// Override for the thermalState seen by the thermal guard. For A/B runs that reproduce iOS's
    /// serious-state slowdown on the Mac to measure GPU energy (CLI THERMAL_SIM). nil uses the
    /// actual value.
    public nonisolated(unsafe) static var simulatedThermalState: ProcessInfo.ThermalState?

    public struct Result {
        public let gaussianCount: Int
        public let viewCount: Int
        public let lossHistory: [Float]
        /// Resumable session for continue training (+N iters). Always set by `runTrainingLoop` /
        /// `continueTraining`.
        public var session: ResumableTrainingSession? = nil

        public init(gaussianCount: Int, viewCount: Int, lossHistory: [Float],
                    session: ResumableTrainingSession? = nil) {
            self.gaussianCount = gaussianCount
            self.viewCount = viewCount
            self.lossHistory = lossHistory
            self.session = session
        }
    }

    /// State held for continue training (in memory only). Holds the `GaussianTrainer` (model + Adam
    /// m/v/stepCount) and the views. Training and continuation run only on the same serial queue by
    /// contract, hence `@unchecked Sendable` (so the reference can be passed to main).
    public final class ResumableTrainingSession: @unchecked Sendable {
        public let trainer: GaussianTrainer
        public let views: [TrainView]
        public let viewSampler: ViewSampler
        public let sceneScale: Float
        public let maxShDegree: Int
        /// The geometric regularization of the run (nil: none); continue training keeps it.
        public let geometryRegularization: GeometryRegularization?
        /// Cumulative iteration count (initial run + continuations).
        public internal(set) var totalIterations: Int

        init(trainer: GaussianTrainer, views: [TrainView], viewSampler: ViewSampler,
             sceneScale: Float, maxShDegree: Int, geometryRegularization: GeometryRegularization?,
             totalIterations: Int) {
            self.trainer = trainer
            self.views = views
            self.viewSampler = viewSampler
            self.sceneScale = sceneScale
            self.maxShDegree = maxShDegree
            self.geometryRegularization = geometryRegularization
            self.totalIterations = totalIterations
        }
    }

    /// gsplat-style scene scale (max distance from the centroid of camera centers × 1.1). Makes the
    /// means LR and MCMC noise proportional to scene size.
    public static func sceneScale(viewmats: [[Float]]) -> Float {
        guard !viewmats.isEmpty else { return 1 }
        let centers = viewmats.map { FrameSubsampler.cameraCenter(viewmat: $0) }
        let centroid = centers.reduce(SIMD3<Float>(repeating: 0), +) / Float(centers.count)
        var maxDist: Float = 0
        for c in centers {
            let d = c - centroid
            maxDist = Swift.max(maxDist, (d.x * d.x + d.y * d.y + d.z * d.z).squareRoot())
        }
        return Swift.max(maxDist * 1.1, 1e-3)
    }

    static func sceneScale(_ cameras: [ColmapCamera]) -> Float {
        sceneScale(viewmats: cameras.map(\.viewmat))
    }

    /// Initialization scheme from the point cloud. `.pocketGSDisc` is PocketGS Operator I
    /// (PocketGS, arXiv 2601.17354, Sec. III-B).
    public enum InitKind: Sendable, Equatable {
        /// gsplat default: 3-NN RMS distance broadcast to all axes (isotropic).
        case gsplatIsotropic
        /// PocketGS Operator I: normal from K_n-neighbor PCA, tangent scale from the K_s-neighbor
        /// mean, disc-like (s_t, s_t, r_normal·s_t). The default `initScale` / `scaleCap` (0.1 /
        /// 0.05) target sparse input. Following paper eq. 6 exactly is 1.0 / .infinity (the MVS
        /// seed uses that).
        case pocketGSDisc(
            kN: Int = 16, kS: Int = 3, rNormal: Float = 0.3,
            initScale: Float = 0.1, scaleCap: Float = 0.05
        )
    }

    /// Builds a dense point cloud with MVS (CostVolumeMVS / PocketGS Operator G) (displacement gate
    /// → scene center → build). nil if no points come out. `depthMaps` receives the per-frame
    /// depth maps behind the points (for meshing, see `CostVolumeMVS.build`).
    public static func runGeometryPrior(
        viewmats: [[Float]],
        intrinsics: [Intrinsics],
        images: [MLXArray],
        sparsePoints: [SIMD3<Float>],
        parameters: CostVolumeMVS.Parameters,
        // Uniform value given to MVS points in Operator G's confidence-weighted fusion. Already
        // filtered in CostVolumeMVS, so a middling value.
        mvsConfidence: Float = 0.6,
        depthMaps: ((CostVolumeMVS.DepthView) -> Void)? = nil,
        log: (String) -> Void = { print($0) }
    ) -> (positions: MLXArray, colors: MLXArray, confidences: MLXArray)? {
        precondition(viewmats.count == intrinsics.count
                     && intrinsics.count == images.count,
                     "viewmats / intrinsics / images counts must match")

        // 1. FrameSubsampler: drop redundant (low-parallax) viewpoints. Blurred frames were already
        // dropped by the intake quality filter (CaptureFrameQualityFilter).
        let keptIndices = FrameSubsampler.selectFrames(viewmats: viewmats)
        log("MVS frame subsampling kept \(keptIndices.count)/\(viewmats.count) frames")
        let mvsViewmats = keptIndices.map { viewmats[$0] }
        let mvsIntrTuples = keptIndices.map { intrinsics[$0] }
        let mvsImages = keptIndices.map { images[$0] }

        // 2. scene center = centroid of camera centers.
        var cx: Float = 0, cy: Float = 0, cz: Float = 0
        for vm in mvsViewmats {
            let c = FrameSubsampler.cameraCenter(viewmat: vm)
            cx += c.x; cy += c.y; cz += c.z
        }
        let invN = 1.0 / Float(max(mvsViewmats.count, 1))
        let sceneCenter = SIMD3<Float>(cx * invN, cy * invN, cz * invN)

        // 3. CostVolumeMVS
        log("running MVS geometry prior (CostVolumeMVS) over \(mvsViewmats.count) views...")
        func mvsLog(_ msg: String) { log(msg) }
        let mvs = CostVolumeMVS.build(
            inputs: CostVolumeMVS.Inputs(
                viewmats: mvsViewmats, intrinsics: mvsIntrTuples,
                rgbImages: mvsImages, sparsePoints: sparsePoints,
                sceneCenter: sceneCenter
            ),
            parameters: parameters,
            depthMaps: depthMaps,
            log: mvsLog
        )
        let denseCount = mvs.positions.shape[0]
        let frameCounts = mvs.perFramePointCount
        let positiveCounts = frameCounts.filter { $0 > 0 }
        let pcMin = positiveCounts.min() ?? 0
        let pcMax = positiveCounts.max() ?? 0
        let pcMean = positiveCounts.isEmpty ? 0 : positiveCounts.reduce(0, +) / positiveCounts.count
        log("CostVolumeMVS produced \(denseCount) points (post-fusion); "
            + "per-frame min=\(pcMin) mean=\(pcMean) max=\(pcMax), "
            + "zero-frames=\(frameCounts.count - positiveCounts.count)")
        guard denseCount > 0 else { return nil }
        let confidences = MLXArray([Float](repeating: mvsConfidence, count: denseCount), [denseCount])
        eval(confidences)
        return (positions: mvs.positions, colors: mvs.colors, confidences: confidences)
    }

    /// Reads a COLMAP scene, initializes from its point cloud, and trains for `iterations`
    /// iterations.
    public static func trainColmap(
        sceneDirectory: URL,
        iterations: Int = 500,
        maxImageDimension: Int = 512,
        maxPoints: Int = 30_000,
        tileSize: Int = 16,
        meansLR: Float = 1.6e-4,
        logEvery: Int = 50,
        seed: UInt64 = 0,
        strategy: (any DensificationStrategy)? = nil,
        opacityReg: Float = 0,
        scaleReg: Float = 0,
        lambdaSSIM: Float = 0.2,
        initOpacity: Float = 0.5,
        initScale: Float = 0.1,
        initKind: InitKind = .gsplatIsotropic,
        // On by default to favor thermal sustainability (same as the App). Off uses the faster coop
        // backward but runs hotter.
        useReplayCache: Bool = true,
        replayKMax: Int = 64,
        outputPLY: URL? = nil,
        onSnapshot: ((GaussianModel, Int) -> Void)? = nil,
        log: (String) -> Void = { print($0) }
    ) throws -> Result {
        let scene = try ColmapLoader.load(directory: sceneDirectory)
        log("loaded \(scene.cameras.count) cameras, \(scene.pointPositions.shape[0]) sparse points")

        let background = MLXArray([Float(0), 0, 0], [3])
        let imagesDir = sceneDirectory.appendingPathComponent("images")
        let views = ColmapTraining.makeViews(
            scene: scene, imagesDirectory: imagesDir,
            maxDimension: maxImageDimension, tileSize: tileSize, background: background
        )
        log("built \(views.count) training views (max dim \(maxImageDimension))")
        guard !views.isEmpty else {
            throw ColmapLoader.Error.parse("no usable training views (missing images in \(imagesDir.path)?)")
        }

        let primaryPositions = scene.pointPositions
        let primaryColors = scene.pointColors

        // Details of the disc path: PocketGS Sec. III-B 1-2 and the fromPointCloudDisc doc.
        let model: GaussianModel
        switch initKind {
        case .gsplatIsotropic:
            model = GaussianModel.fromPointCloud(
                positions: primaryPositions, colors: primaryColors,
                initOpacity: initOpacity, initScale: initScale, maxPoints: maxPoints
            )
            log("initialised \(model.count) gaussians from point cloud "
                + "(gsplat isotropic; init opacity \(initOpacity), init scale ×\(initScale))")
        case .pocketGSDisc(let kN, let kS, let rNormal, let discInitScale, let discScaleCap):
            // Use the paper's default opacity 0.1 (initOpacity is for the isotropic path).
            model = GaussianModel.fromPointCloudDisc(
                positions: primaryPositions, colors: primaryColors,
                kN: kN, kS: kS, rNormal: rNormal,
                initOpacity: 0.1,
                initScale: discInitScale, scaleCap: discScaleCap,
                maxPoints: maxPoints
            )
            log("initialised \(model.count) gaussians from point cloud "
                + "(PocketGS disc-like; K_n=\(kN), K_s=\(kS), r_normal=\(rNormal), "
                + "initScale=\(discInitScale), scaleCap=\(discScaleCap))")
        }

        let sceneScale = Self.sceneScale(scene.cameras)
        return try runTrainingLoop(
            model: model, views: views, sceneScale: sceneScale,
            iterations: iterations, meansLR: meansLR, logEvery: logEvery,
            seed: seed, strategy: strategy,
            opacityReg: opacityReg, scaleReg: scaleReg, lambdaSSIM: lambdaSSIM,
            replayCache: useReplayCache ? .on : .off, replayKMax: replayKMax,
            outputPLY: outputPLY, onSnapshot: onSnapshot, log: log
        )
    }
}
