// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import Trainer
import Capture

func usage() {
    print("splatelier v\(Splatelier.version)")
    print("")
    print("usage:")
    print("  splatelier train <colmap_scene_dir> <output.ply> [iterations] [maxDim] [maxPoints] [capMax] [opacityReg] [scaleReg] [init] [replayCache] [geometryPrior] [densification]")
    print("")
    print("  Trains with MCMC densification (relocate dead gaussians, grow toward")
    print("  capMax, perturb). maxPoints is the initial sub-sample; capMax (default")
    print("  1,000,000) is the population ceiling.")
    print("")
    print("  [init] is the point-cloud initialiser (default `iso`):")
    print("    iso  — gsplat-style isotropic 3-NN RMS scale + random quat")
    print("    disc — PocketGS Operator I: K_n=16 PCA normal + K_s=3 tangential")
    print("           scale + disc-like anisotropic seed (ref/pocketGS.pdf Sec. III-B)")
    print("  [replayCache] enables PocketGS Operator T (default `false`):")
    print("    false / off / no  — existing T-reconstruction backward")
    print("    true  / on  / yes — forward replay cache S = {id, C_in, α} + paper")
    print("                        eq. 8 backward (ref/pocketGS.pdf Sec. III-C)")
    print("  [geometryPrior] selects the input point cloud (default `colmap`):")
    print("    colmap   — COLMAP sparse cloud (only option)")
    print("  [densification] picks the growth strategy (default `mcmc`):")
    print("    mcmc     — existing gsplat-style MCMC: relocate + sampleAdd + noise")
    print("               (aggressive growth toward capMax)")
    print("    mcmc-fixed — no relocate/growth; prune dead Gaussians in-place")
    print("                 (useful with PocketGS-style dense init)")
    print("    igsplus  — ImprovedGS+ style Long-Axis-Split (LAS) growth")
    print("               (deterministic split of large visible Gaussians)")
    print("    none     — no MCMC strategy at all (raw gradient descent)")
    print("")
    print("  splatelier train-capture <dataset.valset> <output.ply> [iterations] [densification] [lambdaSSIM]")
    print("    Loads an iOS app \"Validation datasets\" entry (.valset directory = captured frames + poses + points)")
    print("    and retrains it on the Mac with the app's configuration. Pass the directory extracted from an")
    print("    AirDropped .valset.zip. iterations defaults to the app's recommendation for the frame count;")
    print("    densification defaults to the app's Hybrid (or one of train's: mcmc | mcmc-fixed | igsplus |")
    print("    hybrid | none).")
    print("    env: UNDISTORT_MIN_TRACK=<n> (min observations for points used in distortion estimation, default 3)")
    print("         EVAL_RAW=1 (with EVAL_HOLDOUT: also evaluates held-out views on the raw captured images)")
    print("         EVAL_HOLDOUT_ALL=1 (with EVAL_HOLDOUT: holds out from all frames before VIEW_CAP thinning)")
    print("         VIEW_CAP=<n> (training view cap; default 0.4 view/frame, at least 120, as in the app;")
    print("           0 = all frames)")
    print("         THERMAL_SIM=nominal|fair|serious fixes the thermal guard's thermalState (reproduces the serious slow-down)")
    print("         GEO_REG=0 turns off the geometric regularization (on as in the app)")
    print("         MESH_OBJ=<path.obj> after training, builds the mesh as the app's Generate mesh: from the trained")
    print("           Gaussians' surface depth (from training's MVS depth with GEO_REG=0); MESH_VOXEL=<m> (default")
    print("           0.005) / MESH_TRUNC=<voxels> (default 4) / MESH_MIN_WEIGHT (default 3) /")
    print("           MESH_MIN_COMPONENT=<tris> (default 200) / MESH_MAX_VOXELS (default 12M)")
    print("         MESH_HYBRID=<path.obj> after training, the MVS mesh with holes filled from the trained Gaussians;")
    print("         MESH_GOF=<path.obj> after training, the level set of the trained Gaussians' opacity field;")
    print("           MESH_GS_TSDF=<path.obj> TSDF of the same Gaussian depth maps (GOF_* knobs in MeshCapture.swift)")
    print("         gpu energy= in the training log is the IOReport GPU energy (system-wide value)")
    print("")
    print("  splatelier gs-mesh <dataset.valset> <trained.ply>")
    print("    The MESH_OBJ (MVS depth) / MESH_HYBRID / MESH_GOF / MESH_GS_TSDF meshes from a PLY that")
    print("    train-capture wrote, without training (BA/MVS are redone from the valset).")
    print("    GS_DEPTH_PROFILE=gof,solid compares profiles.")
    print("")
    print("  splatelier mesh-eval <dataset.valset> <mesh.obj>...")
    print("    Compares meshes on the views train-capture held out (EVAL_HOLDOUT, default 8): coverage, PSNR of")
    print("    the vertex colors, agreement with the views' MVS depth, roughness. EVAL_SCALE, EVAL_IMAGES=<dir>.")
    print("")
    print("  splatelier dump-sparse <dataset.valset> <output.ply>")
    print("    Writes the .valset sparse points (ORB+BA refined) as a gaussian PLY for inspection in a viewer.")
    print("")
    print("  splatelier view <ply>")
    print("    Open a window and render a .ply (or .splat/.spz) with MetalSplatter.")
    print("    Drag to orbit, scroll to zoom; close the window to quit.")
    print("    env: VIEWER_TRANSPARENCY=sorted|stochastic|movement (default movement = stochastic alpha only while interacting)")
    print("")
    print("  splatelier train-view <colmap_scene_dir> [iterations] [maxDim] [maxPoints] [output.ply] [capMax] [init] [replayCache]")
    print("    Train with MCMC densification in a live window that updates as it")
    print("    optimizes. Same training as `train`; pass output.ply to also save the")
    print("    final model when training finishes. Accepts the same [init] and")
    print("    [replayCache] values as `train`.")
    print("")
    print("note: training uses the GPU via MLX, which needs the Metal library that")
    print("only an Xcode build produces. `swift run` will fail to find it; build the")
    print("executable with xcodebuild (or run the env-gated training test).")
}

let args = CommandLine.arguments
guard args.count >= 2 else { usage(); exit(0) }

// Positional-argument parser. Unknown tokens print an error and exit(2) so a typo doesn't silently
// fall through to the default path. Don't change the positional order (compatibility).
func parseInitKind(_ token: String) -> TrainingDriver.InitKind {
    switch token.lowercased() {
    case "iso", "isotropic", "gsplat": return .gsplatIsotropic
    case "disc", "pocketgs", "pocketgs-disc": return .pocketGSDisc()
    default:
        FileHandle.standardError.write(Data(
            "error: unknown [init] value '\(token)' (expected: iso | disc)\n".utf8))
        exit(2)
    }
}

func parseReplayCache(_ token: String) -> Bool {
    switch token.lowercased() {
    case "true", "on", "yes", "1", "replay", "pocketgs-t": return true
    case "false", "off", "no", "0": return false
    default:
        FileHandle.standardError.write(Data(
            "error: unknown [replayCache] value '\(token)' (expected: true | false)\n".utf8))
        exit(2)
    }
}

/// Value of `MVS_COVIS`: "off" etc. gives nil (pick references by camera position only); a number
/// is the lower bound on covisibility (0...1).
func mvsCovisibility(_ token: String) -> Float? {
    if ["0", "off", "no", "false"].contains(token.lowercased()) { return nil }
    guard let v = Float(token), v > 0, v <= 1 else {
        FileHandle.standardError.write(Data(
            "error: unknown MVS_COVIS value '\(token)' (expected: 0<v<=1 | off)\n".utf8))
        exit(2)
    }
    return v
}

/// `[geometryPrior]`. The input point cloud is COLMAP only, but the argument stays to keep the
/// order of later positional arguments.
func validateGeometryPrior(_ token: String) {
    guard token.lowercased() == "colmap" else {
        FileHandle.standardError.write(Data(
            "error: unknown [geometryPrior] value '\(token)' (expected: colmap)\n".utf8))
        exit(2)
    }
}

/// Resolves `[densification]` to a strategy. `capMax` is used only by the paths that densify.
func parseDensificationKind(
    _ token: String,
    capMax: Int,
    iterations: Int
) -> (any DensificationStrategy)? {
    switch token.lowercased() {
    case "mcmc": return MCMCStrategy.scaled(iterations: iterations, capMax: capMax)
    case "mcmc-fixed", "fixed", "fixed-count", "nogrowth", "no-growth", "pocketgs":
        return MCMCStrategy.fixedCount()
    case "igsplus", "igs+", "improvedgs+":
        return IGSPlusStrategy.scaled(iterations: iterations, capMax: capMax)
    case "hybrid":
        // Same as the app's Hybrid (IGS+ LAS growth + MCMC dead prune + cleanup of bloated blobs).
        var s = IGSPlusStrategy.scaled(iterations: iterations, capMax: capMax)
        s.pruneDeadGaussians = true
        s.bloatScaleMultiple = 100
        return s
    case "none", "off", "no", "0": return nil
    default:
        FileHandle.standardError.write(Data(
            "error: unknown [densification] value '\(token)' (expected: mcmc | mcmc-fixed | igsplus | none)\n".utf8))
        exit(2)
    }
}

// GPU energy meter (IOReport). The training profiler reports GPU energy per iter for each window.
#if os(macOS)
let gpuEnergyMeter = GPUEnergyMeter()
if let meter = gpuEnergyMeter {
    TrainingDriver.gpuEnergyReader = { meter.joules() }
}
#endif

// THERMAL_SIM=nominal|fair|serious: pins the thermalState seen by the thermal guard and reproduces
// on the Mac iOS's slowdown at serious (sleep between steps).
if let token = ProcessInfo.processInfo.environment["THERMAL_SIM"]?.lowercased() {
    switch token {
    case "nominal": TrainingDriver.simulatedThermalState = .nominal
    case "fair": TrainingDriver.simulatedThermalState = .fair
    case "serious": TrainingDriver.simulatedThermalState = .serious
    default:
        FileHandle.standardError.write(Data(
            "error: unknown THERMAL_SIM value '\(token)' (expected: nominal | fair | serious)\n".utf8))
        exit(2)
    }
}

switch args[1] {
case "train":
    guard args.count >= 4 else { usage(); exit(1) }
    let dir = URL(fileURLWithPath: args[2])
    let outputPLY = URL(fileURLWithPath: args[3])
    let iterations = args.count > 4 ? (Int(args[4]) ?? 500) : 500
    let maxDim = args.count > 5 ? (Int(args[5]) ?? 512) : 512
    let maxPoints = args.count > 6 ? (Int(args[6]) ?? 30_000) : 30_000
    let capMax = args.count > 7 ? (Int(args[7]) ?? 1_000_000) : 1_000_000
    let opacityRegArg: Float = args.count > 8 ? (Float(args[8]) ?? 0.01) : 0.01
    let scaleRegArg: Float = args.count > 9 ? (Float(args[9]) ?? 0.01) : 0.01
    let initKind = args.count > 10 ? parseInitKind(args[10]) : .gsplatIsotropic
    let useReplayCache = args.count > 11 ? parseReplayCache(args[11]) : false
    if args.count > 12 { validateGeometryPrior(args[12]) }
    // MCMCStrategy's raw defaults (start 500 / stop 25000) never fire within 500 iters, so
    // scale them so refinement runs over 10–80% of the horizon.
    let densificationStrategy: (any DensificationStrategy)? = args.count > 13
        ? parseDensificationKind(args[13], capMax: capMax, iterations: iterations)
        : MCMCStrategy.scaled(iterations: iterations, capMax: capMax)
    let initLabel: String = {
        switch initKind {
        case .gsplatIsotropic: return "iso"
        case .pocketGSDisc: return "disc"
        }
    }()
    let densLabel: String = {
        if densificationStrategy == nil { return "none" }
        if densificationStrategy is IGSPlusStrategy { return "igsplus" }
        // fixedCount is identified by growthFactor=1.0.
        if let m = densificationStrategy as? MCMCStrategy, m.growthFactor == 1.0 {
            return "mcmc-fixed"
        }
        return "mcmc"
    }()
    FileHandle.standardError.write(Data(
        ("config: init=\(initLabel) replayCache=\(useReplayCache) "
         + "densification=\(densLabel)\n").utf8))
    MLX.Memory.peakMemory = 0
    let memStart = MLX.Memory.snapshot()
    do {
        let result = try TrainingDriver.trainColmap(
            sceneDirectory: dir,
            iterations: iterations,
            maxImageDimension: maxDim,
            maxPoints: maxPoints,
            strategy: densificationStrategy,
            opacityReg: opacityRegArg,
            scaleReg: scaleRegArg,
            initKind: initKind,
            useReplayCache: useReplayCache,
            outputPLY: outputPLY
        )
        let finalLoss = result.lossHistory.last ?? -1
        let memEnd = MLX.Memory.snapshot()
        let mb = { (b: Int) in String(format: "%.1f", Double(b) / (1024 * 1024)) }
        print("done: \(result.gaussianCount) gaussians, \(result.viewCount) views, "
            + "final loss \(finalLoss)")
        print("GPU mem: peak=\(mb(memEnd.peakMemory))M  active=\(mb(memEnd.activeMemory))M  "
            + "cache=\(mb(memEnd.cacheMemory))M  (start active=\(mb(memStart.activeMemory))M)")
    } catch {
        print("training failed: \(error)")
        exit(1)
    }

case "train-capture":
    // Re-train an iOS validation dataset (.valset) on the Mac with the app's configuration
    // (CaptureTraining.AppPreset, train's defaults). The env variables are knobs for A/B.
    guard args.count >= 4 else {
        print("usage: splatelier train-capture <dataset.valset> <output.ply> "
              + "[iterations=app recommendation] [densification=hybrid] [lambdaSSIM=0.2]")
        exit(1)
    }
    let valsetURL = URL(fileURLWithPath: args[2])
    let outputPLY = URL(fileURLWithPath: args[3])
    let env = ProcessInfo.processInfo.environment
    let capMax = env["CAP_MAX"].flatMap { Int($0) } ?? CaptureTraining.AppPreset.defaultGaussianCap
    // <start>[,<end>] for MCMC_SCALE_LR (a single value means constant).
    func scaleLRPair(_ key: String) -> (start: Float, end: Float)? {
        guard let lr = env[key]?.split(separator: ",").compactMap({ Float($0) }),
              let start = lr.first else { return nil }
        return (start, lr.count > 1 ? lr[1] : start)
    }
    let lambdaSSIM: Float = args.count > 6 ? (Float(args[6]) ?? 0.2) : 0.2
    MLX.Memory.peakMemory = 0
    #if os(macOS)
    // GPU energy is system-wide. If other processes are using the GPU the A/B gets mixed, so check
    // first.
    if let idle = gpuEnergyMeter?.idleWatts() {
        FileHandle.standardError.write(Data(String(format: "[energy] GPU power before training %.2fW%@\n", idle,
            (idle > 1.0 ? " ⚠️ another process is using the GPU; it will be mixed into the energy values" : "") as NSString).utf8))
    }
    #endif
    do {
        FileHandle.standardError.write(Data("loading \(valsetURL.lastPathComponent)…\n".utf8))
        let scene = try CaptureDatasetStore.load(from: valsetURL)
        FileHandle.standardError.write(Data(
            "loaded \(scene.frames.count) frames, \(scene.sparsePoints.count) sparse points\n".utf8))
        // Default: the app's recommendation for the capture's frame count.
        let iterations = (args.count > 4 ? Int(args[4]) : nil)
            ?? CaptureTraining.recommendedIterations(frameCount: scene.frames.count, mvs: true)
        // [densification] (args[5]) picks one of the `train` strategies. Otherwise
        // DENSIFY=hybrid (the app's, default) | igsplus (the app's schedule without the dead prune) |
        // growth | relocate | prune (MCMC, for comparison).
        let densificationStrategy: (any DensificationStrategy)? = {
            if args.count > 5 {
                return parseDensificationKind(args[5], capMax: capMax, iterations: iterations)
            }
            let horizon = max(1, iterations)
            let mode = env["DENSIFY"]?.lowercased() ?? "hybrid"
            switch mode {
            case "igsplus", "igs+", "hybrid":
                var s = CaptureTraining.AppPreset.strategy(iterations: iterations, cap: capMax)
                if mode != "hybrid" {
                    s.pruneDeadGaussians = false
                    s.bloatScaleMultiple = 0
                }
                return s
            case "relocate", "prune":
                // Fixed count: no growth, dead Gaussians relocated or pruned.
                let start = max(50, horizon / 60)
                var s = MCMCStrategy.fixedCount(
                    refineStartIter: start,
                    refineStopIter: max(start + 1, (horizon * 4) / 5),
                    refineEvery: max(50, horizon / 60),
                    minOpacity: 0.005,
                    relocateDeadGaussians: mode == "relocate",
                    pruneDeadGaussians: mode == "prune",
                    seed: 0)
                // MCMC_SCALE_LR=<start>[,<end>]: give MCMC a decaying scale LR (for experiments).
                s.scaleLRSchedule = scaleLRPair("MCMC_SCALE_LR")
                return s
            default:
                // MCMC growth toward capMax.
                var s = MCMCStrategy(
                    capMax: capMax,
                    refineStartIter: max(50, horizon / 20),
                    refineStopIter: max(100, (horizon * 9) / 10),
                    refineEvery: max(10, horizon / 60),
                    minOpacity: 0.005)
                s.scaleLRSchedule = scaleLRPair("MCMC_SCALE_LR")
                return s
            }
        }()
        let logErr: (String) -> Void = { FileHandle.standardError.write(Data(($0 + "\n").utf8)) }
        let geometry = CaptureTraining.prepareGeometry(
            scene: scene,
            // UNDISTORT_MIN_TRACK=2 also uses points seen in only 2 views for distortion estimation
            // (default 3; for checking when it is rejected for lack of edge observations).
            lensDistortion: .init(minTrackLength: env["UNDISTORT_MIN_TRACK"].flatMap { Int($0) } ?? 3),
            mvsParameters: {
                var p = CaptureTraining.MVSMode.mvsParameters
                // MVS_FVMIN / MVS_M override the number of agreeing views / views in the multi-view
                // cost, MVS_DEDUP the voxel, MVS_CONF the confidence threshold and MVS_COVIS the lower
                // bound on reference covisibility (off disables).
                if env["MVS_FVMIN"] != nil || env["MVS_M"] != nil {
                    p = CostVolumeMVS.Parameters.mobile3DRecon(
                        m: env["MVS_M"].flatMap { Int($0) } ?? 5,
                        voxelDedupSize: p.voxelDedupSize,
                        fusionConsistency: CostVolumeMVS.FusionConsistency(
                            vMin: env["MVS_FVMIN"].flatMap { Int($0) } ?? 3),
                        refSelection: p.refSelection)
                }
                if let d = env["MVS_DEDUP"].flatMap({ Float($0) }) { p.voxelDedupSize = d }
                if let c = env["MVS_CONF"].flatMap({ Float($0) }) { p.confidenceThreshold = c }
                if let c = env["MVS_COVIS"] { p.refSelection.minCovisibility = mvsCovisibility(c) }
                return p
            }(),
            // MVS_DOWNSCALE=<s>: downscale MVS input to 1/s. Unset = auto (long side ~320px), 1 =
            // full size.
            mvsDownscale: env["MVS_DOWNSCALE"].flatMap { Int($0) },
            // POSE_DUMP=<path.json> writes poses before and after BA (for evaluating pose
            // accuracy).
            poseDumpURL: env["POSE_DUMP"].map { URL(fileURLWithPath: $0) },
            log: logErr)
        let result = try CaptureTraining.train(
            geometry: geometry,
            iterations: iterations,
            strategy: densificationStrategy,
            // REG=<w>: overrides the opacity / scale regularization weight.
            opacityReg: env["REG"].flatMap { Float($0) } ?? CaptureTraining.AppPreset.regularization,
            scaleReg: env["REG"].flatMap { Float($0) } ?? CaptureTraining.AppPreset.regularization,
            lambdaSSIM: lambdaSSIM,
            // EVAL_HOLDOUT=N: holds out 1 in N images from training and logs held-out PSNR.
            evalHoldoutEvery: env["EVAL_HOLDOUT"].flatMap { Int($0) } ?? 0,
            // EVAL_HOLDOUT_ALL=1: take the held-out views from all frames before thinning to
            // VIEW_CAP, so the held-out set is the same for every VIEW_CAP.
            evalHoldoutBeforeViewSelection: ["1", "on", "yes", "true"].contains(
                env["EVAL_HOLDOUT_ALL"]?.lowercased() ?? ""),
            // EVAL_RAW=1: also evaluates held-out on the images as captured (compares with/without
            // undistortion against the same ground truth).
            evalAgainstCapturedImages: ["1", "on", "yes", "true"].contains(env["EVAL_RAW"]?.lowercased() ?? ""),
            replayKMax: CaptureTraining.AppPreset.replayKMax,
            // INIT_CAP=<n>: initial Gaussian count cap (default 5/6 of CAP_MAX, as in the app).
            // VIEW_CAP=<n>: cap on training views (0 = all frames). Default: the app's 0.4 view/frame,
            // at least 120.
            trainingViewCap: env["VIEW_CAP"].flatMap { Int($0) },
            maxInitGaussians: env["INIT_CAP"].flatMap { Int($0) }
                ?? CaptureTraining.AppPreset.initialGaussianCap(cap: capMax),
            // C2F=<fraction>: share of iterations at 1/C2F_FACTOR of the captured resolution.
            // FINE_RES=<num>/<den> | full: resolution of the rest. The defaults are the app's (0.5,
            // 1/3 and 2/3: the coarse phase is half the fine resolution); with FINE_RES the coarse
            // factor is 2.
            coarseToFineFactor: env["C2F_FACTOR"].flatMap { Int($0) }
                ?? (env["FINE_RES"] == nil ? CaptureTraining.AppPreset.coarseToFineFactor : 2),
            coarseToFineFraction: env["C2F"].flatMap { Float($0) } ?? CaptureTraining.AppPreset.coarseToFineFraction,
            fineResample: {
                if let v = env["FINE_RES"] {
                    if v.lowercased() == "full" { return (1, 1) }
                    let p = v.split(separator: "/").compactMap { Int($0) }
                    if p.count == 2, p[0] > 0, p[1] >= p[0] { return (p[0], p[1]) }
                }
                return CaptureTraining.AppPreset.fineResample
            }(),
            outputPLY: outputPLY,
            // SEED_PLY=<path>: writes the pre-training seed point cloud + cameras.
            seedInspectionPLY: env["SEED_PLY"].map { URL(fileURLWithPath: $0) },
            // Geometric regularization (G³S surface depth with normal consistency and the MVS
            // depth), on as in the app; GEO_REG=0 turns it off. GEO_WN / GEO_WMVS weights,
            // GEO_START fraction, GEO_EVERY, GEO_PROFILE gof|solid, GEO_DEPTH_ITERS,
            // GEO_SAMPLE_WIDTH, GEO_PROFILE_STAGES=1.
            geometryRegularization: {
                if ["0", "off", "no", "false"].contains(env["GEO_REG"]?.lowercased() ?? "") { return nil }
                guard var g = CaptureTraining.AppPreset.geometryRegularization else { return nil }
                if let v = env["GEO_WN"].flatMap({ Float($0) }) { g.normalWeight = v }
                if let v = env["GEO_WMVS"].flatMap({ Float($0) }) { g.mvsDepthWeight = v }
                if let v = env["GEO_START"].flatMap({ Float($0) }) { g.startFraction = v }
                if let v = env["GEO_EVERY"].flatMap({ Int($0) }) { g.every = max(1, v) }
                if env["GEO_PROFILE"] == "gof" { g.depth.profile = .opacityField }
                if let v = env["GEO_DEPTH_ITERS"].flatMap({ Int($0) }) { g.depth.bisectionIterations = v }
                if let v = env["GEO_SAMPLE_WIDTH"].flatMap({ Int($0) }) { g.sampleWidth = v }
                GeometryRegularization.profileStages = env["GEO_PROFILE_STAGES"] == "1"
                return g
            }(),
            log: logErr
        )
        // With EVAL_HOLDOUT the meshes leave out the MVS depth of the held-out views too, so
        // `mesh-eval` compares them on views none of them saw.
        // The split is also written to <output>.split.json for mesh-eval / gs-mesh (EVAL_SPLIT).
        let holdoutEvery = env["EVAL_HOLDOUT"].flatMap { Int($0) } ?? 0
        var heldOutViews: Set<[Float]> = []
        if holdoutEvery > 1 {
            let split = CaptureTraining.evaluationSplit(
                geometry: geometry, holdoutEvery: holdoutEvery,
                trainingViewCap: env["VIEW_CAP"].flatMap { Int($0) },
                holdoutBeforeSelection: ["1", "on", "yes", "true"].contains(env["EVAL_HOLDOUT_ALL"]?.lowercased() ?? ""))
            heldOutViews = Set(split.heldOut.map(\.viewmat))
            try EvaluationSplit.write(
                training: split.training, heldOut: split.heldOut,
                to: outputPLY.deletingPathExtension().appendingPathExtension("split.json"))
        }
        // MESH_OBJ=<path.obj>: after training, build the mesh as the app's Generate mesh does
        // (MESH_* knobs in MeshCapture.swift).
        if let meshOBJ = env["MESH_OBJ"].map({ URL(fileURLWithPath: $0) }),
           !(try writeCaptureMesh(geometry: geometry, session: result.session,
                                  sourceName: valsetURL.lastPathComponent,
                                  to: meshOBJ, excluding: heldOutViews, log: logErr)) {
            logErr("mesh: no surface reconstructed")
        }
        // MESH_HYBRID / MESH_GOF / MESH_GS_TSDF: meshes using the trained Gaussians (MeshCapture.swift).
        if let session = result.session {
            _ = try writeGaussianMeshes(
                model: session.trainer.model, cameras: session.views.map(\.camera),
                shDegree: session.maxShDegree, geometry: geometry,
                sourceName: valsetURL.lastPathComponent, excluding: heldOutViews, log: logErr)
        }
        let finalLoss = result.lossHistory.last ?? -1
        let mem = MLX.Memory.snapshot()
        let mb = { (b: Int) in String(format: "%.1f", Double(b) / (1024 * 1024)) }
        print("done: \(result.gaussianCount) gaussians, \(result.viewCount) views, "
            + "final loss \(finalLoss)")
        print("wrote \(outputPLY.path)")
        print("GPU mem: peak=\(mb(mem.peakMemory))M  active=\(mb(mem.activeMemory))M")
    } catch {
        print("train-capture failed: \(error)")
        exit(1)
    }

case "mesh-eval":
    // Meshes compared on the views train-capture held out (MeshEvaluation.swift).
    guard args.count >= 4 else {
        print("usage: splatelier mesh-eval <dataset.valset> <mesh.obj>...  (EVAL_HOLDOUT / EVAL_SCALE / EVAL_IMAGES)")
        exit(1)
    }
    do {
        try runMeshEvaluation(
            valsetURL: URL(fileURLWithPath: args[2]), meshURLs: args[3...].map { URL(fileURLWithPath: $0) },
            log: { FileHandle.standardError.write(Data(($0 + "\n").utf8)) })
    } catch {
        print("mesh-eval failed: \(error)")
        exit(1)
    }

case "gs-mesh":
    // Meshes from an already trained PLY, without training (MeshCapture.swift).
    guard args.count >= 4 else {
        print("usage: splatelier gs-mesh <dataset.valset> <trained.ply>  (MESH_OBJ / MESH_HYBRID / MESH_GOF / MESH_GS_TSDF)")
        exit(1)
    }
    do {
        try runGaussianMeshCommand(
            valsetURL: URL(fileURLWithPath: args[2]), plyURL: URL(fileURLWithPath: args[3]),
            log: { FileHandle.standardError.write(Data(($0 + "\n").utf8)) })
    } catch {
        print("gs-mesh failed: \(error)")
        exit(1)
    }

case "dump-sparse":
    // Turns the .valset's sparse points (ORB + BA refined) into a gaussian PLY.
    guard args.count >= 4 else {
        print("usage: splatelier dump-sparse <dataset.valset> <output.ply>")
        exit(1)
    }
    let valsetURL = URL(fileURLWithPath: args[2])
    let outputPLY = URL(fileURLWithPath: args[3])
    do {
        let logErr: (String) -> Void = { FileHandle.standardError.write(Data(($0 + "\n").utf8)) }
        let scene = try CaptureDatasetStore.load(from: valsetURL)
        logErr("loaded \(scene.frames.count) frames, \(scene.sparsePoints.count) raw sparse points")
        // Same BA as training.
        let refined = CaptureBA.refine(scene: scene, log: logErr)
        let pts: [SIMD3<Float>] = (refined.baRefinedPoints?.isEmpty == false)
            ? refined.baRefinedPoints! : scene.sparsePoints
        guard !pts.isEmpty else { print("dump-sparse: no sparse points"); exit(1) }
        var flat = [Float](); flat.reserveCapacity(pts.count * 3)
        for p in pts { flat.append(p.x); flat.append(p.y); flat.append(p.z) }
        let positions = MLXArray(flat, [pts.count, 3])
        // No color information, so uniform gray.
        let colors = MLXArray([Float](repeating: 0.7, count: pts.count * 3), [pts.count, 3])
        let model = GaussianModel.fromPointCloud(positions: positions, colors: colors)
        let stats = try PLYWriter.writeGaussianPLY(model: model, to: outputPLY)
        print("dump-sparse: wrote \(stats.kept) sparse points (BA-refined "
            + "\(refined.baRefinedPoints?.count ?? 0), raw \(scene.sparsePoints.count)) to \(outputPLY.path)")
    } catch {
        print("dump-sparse failed: \(error)")
        exit(1)
    }

case "live-mvs-replay":
    // splatelier live-mvs-replay <dataset.valset> <outDir>
    // Feasibility check for in-capture MVS (capture guide): feeds keyframes in capture order and
    // measures incremental MVS time and point cloud.
    guard args.count >= 4 else {
        print("usage: splatelier live-mvs-replay <dataset.valset> <outDir>")
        exit(1)
    }
    do {
        try runLiveMVSReplay(valsetURL: URL(fileURLWithPath: args[2]),
                             outDir: URL(fileURLWithPath: args[3]))
    } catch {
        print("live-mvs-replay failed: \(error)")
        exit(1)
    }

case "eval":
    guard args.count >= 4 else {
        print("usage: splatelier eval <ply> <colmap_scene_dir> [maxDim=256]")
        exit(1)
    }
    let plyURL = URL(fileURLWithPath: args[2])
    let dir = URL(fileURLWithPath: args[3])
    let maxDim = args.count > 4 ? (Int(args[4]) ?? 256) : 256
    MLX.Memory.peakMemory = 0
    do {
        let r = try TrainingDriver.evaluate(
            plyURL: plyURL, sceneDirectory: dir, maxImageDimension: maxDim
        )
        let psnrs = r.perViewPSNR.sorted()
        let ssims = r.perViewSSIM.sorted()
        let psnrsCC = r.perViewPSNRCC.sorted()
        let ssimsCC = r.perViewSSIMCC.sorted()
        let med = { (xs: [Float]) -> Float in xs.isEmpty ? 0 : xs[xs.count / 2] }
        print(String(format: "PSNR      mean=%.3f dB  median=%.3f dB  min=%.3f dB  max=%.3f dB",
                     r.meanPSNR, med(psnrs), psnrs.first ?? 0, psnrs.last ?? 0))
        print(String(format: "SSIM      mean=%.4f  median=%.4f  min=%.4f  max=%.4f",
                     r.meanSSIM, med(ssims), ssims.first ?? 0, ssims.last ?? 0))
        print(String(format: "PSNR (cc) mean=%.3f dB  median=%.3f dB  min=%.3f dB  max=%.3f dB",
                     r.meanPSNRCC, med(psnrsCC), psnrsCC.first ?? 0, psnrsCC.last ?? 0))
        print(String(format: "SSIM (cc) mean=%.4f  median=%.4f  min=%.4f  max=%.4f",
                     r.meanSSIMCC, med(ssimsCC), ssimsCC.first ?? 0, ssimsCC.last ?? 0))
        print(String(format: "Δ from color correction: PSNR +%.3f dB, SSIM +%.4f  (larger means color shifts such as AE/AWB drift dominate)",
                     r.meanPSNRCC - r.meanPSNR, r.meanSSIMCC - r.meanSSIM))
        let mem = MLX.Memory.snapshot()
        let mb = { (b: Int) in String(format: "%.1f", Double(b) / (1024 * 1024)) }
        print("GPU mem: peak=\(mb(mem.peakMemory))M  active=\(mb(mem.activeMemory))M  "
            + "cache=\(mb(mem.cacheMemory))M")
    } catch {
        print("eval failed: \(error)")
        exit(1)
    }

case "compare":
    // Renders 2 PLYs from the same views and measures divergence (worst-case) (for validating
    // fp16).
    guard args.count >= 5 else {
        print("usage: splatelier compare <plyA> <plyB> <colmap_scene_dir> [maxDim=256]")
        exit(1)
    }
    let plyA = URL(fileURLWithPath: args[2])
    let plyB = URL(fileURLWithPath: args[3])
    let dir = URL(fileURLWithPath: args[4])
    let maxDim = args.count > 5 ? (Int(args[5]) ?? 256) : 256
    MLX.Memory.peakMemory = 0
    do {
        let c = try TrainingDriver.compareModels(
            plyA: plyA, plyB: plyB, sceneDirectory: dir, maxImageDimension: maxDim
        )
        let psnrs = c.perViewPSNR.sorted()
        let med = { (xs: [Float]) -> Float in xs.isEmpty ? 0 : xs[xs.count / 2] }
        print(String(format: "render divergence (A vs B): minPSNR=%.2f dB  median=%.2f dB  worstMaxPixelErr=%.4f @ %@",
                     c.minPSNR, med(psnrs), c.worstMaxErr, c.worstViewName))
        if c.nonFiniteViews.isEmpty {
            print("NaN/Inf canary: clean (all views finite)")
        } else {
            print("⚠️ NaN/Inf in \(c.nonFiniteViews.count) view(s): \(c.nonFiniteViews.prefix(5).joined(separator: ", "))")
        }
    } catch {
        print("compare failed: \(error)")
        exit(1)
    }

case "view":
    guard args.count >= 3 else { usage(); exit(1) }
    let plyURL = URL(fileURLWithPath: args[2])
    #if os(macOS)
    MainActor.assumeIsolated {
        SplatViewerApp.run(url: plyURL, title: "splatelier — \(plyURL.lastPathComponent)")
    }
    #else
    print("view is only supported on macOS")
    exit(1)
    #endif

case "train-view":
    guard args.count >= 3 else { usage(); exit(1) }
    let dir = URL(fileURLWithPath: args[2])
    let iterations = args.count > 3 ? (Int(args[3]) ?? 500) : 500
    let maxDim = args.count > 4 ? (Int(args[4]) ?? 512) : 512
    let maxPoints = args.count > 5 ? (Int(args[5]) ?? 30_000) : 30_000
    let outputPLY: URL? = args.count > 6 ? URL(fileURLWithPath: args[6]) : nil
    let capMax = args.count > 7 ? (Int(args[7]) ?? 1_000_000) : 1_000_000
    // args[8]=[init], args[9]=[replayCache]
    let initKind = args.count > 8 ? parseInitKind(args[8]) : .gsplatIsotropic
    let useReplayCache = args.count > 9 ? parseReplayCache(args[9]) : false
    let snapshotEvery = max(1, iterations / 60)
    #if os(macOS)
    MainActor.assumeIsolated {
        SplatViewerApp.runLive(title: "splatelier — \(dir.lastPathComponent)") { box in
            // Scale to the horizon, as in train.
            let strategy = MCMCStrategy.scaled(iterations: iterations, capMax: capMax)
            do {
                _ = try TrainingDriver.trainColmap(
                    sceneDirectory: dir,
                    iterations: iterations,
                    maxImageDimension: maxDim,
                    maxPoints: maxPoints,
                    strategy: strategy,
                    opacityReg: 0.01,
                    scaleReg: 0.01,
                    initKind: initKind,
                    useReplayCache: useReplayCache,
                    outputPLY: outputPLY,
                    onSnapshot: { model, it in
                        if it % snapshotEvery == 0 || it == iterations - 1 {
                            box.publish(modelToSplatPoints(model))
                        }
                    },
                    log: { msg in FileHandle.standardError.write(Data((msg + "\n").utf8)) }
                )
                print("train-view: training complete")
            } catch {
                print("train-view: training failed: \(error)")
            }
        }
    }
    #else
    print("train-view is only supported on macOS")
    exit(1)
    #endif

case "capture":
    guard args.count >= 4 else { usage(); exit(1) }
    let plyURL = URL(fileURLWithPath: args[2])
    let pngURL = URL(fileURLWithPath: args[3])
    let w = args.count > 4 ? (Int(args[4]) ?? 1024) : 1024
    let h = args.count > 5 ? (Int(args[5]) ?? 768) : 768
    #if os(macOS)
    captureSplatPNG(url: plyURL, outURL: pngURL, width: w, height: h)
    #else
    print("capture is only supported on macOS")
    exit(1)
    #endif

case "render":
    // Renders COLMAP training views with the training renderer and outputs render / target PNGs and
    // coverage (for comparison with external viewers).
    guard args.count >= 5 else { usage(); exit(1) }
    let plyURL = URL(fileURLWithPath: args[2])
    let dir = URL(fileURLWithPath: args[3])
    let outPrefix = args[4]
    let viewIdx = args.count > 5 ? (Int(args[5]) ?? 0) : 0
    let maxDim = args.count > 6 ? (Int(args[6]) ?? 512) : 512
    do {
        let model = try PLYReader.readGaussianPLY(url: plyURL)
        let scene = try ColmapLoader.load(directory: dir)
        let bg = MLXArray([Float(0), 0, 0], [3])
        let views = ColmapTraining.makeViews(
            scene: scene, imagesDirectory: dir.appendingPathComponent("images"),
            maxDimension: maxDim, background: bg
        )
        guard viewIdx >= 0, viewIdx < views.count else {
            print("viewIdx out of range (have \(views.count) views)"); exit(1)
        }
        let v = views[viewIdx]
        let renderer = try GaussianRenderer()
        let (render, alpha, _) = renderer.forward(model: model, camera: v.camera)
        eval(render, alpha)
        let cov = alpha.mean().item(Float.self)
        let rmean = render.mean().item(Float.self)
        let rmax = render.max().item(Float.self)
        print("rendered view \(viewIdx) (\(v.name)) with our renderer:")
        print("  coverage (mean render_alpha) = \(cov)")
        print("  render mean = \(rmean), max = \(rmax)")
        #if os(macOS)
        saveImagePNG(render, to: URL(fileURLWithPath: outPrefix + "_render.png"))
        saveImagePNG(v.image, to: URL(fileURLWithPath: outPrefix + "_target.png"))
        #endif
    } catch {
        print("render failed: \(error)"); exit(1)
    }

case "stats":
    // Prints per-gaussian opacity / color / scale distributions of a PLY.
    guard args.count >= 3 else { usage(); exit(1) }
    let plyURL = URL(fileURLWithPath: args[2])
    do {
        let model = try PLYReader.readGaussianPLY(url: plyURL)
        let op = model.opacities()
        let scales = model.scales()
        // DC color (before clamp) = C0·sh0 + 0.5. frac>1 detects color clipping.
        let rgb = GaussianModel.shC0 * model.sh0 + 0.5
        eval(op, rgb, scales)
        let opH = op.asArray(Float.self)
        let colH = rgb.asArray(Float.self)
        let scH = scales.asArray(Float.self)
        func median(_ a: [Float]) -> Float { let s = a.sorted(); return s.isEmpty ? 0 : s[s.count / 2] }
        func mean(_ a: [Float]) -> Float { a.isEmpty ? 0 : a.reduce(0, +) / Float(a.count) }
        func frac(_ a: [Float], _ pred: (Float) -> Bool) -> Float {
            a.isEmpty ? 0 : Float(a.filter(pred).count) / Float(a.count)
        }
        print("gaussians: \(model.count)")
        print("opacity:  median \(median(opH))  mean \(mean(opH))  frac<0.05 \(frac(opH) { $0 < 0.05 })")
        print("color:    median \(median(colH))  mean \(mean(colH))  max \(colH.max() ?? 0)  frac>1 \(frac(colH) { $0 > 1 })  frac<0 \(frac(colH) { $0 < 0 })")
        print("scale:    median \(median(scH))  mean \(mean(scH))  max \(scH.max() ?? 0)")
    } catch {
        print("stats failed: \(error)"); exit(1)
    }

case "cull":
    // Removes off-position, transparent, and huge splats from an existing PLY (like the cull in
    // gsplat exporter.py). Keeps auto-fit in SuperSplat etc. from being thrown off by outliers.
    guard args.count >= 4 else {
        print("usage: splatelier cull <in.ply> <out.ply> "
              + "[minOpacity=0.00392] [maxScale=3.0] [maxAbsCoord=50.0]")
        exit(1)
    }
    let inURL = URL(fileURLWithPath: args[2])
    let outURL = URL(fileURLWithPath: args[3])
    let minOpacity = args.count > 4 ? (Float(args[4]) ?? 1.0 / 255.0) : 1.0 / 255.0
    let maxScale = args.count > 5 ? (Float(args[5]) ?? 3.0) : 3.0
    let maxAbsCoord = args.count > 6 ? (Float(args[6]) ?? 50.0) : 50.0
    do {
        let model = try PLYReader.readGaussianPLY(url: inURL)
        let stats = try PLYWriter.writeGaussianPLY(
            model: model, to: outURL,
            cull: .init(minOpacity: minOpacity, maxScale: maxScale, maxAbsCoord: maxAbsCoord)
        )
        print("cull: \(stats.total) → \(stats.kept) (dropped \(stats.total - stats.kept))")
        print(String(format: "  NaN/Inf:        %d", stats.droppedNonFinite))
        print(String(format: "  opacity ≤ %.4f: %d", minOpacity, stats.droppedLowOpacity))
        print(String(format: "  scale ≥ %.2f:   %d", maxScale, stats.droppedGiantScale))
        print(String(format: "  |coord| ≥ %.2f: %d", maxAbsCoord, stats.droppedOutlierPosition))
        print("wrote \(outURL.path)")
    } catch {
        print("cull failed: \(error)"); exit(1)
    }

default:
    usage()
}
