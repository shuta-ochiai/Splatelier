// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation

/// Strategy that grows, shrinks and perturbs the Gaussian set during training (gsplat's
/// `Strategy`).
public protocol DensificationStrategy {
    /// Called after every optimizer step. `step` is 0-based; `lr` is the current means LR (scale of
    /// MCMC noise).
    mutating func postStep(store: GaussianParamStore, step: Int, lr: Float)
}

/// Exponential decay from `start` at step 0 to `end` at the last step. Non-positive ends fall back
/// to `max(end, 0)`.
func exponentialLearningRate(start: Float, end: Float, step: Int, iterations: Int) -> Float {
    guard iterations > 1, start > 0, end > 0 else { return Swift.max(end, 0) }
    let t = Float(Swift.max(0, Swift.min(step, iterations - 1))) / Float(iterations - 1)
    return start * pow(end / start, t)
}

/// 3DGS as MCMC (Kheradmand et al., 2024), following gsplat's `MCMCStrategy`. Periodically
/// relocates dead Gaussians and grows toward capMax, adding Langevin position noise every step. The
/// fixed-count preset stops relocation / growth and prunes dead Gaussians instead.
public struct MCMCStrategy: DensificationStrategy {
    /// Gaussian count growth cap.
    public var capMax: Int
    /// Scale of MCMC noise (lr · noiseLR per step).
    public var noiseLR: Float
    /// Refine only while refineStartIter < step < refineStopIter.
    public var refineStartIter: Int
    public var refineStopIter: Int
    /// Refine interval (steps).
    public var refineEvery: Int
    /// Dead if opacity is at or below this.
    public var minOpacity: Float
    /// Each refine grows to min(capMax, growthFactor × current count).
    public var growthFactor: Float
    /// If > 0, maxScale > multiple × median also counts as dead (relocated or pruned). 0 disables
    /// (paper default).
    public var bloatScaleMultiple: Float
    /// Perform MCMC relocation of dead Gaussians onto live ones.
    public var relocateDeadGaussians: Bool
    /// Instead of relocating, delete dead Gaussians and compact model / optimizer state.
    public var pruneDeadGaussians: Bool
    /// Max fraction prune may remove per fire. Prevents sudden coverage collapse.
    public var maxPruneFraction: Float
    /// Max fraction (×N) relocate may move per fire, since relocating all at once OOMs. 1.0 is
    /// unlimited.
    public var maxRelocateFraction: Float
    /// Minimum number of Gaussians prune keeps.
    public var minRemainingGaussians: Int
    /// Diagnostic string from the most recent postStep. nil on steps without refine.
    public var lastDiagnostic: String?
    /// When set, the scale LR decays exponentially start → end like IGSPlus (0.020 → 0.004). nil
    /// keeps the trainer's constant scale LR.
    public var scaleLRSchedule: (start: Float, end: Float)?

    private let binoms: [Float]
    private var rng: SplitMix64

    public init(
        capMax: Int = 1_000_000,
        noiseLR: Float = 5e5,
        refineStartIter: Int = 500,
        refineStopIter: Int = 25_000,
        refineEvery: Int = 100,
        minOpacity: Float = 0.005,
        growthFactor: Float = 1.05,
        bloatScaleMultiple: Float = 0,
        relocateDeadGaussians: Bool = true,
        pruneDeadGaussians: Bool = false,
        maxPruneFraction: Float = 0.25,
        maxRelocateFraction: Float = 0.15,
        minRemainingGaussians: Int = 1,
        seed: UInt64 = 0
    ) {
        self.capMax = capMax
        self.noiseLR = noiseLR
        self.refineStartIter = refineStartIter
        self.refineStopIter = refineStopIter
        self.refineEvery = refineEvery
        self.minOpacity = minOpacity
        self.growthFactor = growthFactor
        self.bloatScaleMultiple = bloatScaleMultiple
        self.relocateDeadGaussians = relocateDeadGaussians
        self.pruneDeadGaussians = pruneDeadGaussians
        self.maxPruneFraction = maxPruneFraction
        self.maxRelocateFraction = maxRelocateFraction
        self.minRemainingGaussians = minRemainingGaussians
        self.binoms = Densification.binomialTable()
        self.rng = SplitMix64(seed: seed)
    }

    /// For short training. With the paper's refine start of 500 iters, runs of 500 iters or fewer
    /// never densify, so the schedule scales with total iters.
    public static func scaled(
        iterations: Int,
        capMax: Int = 1_000_000,
        growthFactor: Float = 1.05
    ) -> MCMCStrategy {
        MCMCStrategy(
            capMax: capMax,
            refineStartIter: Swift.max(50, iterations / 10),
            refineStopIter: Swift.max(100, (iterations * 4) / 5),
            refineEvery: Swift.max(10, iterations / 60),
            growthFactor: growthFactor
        )
    }

    /// Fixed-count mobile preset. Stops relocation and growth and periodically prunes dead
    /// Gaussians (the count shrinks but never grows). Meant for keeping a PocketGS-style dense
    /// initial seed intact under a short budget. Defaults target the paper's 500-iter budget, with
    /// no pruning in the first 10% and last 20%. noiseLR=0 matches the deterministic (Adam-only)
    /// contract of paper v4.
    public static func fixedCount(
        noiseLR: Float = 0,
        refineStartIter: Int = 50,
        refineStopIter: Int = 400,
        refineEvery: Int = 50,
        minOpacity: Float = 0.005,
        bloatScaleMultiple: Float = 100,
        maxPruneFraction: Float = 0.25,
        minRemainingGaussians: Int = 1,
        // Dead handling: prune (delete) / relocate (reuse, count unchanged) / neither if both
        // false. Growth is always off.
        relocateDeadGaussians: Bool = false,
        pruneDeadGaussians: Bool = true,
        seed: UInt64 = 0
    ) -> MCMCStrategy {
        // capMax is large enough that target = min(capMax, current) equals current.
        // bloatScaleMultiple defaults to 100: high-opacity giant blobs never die by the opacity
        // test.
        MCMCStrategy(
            capMax: Int.max / 2,
            noiseLR: noiseLR,
            refineStartIter: refineStartIter,
            refineStopIter: refineStopIter,
            refineEvery: refineEvery,
            minOpacity: minOpacity,
            growthFactor: 1.0,
            bloatScaleMultiple: bloatScaleMultiple,
            relocateDeadGaussians: relocateDeadGaussians,
            pruneDeadGaussians: pruneDeadGaussians,
            maxPruneFraction: maxPruneFraction,
            minRemainingGaussians: minRemainingGaussians,
            seed: seed
        )
    }

    /// Scale LR at `step` when `scaleLRSchedule` is set, otherwise nil.
    public func scaleLearningRate(step: Int, iterations: Int) -> Float? {
        guard let schedule = scaleLRSchedule else { return nil }
        return exponentialLearningRate(
            start: schedule.start, end: schedule.end, step: step, iterations: iterations)
    }

    public mutating func postStep(store: GaussianParamStore, step: Int, lr: Float) {
        lastDiagnostic = nil
        if step > refineStartIter && step < refineStopIter && step % refineEvery == 0 {
            // Relocate / prune dead Gaussians and grow toward the cap. The host snapshot is taken
            // once at the start of the fire.
            let snapshot1 = Densification.snapshotHost(store: store)
            var moved = 0
            var pruned = 0
            if relocateDeadGaussians {
                moved = Densification.relocate(
                    store: store, snapshot: snapshot1,
                    binoms: binoms, minOpacity: minOpacity,
                    bloatScaleMultiple: bloatScaleMultiple,
                    maxRelocateFraction: maxRelocateFraction, using: &rng
                )
            } else if pruneDeadGaussians {
                pruned = Densification.pruneDead(
                    store: store, snapshot: snapshot1,
                    minOpacity: minOpacity,
                    bloatScaleMultiple: bloatScaleMultiple,
                    maxPruneFraction: maxPruneFraction,
                    minRemaining: minRemainingGaussians
                )
            }
            let current = store.gaussianCount
            let target = Swift.min(capMax, Int(growthFactor * Float(current)))
            let nAdd = Swift.max(0, target - current)
            var added = 0
            if nAdd > 0 {
                // Relocate changed opacities, so refetch the snapshot.
                let snapshot2 = Densification.snapshotHost(store: store)
                added = Densification.sampleAdd(
                    store: store, count: nAdd, snapshot: snapshot2,
                    binoms: binoms, minOpacity: minOpacity, using: &rng
                )
            }
            if pruneDeadGaussians && !relocateDeadGaussians {
                lastDiagnostic = "[MCMC-fixed step=\(step)] prune removed=\(pruned) added=\(added) total=\(store.gaussianCount)"
            } else {
                lastDiagnostic = "[MCMC step=\(step)] relocate moved=\(moved) added=\(added) total=\(store.gaussianCount)"
            }
        }
        // Langevin perturbation (every step). noiseLR=0 makes it deterministic (paper v4).
        if noiseLR > 0 {
            Densification.injectNoiseToPosition(store: store, scaler: lr * noiseLR)
        }
    }
}

/// ImprovedGS+ (arXiv 2603.08661) densification. Long-Axis-Splits the top edge-score Gaussians and
/// decays the scale LR exponentially. Without edge scores, LAS uses a deterministic proxy.
public struct IGSPlusStrategy: DensificationStrategy {
    /// Gaussian count growth cap.
    public var capMax: Int
    /// LAS active range and interval.
    public var refineStartIter: Int
    public var refineStopIter: Int
    public var refineEvery: Int
    /// Each LAS grows to min(capMax, growthFactor × current count).
    public var growthFactor: Float
    /// Gaussians with opacity below this are not split.
    public var minOpacity: Float
    /// Lower bound on major-axis scale for split candidates.
    public var minSplitScale: Float
    /// The edge map is median-normalized, so the default 1 keeps those at or above the median
    /// positive edge.
    public var minImportance: Float
    /// The first N LAS rounds ignore `minImportance` (the paper's warm-up).
    public var warmupDensifySteps: Int
    /// LAS coefficients from ImprovedGS+ Algorithm 1.
    public var alpha: Float
    public var secondaryScaleFactor: Float
    public var opacityFactor: Float
    /// Exponential decay of the position LR. In normalized scene units; the training loop
    /// multiplies by sceneScale.
    public var meansLRStart: Float
    public var meansLREnd: Float
    /// Exponential decay of the scale LR (paper: 0.020 → 0.002). Spreads early, stabilizes late.
    public var scaleLRStart: Float
    public var scaleLREnd: Float
    /// Hybrid (MCMC×IGS+): each fire runs pruneDead before LAS and refills the freed slots by
    /// importance split (IGS+ alone never reclaims dead Gaussians). false gives pure IGS+.
    public var pruneDeadGaussians: Bool
    /// Max fraction pruneDead may remove per fire.
    public var maxPruneFraction: Float
    /// pruneDead bloat test (maxScale > multiple × median counts as dead). 0 disables.
    public var bloatScaleMultiple: Float
    /// Minimum number of Gaussians pruneDead keeps.
    public var minRemainingGaussians: Int
    /// Diagnostic string from the most recent postStep.
    public var lastDiagnostic: String?

    private var densifyFireCount: Int = 0
    private var pendingImportanceScores: [Float]?

    public init(
        capMax: Int = 1_000_000,
        refineStartIter: Int = 500,
        refineStopIter: Int = 15_000,
        refineEvery: Int = 500,
        growthFactor: Float = 1.05,
        minOpacity: Float = 0.005,
        minSplitScale: Float = 0,
        minImportance: Float = 1,
        warmupDensifySteps: Int = 3,
        alpha: Float = 0.5,
        secondaryScaleFactor: Float = 0.85,
        opacityFactor: Float = 0.6,
        // In short runs the LR hits its floor early and the second half does little, so end is set
        // higher than the paper (×0.1).
        meansLRStart: Float = 0.000128,
        meansLREnd: Float = 0.000038,   // start ×0.3
        scaleLRStart: Float = 0.020,
        scaleLREnd: Float = 0.004,      // start ×0.2
        pruneDeadGaussians: Bool = false,
        maxPruneFraction: Float = 0.25,
        bloatScaleMultiple: Float = 0,
        minRemainingGaussians: Int = 1
    ) {
        self.capMax = capMax
        self.refineStartIter = refineStartIter
        self.refineStopIter = refineStopIter
        self.refineEvery = refineEvery
        self.growthFactor = growthFactor
        self.minOpacity = minOpacity
        self.minSplitScale = minSplitScale
        self.minImportance = minImportance
        self.warmupDensifySteps = warmupDensifySteps
        self.alpha = alpha
        self.secondaryScaleFactor = secondaryScaleFactor
        self.opacityFactor = opacityFactor
        self.meansLRStart = meansLRStart
        self.meansLREnd = meansLREnd
        self.scaleLRStart = scaleLRStart
        self.scaleLREnd = scaleLREnd
        self.pruneDeadGaussians = pruneDeadGaussians
        self.maxPruneFraction = maxPruneFraction
        self.bloatScaleMultiple = bloatScaleMultiple
        self.minRemainingGaussians = minRemainingGaussians
    }

    /// For short training. Scales the paper's 500..15k / every 500 to total iters (densify mid-run,
    /// stabilize at the end).
    public static func scaled(
        iterations: Int,
        capMax: Int = 1_000_000,
        growthFactor: Float = 1.05
    ) -> IGSPlusStrategy {
        IGSPlusStrategy(
            capMax: capMax,
            refineStartIter: Swift.max(10, iterations / 10),
            refineStopIter: Swift.max(50, (iterations * 4) / 5),
            refineEvery: Swift.max(10, iterations / 30),
            growthFactor: growthFactor
        )
    }

    public func meansLearningRate(step: Int, iterations: Int, sceneScale: Float) -> Float {
        exponentialLearningRate(
            start: meansLRStart, end: meansLREnd, step: step, iterations: iterations) * sceneScale
    }

    public func scaleLearningRate(step: Int, iterations: Int) -> Float {
        exponentialLearningRate(
            start: scaleLRStart, end: scaleLREnd, step: step, iterations: iterations)
    }

    public func shouldPrepareImportance(step: Int, gaussianCount: Int) -> Bool {
        guard refineEvery > 0,
              step >= refineStartIter,
              step <= refineStopIter,
              step % refineEvery == 0
        else { return false }
        // Hybrid frees slots by pruning, so prepare scores for the split even at the cap
        // (without them it falls back to proxy order and, after warm-up, the threshold
        // rejects every candidate).
        if pruneDeadGaussians { return true }
        let target = Swift.min(capMax, Int(growthFactor * Float(gaussianCount)))
        return target > gaussianCount
    }

    public mutating func setImportanceScores(_ scores: [Float]?) {
        pendingImportanceScores = scores
    }

    public mutating func postStep(store: GaussianParamStore, step: Int, lr: Float) {
        lastDiagnostic = nil
        defer { pendingImportanceScores = nil }
        guard refineEvery > 0,
              step >= refineStartIter,
              step <= refineStopIter,
              step % refineEvery == 0
        else { return }

        // Hybrid: pruneDead removes dead Gaussians / floaters before LAS.
        var pruned = 0
        if pruneDeadGaussians {
            let before = store.gaussianCount
            if let keep = Densification.pruneDeadKeepingIndices(
                store: store, snapshot: Densification.snapshotHost(store: store),
                minOpacity: minOpacity,
                bloatScaleMultiple: bloatScaleMultiple,
                maxPruneFraction: maxPruneFraction,
                minRemaining: minRemainingGaussians)
            {
                pruned = before - keep.count
                // Edge scores are in pre-prune row order, so compact them with the keep index
                // (on a length mismatch longAxisSplit discards the scores).
                if let scores = pendingImportanceScores, scores.count == before {
                    pendingImportanceScores = keep.map { scores[Int($0)] }
                }
            }
        }

        let current = store.gaussianCount   // growth is based on the post-prune count
        let target = Swift.min(capMax, Int(growthFactor * Float(current)))
        let nSplit = Swift.max(0, target - current)
        guard nSplit > 0 else {
            lastDiagnostic = "[IGSPlus step=\(step)] prune=\(pruned) split=0 total=\(current)"
            return
        }

        let useThreshold = densifyFireCount >= warmupDensifySteps
        let edgeStats: (positive: Int, aboveThreshold: Int, maxScore: Float)?
        if let scores = pendingImportanceScores {
            var positive = 0
            var aboveThreshold = 0
            var maxScore: Float = 0
            for score in scores where score.isFinite && score > 0 {
                positive += 1
                if score >= minImportance { aboveThreshold += 1 }
                if score > maxScore { maxScore = score }
            }
            edgeStats = (positive, aboveThreshold, maxScore)
        } else {
            edgeStats = nil
        }
        let split = Densification.longAxisSplit(
            store: store,
            count: nSplit,
            importanceScores: pendingImportanceScores,
            alpha: alpha,
            secondaryScaleFactor: secondaryScaleFactor,
            opacityFactor: opacityFactor,
            minOpacity: minOpacity,
            minSplitScale: minSplitScale,
            minImportance: minImportance,
            useImportanceThreshold: useThreshold
        )
        densifyFireCount += 1
        let prefix = pruneDeadGaussians ? "[Hybrid step=\(step)] prune=\(pruned) " : "[IGSPlus step=\(step)] "
        if let edgeStats {
            let mask = useThreshold ? "on" : "warmup"
            lastDiagnostic = prefix + "split=\(split) total=\(store.gaussianCount) " +
                "edge=\(edgeStats.positive) edge>=thr=\(edgeStats.aboveThreshold) " +
                "edgeMax=\(edgeStats.maxScore) mask=\(mask)"
        } else {
            lastDiagnostic = prefix + "split=\(split) total=\(store.gaussianCount)"
        }
    }
}
