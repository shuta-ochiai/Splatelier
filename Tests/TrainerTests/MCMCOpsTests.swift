// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXRandom
import Testing
@testable import Trainer

// Tests applying MCMC densification ops (relocate / sampleAdd) to a real `GaussianParamStore`
// (trainer). Uses the GPU renderer to confirm a training step still runs after the model and
// optimizer state are resized.

private func makeOpsCamera() -> Camera {
    let viewmat = MLXArray(
        [Float(1), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [16]
    )
    return Camera(
        viewmat: viewmat,
        intrinsics: MLXArray([Float(80), 80, 16, 16], [4]),
        width: 32, height: 32, tileSize: 16,
        background: MLXArray([Float(0), 0, 0], [3])
    )
}

/// A model with `live` Gaussians at opacity 0.8 + `dead` at opacity 0.001, placed in view at z = 4.
private func makeMixedModel(live: Int, dead: Int) -> GaussianModel {
    let n = live + dead
    var means: [Float] = []
    for i in 0..<n { means += [Float(i) * 0.1 - 0.5, 0, 4.0] }
    let quats = (0..<n).flatMap { _ -> [Float] in [1, 0, 0, 0] }
    let logScales = [Float](repeating: log(0.1), count: n * 3)
    var logitOp: [Float] = []
    for i in 0..<n { logitOp.append(i < live ? logit(0.8) : logit(0.001)) }
    return GaussianModel(
        means: MLXArray(means, [n, 3]),
        quats: MLXArray(quats, [n, 4]),
        logScales: MLXArray(logScales, [n, 3]),
        logitOpacities: MLXArray(logitOp, [n]),
        sh0: MLXArray.zeros([n, 3]),   // rgb 0.5 ⇒ sh0 = 0
        shN: MLXArray.zeros([n, GaussianModel.shRestBands, 3])
    )
}

@Test
func relocateMovesDeadOntoLiveAndPreservesCount() throws {
    let renderer = try GaussianRenderer()
    let model = makeMixedModel(live: 5, dead: 3)
    let trainer = try GaussianTrainer(model: model, renderer: renderer)
    var rng = SplitMix64(seed: 1)
    let minOpacity: Float = 0.005

    let nReloc = Densification.relocate(
        store: trainer, binoms: Densification.binomialTable(), minOpacity: minOpacity, using: &rng
    )
    #expect(nReloc == 3)
    #expect(trainer.gaussianCount == 8, "relocate must preserve count")

    let deadAfter = model.opacities().asArray(Float.self).filter { $0 <= minOpacity }.count
    #expect(deadAfter == 0, "still \(deadAfter) dead Gaussians after relocate")

    let cam = makeOpsCamera()
    let target = MLXArray.zeros([cam.height, cam.width, 3])
    #expect(trainer.step(camera: cam, targetImage: target).isFinite)
    #expect(trainer.gaussianCount == 8)
}

/// Per-fire relocate cap (maxRelocateFraction). Even with many dead, cap at the limit and carry the
/// rest to the next fire (prevents OOM from relocating all at once).
@Test
func relocateRespectsMaxFractionCap() throws {
    let renderer = try GaussianRenderer()
    let model = makeMixedModel(live: 2, dead: 18)   // 20 total, 18 of them dead
    let trainer = try GaussianTrainer(model: model, renderer: renderer)
    var rng = SplitMix64(seed: 1)
    // cap = max(1, Int(20 × 0.1)) = 2 → of the 18 dead, only the 2 most dead are relocated.
    let nReloc = Densification.relocate(
        store: trainer, binoms: Densification.binomialTable(), minOpacity: 0.005,
        bloatScaleMultiple: 0, maxRelocateFraction: 0.1, using: &rng)
    #expect(nReloc == 2, "relocate cap should limit to 2 (10% of 20), got \(nReloc)")
    #expect(trainer.gaussianCount == 20, "count preserved")
    let deadAfter = model.opacities().asArray(Float.self).filter { $0 <= 0.005 }.count
    #expect(deadAfter == 16, "16 dead should remain (18-2), got \(deadAfter)")
}

@Test
func relocateIsNoopWhenNoDead() throws {
    let renderer = try GaussianRenderer()
    let model = makeMixedModel(live: 6, dead: 0)
    let trainer = try GaussianTrainer(model: model, renderer: renderer)
    var rng = SplitMix64(seed: 1)
    let n = Densification.relocate(
        store: trainer, binoms: Densification.binomialTable(), minOpacity: 0.005, using: &rng
    )
    #expect(n == 0)
    #expect(trainer.gaussianCount == 6)
}

@Test
func pruneDeadRemovesLowOpacityRowsAndKeepsTrainable() throws {
    let renderer = try GaussianRenderer()
    let model = makeMixedModel(live: 5, dead: 3)
    let trainer = try GaussianTrainer(model: model, renderer: renderer)
    let minOpacity: Float = 0.005

    let pruned = Densification.pruneDead(
        store: trainer,
        minOpacity: minOpacity,
        maxPruneFraction: 1
    )
    #expect(pruned == 3)
    #expect(trainer.gaussianCount == 5, "prune should compact away dead rows")

    let deadAfter = model.opacities().asArray(Float.self).filter { $0 <= minOpacity }.count
    #expect(deadAfter == 0, "still \(deadAfter) dead Gaussians after prune")

    let cam = makeOpsCamera()
    let target = MLXArray.zeros([cam.height, cam.width, 3])
    #expect(trainer.step(camera: cam, targetImage: target).isFinite)
    #expect(trainer.gaussianCount == 5)
}

@Test
func sampleAddGrowsCountAndKeepsTrainable() throws {
    let renderer = try GaussianRenderer()
    let model = makeMixedModel(live: 5, dead: 0)
    let trainer = try GaussianTrainer(model: model, renderer: renderer)
    var rng = SplitMix64(seed: 3)

    let added = Densification.sampleAdd(
        store: trainer, count: 4, binoms: Densification.binomialTable(), minOpacity: 0.005, using: &rng
    )
    #expect(added == 4)
    #expect(trainer.gaussianCount == 9)
    #expect(model.means.shape == [9, 3])
    #expect(model.logitOpacities.shape == [9])

    let cam = makeOpsCamera()
    let target = MLXArray.zeros([cam.height, cam.width, 3])
    #expect(trainer.step(camera: cam, targetImage: target).isFinite)
}

@Test
func longAxisSplitOffsetsAlongPrincipalAxisAndKeepsTrainable() throws {
    let renderer = try GaussianRenderer()
    let model = GaussianModel(
        means: MLXArray([Float(0), 0, 4], [1, 3]),
        quats: MLXArray([Float(1), 0, 0, 0], [1, 4]),
        logScales: MLXArray([log(Float(0.2)), log(Float(0.1)), log(Float(0.05))], [1, 3]),
        logitOpacities: MLXArray([logit(Float(0.8))], [1]),
        sh0: MLXArray.zeros([1, 3]),
        shN: MLXArray.zeros([1, GaussianModel.shRestBands, 3])
    )
    let trainer = try GaussianTrainer(model: model, renderer: renderer)

    let split = Densification.longAxisSplit(
        store: trainer,
        count: 1,
        alpha: 0.5,
        secondaryScaleFactor: 0.85,
        opacityFactor: 0.6,
        minOpacity: 0.005
    )

    #expect(split == 1)
    #expect(trainer.gaussianCount == 2)
    let means = model.means.asArray(Float.self)
    #expect(abs(means[0] - 0.1) < 1e-5)
    #expect(abs(means[3] + 0.1) < 1e-5)
    #expect(abs(means[2] - 4) < 1e-5)
    #expect(abs(means[5] - 4) < 1e-5)

    let opac = model.opacities().asArray(Float.self)
    #expect(abs(opac[0] - 0.48) < 1e-5)
    #expect(abs(opac[1] - 0.48) < 1e-5)

    let cam = makeOpsCamera()
    let target = MLXArray.zeros([cam.height, cam.width, 3])
    #expect(trainer.step(camera: cam, targetImage: target).isFinite)
}

@Test
func longAxisSplitUsesEdgeImportanceForCandidateSelection() throws {
    let renderer = try GaussianRenderer()
    let model = GaussianModel(
        means: MLXArray([Float(0), 0, 4, 1, 0, 4], [2, 3]),
        quats: MLXArray([Float(1), 0, 0, 0, 1, 0, 0, 0], [2, 4]),
        logScales: MLXArray(
            [
                log(Float(0.4)), log(Float(0.1)), log(Float(0.1)),
                log(Float(0.2)), log(Float(0.1)), log(Float(0.1)),
            ],
            [2, 3]
        ),
        logitOpacities: MLXArray([logit(Float(0.8)), logit(Float(0.8))], [2]),
        sh0: MLXArray.zeros([2, 3]),
        shN: MLXArray.zeros([2, GaussianModel.shRestBands, 3])
    )
    let trainer = try GaussianTrainer(model: model, renderer: renderer)

    let split = Densification.longAxisSplit(
        store: trainer,
        count: 1,
        importanceScores: [0.1, 10.0],
        alpha: 0.5,
        secondaryScaleFactor: 0.85,
        opacityFactor: 0.6,
        minOpacity: 0.005
    )

    #expect(split == 1)
    #expect(trainer.gaussianCount == 3)
    let means = model.means.asArray(Float.self)
    #expect(abs(means[0] - 0) < 1e-5, "low-edge larger gaussian should not split")
    #expect(abs(means[3] - 1.1) < 1e-5, "high-edge gaussian should become +offset copy")
    #expect(abs(means[6] - 0.9) < 1e-5, "high-edge gaussian should append -offset copy")

    let cam = makeOpsCamera()
    let target = MLXArray.zeros([cam.height, cam.width, 3])
    #expect(trainer.step(camera: cam, targetImage: target).isFinite)
}

@Test
func injectNoisePerturbsLowOpacityMoreThanHigh() throws {
    let renderer = try GaussianRenderer()
    // One nearly opaque (barely moves) and one nearly dead.
    let model = GaussianModel(
        means: MLXArray([Float(0), 0, 4, 1, 0, 4], [2, 3]),
        quats: MLXArray([Float(1), 0, 0, 0, 1, 0, 0, 0], [2, 4]),
        logScales: MLXArray([Float](repeating: log(0.1), count: 6), [2, 3]),
        logitOpacities: MLXArray([logit(0.9), logit(0.001)], [2]),
        sh0: MLXArray.zeros([2, 3]),
        shN: MLXArray.zeros([2, GaussianModel.shRestBands, 3])
    )
    let trainer = try GaussianTrainer(model: model, renderer: renderer)
    MLXRandom.seed(123)
    let before = model.means.asArray(Float.self)

    Densification.injectNoiseToPosition(store: trainer, scaler: 10.0)

    let after = model.means.asArray(Float.self)
    func displacement(_ i: Int) -> Float {
        let dx = after[i * 3] - before[i * 3]
        let dy = after[i * 3 + 1] - before[i * 3 + 1]
        let dz = after[i * 3 + 2] - before[i * 3 + 2]
        return (dx * dx + dy * dy + dz * dz).squareRoot()
    }
    let dHigh = displacement(0)   // opacity 0.9 -> gate ~ 0
    let dLow = displacement(1)    // opacity 0.001 -> gate ~ 0.6
    #expect(dLow > dHigh, "low-opacity should move more: low=\(dLow), high=\(dHigh)")
    #expect(dHigh < 1e-3, "near-opaque Gaussian should barely move: \(dHigh)")
    #expect(dLow > 1e-3, "near-dead Gaussian should move appreciably: \(dLow)")
}

@Test
func mcmcStrategyGrowsTowardCapAndStaysTrainable() throws {
    let renderer = try GaussianRenderer()
    let model = makeMixedModel(live: 20, dead: 0)
    let trainer = try GaussianTrainer(model: model, renderer: renderer)
    var strat = MCMCStrategy(
        capMax: 40, noiseLR: 1e4,
        refineStartIter: 0, refineStopIter: 1000, refineEvery: 5,
        minOpacity: 0.005, growthFactor: 1.5, seed: 1
    )
    let cam = makeOpsCamera()
    let target = MLXArray.zeros([cam.height, cam.width, 3])

    for step in 0..<20 {
        trainer.step(camera: cam, targetImage: target)
        strat.postStep(store: trainer, step: step, lr: 1.6e-4)
    }

    // Grows on the refine period (step 5/10/15) and saturates at capMax 40.
    #expect(trainer.gaussianCount > 20, "strategy should grow the count, got \(trainer.gaussianCount)")
    #expect(trainer.gaussianCount <= 40, "strategy must respect capMax, got \(trainer.gaussianCount)")
    #expect(trainer.step(camera: cam, targetImage: target).isFinite, "not trainable after densification")
}

@Test
func igsPlusStrategyGrowsWithLongAxisSplit() throws {
    let renderer = try GaussianRenderer()
    let model = makeMixedModel(live: 20, dead: 0)
    let trainer = try GaussianTrainer(model: model, renderer: renderer)
    var strat = IGSPlusStrategy(
        capMax: 40,
        refineStartIter: 0,
        refineStopIter: 100,
        refineEvery: 5,
        growthFactor: 1.5
    )
    let cam = makeOpsCamera()
    let target = MLXArray.zeros([cam.height, cam.width, 3])

    var sawDiagnostic = false
    for step in 0..<10 {
        trainer.step(camera: cam, targetImage: target)
        strat.postStep(store: trainer, step: step, lr: 1.6e-4)
        sawDiagnostic = sawDiagnostic || (strat.lastDiagnostic?.contains("IGSPlus") == true)
    }

    #expect(trainer.gaussianCount > 20, "IGSPlus should grow the count")
    #expect(trainer.gaussianCount <= 40, "IGSPlus must respect capMax")
    #expect(sawDiagnostic)
    #expect(trainer.step(camera: cam, targetImage: target).isFinite)
}

/// Hybrid (IGS+ + prune): IGSPlusStrategy with pruneDeadGaussians=true removes dead in postStep
/// and then grows via LAS (MCMC×IGS+ hybrid).
@Test
func hybridIGSPlusPrunesDeadThenGrows() throws {
    let renderer = try GaussianRenderer()
    let model = makeMixedModel(live: 16, dead: 8)   // 24 total, 8 of them dead
    let trainer = try GaussianTrainer(model: model, renderer: renderer)
    var strat = IGSPlusStrategy(
        capMax: 60, refineStartIter: 0, refineStopIter: 100, refineEvery: 5,
        growthFactor: 1.5, minOpacity: 0.005,
        pruneDeadGaussians: true)
    let cam = makeOpsCamera()
    let target = MLXArray.zeros([cam.height, cam.width, 3])

    var sawHybridDiag = false
    for step in 0..<10 {
        trainer.step(camera: cam, targetImage: target)
        strat.postStep(store: trainer, step: step, lr: 1.6e-4)
        sawHybridDiag = sawHybridDiag || (strat.lastDiagnostic?.contains("Hybrid") == true)
    }

    #expect(sawHybridDiag, "Hybrid diagnostic (prune=) should appear")
    // prune reduces dead and LAS grows; the step stays finite.
    let deadFinal = model.opacities().asArray(Float.self).filter { $0 <= 0.005 }.count
    #expect(deadFinal < 8, "hybrid prune should reduce dead count (was 8, now \(deadFinal))")
    #expect(trainer.gaussianCount > 16, "hybrid should still grow via LAS")
    #expect(trainer.gaussianCount <= 60, "respect capMax")
    #expect(trainer.step(camera: cam, targetImage: target).isFinite)
}

/// Fixed-count MCMC path: `MCMCStrategy.fixedCount()` is built with growthFactor=1.0;
/// it prunes dead gaussians instead of relocating. sampleAdd is not called.
@Test
func mcmcStrategyFixedCountDoesNotGrow() throws {
    let renderer = try GaussianRenderer()
    // Make half dead. A regular MCMCStrategy would grow via sampleAdd, but fixed-count compacts
    // without growing.
    let model = makeMixedModel(live: 10, dead: 10)
    let trainer = try GaussianTrainer(model: model, renderer: renderer)
    var strat = MCMCStrategy.fixedCount(
        noiseLR: 1e4, refineStartIter: 0, refineStopIter: 1000, refineEvery: 5,
        seed: 1
    )
    let cam = makeOpsCamera()
    let target = MLXArray.zeros([cam.height, cam.width, 3])

    let initialCount = trainer.gaussianCount
    #expect(initialCount == 20)
    for step in 0..<20 {
        trainer.step(camera: cam, targetImage: target)
        strat.postStep(store: trainer, step: step, lr: 1.6e-4)
    }
    #expect(trainer.gaussianCount <= initialCount,
        "fixed-count MCMC should NOT grow; got \(trainer.gaussianCount) from \(initialCount)")
    #expect(trainer.gaussianCount < initialCount,
        "fixed-count MCMC should prune dead rows; got \(trainer.gaussianCount)")
    // Still trainable.
    #expect(trainer.step(camera: cam, targetImage: target).isFinite)
}

/// A single high-opacity + huge-scale blob, though not dead by opacity, is relocated by scale_max >
/// median × k (bloatScaleMultiple).
@Test
func relocateMovesBloatedGaussianWhenBloatSuppressionEnabled() throws {
    let renderer = try GaussianRenderer()
    // 6 healthy (opa 0.8, scale 0.1) + 1 blob (opa 0.9, scale 100). All have opa > minOpacity, so
    // none are dead by the opacity criterion.
    let n = 7
    var means: [Float] = []
    for i in 0..<n { means += [Float(i) * 0.1 - 0.3, 0, 4.0] }
    let quats = (0..<n).flatMap { _ -> [Float] in [1, 0, 0, 0] }
    // median scale 0.1 × k=100 = threshold 10; scale 100 exceeds it, so it counts as a blob.
    var logScales: [Float] = []
    for i in 0..<n {
        let s = (i == n - 1) ? Float(100) : Float(0.1)
        logScales += [log(s), log(s), log(s)]
    }
    let logitOp: [Float] = (0..<n).map { i in i == n - 1 ? logit(0.9) : logit(0.8) }
    let model = GaussianModel(
        means: MLXArray(means, [n, 3]),
        quats: MLXArray(quats, [n, 4]),
        logScales: MLXArray(logScales, [n, 3]),
        logitOpacities: MLXArray(logitOp, [n]),
        sh0: MLXArray.zeros([n, 3]),
        shN: MLXArray.zeros([n, GaussianModel.shRestBands, 3])
    )
    let trainer = try GaussianTrainer(model: model, renderer: renderer)
    var rng = SplitMix64(seed: 1)

    // With bloatScaleMultiple=0 (disabled), 0 relocated.
    let nRelocOff = Densification.relocate(
        store: trainer, binoms: Densification.binomialTable(),
        minOpacity: 0.005, bloatScaleMultiple: 0, using: &rng
    )
    #expect(nRelocOff == 0, "OFF path must not relocate (no dead by opa)")

    // With bloatScaleMultiple=100, the 1 blob is relocated.
    let nRelocOn = Densification.relocate(
        store: trainer, binoms: Densification.binomialTable(),
        minOpacity: 0.005, bloatScaleMultiple: 100, using: &rng
    )
    #expect(nRelocOn == 1, "blob suppression must catch the 1 huge-scale gaussian, got \(nRelocOn)")
    #expect(trainer.gaussianCount == n, "relocate must preserve count")

    // After teleport, scale_max drops substantially.
    let scalesAfter = model.scales().asArray(Float.self)
    var maxScaleAfter: Float = 0
    for i in 0..<n {
        let sx = scalesAfter[i * 3 + 0]
        let sy = scalesAfter[i * 3 + 1]
        let sz = scalesAfter[i * 3 + 2]
        maxScaleAfter = Swift.max(maxScaleAfter, Swift.max(sx, Swift.max(sy, sz)))
    }
    #expect(maxScaleAfter < 1.0, "after blob teleport, max scale should be << 100, got \(maxScaleAfter)")

    // Still trainable.
    let cam = makeOpsCamera()
    let tgt = MLXArray.zeros([cam.height, cam.width, 3])
    #expect(trainer.step(camera: cam, targetImage: tgt).isFinite)
}

/// With the fixed-count preset default (bloatScaleMultiple=100), the blob is pruned through the
/// strategy path too.
@Test
func mcmcStrategyFixedCountCatchesBloatViaDefault() throws {
    let renderer = try GaussianRenderer()
    // 10 healthy + 1 blob.
    let n = 11
    var means: [Float] = []
    for i in 0..<n { means += [Float(i) * 0.1 - 0.5, 0, 4.0] }
    let quats = (0..<n).flatMap { _ -> [Float] in [1, 0, 0, 0] }
    var logScales: [Float] = []
    for i in 0..<n {
        let s = (i == n - 1) ? Float(50) : Float(0.1)
        logScales += [log(s), log(s), log(s)]
    }
    let logitOp: [Float] = (0..<n).map { _ in logit(0.7) }   // all healthy opacity
    let model = GaussianModel(
        means: MLXArray(means, [n, 3]),
        quats: MLXArray(quats, [n, 4]),
        logScales: MLXArray(logScales, [n, 3]),
        logitOpacities: MLXArray(logitOp, [n]),
        sh0: MLXArray.zeros([n, 3]),
        shN: MLXArray.zeros([n, GaussianModel.shRestBands, 3])
    )
    let trainer = try GaussianTrainer(model: model, renderer: renderer)
    var strat = MCMCStrategy.fixedCount(
        noiseLR: 0, refineStartIter: 0, refineStopIter: 100, refineEvery: 1,
        seed: 1
    )
    let cam = makeOpsCamera()
    let tgt = MLXArray.zeros([cam.height, cam.width, 3])

    let scaleBefore = model.scales().asArray(Float.self).max() ?? 0
    #expect(scaleBefore > 40, "initial scale_max should be ~50")

    // Run one fire (refineEvery=1, step=1).
    trainer.step(camera: cam, targetImage: tgt)
    strat.postStep(store: trainer, step: 1, lr: 1.6e-4)
    let scaleAfter = model.scales().asArray(Float.self).max() ?? 0
    #expect(scaleAfter < 1.0,
        "fixed-count MCMC default (bloatScaleMultiple=100) should prune the 50x blob, got max=\(scaleAfter)")
    #expect(trainer.gaussianCount == n - 1, "blob row should be removed")
}

@Test
func opacityRegLowersMeanOpacity() throws {
    let renderer = try GaussianRenderer()
    let model = makeMixedModel(live: 6, dead: 0)   // starts at opacity 0.8
    let cam = makeOpsCamera()
    // Use its own render as the target so the photometric gradient is ~0 and only the opacity
    // regularizer's effect shows.
    let (target, _, _) = renderer.forward(model: model, camera: cam)
    eval(target)

    let trainer = try GaussianTrainer(model: model, renderer: renderer, lambdaSSIM: 0, opacityReg: 10.0)
    let opBefore = model.opacities().mean().item(Float.self)
    for _ in 0..<15 { trainer.step(camera: cam, targetImage: target) }
    let opAfter = model.opacities().mean().item(Float.self)
    #expect(opAfter < opBefore, "opacity regulariser should lower mean opacity: \(opBefore) -> \(opAfter)")
}

/// Hybrid repacks the edge scores into the rows kept after prune removes rows (if the scores were
/// dropped on a length mismatch, the threshold would reject every candidate and split=0).
@Test
func hybridKeepsEdgeScoresAlignedAfterPrune() throws {
    let renderer = try GaussianRenderer()
    let model = makeMixedModel(live: 16, dead: 8)
    let trainer = try GaussianTrainer(model: model, renderer: renderer)
    var strat = IGSPlusStrategy(
        capMax: 100, refineStartIter: 0, refineStopIter: 100, refineEvery: 1,
        growthFactor: 1.5, minOpacity: 0.005, minImportance: 1, warmupDensifySteps: 0,
        pruneDeadGaussians: true)

    strat.setImportanceScores([Float](repeating: 10, count: 24))
    strat.postStep(store: trainer, step: 0, lr: 1.6e-4)

    // maxPruneFraction 0.25 prunes 6 → 18, LAS ×1.5 → 27.
    let diag = try #require(strat.lastDiagnostic)
    #expect(diag.contains("prune=6 split=9"), "\(diag)")
    #expect(trainer.gaussianCount == 27, "\(diag)")
}

/// Bound after an Adam step: scale ≤ maxScale. Holds even when densify changes the row count.
@Test
func trainerCapsScaleAfterStep() throws {
    let renderer = try GaussianRenderer()
    // Scale 0.1, above the cap.
    let model = makeMixedModel(live: 5, dead: 0)
    let trainer = try GaussianTrainer(model: model, renderer: renderer, maxScale: 0.05)
    let cam = makeOpsCamera()
    let target = MLXArray.zeros([cam.height, cam.width, 3])

    trainer.step(camera: cam, targetImage: target)
    #expect(model.scales().max().item(Float.self) <= 0.05 * 1.0001)

    #expect(Densification.longAxisSplit(store: trainer, count: 2) == 2)
    trainer.step(camera: cam, targetImage: target)
    #expect(trainer.gaussianCount == 7)
    #expect(model.scales().max().item(Float.self) <= 0.05 * 1.0001)
}

/// Noise injection's Σ·v (computed without building Σ = R diag(s²) Rᵀ) matches building Σ on the
/// CPU and multiplying.
@Test
func covarianceTimesMatchesExplicitCovariance() {
    let n = 257
    var lcg = TestLCG(seed: 11)
    var quats: [Float] = []
    var scales: [Float] = []
    var vectors: [Float] = []
    for _ in 0..<n {
        for _ in 0..<4 { quats.append(lcg.next() * 2 - 1) }          // unnormalized quaternion
        for _ in 0..<3 { scales.append(0.01 * powf(100, lcg.next())) }  // 0.01-1
        for _ in 0..<3 { vectors.append(lcg.next() * 2 - 1) }
    }

    let got = Densification.covarianceTimes(
        quats: MLXArray(quats, [n, 4]), scales: MLXArray(scales, [n, 3]), vectors: MLXArray(vectors, [n, 3])
    ).asArray(Float.self)

    var maxRelErr = 0.0
    for i in 0..<n {
        let q = (0..<4).map { Double(quats[i * 4 + $0]) }
        let norm = (q.map { $0 * $0 }.reduce(0, +)).squareRoot()
        let (w, x, y, z) = (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm)
        let r: [[Double]] = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
        let s2 = (0..<3).map { Double(scales[i * 3 + $0]) * Double(scales[i * 3 + $0]) }
        let v = (0..<3).map { Double(vectors[i * 3 + $0]) }
        var sigmaV = [0.0, 0.0, 0.0]
        for a in 0..<3 {
            for b in 0..<3 {
                var sigmaAB = 0.0
                for k in 0..<3 { sigmaAB += r[a][k] * s2[k] * r[b][k] }
                sigmaV[a] += sigmaAB * v[b]
            }
        }
        let scale = max(1e-6, sigmaV.map(abs).max() ?? 0)
        for a in 0..<3 {
            maxRelErr = max(maxRelErr, abs(Double(got[i * 3 + a]) - sigmaV[a]) / scale)
        }
    }
    #expect(maxRelErr < 1e-4, "max relative error \(maxRelErr)")
}
