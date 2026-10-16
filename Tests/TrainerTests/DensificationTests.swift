// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import Testing
@testable import Trainer

// Numeric tests for densification (no GPU needed). The relocation opacity/scale correction is
// gsplat Eq.9.

@Test
func binomialTableMatchesKnownValues() {
    let nMax = Densification.nMax
    let b = Densification.binomialTable(nMax: nMax)
    #expect(b[0 * nMax + 0] == 1)      // C(0,0)
    #expect(b[4 * nMax + 2] == 6)      // C(4,2)
    #expect(b[5 * nMax + 0] == 1)      // C(5,0)
    #expect(b[5 * nMax + 5] == 1)      // C(5,5)
    #expect(b[10 * nMax + 3] == 120)   // C(10,3)
    #expect(b[6 * nMax + 3] == 20)     // C(6,3)
}

@Test
func computeRelocationIsIdentityForRatioOne() {
    let b = Densification.binomialTable()
    let scales: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    let (op, sc) = Densification.computeRelocation(
        opacities: [0.3, 0.7], scalesFlat: scales, ratios: [1, 1], binoms: b
    )
    #expect(abs(op[0] - 0.3) < 1e-5)
    #expect(abs(op[1] - 0.7) < 1e-5)
    for i in 0..<6 {
        #expect(abs(sc[i] - scales[i]) < 1e-5, "scale[\(i)] changed under ratio-1")
    }
}

@Test
func computeRelocationSplitLowersOpacityAndScale() {
    let b = Densification.binomialTable()
    let op0: Float = 0.6
    let s0: Float = 0.5
    let (op, sc) = Densification.computeRelocation(
        opacities: [op0], scalesFlat: [s0, s0, s0], ratios: [2], binoms: b
    )

    // o_new = 1 - sqrt(1 - o0)
    let newOp = 1 - (1 - op0).squareRoot()
    #expect(abs(op[0] - newOp) < 1e-5)
    #expect(op[0] < op0, "splitting should lower per-copy opacity")

    // denom (n=2) = 2·o_new - o_new² / √2 ; coeff = o0 / denom
    let denom = 2 * newOp - newOp * newOp / Float(2).squareRoot()
    let coeff = op0 / denom
    #expect(abs(sc[0] - coeff * s0) < 1e-4)
    #expect(sc[0] < s0, "splitting should shrink scale")
    #expect(sc[0] == sc[1] && sc[1] == sc[2], "isotropic input stays isotropic")
}

@Test
func multinomialSampleRespectsWeights() {
    // All probability mass is on index 2.
    var rng = SplitMix64(seed: 42)
    let onlyTwo = Densification.multinomialSample(weights: [0, 0, 5, 0], count: 50, using: &rng)
    #expect(onlyTwo.allSatisfy { $0 == 2 }, "zero-weight indices were sampled")

    // Zero-weight indices never appear; the heavy index dominates.
    var rng2 = SplitMix64(seed: 7)
    let s = Densification.multinomialSample(weights: [1, 0, 0, 9], count: 2000, using: &rng2)
    #expect(s.allSatisfy { $0 == 0 || $0 == 3 }, "sampled a zero-weight index")
    let heavy = s.filter { $0 == 3 }.count
    #expect(heavy > s.count / 2, "index 3 (weight 0.9) should dominate, got \(heavy)/\(s.count)")
}

@Test
func igsPlusScaleLearningRateUsesPaperSchedule() {
    let strat = IGSPlusStrategy(scaleLRStart: 0.020, scaleLREnd: 0.002)

    let first = strat.scaleLearningRate(step: 0, iterations: 500)
    let middle = strat.scaleLearningRate(step: 249, iterations: 500)
    let last = strat.scaleLearningRate(step: 499, iterations: 500)

    #expect(abs(first - 0.020) < 1e-7)
    #expect(abs(last - 0.002) < 1e-7)
    #expect(middle < first)
    #expect(middle > last)
    #expect(abs(middle - Float(sqrt(0.020 * 0.002))) < 2e-5)
}

@Test
func igsPlusMeansLearningRateUsesPaperBoundaryWithSceneScale() {
    let strat = IGSPlusStrategy(meansLRStart: 0.000128, meansLREnd: 0.0000128)

    let first = strat.meansLearningRate(step: 0, iterations: 500, sceneScale: 2)
    let middle = strat.meansLearningRate(step: 249, iterations: 500, sceneScale: 2)
    let last = strat.meansLearningRate(step: 499, iterations: 500, sceneScale: 2)

    #expect(abs(first - 0.000256) < 1e-9)
    #expect(abs(last - 0.0000256) < 1e-9)
    #expect(middle < first)
    #expect(middle > last)
    #expect(abs(middle - 2 * Float(sqrt(0.000128 * 0.0000128))) < 2e-7)
}

@Test
func shortBudgetSchedulesDensifyInsideTrainingWindow() {
    let mcmc = MCMCStrategy.scaled(iterations: 500, capMax: 6_000)
    #expect(mcmc.capMax == 6_000)
    #expect(mcmc.refineStartIter == 50)
    #expect(mcmc.refineStopIter == 400)
    #expect(mcmc.refineEvery == 10)
    #expect(60 > mcmc.refineStartIter)
    #expect(60 < mcmc.refineStopIter)
    #expect(60 % mcmc.refineEvery == 0)

    let igs = IGSPlusStrategy.scaled(iterations: 500, capMax: 6_000)
    #expect(igs.capMax == 6_000)
    #expect(igs.refineStartIter == 50)
    #expect(igs.refineStopIter == 400)
    #expect(igs.refineEvery == 16)
    #expect(igs.minImportance == 1)
    #expect(64 >= igs.refineStartIter)
    #expect(64 <= igs.refineStopIter)
    #expect(64 % igs.refineEvery == 0)
}

@Test
func edgeImportanceMapDetectsImageBoundary() {
    let width = 16
    let height = 16
    var rgb = [Float](repeating: 0, count: width * height * 3)
    for y in 0..<height {
        for x in 0..<width where x >= width / 2 {
            let i = (y * width + x) * 3
            rgb[i + 0] = 1
            rgb[i + 1] = 1
            rgb[i + 2] = 1
        }
    }

    let map = Densification.edgeImportanceMap(rgb: rgb, width: width, height: height)
    let edge = map.sampleMax(x: 8, y: 8, radius: 2)
    let flat = map.sampleMax(x: 2, y: 8, radius: 1)

    #expect(edge > 0)
    #expect(edge > flat)
}

@Test
func edgeImportanceScoresUseContributingFootprint() {
    let width = 9
    let height = 9
    var scores = [Float](repeating: 0, count: width * height)
    scores[4 * width + 6] = 1
    let edgeMap = Densification.EdgeImportanceMap(width: width, height: height, scores: scores)

    let gaussianScores = Densification.edgeImportanceScores(
        edgeMap: edgeMap,
        means2d: [4, 4],
        radii: [4, 4],
        depths: [1],
        conics: [0.05, 0, 0.05],
        opacities: [1]
    )

    #expect(gaussianScores.count == 1)
    #expect(gaussianScores[0] > 0, "edge inside the projected footprint should count")
}

@Test
func edgeImportanceScoresIgnoreNonContributingFootprint() {
    let width = 9
    let height = 9
    var scores = [Float](repeating: 0, count: width * height)
    scores[4 * width + 4] = 1
    let edgeMap = Densification.EdgeImportanceMap(width: width, height: height, scores: scores)

    let gaussianScores = Densification.edgeImportanceScores(
        edgeMap: edgeMap,
        means2d: [4, 4],
        radii: [4, 4],
        depths: [1],
        conics: [0.05, 0, 0.05],
        opacities: [0.001]
    )

    #expect(gaussianScores == [0], "opacity below the rasterizer alpha threshold should not score")
}

/// Pixel centers in continuous coordinates are at i + 0.5 (same convention as the rasterizer), so
/// (2.5, 2.5) is pixel (2, 2).
@Test
func edgeImportanceSampleUsesPixelCenterConvention() {
    var scores = [Float](repeating: 0, count: 25)
    scores[0] = 0.5
    scores[2 * 5 + 2] = 1
    let map = Densification.EdgeImportanceMap(width: 5, height: 5, scores: scores)
    #expect(map.sampleMax(x: 2.5, y: 2.5, radius: 0) == 1)
    #expect(map.sampleMax(x: 2.99, y: 2.0, radius: 0) == 1)
    #expect(map.sampleMax(x: -0.4, y: 0, radius: 0) == 0)
}

/// In coarse / fine-resample steps means2d is in reduced-resolution pixels, so the full-res edge
/// map is sampled with a scale.
@Test
func edgeImportanceScoresMapStepPixelsToFullResEdgeMap() {
    // Full-res 16x16 with an edge only at pixel (12, 12). A Gaussian at full-res (12.5, 12.5) is at
    // (6.25, 6.25) in a 1/2 step.
    var scores = [Float](repeating: 0, count: 16 * 16)
    scores[12 * 16 + 12] = 1
    let edgeMap = Densification.EdgeImportanceMap(width: 16, height: 16, scores: scores)
    func score(_ scale: Float) -> Float {
        Densification.edgeImportanceScores(
            edgeMap: edgeMap,
            means2d: [6.25, 6.25],
            radii: [1, 1],
            depths: [1],
            conics: [1, 0, 1],
            opacities: [1],
            stepToMapScale: scale
        )[0]
    }
    #expect(score(2) == 1)
    #expect(score(1) == 0, "without the scale it samples near (6, 6) and misses the edge")
}

/// `windowMax(radius:)` gives `sampleMax` with that radius bit for bit, at borders and with
/// negative, -0, NaN and tied scores.
@Test
func edgeImportanceWindowMaxMatchesSampleMax() {
    var rng = TestLCG(seed: 7)
    for (width, height) in [(1, 1), (5, 3), (17, 11)] {
        let scores: [Float] = (0..<(width * height)).map { _ in
            let u = rng.next()
            if u < 0.1 { return -0.0 }
            if u < 0.15 { return .nan }
            if u < 0.3 { return 0.5 }   // ties
            return rng.next() * 2 - 0.5
        }
        let map = Densification.EdgeImportanceMap(width: width, height: height, scores: scores)
        for radius in 0...4 {
            let windowed = map.windowMax(radius: radius)
            for qy in -2...(height + 1) {
                for qx in -2...(width + 1) {
                    let x = Float(qx) + rng.next(), y = Float(qy) + rng.next()
                    let direct = map.sampleMax(x: x, y: y, radius: radius)
                    let lookup = windowed.sampleMax(x: x, y: y, radius: 0)
                    #expect(lookup.bitPattern == direct.bitPattern,
                            "\(width)x\(height) r=\(radius) at (\(x), \(y)): \(lookup) vs \(direct)")
                }
            }
        }
    }
}

/// The edge score samples the window-max map once per footprint sample; it must equal the window
/// search it replaced (`sampleMax` with radius ceil(stepToMapScale)) bit for bit.
@Test
func edgeImportanceScoresMatchWindowSearch() {
    var rng = TestLCG(seed: 21)
    let width = 40, height = 30
    let map = Densification.EdgeImportanceMap(
        width: width, height: height,
        scores: (0..<(width * height)).map { _ in rng.next() < 0.7 ? 0 : rng.next() * 3 })
    let n = 200
    var means2d: [Float] = [], radii: [Int32] = [], depths: [Float] = []
    var conics: [Float] = [], opacities: [Float] = []
    for _ in 0..<n {
        means2d += [rng.next() * 30 - 5, rng.next() * 24 - 4]
        radii += [Int32(rng.next() * 6), Int32(rng.next() * 6)]
        depths.append(rng.next() < 0.1 ? -1 : 1)
        conics += [rng.next() * 0.5 + 0.01, rng.next() * 0.1 - 0.05, rng.next() * 0.5 + 0.01]
        opacities.append(rng.next())
    }
    func offsets(_ radius: Int) -> [Float] {
        let r = max(1, radius)
        if r == 1 { return [-1, 0, 1] }
        let half = max(1, r / 2)
        return [Float(-r), Float(-half), 0, Float(half), Float(r)]
    }
    for scale: Float in [1, 1.5, 2, 3] {
        let window = max(1, Int(scale.rounded(.up)))
        let got = Densification.edgeImportanceScores(
            edgeMap: map, means2d: means2d, radii: radii, depths: depths, conics: conics,
            opacities: opacities, stepToMapScale: scale)
        for i in 0..<n {
            var best: Float = 0
            let a = conics[i * 3], b = conics[i * 3 + 1], c = conics[i * 3 + 2]
            if depths[i] > 0, radii[i * 2] > 0, radii[i * 2 + 1] > 0, opacities[i] > 1 / 255 {
                for dy in offsets(Int(radii[i * 2 + 1])) {
                    for dx in offsets(Int(radii[i * 2])) {
                        let sigma = 0.5 * (a * dx * dx + c * dy * dy) + b * dx * dy
                        guard sigma >= 0, sigma.isFinite else { continue }
                        guard min(0.99, opacities[i] * exp(-sigma)) >= 1 / 255 else { continue }
                        let edge = map.sampleMax(
                            x: (means2d[i * 2] + dx) * scale, y: (means2d[i * 2 + 1] + dy) * scale,
                            radius: window)
                        if edge > best { best = edge }
                    }
                }
            }
            #expect(got[i].bitPattern == best.bitPattern, "scale \(scale) gaussian \(i)")
        }
    }
}

/// Hybrid frees slots by pruning, so it prepares edge scores even at the cap. Plain IGS+ does not
/// split at the cap, so it does not need them.
@Test
func hybridPreparesImportanceAtCap() {
    var hybrid = IGSPlusStrategy(capMax: 100, refineStartIter: 10, refineStopIter: 100, refineEvery: 10)
    hybrid.pruneDeadGaussians = true
    let igs = IGSPlusStrategy(capMax: 100, refineStartIter: 10, refineStopIter: 100, refineEvery: 10)
    #expect(hybrid.shouldPrepareImportance(step: 20, gaussianCount: 100))
    #expect(!igs.shouldPrepareImportance(step: 20, gaussianCount: 100))
    #expect(igs.shouldPrepareImportance(step: 20, gaussianCount: 50))
    #expect(!hybrid.shouldPrepareImportance(step: 25, gaussianCount: 100), "not prepared on steps that do not fire")
}

/// MCMC keeps the trainer's constant scale LR unless a schedule is set, then decays start → end.
@Test
func mcmcScaleLRScheduleDecaysOnlyWhenSet() {
    var mcmc = MCMCStrategy.fixedCount()
    #expect(mcmc.scaleLearningRate(step: 0, iterations: 100) == nil)
    mcmc.scaleLRSchedule = (start: 0.02, end: 0.004)
    #expect(abs(mcmc.scaleLearningRate(step: 0, iterations: 100)! - 0.02) < 1e-7)
    #expect(abs(mcmc.scaleLearningRate(step: 99, iterations: 100)! - 0.004) < 1e-7)
}
