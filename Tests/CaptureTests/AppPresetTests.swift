// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Testing
import Trainer
@testable import Capture

/// `CaptureTraining.AppPreset`: the app's Hybrid schedule (start 1/5, stop 3/4, 16 fires, at least
/// 25 iterations apart).
@Test
func appPresetScheduleMatchesTheApp() {
    // (iterations, start, stop, every, fires)
    let cases = [
        (200, 100, 150, 25, 2),
        (500, 100, 375, 25, 11),
        (1200, 240, 900, 41, 16),
        (2400, 480, 1800, 82, 16),
        (5000, 1000, 3750, 171, 16),
    ]
    for (iterations, start, stop, every, fires) in cases {
        let s = CaptureTraining.AppPreset.schedule(iterations: iterations)
        #expect(s.refineStartIter == start, "\(iterations) iters")
        #expect(s.refineStopIter == stop, "\(iterations) iters")
        #expect(s.refineEvery == every, "\(iterations) iters")
        #expect(s.approximateFireCount == fires, "\(iterations) iters")
    }
}

@Test
func appPresetStrategyIsHybrid() {
    let strategy = CaptureTraining.AppPreset.strategy(iterations: 1200, cap: 300_000)
    #expect(strategy.capMax == 300_000)
    #expect(strategy.refineStartIter == 240)
    #expect(strategy.refineStopIter == 900)
    #expect(strategy.refineEvery == 41)
    #expect(strategy.minOpacity == 0.15)
    #expect(strategy.pruneDeadGaussians)
    #expect(strategy.bloatScaleMultiple == 100)
    #expect(strategy.faintPruneScaleMultiple == 1)
}
