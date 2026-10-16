// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Testing
import Trainer
@testable import Capture

/// `CaptureTraining.AppPreset`: the app's Hybrid schedule (start 1/5, stop 3/4, 8-16 fires).
@Test
func appPresetScheduleMatchesTheApp() {
    // (iterations, start, stop, every, fires)
    let cases = [
        (200, 100, 150, 25, 2),
        (500, 100, 375, 34, 8),
        (1200, 240, 900, 82, 8),
        (2400, 480, 1800, 132, 10),
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
    #expect(strategy.refineEvery == 82)
    #expect(strategy.minOpacity == 0.005)
    #expect(strategy.pruneDeadGaussians)
    #expect(strategy.bloatScaleMultiple == 100)
}
