// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Testing
@testable import Capture

/// `CaptureTraining.recommendedIterations(frameCount:)`: 8 iter/frame, [500, 1350] clamp, rounded
/// to multiples of 50.
@Test
func recommendedIterationsScalesWithFrameCount() {
    #expect(CaptureTraining.recommendedIterations(frameCount: 117) == 950)
    #expect(CaptureTraining.recommendedIterations(frameCount: 100) == 800)
    #expect(CaptureTraining.recommendedIterations(frameCount: 120) == 950)
}

@Test
func recommendedIterationsClampsBothEnds() {
    #expect(CaptureTraining.recommendedIterations(frameCount: 50) == 500)
    #expect(CaptureTraining.recommendedIterations(frameCount: 0) == 500)
    #expect(CaptureTraining.recommendedIterations(frameCount: 10) == 500)
    #expect(CaptureTraining.recommendedIterations(frameCount: 200) == 1350)
    #expect(CaptureTraining.recommendedIterations(frameCount: 500) == 1350)
}

@Test
func recommendedIterationsIsMultipleOf50() {
    for n in stride(from: 40, through: 220, by: 7) {
        let iters = CaptureTraining.recommendedIterations(frameCount: n)
        #expect(iters % 50 == 0, "frameCount \(n) → \(iters) not a multiple of 50")
        #expect(iters >= 500 && iters <= 1350, "frameCount \(n) → \(iters) out of clamp range")
    }
}

/// MVS mode: 4 iter/frame, [500, 2400] clamp.
@Test
func recommendedIterationsForMVSMode() {
    #expect(CaptureTraining.recommendedIterations(frameCount: 300, mvs: true) == 1200)
    #expect(CaptureTraining.recommendedIterations(frameCount: 100, mvs: true) == 500)
    #expect(CaptureTraining.recommendedIterations(frameCount: 200, mvs: true) == 800)
    #expect(CaptureTraining.recommendedIterations(frameCount: 500, mvs: true) == 2000)
    #expect(CaptureTraining.recommendedIterations(frameCount: 600, mvs: true) == 2400)
    #expect(CaptureTraining.recommendedIterations(frameCount: 800, mvs: true) == 2400)
}

/// Training view cap: 0.4 view/frame, at least 120.
@Test
func recommendedTrainingViewCapScalesWithFrameCount() {
    #expect(CaptureTraining.recommendedTrainingViewCap(frameCount: 300) == 120)
    #expect(CaptureTraining.recommendedTrainingViewCap(frameCount: 600) == 240)
    #expect(CaptureTraining.recommendedTrainingViewCap(frameCount: 450) == 180)
    #expect(CaptureTraining.recommendedTrainingViewCap(frameCount: 100) == 120)
    #expect(CaptureTraining.recommendedTrainingViewCap(frameCount: 0) == 120)
}
