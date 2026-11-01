// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import Testing
import simd
@testable import Capture

private func pose(x: Float = 0, yaw: Float = 0) -> simd_float4x4 {
    var m = simd_float4x4(simd_quatf(angle: yaw, axis: SIMD3<Float>(0, 1, 0)))
    m.columns.3 = SIMD4<Float>(x, 0, 0, 1)
    return m
}

/// Feed poses at 30 fps for `seconds`, starting at `start`. Returns the last result.
private func feed(_ detector: inout CaptureStillnessDetector, from start: TimeInterval,
                  seconds: TimeInterval, pose makePose: (TimeInterval) -> simd_float4x4) -> Bool {
    var still = false
    var t = start
    while t <= start + seconds {
        still = detector.update(cameraToWorld: makePose(t), timestamp: t)
        t += 1.0 / 30.0
    }
    return still
}

@Test
func stillnessDetectorFlagsStandingStill() {
    var detector = CaptureStillnessDetector()
    let early = feed(&detector, from: 0, seconds: 1.5) { _ in pose() }
    let later = feed(&detector, from: 1.6, seconds: 0.6) { _ in pose() }
    #expect(!early)
    #expect(later)
}

@Test
func stillnessDetectorIgnoresHandShake() {
    // Sub-centimetre jitter does not count as moving.
    var detector = CaptureStillnessDetector()
    let still = feed(&detector, from: 0, seconds: 3) { t in pose(x: 0.005 * Float(sin(t * 20))) }
    #expect(still)
}

@Test
func stillnessDetectorAcceptsWalking() {
    // 10 cm/s: well above the threshold.
    var detector = CaptureStillnessDetector()
    let still = feed(&detector, from: 0, seconds: 5) { t in pose(x: 0.1 * Float(t)) }
    #expect(!still)
}

@Test
func stillnessDetectorAcceptsTurning() {
    // 10°/s in place counts as moving, like the frame gate.
    var detector = CaptureStillnessDetector()
    let still = feed(&detector, from: 0, seconds: 5) { t in pose(yaw: 0.17 * Float(t)) }
    #expect(!still)
}

@Test
func stillnessDetectorClearsWhenMovingAgain() {
    var detector = CaptureStillnessDetector()
    let before = feed(&detector, from: 0, seconds: 3) { _ in pose() }
    let after = detector.update(cameraToWorld: pose(x: 0.05), timestamp: 3.1)
    #expect(before)
    #expect(!after)
}

@Test
func stillnessDetectorResetStartsOver() {
    var detector = CaptureStillnessDetector()
    let before = feed(&detector, from: 0, seconds: 3) { _ in pose() }
    detector.reset()
    let first = detector.update(cameraToWorld: pose(), timestamp: 3.1)
    let second = detector.update(cameraToWorld: pose(), timestamp: 4.0)
    #expect(before)
    #expect(!first)
    #expect(!second)
}
