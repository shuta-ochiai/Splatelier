// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Testing
@testable import Capture

private func realtimeCheckerBytes(width: Int, height: Int) -> [UInt8] {
    var bytes: [UInt8] = []
    bytes.reserveCapacity(width * height * 3)
    for y in 0..<height {
        for x in 0..<width {
            let v: UInt8 = ((x + y) % 2 == 0) ? 192 : 64
            bytes += [v, v, v]
        }
    }
    return bytes
}

private func realtimeFlatBytes(width: Int, height: Int, value: UInt8) -> [UInt8] {
    [UInt8](repeating: value, count: width * height * 3)
}

@Test
func realtimeSelectorRejectsBlurAfterWarmup() {
    var selector = CaptureRealtimeFrameSelector(parameters: .init(
        warmupAcceptedFrames: 2,
        sharpnessStride: 1,
        exposureSampleStride: 1,
        minSharpnessMedianRatio: 0.4
    ))
    let sharp = realtimeCheckerBytes(width: 16, height: 16)
    let blur = realtimeFlatBytes(width: 16, height: 16, value: 128)

    #expect(selector.evaluate(rgbBytes: sharp, width: 16, height: 16).accepted)
    #expect(selector.evaluate(rgbBytes: sharp, width: 16, height: 16).accepted)
    let decision = selector.evaluate(rgbBytes: blur, width: 16, height: 16)
    #expect(!decision.accepted)
    #expect(decision.reasons.contains(.blur))
}

@Test
func realtimeSelectorRejectsExposureFailureWithoutWarmup() {
    var selector = CaptureRealtimeFrameSelector(parameters: .init(
        warmupAcceptedFrames: 4,
        exposureSampleStride: 1,
        maxClippedLumaFraction: 0.5
    ))
    let overexposed = realtimeFlatBytes(width: 8, height: 8, value: 255)
    let decision = selector.evaluate(rgbBytes: overexposed, width: 8, height: 8)
    #expect(!decision.accepted)
    #expect(decision.reasons.contains(.exposure))
}

@Test
func realtimeSelectorDisabledAcceptsEverything() {
    var selector = CaptureRealtimeFrameSelector(parameters: .init(enabled: false))
    let overexposed = realtimeFlatBytes(width: 8, height: 8, value: 255)
    let decision = selector.evaluate(rgbBytes: overexposed, width: 8, height: 8)
    #expect(decision.accepted)
    #expect(decision.reasons.isEmpty)
}
