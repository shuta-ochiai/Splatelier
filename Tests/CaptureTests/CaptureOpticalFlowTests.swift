// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import Testing
@testable import Capture

private func flowFrame(index: Int, width: Int, height: Int, shiftX: Int, shiftY: Int) -> CapturedFrame {
    var pixels: [Float] = []
    pixels.reserveCapacity(width * height * 3)
    for y in 0..<height {
        for x in 0..<width {
            let sx = x - shiftX
            let sy = y - shiftY
            let raw = (sx * 37 + sy * 19 + sx * sy * 7) & 255
            let v = 0.2 + 0.6 * Float(raw) / 255
            pixels += [v, v, v]
        }
    }
    return CapturedFrame(
        image: MLXArray(pixels, [height, width, 3]),
        viewmat: [
            1, 0, 0, -Float(index) * 0.05,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ],
        intrinsics: (fx: 100, fy: 100, cx: Float(width) / 2, cy: Float(height) / 2),
        width: width, height: height,
        timestamp: TimeInterval(index)
    )
}

private func flatFlowFrame(index: Int, width: Int, height: Int) -> CapturedFrame {
    CapturedFrame(
        image: MLXArray([Float](repeating: 0.5, count: width * height * 3), [height, width, 3]),
        viewmat: [
            1, 0, 0, -Float(index) * 0.05,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ],
        intrinsics: (fx: 100, fy: 100, cx: Float(width) / 2, cy: Float(height) / 2),
        width: width, height: height,
        timestamp: TimeInterval(index)
    )
}

private func flowPair(_ from: Int, _ to: Int, residual: Float) -> CaptureOpticalFlow.PairMetrics {
    CaptureOpticalFlow.PairMetrics(
        fromIndex: from, toIndex: to,
        candidateCount: 100, vectorCount: 80,
        medianDxPixels: 5, medianDyPixels: 0,
        medianMagnitudePixels: 5, p90MagnitudePixels: 8,
        meanResidual: residual
    )
}

private func flowSummary(pairs: [CaptureOpticalFlow.PairMetrics]) -> CaptureOpticalFlow.Summary {
    CaptureOpticalFlow.Summary(
        pairs: pairs,
        medianPairMagnitudePixels: 5,
        medianTextureCoverage: 0.8,
        maxP90MagnitudePixels: 8,
        meanResidual: pairs.map(\.meanResidual).reduce(0, +) / Float(max(1, pairs.count))
    )
}

@Test
func flowFrameWeightsDownweightsHighResidualLink() {
    // Of the 4 neighboring pairs, only 2→3 has a large residual (0.20, others 0.05). Its end frames
    // 2 and 3 drop to the floor; the others stay at 1.0.
    let summary = flowSummary(pairs: [
        flowPair(0, 1, residual: 0.05),
        flowPair(1, 2, residual: 0.05),
        flowPair(2, 3, residual: 0.20),
        flowPair(3, 4, residual: 0.05),
    ])
    var logs: [String] = []
    let weights = CaptureOpticalFlow.frameWeights(
        frameCount: 5, summary: summary, parameters: .init(floor: 0.3)
    ) { logs.append($0) }
    #expect(weights.count == 5)
    #expect(weights[0] == 1.0)
    #expect(weights[1] == 1.0)
    #expect(abs(weights[2] - 0.3) < 1e-5)
    #expect(abs(weights[3] - 0.3) < 1e-5)
    #expect(weights[4] == 1.0)
    #expect(logs.contains { $0.contains("[ViewWeight]") && $0.contains("down-weighted 2/5") })
}

@Test
func flowFrameWeightsDisabledReturnsOnes() {
    let summary = flowSummary(pairs: [
        flowPair(0, 1, residual: 0.05),
        flowPair(1, 2, residual: 0.90),
    ])
    let weights = CaptureOpticalFlow.frameWeights(
        frameCount: 3, summary: summary, parameters: .disabled
    ) { _ in }
    #expect(weights == [1.0, 1.0, 1.0])
}

@Test
func flowFrameWeightsUniformResidualStaysOnes() {
    // With no spread in the residuals nothing can be distinguished, so all are 1.0.
    let summary = flowSummary(pairs: [
        flowPair(0, 1, residual: 0.07),
        flowPair(1, 2, residual: 0.07),
        flowPair(2, 3, residual: 0.07),
    ])
    let weights = CaptureOpticalFlow.frameWeights(
        frameCount: 4, summary: summary, parameters: .init(floor: 0.3)
    ) { _ in }
    #expect(weights == [1.0, 1.0, 1.0, 1.0])
}

@Test
func opticalFlowDetectsSyntheticShift() throws {
    let frames = [
        flowFrame(index: 0, width: 64, height: 48, shiftX: 0, shiftY: 0),
        flowFrame(index: 1, width: 64, height: 48, shiftX: 3, shiftY: -2),
    ]
    var logs: [String] = []
    let summary = CaptureOpticalFlow.analyze(
        frames: frames,
        parameters: .init(
            targetWidth: 64,
            blockRadius: 1,
            blockStride: 4,
            searchRadius: 4,
            minTextureStd: 0.01,
            logPairLimit: 1
        )
    ) { logs.append($0) }
    let pair = try #require(summary?.pairs.first)
    #expect(abs(pair.medianDxPixels - 3) <= 1.1, "dx \(pair.medianDxPixels)")
    #expect(abs(pair.medianDyPixels - (-2)) <= 1.1, "dy \(pair.medianDyPixels)")
    #expect(pair.textureCoverage > 0.5)
    #expect(logs.contains { $0.contains("[OpticalFlow] pairs=1") })
}

@Test
func opticalFlowIgnoresFlatLowTextureBlocks() throws {
    let frames = [
        flatFlowFrame(index: 0, width: 48, height: 48),
        flatFlowFrame(index: 1, width: 48, height: 48),
    ]
    let summary = CaptureOpticalFlow.analyze(
        frames: frames,
        parameters: .init(
            targetWidth: 48,
            blockRadius: 1,
            blockStride: 4,
            searchRadius: 3,
            minTextureStd: 0.01,
            logPairLimit: 0
        )
    ) { _ in }
    let pair = try #require(summary?.pairs.first)
    #expect(pair.vectorCount == 0)
    #expect(pair.textureCoverage == 0)
}
