// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import simd
import Testing
@testable import Capture

private func qualityViewmatAt(_ x: Float) -> [Float] {
    [
        1, 0, 0, -x,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
}

private func qualityFrame(index: Int, pixels: [Float], width: Int, height: Int) -> CapturedFrame {
    CapturedFrame(
        image: MLXArray(pixels, [height, width, 3]),
        viewmat: qualityViewmatAt(Float(index) * 0.05),
        intrinsics: (fx: 100, fy: 100, cx: Float(width) / 2, cy: Float(height) / 2),
        width: width, height: height,
        timestamp: TimeInterval(index)
    )
}

private func checkerPixels(width: Int, height: Int) -> [Float] {
    var pixels: [Float] = []
    pixels.reserveCapacity(width * height * 3)
    for y in 0..<height {
        for x in 0..<width {
            let v: Float = ((x + y) % 2 == 0) ? 0.75 : 0.25
            pixels += [v, v, v]
        }
    }
    return pixels
}

private func flatPixels(width: Int, height: Int, value: Float) -> [Float] {
    [Float](repeating: value, count: width * height * 3)
}

/// Checker with variable contrast amplitude. sharpnessScore (mean of neighbor differences) is
/// linear in amp, so sharpness at amp=0.125 is half that at 0.25.
private func checkerPixelsAmp(width: Int, height: Int, amp: Float) -> [Float] {
    var pixels: [Float] = []
    pixels.reserveCapacity(width * height * 3)
    for y in 0..<height {
        for x in 0..<width {
            let v: Float = ((x + y) % 2 == 0) ? (0.5 + amp) : (0.5 - amp)
            pixels += [v, v, v]
        }
    }
    return pixels
}

@Test
func frameQualityFilterDropsSharpnessOutlier() {
    let w = 16, h = 16
    var frames: [CapturedFrame] = []
    for i in 0..<12 {
        let pixels = i == 5
            ? flatPixels(width: w, height: h, value: 0.5)
            : checkerPixels(width: w, height: h)
        frames.append(qualityFrame(index: i, pixels: pixels, width: w, height: h))
    }
    let scene = CapturedScene(frames: frames, sparsePoints: [])
    let params = CaptureFrameQualityFilter.Parameters(
        minFrameCountForFiltering: 4,
        minKeptFrames: 4,
        minKeepFraction: 0.5,
        sharpnessStride: 1,
        exposureSampleStride: 1,
        minSharpnessMedianRatio: 0.35
    )
    var logs: [String] = []
    let filtered = CaptureFrameQualityFilter.filter(scene: scene, parameters: params) {
        logs.append($0)
    }
    #expect(filtered.frames.count == 11)
    #expect(logs.contains { $0.contains("blur=1") })
    #expect(logs.contains { $0.contains("drop frame[5]") })
}

/// `review()` returns the same keep set as `filter()` (the images shown in the UI match those used
/// for training) and a per-frame drop reason.
@Test
func frameQualityReviewMatchesFilter() {
    let w = 16, h = 16
    var frames: [CapturedFrame] = []
    for i in 0..<12 {
        let pixels = i == 5
            ? flatPixels(width: w, height: h, value: 0.5)   // frame 5 = blur outlier
            : checkerPixels(width: w, height: h)
        frames.append(qualityFrame(index: i, pixels: pixels, width: w, height: h))
    }
    let scene = CapturedScene(frames: frames, sparsePoints: [])
    let params = CaptureFrameQualityFilter.Parameters(
        minFrameCountForFiltering: 4,
        minKeptFrames: 4,
        minKeepFraction: 0.5,
        sharpnessStride: 1,
        exposureSampleStride: 1,
        minSharpnessMedianRatio: 0.35
    )

    let reviews = CaptureFrameQualityFilter.review(scene: scene, parameters: params)
    let filtered = CaptureFrameQualityFilter.filter(scene: scene, parameters: params) { _ in }

    // review returns an entry for every frame.
    #expect(reviews.count == 12)
    // review's kept set == the frames filter actually kept (matched by timestamp=index).
    let reviewKept = Set(reviews.filter(\.kept).map(\.index))
    let filterKept = Set(filtered.frames.map { Int($0.timestamp) })
    #expect(reviewKept == filterKept, "review kept \(reviewKept.sorted()) != filter kept \(filterKept.sorted())")
    // frame 5 is dropped for blur, with a reason.
    let f5 = reviews.first { $0.index == 5 }
    #expect(f5?.kept == false, "frame 5 should be dropped")
    #expect(f5?.reasons.contains("blur") == true, "frame 5 drop reason should be blur")
}

/// Absolute blur gate (normalized sharpness floor). If everything is uniformly sharp, the relative
/// check drops nothing, but with the floor above the checker's normSharp (~2.0) every frame is
/// judged blurred and reduced to minKeep (catches uniform blur that slips past the relative check).
@Test
func frameQualityAbsoluteGateDropsLowNormalizedSharpness() {
    let w = 16, h = 16
    var frames: [CapturedFrame] = []
    for i in 0..<12 {
        frames.append(qualityFrame(index: i, pixels: checkerPixels(width: w, height: h),
                                   width: w, height: h))
    }
    let scene = CapturedScene(frames: frames, sparsePoints: [])
    let base = CaptureFrameQualityFilter.Parameters(
        minFrameCountForFiltering: 4, minKeptFrames: 4, minKeepFraction: 0.5,
        sharpnessStride: 1, exposureSampleStride: 1, minSharpnessMedianRatio: 0.35)

    // gate off (floor=0): uniformly sharp, so the relative check keeps everything.
    let off = CaptureFrameQualityFilter.review(scene: scene, parameters: base)
    let keptOff = off.filter { $0.kept }.count
    #expect(keptOff == 12, "uniform sharp frames all kept with absolute gate off (kept=\(keptOff))")

    // gate on (floor well above the actual normSharp): the absolute gate flags every frame → down
    // to minKeep.
    var on = base
    on.minNormalizedSharpness = 100.0
    let onR = CaptureFrameQualityFilter.review(scene: scene, parameters: on)
    let keptOn = onR.filter { $0.kept }.count
    #expect(keptOn < 12, "absolute gate drops uniformly-low-normSharp frames (kept=\(keptOn))")
    #expect(keptOn >= 4, "minKept floor still respected (kept=\(keptOn))")
}

@Test
func frameQualityFilterDropsLocalBlurOutlier() {
    // frame 6 has half the sharpness of its neighbors (amp 0.125 vs 0.25). Its ratio of 0.5 to the
    // global median passes the 0.35 threshold, but the local window gate (ratio 0.6) drops it.
    // blurScore's rescue and absolute gate are turned off to test the relative gate alone (by
    // default blurScore rescues it: frameQualityRescuesLowContrastSharpFrame).
    let w = 16, h = 16
    var frames: [CapturedFrame] = []
    for i in 0..<12 {
        let amp: Float = (i == 6) ? 0.125 : 0.25
        frames.append(qualityFrame(
            index: i,
            pixels: checkerPixelsAmp(width: w, height: h, amp: amp),
            width: w, height: h
        ))
    }
    let scene = CapturedScene(frames: frames, sparsePoints: [])
    let params = CaptureFrameQualityFilter.Parameters(
        minFrameCountForFiltering: 4,
        minKeptFrames: 4,
        minKeepFraction: 0.5,
        sharpnessStride: 1,
        exposureSampleStride: 1,
        minSharpnessMedianRatio: 0.35,
        maxBlurScore: 0,             // Absolute blur gate disabled (relative only)
        blurScoreSharpCeiling: 0,    // Rescue disabled (relative only)
        localBlurEnabled: true,
        localBlurWindow: 3,
        localBlurRatio: 0.6,
        localBlurMinNeighbors: 3
    )
    var logs: [String] = []
    let filtered = CaptureFrameQualityFilter.filter(scene: scene, parameters: params) {
        logs.append($0)
    }
    #expect(filtered.frames.count == 11)
    #expect(logs.contains { $0.contains("blur=0 localblur=1") })
    #expect(logs.contains { $0.contains("drop frame[6]") })
}

/// blurScore rescues false drops. A low-contrast (amp 0.125) checker with sharp edges is treated as
/// an outlier by the gradient-based local relative gate, but the edge-width blurScore judges it
/// sharp (< ceiling), so by default it is kept.
@Test
func frameQualityRescuesLowContrastSharpFrame() {
    let w = 16, h = 16
    var frames: [CapturedFrame] = []
    for i in 0..<12 {
        let amp: Float = (i == 6) ? 0.125 : 0.25
        frames.append(qualityFrame(
            index: i,
            pixels: checkerPixelsAmp(width: w, height: h, amp: amp),
            width: w, height: h
        ))
    }
    let scene = CapturedScene(frames: frames, sparsePoints: [])
    // Apply the default rescue (blurScoreSharpCeiling), with the localblur gate ON.
    let params = CaptureFrameQualityFilter.Parameters(
        minFrameCountForFiltering: 4,
        minKeptFrames: 4,
        minKeepFraction: 0.5,
        sharpnessStride: 1,
        exposureSampleStride: 1,
        minSharpnessMedianRatio: 0.35,
        localBlurEnabled: true,
        localBlurWindow: 3,
        localBlurRatio: 0.6,
        localBlurMinNeighbors: 3
    )
    let reviews = CaptureFrameQualityFilter.review(scene: scene, parameters: params)
    let f6 = reviews.first { $0.index == 6 }
    #expect(f6?.kept == true, "low-contrast but sharp frame 6 should be rescued by blurScore")
    #expect(reviews.filter(\.kept).count == 12, "all frames kept (no genuine blur)")
}

/// blurScore absolute gate. A flat (no high-frequency) frame mixed in with sharp frames is removed
/// by blurScore > maxBlurScore even with the gradient relative gates off.
@Test
func frameQualityAbsoluteBlurGateDropsBlurryFrame() {
    let w = 16, h = 16
    var frames: [CapturedFrame] = []
    for i in 0..<12 {
        // frame 4 = flat (no high frequencies → blurScore=1); the others are sharp checkers
        // (blurScore~0.2).
        let pixels = i == 4
            ? flatPixels(width: w, height: h, value: 0.5)
            : checkerPixels(width: w, height: h)
        frames.append(qualityFrame(index: i, pixels: pixels, width: w, height: h))
    }
    let scene = CapturedScene(frames: frames, sparsePoints: [])
    // Turn off the relative gates and test the absolute blur gate alone.
    let params = CaptureFrameQualityFilter.Parameters(
        minFrameCountForFiltering: 4,
        minKeptFrames: 4,
        minKeepFraction: 0.5,
        sharpnessStride: 1,
        exposureSampleStride: 1,
        minSharpnessMedianRatio: 0,   // global relative gate disabled
        maxBlurScore: 0.62,
        localBlurEnabled: false
    )
    var logs: [String] = []
    let filtered = CaptureFrameQualityFilter.filter(scene: scene, parameters: params) {
        logs.append($0)
    }
    #expect(filtered.frames.count == 11, "flat blurry frame dropped by absolute gate")
    #expect(logs.contains { $0.contains("drop frame[4]") })
}

/// The edge-width blurScore clearly separates sharp (checker / sharp step) from flat and wide-step
/// (blurred) images (the core of motion blur detection).
@Test
func edgeWidthBlurScoreSeparatesSharpFromBlurry() {
    let w = 48, h = 48
    func img(_ px: [Float]) -> MLXArray { MLXArray(px, [h, w, 3]) }

    // A **sharp step** with dark/bright vertical halves (1px transition): sharp.
    func sharpStepPx() -> [Float] {
        var p: [Float] = []
        for _ in 0..<h { for x in 0..<w { let v: Float = x < w / 2 ? 0.2 : 0.8; p += [v, v, v] } }
        return p
    }
    // A **wide step** with the middle ~16px joined by a ramp (blurred transition): blurred.
    func blurryStepPx(rampWidth: Int) -> [Float] {
        var p: [Float] = []
        let lo = w / 2 - rampWidth / 2
        for _ in 0..<h {
            for x in 0..<w {
                let t = Float(x - lo) / Float(rampWidth)
                let v: Float = x < lo ? 0.2 : (x >= lo + rampWidth ? 0.8 : 0.2 + 0.6 * t)
                p += [v, v, v]
            }
        }
        return p
    }

    let bChecker = CaptureFrameQualityFilter.edgeWidthBlurScore(
        rgb: img(checkerPixels(width: w, height: h)), lineStride: 1)
    let bFlat = CaptureFrameQualityFilter.edgeWidthBlurScore(
        rgb: img(flatPixels(width: w, height: h, value: 0.5)), lineStride: 1)
    let bSharpStep = CaptureFrameQualityFilter.edgeWidthBlurScore(
        rgb: img(sharpStepPx()), lineStride: 1)
    let bBlurryStep = CaptureFrameQualityFilter.edgeWidthBlurScore(
        rgb: img(blurryStepPx(rampWidth: 16)), lineStride: 1)

    #expect(bChecker < 0.40, "sharp checker → low blurScore (got \(bChecker))")
    #expect(bSharpStep < 0.40, "sharp 1px step → low blurScore (got \(bSharpStep))")
    #expect(bFlat > 0.9, "featureless flat → blurScore≈1 (got \(bFlat))")
    #expect(bBlurryStep > 0.62, "wide blurred edge → high blurScore (got \(bBlurryStep))")
    #expect(bBlurryStep - bSharpStep > 0.25,
            "clear separation sharp vs blurry (\(bSharpStep) vs \(bBlurryStep))")
}

@Test
func frameQualityFilterLocalBlurDisabledKeepsAll() {
    // Same scene as above, but with the local-blur gate off nothing is dropped.
    let w = 16, h = 16
    var frames: [CapturedFrame] = []
    for i in 0..<12 {
        let amp: Float = (i == 6) ? 0.125 : 0.25
        frames.append(qualityFrame(
            index: i,
            pixels: checkerPixelsAmp(width: w, height: h, amp: amp),
            width: w, height: h
        ))
    }
    let scene = CapturedScene(frames: frames, sparsePoints: [])
    let params = CaptureFrameQualityFilter.Parameters(
        minFrameCountForFiltering: 4,
        minKeptFrames: 4,
        minKeepFraction: 0.5,
        sharpnessStride: 1,
        exposureSampleStride: 1,
        minSharpnessMedianRatio: 0.35,
        localBlurEnabled: false
    )
    var logs: [String] = []
    let filtered = CaptureFrameQualityFilter.filter(scene: scene, parameters: params) {
        logs.append($0)
    }
    #expect(filtered.frames.count == 12)
    #expect(logs.contains { $0.contains("kept 12/12") })
}

@Test
func frameQualityFilterDropsExposureFailure() {
    let w = 12, h = 12
    var frames: [CapturedFrame] = []
    for i in 0..<10 {
        let pixels = i == 3
            ? flatPixels(width: w, height: h, value: 1.0)
            : checkerPixels(width: w, height: h)
        frames.append(qualityFrame(index: i, pixels: pixels, width: w, height: h))
    }
    let scene = CapturedScene(frames: frames, sparsePoints: [])
    let params = CaptureFrameQualityFilter.Parameters(
        minFrameCountForFiltering: 4,
        minKeptFrames: 4,
        minKeepFraction: 0.5,
        sharpnessStride: 1,
        exposureSampleStride: 1,
        maxClippedLumaFraction: 0.8
    )
    var logs: [String] = []
    let filtered = CaptureFrameQualityFilter.filter(scene: scene, parameters: params) {
        logs.append($0)
    }
    #expect(filtered.frames.count == 9)
    #expect(logs.contains { $0.contains("exposure=1") })
    #expect(logs.contains { $0.contains("drop frame[3]") })
}

@Test
func frameQualityFilterSkipsSmallScenes() {
    let w = 8, h = 8
    let frames = (0..<3).map {
        qualityFrame(
            index: $0,
            pixels: flatPixels(width: w, height: h, value: 1.0),
            width: w,
            height: h
        )
    }
    let scene = CapturedScene(frames: frames, sparsePoints: [])
    var logs: [String] = []
    let filtered = CaptureFrameQualityFilter.filter(scene: scene) {
        logs.append($0)
    }
    #expect(filtered.frames.count == 3)
    #expect(logs.contains { $0.contains("skipped") })
}
