// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import Trainer

/// Thins captured frames by blur and exposure before BA/MVS/training. The removal rate is capped so
/// low-texture scenes do not run out of frames. `sharpness` (gradient energy, PocketGS eq.1)
/// depends on image content, so it is complemented by the content-invariant `blurScore` (edge-width
/// method): if large, remove via the absolute gate; if small, cancel the relative gate's verdict.
public enum CaptureFrameQualityFilter {

    public struct Parameters: Sendable, Equatable {
        public var enabled: Bool
        public var minFrameCountForFiltering: Int
        public var minKeptFrames: Int
        public var minKeepFraction: Float
        public var sharpnessStride: Int
        public var exposureSampleStride: Int
        public var minSharpnessMedianRatio: Float
        public var minSharpnessAbsolute: Float
        /// Safety-net absolute gate: remove if sharpness / (lumaStd+ε) is below this. 0 disables.
        public var minNormalizedSharpness: Float
        /// Remove if blurScore exceeds this. 0 disables. The default was calibrated visually on
        /// device.
        public var maxBlurScore: Float
        /// Keep a frame flagged only by the relative gates if its blurScore is below this (protects
        /// sharp frames that are merely low-texture). 0 disables. Default calibrated on device.
        public var blurScoreSharpCeiling: Float
        /// Spacing (px) of the scanlines used to measure edge width. The width walk itself always
        /// steps 1px at a time.
        public var blurEdgeStride: Int
        public var localBlurEnabled: Bool
        public var localBlurWindow: Int
        public var localBlurRatio: Float
        public var localBlurMinNeighbors: Int
        public var darkMeanThreshold: Float
        public var brightMeanThreshold: Float
        public var lowClipThreshold: Float
        public var highClipThreshold: Float
        public var maxClippedLumaFraction: Float
        public var logDropLimit: Int

        public init(
            enabled: Bool = true,
            minFrameCountForFiltering: Int = 12,
            minKeptFrames: Int = 12,
            minKeepFraction: Float = 0.65,
            sharpnessStride: Int = 4,
            exposureSampleStride: Int = 4,
            minSharpnessMedianRatio: Float = 0.55,
            minSharpnessAbsolute: Float = 1e-4,
            minNormalizedSharpness: Float = 0,
            maxBlurScore: Float = 0.65,
            blurScoreSharpCeiling: Float = 0.60,
            blurEdgeStride: Int = 2,
            localBlurEnabled: Bool = true,
            localBlurWindow: Int = 6,
            localBlurRatio: Float = 0.80,
            localBlurMinNeighbors: Int = 3,
            darkMeanThreshold: Float = 0.04,
            brightMeanThreshold: Float = 0.96,
            lowClipThreshold: Float = 0.01,
            highClipThreshold: Float = 0.99,
            maxClippedLumaFraction: Float = 0.35,
            logDropLimit: Int = 12
        ) {
            precondition(minFrameCountForFiltering >= 1)
            precondition(minKeptFrames >= 1)
            precondition(minKeepFraction > 0 && minKeepFraction <= 1)
            precondition(sharpnessStride >= 1)
            precondition(exposureSampleStride >= 1)
            precondition(minSharpnessMedianRatio >= 0)
            precondition(minSharpnessAbsolute >= 0)
            precondition(minNormalizedSharpness >= 0)
            precondition(maxBlurScore >= 0 && maxBlurScore <= 1)
            precondition(blurScoreSharpCeiling >= 0 && blurScoreSharpCeiling <= 1)
            precondition(blurEdgeStride >= 1)
            precondition(localBlurWindow >= 1)
            precondition(localBlurRatio >= 0 && localBlurRatio <= 1)
            precondition(localBlurMinNeighbors >= 1)
            precondition(darkMeanThreshold >= 0 && darkMeanThreshold <= 1)
            precondition(brightMeanThreshold >= 0 && brightMeanThreshold <= 1)
            precondition(lowClipThreshold >= 0 && lowClipThreshold <= 1)
            precondition(highClipThreshold >= 0 && highClipThreshold <= 1)
            precondition(maxClippedLumaFraction >= 0 && maxClippedLumaFraction <= 1)
            precondition(logDropLimit >= 0)
            self.enabled = enabled
            self.minFrameCountForFiltering = minFrameCountForFiltering
            self.minKeptFrames = minKeptFrames
            self.minKeepFraction = minKeepFraction
            self.sharpnessStride = sharpnessStride
            self.exposureSampleStride = exposureSampleStride
            self.minSharpnessMedianRatio = minSharpnessMedianRatio
            self.minSharpnessAbsolute = minSharpnessAbsolute
            self.minNormalizedSharpness = minNormalizedSharpness
            self.maxBlurScore = maxBlurScore
            self.blurScoreSharpCeiling = blurScoreSharpCeiling
            self.blurEdgeStride = blurEdgeStride
            self.localBlurEnabled = localBlurEnabled
            self.localBlurWindow = localBlurWindow
            self.localBlurRatio = localBlurRatio
            self.localBlurMinNeighbors = localBlurMinNeighbors
            self.darkMeanThreshold = darkMeanThreshold
            self.brightMeanThreshold = brightMeanThreshold
            self.lowClipThreshold = lowClipThreshold
            self.highClipThreshold = highClipThreshold
            self.maxClippedLumaFraction = maxClippedLumaFraction
            self.logDropLimit = logDropLimit
        }
    }

    public struct Metrics: Sendable {
        public let index: Int
        public let sharpness: Float
        /// Content-invariant blurriness [0,1] (0=sharp, 1=blurry). Edge-width method.
        public let blurScore: Float
        public let lumaMean: Float
        public let lumaStd: Float
        public let lowClipFraction: Float
        public let highClipFraction: Float

        public var clippedLumaFraction: Float {
            Swift.max(lowClipFraction, highClipFraction)
        }
    }

    private struct Decision {
        let metrics: Metrics
        let reasons: [String]
        let score: Float
    }

    public static func metrics(
        frames: [CapturedFrame],
        parameters: Parameters = Parameters()
    ) -> [Metrics] {
        frames.enumerated().map { index, frame in
            let img = frame.imageFloat
            let sharpness = FrameSubsampler.sharpnessScore(
                rgb: img,
                stride: parameters.sharpnessStride
            )
            // Blur metrics and exposure stats share the same host array (one full-res readback).
            let flat = img.asArray(Float.self)
            let (h, w) = (img.shape[0], img.shape[1])
            let blurScore = edgeWidthBlurScore(flat: flat, height: h, width: w, lineStride: parameters.blurEdgeStride)
            let exposure = exposureStats(
                flat: flat, height: h, width: w,
                sampleStride: parameters.exposureSampleStride,
                lowClipThreshold: parameters.lowClipThreshold,
                highClipThreshold: parameters.highClipThreshold
            )
            return Metrics(
                index: index,
                sharpness: sharpness,
                blurScore: blurScore,
                lumaMean: exposure.mean,
                lumaStd: exposure.std,
                lowClipFraction: exposure.lowClipFraction,
                highClipFraction: exposure.highClipFraction
            )
        }
    }

    /// Per-frame verdict (for UI viewing). `kept` matches the set `filter` keeps.
    public struct FrameReview: Sendable, Equatable {
        public let index: Int
        public let kept: Bool
        public let reasons: [String]   // "blur" / "localblur" / "exposure"
        public let sharpness: Float
        /// Shown in the UI and used for threshold calibration.
        public let blurScore: Float

        public init(index: Int, kept: Bool, reasons: [String], sharpness: Float,
                    blurScore: Float = 0) {
            self.index = index
            self.kept = kept
            self.reasons = reasons
            self.sharpness = sharpness
            self.blurScore = blurScore
        }
    }

    /// Returns per-frame keep/drop and reasons using the same verdict as `filter`.
    /// When disabled or with too few frames, all frames are kept.
    public static func review(
        scene: CapturedScene,
        parameters: Parameters = Parameters()
    ) -> [FrameReview] {
        let frames = scene.frames
        guard parameters.enabled,
              frames.count >= parameters.minFrameCountForFiltering,
              let ev = evaluate(frames: frames, parameters: parameters)
        else {
            return metrics(frames: frames, parameters: parameters).map {
                FrameReview(index: $0.index, kept: true, reasons: [],
                            sharpness: $0.sharpness, blurScore: $0.blurScore)
            }
        }
        return ev.decisions.map {
            FrameReview(index: $0.metrics.index,
                        kept: ev.keep.contains($0.metrics.index),
                        reasons: $0.reasons,
                        sharpness: $0.metrics.sharpness,
                        blurScore: $0.metrics.blurScore)
        }
    }

    /// Verdict shared by `review` and `filter` (tests guarantee the keep sets match).
    private struct Evaluation {
        let decisions: [Decision]
        let keep: Set<Int>
        let minKeep: Int
        let proposedDropCount: Int
        let rescuedCount: Int
        let sharpMedian: Float
        let sharpThreshold: Float
        let sharpQ10: Float
        let sharpQ90: Float
        let clipMedian: Float
        let blurMedian: Float
        let sharpByIndex: [Float]
    }

    /// metrics → per-frame verdict → keep set including the minKeep top-up.
    private static func evaluate(
        frames: [CapturedFrame], parameters: Parameters
    ) -> Evaluation? {
        let n = frames.count
        let allMetrics = metrics(frames: frames, parameters: parameters)
        guard !allMetrics.isEmpty else { return nil }

        let sharpSorted = allMetrics.map(\.sharpness).sorted()
        let clipSorted = allMetrics.map(\.clippedLumaFraction).sorted()
        let blurSorted = allMetrics.map(\.blurScore).sorted()
        let sharpMedian = quantile(sorted: sharpSorted, q: 0.5)
        let sharpQ10 = quantile(sorted: sharpSorted, q: 0.1)
        let sharpQ90 = quantile(sorted: sharpSorted, q: 0.9)
        let clipMedian = quantile(sorted: clipSorted, q: 0.5)
        let blurMedian = quantile(sorted: blurSorted, q: 0.5)
        let sharpThreshold = Swift.max(
            parameters.minSharpnessAbsolute,
            sharpMedian * parameters.minSharpnessMedianRatio
        )

        // Ratio to the temporal-neighborhood median. During an orbit lighting and texture change,
        // so the global median is weak; this catches isolated blur between sharp neighbors. In a
        // uniformly soft stretch the neighborhood median also drops, so the whole stretch is never
        // removed.
        let sharpByIndex = allMetrics.map(\.sharpness)
        func localSharpMedian(_ i: Int) -> Float {
            let lo = Swift.max(0, i - parameters.localBlurWindow)
            let hi = Swift.min(n - 1, i + parameters.localBlurWindow)
            var window: [Float] = []
            window.reserveCapacity(hi - lo)
            for j in lo...hi where j != i { window.append(sharpByIndex[j]) }
            guard window.count >= parameters.localBlurMinNeighbors else { return 0 }
            return quantile(sorted: window.sorted(), q: 0.5)
        }

        let decisions = allMetrics.map { m -> Decision in
            var reasons: [String] = []
            // Relative gate: ratio to the global median
            if sharpMedian > parameters.minSharpnessAbsolute,
               m.sharpness < sharpThreshold {
                reasons.append("blur")
            }
            // Relative gate: ratio to the neighborhood median
            if parameters.localBlurEnabled, !reasons.contains("blur") {
                let localMedian = localSharpMedian(m.index)
                if localMedian > parameters.minSharpnessAbsolute,
                   m.sharpness < localMedian * parameters.localBlurRatio {
                    reasons.append("localblur")
                }
            }
            // Rescue: cancel the relative gates' verdict if blurScore is small enough
            if parameters.blurScoreSharpCeiling > 0,
               m.blurScore < parameters.blurScoreSharpCeiling {
                reasons.removeAll { $0 == "blur" || $0 == "localblur" }
            }
            // Absolute gate: catches high-texture blur that slipped past the relative gates.
            // Evaluated after the rescue.
            if parameters.maxBlurScore > 0, !reasons.contains("blur"),
               m.blurScore > parameters.maxBlurScore {
                reasons.append("blur")
            }
            if parameters.minNormalizedSharpness > 0, !reasons.contains("blur") {
                if m.sharpness / (m.lumaStd + 1e-6) < parameters.minNormalizedSharpness {
                    reasons.append("blur")
                }
            }
            if m.lumaMean < parameters.darkMeanThreshold
                || m.lumaMean > parameters.brightMeanThreshold
                || m.clippedLumaFraction > parameters.maxClippedLumaFraction
            {
                reasons.append("exposure")
            }
            // Priority for the minKeep top-up: sharp, structured, well exposed
            let sharpNorm = sharpMedian > 1e-7 ? m.sharpness / sharpMedian : 1
            let exposurePenalty = m.clippedLumaFraction
                / Swift.max(parameters.maxClippedLumaFraction, 1e-6)
            let score = Swift.min(sharpNorm, 3) + (1 - m.blurScore) - exposurePenalty
            return Decision(metrics: m, reasons: reasons, score: score)
        }

        var keep = Set(decisions.filter { $0.reasons.isEmpty }.map { $0.metrics.index })
        let minKeepByFraction = Int(ceil(Float(n) * parameters.minKeepFraction))
        let minKeep = Swift.min(n, Swift.max(parameters.minKeptFrames, minKeepByFraction))
        let proposedDropCount = n - keep.count
        var rescuedCount = 0
        if keep.count < minKeep {
            let candidates = decisions
                .filter { !keep.contains($0.metrics.index) }
                .sorted { a, b in
                    if a.score == b.score { return a.metrics.index < b.metrics.index }
                    return a.score > b.score
                }
            for d in candidates.prefix(minKeep - keep.count) {
                keep.insert(d.metrics.index)
                rescuedCount += 1
            }
        }

        return Evaluation(
            decisions: decisions, keep: keep, minKeep: minKeep,
            proposedDropCount: proposedDropCount, rescuedCount: rescuedCount,
            sharpMedian: sharpMedian, sharpThreshold: sharpThreshold,
            sharpQ10: sharpQ10, sharpQ90: sharpQ90,
            clipMedian: clipMedian, blurMedian: blurMedian,
            sharpByIndex: sharpByIndex
        )
    }

    public static func filter(
        scene: CapturedScene,
        parameters: Parameters = Parameters(),
        log: (String) -> Void
    ) -> CapturedScene {
        guard parameters.enabled else {
            log("[FrameQuality] disabled")
            return scene
        }
        let n = scene.frames.count
        guard n >= parameters.minFrameCountForFiltering else {
            log("[FrameQuality] skipped: only \(n) frames (<\(parameters.minFrameCountForFiltering))")
            return scene
        }
        guard let ev = evaluate(frames: scene.frames, parameters: parameters) else {
            return scene
        }
        let keep = ev.keep
        let decisions = ev.decisions

        func localSharpMedian(_ i: Int) -> Float {
            let lo = Swift.max(0, i - parameters.localBlurWindow)
            let hi = Swift.min(n - 1, i + parameters.localBlurWindow)
            var window: [Float] = []
            window.reserveCapacity(hi - lo)
            for j in lo...hi where j != i { window.append(ev.sharpByIndex[j]) }
            guard window.count >= parameters.localBlurMinNeighbors else { return 0 }
            return quantile(sorted: window.sorted(), q: 0.5)
        }

        let finalDropped = decisions
            .filter { !keep.contains($0.metrics.index) }
            .sorted { $0.metrics.index < $1.metrics.index }
        let blurDrops = finalDropped.filter { $0.reasons.contains("blur") }.count
        let localBlurDrops = finalDropped.filter { $0.reasons.contains("localblur") }.count
        let exposureDrops = finalDropped.filter { $0.reasons.contains("exposure") }.count
        log(String(format:
            "[FrameQuality] sharp q10/med/q90=%.5f/%.5f/%.5f clipMed=%.3f threshold=%.5f blurMed=%.3f maxBlur=%.2f",
            ev.sharpQ10, ev.sharpMedian, ev.sharpQ90, ev.clipMedian, ev.sharpThreshold,
            ev.blurMedian, parameters.maxBlurScore
        ))
        log(String(format:
            "[FrameQuality] kept %d/%d frames (dropped=%d blur=%d localblur=%d exposure=%d proposed=%d rescued=%d minKeep=%d)",
            keep.count, n, finalDropped.count, blurDrops, localBlurDrops, exposureDrops,
            ev.proposedDropCount, ev.rescuedCount, ev.minKeep
        ))

        if !finalDropped.isEmpty && parameters.logDropLimit > 0 {
            for d in finalDropped.prefix(parameters.logDropLimit) {
                let m = d.metrics
                let rel = ev.sharpMedian > 1e-7 ? m.sharpness / ev.sharpMedian : 1
                let localMed = localSharpMedian(m.index)
                let localRel = localMed > 1e-7 ? m.sharpness / localMed : rel
                log(String(format:
                    "  [FrameQuality] drop frame[%d] reason=%@ sharp=%.5f blur=%.3f rel=%.2f localRel=%.2f mean=%.3f std=%.3f clip=%.3f",
                    m.index, d.reasons.joined(separator: "+") as NSString,
                    m.sharpness, m.blurScore, rel, localRel, m.lumaMean, m.lumaStd, m.clippedLumaFraction
                ))
            }
            if finalDropped.count > parameters.logDropLimit {
                log("  [FrameQuality] ... \(finalDropped.count - parameters.logDropLimit) more dropped frames")
            }
        }

        // Log the top blurScores on one line for threshold calibration (✗ marks dropped frames).
        // blurMed / maxBlur are on the first line.
        let blurriest = decisions.sorted { $0.metrics.blurScore > $1.metrics.blurScore }.prefix(5)
        if !blurriest.isEmpty {
            let entries = blurriest.map { d -> String in
                let m = d.metrics
                let rel = ev.sharpMedian > 1e-7 ? m.sharpness / ev.sharpMedian : 1
                return String(format: "[%d]%@ blur=%.3f rel=%.2f",
                              m.index, (keep.contains(m.index) ? "" : "✗") as NSString, m.blurScore, rel)
            }
            log(String(format: "[FrameQuality] blurriest (ceil=%.2f): %@",
                       parameters.blurScoreSharpCeiling, entries.joined(separator: ", ") as NSString))
        }

        if finalDropped.isEmpty { return scene }
        let keepIndices = scene.frames.indices.filter { keep.contains($0) }
        return scene.withFrames(keepIndices.map { scene.frames[$0] })
    }

    private static func quantile(sorted: [Float], q: Float) -> Float {
        guard !sorted.isEmpty else { return 0 }
        let clamped = Swift.max(0, Swift.min(q, 1))
        let idx = Int((Float(sorted.count - 1) * clamped).rounded())
        return sorted[idx]
    }

    /// No-reference blurriness [0,1] (edge-width method, Marziliano et al. 2002). Measures the
    /// transition width at local gradient maxima and maps the mean width w to w/(w+k). It looks at
    /// width, not edge count, so it is content-invariant. Takes the max of the horizontal and
    /// vertical mean widths to handle directional blur. 1 if there are no edges.
    static func edgeWidthBlurScore(rgb: MLXArray, lineStride: Int = 2) -> Float {
        precondition(rgb.shape.count == 3 && rgb.shape[2] == 3, "rgb must be [H, W, 3]")
        return edgeWidthBlurScore(
            flat: rgb.asArray(Float.self), height: rgb.shape[0], width: rgb.shape[1], lineStride: lineStride)
    }

    /// `flat` is a [H, W, 3] host array.
    static func edgeWidthBlurScore(flat: [Float], height h: Int, width w: Int, lineStride: Int = 2) -> Float {
        guard h >= 5, w >= 5 else { return 1 }

        var luma = [Float](repeating: 0, count: h * w)
        for i in 0..<(h * w) {
            let o = i * 3
            let r = flat[o], g = flat[o + 1], b = flat[o + 2]
            luma[i] = r.isFinite && g.isFinite && b.isFinite
                ? 0.299 * r + 0.587 * g + 0.114 * b : 0
        }

        let ls = Swift.max(1, lineStride)
        let gradThresh: Float = 0.025   // minimum edge step, to reject noise
        let maxWalk = 25                // max walk per side (px)
        let mapK: Float = 3

        func scanLine(count: Int, at: (Int) -> Float, into sumW: inout Float, cnt: inout Int) {
            guard count >= 3 else { return }
            var i = 1
            while i < count - 1 {
                let dPrev = at(i) - at(i - 1)
                let dHere = at(i + 1) - at(i)
                // Treat positions where dHere is a local maximum as edges
                let dNext = (i + 2 < count) ? at(i + 2) - at(i + 1) : 0
                let a = Swift.abs(dHere)
                guard a >= gradThresh, a >= Swift.abs(dPrev), a >= Swift.abs(dNext) else {
                    i += 1; continue
                }
                let rising = dHere > 0
                // Walk left/right only while the transition continues
                var l = i
                var steps = 0
                while l > 0, steps < maxWalk {
                    let cont = rising ? (at(l - 1) < at(l)) : (at(l - 1) > at(l))
                    if !cont { break }
                    l -= 1; steps += 1
                }
                var rr = i + 1
                steps = 0
                while rr < count - 1, steps < maxWalk {
                    let cont = rising ? (at(rr + 1) > at(rr)) : (at(rr + 1) < at(rr))
                    if !cont { break }
                    rr += 1; steps += 1
                }
                let width = Swift.max(1, rr - l)
                sumW += Float(width)
                cnt += 1
                i = Swift.max(i + 1, rr)   // do not count the same transition twice
            }
        }

        // Row direction
        var sumW_h: Float = 0, cnt_h = 0
        var y = 0
        while y < h {
            let base = y * w
            scanLine(count: w, at: { luma[base + $0] }, into: &sumW_h, cnt: &cnt_h)
            y += ls
        }
        // Column direction
        var sumW_v: Float = 0, cnt_v = 0
        var x = 0
        while x < w {
            scanLine(count: h, at: { luma[$0 * w + x] }, into: &sumW_v, cnt: &cnt_v)
            x += ls
        }

        let aH: Float? = cnt_h > 0 ? sumW_h / Float(cnt_h) : nil
        let aV: Float? = cnt_v > 0 ? sumW_v / Float(cnt_v) : nil
        guard let avgW = [aH, aV].compactMap({ $0 }).max() else { return 1 }
        return avgW / (avgW + mapK)
    }

    /// `flat` is a [H, W, 3] host array.
    private static func exposureStats(
        flat: [Float], height h: Int, width w: Int,
        sampleStride: Int,
        lowClipThreshold: Float,
        highClipThreshold: Float
    ) -> (mean: Float, std: Float, lowClipFraction: Float, highClipFraction: Float) {
        let step = Swift.max(1, sampleStride)
        var count = 0
        var sum: Float = 0
        var sumSq: Float = 0
        var lowClip = 0
        var highClip = 0

        for y in Swift.stride(from: 0, to: h, by: step) {
            for x in Swift.stride(from: 0, to: w, by: step) {
                let o = (y * w + x) * 3
                guard o + 2 < flat.count else { continue }
                let r = flat[o]
                let g = flat[o + 1]
                let b = flat[o + 2]
                guard r.isFinite, g.isFinite, b.isFinite else { continue }
                let luma = 0.299 * r + 0.587 * g + 0.114 * b
                sum += luma
                sumSq += luma * luma
                if luma <= lowClipThreshold { lowClip += 1 }
                if luma >= highClipThreshold { highClip += 1 }
                count += 1
            }
        }
        guard count > 0 else { return (0, 0, 1, 1) }
        let inv = 1 / Float(count)
        let mean = sum * inv
        let variance = Swift.max(0, sumSq * inv - mean * mean)
        return (
            mean: mean,
            std: variance.squareRoot(),
            lowClipFraction: Float(lowClip) * inv,
            highClipFraction: Float(highClip) * inv
        )
    }
}
