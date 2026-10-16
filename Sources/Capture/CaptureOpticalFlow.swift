// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX

/// Lightweight optical flow diagnostics between adjacent frames (for capture quality). Not dense
/// flow but sparse block matching statistics, cheaply measuring overlap, excessive motion, low
/// texture and inconsistency.
public enum CaptureOpticalFlow {

    public struct Parameters: Sendable, Equatable {
        public var enabled: Bool
        public var targetWidth: Int
        public var blockRadius: Int
        public var blockStride: Int
        public var searchRadius: Int
        public var minTextureStd: Float
        public var logPairLimit: Int

        public init(
            enabled: Bool = true,
            targetWidth: Int = 64,
            blockRadius: Int = 1,
            blockStride: Int = 6,
            searchRadius: Int = 4,
            minTextureStd: Float = 0.025,
            logPairLimit: Int = 3
        ) {
            precondition(targetWidth >= 8)
            precondition(blockRadius >= 1)
            precondition(blockStride >= 1)
            precondition(searchRadius >= 1)
            precondition(minTextureStd >= 0)
            precondition(logPairLimit >= 0)
            self.enabled = enabled
            self.targetWidth = targetWidth
            self.blockRadius = blockRadius
            self.blockStride = blockStride
            self.searchRadius = searchRadius
            self.minTextureStd = minTextureStd
            self.logPairLimit = logPairLimit
        }
    }

    public struct PairMetrics: Sendable {
        public let fromIndex: Int
        public let toIndex: Int
        public let candidateCount: Int
        public let vectorCount: Int
        public let medianDxPixels: Float
        public let medianDyPixels: Float
        public let medianMagnitudePixels: Float
        public let p90MagnitudePixels: Float
        public let meanResidual: Float

        public var textureCoverage: Float {
            candidateCount > 0 ? Float(vectorCount) / Float(candidateCount) : 0
        }
    }

    public struct Summary: Sendable {
        public let pairs: [PairMetrics]
        public let medianPairMagnitudePixels: Float
        public let medianTextureCoverage: Float
        public let maxP90MagnitudePixels: Float
        public let meanResidual: Float
    }

    /// Mapping from adjacent-flow SAD residual → per-frame training sampling weight. Frames with
    /// large residual (blur, parallax, tracking failure) get fewer steps. Normalized by quantiles
    /// within the capture so it does not depend on absolute values.
    public struct WeightParameters: Sendable, Equatable {
        public var enabled: Bool
        /// Lower bound of the weight. 0 would mean that view is never trained, so avoid it.
        public var floor: Float
        /// Residuals at or below this quantile get weight 1.
        public var lowQuantile: Float
        /// Residuals at or above this quantile get weight `floor`.
        public var highQuantile: Float

        public init(
            enabled: Bool = true,
            floor: Float = 0.3,
            lowQuantile: Float = 0.6,
            highQuantile: Float = 0.95
        ) {
            precondition(floor >= 0 && floor <= 1)
            precondition(lowQuantile >= 0 && lowQuantile <= 1)
            precondition(highQuantile >= lowQuantile && highQuantile <= 1)
            self.enabled = enabled
            self.floor = floor
            self.lowQuantile = lowQuantile
            self.highQuantile = highQuantile
        }

        public static var disabled: WeightParameters {
            WeightParameters(enabled: false)
        }
    }

    /// Returns per-frame sampling weights [floor, 1]. A frame's residual is the max over its
    /// adjacent pairs. All 1 when disabled or when the residual spread is degenerate.
    public static func frameWeights(
        frameCount: Int,
        summary: Summary?,
        parameters: WeightParameters = WeightParameters(),
        log: (String) -> Void
    ) -> [Float] {
        let ones = [Float](repeating: 1, count: Swift.max(0, frameCount))
        guard parameters.enabled else { return ones }
        guard let summary, !summary.pairs.isEmpty, frameCount > 0 else { return ones }

        var frameResidual = [Float](repeating: 0, count: frameCount)
        var hasLink = [Bool](repeating: false, count: frameCount)
        for p in summary.pairs {
            for idx in [p.fromIndex, p.toIndex] where idx >= 0 && idx < frameCount {
                frameResidual[idx] = Swift.max(frameResidual[idx], p.meanResidual)
                hasLink[idx] = true
            }
        }

        let residualDist = summary.pairs.map(\.meanResidual).sorted()
        let lowR = quantile(residualDist, q: parameters.lowQuantile)
        let highR = quantile(residualDist, q: parameters.highQuantile)
        let span = highR - lowR
        guard span > 1e-6 else {
            log("[ViewWeight] residual spread degenerate (lowR=\(lowR)); all frames weight=1.0")
            return ones
        }

        var weights = ones
        var downCount = 0
        for i in 0..<frameCount where hasLink[i] {
            let excess = Swift.max(0, Swift.min(1, (frameResidual[i] - lowR) / span))
            let w = 1 - (1 - parameters.floor) * excess
            weights[i] = w
            if w < 0.999 { downCount += 1 }
        }
        let minW = weights.min() ?? 1
        log(String(format:
            "[ViewWeight] flow residual band low=%.4f high=%.4f → down-weighted %d/%d frames (minWeight=%.2f, floor=%.2f)",
            lowR, highR, downCount, frameCount, minW, parameters.floor))
        return weights
    }

    private struct LumaFrame {
        let width: Int
        let height: Int
        let values: [Float]
        let scaleX: Float
        let scaleY: Float

        func value(_ x: Int, _ y: Int) -> Float {
            values[y * width + x]
        }
    }

    public static func analyze(
        frames: [CapturedFrame],
        parameters: Parameters = Parameters(),
        log: (String) -> Void
    ) -> Summary? {
        guard parameters.enabled else {
            log("[OpticalFlow] disabled")
            return nil
        }
        guard frames.count >= 2 else {
            log("[OpticalFlow] skipped: need at least 2 frames")
            return nil
        }

        let lumaFrames = frames.map { makeLumaFrame(frame: $0, targetWidth: parameters.targetWidth) }
        var pairs: [PairMetrics] = []
        pairs.reserveCapacity(frames.count - 1)
        for i in 0..<(lumaFrames.count - 1) {
            pairs.append(pairMetrics(
                fromIndex: i,
                toIndex: i + 1,
                a: lumaFrames[i],
                b: lumaFrames[i + 1],
                parameters: parameters
            ))
        }
        guard !pairs.isEmpty else { return nil }

        let medianPairMagnitude = median(pairs.map(\.medianMagnitudePixels))
        let medianCoverage = median(pairs.map(\.textureCoverage))
        let maxP90 = pairs.map(\.p90MagnitudePixels).max() ?? 0
        let residual = pairs.map(\.meanResidual).reduce(0, +) / Float(pairs.count)
        let summary = Summary(
            pairs: pairs,
            medianPairMagnitudePixels: medianPairMagnitude,
            medianTextureCoverage: medianCoverage,
            maxP90MagnitudePixels: maxP90,
            meanResidual: residual
        )

        log(String(format:
            "[OpticalFlow] pairs=%d gridWidth=%d medianMag=%.1fpx maxP90=%.1fpx medianCoverage=%.2f meanResidual=%.4f",
            pairs.count, parameters.targetWidth, medianPairMagnitude, maxP90,
            medianCoverage, residual
        ))
        if parameters.logPairLimit > 0 {
            let interesting = pairs.sorted { a, b in
                if a.p90MagnitudePixels == b.p90MagnitudePixels {
                    return a.meanResidual > b.meanResidual
                }
                return a.p90MagnitudePixels > b.p90MagnitudePixels
            }
            for p in interesting.prefix(parameters.logPairLimit) {
                log(String(format:
                    "  [OpticalFlow] pair[%d->%d] med=(%.1f, %.1f) mag=%.1f p90=%.1f coverage=%.2f residual=%.4f vectors=%d/%d",
                    p.fromIndex, p.toIndex,
                    p.medianDxPixels, p.medianDyPixels,
                    p.medianMagnitudePixels, p.p90MagnitudePixels,
                    p.textureCoverage, p.meanResidual,
                    p.vectorCount, p.candidateCount
                ))
            }
        }
        return summary
    }

    private static func makeLumaFrame(frame: CapturedFrame, targetWidth: Int) -> LumaFrame {
        precondition(frame.image.shape.count == 3 && frame.image.shape[2] == 3)
        let srcW = frame.width
        let srcH = frame.height
        let w = Swift.max(8, Swift.min(targetWidth, srcW))
        let h = Swift.max(8, Int((Float(srcH) * Float(w) / Float(srcW)).rounded()))
        // Gather only the nearest grid points on the GPU before reading back (do not copy full-res
        // to host).
        let rows = (0..<h).map { Int32(Swift.min(srcH - 1, Int((Float($0) + 0.5) * Float(srcH) / Float(h)))) }
        let cols = (0..<w).map { Int32(Swift.min(srcW - 1, Int((Float($0) + 0.5) * Float(srcW) / Float(w)))) }
        let flat = frame.imageFloat
            .take(MLXArray(rows), axis: 0)
            .take(MLXArray(cols), axis: 1)
            .asArray(Float.self)   // [h, w, 3]
        var luma = [Float](repeating: 0, count: w * h)
        for i in 0..<(w * h) {
            let off = i * 3
            luma[i] = 0.299 * flat[off] + 0.587 * flat[off + 1] + 0.114 * flat[off + 2]
        }
        return LumaFrame(
            width: w,
            height: h,
            values: luma,
            scaleX: Float(srcW) / Float(w),
            scaleY: Float(srcH) / Float(h)
        )
    }

    private static func pairMetrics(
        fromIndex: Int,
        toIndex: Int,
        a: LumaFrame,
        b: LumaFrame,
        parameters: Parameters
    ) -> PairMetrics {
        let w = Swift.min(a.width, b.width)
        let h = Swift.min(a.height, b.height)
        let r = parameters.blockRadius
        let search = parameters.searchRadius
        let margin = r + search
        guard w > margin * 2 + 1, h > margin * 2 + 1 else {
            return PairMetrics(
                fromIndex: fromIndex, toIndex: toIndex,
                candidateCount: 0, vectorCount: 0,
                medianDxPixels: 0, medianDyPixels: 0,
                medianMagnitudePixels: 0, p90MagnitudePixels: 0,
                meanResidual: 0
            )
        }

        var dxs: [Float] = []
        var dys: [Float] = []
        var mags: [Float] = []
        var residuals: [Float] = []
        var candidates = 0
        let blockCount = Float((2 * r + 1) * (2 * r + 1))

        for y in Swift.stride(from: margin, to: h - margin, by: parameters.blockStride) {
            for x in Swift.stride(from: margin, to: w - margin, by: parameters.blockStride) {
                candidates += 1
                let std = blockStd(a, x: x, y: y, radius: r)
                if std < parameters.minTextureStd { continue }

                var bestDx = 0
                var bestDy = 0
                var bestSad = Float.greatestFiniteMagnitude
                for dy in -search...search {
                    for dx in -search...search {
                        let sad = blockSAD(a, b, x: x, y: y, dx: dx, dy: dy, radius: r)
                        if sad < bestSad {
                            bestSad = sad
                            bestDx = dx
                            bestDy = dy
                        }
                    }
                }
                let dxPixels = Float(bestDx) * a.scaleX
                let dyPixels = Float(bestDy) * a.scaleY
                dxs.append(dxPixels)
                dys.append(dyPixels)
                mags.append((dxPixels * dxPixels + dyPixels * dyPixels).squareRoot())
                residuals.append(bestSad / blockCount)
            }
        }

        return PairMetrics(
            fromIndex: fromIndex,
            toIndex: toIndex,
            candidateCount: candidates,
            vectorCount: mags.count,
            medianDxPixels: median(dxs),
            medianDyPixels: median(dys),
            medianMagnitudePixels: median(mags),
            p90MagnitudePixels: quantile(mags, q: 0.90),
            meanResidual: residuals.isEmpty ? 0 : residuals.reduce(0, +) / Float(residuals.count)
        )
    }

    private static func blockStd(_ f: LumaFrame, x: Int, y: Int, radius: Int) -> Float {
        var sum: Float = 0
        var sumSq: Float = 0
        var count: Float = 0
        for yy in (y - radius)...(y + radius) {
            for xx in (x - radius)...(x + radius) {
                let v = f.value(xx, yy)
                sum += v
                sumSq += v * v
                count += 1
            }
        }
        let mean = sum / count
        return Swift.max(0, sumSq / count - mean * mean).squareRoot()
    }

    private static func blockSAD(
        _ a: LumaFrame,
        _ b: LumaFrame,
        x: Int,
        y: Int,
        dx: Int,
        dy: Int,
        radius: Int
    ) -> Float {
        var sad: Float = 0
        for yy in -radius...radius {
            for xx in -radius...radius {
                sad += abs(a.value(x + xx, y + yy) - b.value(x + dx + xx, y + dy + yy))
            }
        }
        return sad
    }

    private static func median(_ xs: [Float]) -> Float {
        quantile(xs, q: 0.5)
    }

    private static func quantile(_ xs: [Float], q: Float) -> Float {
        guard !xs.isEmpty else { return 0 }
        let sorted = xs.sorted()
        let idx = Int((Float(sorted.count - 1) * Swift.max(0, Swift.min(q, 1))).rounded())
        return sorted[idx]
    }
}
