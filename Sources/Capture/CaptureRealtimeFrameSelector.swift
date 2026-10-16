// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation

/// Online image quality gate during capture (blur, exposure). Applied directly to the downscaled
/// RGB UInt8; rejected frames do not become CapturedFrames / anchors / coverage points.
public struct CaptureRealtimeFrameSelector: Sendable {

    public enum RejectReason: String, Sendable, Equatable {
        case blur
        case exposure
    }

    public struct Parameters: Sendable, Equatable {
        public var enabled: Bool
        public var warmupAcceptedFrames: Int
        public var sharpnessStride: Int
        public var exposureSampleStride: Int
        public var minSharpnessMedianRatio: Float
        public var minSharpnessAbsolute: Float
        public var darkMeanThreshold: Float
        public var brightMeanThreshold: Float
        public var lowClipThreshold: Float
        public var highClipThreshold: Float
        public var maxClippedLumaFraction: Float

        public init(
            enabled: Bool = true,
            warmupAcceptedFrames: Int = 4,
            sharpnessStride: Int = 4,
            exposureSampleStride: Int = 4,
            minSharpnessMedianRatio: Float = 0.30,
            minSharpnessAbsolute: Float = 1e-4,
            darkMeanThreshold: Float = 0.04,
            brightMeanThreshold: Float = 0.96,
            lowClipThreshold: Float = 0.01,
            highClipThreshold: Float = 0.99,
            maxClippedLumaFraction: Float = 0.35
        ) {
            precondition(warmupAcceptedFrames >= 0)
            precondition(sharpnessStride >= 1)
            precondition(exposureSampleStride >= 1)
            precondition(minSharpnessMedianRatio >= 0)
            precondition(minSharpnessAbsolute >= 0)
            precondition(darkMeanThreshold >= 0 && darkMeanThreshold <= 1)
            precondition(brightMeanThreshold >= 0 && brightMeanThreshold <= 1)
            precondition(lowClipThreshold >= 0 && lowClipThreshold <= 1)
            precondition(highClipThreshold >= 0 && highClipThreshold <= 1)
            precondition(maxClippedLumaFraction >= 0 && maxClippedLumaFraction <= 1)
            self.enabled = enabled
            self.warmupAcceptedFrames = warmupAcceptedFrames
            self.sharpnessStride = sharpnessStride
            self.exposureSampleStride = exposureSampleStride
            self.minSharpnessMedianRatio = minSharpnessMedianRatio
            self.minSharpnessAbsolute = minSharpnessAbsolute
            self.darkMeanThreshold = darkMeanThreshold
            self.brightMeanThreshold = brightMeanThreshold
            self.lowClipThreshold = lowClipThreshold
            self.highClipThreshold = highClipThreshold
            self.maxClippedLumaFraction = maxClippedLumaFraction
        }
    }

    public struct Metrics: Sendable, Equatable {
        public let sharpness: Float
        public let lumaMean: Float
        public let lumaStd: Float
        public let lowClipFraction: Float
        public let highClipFraction: Float
        public let acceptedSharpnessMedian: Float

        public var clippedLumaFraction: Float {
            Swift.max(lowClipFraction, highClipFraction)
        }
    }

    public struct Decision: Sendable, Equatable {
        public let accepted: Bool
        public let reasons: [RejectReason]
        public let metrics: Metrics
    }

    public let parameters: Parameters
    private var acceptedSharpness: [Float] = []

    public init(parameters: Parameters = Parameters()) {
        self.parameters = parameters
    }

    public var acceptedCount: Int { acceptedSharpness.count }

    public mutating func reset() {
        acceptedSharpness.removeAll(keepingCapacity: true)
    }

    public mutating func evaluate(
        rgbBytes: [UInt8],
        width: Int,
        height: Int
    ) -> Decision {
        let metrics = Self.metrics(
            rgbBytes: rgbBytes,
            width: width,
            height: height,
            parameters: parameters,
            acceptedSharpness: acceptedSharpness
        )
        guard parameters.enabled else {
            acceptedSharpness.append(metrics.sharpness)
            return Decision(accepted: true, reasons: [], metrics: metrics)
        }

        var reasons: [RejectReason] = []
        if metrics.lumaMean < parameters.darkMeanThreshold
            || metrics.lumaMean > parameters.brightMeanThreshold
            || metrics.clippedLumaFraction > parameters.maxClippedLumaFraction
        {
            reasons.append(.exposure)
        }

        if acceptedSharpness.count >= parameters.warmupAcceptedFrames,
           metrics.acceptedSharpnessMedian > parameters.minSharpnessAbsolute
        {
            let threshold = Swift.max(
                parameters.minSharpnessAbsolute,
                metrics.acceptedSharpnessMedian * parameters.minSharpnessMedianRatio
            )
            if metrics.sharpness < threshold {
                reasons.append(.blur)
            }
        }

        let accepted = reasons.isEmpty
        if accepted {
            acceptedSharpness.append(metrics.sharpness)
        }
        return Decision(accepted: accepted, reasons: reasons, metrics: metrics)
    }

    public static func metrics(
        rgbBytes: [UInt8],
        width: Int,
        height: Int,
        parameters: Parameters = Parameters(),
        acceptedSharpness: [Float] = []
    ) -> Metrics {
        precondition(width > 0 && height > 0)
        precondition(rgbBytes.count >= width * height * 3)

        let sharpness = sharpnessScore(
            rgbBytes: rgbBytes,
            width: width,
            height: height,
            stride: parameters.sharpnessStride
        )
        let exposure = exposureStats(
            rgbBytes: rgbBytes,
            width: width,
            height: height,
            sampleStride: parameters.exposureSampleStride,
            lowClipThreshold: parameters.lowClipThreshold,
            highClipThreshold: parameters.highClipThreshold
        )
        return Metrics(
            sharpness: sharpness,
            lumaMean: exposure.mean,
            lumaStd: exposure.std,
            lowClipFraction: exposure.lowClipFraction,
            highClipFraction: exposure.highClipFraction,
            acceptedSharpnessMedian: median(acceptedSharpness)
        )
    }

    private static func sharpnessScore(
        rgbBytes: [UInt8],
        width: Int,
        height: Int,
        stride: Int
    ) -> Float {
        guard width > 1, height > 1 else { return 0 }
        let step = Swift.max(1, stride)
        var sum: Float = 0
        var count: Float = 0
        for y in Swift.stride(from: 0, to: height - 1, by: step) {
            for x in Swift.stride(from: 0, to: width - 1, by: step) {
                let c = luma(rgbBytes, width: width, x: x, y: y)
                let cx = luma(rgbBytes, width: width, x: x + 1, y: y)
                let cy = luma(rgbBytes, width: width, x: x, y: y + 1)
                sum += abs(cx - c) + abs(cy - c)
                count += 1
            }
        }
        return count > 0 ? sum / count : 0
    }

    private static func exposureStats(
        rgbBytes: [UInt8],
        width: Int,
        height: Int,
        sampleStride: Int,
        lowClipThreshold: Float,
        highClipThreshold: Float
    ) -> (mean: Float, std: Float, lowClipFraction: Float, highClipFraction: Float) {
        let step = Swift.max(1, sampleStride)
        var count: Float = 0
        var sum: Float = 0
        var sumSq: Float = 0
        var low: Float = 0
        var high: Float = 0
        for y in Swift.stride(from: 0, to: height, by: step) {
            for x in Swift.stride(from: 0, to: width, by: step) {
                let v = luma(rgbBytes, width: width, x: x, y: y)
                sum += v
                sumSq += v * v
                if v <= lowClipThreshold { low += 1 }
                if v >= highClipThreshold { high += 1 }
                count += 1
            }
        }
        guard count > 0 else { return (0, 0, 1, 1) }
        let mean = sum / count
        let variance = Swift.max(0, sumSq / count - mean * mean)
        return (mean, variance.squareRoot(), low / count, high / count)
    }

    private static func luma(_ rgbBytes: [UInt8], width: Int, x: Int, y: Int) -> Float {
        let o = (y * width + x) * 3
        let r = Float(rgbBytes[o]) / 255
        let g = Float(rgbBytes[o + 1]) / 255
        let b = Float(rgbBytes[o + 2]) / 255
        return 0.299 * r + 0.587 * g + 0.114 * b
    }

    private static func median(_ xs: [Float]) -> Float {
        guard !xs.isEmpty else { return 0 }
        let sorted = xs.sorted()
        return sorted[sorted.count / 2]
    }
}
