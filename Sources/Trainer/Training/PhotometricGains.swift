// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation

/// Settings of the per-view gains (`PhotometricGains`).
public struct PhotometricGainConfig: Sendable, Equatable {
    /// Fraction of the iterations during which the gains stay 1, so they are not fitted to a
    /// render that is still far from the scene.
    public var startFraction: Float
    /// Share of the distance to the fitted gain that one visit of a view covers. A view is only
    /// drawn a few times per run, so a gradient step of a usual learning rate would barely move it.
    public var rate: Float
    /// Bound on a gain's natural log (0.7 is a factor of 2, a stop of exposure), so a gain cannot
    /// stand in for content the model has not learned yet.
    public var maxLogGain: Float

    public init(startFraction: Float = 0.3, rate: Float = 0.5, maxLogGain: Float = 0.7) {
        self.startFraction = startFraction
        self.rate = rate
        self.maxLogGain = maxLogGain
    }
}

/// A gain per view and colour channel that the render is multiplied by before it meets the image,
/// so frames shot with a different exposure or white balance (auto exposure before it locks, a
/// light that changes) do not pull the scene's colours towards themselves. It is the exposure and
/// colour part of PPISP (Deutsch et al.) without the vignetting, the camera response and the
/// controller; the exposure term is what its ablation finds matters most. The gains exist only in
/// training: the splats are the scene as seen at the average gain, which is what the viewer shows.
///
/// A gain is fitted in closed form instead of by gradient descent. With `x` the render and `t` the
/// image, the gain that minimises `Σ(g·x − t)²` per channel is `Σxt / Σx²`; each visit moves the
/// stored value part of the way there. The mean of the stored log gains over the views is
/// subtracted when a gain is read (the scene brightness and a common gain cannot be told apart).
public final class PhotometricGains {
    public let config: PhotometricGainConfig
    public let viewCount: Int
    /// Natural log of each view's gain, `viewCount × 3`, before the mean is removed.
    public private(set) var logGains: [Float]
    private var logSum = SIMD3<Float>(repeating: 0)

    public init(viewCount: Int, config: PhotometricGainConfig = PhotometricGainConfig()) {
        self.config = config
        self.viewCount = viewCount
        self.logGains = [Float](repeating: 0, count: viewCount * 3)
    }

    private func stored(_ view: Int) -> SIMD3<Float> {
        SIMD3(logGains[view * 3], logGains[view * 3 + 1], logGains[view * 3 + 2])
    }

    /// Natural log of the view's gain with the mean over the views removed.
    public func logGain(view: Int) -> SIMD3<Float> {
        stored(view) - logSum / Float(Swift.max(1, viewCount))
    }

    /// The factor to multiply the view's render by (1 before the view has been fitted).
    public func gain(view: Int) -> SIMD3<Float> {
        let l = logGain(view: view)
        return SIMD3(exp(l.x), exp(l.y), exp(l.z))
    }

    /// Moves the view's gain towards the one that fits `crossSum = Σxt` to `squareSum = Σx²` (each
    /// per channel, over the pixels of the view).
    public func fit(view: Int, crossSum: SIMD3<Float>, squareSum: SIMD3<Float>) {
        let bound = config.maxLogGain
        var target = SIMD3<Float>(repeating: 0)
        for c in 0..<3 {
            // A channel with no render in it (or no image) says nothing about the gain.
            guard squareSum[c] > 1e-6, crossSum[c] > 1e-6 else { continue }
            target[c] = Swift.min(bound, Swift.max(-bound, Foundation.log(crossSum[c] / squareSum[c])))
        }
        let old = stored(view)
        let moved = old + config.rate * (target - logGain(view: view))
        for c in 0..<3 { logGains[view * 3 + c] = moved[c] }
        logSum += moved - old
    }

    /// Spread of the exposure (the mean over channels of each view's log gain, in stops) over the
    /// views: smallest, largest and standard deviation. For the training log.
    public func exposureSummary() -> (minStops: Float, maxStops: Float, stdStops: Float) {
        guard viewCount > 0 else { return (0, 0, 0) }
        let stops = (0..<viewCount).map { (logGain(view: $0).sum() / 3) / Float(Foundation.log(2.0)) }
        let mean = stops.reduce(0, +) / Float(viewCount)
        let variance = stops.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(viewCount)
        return (stops.min() ?? 0, stops.max() ?? 0, variance.squareRoot())
    }
}
