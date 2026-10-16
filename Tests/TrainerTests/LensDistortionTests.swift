// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Estimating radial lens distortion (k1, k2) and undistorting images.

import Foundation
import MLX
import simd
import Testing
@testable import Trainer

@Suite struct LensDistortionTests {

    /// Deterministic pseudo-random numbers (LCG).
    struct LCG {
        var state: UInt64
        mutating func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(state >> 40) / Float(1 << 24)
        }
        mutating func uniform(_ lo: Float, _ hi: Float) -> Float { lo + (hi - lo) * next() }
    }

    static let width = 640, height = 480
    static let k: Intrinsics = (fx: 500, fy: 500, cx: 320, cy: 240)

    /// Distorted observations of points in a box seen from 8 cameras lined up along x (no
    /// rotation).
    static func syntheticScene(
        distortion: RadialDistortion, noisePx: Float = 0
    ) -> (state: BundleAdjustment.State, observations: [BundleAdjustment.Observation]) {
        var rng = LCG(state: 7)
        let viewmats: [[Float]] = (0..<8).map { i in
            let cx = Float(i) * 0.15 - 0.5
            return [1, 0, 0, -cx, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        }
        let points: [SIMD3<Float>] = (0..<600).map { _ in
            SIMD3(rng.uniform(-2.6, 2.6), rng.uniform(-1.9, 1.9), rng.uniform(2.0, 4.0))
        }
        var obs: [BundleAdjustment.Observation] = []
        for (c, vm) in viewmats.enumerated() {
            for (j, p) in points.enumerated() {
                let pc = SIMD3(p.x + vm[3], p.y + vm[7], p.z + vm[11])
                let pd = distortion.distort(SIMD2(pc.x / pc.z, pc.y / pc.z))
                let uv = SIMD2(k.fx * pd.x + k.cx + noisePx * rng.uniform(-1, 1),
                               k.fy * pd.y + k.cy + noisePx * rng.uniform(-1, 1))
                guard uv.x >= 0, uv.y >= 0, uv.x < Float(width), uv.y < Float(height) else { continue }
                obs.append(BundleAdjustment.Observation(cameraIndex: c, pointIndex: j, pixel: uv))
            }
        }
        return (BundleAdjustment.State(viewmats: viewmats, points: points), obs)
    }

    static func problem(
        _ scene: (state: BundleAdjustment.State, observations: [BundleAdjustment.Observation]),
        parameters: LensDistortionEstimation.Parameters = .init()
    ) -> LensDistortionEstimation.Problem {
        let n = scene.state.viewmats.count
        return LensDistortionEstimation.Problem(
            state: scene.state, intrinsics: Array(repeating: k, count: n),
            sizes: Array(repeating: (width: width, height: height), count: n),
            observations: scene.observations, inlierObservations: scene.observations,
            parameters: parameters)
    }

    @Test func undistortInvertsDistort() {
        let d = RadialDistortion(k1: -0.08, k2: 0.02)
        for p in [SIMD2<Float>(0, 0), SIMD2(0.3, -0.2), SIMD2(-0.6, 0.45), SIMD2(0.7, 0.5)] {
            guard let back = d.undistort(d.distort(p)) else {
                Issue.record("undistort failed for \(p)")
                continue
            }
            #expect(simd_length(back - p) < 1e-5)
        }
        let uv = SIMD2<Float>(37.5, 451.5)
        let pd = SIMD2((uv.x - Self.k.cx) / Self.k.fx, (uv.y - Self.k.cy) / Self.k.fy)
        let u = d.undistortPixel(uv, Self.k)!
        let pu = SIMD2((u.x - Self.k.cx) / Self.k.fx, (u.y - Self.k.cy) / Self.k.fy)
        #expect(simd_length(d.distort(pu) - pd) < 1e-5)
        #expect(d.roundTripValid(Self.k, width: Self.width, height: Self.height))
    }

    /// Where the distorted radius is not monotonic (no inverse exists), the inverse map reports
    /// failure and the round-trip check fails.
    @Test func undistortFailsBeyondInvertibleRange() {
        // r·(1 − 0.5 r²) peaks at 0.544 at r = 0.816. Distorted radii beyond that have no solution.
        let d = RadialDistortion(k1: -0.5, k2: 0)
        #expect(d.undistort(SIMD2(0.6, 0)) == nil)
        #expect(d.undistort(SIMD2(0.3, 0)) != nil)
        // An image whose corner normalized radius is 0.8 (> 0.544) fails the round-trip check.
        #expect(!d.roundTripValid(Self.k, width: Self.width, height: Self.height))
    }

    /// Given the true poses and points, k1, k2 are recovered from noisy observations and adopted.
    @Test func estimateRecoversSyntheticDistortion() {
        let truth = RadialDistortion(k1: -0.06, k2: 0.015)
        let p = Self.problem(Self.syntheticScene(distortion: truth, noisePx: 0.5))
        let est = p.estimate(.init())
        #expect(est.accepted, "\(est.rejectReason ?? "")")
        #expect(abs(est.distortion.k1 - truth.k1) < 0.003)
        #expect(abs(est.distortion.k2 - truth.k2) < 0.005)
        #expect(abs(est.cornerShiftPx - p.cornerShiftPx(truth)) < 0.3)
        // After correction the outer residuals shrink.
        let before = p.radialProfile(distortion: .none)
        let after = p.radialProfile(distortion: est.distortion)
        #expect(after.last!.medianPx < 0.5 * before.last!.medianPx)
    }

    /// Even if the BA points have shifted to absorb the distortion, points and k are solved
    /// jointly, so k is recovered.
    @Test func estimateRecoversDistortionFromPerturbedPoints() {
        let truth = RadialDistortion(k1: -0.06, k2: 0.015)
        var scene = Self.syntheticScene(distortion: truth, noisePx: 0.3)
        var rng = LCG(state: 11)
        scene.state = BundleAdjustment.State(
            viewmats: scene.state.viewmats,
            points: scene.state.points.map {
                $0 + SIMD3(rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02))
            })
        let est = Self.problem(scene).estimate(.init())
        #expect(est.accepted, "\(est.rejectReason ?? "")")
        #expect(abs(est.distortion.k1 - truth.k1) < 0.003)
        #expect(abs(est.distortion.k2 - truth.k2) < 0.005)
    }

    /// Points seen from only 2 views are not used for estimation by default (track length 3 or
    /// more).
    @Test func estimateIgnoresTwoViewTracks() {
        let truth = RadialDistortion(k1: -0.06, k2: 0.015)
        var scene = Self.syntheticScene(distortion: truth, noisePx: 0.3)
        var seen = [Int: Int]()
        scene.observations = scene.observations.filter { o in
            seen[o.pointIndex, default: 0] += 1
            return seen[o.pointIndex]! <= 2
        }
        let strict = Self.problem(scene).estimate(.init())
        #expect(strict.pointCount == 0)
        #expect(!strict.accepted)
        let loose = LensDistortionEstimation.Parameters(minTrackLength: 2)
        let est = Self.problem(scene, parameters: loose).estimate(loose)
        #expect(est.pointCount > 0)
        #expect(abs(est.distortion.k1 - truth.k1) < 0.01)
    }

    /// Without distortion it is not adopted (correction below the threshold).
    @Test func estimateRejectsWithoutDistortion() {
        let p = Self.problem(Self.syntheticScene(distortion: .none, noisePx: 0.5))
        let est = p.estimate(.init())
        #expect(!est.accepted)
        #expect(abs(est.cornerShiftPx) < 0.5)
    }

    /// Without observations near the image edges k is undetermined, so it is not adopted.
    @Test func estimateRejectsWithoutEdgeObservations() {
        let truth = RadialDistortion(k1: -0.06, k2: 0.015)
        var scene = Self.syntheticScene(distortion: truth, noisePx: 0.5)
        scene.observations = scene.observations.filter {
            abs($0.pixel.x - Self.k.cx) < 120 && abs($0.pixel.y - Self.k.cy) < 90
        }
        let est = Self.problem(scene).estimate(.init(minObservations: 100))
        #expect(!est.accepted)
    }

    /// When BA returns ground truth, the alternating estimation converges in one correction and
    /// returns the true distortion.
    @Test func refineAlternatingConverges() {
        let truth = RadialDistortion(k1: -0.06, k2: 0.015)
        let scene = Self.syntheticScene(distortion: truth, noisePx: 0.3)
        let n = scene.state.viewmats.count
        let initial = IterativeGeometricRefinement.Result(
            state: scene.state, observations: scene.observations, rounds: [])
        var solveCalls = 0
        var logs: [String] = []
        let (_, d) = LensDistortionEstimation.refineAlternating(
            initial: initial, intrinsics: Array(repeating: Self.k, count: n),
            sizes: Array(repeating: (width: Self.width, height: Self.height), count: n),
            observations: scene.observations, parameters: .init(maxAlternations: 3),
            solve: { obs in
                solveCalls += 1
                return IterativeGeometricRefinement.Result(state: scene.state, observations: obs, rounds: [])
            },
            log: { logs.append($0) })
        #expect(solveCalls == 1)
        #expect(logs.contains { $0.contains("converged") })
        #expect(abs(d.k1 - truth.k1) < 0.003)
    }

    @Test func refineAlternatingDisabledIsIdentity() {
        let scene = Self.syntheticScene(distortion: RadialDistortion(k1: -0.06, k2: 0), noisePx: 0)
        let initial = IterativeGeometricRefinement.Result(
            state: scene.state, observations: scene.observations, rounds: [])
        let (_, d) = LensDistortionEstimation.refineAlternating(
            initial: initial, intrinsics: [], sizes: [], observations: scene.observations,
            parameters: .disabled, solve: { _ in fatalError("must not re-solve") }, log: { _ in })
        #expect(d.isNone)
    }

    /// Barrel shrinks the focal length to widen the field of view; pincushion stretches it to avoid
    /// black borders. No distortion keeps K as is.
    @Test func undistortedIntrinsicsKeepsBorderInside() {
        let k = Self.k
        let same = LensUndistortion.undistortedIntrinsics(k, width: Self.width, height: Self.height, distortion: .none)
        #expect(same?.fx == k.fx)
        for (d, shrinks) in [(RadialDistortion(k1: -0.05, k2: 0), true), (RadialDistortion(k1: 0.05, k2: 0), false)] {
            guard let t = LensUndistortion.undistortedIntrinsics(
                k, width: Self.width, height: Self.height, distortion: d) else {
                Issue.record("no crop found for \(d)")
                continue
            }
            #expect(shrinks ? t.fx < k.fx : t.fx > k.fx)
            #expect(t.cx == k.cx && t.cy == k.cy)
            // The output's 4 corners and edge midpoints all sample within the range of input pixel
            // centers.
            let w = Float(Self.width), h = Float(Self.height)
            for (u, v) in [(0.5, 0.5), (w - 0.5, 0.5), (0.5, h - 0.5), (w - 0.5, h - 0.5),
                           (w / 2, 0.5), (0.5, h / 2)] as [(Float, Float)] {
                let p = d.distort(SIMD2((u - t.cx) / t.fx, (v - t.cy) / t.fy))
                let ui = k.fx * p.x + k.cx, vi = k.fy * p.y + k.cy
                // The search picks right at the boundary (1e-3 px tolerance), so the check allows a
                // 0.01 px margin.
                #expect(ui >= 0.49 && ui <= w - 0.49 && vi >= 0.49 && vi <= h - 0.49, "\(d): (\(ui), \(vi))")
            }
        }
    }

    /// With no distortion and the same K, pixels are unchanged (Catmull-Rom is the identity at
    /// integer positions).
    @Test func undistortImageIdentity() {
        var rng = LCG(state: 3)
        let h = 24, w = 32
        let bytes = (0..<(h * w * 3)).map { _ in UInt8(rng.uniform(0, 255.99)) }
        let image = MLXArray(bytes, [h, w, 3])
        let k: Intrinsics = (fx: 30, fy: 30, cx: 16, cy: 12)
        let out = LensUndistortion.undistortImage(image, source: k, target: k, distortion: .none)
        #expect(out.dtype == .uint8)
        #expect(out.asArray(UInt8.self) == bytes)
    }

    /// Matches a naive CPU bicubic implementation pixel for pixel (a high-frequency noise image
    /// also catches tap offset errors).
    @Test func undistortImageMatchesCPUReference() {
        var rng = LCG(state: 5)
        let h = 37, w = 53
        let pixels = (0..<(h * w * 3)).map { _ in rng.next() }
        let source: Intrinsics = (fx: 40, fy: 41, cx: 26.3, cy: 18.1)
        let target: Intrinsics = (fx: 40 * 1.03, fy: 41 * 1.03, cx: 26.3, cy: 18.1)
        let d = RadialDistortion(k1: 0.07, k2: -0.04)
        let out = LensUndistortion.undistortImage(
            MLXArray(pixels, [h, w, 3]), source: source, target: target, distortion: d
        ).asArray(Float.self)
        func weights(_ t: Float) -> [Float] {
            let t2 = t * t, t3 = t2 * t
            return [0.5 * (-t3 + 2 * t2 - t), 0.5 * (3 * t3 - 5 * t2 + 2),
                    0.5 * (-3 * t3 + 4 * t2 + t), 0.5 * (t3 - t2)]
        }
        var maxErr: Float = 0
        for y in 0..<h {
            for x in 0..<w {
                let xu = (Float(x) + 0.5 - target.cx) / target.fx
                let yu = (Float(y) + 0.5 - target.cy) / target.fy
                let f = d.factor(r2: xu * xu + yu * yu)
                let ui = source.fx * xu * f + source.cx - 0.5
                let vi = source.fy * yu * f + source.cy - 0.5
                let x0 = ui.rounded(.down), y0 = vi.rounded(.down)
                let wx = weights(ui - x0), wy = weights(vi - y0)
                for c in 0..<3 {
                    var v: Float = 0
                    for j in 0..<4 {
                        let sy = min(max(Int(y0) + j - 1, 0), h - 1)
                        for i in 0..<4 {
                            let sx = min(max(Int(x0) + i - 1, 0), w - 1)
                            v += wy[j] * wx[i] * pixels[(sy * w + sx) * 3 + c]
                        }
                    }
                    maxErr = max(maxErr, abs(out[(y * w + x) * 3 + c] - min(max(v, 0), 1)))
                }
            }
        }
        #expect(maxErr < 1e-4, "max error \(maxErr)")
    }

    /// Sampling at integer positions returns the pixel itself (bicubic sampler for evaluation).
    @Test func sampleBicubicIsIdentityAtPixelCenters() {
        var rng = LCG(state: 9)
        let h = 6, w = 7
        let pixels = (0..<(h * w * 3)).map { _ in rng.next() }
        let xs = (0..<(h * w)).map { Float($0 % w) }
        let ys = (0..<(h * w)).map { Float($0 / w) }
        let out = LensUndistortion.sampleBicubic(
            MLXArray(pixels, [h, w, 3]), x: MLXArray(xs, [xs.count]), y: MLXArray(ys, [ys.count])
        ).asArray(Float.self)
        for i in pixels.indices { #expect(abs(out[i] - pixels[i]) < 1e-6) }
    }

    /// Captured pixel → model render coordinates: with no distortion and the same K, the index into
    /// a 2x render is 2x + 0.5. With distortion, re-distorting that position returns the captured
    /// pixel.
    @Test func captureToModelCoordinatesRoundTrips() {
        let k = Self.k
        let plain = LensUndistortion.captureToModelCoordinates(
            width: 4, height: 3, capture: k, model: k, distortion: .none, scale: 2)
        #expect(abs(plain.x[1] - 2.5) < 1e-4 && abs(plain.y[4] - 2.5) < 1e-4)

        let d = RadialDistortion(k1: -0.06, k2: 0.015)
        let model: Intrinsics = (fx: k.fx * 0.98, fy: k.fy * 0.98, cx: k.cx, cy: k.cy)
        let map = LensUndistortion.captureToModelCoordinates(
            width: Self.width, height: Self.height, capture: k, model: model, distortion: d, scale: 2)
        for (x, y) in [(0, 0), (320, 240), (639, 10), (100, 479)] {
            let i = y * Self.width + x
            let p = SIMD2(((map.x[i] + 0.5) / 2 - model.cx) / model.fx, ((map.y[i] + 0.5) / 2 - model.cy) / model.fy)
            let back = d.distort(p)
            #expect(abs(k.fx * back.x + k.cx - (Float(x) + 0.5)) < 1e-3)
            #expect(abs(k.fy * back.y + k.cy - (Float(y) + 0.5)) < 1e-3)
        }
    }

    /// Undistorting an image made by distorting a smooth undistorted pattern recovers the original
    /// pattern (checks map direction).
    @Test func undistortImageRecoversPattern() {
        let h = 120, w = 160
        let source: Intrinsics = (fx: 120, fy: 120, cx: 80, cy: 60)
        let d = RadialDistortion(k1: -0.1, k2: 0.01)
        guard let target = LensUndistortion.undistortedIntrinsics(source, width: w, height: h, distortion: d) else {
            Issue.record("no crop found")
            return
        }
        func pattern(_ u: Float, _ v: Float) -> Float { 0.5 + 0.4 * sin(u * 0.08) * cos(v * 0.06) }
        // Distorted image: undistort each pixel center, map it to a target pixel, and sample the
        // pattern.
        var distorted = [Float](repeating: 0, count: h * w * 3)
        for y in 0..<h {
            for x in 0..<w {
                let pd = SIMD2((Float(x) + 0.5 - source.cx) / source.fx, (Float(y) + 0.5 - source.cy) / source.fy)
                let p = d.undistort(pd)!
                let value = pattern(target.fx * p.x + target.cx, target.fy * p.y + target.cy)
                for c in 0..<3 { distorted[(y * w + x) * 3 + c] = value }
            }
        }
        let out = LensUndistortion.undistortImage(
            MLXArray(distorted, [h, w, 3]), source: source, target: target, distortion: d
        ).asArray(Float.self)
        var maxErr: Float = 0
        for y in 3..<(h - 3) {
            for x in 3..<(w - 3) {
                maxErr = max(maxErr, abs(out[(y * w + x) * 3] - pattern(Float(x) + 0.5, Float(y) + 0.5)))
            }
        }
        #expect(maxErr < 0.01, "max error \(maxErr)")
        // Without correction the pattern is offset (confirms the test does not pass trivially).
        var rawErr: Float = 0
        for y in 3..<(h - 3) {
            for x in 3..<(w - 3) {
                rawErr = max(rawErr, abs(distorted[(y * w + x) * 3] - pattern(Float(x) + 0.5, Float(y) + 0.5)))
            }
        }
        #expect(rawErr > 0.1)
    }
}
