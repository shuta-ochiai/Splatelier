// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Lens radial distortion (k1, k2) estimation and image undistortion. The model matches OpenCV /
// COLMAP RADIAL: in normalized camera coordinates, x_d = x_u·(1 + k1 r² + k2 r⁴), r² = |x_u|².
// Estimation: fix the post-BA poses and solve points and k1, k2 jointly with Gauss-Newton (points
// Schur-eliminated to 2×2). Fixing the points too would underestimate k by the distortion the
// points absorbed in BA and make σ optimistic. The poses also absorb part of the distortion, so BA
// is rerun on undistorted observations and k re-estimated (alternating, `refineAlternating`).
// Images: for each output (undistorted) pixel, look up the input (distorted) position and resample
// with Catmull-Rom bicubic.

import Foundation
import MLX
import MLXFast
import simd

/// Radial distortion coefficients (normalized camera coordinates). The intrinsics are used as is,
/// as the undistorted pinhole.
public struct RadialDistortion: Sendable, Equatable {
    public var k1: Float
    public var k2: Float

    public init(k1: Float, k2: Float) {
        self.k1 = k1
        self.k2 = k2
    }

    public static let none = RadialDistortion(k1: 0, k2: 0)
    public var isNone: Bool { k1 == 0 && k2 == 0 }

    /// Factor 1 + k1 r² + k2 r⁴ for r² of undistorted normalized coordinates.
    public func factor(r2: Float) -> Float { 1 + k1 * r2 + k2 * r2 * r2 }

    /// Undistorted normalized coordinates → distorted.
    public func distort(_ p: SIMD2<Float>) -> SIMD2<Float> {
        p * factor(r2: simd_length_squared(p))
    }

    /// Inverse of `distort`. Solves r·f(r²) = r_d with 1D Newton on the radius r (step capped at
    /// max(0.1 r, 0.1)). Returns nil when it enters a non-monotonic range or fails to converge
    /// (that observation is not used).
    public func undistort(_ pd: SIMD2<Float>) -> SIMD2<Float>? {
        let rd = Double(simd_length(pd))
        guard !isNone, rd > 0 else { return pd }
        let a = Double(k1), b = Double(k2)
        var r = rd
        for _ in 0..<50 {
            let r2 = r * r
            let g = r * (1 + a * r2 + b * r2 * r2) - rd
            let dg = 1 + 3 * a * r2 + 5 * b * r2 * r2
            guard dg > 1e-6 else { return nil }
            let maxStep = max(0.1 * r, 0.1)
            let step = min(max(g / dg, -maxStep), maxStep)
            r = max(r - step, 0)
            if abs(step) < 1e-10 { return pd * Float(r / rd) }
        }
        return nil
    }

    /// Distorted pixel → undistorted pixel with the same intrinsics. nil if the inverse cannot be
    /// found.
    public func undistortPixel(_ uv: SIMD2<Float>, _ k: Intrinsics) -> SIMD2<Float>? {
        guard let p = undistort(SIMD2((uv.x - k.cx) / k.fx, (uv.y - k.cy) / k.fy)) else { return nil }
        return SIMD2(k.fx * p.x + k.cx, k.fy * p.y + k.cy)
    }

    /// Radial displacement (normalized coordinates) at undistorted radius r: r·(k1 r² + k2 r⁴).
    /// Positive means outward (pincushion).
    public func radialShift(atRadius r: Float) -> Float { r * (factor(r2: r * r) - 1) }

    /// Whether the distorted radius r·f(r²) increases monotonically over undistorted radius [0, r]
    /// (d/dr = 1 + 3k1 r² + 5k2 r⁴). Without monotonicity the inverse is not unique.
    public func isMonotonic(upToRadius r: Float) -> Bool {
        for i in 0...64 {
            let s = r * Float(i) / 64
            let s2 = s * s
            if 1 + 3 * k1 * s2 + 5 * k2 * s2 * s2 <= 0.1 { return false }
        }
        return true
    }

    /// Whether the "undistort → redistort" round-trip error at the bin centers of a 9×9 grid over
    /// the image is within 0.001·max(W, H) px (the same criterion as COLMAP's single-view
    /// calibration sanity check).
    public func roundTripValid(_ k: Intrinsics, width: Int, height: Int) -> Bool {
        let tolerance = 0.001 * Float(max(width, height))
        for j in 0..<9 {
            for i in 0..<9 {
                let uv = SIMD2((Float(i) + 0.5) * Float(width) / 9, (Float(j) + 0.5) * Float(height) / 9)
                guard let p = undistort(SIMD2((uv.x - k.cx) / k.fx, (uv.y - k.cy) / k.fy)) else { return false }
                let back = distort(p)
                let err = SIMD2(k.fx * back.x + k.cx, k.fy * back.y + k.cy) - uv
                guard simd_length(err) <= tolerance else { return false }
            }
        }
        return true
    }
}

public enum LensDistortionEstimation {

    public struct Parameters: Sendable {
        public var enabled: Bool
        /// Maximum number of "estimate → redo BA on undistorted observations" rounds.
        public var maxAlternations: Int
        /// Maximum Gauss-Newton iterations when solving points and k jointly.
        public var maxIterations: Int
        /// Huber threshold for IRLS (px).
        public var huberThresholdPx: Float
        /// During estimation, observations with residuals above this are dropped as mismatches
        /// (px). Image-edge observations that the IGR reprojection threshold (2px) dropped are
        /// still distortion cues, so this is wider than IGR.
        public var maxResidualPx: Float
        /// Conditions on points used for estimation: observation count (track length) and maximum
        /// triangulation angle. Two-view points have only 4 observations for 3 point DOF and absorb
        /// distortion easily, so they are excluded (COLMAP's ignore_two_view_tracks /
        /// filter_min_tri_angle).
        public var minTrackLength: Int
        public var minTriangulationAngleDegrees: Float
        public var minObservations: Int
        /// Minimum number of outer observations (radius ≥ `outerRadiusFraction` of the corner
        /// radius). Without edge observations k is undetermined.
        public var minOuterObservations: Int
        public var outerRadiusFraction: Float
        /// Do not correct if the correction at the corner (px) is below this (resampling blur would
        /// be larger).
        public var minCornerShiftPx: Float
        /// Treat the estimate as broken if the corner correction exceeds this fraction of the
        /// distance to the corner.
        public var maxCornerShiftFraction: Float
        /// The corner correction is significant if it is at least this many times its standard
        /// error (with points marginalized).
        public var minSignificance: Float
        /// From the second round on, converged when the change in corner correction is below this
        /// (px).
        public var convergencePx: Float

        public init(
            enabled: Bool = true,
            maxAlternations: Int = 2,
            maxIterations: Int = 10,
            huberThresholdPx: Float = 3,
            maxResidualPx: Float = 20,
            minTrackLength: Int = 3,
            minTriangulationAngleDegrees: Float = 1.5,
            minObservations: Int = 1000,
            minOuterObservations: Int = 200,
            outerRadiusFraction: Float = 0.6,
            minCornerShiftPx: Float = 0.5,
            maxCornerShiftFraction: Float = 0.05,
            minSignificance: Float = 3,
            convergencePx: Float = 0.2
        ) {
            precondition(maxAlternations >= 1 && maxIterations >= 1)
            precondition(minTrackLength >= 2)
            self.enabled = enabled
            self.maxAlternations = maxAlternations
            self.maxIterations = maxIterations
            self.huberThresholdPx = huberThresholdPx
            self.maxResidualPx = maxResidualPx
            self.minTrackLength = minTrackLength
            self.minTriangulationAngleDegrees = minTriangulationAngleDegrees
            self.minObservations = minObservations
            self.minOuterObservations = minOuterObservations
            self.outerRadiusFraction = outerRadiusFraction
            self.minCornerShiftPx = minCornerShiftPx
            self.maxCornerShiftFraction = maxCornerShiftFraction
            self.minSignificance = minSignificance
            self.convergencePx = convergencePx
        }

        public static let disabled = Parameters(enabled: false)
    }

    public struct Estimate: Sendable {
        public let distortion: RadialDistortion
        /// Number of points and observations used for estimation, and how many of those
        /// observations are outer.
        public let pointCount: Int
        public let observationCount: Int
        public let outerObservationCount: Int
        /// Correction at the corner (the image corner farthest from the principal point) in px,
        /// signed, and its standard error.
        public let cornerShiftPx: Float
        public let cornerShiftSigmaPx: Float
        /// nil means accepted.
        public let rejectReason: String?
        public var accepted: Bool { rejectReason == nil }

        public var summary: String {
            String(format: "k1=%+.5f k2=%+.5f corner shift %+.2f±%.2f px (points %d, obs %d, outer %d)",
                   distortion.k1, distortion.k2, cornerShiftPx, cornerShiftSigmaPx,
                   pointCount, observationCount, outerObservationCount)
        }
    }

    /// Estimation problem built from the BA result (poses and points) and distorted observations.
    /// Poses are fixed; points and k1, k2 are solved (the same "fixed poses" form as COLMAP's
    /// `point_triangulator --refine_intrinsics`).
    public struct Problem {
        struct Camera {
            let rotation: simd_double3x3
            let translation: SIMD3<Double>
            let fx: Double, fy: Double, cx: Double, cy: Double
            /// Normalized radius of the corner.
            let cornerRadius: Double
        }
        struct Observation {
            let camera: Int
            let pixel: SIMD2<Double>
        }

        let cameras: [Camera]
        /// Initial point positions used for estimation (from BA) and per-point observations (CSR:
        /// the observations of points[j] are observations[observationStart[j] ..<
        /// observationStart[j + 1]]).
        let points: [SIMD3<Double>]
        let observationStart: [Int]
        let observations: [Observation]
        /// Corner normalized radius (max over all frames) and representative focal length (median
        /// fx). Used to convert corrections to px.
        let cornerRadius: Float
        let referenceFocal: Float
        /// Intrinsics and size of the frame with the largest corner radius (for the round-trip
        /// error check).
        let referenceIntrinsics: Intrinsics
        let referenceSize: (width: Int, height: Int)

        /// - Parameters:
        ///   - observations: Observations in distorted pixels (before going into BA).
        ///   - inlierObservations: Observations kept by BA (IGR). Only points that keep 2+ views
        ///     here are used
        ///     (triangulation of the others is not trustworthy). Pixel values are not read.
        public init(
            state: BundleAdjustment.State,
            intrinsics: [Intrinsics],
            sizes: [(width: Int, height: Int)],
            observations: [BundleAdjustment.Observation],
            inlierObservations: [BundleAdjustment.Observation],
            parameters: Parameters = .init()
        ) {
            precondition(intrinsics.count == state.viewmats.count && sizes.count == intrinsics.count)
            let cornerRadii: [Double] = zip(intrinsics, sizes).map { k, s in
                var r = 0.0
                for (u, v) in [(0, 0), (s.width, 0), (0, s.height), (s.width, s.height)] {
                    let x = (Double(u) - Double(k.cx)) / Double(k.fx)
                    let y = (Double(v) - Double(k.cy)) / Double(k.fy)
                    r = max(r, (x * x + y * y).squareRoot())
                }
                return r
            }
            let ref = cornerRadii.indices.max { cornerRadii[$0] < cornerRadii[$1] } ?? 0
            cornerRadius = Float(cornerRadii.isEmpty ? 1 : cornerRadii[ref])
            referenceIntrinsics = intrinsics.isEmpty ? (1, 1, 0, 0) : intrinsics[ref]
            referenceSize = sizes.isEmpty ? (1, 1) : sizes[ref]
            let focals = intrinsics.map(\.fx).sorted()
            referenceFocal = focals.isEmpty ? 1 : focals[focals.count / 2]

            let cams: [Camera] = state.viewmats.indices.map { i in
                let vm = state.viewmats[i].map(Double.init)
                let k = intrinsics[i]
                return Camera(
                    rotation: simd_double3x3(rows: [SIMD3(vm[0], vm[1], vm[2]),
                                                    SIMD3(vm[4], vm[5], vm[6]),
                                                    SIMD3(vm[8], vm[9], vm[10])]),
                    translation: SIMD3(vm[3], vm[7], vm[11]),
                    fx: Double(k.fx), fy: Double(k.fy), cx: Double(k.cx), cy: Double(k.cy),
                    cornerRadius: cornerRadii[i])
            }
            cameras = cams

            var inlierCount = [Int](repeating: 0, count: state.points.count)
            for o in inlierObservations { inlierCount[o.pointIndex] += 1 }
            var byPoint = [[BundleAdjustment.Observation]](repeating: [], count: state.points.count)
            for o in observations where inlierCount[o.pointIndex] >= 2 { byPoint[o.pointIndex].append(o) }

            let minCosAngle = cos(Double(parameters.minTriangulationAngleDegrees) * .pi / 180)
            var pts: [SIMD3<Double>] = []
            var start: [Int] = [0]
            var obs: [Observation] = []
            for (j, list) in byPoint.enumerated() where list.count >= parameters.minTrackLength {
                let x = SIMD3<Double>(state.points[j])
                // Keep only observations in front of the camera and measure the widest viewing-ray
                // spread (smallest cos).
                var rays: [SIMD3<Double>] = []
                var kept: [Observation] = []
                for o in list {
                    let c = cams[o.cameraIndex]
                    guard (c.rotation * x + c.translation).z > 1e-3 else { continue }
                    let center = -(c.rotation.transpose * c.translation)
                    rays.append(simd_normalize(x - center))
                    kept.append(Observation(camera: o.cameraIndex, pixel: SIMD2<Double>(o.pixel)))
                }
                guard kept.count >= parameters.minTrackLength else { continue }
                var minCos = 1.0
                for a in 0..<rays.count {
                    for b in (a + 1)..<rays.count { minCos = min(minCos, simd_dot(rays[a], rays[b])) }
                }
                guard minCos <= minCosAngle else { continue }
                pts.append(x)
                obs.append(contentsOf: kept)
                start.append(obs.count)
            }
            points = pts
            observationStart = start
            self.observations = obs
        }

        /// Correction at the corner (px).
        public func cornerShiftPx(_ d: RadialDistortion) -> Float {
            referenceFocal * d.radialShift(atRadius: cornerRadius)
        }

        /// Residual of one observation (observed − predicted, px) and the Jacobian of the
        /// prediction w.r.t. k and the point. nil if behind the camera.
        private func linearize(
            _ c: Camera, _ x: SIMD3<Double>, _ k: SIMD2<Double>, _ pixel: SIMD2<Double>
        ) -> (r: SIMD2<Double>, jk: simd_double2x2, jx: simd_double3x2, radiusFraction: Double)? {
            let pc = c.rotation * x + c.translation
            guard pc.z > 1e-6 else { return nil }
            let iz = 1 / pc.z
            let xu = pc.x * iz, yu = pc.y * iz
            let r2 = xu * xu + yu * yu, r4 = r2 * r2
            let f = 1 + k.x * r2 + k.y * r4
            let fp = k.x + 2 * k.y * r2   // df/d(r²)
            let r = pixel - SIMD2(c.fx * xu * f + c.cx, c.fy * yu * f + c.cy)
            let jk = simd_double2x2(rows: [SIMD2(c.fx * xu * r2, c.fx * xu * r4),
                                           SIMD2(c.fy * yu * r2, c.fy * yu * r4)])
            // d(predicted)/d(xu, yu) · d(xu, yu)/d(Pc) · d(Pc)/d(X)
            let dPixel = simd_double2x2(rows: [SIMD2(c.fx * (f + 2 * xu * xu * fp), c.fx * 2 * xu * yu * fp),
                                               SIMD2(c.fy * 2 * xu * yu * fp, c.fy * (f + 2 * yu * yu * fp))])
            let dNormalized = simd_double3x2(rows: [SIMD3(iz, 0, -xu * iz), SIMD3(0, iz, -yu * iz)])
            let jx = simd_mul(simd_mul(dPixel, dNormalized), c.rotation)
            return (r, jk, jx, r2.squareRoot() / c.cornerRadius)
        }

        /// Solves points and k jointly with Huber-IRLS Gauss-Newton and decides acceptance. σ comes
        /// from the covariance of k with points marginalized, σ²·S⁻¹ (S is the 2×2 after
        /// Schur-eliminating the points).
        public func estimate(_ parameters: Parameters) -> Estimate {
            let nPoints = points.count
            var x = points
            var k = SIMD2<Double>(0, 0)
            var schur = simd_double2x2(0)
            var used = 0, outer = 0, usedPoints = 0
            var weightedSq = 0.0
            var hxx = [simd_double3x3](repeating: simd_double3x3(0), count: nPoints)
            var hkx = [simd_double3x2](repeating: simd_double3x2(0), count: nPoints)
            var bx = [SIMD3<Double>](repeating: .zero, count: nPoints)
            var hxxInv = [simd_double3x3?](repeating: nil, count: nPoints)
            let huber = Double(parameters.huberThresholdPx)
            let maxResidual = Double(parameters.maxResidualPx)

            for _ in 0..<parameters.maxIterations {
                schur = simd_double2x2(0)
                var g = SIMD2<Double>(0, 0)
                used = 0; outer = 0; usedPoints = 0; weightedSq = 0
                for j in 0..<nPoints {
                    var hj = simd_double3x3(0), hkj = simd_double3x2(0)
                    var hkk = simd_double2x2(0)
                    var bj = SIMD3<Double>(0, 0, 0), bkj = SIMD2<Double>(0, 0)
                    var nj = 0, outerJ = 0, sqJ = 0.0
                    for o in observations[observationStart[j]..<observationStart[j + 1]] {
                        guard let lin = linearize(cameras[o.camera], x[j], k, o.pixel) else { continue }
                        let n = simd_length(lin.r)
                        guard n <= maxResidual else { continue }
                        let w = n <= huber ? 1 : huber / n
                        hkk += w * simd_mul(lin.jk.transpose, lin.jk)
                        bkj += w * simd_mul(lin.jk.transpose, lin.r)
                        hj += w * simd_mul(lin.jx.transpose, lin.jx)
                        hkj += w * simd_mul(lin.jk.transpose, lin.jx)
                        bj += w * simd_mul(lin.jx.transpose, lin.r)
                        sqJ += w * n * n
                        nj += 1
                        if lin.radiusFraction >= Double(parameters.outerRadiusFraction) { outerJ += 1 }
                    }
                    // Points with fewer than 2 observations or degenerate ones cannot move freely,
                    // so they are not used.
                    hxxInv[j] = nil
                    guard nj >= 2 else { continue }
                    // Tiny Levenberg damping for numerical stability.
                    let damp = 1e-9 * (hj[0, 0] + hj[1, 1] + hj[2, 2]) + 1e-12
                    let damped = hj + simd_double3x3(diagonal: SIMD3(repeating: damp))
                    guard abs(damped.determinant) > 1e-30 else { continue }
                    let inv = damped.inverse
                    hxxInv[j] = inv
                    hxx[j] = hj; hkx[j] = hkj; bx[j] = bj
                    let t = simd_mul(hkj, inv)
                    schur += hkk - simd_mul(t, hkj.transpose)
                    g += bkj - simd_mul(t, bj)
                    used += nj; outer += outerJ; weightedSq += sqJ; usedPoints += 1
                }
                guard abs(schur.determinant) > 1e-30 else { break }
                let dk = simd_mul(schur.inverse, g)
                k += dk
                for j in 0..<nPoints {
                    guard let inv = hxxInv[j] else { continue }
                    x[j] += simd_mul(inv, bx[j] - simd_mul(hkx[j].transpose, dk))
                }
                if max(abs(dk.x), abs(dk.y)) < 1e-9 { break }
            }

            let d = RadialDistortion(k1: Float(k.x), k2: Float(k.y))
            let shift = cornerShiftPx(d)
            var sigma = Float.infinity
            let dof = 2 * used - 3 * usedPoints - 2
            if dof > 0, abs(schur.determinant) > 1e-30 {
                // Cov(k) = σ² S⁻¹, with σ² the weighted residual per degree of freedom.
                let variance = weightedSq / Double(dof)
                let rc = Double(cornerRadius)
                let grad = Double(referenceFocal) * SIMD2(rc * rc * rc, rc * rc * rc * rc * rc)
                sigma = Float((variance * simd_dot(grad, simd_mul(schur.inverse, grad))).squareRoot())
            }

            let reason: String?
            if used < parameters.minObservations {
                reason = "too few observations (\(used) < \(parameters.minObservations))"
            } else if outer < parameters.minOuterObservations {
                reason = "too few observations near the image edge (\(outer) < \(parameters.minOuterObservations))"
            } else if !d.isMonotonic(upToRadius: 1.5 * cornerRadius) {
                reason = "non-monotonic distortion"
            } else if !d.roundTripValid(referenceIntrinsics, width: referenceSize.width,
                                        height: referenceSize.height) {
                reason = "round-trip check failed"
            } else if abs(shift) > parameters.maxCornerShiftFraction * referenceFocal * cornerRadius {
                reason = String(format: "implausibly large corner shift (%.1f px)", shift)
            } else if !(abs(shift) >= parameters.minSignificance * sigma) {
                reason = String(format: "not significant (%.2f px < %.0fσ = %.2f px)",
                                abs(shift), parameters.minSignificance, parameters.minSignificance * sigma)
            } else if abs(shift) < parameters.minCornerShiftPx {
                reason = String(format: "corner shift %.2f px < %.2f px (not worth resampling)",
                                abs(shift), parameters.minCornerShiftPx)
            } else {
                reason = nil
            }
            return Estimate(distortion: d, pointCount: usedPoints, observationCount: used,
                            outerObservationCount: outer, cornerShiftPx: shift,
                            cornerShiftSigmaPx: sigma, rejectReason: reason)
        }

        /// Median reprojection error per radial band (fraction of the distance to the corner), with
        /// points kept at their BA positions. Residual distortion makes it grow toward the outside.
        public func radialProfile(
            distortion: RadialDistortion, bins: Int = 5, maxResidualPx: Float = 20
        ) -> [(upperFraction: Float, count: Int, medianPx: Float)] {
            let k = SIMD2(Double(distortion.k1), Double(distortion.k2))
            var byBin = [[Float]](repeating: [], count: bins)
            for j in points.indices {
                for o in observations[observationStart[j]..<observationStart[j + 1]] {
                    guard let lin = linearize(cameras[o.camera], points[j], k, o.pixel) else { continue }
                    let n = Float(simd_length(lin.r))
                    guard n <= maxResidualPx else { continue }
                    byBin[min(bins - 1, Int(lin.radiusFraction * Double(bins)))].append(n)
                }
            }
            return byBin.enumerated().map { i, values in
                let sorted = values.sorted()
                return (Float(i + 1) / Float(bins), sorted.count,
                        sorted.isEmpty ? 0 : sorted[sorted.count / 2])
            }
        }

        public func radialProfileSummary(distortion: RadialDistortion, maxResidualPx: Float = 20) -> String {
            var lower: Float = 0
            return radialProfile(distortion: distortion, maxResidualPx: maxResidualPx).map { bin in
                defer { lower = bin.upperFraction }
                return String(format: "r %.0f-%.0f%%: n=%d med %.2fpx",
                              lower * 100, bin.upperFraction * 100, bin.count, bin.medianPx)
            }.joined(separator: " | ")
        }
    }

    /// Alternates BA (IGR) and distortion estimation. `initial` is the result solved on the raw
    /// distorted observations. If a distortion is accepted, the last BA is the one solved on
    /// observations with that distortion removed (consistent with the returned state).
    /// - Parameter solve: Takes observations (pixels) and redoes BA from the initial poses.
    public static func refineAlternating(
        initial: IterativeGeometricRefinement.Result,
        intrinsics: [Intrinsics],
        sizes: [(width: Int, height: Int)],
        observations: [BundleAdjustment.Observation],
        parameters: Parameters,
        solve: ([BundleAdjustment.Observation]) -> IterativeGeometricRefinement.Result,
        log: (String) -> Void
    ) -> (result: IterativeGeometricRefinement.Result, distortion: RadialDistortion) {
        guard parameters.enabled else { return (initial, .none) }
        var result = initial
        var applied = RadialDistortion.none
        func makeProblem() -> Problem {
            Problem(state: result.state, intrinsics: intrinsics, sizes: sizes,
                    observations: observations, inlierObservations: result.observations,
                    parameters: parameters)
        }
        var problem = makeProblem()
        log("lens distortion: residual by radius (pinhole): "
            + problem.radialProfileSummary(distortion: .none, maxResidualPx: parameters.maxResidualPx))
        for round in 0..<parameters.maxAlternations {
            if round > 0 { problem = makeProblem() }
            let est = problem.estimate(parameters)
            guard est.accepted else {
                log("lens distortion estimate #\(round): \(est.summary) — rejected: \(est.rejectReason ?? "")"
                    + (applied.isNone ? "; not applied" : "; keeping previous estimate"))
                break
            }
            log("lens distortion estimate #\(round): \(est.summary)")
            if !applied.isNone,
               abs(est.cornerShiftPx - problem.cornerShiftPx(applied)) < parameters.convergencePx {
                log("lens distortion: converged")
                break
            }
            applied = est.distortion
            // Drop observations whose inverse cannot be found (normally does not happen for a
            // distortion that passed acceptance).
            let undistorted: [BundleAdjustment.Observation] = observations.compactMap { o in
                applied.undistortPixel(o.pixel, intrinsics[o.cameraIndex]).map {
                    BundleAdjustment.Observation(cameraIndex: o.cameraIndex, pointIndex: o.pointIndex, pixel: $0)
                }
            }
            let dropped = observations.count - undistorted.count
            log("lens distortion: re-running BA on undistorted observations"
                + (dropped > 0 ? " (\(dropped) obs dropped: undistortion did not converge)" : ""))
            result = solve(undistorted)
        }
        if !applied.isNone {
            log(String(format: "lens distortion applied: k1=%+.5f k2=%+.5f; residual by radius (corrected): ",
                       applied.k1, applied.k2)
                + makeProblem().radialProfileSummary(distortion: applied, maxResidualPx: parameters.maxResidualPx))
        }
        return (result, applied)
    }
}

public enum LensUndistortion {

    /// Pinhole intrinsics after undistortion. Keeps the principal point and scales the focal length
    /// by s. s is the smallest value at which every output pixel samples from within the input's
    /// pixel-center range (no black border, maximum field of view). Barrel gives s < 1 (wider field
    /// of view), pincushion s > 1. nil if none is found in s ∈ [0.7, 1.5].
    public static func undistortedIntrinsics(
        _ k: Intrinsics, width: Int, height: Int, distortion: RadialDistortion
    ) -> Intrinsics? {
        guard !distortion.isNone else { return k }
        let w = Float(width), h = Float(height)
        let eps: Float = 1e-3
        func inside(_ u: Float, _ v: Float, _ s: Float) -> Bool {
            let p = distortion.distort(SIMD2((u - k.cx) / (k.fx * s), (v - k.cy) / (k.fy * s)))
            let ui = k.fx * p.x + k.cx, vi = k.fy * p.y + k.cy
            return ui >= 0.5 - eps && ui <= w - 0.5 + eps && vi >= 0.5 - eps && vi <= h - 0.5 + eps
        }
        // If the mapping is monotonic, all interior pixels fit inside the input whenever all border
        // pixels do.
        func valid(_ s: Float) -> Bool {
            for x in 0..<width {
                let u = Float(x) + 0.5
                if !inside(u, 0.5, s) || !inside(u, h - 0.5, s) { return false }
            }
            for y in 0..<height {
                let v = Float(y) + 0.5
                if !inside(0.5, v, s) || !inside(w - 0.5, v, s) { return false }
            }
            return true
        }
        var lo: Float = 0.7, hi: Float = 1.5
        guard valid(hi) else { return nil }
        if !valid(lo) {
            for _ in 0..<30 {
                let mid = 0.5 * (lo + hi)
                if valid(mid) { hi = mid } else { lo = mid }
            }
        } else {
            hi = lo
        }
        return (fx: k.fx * hi, fy: k.fy * hi, cx: k.cx, cy: k.cy)
    }

    private static let kernel = SlangCompiledKernel.required("lens_undistort")

    /// Samples `image` ([H, W, C] float) with Catmull-Rom bicubic at coordinates `x`, `y` ([N]) in
    /// index space (pixel centers at integers) and returns [N, C]. Out-of-range taps are filled
    /// with edge pixels. For evaluation (speed does not matter).
    public static func sampleBicubic(_ image: MLXArray, x: MLXArray, y: MLXArray) -> MLXArray {
        let h = image.shape[0], w = image.shape[1], c = image.shape[2]
        let x0 = MLX.floor(x), y0 = MLX.floor(y)
        func weights(_ t: MLXArray) -> [MLXArray] {
            let t2 = t * t, t3 = t2 * t
            return [0.5 * (-t3 + 2 * t2 - t),
                    0.5 * (3 * t3 - 5 * t2 + 2),
                    0.5 * (-3 * t3 + 4 * t2 + t),
                    0.5 * (t3 - t2)]
        }
        let wx = weights(x - x0), wy = weights(y - y0)
        let src = image.asType(.float32).reshaped([h * w, c])
        var acc = MLXArray.zeros([x.shape[0], c], dtype: .float32)
        for j in 0..<4 {
            let yi = MLX.clip(y0 + Float(j - 1), min: 0, max: Float(h - 1))
            for i in 0..<4 {
                let xi = MLX.clip(x0 + Float(i - 1), min: 0, max: Float(w - 1))
                let idx = (yi * Float(w) + xi).asType(.int32)
                acc = acc + src.take(idx, axis: 0) * (wy[j] * wx[i]).reshaped([-1, 1])
            }
            eval(acc)
        }
        return acc
    }

    /// For each pixel center of the captured image (intrinsics `capture`, distortion `distortion`),
    /// where it lands (index space, row-major [H·W]) in an image of the undistorted model
    /// (intrinsics `model`) rendered at `scale` times the resolution. Pixels whose inverse cannot
    /// be found use the position that ignores distortion.
    public static func captureToModelCoordinates(
        width: Int, height: Int, capture: Intrinsics, model: Intrinsics,
        distortion: RadialDistortion, scale: Float
    ) -> (x: [Float], y: [Float]) {
        var xs = [Float](repeating: 0, count: width * height)
        var ys = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                let pd = SIMD2((Float(x) + 0.5 - capture.cx) / capture.fx,
                               (Float(y) + 0.5 - capture.cy) / capture.fy)
                let p = distortion.undistort(pd) ?? pd
                xs[y * width + x] = scale * (model.fx * p.x + model.cx) - 0.5
                ys[y * width + x] = scale * (model.fy * p.y + model.cy) - 0.5
            }
        }
        return (xs, ys)
    }

    /// Resamples the distorted image `image` ([H, W, 3], uint8 or float [0,1]) into an undistorted
    /// image with intrinsics `target` using Catmull-Rom bicubic (`slang/lens_undistort.slang`).
    /// Output has the same dtype and size as the input. Out-of-range is filled with edge pixels.
    /// Does not evaluate (returns lazily).
    public static func undistortImage(
        _ image: MLXArray, source: Intrinsics, target: Intrinsics, distortion: RadialDistortion
    ) -> MLXArray {
        precondition(image.shape.count == 3 && image.shape[2] == 3, "image must be [H, W, 3]")
        let h = image.shape[0], w = image.shape[1]
        let params = MLXArray([source.fx, source.fy, source.cx, source.cy,
                               target.fx, target.fy, target.cx, target.cy,
                               distortion.k1, distortion.k2], [10])
        let out = kernel(
            [image.asType(.float32).reshaped([h * w * 3]), params, MLXArray([Int32(w), Int32(h)], [2])],
            grid: (w, h, 1),
            threadGroup: (16, 16, 1),
            outputShapes: [[h * w * 3]],
            outputDTypes: [.float32]
        )[0].reshaped([h, w, 3])
        if image.dtype == .uint8 {
            return (MLX.clip(out, min: 0, max: 255) + 0.5).asType(.uint8)
        }
        return MLX.clip(out, min: 0, max: 1).asType(image.dtype)
    }
}
