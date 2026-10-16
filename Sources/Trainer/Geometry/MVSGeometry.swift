// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// CPU helpers for MVS preprocessing (PocketGS Operator G 3b-5, Sec. III-A 3): reference view
// selection and depth range estimation.

import Foundation

// MARK: - Reference view selection

/// MVS reference view selection (ordered by the PocketGS eq. 4 score).
public enum ReferenceSelection {

    /// The paper does not disclose values. `bTarget` (m) / `sigmaB` form the Gaussian baseline
    /// term; `alphaMin` is the lower bound on viewing angle (rad; 5° ≈ 0.087 is a standard lower
    /// bound for dense MVS).
    public struct Parameters: Sendable, Equatable {
        public var bTarget: Float
        public var sigmaB: Float
        public var alphaMin: Float
        /// When set, the baseline is chosen relative to the target's median depth z (`bTarget =
        /// baselineToDepth·z`, `sigmaB = 0.5·bTarget`), and α is also measured at the target's
        /// fixation point (camera center + optical axis·z). With `nil`, it uses the absolute
        /// `bTarget` and a shared sceneCenter, which picks refs that are too wide-angle in
        /// close-range scenes and Census matching fails. Per-target resolution is done by
        /// CostVolumeMVS.
        public var baselineToDepth: Float?
        /// Lower bound on covisibility (`covisibility`: the fraction of sparse points visible in
        /// the target that are also visible in the ref). When set, candidates at or above the bound
        /// are picked first in score order, and the rest are filled in order of covisibility ×
        /// score. `nil` disables it (selection by camera position only). Position alone also picks
        /// refs facing away from the target, and the median of the multi-view cost (out-of-view is
        /// max cost) goes flat where few refs see the pixel, so no depth comes out (2026-10-09
        /// Jonathan: the wall split into 3, and in about 90% of hole pixels at most 2 of the 5 refs
        /// saw the pixel).
        public var minCovisibility: Float?
        public init(
            bTarget: Float = 0.5, sigmaB: Float = 0.3, alphaMin: Float = 0.087,
            baselineToDepth: Float? = 0.2, minCovisibility: Float? = nil
        ) {
            precondition(sigmaB > 0)
            precondition(alphaMin > 0)
            if let r = baselineToDepth { precondition(r > 0) }
            if let c = minCovisibility { precondition(c >= 0 && c <= 1) }
            self.bTarget = bTarget
            self.sigmaB = sigmaB
            self.alphaMin = alphaMin
            self.baselineToDepth = baselineToDepth
            self.minCovisibility = minCovisibility
        }
    }

    /// Camera used for covisibility (viewmat / intrinsics / image size at MVS resolution).
    public struct View {
        public let viewmat: [Float]
        public let intrinsics: Intrinsics
        public let width: Int
        public let height: Int
        public init(viewmat: [Float], intrinsics: Intrinsics, width: Int, height: Int) {
            precondition(viewmat.count == 16, "viewmat must be 16 floats")
            self.viewmat = viewmat
            self.intrinsics = intrinsics
            self.width = width
            self.height = height
        }

        /// Whether `p` is in front (z > 0.05) and lands inside the image.
        @inline(__always)
        func sees(_ p: SIMD3<Float>) -> Bool {
            let m = viewmat
            let z = m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]
            guard z > 0.05 else { return false }
            let u = intrinsics.fx * (m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3]) / z + intrinsics.cx
            let v = intrinsics.fy * (m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7]) / z + intrinsics.cy
            return u >= 0 && u < Float(width) && v >= 0 && v < Float(height)
        }
    }

    /// Fraction of sparse points visible in the target that are also visible in each camera (1 for
    /// the target itself). nil if no points are visible in the target. If more than `maxSamples`
    /// points are visible in the target, they are subsampled at even intervals.
    public static func covisibility(
        targetIndex: Int, views: [View], points: [SIMD3<Float>], maxSamples: Int = 256
    ) -> [Float]? {
        precondition(targetIndex >= 0 && targetIndex < views.count, "targetIndex out of range")
        precondition(maxSamples >= 1)
        let target = views[targetIndex]
        var seen = points.filter { target.sees($0) }
        guard !seen.isEmpty else { return nil }
        if seen.count > maxSamples {
            let n = seen.count
            seen = (0..<maxSamples).map { seen[$0 * n / maxSamples] }
        }
        let inv = 1 / Float(seen.count)
        return views.indices.map { j in
            if j == targetIndex { return 1 }
            let view = views[j]
            var c = 0
            for p in seen where view.sees(p) { c += 1 }
            return Float(c) * inv
        }
    }

    /// PocketGS eq. 4: `S_ref = exp(-(b - b_target)² / 2σ_b²) · max(α / α_min, 1)`.
    public static func score(
        targetCenter: SIMD3<Float>, refCenter: SIMD3<Float>,
        sceneCenter: SIMD3<Float>, parameters: Parameters
    ) -> Float {
        let dx = targetCenter.x - refCenter.x
        let dy = targetCenter.y - refCenter.y
        let dz = targetCenter.z - refCenter.z
        let baseline = (dx * dx + dy * dy + dz * dz).squareRoot()
        let dB = baseline - parameters.bTarget
        let baselineTerm = exp(-(dB * dB) / (2 * parameters.sigmaB * parameters.sigmaB))

        // α = angle between target and ref as seen from sceneCenter.
        func unit(_ v: SIMD3<Float>) -> SIMD3<Float> {
            let m = (v.x * v.x + v.y * v.y + v.z * v.z).squareRoot()
            if m < 1e-9 { return SIMD3<Float>(0, 0, 1) }
            return SIMD3<Float>(v.x / m, v.y / m, v.z / m)
        }
        let toTarget = unit(targetCenter - sceneCenter)
        let toRef = unit(refCenter - sceneCenter)
        let cosAlpha = toTarget.x * toRef.x + toTarget.y * toRef.y + toTarget.z * toRef.z
        let clamped = min(Float(1), max(Float(-1), cosAlpha))
        let alpha = acos(clamped)
        let angleTerm = max(alpha / parameters.alphaMin, 1)

        return baselineTerm * angleTerm
    }

    /// Top-K refs in descending score order (for multi-view sweeps). Returns `min(k, camera count -
    /// 1)` refs. With `minCovisibility` set and `covisibility` passed (in camera order, from
    /// `covisibility(targetIndex:views:points:)`), candidates meeting the covisibility bound are
    /// prioritized.
    public static func selectTopK(
        targetIndex: Int,
        cameraCenters: [SIMD3<Float>],
        sceneCenter: SIMD3<Float>,
        k: Int,
        parameters: Parameters = Parameters(),
        covisibility: [Float]? = nil
    ) -> [Int] {
        precondition(targetIndex >= 0 && targetIndex < cameraCenters.count,
                     "targetIndex out of range")
        precondition(k >= 1, "k must be >= 1")
        guard cameraCenters.count >= 2 else { return [] }
        let tc = cameraCenters[targetIndex]
        var scored: [(Int, Float)] = []
        scored.reserveCapacity(cameraCenters.count - 1)
        for (i, rc) in cameraCenters.enumerated() where i != targetIndex {
            let s = score(targetCenter: tc, refCenter: rc,
                          sceneCenter: sceneCenter, parameters: parameters)
            scored.append((i, s))
        }
        scored.sort { $0.1 > $1.1 }
        if let minCov = parameters.minCovisibility, let cov = covisibility {
            precondition(cov.count == cameraCenters.count, "covisibility must have one value per camera")
            let passing = scored.filter { cov[$0.0] >= minCov }
            let rest = scored.filter { cov[$0.0] < minCov }.sorted {
                let a = cov[$0.0] * $0.1, b = cov[$1.0] * $1.1
                return a != b ? a > b : $0.1 > $1.1
            }
            scored = passing + rest
        }
        return scored.prefix(k).map { $0.0 }
    }
}

// MARK: - Depth range estimation

/// Sets the search range [q_0.05·0.5, q_0.95·1.5] from quantiles of sparse-point depths projected
/// into the target (PocketGS Sec. III-A 3). The factors are a margin for BA residual noise.
public enum DepthRangeEstimator {

    /// `minDepth < maxDepth`.
    public struct Range: Sendable, Equatable {
        public let minDepth: Float
        public let maxDepth: Float
    }

    /// Uses only points inside the frustum (inside the image and z > 0). nil if there are no valid
    /// points.
    public static func estimate(
        viewmat: [Float],
        imageWidth: Int, imageHeight: Int,
        intrinsics: Intrinsics,
        points: [SIMD3<Float>],
        lowQuantile: Float = 0.05, highQuantile: Float = 0.95,
        lowMultiplier: Float = 0.5, highMultiplier: Float = 1.5
    ) -> Range? {
        precondition(viewmat.count == 16, "viewmat must be 16 floats")
        precondition(0 <= lowQuantile && lowQuantile <= highQuantile && highQuantile <= 1)
        let fx = intrinsics.fx
        let fy = intrinsics.fy
        let cx = intrinsics.cx
        let cy = intrinsics.cy

        var depths: [Float] = []
        depths.reserveCapacity(points.count)
        for p in points {
            let xc = viewmat[0] * p.x + viewmat[1] * p.y + viewmat[2]  * p.z + viewmat[3]
            let yc = viewmat[4] * p.x + viewmat[5] * p.y + viewmat[6]  * p.z + viewmat[7]
            let zc = viewmat[8] * p.x + viewmat[9] * p.y + viewmat[10] * p.z + viewmat[11]
            if zc <= 1e-3 { continue }
            let uProj = fx * xc / zc + cx
            let vProj = fy * yc / zc + cy
            if uProj < 0 || uProj >= Float(imageWidth) { continue }
            if vProj < 0 || vProj >= Float(imageHeight) { continue }
            depths.append(zc)
        }
        guard !depths.isEmpty else { return nil }
        let n = depths.count
        let lowI = max(0, min(n - 1, Int(Float(n) * lowQuantile)))
        let highI = max(0, min(n - 1, Int(Float(n) * highQuantile)))
        // Select only the values at two ranks without sorting (sorting tens of thousands of points
        // per target took about 1/4 of MVS). Selecting highI first leaves only values <= highI
        // before it, so lowI can be selected from that part.
        let highDepth = OrderStatistic.select(&depths, highI)
        let lowDepth = lowI < highI ? OrderStatistic.select(&depths, lowI, in: 0..<highI) : highDepth
        let lo = lowDepth * lowMultiplier
        let hi = highDepth * highMultiplier
        // In the degenerate case hi <= lo (e.g. a single point), make hi slightly larger than lo
        let safeHi = (hi > lo) ? hi : (lo * 1.01 + 1e-3)
        return Range(minDepth: lo, maxDepth: safeHi)
    }
}

// MARK: - Order statistics

/// Finds the value `a[k]` would have after sorting, without sorting everything (quickselect,
/// average O(n)).
enum OrderStatistic {
    /// Returns the value that lands at position `k` when `range` (default: all) is sorted.
    /// Afterwards `a[k]` holds that value, and within `range` everything before `k` is <= it and
    /// everything after is >= it. Uses a 3-way partition to stay linear with many duplicates.
    static func select(_ a: inout [Float], _ k: Int, in range: Range<Int>? = nil) -> Float {
        var lo = range?.lowerBound ?? 0
        var hi = (range?.upperBound ?? a.count) - 1
        precondition(lo <= k && k <= hi, "k out of range")
        var rounds = 0
        while lo < hi {
            rounds += 1
            if rounds > 64 {
                // After a run of bad pivots, sort the rest (avoids worst-case O(n²))
                a[lo...hi].sort()
                return a[k]
            }
            let x = a[lo], y = a[lo + (hi - lo) / 2], z = a[hi]
            let pivot = max(min(x, y), min(max(x, y), z))   // median of three
            // [lo, lt) < pivot, [lt, i) == pivot, (gt, hi] > pivot
            var lt = lo, i = lo, gt = hi
            while i <= gt {
                let v = a[i]
                if v < pivot {
                    a.swapAt(lt, i); lt += 1; i += 1
                } else if v > pivot {
                    a.swapAt(i, gt); gt -= 1
                } else {
                    i += 1
                }
            }
            if k < lt {
                hi = lt - 1
            } else if k > gt {
                lo = gt + 1
            } else {
                return pivot
            }
        }
        return a[k]
    }
}
