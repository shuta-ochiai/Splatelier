// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import simd
import MLXFast
import MLXRandom

/// Learnable Gaussian parameters. Stored raw (pre-activation): scales = exp(logScales), opacities =
/// sigmoid(logitOpacities). Color uses gsplat-style SH: RGB = clamp(SH_eval + 0.5, 0). sh0 (DC) =
/// (rgb - 0.5) / C0; shN (degree 1) starts at 0 with a lower LR to avoid overfitting the DC.
public final class GaussianModel {

    /// PocketGS Operator I kernel: per-point 3×3 symmetric eigendecomposition (Smith 1961 / Cardano
    /// formula).
    private static let pcaNormalsKernel = SlangCompiledKernel.required("pca_normals_3x3")

    /// PocketGS Operator I KNN kernel. One thread per point keeps the top-K via insertion sort in
    /// registers.
    private static let knnTopKKernel = SlangCompiledKernel.required("knn_topk")
    /// Must match MAX_K in `knn_topk.slang`.
    public static let knnMaxK: Int = 16

    /// SH degree-0 normalization constant `1 / (2√π)` (gsplat's C0).
    public static let shC0: Float = 0.28209479177387814

    public var means: MLXArray              // [N, 3]   world-space centers
    public var quats: MLXArray              // [N, 4]   (w, x, y, z), unnormalized
    public var logScales: MLXArray          // [N, 3]
    public var logitOpacities: MLXArray     // [N]
    public var sh0: MLXArray                // [N, 3]      DC coefficients
    public var shN: MLXArray                // [N, 3, 3]  higher-order coefficients (degree 1)

    /// The last `environmentCount` rows are the environment (far background) sphere: the bg-SH
    /// baked into a far shell on the export copy (`appendBakedBackgroundSphere`). PLYWriter exempts
    /// them from culling and reports them for the background sidecar. 0 during training.
    public var environmentCount: Int = 0

    /// Shallow copy sharing the parameter arrays. Rebinding leaves the original model unchanged, so
    /// the bg-SH bake at export can run without disturbing the training model.
    public func shallowCopy() -> GaussianModel {
        let c = GaussianModel(
            means: means, quats: quats, logScales: logScales,
            logitOpacities: logitOpacities, sh0: sh0, shN: shN
        )
        c.environmentCount = environmentCount
        return c
    }

    public init(
        means: MLXArray,
        quats: MLXArray,
        logScales: MLXArray,
        logitOpacities: MLXArray,
        sh0: MLXArray,
        shN: MLXArray
    ) {
        let n = means.shape[0]
        precondition(means.shape == [n, 3], "means must be [N, 3]")
        precondition(quats.shape == [n, 4], "quats must be [N, 4]")
        precondition(logScales.shape == [n, 3], "logScales must be [N, 3]")
        precondition(logitOpacities.shape == [n], "logitOpacities must be [N]")
        precondition(sh0.shape == [n, 3], "sh0 must be [N, 3]")
        precondition(shN.shape == [n, GaussianModel.shRestBands, 3],
                     "shN must be [N, \(GaussianModel.shRestBands), 3]")
        self.means = means
        self.quats = quats
        self.logScales = logScales
        self.logitOpacities = logitOpacities
        self.sh0 = sh0
        self.shN = shN
    }

    public var count: Int { means.shape[0] }

    /// Number of higher-order foreground SH coefficients (3, since degree 1). sh_eval_deg3.slang
    /// also caps at degree 1.
    public static let shRestBands = 3

    /// Max foreground SH degree. The kernel caps at degree 1, so larger values add no bands.
    public static let maxShDegree = 1

    /// Activated per-axis standard deviations.
    public func scales() -> MLXArray { MLX.exp(logScales) }

    /// Activated opacity (0, 1).
    public func opacities() -> MLXArray { MLX.sigmoid(logitOpacities) }

    /// DC color `clamp(C0·sh0 + 0.5, 0, 1)` for display / export (view-independent).
    public func dcColors() -> MLXArray {
        let rgb = Self.shC0 * sh0 + 0.5
        return MLX.minimum(MLX.maximum(rgb, MLXArray(Float(0))), MLXArray(Float(1)))
    }

    /// Bakes the trained bg-SH (`coeffs` [bands, 3]) into the environment sphere's DC (sh0 =
    /// (C-0.5)/C0, shN=0) for export. Meant to be called once at export. Counted in
    /// environmentCount so PLYWriter exempts it from culling.
    public func appendBakedBackgroundSphere(
        coeffs: MLXArray, degree: Int,
        center: SIMD3<Float>, radius: Float, count m: Int,
        opacity: Float, scaleFactor: Float = 1.2
    ) {
        guard m > 0, radius > 0 else { return }
        let (pos, dir) = Self.fibonacciSphere(center: center, radius: radius, count: m)
        // Convert the raw SH color C_i = basis(dir_i)·coeffs into viewer-convention sh0.
        let dirs = MLXArray(dir, [m, 3])
        let colors = BackgroundSH.colors(dirs: dirs, coeffs: coeffs, degree: degree)   // [m,3]
        appendSphereRows(pos: pos, sh0: (colors - 0.5) / Self.shC0, radius: radius,
                         opacity: opacity, scaleFactor: scaleFactor)
    }

    /// Positions and unit directions (each [m*3]) of `m` points on a Fibonacci sphere with center
    /// `center` and radius `radius`.
    private static func fibonacciSphere(
        center: SIMD3<Float>, radius: Float, count m: Int
    ) -> (pos: [Float], dir: [Float]) {
        var pos = [Float](); pos.reserveCapacity(m * 3)
        var dir = [Float](); dir.reserveCapacity(m * 3)
        let golden = Double.pi * (3.0 - 5.0.squareRoot())   // golden angle
        for i in 0..<m {
            let y = 1.0 - 2.0 * (Double(i) + 0.5) / Double(m)   // [-1, 1]
            let rxy = (Swift.max(0.0, 1.0 - y * y)).squareRoot()
            let theta = golden * Double(i)
            let x = Foundation.cos(theta) * rxy
            let z = Foundation.sin(theta) * rxy
            pos.append(center.x + radius * Float(x))
            pos.append(center.y + radius * Float(y))
            pos.append(center.z + radius * Float(z))
            dir.append(Float(x)); dir.append(Float(y)); dir.append(Float(z))
        }
        return (pos, dir)
    }

    /// Appends sphere rows as the environment sphere. Identity quats, isotropic scale = point
    /// spacing sqrt(4πR²/m) × scaleFactor.
    private func appendSphereRows(
        pos: [Float], sh0 sphereSh0: MLXArray, radius: Float, opacity: Float, scaleFactor: Float
    ) {
        let m = pos.count / 3
        var quatFlat = [Float](repeating: 0, count: m * 4)
        for i in 0..<m { quatFlat[i * 4] = 1 }
        let spacing = radius * Float((4.0 * Double.pi / Double(m)).squareRoot())
        let s = Swift.max(spacing * scaleFactor, 1e-4)
        means = MLX.concatenated([means, MLXArray(pos, [m, 3])], axis: 0)
        quats = MLX.concatenated([quats, MLXArray(quatFlat, [m, 4])], axis: 0)
        logScales = MLX.concatenated(
            [logScales, MLXArray([Float](repeating: Foundation.log(s), count: m * 3), [m, 3])], axis: 0)
        logitOpacities = MLX.concatenated(
            [logitOpacities, MLXArray([Float](repeating: logit(opacity), count: m), [m])], axis: 0)
        sh0 = MLX.concatenated([sh0, sphereSh0], axis: 0)
        shN = MLX.concatenated([shN, MLXArray.zeros([m, Self.shRestBands, 3])], axis: 0)
        environmentCount += m
        eval(means, quats, logScales, logitOpacities, sh0, shN)
    }
}

/// Inverse sigmoid `log(p / (1 - p))`.
public func logit(_ p: Float) -> Float {
    let q = max(min(p, 1 - 1e-6), 1e-6)
    return log(q / (1 - q))
}

extension GaussianModel {

    /// Indices picking exactly `count` of `total` at even spacing (`floor(i·total/count)`).
    /// Avoids the integer-stride behavior where inputs 1-2x the cap get no thinning / get halved.
    static func evenSubsampleIndices(total: Int, count: Int) -> MLXArray {
        precondition(count > 0 && count <= total)
        let idxHost = (0..<count).map { Int32((Int64($0) * Int64(total)) / Int64(count)) }
        return MLXArray(idxHost, [count])
    }

    /// Initializes from a point cloud. Isotropic scale = initScale × RMS distance to the 3 nearest
    /// neighbors, uniformly random quats, low constant opacity.
    ///
    /// - Parameters:
    ///   - colors: `[N, 3]` linear RGB `[0, 1]`.
    ///   - initScale: Multiplier on the 3-NN distance (gsplat: default 1.0, MCMC 0.1).
    ///   - maxPoints: If exceeded, thin evenly to exactly this many.
    public static func fromPointCloud(
        positions: MLXArray,
        colors: MLXArray,
        initOpacity: Float = 0.1,
        initScale: Float = 1.0,
        maxPoints: Int? = nil
    ) -> GaussianModel {
        precondition(positions.shape.count == 2 && positions.shape[1] == 3, "positions must be [N, 3]")
        let total = positions.shape[0]
        precondition(colors.shape == [total, 3], "colors must match positions [N, 3]")

        var pos = positions
        var col = colors
        if let maxPoints, total > maxPoints, maxPoints > 0 {
            let idx = evenSubsampleIndices(total: total, count: maxPoints)
            pos = positions[idx]
            col = colors[idx]
        }
        let n = pos.shape[0]

        let dist = meanDistanceToNearestNeighbors(pos, k: 3)   // [N]
        let safeDist = MLX.maximum(dist, MLXArray(1e-7 as Float))
        let logScale1 = MLX.log(safeDist * initScale).expandedDimensions(axis: -1)   // [N, 1]
        let logScales = MLX.broadcast(logScale1, to: [n, 3])

        // Uniform [0,1)^4 quats as in gsplat. Scatters orientations so MCMC noise = Σ·randn isn't
        // only axis-aligned.
        let quats = MLXRandom.uniform(low: Float(0), high: Float(1), [n, 4])
        let logitVal = logit(initOpacity)
        let logitOpacities = MLXArray([Float](repeating: logitVal, count: n), [n])

        // gsplat rgb_to_sh: (rgb - 0.5) / C0. Higher orders are 0.
        let sh0 = (col - 0.5) / shC0
        let shN = MLXArray.zeros([n, shRestBands, 3])

        eval(pos, sh0, shN, logScales, quats, logitOpacities)
        return GaussianModel(
            means: pos, quats: quats,
            logScales: logScales, logitOpacities: logitOpacities, sh0: sh0, shN: shN
        )
    }

    /// RMS distance `sqrt(mean(d²))` from each point to its k nearest neighbors (excluding itself).
    /// Same definition as gsplat's `dist_avg`.
    static func meanDistanceToNearestNeighbors(_ points: MLXArray, k: Int) -> MLXArray {
        let n = points.shape[0]
        if n <= 1 { return MLXArray([Float](repeating: 0, count: n), [n]) }
        let kk = min(k, n - 1)
        let (_, dists) = knnIndicesAndDistances(points, k: kk)              // [N, kk] distances (not d²)
        let rms = MLX.sqrt((dists * dists).mean(axis: -1))                  // [N]
        eval(rms)
        return rms
    }

    /// k nearest neighbors of each point on the GPU (`knn_topk.slang`). For PocketGS Operator I
    /// (Sec. III-B 1, eq. 5-6).
    /// - Returns: `indices [N, k]` int32 and `distances [N, k]` float32, excluding self, ascending
    ///   by distance.
    static func knnIndicesAndDistances(
        _ points: MLXArray, k: Int
    ) -> (indices: MLXArray, distances: MLXArray) {
        precondition(points.shape.count == 2 && points.shape[1] == 3, "points must be [N, 3]")
        precondition(k >= 1, "k must be >= 1")
        precondition(k <= knnMaxK,
                     "k must be <= knnMaxK (\(knnMaxK)); recompile knn_topk.slang with larger MAX_K")
        let n = points.shape[0]
        if n <= 1 {
            return (
                indices: MLXArray.zeros([n, 0], dtype: .int32),
                distances: MLXArray.zeros([n, 0])
            )
        }
        let kk = min(k, n - 1)
        if n >= knnGridMinPoints, let grid = knnGrid(points, k: kk) {
            return grid
        }
        return knnBruteForce(points, k: kk)
    }

    /// Brute-force k nearest neighbors (`knn_topk.slang`). `k <= N - 1`.
    static func knnBruteForce(_ points: MLXArray, k kk: Int) -> (indices: MLXArray, distances: MLXArray) {
        let n = points.shape[0]
        let posFlat = points.reshaped([n * 3])
        let dimsBuf = MLXArray([Int32(n), Int32(kk)], [2])
        let out = Self.knnTopKKernel(
            [posFlat, dimsBuf],
            grid: (((n + 63) / 64) * 64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[n * kk], [n * kk]],
            outputDTypes: [.int32, .float32]
        )
        let indices = out[0].reshaped([n, kk])
        let distances = out[1].reshaped([n, kk])
        return (indices: indices, distances: distances)
    }

    private static let knnGridKernel = SlangCompiledKernel.required("knn_grid")
    /// At or above this point count, search with a grid (brute force is N², and at the seed cap of
    /// 300k points it dominated init).
    static let knnGridMinPoints = 4096
    /// Target points per grid cell. Surface points within radius h number about π times this, so
    /// even K=16 resolves within 1-2 shells.
    static let knnGridPointsPerCell: Float = 8

    /// k nearest neighbors via a grid (`knn_grid.slang`, same result as brute force). nil for
    /// extreme extents where cell coordinates don't fit.
    static func knnGrid(_ points: MLXArray, k: Int) -> (indices: MLXArray, distances: MLXArray)? {
        let n = points.shape[0]
        let p = points.asArray(Float.self)
        var lo = SIMD3<Float>(repeating: .infinity)
        var hi = SIMD3<Float>(repeating: -.infinity)
        for i in 0..<n {
            let v = SIMD3<Float>(p[i * 3], p[i * 3 + 1], p[i * 3 + 2])
            lo = simd_min(lo, v)
            hi = simd_max(hi, v)
        }
        guard lo.x.isFinite, hi.x.isFinite else { return nil }
        func cellOf(_ i: Int, _ h: Float) -> SIMD3<Int32> {
            SIMD3<Int32>(Int32((p[i * 3] - lo.x) / h), Int32((p[i * 3 + 1] - lo.y) / h), Int32((p[i * 3 + 2] - lo.z) / h))
        }
        // Cell edge: tuned in a few passes so points per non-empty cell hit the target (surface
        // points ∝ h²).
        let extent = hi - lo
        var h = Swift.max(extent.max() / Float(n).squareRoot(), 1e-6)
        for _ in 0..<4 {
            guard extent.max() / h < Float(1 << 20) else { return nil }
            var cells = Set<SIMD3<Int32>>()
            cells.reserveCapacity(n)
            for i in 0..<n { cells.insert(cellOf(i, h)) }
            let ratio = knnGridPointsPerCell / (Float(n) / Float(cells.count))
            if abs(ratio - 1) < 0.25 { break }
            h *= ratio.squareRoot()
        }
        guard extent.max() / h < Float(1 << 20) else { return nil }
        // Order cells lexicographically (x, y, z) and pack points in cell order.
        let cellsOfPoints = (0..<n).map { cellOf($0, h) }
        func key(_ c: SIMD3<Int32>) -> Int64 { Int64(c.x) << 42 | Int64(c.y) << 21 | Int64(c.z) }
        let order = (0..<n).sorted { key(cellsOfPoints[$0]) < key(cellsOfPoints[$1]) }
        var sortedPos = [Float](repeating: 0, count: n * 3)
        var sortedOrig = [Int32](repeating: 0, count: n)
        var cellCoords: [Int32] = []
        var cellStart: [Int32] = []
        var prev: Int64 = -1
        for (s, i) in order.enumerated() {
            sortedPos[s * 3] = p[i * 3]
            sortedPos[s * 3 + 1] = p[i * 3 + 1]
            sortedPos[s * 3 + 2] = p[i * 3 + 2]
            sortedOrig[s] = Int32(i)
            let c = cellsOfPoints[i]
            if key(c) != prev {
                prev = key(c)
                cellCoords += [c.x, c.y, c.z]
                cellStart.append(Int32(s))
            }
        }
        cellStart.append(Int32(n))
        let queryCell = cellsOfPoints.flatMap { [$0.x, $0.y, $0.z] }
        let out = knnGridKernel(
            [points.reshaped([n * 3]), MLXArray(queryCell), MLXArray(sortedPos), MLXArray(sortedOrig),
             MLXArray(cellCoords), MLXArray(cellStart),
             MLXArray([Int32(n), Int32(k), Int32(cellStart.count - 1), 4]), MLXArray([h])],
            grid: (((n + 63) / 64) * 64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[n * k], [n * k]],
            outputDTypes: [.int32, .float32]
        )
        return (indices: out[0].reshaped([n, k]), distances: out[1].reshaped([n, k]))
    }

    /// PocketGS Operator I disc-like initialization (Sec. III-B 1-2, eq. 5-6). The normal is the
    /// smallest eigenvector of the K_n-neighbor covariance, the tangent scale s_t is the arithmetic
    /// mean distance (not RMS) to the K_s neighbors, the normal scale is s_n = rNormal·s_t, and the
    /// quat is the closed form with R(q)·ẑ = n.
    ///
    /// - Parameters:
    ///   - rNormal: s_n / s_t (paper v4 = 0.3).
    ///   - initScale: Multiplier on s_t. 1.0 per the paper; 0.1 for sparse point clouds, where
    ///     large NN
    ///     distances make the discs huge.
    ///   - scaleCap: Absolute cap on s_t (m). `.infinity` per the paper, 0.05 for sparse point
    ///     clouds.
    ///   - maxPoints: If exceeded, thin evenly to exactly this many.
    public static func fromPointCloudDisc(
        positions: MLXArray,
        colors: MLXArray,
        kN: Int = 16,
        kS: Int = 3,
        rNormal: Float = 0.3,
        initOpacity: Float = 0.1,
        initScale: Float = 0.1,
        scaleCap: Float = 0.05,
        maxPoints: Int? = nil
    ) -> GaussianModel {
        precondition(positions.shape.count == 2 && positions.shape[1] == 3, "positions must be [N, 3]")
        let total = positions.shape[0]
        precondition(colors.shape == [total, 3], "colors must match positions [N, 3]")
        precondition(kN >= 3 && kS >= 1, "kN must be >= 3, kS must be >= 1")

        var pos = positions
        var col = colors
        if let maxPoints, total > maxPoints, maxPoints > 0 {
            let idx = evenSubsampleIndices(total: total, count: maxPoints)
            pos = positions[idx]
            col = colors[idx]
        }
        let n = pos.shape[0]

        // One KNN pass serves both K_n and K_s (ascending by distance, so the first kS columns are
        // the K_s neighbors). Normals from eq. 5 via `pca_normals_3x3.slang`.
        precondition(kN >= kS, "kN must be >= kS to share the same KNN pass")
        let (knIdx, knDist) = knnIndicesAndDistances(pos, k: kN)    // [N, kN] int32 / [N, kN] f32
        let posFlat = pos.reshaped([n * 3])                         // [N*3]
        let knIdxFlat = knIdx.reshaped([n * kN])                    // [N*kN]
        let dimsBuf = MLXArray([Int32(n), Int32(kN)], [2])
        let normalsOut = Self.pcaNormalsKernel(
            [posFlat, knIdxFlat, dimsBuf],
            grid: (((n + 63) / 64) * 64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[n * 3]],
            outputDTypes: [.float32]
        )
        let normalsRaw = normalsOut[0].reshaped([n, 3])
        // Materialize here to free the intermediate buffers.
        eval(normalsRaw)
        MLX.Memory.clearCache()
        // Already unit length from the kernel; renormalize against numerical error.
        let normalsNorm = MLX.sqrt(MLX.maximum(
            (normalsRaw * normalsRaw).sum(axis: -1, keepDims: true),
            MLXArray(Float(1e-20))
        ))                                                            // [N, 1]
        let normalsUnit = normalsRaw / normalsNorm                    // [N, 3]

        // s_t = arithmetic mean distance to the K_s neighbors (eq. 6). Detached as contiguous so
        // the slice doesn't pin the parent [N, kN].
        let ksDist = MLX.contiguous(knDist[0..., 0..<kS])             // [N, kS]
        let sT = ksDist.mean(axis: -1)                                // [N]
        let scaled = sT * initScale                                   // [N]
        let cappedST = MLX.minimum(scaled, MLXArray(scaleCap))        // [N]
        let safeST = MLX.maximum(cappedST, MLXArray(Float(1e-7)))

        let sN = rNormal * safeST                                     // [N]
        let safeSN = MLX.maximum(sN, MLXArray(Float(1e-9)))

        // log scale = (s_t, s_t, s_n). The z axis maps to the normal after rotation.
        let logSt = MLX.log(safeST).expandedDimensions(axis: -1)      // [N, 1]
        let logSn = MLX.log(safeSN).expandedDimensions(axis: -1)      // [N, 1]
        let logScales = MLX.concatenated([logSt, logSt, logSn], axis: -1)  // [N, 3]

        let quats = quatFromZToNormals(normalsUnit)                   // [N, 4]

        let logitVal = logit(initOpacity)
        let logitOpacities = MLXArray([Float](repeating: logitVal, count: n), [n])
        let sh0 = (col - 0.5) / shC0
        let shN = MLXArray.zeros([n, shRestBands, 3])

        eval(pos, quats, logScales, logitOpacities, sh0, shN)
        return GaussianModel(
            means: pos, quats: quats,
            logScales: logScales, logitOpacities: logitOpacities, sh0: sh0, shN: shN
        )
    }

    /// Quat (w, x, y, z) with R(q)·ẑ = n from unit normals n [N, 3] (PocketGS Sec. III-B 2).
    /// Half-angle formula gives q ∝ (1 + n_z, -n_y, n_x, 0). Near n ≈ -ẑ the axis is undefined, so
    /// use 180° about the x axis.
    static func quatFromZToNormals(_ normals: MLXArray) -> MLXArray {
        precondition(normals.shape.count == 2 && normals.shape[1] == 3, "normals must be [N, 3]")
        let nx = normals[0..., 0]                                     // [N]
        let ny = normals[0..., 1]
        let nz = normals[0..., 2]
        let onePlusNz = nz + Float(1)                                  // [N]
        let w = onePlusNz                                              // [N]
        let x = -ny                                                    // [N]
        let y = nx                                                     // [N]
        let z = MLXArray.zeros(like: nz)                               // [N]
        let qUnnorm = MLX.stacked([w, x, y, z], axis: -1)              // [N, 4]
        let qNorm = MLX.sqrt(MLX.maximum(
            (qUnnorm * qUnnorm).sum(axis: -1, keepDims: true),
            MLXArray(Float(1e-20))
        ))                                                              // [N, 1]
        let qNormalised = qUnnorm / qNorm                              // [N, 4]
        // Antipodal case (1+nz ≈ 0) is overwritten with (0, 1, 0, 0)
        let n = normals.shape[0]
        let fallback = MLX.broadcast(
            MLXArray([Float(0), 1, 0, 0], [1, 4]), to: [n, 4]
        )                                                              // [N, 4]
        let antipodalMask = (onePlusNz .< MLXArray(Float(1e-6)))
            .expandedDimensions(axis: -1)                               // [N, 1]
        return MLX.which(antipodalMask, fallback, qNormalised)
    }

    /// Random spherical cloud for tests / toy experiments. means ~ N(0, meansStd²), quats near
    /// identity.
    public static func random(
        count: Int,
        meansStd: Float = 0.5,
        initLogScale: Float = -2.0,
        initOpacity: Float = 0.5,
        initColor: Float = 0.5,
        seed: UInt64 = 0
    ) -> GaussianModel {
        MLXRandom.seed(seed)
        let means = MLXRandom.normal([count, 3]) * meansStd
        var quats = MLXRandom.normal([count, 4]) * 0.1
        let identityCol = MLXArray(
            (0..<count).flatMap { _ -> [Float] in [1, 0, 0, 0] },
            [count, 4]
        )
        quats = quats + identityCol
        let logScales = MLXArray(
            [Float](repeating: initLogScale, count: count * 3), [count, 3]
        )
        let logitVal = logit(initOpacity)
        let logitOpacities = MLXArray(
            [Float](repeating: logitVal, count: count), [count]
        )
        let sh0Val = (initColor - 0.5) / shC0
        let sh0 = MLXArray(
            [Float](repeating: sh0Val, count: count * 3), [count, 3]
        )
        let shN = MLXArray.zeros([count, shRestBands, 3])
        eval(means, quats, logScales, logitOpacities, sh0, shN)
        return GaussianModel(
            means: means, quats: quats,
            logScales: logScales, logitOpacities: logitOpacities, sh0: sh0, shN: shN
        )
    }
}
