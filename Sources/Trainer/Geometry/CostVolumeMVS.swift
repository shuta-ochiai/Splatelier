// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Cost-volume MVS (PocketGS Operator G 3b-6, Sec. III-A 3). Estimates a depth map per target frame
// and returns a dense point cloud unprojected to world (compatible with
// `GaussianModel.fromPointCloudDisc`). Per target: top-M ref selection → depth range → census →
// plane sweep fused over the refs (Mobile3DRecon eq. 7) → SGM (8 paths) → depth filter; then
// multi-view consistency fusion → unproject.

import Foundation
import MLX
import MLXFast

public enum CostVolumeMVS {

    /// Settings for the multi-view depth-fusion consistency filter.
    public struct FusionConsistency: Sendable, Equatable {
        /// Number of neighbor frames to reproject into (top-K of ref selection).
        public var neighborCount: Int
        /// Number of depth-consistent views required to accept.
        public var vMin: Int
        /// Allowed relative error for depth consistency, `|z - D_j| / z`.
        public var relTol: Float
        /// If true, converts the matching neighbors' depths onto the target pixel ray, averages
        /// them with its own depth, and places the point there (the same averaging as COLMAP-style
        /// depth fusion).
        public var averageDepth: Bool
        public init(neighborCount: Int = 8, vMin: Int = 2, relTol: Float = 0.02,
                    averageDepth: Bool = true) {
            precondition(neighborCount >= 1 && vMin >= 1 && relTol > 0)
            self.neighborCount = neighborCount
            self.vMin = vMin
            self.relTol = relTol
            self.averageDepth = averageDepth
        }
    }

    public struct Parameters: Sendable {
        public var refSelection: ReferenceSelection.Parameters
        public var depthHypotheses: Int          // K
        public var sgmPenalties: SGMAggregation.Penalties
        public var confidenceThreshold: Float
        /// See `DepthFilter.filter(excludeRadius:)`.
        public var secondMinExcludeRadius: Int
        public var lowQuantile: Float
        public var highQuantile: Float
        public var lowMultiplier: Float
        public var highMultiplier: Float
        /// Multi-view depth-fusion consistency filter (fusibile / Galliani family). Removes
        /// repeated-pattern mismatches (ray-direction spray) that confidence can't, using
        /// multi-view depth agreement. Points are emitted only through it.
        public var fusionConsistency: FusionConsistency
        /// Voxel dedup edge length (m); `nil` disables (Sec. III-A 3 "fused into P"). Leftover
        /// duplicate points collapse Operator I's K_s=3 mean neighbor distance to 0 and degenerate
        /// the disc seed.
        public var voxelDedupSize: Float?
        /// Census variant shared by target and refs.
        public var censusVariant: CensusTransform.Variant
        /// M: number of refs whose census costs are fused into one cost volume before SGM / WTA
        /// (Mobile3DRecon eq. 7).
        public var referenceCount: Int

        /// The defaults are `mobile3DRecon()`'s.
        public init(
            refSelection: ReferenceSelection.Parameters = ReferenceSelection.Parameters(),
            depthHypotheses: Int = 64,
            sgmPenalties: SGMAggregation.Penalties = SGMAggregation.Penalties(),
            confidenceThreshold: Float = 0.15,
            secondMinExcludeRadius: Int = 2,
            lowQuantile: Float = 0.05,
            highQuantile: Float = 0.95,
            lowMultiplier: Float = 0.5,
            highMultiplier: Float = 1.5,
            fusionConsistency: FusionConsistency = FusionConsistency(),
            voxelDedupSize: Float? = 0.005,
            censusVariant: CensusTransform.Variant = .improved,
            referenceCount: Int = 5
        ) {
            precondition(depthHypotheses >= 2 && depthHypotheses <= SGMAggregation.kMax,
                         "depthHypotheses must be 2..\(SGMAggregation.kMax)")
            precondition(secondMinExcludeRadius >= 0, "secondMinExcludeRadius must be >= 0")
            if let v = voxelDedupSize { precondition(v > 0, "voxelDedupSize must be > 0") }
            self.refSelection = refSelection
            self.depthHypotheses = depthHypotheses
            self.sgmPenalties = sgmPenalties
            self.confidenceThreshold = confidenceThreshold
            self.secondMinExcludeRadius = secondMinExcludeRadius
            self.lowQuantile = lowQuantile
            self.highQuantile = highQuantile
            self.lowMultiplier = lowMultiplier
            self.highMultiplier = highMultiplier
            self.fusionConsistency = fusionConsistency
            self.voxelDedupSize = voxelDedupSize
            self.censusVariant = censusVariant
            self.referenceCount = referenceCount
        }

        /// Defaults for the Mobile3DRecon eq. 7 multi-view cost.
        /// - Parameter m: Number of refs to fuse.
        /// - Parameter confidenceThreshold: 0.15, the value from Mobile3DRecon §4.3.1.
        public static func mobile3DRecon(
            m: Int = 5,
            confidenceThreshold: Float = 0.15,
            voxelDedupSize: Float? = 0.005,
            fusionConsistency: FusionConsistency = FusionConsistency(),
            refSelection: ReferenceSelection.Parameters = ReferenceSelection.Parameters()
        ) -> Parameters {
            Parameters(
                refSelection: refSelection,
                confidenceThreshold: confidenceThreshold,
                fusionConsistency: fusionConsistency,
                voxelDedupSize: voxelDedupSize,
                referenceCount: m
            )
        }
    }

    /// `sceneCenter` is used for the viewing-angle computation in ref selection.
    public struct Inputs {
        public let viewmats: [[Float]]                                            // [F][16]
        public let intrinsics: [Intrinsics]      // [F]
        public let rgbImages: [MLXArray]                                          // [F] × [H, W, 3]
        public let sparsePoints: [SIMD3<Float>]                                    // [P]
        public let sceneCenter: SIMD3<Float>

        public init(
            viewmats: [[Float]],
            intrinsics: [Intrinsics],
            rgbImages: [MLXArray],
            sparsePoints: [SIMD3<Float>],
            sceneCenter: SIMD3<Float>
        ) {
            precondition(viewmats.count == intrinsics.count
                         && intrinsics.count == rgbImages.count)
            self.viewmats = viewmats
            self.intrinsics = intrinsics
            self.rgbImages = rgbImages
            self.sparsePoints = sparsePoints
            self.sceneCenter = sceneCenter
        }
    }

    /// Depth map for one view. The same depth the point cloud was built from (after the consistency
    /// filter, and depth averaging with `averageDepth`).
    public struct DepthView: @unchecked Sendable {
        public let width: Int
        public let height: Int
        /// [H*W] row-major. <= 0 is an invalid pixel.
        public let depth: [Float]
        /// [H*W*3] 8-bit RGB.
        public let rgb: [UInt8]
        public let viewmat: [Float]      // [16] row-major world→camera
        public let intrinsics: Intrinsics

        public init(width: Int, height: Int, depth: [Float], rgb: [UInt8],
                    viewmat: [Float], intrinsics: Intrinsics) {
            precondition(depth.count == width * height && rgb.count == width * height * 3
                         && viewmat.count == 16)
            self.width = width
            self.height = height
            self.depth = depth
            self.rgb = rgb
            self.viewmat = viewmat
            self.intrinsics = intrinsics
        }
    }

    public struct Output {
        public let positions: MLXArray   // [N, 3] float32, world coordinates
        public let colors: MLXArray      // [N, 3] float32 in [0, 1]
        /// Accepted pixel count per frame (for debugging).
        public let perFramePointCount: [Int]
    }

    /// `depthMaps` receives each frame's depth map behind the point cloud (in frame order, as fusion
    /// accepts it) for TSDF fusion when meshing; the caller can stream them out instead of keeping
    /// them all.
    public static func build(
        inputs: Inputs,
        parameters: Parameters = Parameters(),
        depthMaps: ((DepthView) -> Void)? = nil,
        log: (String) -> Void = { print($0) }
    ) -> Output {
        let frameCount = inputs.viewmats.count
        precondition(frameCount >= 2, "need at least 2 frames for stereo MVS")

        let cameraCenters: [SIMD3<Float>] = inputs.viewmats.map {
            FrameSubsampler.cameraCenter(viewmat: $0)
        }

        // Census is computed once per frame and reused.
        var cachedCensus: [MLXArray?] = Array(repeating: nil, count: frameCount)
        func census(for i: Int) -> MLXArray {
            if let c = cachedCensus[i] { return c }
            let luma = CensusTransform.rgbToLuma(inputs.rgbImages[i])
            let c = CensusTransform.compute(luma: luma, variant: parameters.censusVariant)
            cachedCensus[i] = c
            return c
        }

        // With `baselineToDepth` set, per frame the median depth z of the sparse points determines
        // the gaze point (center + optical axis·z) and bTarget = r·z.
        var selCenter = [SIMD3<Float>](repeating: inputs.sceneCenter, count: frameCount)
        var selParams = [ReferenceSelection.Parameters](
            repeating: parameters.refSelection, count: frameCount)
        if parameters.refSelection.baselineToDepth != nil {
            for i in 0..<frameCount {
                let shape = inputs.rgbImages[i].shape
                guard let g = referenceGaze(
                    viewmat: inputs.viewmats[i], intrinsics: inputs.intrinsics[i],
                    width: shape[1], height: shape[0], cameraCenter: cameraCenters[i],
                    sparsePoints: inputs.sparsePoints, parameters: parameters.refSelection
                ) else { continue }
                selCenter[i] = g.center
                selParams[i] = g.parameters
            }
        }

        // With `minCovisibility` set, per-target covisibility (used for ref selection and for
        // choosing neighbors in the consistency check).
        var covisibility = [[Float]?](repeating: nil, count: frameCount)
        if parameters.refSelection.minCovisibility != nil {
            let tCovis = Date()
            defer { if profiling { profile["covisibility(CPU)", default: 0] += Date().timeIntervalSince(tCovis) } }
            let views = (0..<frameCount).map { i -> ReferenceSelection.View in
                let shape = inputs.rgbImages[i].shape
                return ReferenceSelection.View(
                    viewmat: inputs.viewmats[i], intrinsics: inputs.intrinsics[i],
                    width: shape[1], height: shape[0])
            }
            for i in 0..<frameCount {
                covisibility[i] = ReferenceSelection.covisibility(
                    targetIndex: i, views: views, points: inputs.sparsePoints)
            }
        }

        var perFrameCount: [Int] = []
        // The consistency check needs every frame's depth map, so unproject only after all are
        // ready.
        var fusionDepths: [Int: [Float]] = [:]
        var fusionDims: [Int: (w: Int, h: Int)] = [:]
        func emitDepthView(_ i: Int, _ depth: [Float]) {
            let shape = inputs.rgbImages[i].shape
            let rgb8 = (MLX.clip(inputs.rgbImages[i], min: 0.0, max: 1.0) * 255.0 + 0.5)
                .asType(.uint8).asArray(UInt8.self)
            depthMaps?(DepthView(
                width: shape[1], height: shape[0], depth: depth, rgb: rgb8,
                viewmat: inputs.viewmats[i], intrinsics: inputs.intrinsics[i]))
        }
        // For diagnostic logging
        var droppedAtRef = 0
        var droppedAtRange = 0
        var depthFilterKept: [Int] = []

        let referenceCount = max(1, parameters.referenceCount)

        for targetIdx in 0..<frameCount {
            let refIndices = ReferenceSelection.selectTopK(
                targetIndex: targetIdx,
                cameraCenters: cameraCenters,
                sceneCenter: selCenter[targetIdx],
                k: referenceCount,
                parameters: selParams[targetIdx],
                covisibility: covisibility[targetIdx]
            )
            guard !refIndices.isEmpty else {
                perFrameCount.append(0)
                droppedAtRef += 1
                continue
            }

            let tShape = inputs.rgbImages[targetIdx].shape
            let h = tShape[0], w = tShape[1]
            guard let depthRange = DepthRangeEstimator.estimate(
                viewmat: inputs.viewmats[targetIdx],
                imageWidth: w, imageHeight: h,
                intrinsics: inputs.intrinsics[targetIdx],
                points: inputs.sparsePoints,
                lowQuantile: parameters.lowQuantile,
                highQuantile: parameters.highQuantile,
                lowMultiplier: parameters.lowMultiplier,
                highMultiplier: parameters.highMultiplier
            ) else {
                perFrameCount.append(0)
                droppedAtRange += 1
                continue
            }

            // Depth hypotheses evenly spaced in inverse depth (denser at close range)
            let depths = inverseDepthHypotheses(
                range: depthRange, count: parameters.depthHypotheses)

            let intrT = inputs.intrinsics[targetIdx]
            let depthHost = runMultiViewCostMVS(
                targetIdx: targetIdx, refIndices: refIndices, depths: depths,
                census: census, viewmats: inputs.viewmats,
                intrinsics: inputs.intrinsics, parameters: parameters
            )
            depthFilterKept.append(depthHost.lazy.filter { $0 > 0 }.count)

            precondition(intrT.fx > 0 && intrT.fy > 0,
                         "intrinsics fx/fy must be positive (frame \(targetIdx))")
            fusionDepths[targetIdx] = depthHost
            fusionDims[targetIdx] = (w: w, h: h)
            perFrameCount.append(depthHost.lazy.filter { $0 > 0 }.count)  // before fusion

            // Clear the cache once per target (otherwise SGM and other transients accumulate and
            // OOM). Not called inside the ref loop (lazy census got re-evaluated and broke the
            // depth-filter pass rate).
            MLX.Memory.clearCache()
        }

        var positions = MLXArray.zeros([0, 3])
        var colors = MLXArray.zeros([0, 3])
        if !fusionDepths.isEmpty {
            let fp = parameters.fusionConsistency
            let tFuse = Date()
            let (fpos, fcol, kept, total) = fuseWithConsistency(
                depths: fusionDepths, dims: fusionDims,
                viewmats: inputs.viewmats, intrinsics: inputs.intrinsics,
                rgb: { inputs.rgbImages[$0] }, cameraCenters: cameraCenters,
                selCenter: selCenter, selParams: selParams, covisibility: { covisibility[$0] },
                fusion: fp,
                filteredDepths: depthMaps != nil ? { emitDepthView($0, $1) } : nil
            )
            if profiling { profile["fusionConsistency(CPU)", default: 0] += Date().timeIntervalSince(tFuse) }
            log(String(format:
                "[MVS fusion] kept %d/%d points (%.1f%%) via vMin=%d neighbors=%d relTol=%.3f",
                kept, total, total > 0 ? 100.0 * Double(kept) / Double(total) : 0.0,
                fp.vMin, fp.neighborCount, fp.relTol))
            if kept > 0 {
                positions = fpos
                colors = fcol
                eval(positions, colors)
            }
        }
        let preDedupCount = positions.shape[0]
        let zeroFrames = perFrameCount.lazy.filter { $0 == 0 }.count
        let avgDfKept = depthFilterKept.isEmpty
            ? 0 : Double(depthFilterKept.reduce(0, +)) / Double(depthFilterKept.count)
        log(String(
            format: "[MVS diag] frames=%d zero=%d (ref-drop=%d range-drop=%d) "
                  + "avgDepthFilterKept=%.0f preDedup=%d",
            frameCount, zeroFrames, droppedAtRef, droppedAtRange, avgDfKept, preDedupCount
        ))

        if let voxelSize = parameters.voxelDedupSize,
           positions.shape[0] > 0
        {
            let tDedup = Date()
            defer { if profiling { profile["voxelDedup(CPU)", default: 0] += Date().timeIntervalSince(tDedup) } }
            let (fusedPos, fusedCol) = voxelDedup(
                positions: positions, colors: colors, voxelSize: voxelSize
            )
            eval(fusedPos, fusedCol)
            let postDedup = fusedPos.shape[0]
            let compress = preDedupCount > 0 ? Double(preDedupCount) / Double(postDedup) : 0
            log(String(
                format: "[MVS diag] postDedup=%d (compression=%.2fx, voxelSize=%.3fm)",
                postDedup, compress, voxelSize
            ))
            return Output(positions: fusedPos, colors: fusedCol, perFramePointCount: perFrameCount)
        }
        return Output(positions: positions, colors: colors, perFramePointCount: perFrameCount)
    }

    /// Gaze point and parameters for reference selection. The median depth z of the sparse points
    /// visible in the target determines the gaze point (center + optical axis·z) and bTarget = r·z.
    /// nil if `baselineToDepth` is unset or no points are visible (keeps the shared sceneCenter).
    static func referenceGaze(
        viewmat vm: [Float], intrinsics intr: Intrinsics,
        width: Int, height: Int, cameraCenter: SIMD3<Float>,
        sparsePoints: [SIMD3<Float>], parameters: ReferenceSelection.Parameters
    ) -> (center: SIMD3<Float>, parameters: ReferenceSelection.Parameters)? {
        guard let r = parameters.baselineToDepth else { return nil }
        let (h, w) = (Float(height), Float(width))
        var zs: [Float] = []
        for p in sparsePoints {
            let z = vm[8] * p.x + vm[9] * p.y + vm[10] * p.z + vm[11]
            guard z > 1e-3 else { continue }
            let x = vm[0] * p.x + vm[1] * p.y + vm[2] * p.z + vm[3]
            let y = vm[4] * p.x + vm[5] * p.y + vm[6] * p.z + vm[7]
            let u = intr.fx * x / z + intr.cx
            let v = intr.fy * y / z + intr.cy
            if u >= 0, u < w, v >= 0, v < h { zs.append(z) }
        }
        guard !zs.isEmpty else { return nil }
        let zMed = OrderStatistic.select(&zs, zs.count / 2)
        // World direction of the optical axis (cam +z) = row 3 of R.
        let fwd = SIMD3<Float>(vm[8], vm[9], vm[10])
        var p = parameters
        p.bTarget = r * zMed
        p.sigmaB = 0.5 * r * zMed
        return (cameraCenter + zMed * fwd, p)
    }

    /// Hypothesis depths splitting the depth range into `count` steps evenly in inverse depth
    /// (disparity) (denser at close range).
    static func inverseDepthHypotheses(
        range: DepthRangeEstimator.Range, count K: Int
    ) -> [Float] {
        let invMin = 1.0 / range.maxDepth
        let invMax = 1.0 / range.minDepth
        var depths: [Float] = []
        depths.reserveCapacity(K)
        for k in 0..<K {
            let t = (K == 1) ? 0 : Float(k) / Float(K - 1)
            let inv = invMin + t * (invMax - invMin)
            depths.append(1.0 / max(inv, 1e-6))
        }
        return depths
    }

    /// Averages position and color per voxel into one point (CPU).
    static func voxelDedup(
        positions: MLXArray, colors: MLXArray, voxelSize: Float
    ) -> (positions: MLXArray, colors: MLXArray) {
        let n = positions.shape[0]
        let pos = positions.asArray(Float.self)
        let col = colors.asArray(Float.self)
        // Pack voxel indices into an Int64 key, 21 bits each with a bias.
        let bias: Int64 = 1 << 20    // supports ±2^20 voxels (= ±~10 km @ 1 cm)
        var sums: [Int64: (px: Double, py: Double, pz: Double,
                            cr: Double, cg: Double, cb: Double, count: Int)] = [:]
        sums.reserveCapacity(n / 4)
        let invSize = 1.0 / voxelSize
        for i in 0..<n {
            let x = pos[i * 3 + 0]
            let y = pos[i * 3 + 1]
            let z = pos[i * 3 + 2]
            // Drop out-of-range points
            let vx = Int64((x * invSize).rounded(.down))
            let vy = Int64((y * invSize).rounded(.down))
            let vz = Int64((z * invSize).rounded(.down))
            if abs(vx) >= bias || abs(vy) >= bias || abs(vz) >= bias { continue }
            let key = ((vx + bias) << 42) | ((vy + bias) << 21) | (vz + bias)
            var entry = sums[key] ?? (0, 0, 0, 0, 0, 0, 0)
            entry.px += Double(x); entry.py += Double(y); entry.pz += Double(z)
            entry.cr += Double(col[i * 3 + 0]); entry.cg += Double(col[i * 3 + 1])
            entry.cb += Double(col[i * 3 + 2])
            entry.count += 1
            sums[key] = entry
        }
        var outPos: [Float] = []
        var outCol: [Float] = []
        outPos.reserveCapacity(sums.count * 3)
        outCol.reserveCapacity(sums.count * 3)
        for (_, e) in sums {
            let inv = 1.0 / Double(e.count)
            outPos += [Float(e.px * inv), Float(e.py * inv), Float(e.pz * inv)]
            outCol += [Float(e.cr * inv), Float(e.cg * inv), Float(e.cb * inv)]
        }
        let m = sums.count
        if m == 0 {
            return (positions: MLXArray.zeros([0, 3]), colors: MLXArray.zeros([0, 3]))
        }
        return (
            positions: MLXArray(outPos, [m, 3]),
            colors: MLXArray(outCol, [m, 3])
        )
    }

    /// Elementwise median along the ref axis (matches `MLX.sorted(stacked(vols), axis: 0)[n/2]`).
    /// stack + sort is slow, so use a min/max odd-even transposition sort (sorted in n stages) with
    /// only elementwise ops.
    static func elementwiseMedian(_ vols: [MLXArray]) -> MLXArray {
        precondition(!vols.isEmpty)
        var a = vols
        let n = a.count
        for round in 0..<n {
            for i in stride(from: round % 2, to: n - 1, by: 2) {
                let lo = MLX.minimum(a[i], a[i + 1])
                let hi = MLX.maximum(a[i], a[i + 1])
                a[i] = lo
                a[i + 1] = hi
            }
        }
        return a[n / 2]
    }

    /// Records cumulative per-stage time (seconds) only when MVS_PROFILE=1 (each stage evals, so
    /// normally off).
    nonisolated(unsafe) public static var profile: [String: Double] = [:]
    public static let profiling = ProcessInfo.processInfo.environment["MVS_PROFILE"] != nil
    static func timed<T>(_ key: String, _ body: () -> T) -> T {
        guard profiling else { return body() }
        let t0 = Date()
        let r = body()
        profile[key, default: 0] += Date().timeIntervalSince(t0)
        return r
    }

    /// Mobile3DRecon eq. 7: fuses the top-M refs' cost volumes by the median along the ref axis,
    /// then runs SGM once. Out-of-view hypotheses get the max cost 25, so hypotheses seen from more
    /// refs tend to win; the median (unlike the mean) keeps the true depth winning when some refs
    /// do not see it.
    static func runMultiViewCostMVS(
        targetIdx: Int, refIndices: [Int], depths: [Float],
        census: (Int) -> MLXArray,
        viewmats: [[Float]],
        intrinsics: [Intrinsics],
        parameters: Parameters
    ) -> [Float] {
        guard !refIndices.isEmpty else { return [] }
        let targetCensus = timed("census") { () -> MLXArray in
            let c = census(targetIdx); if profiling { eval(c) }; return c }
        let intrT = intrinsics[targetIdx]
        func costVolume(_ rj: Int) -> MLXArray {
            let intrR = intrinsics[rj]
            let rt = PlaneSweepCostVolume.relativePose(
                targetViewmat: viewmats[targetIdx], refViewmat: viewmats[rj]
            )
            return PlaneSweepCostVolume.compute(
                targetCensus: targetCensus, refCensus: census(rj),
                intrinsicsTarget: [intrT.fx, intrT.fy, intrT.cx, intrT.cy],
                intrinsicsRef: [intrR.fx, intrR.fy, intrR.cx, intrR.cy],
                relativeRT: rt, depths: depths
            )
        }
        let fused: MLXArray
        if refIndices.count <= PlaneSweepCostVolume.medianMaxRefs {
            // Compute per-ref cost and the median along the ref axis in one kernel (no intermediate
            // [M, H, W, K]).
            fused = timed("costVolume+median(fused)") { () -> MLXArray in
                let f = PlaneSweepCostVolume.computeMedian(
                    targetCensus: targetCensus, refCensuses: refIndices.map { census($0) },
                    intrinsicsTarget: [intrT.fx, intrT.fy, intrT.cx, intrT.cy],
                    intrinsicsRefs: refIndices.map {
                        [intrinsics[$0].fx, intrinsics[$0].fy, intrinsics[$0].cx, intrinsics[$0].cy]
                    },
                    relativeRTs: refIndices.map {
                        PlaneSweepCostVolume.relativePose(
                            targetViewmat: viewmats[targetIdx], refViewmat: viewmats[$0])
                    },
                    depths: depths
                )
                // No eval except when profiling. fused → SGM → depth filter flows to a single
                // readback without syncs (in the gaps between syncs the GPU clock drops and the
                // following SGM slows down).
                if profiling { eval(f) }
                return f }
        } else {
            // Only when the reference count exceeds the fused kernel's limit: per-volume +
            // elementwise median.
            let vols = refIndices.map { costVolume($0).asType(.float32) }
            fused = elementwiseMedian(vols).asType(.int32)
            eval(fused)
        }
        let agg = timed("sgm") { () -> MLXArray in
            let a = SGMAggregation.aggregate(costVolume: fused, penalties: parameters.sgmPenalties)
            if profiling { eval(a) }; return a }
        return timed("depthFilter+readback") {
            let filt = DepthFilter.filter(
                aggregatedCost: agg, depths: depths,
                threshold: parameters.confidenceThreshold,
                excludeRadius: parameters.secondMinExcludeRadius
            )
            return filt.depthMap.asArray(Float.self)
        }
    }

    /// Multi-view depth-fusion consistency filter (CPU). Maps each valid pixel back to world,
    /// reprojects into the top-K neighbors' depth maps, and keeps only points with at least `vMin`
    /// neighbors whose depth agrees within relative error < relTol. With `targets`, emits points
    /// only for those frames (neighbors are chosen from all `depths`; for `StreamingMVS`).
    /// `filteredDepths` receives per-frame depth maps of accepted pixels only (rejected = -1; the
    /// averaged depth with `averageDepth`). `covisibility` is the per-frame covisibility used for
    /// neighbor selection (see `ReferenceSelection.selectTopK`). The per-pixel test runs on the GPU
    /// (`mvs_fusion_consistency`); points are ordered by pixel on the CPU (same order as the CPU
    /// version `fuseWithConsistencyCPU`, the test reference). With `emitPoints` false only
    /// `filteredDepths` is produced (`rgb` is not read and no points are returned).
    static func fuseWithConsistency(
        depths: [Int: [Float]], dims: [Int: (w: Int, h: Int)],
        viewmats: [[Float]], intrinsics: [Intrinsics],
        rgb: (Int) -> MLXArray, cameraCenters: [SIMD3<Float>],
        selCenter: [SIMD3<Float>], selParams: [ReferenceSelection.Parameters],
        covisibility: (Int) -> [Float]? = { _ in nil },
        fusion: FusionConsistency, targets: [Int]? = nil,
        filteredDepths: ((Int, [Float]) -> Void)? = nil,
        emitPoints: Bool = true
    ) -> (positions: MLXArray, colors: MLXArray, kept: Int, total: Int) {
        let frames = (targets ?? depths.keys.sorted()).filter { dims[$0] != nil && depths[$0] != nil }
        let k = fusion.neighborCount
        var keptPos: [Float] = []
        var keptCol: [Float] = []
        var total = 0
        var accepted = 0
        for start in stride(from: 0, to: frames.count, by: fusionTargetsPerDispatch) {
            let chunk = Array(frames[start..<Swift.min(frames.count, start + fusionTargetsPerDispatch)])
            // Top-K neighbors with depth maps (excluding self, chosen the same way as the CPU
            // version)
            let neighborLists = chunk.map { i in
                ReferenceSelection.selectTopK(
                    targetIndex: i, cameraCenters: cameraCenters, sceneCenter: selCenter[i],
                    k: k, parameters: selParams[i], covisibility: covisibility(i)
                ).filter { $0 != i && depths[$0] != nil }
            }
            // Assign slot numbers to the chunk's targets and neighbor frames, and pass the
            // concatenated depth maps.
            var slotOf: [Int: Int32] = [:]
            var depthFlat: [Float] = []
            var frameInfo: [Float] = []
            var frameDims: [Int32] = []
            for f in chunk + neighborLists.joined() where slotOf[f] == nil {
                slotOf[f] = Int32(slotOf.count)
                let (w, h) = dims[f]!
                frameDims += [Int32(w), Int32(h), Int32(depthFlat.count)]
                depthFlat += depths[f]!
                let it = intrinsics[f]
                frameInfo += viewmats[f][0..<12]
                frameInfo += [it.fx, it.fy, it.cx, it.cy]
            }
            var targetInfo: [Int32] = []
            var outCount = 0
            var maxW = 0, maxH = 0
            for i in chunk {
                let (w, h) = dims[i]!
                targetInfo += [slotOf[i]!, Int32(outCount)]
                outCount += w * h
                maxW = Swift.max(maxW, w)
                maxH = Swift.max(maxH, h)
            }
            var neighborSlots = [Int32](repeating: -1, count: chunk.count * k)
            for (t, list) in neighborLists.enumerated() {
                for (j, f) in list.prefix(k).enumerated() { neighborSlots[t * k + j] = slotOf[f]! }
            }
            let out = fusionKernel(
                [MLXArray(depthFlat), MLXArray(frameInfo), MLXArray(frameDims), MLXArray(targetInfo),
                 MLXArray(neighborSlots),
                 MLXArray([Int32(k), Int32(fusion.vMin), fusion.averageDepth ? 1 : 0]),
                 MLXArray([fusion.relTol])],
                grid: ((maxW + 15) / 16 * 16, (maxH + 15) / 16 * 16, chunk.count),
                threadGroup: (16, 16, 1),
                outputShapes: [[outCount]], outputDTypes: [.float32]
            )[0].asArray(Float.self)

            // Map accepted pixels back to world in pixel order (placed on the target pixel ray at
            // the accepted depth).
            var offset = 0
            for i in chunk {
                let (w, h) = dims[i]!
                let dHost = depths[i]!
                let intr = intrinsics[i]
                let vm = viewmats[i]
                let rgbHost = emitPoints ? rgb(i).asArray(Float.self) : []   // [h*w*3] row-major
                var filtered = filteredDepths != nil ? [Float](repeating: -1, count: w * h) : []
                for pix in 0..<(w * h) {
                    if dHost[pix] > 0 { total += 1 }
                    let z = out[offset + pix]
                    guard z > 0 else { continue }
                    accepted += 1
                    if filteredDepths != nil { filtered[pix] = z }
                    guard emitPoints else { continue }
                    let x = pix % w, y = pix / w
                    let ax = (Float(x) + 0.5 - intr.cx) / intr.fx * z - vm[3]
                    let ay = (Float(y) + 0.5 - intr.cy) / intr.fy * z - vm[7]
                    let az = z - vm[11]
                    keptPos.append(vm[0] * ax + vm[4] * ay + vm[8] * az)
                    keptPos.append(vm[1] * ax + vm[5] * ay + vm[9] * az)
                    keptPos.append(vm[2] * ax + vm[6] * ay + vm[10] * az)
                    keptCol.append(rgbHost[pix * 3])
                    keptCol.append(rgbHost[pix * 3 + 1])
                    keptCol.append(rgbHost[pix * 3 + 2])
                }
                filteredDepths?(i, filtered)
                offset += w * h
            }
        }
        let kept = keptPos.count / 3
        if kept == 0 {
            return (MLXArray.zeros([0, 3]), MLXArray.zeros([0, 3]), accepted, total)
        }
        return (MLXArray(keptPos, [kept, 3]), MLXArray(keptCol, [kept, 3]), accepted, total)
    }

    private static let fusionKernel = SlangCompiledKernel.required("mvs_fusion_consistency")
    /// Targets per consistency-check dispatch (trade-off between depth-map transfer including
    /// neighbors and sync count).
    static let fusionTargetsPerDispatch = 32

    /// CPU version of `fuseWithConsistency` (reference for the GPU version).
    static func fuseWithConsistencyCPU(
        depths: [Int: [Float]], dims: [Int: (w: Int, h: Int)],
        viewmats: [[Float]], intrinsics: [Intrinsics],
        rgb: (Int) -> MLXArray, cameraCenters: [SIMD3<Float>],
        selCenter: [SIMD3<Float>], selParams: [ReferenceSelection.Parameters],
        covisibility: (Int) -> [Float]? = { _ in nil },
        fusion: FusionConsistency, targets: [Int]? = nil,
        filteredDepths: ((Int, [Float]) -> Void)? = nil
    ) -> (positions: MLXArray, colors: MLXArray, kept: Int, total: Int) {
        let frames = targets ?? depths.keys.sorted()
        let relTol = fusion.relTol
        var keptPos: [Float] = []
        var keptCol: [Float] = []
        var total = 0
        for i in frames {
            guard let (w, h) = dims[i], let dHost = depths[i] else { continue }
            let intr = intrinsics[i]
            let vm = viewmats[i]
            let rgbHost = rgb(i).asArray(Float.self)   // [h*w*3] row-major
            // Top-K neighbors with depth maps (excluding self)
            let neighbors = ReferenceSelection.selectTopK(
                targetIndex: i, cameraCenters: cameraCenters, sceneCenter: selCenter[i],
                k: fusion.neighborCount, parameters: selParams[i], covisibility: covisibility(i)
            ).filter { $0 != i && depths[$0] != nil }
            let nData = neighbors.map {
                (d: depths[$0]!, w: dims[$0]!.w, h: dims[$0]!.h,
                 vm: viewmats[$0], intr: intrinsics[$0])
            }
            var filtered = filteredDepths != nil ? [Float](repeating: -1, count: w * h) : []
            for y in 0..<h {
                for x in 0..<w {
                    let pix = y * w + x
                    let d = dHost[pix]
                    if d <= 0 { continue }
                    total += 1
                    // world = Rᵀ(cam - t)
                    let xc = (Float(x) + 0.5 - intr.cx) / intr.fx * d
                    let yc = (Float(y) + 0.5 - intr.cy) / intr.fy * d
                    let dx = xc - vm[3], dy = yc - vm[7], dz = d - vm[11]
                    let wx = vm[0] * dx + vm[4] * dy + vm[8] * dz
                    let wy = vm[1] * dx + vm[5] * dy + vm[9] * dz
                    let wz = vm[2] * dx + vm[6] * dy + vm[10] * dz
                    // Reproject into the neighbor and check depth agreement
                    var conf = 0
                    var depthSum = d
                    for nd in nData {
                        let vj = nd.vm
                        let cz = vj[8] * wx + vj[9] * wy + vj[10] * wz + vj[11]
                        if cz <= 1e-6 { continue }
                        let cxx = vj[0] * wx + vj[1] * wy + vj[2] * wz + vj[3]
                        let cyy = vj[4] * wx + vj[5] * wy + vj[6] * wz + vj[7]
                        let u = nd.intr.fx * cxx / cz + nd.intr.cx
                        let v = nd.intr.fy * cyy / cz + nd.intr.cy
                        // Pixel center = i + 0.5 convention, so the index is floor
                        let ux = Int(u.rounded(.down)), vy = Int(v.rounded(.down))
                        if ux < 0 || ux >= nd.w || vy < 0 || vy >= nd.h { continue }
                        let dj = nd.d[vy * nd.w + ux]
                        if dj <= 0 { continue }
                        if abs(cz - dj) / cz < relTol {
                            conf += 1
                            if fusion.averageDepth {
                                // Map neighbor pixel (ux, vy) back to world at depth dj and convert
                                // to target depth z
                                let ncx = (Float(ux) + 0.5 - nd.intr.cx) / nd.intr.fx * dj
                                let ncy = (Float(vy) + 0.5 - nd.intr.cy) / nd.intr.fy * dj
                                let ex = ncx - vj[3], ey = ncy - vj[7], ez = dj - vj[11]
                                let px = vj[0] * ex + vj[4] * ey + vj[8] * ez
                                let py = vj[1] * ex + vj[5] * ey + vj[9] * ez
                                let pz = vj[2] * ex + vj[6] * ey + vj[10] * ez
                                depthSum += vm[8] * px + vm[9] * py + vm[10] * pz + vm[11]
                            }
                        }
                    }
                    if conf >= fusion.vMin {
                        var (wx, wy, wz) = (wx, wy, wz)
                        if filteredDepths != nil {
                            filtered[pix] = fusion.averageDepth ? depthSum / Float(conf + 1) : d
                        }
                        if fusion.averageDepth {
                            // Re-place on the target pixel ray at the averaged depth
                            let da = depthSum / Float(conf + 1)
                            let ax = (Float(x) + 0.5 - intr.cx) / intr.fx * da - vm[3]
                            let ay = (Float(y) + 0.5 - intr.cy) / intr.fy * da - vm[7]
                            let az = da - vm[11]
                            wx = vm[0] * ax + vm[4] * ay + vm[8] * az
                            wy = vm[1] * ax + vm[5] * ay + vm[9] * az
                            wz = vm[2] * ax + vm[6] * ay + vm[10] * az
                        }
                        keptPos.append(wx); keptPos.append(wy); keptPos.append(wz)
                        keptCol.append(rgbHost[pix * 3])
                        keptCol.append(rgbHost[pix * 3 + 1])
                        keptCol.append(rgbHost[pix * 3 + 2])
                    }
                }
            }
            filteredDepths?(i, filtered)
        }
        let kept = keptPos.count / 3
        if kept == 0 {
            return (MLXArray.zeros([0, 3]), MLXArray.zeros([0, 3]), 0, total)
        }
        return (MLXArray(keptPos, [kept, 3]), MLXArray(keptCol, [kept, 3]), kept, total)
    }
}
