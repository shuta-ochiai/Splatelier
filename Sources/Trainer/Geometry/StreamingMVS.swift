// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Incremental MVS during capture (Mobile3DRecon-style per-keyframe reconstruction). Each time a
// keyframe arrives:
//   1. Compute and keep that frame's Census (it becomes a reference candidate for later targets)
//   2. Compute one depth map for the frame that arrived `lookahead` frames earlier as the target.
//      References are
//      chosen from all frames received so far with the same reference selection as CostVolumeMVS (=
//      up to lookahead frames ahead)
//   3. For a frame whose depth maps are available up to `fusionDelay` frames ahead, run cross-view
//      consistency
//      (FusionConsistency) against all depth maps at that point and emit points
// The building blocks (reference selection / depth range / multi-view cost / SGM / consistency) are
// the same as CostVolumeMVS's batch path. Setting lookahead / fusionDelay to Int.max defers
// everything to `finish()`, which emits the same point cloud in the same order as
// `CostVolumeMVS.build` (to verify that making it incremental did not change the result).

import Foundation
import MLX
import simd

public final class StreamingMVS {
    public struct Parameters: Sendable {
        public var mvs: CostVolumeMVS.Parameters
        /// How many later frames to wait for before computing the target's depth (number of future
        /// reference candidates).
        public var lookahead: Int
        /// How many frames ahead of depth maps to wait for before the consistency check (number of
        /// future check partners).
        public var fusionDelay: Int

        public init(mvs: CostVolumeMVS.Parameters, lookahead: Int = 2, fusionDelay: Int = 2) {
            precondition(lookahead >= 0 && fusionDelay >= 0)
            self.mvs = mvs
            self.lookahead = lookahead
            self.fusionDelay = fusionDelay
        }
    }

    /// What one `addFrame` / `finish` call processed, and how long it took (seconds).
    public struct Step: Sendable {
        public var censusSeconds: Double = 0
        public var depthSeconds: Double = 0
        public var fusionSeconds: Double = 0
        /// Frames whose depth map was computed in this step (including frames that produced no
        /// depth).
        public var depthTargets: [Int] = []
        /// Frames that emitted points (were consistency-checked) in this step.
        public var fusedFrames: [Int] = []
        /// Number of points emitted by each frame in `fusedFrames` (`positions` is laid out per
        /// frame in this order).
        public var fusedPointCounts: [Int] = []
        public var newPoints = 0
        /// Points emitted in this step (world xyz, flattened) and colors (0..1 rgb, flattened).
        public var positions: [Float] = []
        public var colors: [Float] = []
        /// Only with `emitsSurfels`: per-point normals (world xyz, flattened, facing the camera of
        /// the frame that emitted the point) and the size of the one pixel that saw the point
        /// (depth / fx, in m; ignores surface slant).
        public var normals: [Float] = []
        public var pixelSizes: [Float] = []
        /// Number of points whose normal could not be derived from depth differences and was
        /// replaced by the reversed view direction.
        public var fallbackNormals = 0
        /// Only with `emitsSurfels`: the consistency-checked depth map of the frame that emitted
        /// points (MVS resolution, ≤ 0 where there is no point). Used by the capture-time display
        /// to remove splats floating in front of this surface.
        public var fusedDepths: [(frame: Int, width: Int, height: Int, depth: [Float])] = []
        public var surfelSeconds: Double = 0
    }

    private struct Frame {
        let viewmat: [Float]
        let intrinsics: Intrinsics
        /// [h, w, 3] float 0..1. Color is used only in the frame's own consistency check, so it is
        /// dropped after the check.
        var image: MLXArray?
        let census: MLXArray
        let center: SIMD3<Float>
        let width: Int
        let height: Int
    }

    public let parameters: Parameters
    private var frames: [Frame] = []
    private var sparsePoints: [SIMD3<Float>] = []
    private var depthMaps: [Int: [Float]] = [:]
    private var dims: [Int: (w: Int, h: Int)] = [:]
    private var selCenter: [Int: SIMD3<Float>] = [:]
    private var selParams: [Int: ReferenceSelection.Parameters] = [:]
    /// The most recently computed covisibility (reused if frame and sparse-point counts are
    /// unchanged; live, the depth and consistency steps run on the same input back to back).
    private var covisibilityCache: (target: Int, frames: Int, points: Int, value: [Float]?)?
    /// Frames for which depth was attempted (in computation order), including frames that produced
    /// no depth.
    private var depthOrder: [Int] = []
    private var nextFuse = 0
    private var positionChunks: [MLXArray] = []
    private var colorChunks: [MLXArray] = []

    /// Whether to keep consistency-checked points for `cloud()`. Set to false to save memory when
    /// only each step's `Step.positions` is used, as in the live capture display.
    public let retainsCloud: Bool

    /// Whether to also emit normals and pixel sizes (`Step.normals` / `Step.pixelSizes`) for each
    /// consistency-checked point (for the capture-time surfel display).
    public var emitsSurfels = false
    /// Pixel distance (in MVS-resolution pixels) at which depths are compared when computing
    /// normals.
    public var surfelNormalRadius = 2
    /// If the neighboring pixel's depth differs by more than this relative amount, treat it as a
    /// depth step and do not use it for the normal.
    public var surfelMaxRelativeJump: Float = 0.05

    public init(parameters: Parameters, retainsCloud: Bool = true) {
        self.parameters = parameters
        self.retainsCloud = retainsCloud
    }

    public var frameCount: Int { frames.count }
    /// Number of consistency-checked points (before dedup).
    public private(set) var pointCount = 0

    /// Adds sparse points (ARKit feature points, etc.) used for estimating the depth range and
    /// fixation point.
    public func addSparsePoints(_ points: [SIMD3<Float>]) {
        sparsePoints.append(contentsOf: points)
    }

    /// Adds one keyframe and advances depth computation and consistency checks as far as the
    /// arrived frames allow.
    /// - Parameter image: `[h, w, 3]` float (0..1), already downscaled to MVS resolution.
    @discardableResult
    public func addFrame(
        image: MLXArray, viewmat: [Float],
        intrinsics: Intrinsics
    ) -> Step {
        var step = Step()
        let t0 = Date()
        let luma = CensusTransform.rgbToLuma(image)
        let census = CensusTransform.compute(luma: luma, variant: parameters.mvs.censusVariant)
        eval(census)
        step.censusSeconds = Date().timeIntervalSince(t0)
        frames.append(Frame(
            viewmat: viewmat, intrinsics: intrinsics, image: image, census: census,
            center: FrameSubsampler.cameraCenter(viewmat: viewmat),
            width: image.shape[1], height: image.shape[0]))

        if parameters.lookahead < Int.max {
            let target = frames.count - 1 - parameters.lookahead
            if target >= 0 { computeDepth(target, step: &step) }
        }
        if parameters.fusionDelay < Int.max {
            while nextFuse + parameters.fusionDelay < depthOrder.count {
                fuse(depthOrder[nextFuse], step: &step)
                nextFuse += 1
            }
        }
        return step
    }

    /// End of capture: compute depth for the remaining targets (with all arrived frames as
    /// candidates) and consistency-check everything left.
    @discardableResult
    public func finish() -> Step {
        var step = Step()
        let next = depthOrder.last.map { $0 + 1 } ?? 0
        for t in next..<frames.count { computeDepth(t, step: &step) }
        while nextFuse < depthOrder.count {
            fuse(depthOrder[nextFuse], step: &step)
            nextFuse += 1
        }
        return step
    }

    /// Consistency-checked point cloud. `dedup` applies voxel averaging at
    /// `parameters.mvs.voxelDedupSize` (same dedup as the batch `CostVolumeMVS.build`).
    public func cloud(dedup: Bool = true) -> (positions: MLXArray, colors: MLXArray) {
        guard !positionChunks.isEmpty else {
            return (MLXArray.zeros([0, 3]), MLXArray.zeros([0, 3]))
        }
        let pos = MLX.concatenated(positionChunks, axis: 0)
        let col = MLX.concatenated(colorChunks, axis: 0)
        eval(pos, col)
        if dedup, let v = parameters.mvs.voxelDedupSize {
            return CostVolumeMVS.voxelDedup(positions: pos, colors: col, voxelSize: v)
        }
        return (pos, col)
    }

    /// Centroid of the camera centers (the shared center for reference selection on frames without
    /// a fixation point; same computation as the batch `TrainingDriver.runGeometryPrior`).
    public static func cameraCentroid(_ viewmats: [[Float]]) -> SIMD3<Float> {
        var cx: Float = 0, cy: Float = 0, cz: Float = 0
        for vm in viewmats {
            let c = FrameSubsampler.cameraCenter(viewmat: vm)
            cx += c.x; cy += c.y; cz += c.z
        }
        let invN = 1.0 / Float(max(viewmats.count, 1))
        return SIMD3<Float>(cx * invN, cy * invN, cz * invN)
    }

    /// Downscales `[H, W, 3]` (uint8 or float) by 1/`factor` with a box average into a float 0..1
    /// image (same computation as CaptureTraining's MVS input downscale). The caller scales
    /// intrinsics by 1/factor.
    public static func boxDownscale(_ image: MLXArray, factor s: Int) -> MLXArray {
        let img = image.dtype == .uint8 ? image.asType(.float32) / 255.0 : image
        guard s > 1 else { return img }
        let h = img.shape[0] / s, w = img.shape[1] / s
        return img[0..<(h * s), 0..<(w * s)]
            .reshaped([h, s, w, s, 3]).mean(axes: [1, 3])
    }

    // MARK: - private

    private func computeDepth(_ t: Int, step: inout Step) {
        let t0 = Date()
        defer {
            step.depthSeconds += Date().timeIntervalSince(t0)
            step.depthTargets.append(t)
            depthOrder.append(t)
            MLX.Memory.clearCache()
        }
        let p = parameters.mvs
        let f = frames[t]
        let available = frames.count
        let centers = frames.map(\.center)
        var center = StreamingMVS.cameraCentroid(frames.map(\.viewmat))
        var sel = p.refSelection
        if let g = CostVolumeMVS.referenceGaze(
            viewmat: f.viewmat, intrinsics: f.intrinsics, width: f.width, height: f.height,
            cameraCenter: f.center, sparsePoints: sparsePoints, parameters: p.refSelection
        ) {
            center = g.center
            sel = g.parameters
        }
        selCenter[t] = center
        selParams[t] = sel
        let refs = ReferenceSelection.selectTopK(
            targetIndex: t, cameraCenters: centers, sceneCenter: center,
            k: max(1, p.referenceCount), parameters: sel, covisibility: covisibility(t))
        guard !refs.isEmpty, available >= 2,
              let range = DepthRangeEstimator.estimate(
                viewmat: f.viewmat, imageWidth: f.width, imageHeight: f.height,
                intrinsics: f.intrinsics, points: sparsePoints,
                lowQuantile: p.lowQuantile, highQuantile: p.highQuantile,
                lowMultiplier: p.lowMultiplier, highMultiplier: p.highMultiplier)
        else { return }
        let depths = CostVolumeMVS.inverseDepthHypotheses(range: range, count: p.depthHypotheses)
        let depth = CostVolumeMVS.runMultiViewCostMVS(
            targetIdx: t, refIndices: refs, depths: depths,
            census: { self.frames[$0].census },
            viewmats: frames.map(\.viewmat), intrinsics: frames.map(\.intrinsics),
            parameters: p)
        depthMaps[t] = depth
        dims[t] = (w: f.width, h: f.height)
    }

    /// Covisibility of frame `t` against all arrived frames (nil if `minCovisibility` is not set).
    private func covisibility(_ t: Int) -> [Float]? {
        guard parameters.mvs.refSelection.minCovisibility != nil else { return nil }
        if let c = covisibilityCache, c.target == t, c.frames == frames.count, c.points == sparsePoints.count {
            return c.value
        }
        let views = frames.map {
            ReferenceSelection.View(
                viewmat: $0.viewmat, intrinsics: $0.intrinsics, width: $0.width, height: $0.height)
        }
        let value = ReferenceSelection.covisibility(targetIndex: t, views: views, points: sparsePoints)
        covisibilityCache = (t, frames.count, sparsePoints.count, value)
        return value
    }

    private func fuse(_ i: Int, step: inout Step) {
        defer { frames[i].image = nil }
        guard depthMaps[i] != nil else { return }
        let fusion = parameters.mvs.fusionConsistency
        let t0 = Date()
        let n = frames.count
        let centers = frames.map(\.center)
        let fallback = selCenter[i] ?? centers[i]
        let cov = covisibility(i)   // recomputed against all frames that have arrived at consistency-check time
        var filtered: [Float] = []
        let (pos, col, kept, _) = CostVolumeMVS.fuseWithConsistency(
            depths: depthMaps, dims: dims,
            viewmats: frames.map(\.viewmat), intrinsics: frames.map(\.intrinsics),
            rgb: { self.frames[$0].image! }, cameraCenters: centers,
            selCenter: (0..<n).map { selCenter[$0] ?? fallback },
            selParams: (0..<n).map { selParams[$0] ?? parameters.mvs.refSelection },
            covisibility: { $0 == i ? cov : nil },
            fusion: fusion, targets: [i],
            filteredDepths: emitsSurfels ? { filtered = $1 } : nil)
        if kept > 0 {
            if retainsCloud {
                positionChunks.append(pos)
                colorChunks.append(col)
            }
            step.positions += pos.asArray(Float.self)
            step.colors += col.asArray(Float.self)
            pointCount += kept
            if emitsSurfels { appendSurfels(i, filtered: filtered, step: &step) }
        }
        step.fusionSeconds += Date().timeIntervalSince(t0)
        step.fusedFrames.append(i)
        step.fusedPointCounts.append(kept)
        step.newPoints += kept
    }

    /// Attaches normals and pixel sizes to the consistency-checked points of frame `i` (pixels with
    /// depth > 0 in `filtered`; pixel order = point order of `fuseWithConsistency`). Neighboring
    /// pixels dropped by the consistency check are filled from the original depth map (so normals
    /// do not go missing at surface edges).
    private func appendSurfels(_ i: Int, filtered: [Float], step: inout Step) {
        guard let (w, h) = dims[i], let raw = depthMaps[i], filtered.count == w * h else { return }
        let t0 = Date()
        let intr = frames[i].intrinsics
        let vm = frames[i].viewmat
        let depthAt = { (pix: Int) -> Float in filtered[pix] > 0 ? filtered[pix] : raw[pix] }
        for pix in 0..<(w * h) where filtered[pix] > 0 {
            let x = pix % w, y = pix / w
            let z = filtered[pix]
            let ray = SIMD3<Float>((Float(x) + 0.5 - intr.cx) / intr.fx, (Float(y) + 0.5 - intr.cy) / intr.fy, 1)
            let n = Self.depthNormal(
                x: x, y: y, z: z, w: w, h: h, intrinsics: intr, radius: surfelNormalRadius,
                maxRelativeJump: surfelMaxRelativeJump, depth: depthAt)
            if n == nil { step.fallbackNormals += 1 }
            let nc = n ?? -simd_normalize(ray)
            // The camera → world rotation is the transpose of the viewmat rotation
            step.normals.append(vm[0] * nc.x + vm[4] * nc.y + vm[8] * nc.z)
            step.normals.append(vm[1] * nc.x + vm[5] * nc.y + vm[9] * nc.z)
            step.normals.append(vm[2] * nc.x + vm[6] * nc.y + vm[10] * nc.z)
            step.pixelSizes.append(z / intr.fx)
        }
        step.fusedDepths.append((frame: i, width: w, height: h, depth: filtered))
        step.surfelSeconds += Date().timeIntervalSince(t0)
    }

    /// Normal at depth-map pixel (x, y) with depth `z` (camera coordinates, unit vector facing the
    /// camera). Pixels `radius` away to the left/right and up/down are lifted back to surface
    /// points and their differences used as tangent vectors. If one side is missing or is a depth
    /// step (relative difference > `maxRelativeJump`), the pixel itself is used instead; nil if
    /// neither both left/right nor both up/down are usable.
    static func depthNormal(
        x: Int, y: Int, z: Float, w: Int, h: Int, intrinsics intr: Intrinsics, radius r: Int,
        maxRelativeJump: Float, depth: (Int) -> Float
    ) -> SIMD3<Float>? {
        func point(_ px: Int, _ py: Int, _ d: Float) -> SIMD3<Float> {
            SIMD3<Float>((Float(px) + 0.5 - intr.cx) / intr.fx * d, (Float(py) + 0.5 - intr.cy) / intr.fy * d, d)
        }
        func usable(_ px: Int, _ py: Int) -> SIMD3<Float>? {
            guard px >= 0, px < w, py >= 0, py < h else { return nil }
            let d = depth(py * w + px)
            guard d > 0, abs(d - z) <= maxRelativeJump * z else { return nil }
            return point(px, py, d)
        }
        let c = point(x, y, z)
        func tangent(_ a: SIMD3<Float>?, _ b: SIMD3<Float>?) -> SIMD3<Float>? {
            switch (a, b) {
            case let (a?, b?): return b - a
            case let (a?, nil): return c - a
            case let (nil, b?): return b - c
            default: return nil
            }
        }
        guard let tx = tangent(usable(x - r, y), usable(x + r, y)),
              let ty = tangent(usable(x, y - r), usable(x, y + r)) else { return nil }
        var n = simd_cross(tx, ty)
        let len = simd_length(n)
        guard len > 1e-12 else { return nil }
        n /= len
        return simd_dot(n, c) > 0 ? -n : n
    }
}
