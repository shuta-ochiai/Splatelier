// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Mesh extraction from trained Gaussians through their opacity field. The idea is GOF's (Yu et al.,
// "Gaussian Opacity Fields", TOG 2024): a point's opacity is the minimum, over the training views
// that observe it, of the volume-rendered opacity along the view ray up to the point, and the
// surface is the field's 0.5 level set, located by binary search. Implemented from the papers
// (with the exact 0.5-depth idea from SOF, Radl et al., SIGGRAPH Asia 2025).
//
// Each view's ray opacity is non-decreasing in depth, so a view's opacity at a point is ≥ 0.5
// exactly when the point lies behind that view's 0.5-crossing depth. The field's sign therefore
// follows from one depth map per view (`gof_depth`): a point is inside unless an observing view sees
// it in front of its 0.5 depth (`gof_carve`). This replaces GOF's per-point Gaussian evaluation and
// Delaunay tetrahedralization with depth lookups on the sparse voxel grid of `TSDFFusion`; Marching
// Cubes places a vertex on each sign-changing grid edge, and a binary search along the edge (GOF Sec.
// 3.4) moves it onto the level set.

import Foundation
import MLX
import MLXFast

public enum GaussianOpacityField {

    private static let carveKernel = SlangCompiledKernel.required("gof_carve")

    // MARK: - Surface depth

    /// How one Gaussian attenuates a ray, the factor of the order-free ray transmittance whose 0.5
    /// crossing is the surface depth (`gof_depth.slang`). Both end at the rasterizer's 1 - alpha.
    public enum TransmittanceProfile: Int, Sendable {
        /// 1 - α G(min(t, t*)): all of the attenuation before the peak (GOF).
        case opacityField = 0
        /// The stochastic-solid vacancy √(1 - α G(t)) before the peak, v(t*)² / v(t) after it, smooth
        /// in t (Geometry-Grounded Gaussian Splatting, Zhang et al., arXiv 2601.17835, eq. 12).
        case stochasticSolid = 1
    }

    public struct DepthParameters: Sendable {
        /// Bisection steps for the 0.5 crossing along each pixel ray (precision = range / 2^n). Runs
        /// as ⌈n / 3⌉ passes over the Gaussians, each splitting the interval into 8. 12 matches 15
        /// (0.003% of pixels differ by > 0.1%); with 9, 7% of pixels moved by > 0.1% and the TSDF
        /// mesh got rougher.
        public var bisectionIterations: Int
        /// Camera-space depth below which Gaussians are ignored (as the projection's near plane).
        public var nearPlane: Float
        /// Rendering resolution relative to each camera (1 = the training resolution). With 1080×810
        /// training views, 1/2 matched full resolution in the TSDF mesh (voxels ≈ 1 cm, pixels a
        /// few mm) at a third of the time; 1/3 lost 0.2 dB of mesh color.
        public var resolutionScale: Float
        /// Use at most this many views, evenly spaced (0 = all).
        public var maxViews: Int
        public var profile: TransmittanceProfile

        public init(bisectionIterations: Int = 12, nearPlane: Float = 0.01,
                    resolutionScale: Float = 0.5, maxViews: Int = 0,
                    profile: TransmittanceProfile = .opacityField) {
            precondition(bisectionIterations >= 0 && nearPlane > 0 && resolutionScale > 0 && maxViews >= 0)
            self.bisectionIterations = bisectionIterations
            self.nearPlane = nearPlane
            self.resolutionScale = resolutionScale
            self.maxViews = maxViews
            self.profile = profile
        }
    }

    /// For each camera, the camera-space depth at which the opacity field along each pixel ray
    /// reaches 0.5 (<= 0 where it never does), with the rendered color. The result is a regular depth
    /// map, usable by `extractMesh` and by `TSDFFusion`.
    public static func surfaceDepthViews(
        model: GaussianModel, cameras: [Camera], shDegree: Int,
        parameters: DepthParameters = DepthParameters(),
        log: (String) -> Void = { print($0) }
    ) throws -> [CostVolumeMVS.DepthView] {
        let t0 = Date()
        let foreground = withoutEnvironment(model)
        let renderer = try GaussianRenderer()
        renderer.useReplayCache = false
        let selected = evenlySpaced(cameras, maxCount: parameters.maxViews)
        var views: [CostVolumeMVS.DepthView] = []
        views.reserveCapacity(selected.count)
        var validPixels = 0
        var totalPixels = 0
        for camera in selected {
            let cam = rescaled(camera, by: parameters.resolutionScale)
            let view = renderDepth(model: foreground, camera: cam, shDegree: shDegree,
                                   renderer: renderer, parameters: parameters)
            validPixels += view.depth.lazy.filter { $0 > 0 }.count
            totalPixels += view.depth.count
            views.append(view)
            MLX.Memory.clearCache()
        }
        log(String(format: "[GOF] surface depth: %d views at %dx%d, %.1f%% of pixels reach opacity 0.5 (%.2fs)",
                   views.count, views.first?.width ?? 0, views.first?.height ?? 0,
                   100 * Double(validPixels) / Double(Swift.max(1, totalPixels)),
                   Date().timeIntervalSince(t0)))
        return views
    }

    static func renderDepth(
        model: GaussianModel, camera: Camera, shDegree: Int,
        renderer: GaussianRenderer, parameters: DepthParameters
    ) -> CostVolumeMVS.DepthView {
        let (color, _, cache) = renderer.forward(model: model, camera: camera, shDegree: shDegree)
        let pixels = camera.width * camera.height
        let depth = GaussianSurfaceDepth.render(
            model: model, camera: camera, cache: cache,
            parameters: .init(profile: parameters.profile, bisectionIterations: parameters.bisectionIterations,
                              nearPlane: parameters.nearPlane)
        )?.depth.reshaped([pixels]).asArray(Float.self) ?? [Float](repeating: -1, count: pixels)
        let rgb = (MLX.clip(color, min: 0, max: 1) * 255 + 0.5).asType(.uint8)
            .reshaped([pixels * 3]).asArray(UInt8.self)
        let intr = camera.intrinsics.asArray(Float.self)
        return CostVolumeMVS.DepthView(
            width: camera.width, height: camera.height, depth: depth, rgb: rgb,
            viewmat: camera.viewmat.asArray(Float.self),
            intrinsics: (fx: intr[0], fy: intr[1], cx: intr[2], cy: intr[3]))
    }

    /// Keeps only the depth pixels that at least `fusion.vMin` of each view's `fusion.neighborCount`
    /// neighbors see at the same depth (relative error < `relTol`), the consistency test of MVS
    /// fusion; with `averageDepth` a kept pixel takes the mean of the agreeing depths. Neighbors are
    /// chosen as MVS chooses its references, with the baseline relative to the view's median depth.
    /// Drops floaters that only some views see as surfaces, and smooths the depth.
    public static func consistentDepthViews(
        _ views: [CostVolumeMVS.DepthView],
        fusion: CostVolumeMVS.FusionConsistency = CostVolumeMVS.FusionConsistency(),
        log: (String) -> Void = { print($0) }
    ) -> [CostVolumeMVS.DepthView] {
        guard views.count >= 2 else { return views }
        var depths: [Int: [Float]] = [:]
        var dims: [Int: (w: Int, h: Int)] = [:]
        var centers: [SIMD3<Float>] = []
        var selCenter: [SIMD3<Float>] = []
        var selParams: [ReferenceSelection.Parameters] = []
        for (i, v) in views.enumerated() {
            depths[i] = v.depth
            dims[i] = (v.width, v.height)
            let c = FrameSubsampler.cameraCenter(viewmat: v.viewmat)
            centers.append(c)
            var p = ReferenceSelection.Parameters()
            if let z = medianDepth([v]), let r = p.baselineToDepth {
                p.bTarget = r * z
                p.sigmaB = 0.5 * r * z
                selCenter.append(c + z * SIMD3<Float>(v.viewmat[8], v.viewmat[9], v.viewmat[10]))
            } else {
                selCenter.append(c)
            }
            selParams.append(p)
        }
        var filtered = [[Float]](repeating: [], count: views.count)
        let (_, _, kept, total) = CostVolumeMVS.fuseWithConsistency(
            depths: depths, dims: dims, viewmats: views.map(\.viewmat), intrinsics: views.map(\.intrinsics),
            rgb: { _ in MLXArray.zeros([1]) }, cameraCenters: centers,
            selCenter: selCenter, selParams: selParams, fusion: fusion,
            filteredDepths: { filtered[$0] = $1 }, emitPoints: false)
        log(String(format: "[GOF] depth consistency: kept %d/%d pixels (%.1f%%), vMin=%d of %d neighbors, relTol=%.3f",
                   kept, total, 100 * Double(kept) / Double(Swift.max(1, total)),
                   fusion.vMin, fusion.neighborCount, fusion.relTol))
        return views.enumerated().map { i, v in
            CostVolumeMVS.DepthView(
                width: v.width, height: v.height, depth: filtered[i], rgb: v.rgb,
                viewmat: v.viewmat, intrinsics: v.intrinsics)
        }
    }

    // MARK: - Mesh

    public struct MeshParameters: Sendable {
        /// Voxel edge length (m) of the sampling grid.
        public var voxelSize: Float
        /// Half-width (in voxels) of the band allocated around the depth samples.
        public var bandVoxels: Float
        /// Views that must observe a cube's corners before it may emit a surface.
        public var minObservations: Float
        /// Views allowed to see a point in front of their surface before it counts as outside. 0 is
        /// GOF's minimum over views; a few more tolerate pose error and silhouette pixels.
        public var maxCarvingViews: Int
        /// A view carves a point only if it lies in front of the view's surface depth by more than
        /// `depthToleranceVoxels · voxelSize + depthToleranceRelative · depth`.
        public var depthToleranceVoxels: Float
        public var depthToleranceRelative: Float
        /// Pixel depths are interpolated bilinearly at a point's projection unless the 4 neighbors
        /// differ by more than `interpolationGapVoxels · voxelSize + interpolationGapRelative ·
        /// depth` (a depth discontinuity), where the nearest of them is used instead.
        public var interpolationGapVoxels: Float
        public var interpolationGapRelative: Float
        /// Binary-search steps along each sign-changing edge (GOF uses 8).
        public var bisectionIterations: Int
        /// Keep only triangles whose vertices lie on the surface depth (within 2 voxels) of at
        /// least this many views. Sign changes that no view's depth backs, such as the edges of a
        /// view frustum where carving starts, are not surfaces.
        public var minSupportViews: Float
        /// Place the grid only around depth pixels closer than this factor × the median depth
        /// (far background would widen the grid and coarsen its voxels; 0 disables).
        public var allocationDepthFactor: Float
        /// Drop connected components with fewer triangles than this (0 disables).
        public var minComponentTriangles: Int
        /// Allocation caps, as in `TSDFFusion.Parameters` (voxels grow automatically above them).
        public var maxVoxels: Int
        public var maxBlockCells: Int
        public var boundsQuantile: Float
        /// Points per GPU dispatch.
        public var pointsPerDispatch: Int

        public init(voxelSize: Float = 0.005, bandVoxels: Float = 2, minObservations: Float = 3,
                    maxCarvingViews: Int = 0, depthToleranceVoxels: Float = 0,
                    depthToleranceRelative: Float = 0, interpolationGapVoxels: Float = 4,
                    interpolationGapRelative: Float = 0.02, bisectionIterations: Int = 8,
                    minSupportViews: Float = 1, allocationDepthFactor: Float = 2.5,
                    minComponentTriangles: Int = 200, maxVoxels: Int = 32_000_000,
                    maxBlockCells: Int = 16_000_000, boundsQuantile: Float = 0.005,
                    pointsPerDispatch: Int = 2_000_000) {
            precondition(voxelSize > 0 && bandVoxels > 0 && minObservations >= 1 && maxCarvingViews >= 0)
            precondition(depthToleranceVoxels >= 0 && depthToleranceRelative >= 0 && bisectionIterations >= 0)
            precondition(interpolationGapVoxels >= 0 && interpolationGapRelative >= 0)
            precondition(minSupportViews >= 0 && allocationDepthFactor >= 0)
            precondition(minComponentTriangles >= 0 && maxVoxels > 0 && maxBlockCells > 0 && pointsPerDispatch > 0)
            self.voxelSize = voxelSize
            self.bandVoxels = bandVoxels
            self.minObservations = minObservations
            self.maxCarvingViews = maxCarvingViews
            self.depthToleranceVoxels = depthToleranceVoxels
            self.depthToleranceRelative = depthToleranceRelative
            self.interpolationGapVoxels = interpolationGapVoxels
            self.interpolationGapRelative = interpolationGapRelative
            self.bisectionIterations = bisectionIterations
            self.minSupportViews = minSupportViews
            self.allocationDepthFactor = allocationDepthFactor
            self.minComponentTriangles = minComponentTriangles
            self.maxVoxels = maxVoxels
            self.maxBlockCells = maxBlockCells
            self.boundsQuantile = boundsQuantile
            self.pointsPerDispatch = pointsPerDispatch
        }
    }

    /// Extracts the 0.5 level set of the opacity field given by `surfaceDepthViews`. nil if no
    /// surface remains.
    public static func extractMesh(
        depthViews views: [CostVolumeMVS.DepthView],
        parameters: MeshParameters = MeshParameters(),
        log: (String) -> Void = { print($0) }
    ) -> TriangleMesh? {
        let t0 = Date()
        let maxDepth = parameters.allocationDepthFactor > 0
            ? parameters.allocationDepthFactor * (medianDepth(views) ?? .infinity) : .infinity
        let alloc = TSDFFusion.Parameters(
            voxelSize: parameters.voxelSize, truncationVoxels: parameters.bandVoxels,
            maxVoxels: parameters.maxVoxels, maxBlockCells: parameters.maxBlockCells,
            boundsQuantile: parameters.boundsQuantile, allocationMaxDepth: maxDepth)
        guard let allocation = TSDFFusion.allocate(views: views, parameters: alloc, log: log),
              !allocation.blockCoords.isEmpty else { return nil }
        let voxelSize = allocation.voxelSize
        let blockCount = allocation.blockCoords.count
        let voxelCount = blockCount * TSDFFusion.Volume.voxelsPerBlock
        log(String(format: "[GOF] voxel %.4f m, %d blocks (%d voxels), grid placed from depths <= %.2f m",
                   voxelSize, blockCount, voxelCount, maxDepth))

        let field = FieldEvaluator(
            frames: TSDFFusion.FrameBuffers(views: views), voxelSize: voxelSize, allocation: allocation,
            parameters: parameters)

        // Field sign at the voxel centers.
        var tsdf: [Float] = []
        var weight: [Float] = []
        var color: [UInt32] = []
        tsdf.reserveCapacity(voxelCount)
        weight.reserveCapacity(voxelCount)
        color.reserveCapacity(voxelCount)
        var inside = 0
        field.evaluateVoxels(count: voxelCount) { r in
            for i in 0..<r.carve.count {
                let isIn = field.isInside(carve: r.carve[i], observed: r.observed[i])
                if isIn { inside += 1 }
                tsdf.append(isIn ? -1 : 1)
            }
            weight += r.observed
            color += r.color
        }
        let tCarve = Date()
        log(String(format: "[GOF] carved %d voxels against %d views: %d inside (%.2fs)",
                   voxelCount, views.count, inside, tCarve.timeIntervalSince(t0)))

        let volume = TSDFFusion.Volume(
            voxelSize: voxelSize, truncation: parameters.bandVoxels * voxelSize,
            origin: allocation.origin, blockDims: allocation.dims, blockIndex: allocation.blockIndex,
            blockCoords: allocation.blockCoords, tsdf: tsdf, weight: weight, color: color)
        var (mesh, edges) = MarchingCubes.extract(
            volume: volume, minWeight: parameters.minObservations, recordEdges: true)
        guard mesh.vertexCount > 0 else {
            log("[GOF] no sign change on the grid")
            return nil
        }

        // Binary search along each vertex's edge for the 0.5 level set.
        let vertexCount = mesh.vertexCount
        for _ in 0..<parameters.bisectionIterations {
            let mids = edges.map { 0.5 * ($0.inside + $0.outside) }
            field.evaluatePoints(mids) { offset, r in
                for i in 0..<r.carve.count {
                    let v = offset + i
                    if field.isInside(carve: r.carve[i], observed: r.observed[i]) {
                        edges[v].inside = mids[v]
                    } else {
                        edges[v].outside = mids[v]
                    }
                }
            }
        }
        for v in 0..<vertexCount { mesh.positions[v] = 0.5 * (edges[v].inside + edges[v].outside) }

        // Color and depth support at the final positions.
        var supported = [Bool](repeating: false, count: vertexCount)
        field.evaluatePoints(mesh.positions) { offset, r in
            for i in 0..<r.carve.count {
                let v = offset + i
                supported[v] = r.support[i] >= parameters.minSupportViews
                let c = r.color[i]
                if c != 0 {
                    mesh.colors[v] = SIMD3(Float(c & 0xFF), Float((c >> 8) & 0xFF), Float((c >> 16) & 0xFF)) / 255
                }
            }
        }
        let raw = mesh.triangleCount
        mesh = keepingTriangles(of: mesh) { a, b, c in supported[a] && supported[b] && supported[c] }
        let unsupported = raw - mesh.triangleCount
        mesh.normals = TriangleMesh.vertexNormals(positions: mesh.positions, indices: mesh.indices)
        let beforeComponents = mesh.triangleCount
        mesh = mesh.removingSmallComponents(minTriangles: parameters.minComponentTriangles)
        MLX.Memory.clearCache()
        log(String(format: "[GOF] %d vertices, %d triangles (marching cubes %d, dropped %d without depth support "
                   + "and %d in components < %d tris); carve %.2fs, extract+search %.2fs",
                   mesh.vertexCount, mesh.triangleCount, raw, unsupported,
                   beforeComponents - mesh.triangleCount, parameters.minComponentTriangles,
                   tCarve.timeIntervalSince(t0), Date().timeIntervalSince(tCarve)))
        return mesh.triangleCount > 0 ? mesh : nil
    }

    /// Runs `gof_carve` over voxel centers or explicit points.
    struct FieldEvaluator {
        struct Result {
            var carve: [Float]
            var observed: [Float]
            var color: [UInt32]
            var support: [Float]
        }

        let frames: TSDFFusion.FrameBuffers
        let carveParams: MLXArray
        let coords: MLXArray
        let gridParams: MLXArray
        let maxCarve: Float
        let pointsPerDispatch: Int
        // Placeholders for the inputs of the other point mode (zero-size buffers cannot be bound).
        let unusedPoints = MLXArray([Float(0), 0, 0], [3])
        let unusedCoords = MLXArray([Int32(0), 0, 0], [3])

        init(frames: TSDFFusion.FrameBuffers, voxelSize: Float, allocation: TSDFFusion.Allocation,
             parameters: MeshParameters) {
            self.frames = frames
            self.carveParams = MLXArray(
                [parameters.depthToleranceVoxels * voxelSize, parameters.depthToleranceRelative, 2 * voxelSize,
                 parameters.interpolationGapVoxels * voxelSize, parameters.interpolationGapRelative], [5])
            self.coords = allocation.blockCoordsArray()
            self.gridParams = MLXArray(
                [voxelSize, allocation.origin.x, allocation.origin.y, allocation.origin.z], [4])
            self.maxCarve = Float(parameters.maxCarvingViews)
            self.pointsPerDispatch = parameters.pointsPerDispatch
        }

        func isInside(carve: Float, observed: Float) -> Bool { observed > 0 && carve <= maxCarve }

        /// Voxel centers 0..<count of the allocation's blocks, in dispatch-sized chunks.
        func evaluateVoxels(count: Int, _ body: (Result) -> Void) {
            var start = 0
            while start < count {
                let n = Swift.min(pointsPerDispatch, count - start)
                body(run(mode: 0, start: start, count: n, coords: coords, points: unusedPoints))
                start += n
            }
        }

        /// `points` in chunks; `body` gets each chunk's offset into `points`.
        func evaluatePoints(_ points: [SIMD3<Float>], _ body: (Int, Result) -> Void) {
            guard !points.isEmpty else { return }
            var flat = [Float]()
            flat.reserveCapacity(points.count * 3)
            for p in points { flat += [p.x, p.y, p.z] }
            let arr = MLXArray(flat, [flat.count])
            var start = 0
            while start < points.count {
                let n = Swift.min(pointsPerDispatch, points.count - start)
                body(start, run(mode: 1, start: start, count: n, coords: unusedCoords, points: arr))
                start += n
            }
        }

        private func run(mode: Int32, start: Int, count: Int, coords: MLXArray, points: MLXArray) -> Result {
            let out = carveKernel(
                [frames.depth, frames.rgb, frames.viewmats, frames.intrinsics, frames.meta, carveParams,
                 coords, gridParams, points,
                 MLXArray([mode, Int32(start), Int32(count), Int32(frames.count)], [4])],
                grid: (count, 1, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [[count], [count], [count], [count]],
                outputDTypes: [.float32, .float32, .uint32, .float32]
            )
            eval(out[0], out[1], out[2], out[3])
            return Result(carve: out[0].asArray(Float.self), observed: out[1].asArray(Float.self),
                          color: out[2].asArray(UInt32.self), support: out[3].asArray(Float.self))
        }
    }

    /// Median of the valid depths, estimated on at most ~1M pixels.
    public static func medianDepth(_ views: [CostVolumeMVS.DepthView]) -> Float? {
        let total = views.reduce(0) { $0 + $1.depth.count }
        let stride = Swift.max(1, total / 1_000_000)
        var samples: [Float] = []
        var k = 0
        for v in views {
            for d in v.depth {
                if k % stride == 0 && d > 0 { samples.append(d) }
                k += 1
            }
        }
        guard !samples.isEmpty else { return nil }
        samples.sort()
        return samples[samples.count / 2]
    }

    /// The triangles of `mesh` for which `keep(a, b, c)` holds, with unused vertices removed.
    static func keepingTriangles(of mesh: TriangleMesh, _ keep: (Int, Int, Int) -> Bool) -> TriangleMesh {
        var remap = [Int32](repeating: -1, count: mesh.vertexCount)
        var out = TriangleMesh(positions: [], normals: [], colors: [], indices: [])
        for t in stride(from: 0, to: mesh.indices.count, by: 3) {
            let (a, b, c) = (Int(mesh.indices[t]), Int(mesh.indices[t + 1]), Int(mesh.indices[t + 2]))
            guard keep(a, b, c) else { continue }
            for v in [a, b, c] {
                if remap[v] < 0 {
                    remap[v] = Int32(out.positions.count)
                    out.positions.append(mesh.positions[v])
                    out.normals.append(mesh.normals[v])
                    out.colors.append(mesh.colors[v])
                }
                out.indices.append(UInt32(remap[v]))
            }
        }
        return out
    }

    // MARK: - Helpers

    /// The model without the baked environment sphere (far background shell), which is not part of
    /// the scene's surfaces.
    static func withoutEnvironment(_ model: GaussianModel) -> GaussianModel {
        guard model.environmentCount > 0 else { return model }
        let n = model.count - model.environmentCount
        return GaussianModel(
            means: model.means[0..<n], quats: model.quats[0..<n], logScales: model.logScales[0..<n],
            logitOpacities: model.logitOpacities[0..<n], sh0: model.sh0[0..<n], shN: model.shN[0..<n])
    }

    /// `maxCount` items spaced evenly through `items` (all of them if 0 or more than available).
    static func evenlySpaced<T>(_ items: [T], maxCount: Int) -> [T] {
        guard maxCount > 0, items.count > maxCount else { return items }
        return (0..<maxCount).map { items[$0 * items.count / maxCount] }
    }

    /// `camera` rendered at `scale` × its resolution (intrinsics scaled per axis to the rounded size).
    /// `camera` with its image (and intrinsics) scaled by `scale`.
    public static func rescaled(_ camera: Camera, by scale: Float) -> Camera {
        guard scale != 1 else { return camera }
        let w = Swift.max(1, Int((Float(camera.width) * scale).rounded()))
        let h = Swift.max(1, Int((Float(camera.height) * scale).rounded()))
        let sx = Float(w) / Float(camera.width), sy = Float(h) / Float(camera.height)
        let intr = camera.intrinsics.asArray(Float.self)
        return Camera(
            viewmat: camera.viewmat,
            intrinsics: MLXArray([intr[0] * sx, intr[1] * sy, intr[2] * sx, intr[3] * sy], [4]),
            width: w, height: h, tileSize: camera.tileSize, background: camera.background)
    }
}
