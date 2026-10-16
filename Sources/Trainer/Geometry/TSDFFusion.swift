// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// TSDF fusion of MVS depth maps (Curless & Levoy 1996). The mesh is extracted with
// `MarchingCubes.extract`. Voxels are allocated only in 8³ blocks near the surface: a two-level
// layout where a coarse dense grid (in blocks) holds pool indices, giving constant-time lookup from
// both GPU and CPU without hashing. Allocation runs on the CPU (blocks within ± truncation distance
// of depth pixels); fusion runs on the GPU (`tsdf_integrate`, 1 thread = 1 voxel scanning all
// frames).

import Foundation
import MLX
import MLXFast

public enum TSDFFusion {

    private static let integrateKernel = SlangCompiledKernel.required("tsdf_integrate")

    public struct Parameters: Sendable {
        /// Voxel edge length (m).
        public var voxelSize: Float
        /// Truncation distance (in voxels). Signed distance is stored only within this range in
        /// front of and behind the surface.
        public var truncationVoxels: Float
        /// Cap on allocated voxels. If exceeded, voxelSize is enlarged and allocation redone
        /// (memory ≈ 12 B/voxel × 2).
        public var maxVoxels: Int
        /// Cap on coarse-grid (block) cells. voxelSize is also enlarged when the extent is too
        /// wide.
        public var maxBlockCells: Int
        /// Fraction trimmed from both ends of each axis when estimating the extent (keeps far
        /// outlier depths from bloating the grid).
        public var boundsQuantile: Float
        /// Voxels per dispatch (avoids long-running kernels).
        public var voxelsPerDispatch: Int
        /// Depth pixels farther than this (camera z, m) are not used to place the grid (far
        /// background would otherwise widen it and coarsen the voxels); they still take part in
        /// fusion where the grid reaches them.
        public var allocationMaxDepth: Float
        /// `integrateHybrid` only: voxels the fill may add beyond the primary's, as a fraction of
        /// `maxVoxels`. The voxel size is set by the primary alone, so the fill never coarsens it;
        /// when the fill's blocks exceed this budget, those hit by the most fill pixels are kept.
        public var fillVoxelFraction: Float

        public init(voxelSize: Float = 0.005, truncationVoxels: Float = 4,
                    maxVoxels: Int = 32_000_000, maxBlockCells: Int = 16_000_000,
                    boundsQuantile: Float = 0.005, voxelsPerDispatch: Int = 2_000_000,
                    allocationMaxDepth: Float = .infinity, fillVoxelFraction: Float = 0.25) {
            precondition(voxelSize > 0 && truncationVoxels > 0 && maxVoxels > 0 && maxBlockCells > 0)
            precondition(boundsQuantile >= 0 && boundsQuantile < 0.5 && voxelsPerDispatch > 0)
            precondition(allocationMaxDepth > 0 && fillVoxelFraction >= 0)
            self.voxelSize = voxelSize
            self.truncationVoxels = truncationVoxels
            self.maxVoxels = maxVoxels
            self.maxBlockCells = maxBlockCells
            self.boundsQuantile = boundsQuantile
            self.voxelsPerDispatch = voxelsPerDispatch
            self.allocationMaxDepth = allocationMaxDepth
            self.fillVoxelFraction = fillVoxelFraction
        }
    }

    /// Fusion result. The center of voxel (i, j, k) is `origin + (i + 0.5, j + 0.5, k + 0.5) ·
    /// voxelSize`.
    public struct Volume: Sendable {
        public static let blockSide = 8
        public static let voxelsPerBlock = 512

        public let voxelSize: Float
        public let truncation: Float
        /// World coordinates of the corner of voxel (0, 0, 0).
        public let origin: SIMD3<Float>
        /// Coarse-grid block counts (x, y, z).
        public let blockDims: SIMD3<Int32>
        /// [bx*by*bz] (x fastest) → pool index; -1 if unallocated.
        public let blockIndex: [Int32]
        /// Pool index → block coordinates.
        public let blockCoords: [SIMD3<Int32>]
        /// [pool*512] normalized TSDF (-1..1). Positive = free space on the camera side, negative =
        /// behind the surface.
        public let tsdf: [Float]
        /// [pool*512] observation count (0 = unobserved).
        public let weight: [Float]
        /// [pool*512] r | g<<8 | b<<16.
        public let color: [UInt32]

        public var voxelCount: Int { tsdf.count }

        /// Storage location of a voxel coordinate. nil if unallocated.
        @inline(__always)
        public func slot(_ x: Int32, _ y: Int32, _ z: Int32) -> Int? {
            if x < 0 || y < 0 || z < 0 { return nil }
            let bx = x >> 3, by = y >> 3, bz = z >> 3
            if bx >= blockDims.x || by >= blockDims.y || bz >= blockDims.z { return nil }
            let cell = (Int(bz) * Int(blockDims.y) + Int(by)) * Int(blockDims.x) + Int(bx)
            let b = blockIndex[cell]
            if b < 0 { return nil }
            return Int(b) * 512 + Int(((z & 7) << 6) | ((y & 7) << 3) | (x & 7))
        }
    }

    /// Fuses a set of depth maps. nil if there are no valid pixels.
    public static func integrate(
        views: [CostVolumeMVS.DepthView],
        parameters: Parameters = Parameters(),
        log: (String) -> Void = { print($0) }
    ) -> Volume? {
        guard let allocation = allocate(views: views, parameters: parameters, log: log) else { return nil }
        let voxelSize = allocation.voxelSize
        let trunc = parameters.truncationVoxels * voxelSize
        let blockCount = allocation.blockCoords.count
        let voxelCount = blockCount * Volume.voxelsPerBlock
        log(String(format: "[TSDF] voxel %.4f m trunc %.4f m, grid %dx%dx%d blocks, %d blocks allocated (%d voxels, %.0f MB)",
                   voxelSize, trunc, allocation.dims.x, allocation.dims.y, allocation.dims.z,
                   blockCount, voxelCount, Double(voxelCount * 12) / (1024 * 1024)))
        guard blockCount > 0 else { return nil }

        let (tsdf, weight, color) = fuse(
            views: views, allocation: allocation, truncation: trunc, parameters: parameters)
        let observed = weight.lazy.filter { $0 > 0 }.count
        log(String(format: "[TSDF] integrated %d views, observed voxels %d/%d (%.1f%%)",
                   views.count, observed, voxelCount, 100 * Double(observed) / Double(voxelCount)))
        return Volume(
            voxelSize: voxelSize, truncation: trunc, origin: allocation.origin,
            blockDims: allocation.dims, blockIndex: allocation.blockIndex,
            blockCoords: allocation.blockCoords, tsdf: tsdf, weight: weight, color: color)
    }

    /// Fuses `primary` depth maps, filling only where they leave the grid unobserved with `fill`
    /// depth maps. Both are fused separately on one grid; per voxel the primary TSDF is trusted in
    /// proportion to its observation count (fully from `primaryTrustObservations` on), and the fill
    /// TSDF takes the rest. Primary observations include the free space in front of their surfaces,
    /// so the primary also overrides fill surfaces lying where it saw empty space. The grid is placed
    /// around the primary depths at the primary's own voxel size, plus the fill depths that fall
    /// outside it (the holes) within `Parameters.fillVoxelFraction`.
    public static func integrateHybrid(
        primary: [CostVolumeMVS.DepthView],
        fill: [CostVolumeMVS.DepthView],
        primaryTrustObservations: Float = 3,
        parameters: Parameters = Parameters(),
        log: (String) -> Void = { print($0) }
    ) -> Volume? {
        precondition(primaryTrustObservations > 0)
        guard let allocation = allocate(views: primary, fillViews: fill, parameters: parameters, log: log) else {
            return nil
        }
        let voxelSize = allocation.voxelSize
        let trunc = parameters.truncationVoxels * voxelSize
        let voxelCount = allocation.blockCoords.count * Volume.voxelsPerBlock
        log(String(format: "[TSDF hybrid] voxel %.4f m trunc %.4f m, %d blocks allocated (%d voxels)",
                   voxelSize, trunc, allocation.blockCoords.count, voxelCount))
        guard voxelCount > 0 else { return nil }
        // The result is written over the primary's arrays (no third copy of the volume).
        var (tsdf, weight, color) = fuse(views: primary, allocation: allocation, truncation: trunc, parameters: parameters)
        let f = fuse(views: fill, allocation: allocation, truncation: trunc, parameters: parameters)

        func unpack(_ c: UInt32) -> SIMD3<Float> {
            SIMD3(Float(c & 0xFF), Float((c >> 8) & 0xFF), Float((c >> 16) & 0xFF))
        }
        var primaryOnly = 0, filled = 0, blended = 0
        for i in 0..<voxelCount {
            let wp = Swift.min(1, weight[i] / primaryTrustObservations)
            if wp >= 1 || f.weight[i] <= 0 {
                if weight[i] > 0 { primaryOnly += 1 }
                continue
            }
            if wp <= 0 {
                tsdf[i] = f.tsdf[i]; weight[i] = f.weight[i]; color[i] = f.color[i]
                filled += 1
                continue
            }
            blended += 1
            tsdf[i] = wp * tsdf[i] + (1 - wp) * f.tsdf[i]
            weight[i] += f.weight[i]
            let (cp, cf) = (color[i], f.color[i])
            if cp == 0 || cf == 0 {
                color[i] = cp == 0 ? cf : cp
            } else {
                let c = wp * unpack(cp) + (1 - wp) * unpack(cf)
                color[i] = UInt32(c.x.rounded()) | (UInt32(c.y.rounded()) << 8) | (UInt32(c.z.rounded()) << 16)
            }
        }
        log(String(format: "[TSDF hybrid] %d primary (%d views), %d filled (%d views), %d blended voxels",
                   primaryOnly, primary.count, filled, fill.count, blended))
        return Volume(
            voxelSize: voxelSize, truncation: trunc, origin: allocation.origin,
            blockDims: allocation.dims, blockIndex: allocation.blockIndex,
            blockCoords: allocation.blockCoords, tsdf: tsdf, weight: weight, color: color)
    }

    /// Runs `tsdf_integrate` over every voxel of `allocation`.
    static func fuse(
        views: [CostVolumeMVS.DepthView], allocation: Allocation, truncation trunc: Float,
        parameters: Parameters
    ) -> (tsdf: [Float], weight: [Float], color: [UInt32]) {
        let voxelCount = allocation.blockCoords.count * Volume.voxelsPerBlock
        let frames = FrameBuffers(views: views)
        let coordsArr = allocation.blockCoordsArray()
        let paramsF = MLXArray(
            [allocation.voxelSize, trunc, allocation.origin.x, allocation.origin.y, allocation.origin.z], [5])
        var tsdf: [Float] = []
        var weight: [Float] = []
        var color: [UInt32] = []
        tsdf.reserveCapacity(voxelCount)
        weight.reserveCapacity(voxelCount)
        color.reserveCapacity(voxelCount)
        var start = 0
        while start < voxelCount {
            let count = Swift.min(parameters.voxelsPerDispatch, voxelCount - start)
            let paramsI = MLXArray([Int32(start), Int32(count), Int32(views.count)], [3])
            let out = integrateKernel(
                [coordsArr, frames.depth, frames.rgb, frames.viewmats, frames.intrinsics, frames.meta,
                 paramsF, paramsI],
                grid: (count, 1, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [[count], [count], [count]],
                outputDTypes: [.float32, .float32, .uint32]
            )
            eval(out[0], out[1], out[2])
            tsdf += out[0].asArray(Float.self)
            weight += out[1].asArray(Float.self)
            color += out[2].asArray(UInt32.self)
            start += count
        }
        MLX.Memory.clearCache()
        return (tsdf, weight, color)
    }

    // MARK: - Allocation

    struct Allocation {
        let voxelSize: Float
        let origin: SIMD3<Float>
        let dims: SIMD3<Int32>
        let blockIndex: [Int32]
        let blockCoords: [SIMD3<Int32>]

        /// Block coordinates flattened for the GPU kernels ([nBlocks*3]).
        func blockCoordsArray() -> MLXArray {
            var flat: [Int32] = []
            flat.reserveCapacity(blockCoords.count * 3)
            for c in blockCoords { flat += [c.x, c.y, c.z] }
            return MLXArray(flat, [flat.count])
        }
    }

    /// Allocates the 8³ blocks within ±`truncationVoxels` of the depth pixels. If the grid or the
    /// voxel count exceeds its cap, voxels are enlarged and allocation redone (by the cube root for
    /// the grid, a volume; by a fitted power for the voxels, a band around the surface). nil if there
    /// are no valid depth pixels.
    /// `fillViews` then add blocks at that voxel size only around their pixels whose own block `views`
    /// left unallocated (the holes), up to `fillVoxelFraction · maxVoxels` voxels.
    static func allocate(
        views: [CostVolumeMVS.DepthView], fillViews: [CostVolumeMVS.DepthView] = [],
        parameters: Parameters, log: (String) -> Void
    ) -> Allocation? {
        let maxDepth = parameters.allocationMaxDepth
        guard let (lo, hi) = robustBounds(
            views: views + fillViews, quantile: parameters.boundsQuantile, maxDepth: maxDepth) else {
            log("[TSDF] no valid depth pixels")
            return nil
        }
        var voxelSize = parameters.voxelSize
        // The voxel size is searched on every `step`-th pixel along both axes (~2M points: depth
        // pixels are far denser than 8³ blocks, so they mark nearly the same blocks), and the final
        // allocation redone at that size with every pixel.
        let pixels = views.reduce(0) { $0 + $1.width * $1.height }
        var step = Swift.max(1, Int((Double(pixels) / 2_000_000).squareRoot()))
        // Allocated voxels fall as a power of the voxel size (≈ s⁻² for a surface band, more where
        // sparse points each hold their own blocks); the exponent is refitted from the last two
        // passes at the same sampling.
        var exponent = 2.0
        var last: (size: Float, voxels: Int)?
        while true {
            let trunc = parameters.truncationVoxels * voxelSize
            let margin = trunc + 2 * voxelSize
            let origin = lo - SIMD3<Float>(repeating: margin)
            let blockWorld = voxelSize * Float(Volume.blockSide)
            let ext = (hi + SIMD3<Float>(repeating: margin)) - origin
            // Blocks per axis (capped at 2^20 to fit Int32; the cell-count cap is checked in
            // Double).
            let n = [ext.x, ext.y, ext.z].map { Swift.min(Swift.max(1, ($0 / blockWorld).rounded(.up)), Float(1 << 20)) }
            let cells = Double(n[0]) * Double(n[1]) * Double(n[2])
            if cells > Double(parameters.maxBlockCells) {
                let grow = Float(pow(cells / Double(parameters.maxBlockCells), 1.0 / 3.0)) * 1.01
                log(String(format: "[TSDF] grid %.0fx%.0fx%.0f blocks exceeds cap; voxel %.4f → %.4f m",
                           n[0], n[1], n[2], voxelSize, voxelSize * grow))
                voxelSize *= grow
                continue
            }
            let grid = BlockGrid(
                origin: origin, voxelSize: voxelSize, truncation: trunc,
                dims: SIMD3<Int32>(Int32(n[0]), Int32(n[1]), Int32(n[2])))
            var mark = [Bool](repeating: false, count: grid.cellCount)
            forEachWorldPoint(views: views, maxDepth: maxDepth, step: step) { p in
                grid.forEachCell(around: p) { mark[$0] = true }
            }
            let voxels = mark.lazy.filter { $0 }.count * Volume.voxelsPerBlock
            if voxels > parameters.maxVoxels {
                if let last, last.voxels > voxels, voxelSize > last.size {
                    exponent = Swift.min(3, Swift.max(1, Foundation.log(Double(last.voxels) / Double(voxels))
                                                        / Foundation.log(Double(voxelSize / last.size))))
                }
                last = (voxelSize, voxels)
                let grow = Float(pow(Double(voxels) / Double(parameters.maxVoxels), 1 / exponent)) * 1.01
                log(String(format: "[TSDF] %d voxels exceeds cap %d (every %d px); voxel %.4f → %.4f m",
                           voxels, parameters.maxVoxels, step, voxelSize, voxelSize * grow))
                voxelSize *= grow
                continue
            }
            if step > 1 {
                step = 1
                last = nil
                continue
            }
            if !fillViews.isEmpty {
                let budget = Int(parameters.fillVoxelFraction * Float(parameters.maxVoxels)) / Volume.voxelsPerBlock
                let (added, candidates) = addFillBlocks(
                    fillViews, into: &mark, grid: grid, maxDepth: maxDepth, budget: budget)
                log(String(format: "[TSDF] fill adds %d of %d blocks outside the primary's (%d voxels)",
                           added, candidates, added * Volume.voxelsPerBlock))
            }
            return grid.allocation(mark: mark)
        }
    }

    /// Coarse grid of 8³ blocks and the blocks within ±truncation of a world point.
    struct BlockGrid {
        let origin: SIMD3<Float>
        let voxelSize: Float
        let truncation: Float
        let dims: SIMD3<Int32>

        var cellCount: Int { Int(dims.x) * Int(dims.y) * Int(dims.z) }
        private var invBlock: Float { 1 / (voxelSize * Float(Volume.blockSide)) }

        /// Cell of the block containing `p`; nil outside the grid.
        func cell(of p: SIMD3<Float>) -> Int? {
            let b = ((p - origin) * invBlock).rounded(.down)
            guard b.x >= 0, b.y >= 0, b.z >= 0,
                  b.x < Float(dims.x), b.y < Float(dims.y), b.z < Float(dims.z) else { return nil }
            return (Int(b.z) * Int(dims.y) + Int(b.y)) * Int(dims.x) + Int(b.x)
        }

        /// Calls `body` with each cell touched by a box of ±truncation around `p` (none if the box
        /// misses the grid on some axis: an outlier depth).
        func forEachCell(around p: SIMD3<Float>, _ body: (Int) -> Void) {
            let inv = invBlock
            func range(_ v: Float, _ o: Float, _ n: Int32) -> ClosedRange<Int>? {
                let lo = Int(((v - o - truncation) * inv).rounded(.down))
                let hi = Int(((v - o + truncation) * inv).rounded(.down))
                if hi < 0 || lo >= Int(n) { return nil }
                return Swift.max(lo, 0)...Swift.min(hi, Int(n) - 1)
            }
            guard let rx = range(p.x, origin.x, dims.x), let ry = range(p.y, origin.y, dims.y),
                  let rz = range(p.z, origin.z, dims.z) else { return }
            let (nx, ny) = (Int(dims.x), Int(dims.y))
            for z in rz {
                for y in ry {
                    let row = (z * ny + y) * nx
                    for x in rx { body(row + x) }
                }
            }
        }

        /// Pool indices of the marked cells (x fastest).
        func allocation(mark: [Bool]) -> Allocation {
            var blockIndex = [Int32](repeating: -1, count: cellCount)
            var coords: [SIMD3<Int32>] = []
            var cell = 0
            for z in 0..<dims.z {
                for y in 0..<dims.y {
                    for x in 0..<dims.x {
                        if mark[cell] {
                            blockIndex[cell] = Int32(coords.count)
                            coords.append(SIMD3<Int32>(x, y, z))
                        }
                        cell += 1
                    }
                }
            }
            return Allocation(
                voxelSize: voxelSize, origin: origin, dims: dims, blockIndex: blockIndex, blockCoords: coords)
        }
    }

    /// Marks the blocks around the `fillViews` pixels whose own block `mark` leaves unallocated, at
    /// most `budget` of them: those touched by the most fill pixels (near, densely seen surfaces)
    /// first. Returns the blocks added and the candidates.
    static func addFillBlocks(
        _ fillViews: [CostVolumeMVS.DepthView], into mark: inout [Bool], grid: BlockGrid,
        maxDepth: Float, budget: Int
    ) -> (added: Int, candidates: Int) {
        var hits = [UInt32](repeating: 0, count: mark.count)
        let primary = mark
        forEachWorldPoint(views: fillViews, maxDepth: maxDepth) { p in
            if let c = grid.cell(of: p), primary[c] { return }
            grid.forEachCell(around: p) { if !primary[$0] { hits[$0] &+= 1 } }
        }
        var candidates = hits.indices.filter { hits[$0] > 0 }
        let total = candidates.count
        if total > budget {
            candidates.sort { hits[$0] > hits[$1] }
            candidates.removeLast(total - budget)
        }
        for c in candidates { mark[c] = true }
        return (candidates.count, total)
    }

    /// Depth / color / camera of all frames concatenated for the GPU kernels (layout shared by
    /// `tsdf_integrate` and `gof_carve_*`).
    struct FrameBuffers {
        let depth: MLXArray        // [Σ H*W]
        let rgb: MLXArray          // [Σ H*W] r | g<<8 | b<<16
        let viewmats: MLXArray     // [F*16]
        let intrinsics: MLXArray   // [F*4]
        let meta: MLXArray         // [F*3] (W, H, pixel offset)
        let count: Int

        init(views: [CostVolumeMVS.DepthView]) {
            var depthFlat: [Float] = []
            var rgbFlat: [UInt32] = []
            var viewmatFlat: [Float] = []
            var intrFlat: [Float] = []
            var meta: [Int32] = []
            let pixelTotal = views.reduce(0) { $0 + $1.width * $1.height }
            precondition(pixelTotal < Int(Int32.max), "depth maps too large for int32 offsets")
            depthFlat.reserveCapacity(pixelTotal)
            rgbFlat.reserveCapacity(pixelTotal)
            for v in views {
                meta += [Int32(v.width), Int32(v.height), Int32(depthFlat.count)]
                depthFlat += v.depth
                let n = v.width * v.height
                v.rgb.withUnsafeBufferPointer { p in
                    for i in 0..<n {
                        rgbFlat.append(UInt32(p[i * 3]) | (UInt32(p[i * 3 + 1]) << 8) | (UInt32(p[i * 3 + 2]) << 16))
                    }
                }
                viewmatFlat += v.viewmat
                intrFlat += [v.intrinsics.fx, v.intrinsics.fy, v.intrinsics.cx, v.intrinsics.cy]
            }
            self.depth = MLXArray(depthFlat, [depthFlat.count])
            self.rgb = MLXArray(rgbFlat, [rgbFlat.count])
            self.viewmats = MLXArray(viewmatFlat, [viewmatFlat.count])
            self.intrinsics = MLXArray(intrFlat, [intrFlat.count])
            self.meta = MLXArray(meta, [meta.count])
            self.count = views.count
        }
    }

    /// Per-axis quantile range of the world coordinates of valid depth pixels. Estimated on at most
    /// ~1M points (every `step`-th pixel along both axes).
    static func robustBounds(
        views: [CostVolumeMVS.DepthView], quantile: Float, maxDepth: Float = .infinity
    ) -> (lo: SIMD3<Float>, hi: SIMD3<Float>)? {
        let valid = views.reduce(0) { acc, v in acc + v.depth.lazy.filter { $0 > 0 && $0 <= maxDepth }.count }
        guard valid > 0 else { return nil }
        let step = Swift.max(1, Int((Double(valid) / 1_000_000).squareRoot().rounded(.up)))
        var xs: [Float] = [], ys: [Float] = [], zs: [Float] = []
        let cap = valid / (step * step) + views.count * 64
        xs.reserveCapacity(cap); ys.reserveCapacity(cap); zs.reserveCapacity(cap)
        forEachWorldPoint(views: views, maxDepth: maxDepth, step: step) { p in
            xs.append(p.x); ys.append(p.y); zs.append(p.z)
        }
        func range(_ a: inout [Float]) -> (Float, Float) {
            a.sort()
            let i = Int(Float(a.count - 1) * quantile)
            return (a[i], a[a.count - 1 - i])
        }
        let (x0, x1) = range(&xs), (y0, y1) = range(&ys), (z0, z1) = range(&zs)
        return (SIMD3(x0, y0, z0), SIMD3(x1, y1, z1))
    }

    /// Enumerates valid depth pixels (0 < z <= maxDepth) of all views in world coordinates (pixel
    /// center x + 0.5, P_w = Rᵀ (P_c - t)), every `step`-th pixel along both axes.
    static func forEachWorldPoint(
        views: [CostVolumeMVS.DepthView], maxDepth: Float = .infinity, step: Int = 1,
        _ body: (SIMD3<Float>) -> Void
    ) {
        for v in views {
            let vm = v.viewmat
            let intr = v.intrinsics
            v.depth.withUnsafeBufferPointer { d in
                for y in Swift.stride(from: step / 2, to: v.height, by: step) {
                    let ny = (Float(y) + 0.5 - intr.cy) / intr.fy
                    for x in Swift.stride(from: step / 2, to: v.width, by: step) {
                        let z = d[y * v.width + x]
                        if z <= 0 || z > maxDepth { continue }
                        let dx = (Float(x) + 0.5 - intr.cx) / intr.fx * z - vm[3]
                        let dy = ny * z - vm[7]
                        let dz = z - vm[11]
                        body(SIMD3(vm[0] * dx + vm[4] * dy + vm[8] * dz,
                                   vm[1] * dx + vm[5] * dy + vm[9] * dz,
                                   vm[2] * dx + vm[6] * dy + vm[10] * dz))
                    }
                }
            }
        }
    }
}
