// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import simd

/// Primitive for the coverage display during capture. Not a training Gaussian.
public struct CaptureCoverageSplat: Sendable, Equatable {
    public let position: SIMD3<Float>
    public let radiusMeters: Float
    /// Edge length (m) of the display voxel. Grows when the display is coarsened
    /// (`CaptureCoverageAccumulator.displayVoxelSizeMeters`).
    public let voxelSizeMeters: Float
    public let opacity: Float
    public let observationCount: Int
    /// Mean color (0..1 RGB) of the points in the voxel. Non-nil only when colored points (MVS)
    /// were inserted.
    public let color: SIMD3<Float>?
    /// Number of distinct keyframes that contributed points to this voxel (counted only when
    /// inserted with a frame index, otherwise 0). Low = not enough capture viewpoints there.
    public let viewCount: Int

    public init(
        position: SIMD3<Float>,
        radiusMeters: Float,
        voxelSizeMeters: Float,
        opacity: Float,
        observationCount: Int,
        color: SIMD3<Float>? = nil,
        viewCount: Int = 0
    ) {
        self.position = position
        self.radiusMeters = radiusMeters
        self.voxelSizeMeters = voxelSizeMeters
        self.opacity = opacity
        self.observationCount = observationCount
        self.color = color
        self.viewCount = viewCount
    }
}

/// Aggregates live feature points into voxels to form stable coverage splats.
/// The more observations in a voxel, the larger its opacity and radius.
public struct CaptureCoverageAccumulator: Sendable {
    private struct VoxelKey: Hashable {
        let x: Int
        let y: Int
        let z: Int
    }

    private struct Voxel: Sendable {
        var sum: SIMD3<Float>
        var count: Int
        var lastTimestamp: TimeInterval
        var colorSum = SIMD3<Float>.zero
        var colorCount = 0
        var lastFrame = -1
        var frameCount = 0
        var viewKey: VoxelKey? = nil

        var center: SIMD3<Float> {
            sum / Float(max(1, count))
        }
    }

    public let voxelSizeMeters: Float
    public let maxSplats: Int
    /// Cell edge of the grid used to count `viewCount`. nil means per display voxel. Coarser than
    /// the display voxel, a single-point voxel floating slightly off a surface gets the same view
    /// count as the surrounding surface, so "not enough viewpoints" shows up as regions rather than
    /// point scatter.
    public let viewVoxelSizeMeters: Float?
    /// When the display is coarsened, cells with fewer points than this are hidden. Surfaces merge
    /// into coarse cells and gain points, but isolated points floating off the surface do not merge
    /// and remain as 1-2 point cells, scattering over the surface as under-viewed points
    /// (2026-10-09 Jonathan: 21% of the coarse display had 1 view, 93% of those with ≤ 3 points ->
    /// 4% with ≥ 3 points).
    public let minPointsWhenCoarse: Int
    private var voxels: [VoxelKey: Voxel] = [:]
    private var viewCells: [VoxelKey: (lastFrame: Int, count: Int)] = [:]
    /// When display voxels exceed `maxSplats`, output `coarse` with the display edge alone scaled
    /// by `displayFactor` (2, 3, 4, 6, 8, 12, ...). Keeping only the most-observed voxels makes
    /// captured areas vanish in chunks, so it looks like "captured but no points appear"
    /// (2026-10-09 Jonathan: keeping only the top 80k of 460k-740k 2cm voxels made the wall look
    /// split into 3).
    private var displayFactor = 1
    private var coarse: [VoxelKey: Voxel] = [:]

    public init(voxelSizeMeters: Float = 0.04, maxSplats: Int = 12_000,
                viewVoxelSizeMeters: Float? = nil, minPointsWhenCoarse: Int = 1) {
        precondition(voxelSizeMeters > 0, "voxelSizeMeters must be positive")
        precondition(maxSplats > 0, "maxSplats must be positive")
        precondition(viewVoxelSizeMeters.map { $0 > 0 } ?? true)
        precondition(minPointsWhenCoarse >= 1)
        self.voxelSizeMeters = voxelSizeMeters
        self.maxSplats = maxSplats
        self.viewVoxelSizeMeters = viewVoxelSizeMeters
        self.minPointsWhenCoarse = minPointsWhenCoarse
    }

    public var voxelCount: Int { voxels.count }

    public mutating func reset() {
        voxels.removeAll(keepingCapacity: true)
        viewCells.removeAll(keepingCapacity: true)
        coarse.removeAll(keepingCapacity: true)
        displayFactor = 1
    }

    /// - Parameters:
    ///   - colors: color of each point (0..1 RGB, same count as `points`). Appears in the snapshot
    ///     as the voxel's mean color.
    ///   - frameIndex: keyframe index that produced the points. The number of distinct indices per
    ///     voxel is counted as `viewCount` (insert points of the same frame consecutively).
    public mutating func insert(
        _ points: [SIMD3<Float>], timestamp: TimeInterval,
        colors: [SIMD3<Float>]? = nil, frameIndex: Int? = nil
    ) {
        precondition(colors == nil || colors!.count == points.count)
        for (i, p) in points.enumerated() {
            insert(p, timestamp: timestamp, color: colors?[i], frameIndex: frameIndex)
        }
    }

    public mutating func insert(
        _ point: SIMD3<Float>, timestamp: TimeInterval,
        color: SIMD3<Float>? = nil, frameIndex: Int? = nil
    ) {
        guard point.x.isFinite, point.y.isFinite, point.z.isFinite else { return }
        let key = Self.key(point, voxelSizeMeters)
        var viewKey: VoxelKey? = nil
        if let frameIndex, let vs = viewVoxelSizeMeters {
            let vk = Self.key(point, vs)
            viewKey = vk
            var cell = viewCells[vk] ?? (lastFrame: -1, count: 0)
            if cell.lastFrame != frameIndex {
                cell = (lastFrame: frameIndex, count: cell.count + 1)
                viewCells[vk] = cell
            }
        }
        let views = viewCells
        let viewCount = { (k: VoxelKey) in views[k]?.count ?? 0 }
        Self.add(point, timestamp: timestamp, color: color, frameIndex: frameIndex, viewKey: viewKey,
                 viewCount: viewCount, to: &voxels, at: key)
        if displayFactor > 1 {
            Self.add(point, timestamp: timestamp, color: color, frameIndex: frameIndex, viewKey: viewKey,
                     viewCount: viewCount, to: &coarse, at: Self.coarsen(key, displayFactor))
        }
        while (displayFactor == 1 ? voxels.count : coarse.count) > maxSplats {
            coarsenDisplay()
        }
    }

    private static func key(_ p: SIMD3<Float>, _ size: Float) -> VoxelKey {
        VoxelKey(x: Int(floor(p.x / size)), y: Int(floor(p.y / size)), z: Int(floor(p.z / size)))
    }

    /// Maps a display voxel key to the key of the `factor`-times grid (floor for negative
    /// coordinates too).
    private static func coarsen(_ k: VoxelKey, _ factor: Int) -> VoxelKey {
        func div(_ a: Int) -> Int { a >= 0 ? a / factor : -((-a + factor - 1) / factor) }
        return VoxelKey(x: div(k.x), y: div(k.y), z: div(k.z))
    }

    /// `viewKey` snaps to the view-grid cell with the most views within the voxel (so that even if
    /// a 1-view point floating off the surface enters a coarse display voxel first, the whole
    /// surface does not look under-viewed).
    private static func add(
        _ point: SIMD3<Float>, timestamp: TimeInterval, color: SIMD3<Float>?, frameIndex: Int?,
        viewKey: VoxelKey?, viewCount: (VoxelKey) -> Int,
        to map: inout [VoxelKey: Voxel], at key: VoxelKey
    ) {
        var voxel = map[key] ?? Voxel(sum: .zero, count: 0, lastTimestamp: timestamp)
        voxel.sum += point
        voxel.count += 1
        voxel.lastTimestamp = timestamp
        if let color {
            voxel.colorSum += color
            voxel.colorCount += 1
        }
        if let viewKey {
            if voxel.viewKey.map({ viewCount(viewKey) > viewCount($0) }) ?? true { voxel.viewKey = viewKey }
        } else if let frameIndex, frameIndex != voxel.lastFrame {
            voxel.lastFrame = frameIndex
            voxel.frameCount += 1
        }
        map[key] = voxel
    }

    /// Coarsens the display grid by 1.5x (2 → 3 → 4 → 6 → ..., 2x from 1) and rebuilds `coarse`
    /// from the display voxels. With 2x steps the surface splat count drops by 1/4 each time and
    /// the display gets too coarse. Distinct frame counts cannot be summed across voxels, so they
    /// are approximated by the max (with `viewVoxelSizeMeters`, the view-grid cell with the most
    /// views is used).
    private mutating func coarsenDisplay() {
        let f = displayFactor
        displayFactor = f == 1 ? 2 : (f & (f - 1) == 0 ? f * 3 / 2 : f * 4 / 3)
        let views = viewCells
        let viewCount = { (k: VoxelKey) in views[k]?.count ?? 0 }
        coarse.removeAll(keepingCapacity: true)
        for (k, v) in voxels {
            let ck = Self.coarsen(k, displayFactor)
            guard var c = coarse[ck] else { coarse[ck] = v; continue }
            c.sum += v.sum
            c.count += v.count
            c.lastTimestamp = max(c.lastTimestamp, v.lastTimestamp)
            c.colorSum += v.colorSum
            c.colorCount += v.colorCount
            c.lastFrame = max(c.lastFrame, v.lastFrame)
            c.frameCount = max(c.frameCount, v.frameCount)
            if let vk = v.viewKey, c.viewKey.map({ viewCount(vk) > viewCount($0) }) ?? true { c.viewKey = vk }
            coarse[ck] = c
        }
    }

    /// Display edge length (m). Coarsens when display voxels exceed `maxSplats` (`coarsenDisplay`).
    public var displayVoxelSizeMeters: Float { voxelSizeMeters * Float(displayFactor) }

    /// Splats for the whole captured area (at most `maxSplats`; any excess is merged by coarsening
    /// the display grid).
    public func snapshot() -> [CaptureCoverageSplat] {
        let size = displayVoxelSizeMeters
        let cells = displayFactor == 1
            ? Array(voxels.values) : coarse.values.filter { $0.count >= minPointsWhenCoarse }
        return cells.map { voxel in
            let confidence = min(1, log1p(Float(voxel.count)) / log1p(Float(8)))
            return CaptureCoverageSplat(
                position: voxel.center,
                radiusMeters: size * (1.0 + 1.8 * confidence),
                voxelSizeMeters: size,
                opacity: 0.18 + 0.62 * confidence,
                observationCount: voxel.count,
                color: voxel.colorCount > 0 ? voxel.colorSum / Float(voxel.colorCount) : nil,
                viewCount: voxel.viewKey.flatMap { viewCells[$0]?.count } ?? voxel.frameCount
            )
        }
    }
}
