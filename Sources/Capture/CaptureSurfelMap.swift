// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Display surfel map during capture. For each point, the grid resolution (level) is chosen by "how
// large the pixel that saw the point is on the surface", and cells at finer levels become smaller
// splats. Areas captured only from afar have points only at coarse levels and look large and
// blurry; capturing up close puts points into finer levels and sharpens them, hiding the coarse
// splats that covered the area. No optimization (only the mean of the points in a cell), so the
// look changes only when keyframes are added.
//
// Sizing splats by pixel size follows the same idea as surfel fusion (Keller et al. 2013 "Real-time
// 3D Reconstruction in Dynamic Scenes using Point-based Fusion").

import Foundation
import simd

public struct CaptureSurfelMap: Sendable {
    public struct Surfel: Sendable, Equatable {
        public let position: SIMD3<Float>
        /// Unit normal (mean of the normals of the points in the cell).
        public let normal: SIMD3<Float>
        /// 0..1 RGB (mean color of the points in the cell).
        public let color: SIMD3<Float>
        /// Cell edge length (m). Splat size is proportional to it.
        public let cellMeters: Float
        public let level: Int
        public let pointCount: Int
        /// Number of distinct keyframes that contributed points to the cell.
        public let viewCount: Int

        /// In-plane standard deviation (m) of the splat = cell edge ×
        /// `CaptureSurfelMap.splatSigmaPerCell`.
        public var splatSigma: Float { cellMeters * CaptureSurfelMap.splatSigmaPerCell }
        /// Splat opacity. 0.55 if one keyframe contributed points, 0.95 with 4 or more.
        public var splatOpacity: Float { 0.55 + 0.4 * min(1, Float(max(0, viewCount - 1)) / 3) }
    }

    /// Ratio of the splat's in-plane standard deviation to the cell edge. Large enough to overlap
    /// the neighboring cells' splats without gaps.
    public static let splatSigmaPerCell: Float = 0.5

    private struct Key: Hashable {
        let x: Int32
        let y: Int32
        let z: Int32
    }

    private struct Cell {
        var position = SIMD3<Float>.zero
        var color = SIMD3<Float>.zero
        var normal = SIMD3<Float>.zero
        var count = 0
        var lastFrame = -1
        var frames = 0
        /// Bits for the child cells (8) at the next finer level that hold points (themselves or in
        /// descendants).
        var childMask: UInt8 = 0

        /// Discard the points (descendant marks remain. The parent's mark also remains, but that
        /// only errs toward hiding the parent too much).
        mutating func removeData() {
            position = .zero
            color = .zero
            normal = .zero
            count = 0
            lastFrame = -1
            frames = 0
        }
    }

    /// Cell edge (m) at the finest level. A level-L cell is `finestCellMeters × 2^L`.
    public let finestCellMeters: Float
    public let levelCount: Int
    /// Edge of the cell a point goes into = pixel size on the surface × this (with 1, about one
    /// pixel per cell when facing the surface).
    public let cellsPerPixel: Float
    /// Upper bound on pixel stretch from the angle between surface and view ray (divide by this
    /// even if cos is below it).
    public let minCosine: Float
    /// If at least this many child cells are refined, the cell is not displayed (left to the finer
    /// splats). 1: a thin structure seen up close (a bar refines one or two of the 8 children) is
    /// otherwise covered by the coarse splat that averages it with what lies behind it
    /// (2026-10-14 梁: vertical rebars did not show however closely they were captured).
    public let hideWhenRefinedChildren: Int
    /// Cells with fewer points than this are not displayed (isolated points floating off the
    /// surface).
    public let minPoints: Int

    private var levels: [[Key: Cell]]

    public init(finestCellMeters: Float = 0.005, levelCount: Int = 7, cellsPerPixel: Float = 1,
                minCosine: Float = 0.3, hideWhenRefinedChildren: Int = 1, minPoints: Int = 2) {
        precondition(finestCellMeters > 0 && levelCount >= 1 && cellsPerPixel > 0)
        self.finestCellMeters = finestCellMeters
        self.levelCount = levelCount
        self.cellsPerPixel = cellsPerPixel
        self.minCosine = minCosine
        self.hideWhenRefinedChildren = hideWhenRefinedChildren
        self.minPoints = minPoints
        self.levels = Array(repeating: [:], count: levelCount)
    }

    /// Cell count per level (including parent cells that hold no points).
    public var cellCounts: [Int] { levels.map(\.count) }

    public func cellMeters(level: Int) -> Float { finestCellMeters * Float(1 << level) }

    /// Inserts one point.
    /// - Parameters:
    ///   - normal: unit normal (either orientation, but points on the same surface must face the
    ///     same side).
    ///   - pixelSize: size of one pixel that saw the point (depth / fx, m).
    ///   - cameraCenter: position of the camera that saw the point (accounts for pixel stretch from
    ///     surface tilt).
    ///   - frame: index of the keyframe that produced the point (insert points of the same frame
    ///     consecutively).
    public mutating func insert(
        position p: SIMD3<Float>, normal: SIMD3<Float>, color: SIMD3<Float>,
        pixelSize: Float, cameraCenter: SIMD3<Float>, frame: Int
    ) {
        guard p.x.isFinite, p.y.isFinite, p.z.isFinite, pixelSize > 0 else { return }
        let view = simd_normalize(cameraCenter - p)
        let cosine = max(minCosine, abs(simd_dot(view, normal)))
        let footprint = cellsPerPixel * pixelSize / cosine
        let level = min(levelCount - 1, max(0, Int(ceil(log2(footprint / finestCellMeters)))))
        var key = Self.key(p, cellMeters(level: level))
        var cell = levels[level][key] ?? Cell()
        let wasEmpty = cell.count == 0
        cell.position += p
        cell.color += color
        cell.normal += normal
        cell.count += 1
        if cell.lastFrame != frame {
            cell.lastFrame = frame
            cell.frames += 1
        }
        levels[level][key] = cell
        guard wasEmpty else { return }
        // Mark the parent cells "this child holds points" (if already marked, everything above is
        // marked too).
        var l = level
        while l + 1 < levelCount {
            let bit = UInt8(1) << UInt8((key.x & 1) | (key.y & 1) << 1 | (key.z & 1) << 2)
            key = Key(x: key.x >> 1, y: key.y >> 1, z: key.z >> 1)
            l += 1
            var parent = levels[l][key] ?? Cell()
            if parent.childMask & bit != 0 { break }
            parent.childMask |= bit
            levels[l][key] = parent
        }
    }

    /// Removes cells coarser than the observation that float in front of the depth map surface
    /// (splats where there should be empty space. Points captured from afar have large depth error
    /// and float off the surface, blurring over the front of surfaces captured up close). Cells
    /// finer than the observation are kept, since the observation is the coarser one.
    /// - Parameters:
    ///   - depth: `width × height` depth (camera z, m). Pixels without points are ≤ 0.
    ///   - intrinsics: (fx, fy, cx, cy) at the resolution of `depth`.
    ///   - viewmat: world → camera (row-major 4×4).
    ///   - relativeMargin: remove if in front by more than this fraction of the surface depth + the
    ///     cell edge.
    /// - Returns: number of cells removed.
    @discardableResult
    public mutating func carve(
        depth: [Float], width w: Int, height h: Int,
        intrinsics: (fx: Float, fy: Float, cx: Float, cy: Float), viewmat vm: [Float],
        relativeMargin: Float = 0.05
    ) -> Int {
        var removed = 0
        for level in 1..<levelCount {
            let size = cellMeters(level: level)
            var doomed: [Key] = []
            for (key, cell) in levels[level] where cell.count > 0 {
                let p = cell.position / Float(cell.count)
                let z = vm[8] * p.x + vm[9] * p.y + vm[10] * p.z + vm[11]
                guard z > 1e-3 else { continue }
                let u = intrinsics.fx * (vm[0] * p.x + vm[1] * p.y + vm[2] * p.z + vm[3]) / z + intrinsics.cx
                let v = intrinsics.fy * (vm[4] * p.x + vm[5] * p.y + vm[6] * p.z + vm[7]) / z + intrinsics.cy
                guard u >= 0, v >= 0, u < Float(w), v < Float(h) else { continue }
                let d = depth[Int(v) * w + Int(u)]
                // The observation's cell (pixel × cellsPerPixel) is finer than this cell, and the
                // surface lies sufficiently behind the cell
                guard d > 0, cellsPerPixel * d / intrinsics.fx < size,
                      z < d - relativeMargin * d - size else { continue }
                doomed.append(key)
            }
            for key in doomed { levels[level][key]?.removeData() }
            removed += doomed.count
        }
        return removed
    }

    /// Surfels to display (excluding cells with at least `hideWhenRefinedChildren` refined children
    /// and cells with fewer than `minPoints` points).
    public func snapshot() -> [Surfel] {
        var out: [Surfel] = []
        for (level, cells) in levels.enumerated() {
            let size = cellMeters(level: level)
            for cell in cells.values where cell.count >= minPoints
                && cell.childMask.nonzeroBitCount < hideWhenRefinedChildren {
                let n = simd_length(cell.normal) > 1e-6 ? simd_normalize(cell.normal) : SIMD3<Float>(0, 0, 1)
                let inv = 1 / Float(cell.count)
                out.append(Surfel(
                    position: cell.position * inv, normal: n, color: cell.color * inv,
                    cellMeters: size, level: level, pointCount: cell.count, viewCount: cell.frames))
            }
        }
        return out
    }

    private static func key(_ p: SIMD3<Float>, _ size: Float) -> Key {
        Key(x: Int32(floor(p.x / size)), y: Int32(floor(p.y / size)), z: Int32(floor(p.z / size)))
    }
}
