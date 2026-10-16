// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXFast

/// Tile binning result for the rasterizer. Entries are ordered by (tile_id, depth) for every tile
/// each Gaussian's AABB overlaps. On the GPU: tile_aabb → cumsum → tile_emit_pairs → argSort →
/// tile_lower_bound. Only n_isects is read back to the host.
public struct TileBinningResult {
    public let flattenIds: MLXArray      // [n_isects] int32, front to back within a tile
    public let tileOffsets: MLXArray     // [numTiles + 1] int32, tile t is flattenIds[off[t]..<off[t+1]]
    public let numTilesX: Int
    public let numTilesY: Int
    public var numTiles: Int { numTilesX * numTilesY }
    public var nIsects: Int { flattenIds.shape[0] }
}

public enum TileBinning {

    private static let aabbKernel = SlangCompiledKernel.required("tile_aabb")

    private static let emitKernel = SlangCompiledKernel.required("tile_emit_pairs")

    private static let lowerBoundKernel = SlangCompiledKernel.required("tile_lower_bound")

    /// Same values as the projection near/far cull (nearPlane / farPlane in slang/gs_math.slang).
    /// Depths of Gaussians reaching binning fall within this range, so it is used as the key
    /// quantization range.
    static let nearPlane: Float = 0.01
    static let farPlane: Float = 1e10
    private static let nearPlaneBits = Int32(bitPattern: nearPlane.bitPattern)
    private static let farPlaneBits = Int32(bitPattern: farPlane.bitPattern)

    /// Bit allocation of the sort key (tile_id << depthBits) | quantized depth. tile_id gets
    /// ceil(log2(numTiles)) bits and depth the rest, 31 bits in total (a non-negative int32).
    /// Quantized depth = (asint(depth) - asint(near)) >> depthShift. Bit patterns of positive
    /// floats sort like their values, so the relative precision is about 2^(depthShift - 23)
    /// regardless of depth units (about 1.2e-4 for a 4:3 1024 px width with 16 px tiles).
    static func sortKeyLayout(numTiles: Int) -> (depthBits: Int, depthShift: Int) {
        precondition(numTiles >= 1, "numTiles must be positive")
        let tileBits = numTiles == 1 ? 0 : Int.bitWidth - (numTiles - 1).leadingZeroBitCount
        let depthBits = 31 - tileBits
        precondition(depthBits >= 8, "too many tiles (\(numTiles)) for a 31-bit sort key")
        let range = Int(farPlaneBits) - Int(nearPlaneBits)
        let rangeBits = Int.bitWidth - range.leadingZeroBitCount
        return (depthBits, max(0, rangeBits - depthBits))
    }

    /// Assigns projected Gaussians to tiles. Excludes rx <= 0 / ry <= 0 / depth <= 0.
    /// - Parameters:
    ///   - means2d: [N, 2] centers in pixel coordinates.
    ///   - radii: [N, 2] int32 per-axis pixel radii (rx, ry).
    ///   - depths: [N] camera-space z.
    ///   - conics: [N, 3] (a, b, c) of Σ_2d⁻¹. When passed together with opacities, only tiles the
    ///     ellipse actually covers
    ///     within the AABB are counted (scanline culling). nil means every tile in the AABB.
    ///   - opacities: [N] sigmoid-activated opacities (ellipse size = the same opacity-dependent
    ///     radius as radii).
    public static func bin(
        means2d: MLXArray,
        radii: MLXArray,
        depths: MLXArray,
        conics: MLXArray? = nil,
        opacities: MLXArray? = nil,
        imageWidth: Int,
        imageHeight: Int,
        tileSize: Int = 16
    ) -> TileBinningResult {
        precondition(means2d.shape.count == 2 && means2d.shape[1] == 2, "means2d must be [N, 2]")
        precondition(radii.shape.count == 2 && radii.shape[1] == 2, "radii must be [N, 2] (rx, ry)")
        precondition(depths.shape.count == 1, "depths must be [N]")
        precondition(tileSize > 0, "tileSize must be positive")

        let n = means2d.shape[0]
        precondition(radii.shape[0] == n && depths.shape[0] == n,
                     "means2d/radii/depths must agree on N")

        let numTilesX = (imageWidth + tileSize - 1) / tileSize
        let numTilesY = (imageHeight + tileSize - 1) / tileSize
        let numTiles = numTilesX * numTilesY

        if n == 0 {
            return TileBinningResult(
                flattenIds: MLXArray.zeros([0], dtype: .int32),
                tileOffsets: MLXArray.zeros([numTiles + 1], dtype: .int32),
                numTilesX: numTilesX, numTilesY: numTilesY
            )
        }

        // 1. AABB and intersected tile count. In scanline mode only tiles covered by the ellipse
        //    are counted.
        let scanline = conics != nil && opacities != nil
        if scanline {
            precondition(conics!.size == n * 3 && opacities!.size == n,
                         "conics must be [N, 3] and opacities [N]")
        }
        let aabbDims = MLXArray([
            Int32(numTilesX), Int32(numTilesY), Int32(tileSize), Int32(n), scanline ? 1 : 0,
        ], [5])
        let means2dFlat = means2d.reshaped([n * 2])
        let radiiFlat = radii.reshaped([n * 2])
        // In AABB mode the kernel does not read these, so pass dummies of the right shape.
        let conicsFlat = scanline ? conics!.reshaped([n * 3]) : MLXArray.zeros([1], dtype: .float32)
        let opacitiesFlat = scanline ? opacities!.reshaped([n]) : MLXArray.zeros([1], dtype: .float32)
        let aabbOut = aabbKernel(
            [means2dFlat, radiiFlat, depths, conicsFlat, opacitiesFlat, aabbDims],
            grid: (n, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[n * 4], [n]],
            outputDTypes: [.int32, .int32]
        )
        let aabb = aabbOut[0]
        let countArr = aabbOut[1]

        // 2. Exclusive prefix sum (starts[0] = 0).
        let inclusive = MLX.cumsum(countArr, axis: 0)
        let starts: MLXArray
        if n == 1 {
            starts = MLXArray.zeros([1], dtype: .int32)
        } else {
            let zero1 = MLXArray.zeros([1], dtype: .int32)
            starts = MLX.concatenated([zero1, inclusive[0..<(n - 1)]], axis: 0)
        }
        // Read only n_isects back to the host to fix the output size (the only sync on this path).
        let nIsects: Int
        eval(inclusive)
        nIsects = Int(inclusive[n - 1].item(Int32.self))

        if nIsects == 0 {
            return TileBinningResult(
                flattenIds: MLXArray.zeros([0], dtype: .int32),
                tileOffsets: MLXArray.zeros([numTiles + 1], dtype: .int32),
                numTilesX: numTilesX, numTilesY: numTilesY
            )
        }

        // 3. Emit (tile_id, sort_key, gauss_idx) per intersection.
        let layout = sortKeyLayout(numTiles: numTiles)
        let emitDims = MLXArray([
            Int32(numTilesX), Int32(n), Int32(layout.depthBits), Int32(layout.depthShift),
            nearPlaneBits, farPlaneBits,
        ], [6])
        let emitOut = emitKernel(
            [aabb, countArr, starts, depths, emitDims,
             means2dFlat, radiiFlat, conicsFlat, opacitiesFlat, aabbDims],
            grid: (n, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[nIsects], [nIsects], [nIsects]],
            outputDTypes: [.int32, .int32, .int32]
        )
        let tileIds = emitOut[0]
        let sortKeys = emitOut[1]
        let gaussIndices = emitOut[2]

        // 4. Sort on a single 31-bit non-negative int32 key (half the bandwidth of a 64-bit key,
        //    and MLX's sort blocks
        //    become twice as large).
        let sortedIdx = MLX.argSort(sortKeys)
        let flattenIds = gaussIndices[sortedIdx]

        // 5. tileOffsets via lower_bound on the sorted tile_ids (the last entry is n_isects).
        let sortedTileIds = tileIds[sortedIdx]
        let lbDims = MLXArray([Int32(nIsects), Int32(numTiles)], [2])
        let lbOut = lowerBoundKernel(
            [sortedTileIds, lbDims],
            grid: (numTiles + 1, 1, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [[numTiles + 1]],
            outputDTypes: [.int32]
        )
        let tileOffsets = lbOut[0]
        eval(flattenIds, tileOffsets)

        return TileBinningResult(
            flattenIds: flattenIds,
            tileOffsets: tileOffsets,
            numTilesX: numTilesX,
            numTilesY: numTilesY
        )
    }
}
