// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import Testing
@testable import Trainer

@Test
func tileBinningEmptyResultWhenAllBehindCamera() {
    // Both have depth < 0, so nothing is output.
    let means2d = MLXArray([10.0, 10.0, 20.0, 20.0] as [Float], [2, 2])
    let radii = MLXArray([Int32(5), Int32(5), Int32(5), Int32(5)], [2, 2])
    let depths = MLXArray([Float(-1.0), Float(-2.0)], [2])

    let result = TileBinning.bin(
        means2d: means2d, radii: radii, depths: depths,
        imageWidth: 64, imageHeight: 64, tileSize: 16
    )

    #expect(result.numTilesX == 4)
    #expect(result.numTilesY == 4)
    #expect(result.nIsects == 0)
    #expect(result.tileOffsets.shape[0] == 17)
    let offs = result.tileOffsets.asArray(Int32.self)
    #expect(offs.allSatisfy { $0 == 0 })
}

@Test
func tileBinningSingleGaussianCovers9Tiles() {
    // (40, 40) radius 20 → bbox x, y ∈ [20, 60] → tiles (floor(/16)) ∈ [1, 3]², 9 tiles (4x4 grid).
    let means2d = MLXArray([40.0, 40.0] as [Float], [1, 2])
    let radii = MLXArray([Int32(20), Int32(20)], [1, 2])
    let depths = MLXArray([Float(5.0)], [1])

    let result = TileBinning.bin(
        means2d: means2d, radii: radii, depths: depths,
        imageWidth: 64, imageHeight: 64, tileSize: 16
    )
    #expect(result.nIsects == 9, "expected 9 tile intersections, got \(result.nIsects)")

    let offs = result.tileOffsets.asArray(Int32.self)
    // Exactly 1 entry in each of those 9 tiles, 0 elsewhere.
    let expectedTouched: Set<Int32> = [
        1 * 4 + 1, 1 * 4 + 2, 1 * 4 + 3,
        2 * 4 + 1, 2 * 4 + 2, 2 * 4 + 3,
        3 * 4 + 1, 3 * 4 + 2, 3 * 4 + 3,
    ]
    for t in 0..<result.numTiles {
        let count = offs[t + 1] - offs[t]
        if expectedTouched.contains(Int32(t)) {
            #expect(count == 1, "tile \(t) should have 1 entry, got \(count)")
        } else {
            #expect(count == 0, "tile \(t) should have 0 entries, got \(count)")
        }
    }
}

@Test
func tileBinningClipsAtImageBounds() {
    // The off-image part of the corner Gaussian is clipped, leaving only the 1 corner tile.
    let means2d = MLXArray([2.0, 2.0] as [Float], [1, 2])
    let radii = MLXArray([Int32(8), Int32(8)], [1, 2])
    let depths = MLXArray([Float(5.0)], [1])

    let result = TileBinning.bin(
        means2d: means2d, radii: radii, depths: depths,
        imageWidth: 64, imageHeight: 64, tileSize: 16
    )
    // bbox x ∈ [-6, 10] → clipped to [0, 10] → only tile x = 0.
    #expect(result.nIsects == 1)
}

@Test
func tileBinningDropsBehindCamera() {
    let means2d = MLXArray([40.0, 40.0, 30.0, 30.0] as [Float], [2, 2])
    let radii = MLXArray([Int32(5), Int32(5), Int32(5), Int32(5)], [2, 2])
    let depths = MLXArray([Float(-1.0), Float(2.0)], [2])  // the first is behind the camera

    let result = TileBinning.bin(
        means2d: means2d, radii: radii, depths: depths,
        imageWidth: 64, imageHeight: 64, tileSize: 16
    )
    let flatten = result.flattenIds.asArray(Int32.self)
    #expect(flatten.allSatisfy { $0 == 1 })
}

@Test
func tileBinningSortsByDepthWithinTile() {
    // The 3 in the same tile are in ascending depth order (front first).
    let means2d = MLXArray([8.0, 8.0, 8.0, 8.0, 8.0, 8.0] as [Float], [3, 2])
    let radii = MLXArray(
        [Int32(3), Int32(3), Int32(3), Int32(3), Int32(3), Int32(3)], [3, 2]
    )
    let depths = MLXArray([Float(5.0), Float(1.0), Float(3.0)], [3])

    let result = TileBinning.bin(
        means2d: means2d, radii: radii, depths: depths,
        imageWidth: 16, imageHeight: 16, tileSize: 16
    )
    #expect(result.numTiles == 1)
    #expect(result.nIsects == 3)

    let flatten = result.flattenIds.asArray(Int32.self)
    #expect(flatten == [1, 2, 0])  // depth 1, 3, 5 -> gaussian 1, 2, 0
}

@Test
func tileBinningOffsetsAreCSR() {
    // tileOffsets is a CSR row pointer (starts at 0, ends at nIsects, monotonically increasing).
    let means2d = MLXArray(
        [20.0, 20.0,
         40.0, 40.0,
         50.0, 10.0] as [Float],
        [3, 2]
    )
    let radii = MLXArray(
        [Int32(8), Int32(8), Int32(10), Int32(10), Int32(6), Int32(6)], [3, 2]
    )
    let depths = MLXArray([Float(2.0), Float(5.0), Float(1.5)], [3])

    let result = TileBinning.bin(
        means2d: means2d, radii: radii, depths: depths,
        imageWidth: 64, imageHeight: 64, tileSize: 16
    )
    let offs = result.tileOffsets.asArray(Int32.self)
    #expect(offs.first == 0)
    #expect(Int(offs.last ?? -1) == result.nIsects)
    for t in 0..<result.numTiles {
        #expect(offs[t] <= offs[t + 1], "tile \(t) offsets not monotonic")
    }
}

@Test
func tileBinningSortKeyLayoutFitsInNonNegativeInt32() {
    // (tile_id << depthBits) | quantized depth fits in 31 bits (non-negative int32), and depth does
    // not spill into the tile bits.
    let nearBits = Int(Int32(bitPattern: TileBinning.nearPlane.bitPattern))
    let farBits = Int(Int32(bitPattern: TileBinning.farPlane.bitPattern))
    let maxDepthCode = { (shift: Int) in (farBits - nearBits) >> shift }
    for numTiles in [1, 2, 3, 768, 3072, 4096, 4097, 65_536] {
        let (depthBits, depthShift) = TileBinning.sortKeyLayout(numTiles: numTiles)
        #expect(maxDepthCode(depthShift) < (1 << depthBits),
                "numTiles \(numTiles): depth code spills into tile bits")
        let maxKey = ((numTiles - 1) << depthBits) | maxDepthCode(depthShift)
        #expect(maxKey <= Int(Int32.max), "numTiles \(numTiles): key exceeds 31 bits")
    }
    // 3072 tiles from 1024x768 divided into 16-px tiles: 12 bits of tile + 19 bits of depth.
    #expect(TileBinning.sortKeyLayout(numTiles: 3072).depthBits == 19)
}

@Test
func tileBinningOrdersByDepthAcrossManyTiles() {
    // Scatter random Gaussians over 1024x768 (3072 tiles) and check that each tile's list
    // (1) equals the set of overlapping Gaussians under the same rule as tile_aabb, and
    // (2) is ordered front to back (only pairs that fall into the same quantization step may swap).
    let width = 1024, height = 768, tileSize = 16
    let n = 3000
    var lcg = TestLCG(seed: 2026)
    var means: [Float] = []
    var radiiList: [Int32] = []
    var depthList: [Float] = []
    for _ in 0..<n {
        means.append(lcg.next() * Float(width))
        means.append(lcg.next() * Float(height))
        radiiList.append(Int32(1 + lcg.next() * 40))
        radiiList.append(Int32(1 + lcg.next() * 40))
        depthList.append(0.05 * powf(4000, lcg.next()))  // log-uniform over 0.05-200
    }

    let result = TileBinning.bin(
        means2d: MLXArray(means, [n, 2]), radii: MLXArray(radiiList, [n, 2]),
        depths: MLXArray(depthList, [n]),
        imageWidth: width, imageHeight: height, tileSize: tileSize
    )

    let numTilesX = result.numTilesX
    let numTilesY = result.numTilesY
    let ts = Float(tileSize)
    var expected = Array(repeating: [Int](), count: result.numTiles)
    for i in 0..<n {
        let px = means[2 * i], py = means[2 * i + 1]
        let rx = Float(radiiList[2 * i]), ry = Float(radiiList[2 * i + 1])
        let x0 = max(0, Int(floor((px - rx) / ts)))
        let x1 = min(numTilesX - 1, Int(floor((px + rx) / ts)))
        let y0 = max(0, Int(floor((py - ry) / ts)))
        let y1 = min(numTilesY - 1, Int(floor((py + ry) / ts)))
        if x1 < x0 || y1 < y0 { continue }
        for ty in y0...y1 { for tx in x0...x1 { expected[ty * numTilesX + tx].append(i) } }
    }

    #expect(result.nIsects == expected.reduce(0) { $0 + $1.count })
    let offs = result.tileOffsets.asArray(Int32.self).map(Int.init)
    let flatten = result.flattenIds.asArray(Int32.self).map(Int.init)
    let quantum = 1 << TileBinning.sortKeyLayout(numTiles: result.numTiles).depthShift
    for t in 0..<result.numTiles {
        let ids = Array(flatten[offs[t]..<offs[t + 1]])
        #expect(ids.count == expected[t].count && Set(ids) == Set(expected[t]),
                "tile \(t): membership differs from the AABB rule")
        for (a, b) in zip(ids, ids.dropFirst()) where depthList[a] > depthList[b] {
            let gap = Int(depthList[a].bitPattern) - Int(depthList[b].bitPattern)
            #expect(gap < quantum, "tile \(t): depth \(depthList[a]) is placed before \(depthList[b])")
        }
    }
}

@Test
func tileBinningKeepsOrderOfCloseDepths() {
    // Depths with a relative difference of 1e-3 (8x the ~1.2e-4 quantization step) still sort front
    // to back (tight condition at the 1024x768 tile count).
    let means2d = MLXArray([8.0, 8.0, 8.0, 8.0, 8.0, 8.0] as [Float], [3, 2])
    let radii = MLXArray([Int32](repeating: 3, count: 6), [3, 2])
    let depths = MLXArray([Float(5.0 * 1.002), Float(5.0), Float(5.0 * 1.001)], [3])

    let result = TileBinning.bin(
        means2d: means2d, radii: radii, depths: depths,
        imageWidth: 1024, imageHeight: 768, tileSize: 16
    )
    #expect(result.flattenIds.asArray(Int32.self) == [1, 2, 0])
}
