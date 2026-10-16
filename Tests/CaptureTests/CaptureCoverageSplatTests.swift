// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import simd
import Testing
@testable import Capture

@Test
func coverageAccumulatorMergesRepeatedObservationsInVoxel() {
    var acc = CaptureCoverageAccumulator(voxelSizeMeters: 0.1, maxSplats: 100)

    acc.insert(SIMD3<Float>(0.01, 0.02, 0.03), timestamp: 1)
    acc.insert(SIMD3<Float>(0.02, 0.02, 0.04), timestamp: 2)

    let splats = acc.snapshot()
    #expect(splats.count == 1)
    #expect(splats[0].observationCount == 2)
    #expect(splats[0].opacity > 0.18)
    #expect(abs(splats[0].position.x - 0.015) < 1e-6)
    #expect(splats[0].voxelSizeMeters == 0.1)
}

@Test
func coverageAccumulatorKeepsDistinctVoxels() {
    var acc = CaptureCoverageAccumulator(voxelSizeMeters: 0.1, maxSplats: 100)

    acc.insert([
        SIMD3<Float>(0.01, 0.02, 0.03),
        SIMD3<Float>(0.22, 0.02, 0.03),
    ], timestamp: 1)

    #expect(acc.voxelCount == 2)
    #expect(acc.snapshot().count == 2)
}

/// Past the cap, instead of discarding low-observation voxels, coarsen the display grid by 2× steps
/// and merge all points.
@Test
func coverageAccumulatorOverCapCoarsensInsteadOfDropping() {
    var acc = CaptureCoverageAccumulator(voxelSizeMeters: 0.1, maxSplats: 1)

    acc.insert(SIMD3<Float>(0.21, 0.0, 0.0), timestamp: 1)
    acc.insert(SIMD3<Float>(0.01, 0.0, 0.0), timestamp: 2)   // Still 2 on a 0.2 grid → 1 on a 0.3 grid
    acc.insert(SIMD3<Float>(0.02, 0.0, 0.0), timestamp: 3)   // Added incrementally to the coarse grid

    let splats = acc.snapshot()
    #expect(splats.count == 1)
    #expect(splats[0].observationCount == 3)
    #expect(abs(splats[0].position.x - 0.08) < 1e-6)
    #expect(abs(acc.displayVoxelSizeMeters - 0.3) < 1e-6)
    #expect(splats[0].voxelSizeMeters == acc.displayVoxelSizeMeters, "returns the coarse display voxel edge length")
    #expect(acc.voxelCount == 2, "display voxels themselves are not discarded")
}

/// Incremental additions after coarsening (including negative coordinates) match re-merging all
/// points on the final display grid.
@Test
func coverageAccumulatorCoarseDisplayMatchesRegrouping() {
    var acc = CaptureCoverageAccumulator(voxelSizeMeters: 0.1, maxSplats: 4)
    // x kept away from grid boundaries (steps of 0.1 + 0.03). Inserted in mixed order.
    let xs: [Float] = (0..<20).map { Float($0 - 10) * 0.1 + 0.03 }
    let order = [7, 0, 13, 19, 2, 11, 5, 16, 9, 1, 18, 4, 14, 8, 3, 17, 6, 12, 10, 15]
    for i in order { acc.insert(SIMD3<Float>(xs[i], -0.05, 0.02), timestamp: 0) }

    let size = acc.displayVoxelSizeMeters
    var groups: [Int: [Float]] = [:]
    for x in xs { groups[Int(floor(x / size)), default: []].append(x) }
    let splats = acc.snapshot().sorted { $0.position.x < $1.position.x }
    let expected = groups.sorted { $0.key < $1.key }.map { $0.value }
    #expect(splats.count <= 4)
    #expect(splats.count == expected.count)
    for (s, g) in zip(splats, expected) {
        #expect(s.observationCount == g.count)
        #expect(abs(s.position.x - g.reduce(0, +) / Float(g.count)) < 1e-5)
    }
    #expect(acc.voxelCount == 20)
}

@Test
func coverageAccumulatorAveragesColorAndCountsDistinctFrames() {
    var acc = CaptureCoverageAccumulator(voxelSizeMeters: 0.1, maxSplats: 100)

    // 2 points from frame 3, 1 point from frame 5 → 2 distinct keyframes.
    acc.insert([SIMD3<Float>(0.01, 0, 0), SIMD3<Float>(0.02, 0, 0)], timestamp: 0,
               colors: [SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, 1, 0)], frameIndex: 3)
    acc.insert([SIMD3<Float>(0.03, 0, 0)], timestamp: 0,
               colors: [SIMD3<Float>(0, 0, 1)], frameIndex: 5)

    let splats = acc.snapshot()
    #expect(splats.count == 1)
    #expect(splats[0].viewCount == 2)
    #expect(splats[0].observationCount == 3)
    let c = splats[0].color!
    #expect(abs(c.x - 1.0 / 3) < 1e-6 && abs(c.y - 1.0 / 3) < 1e-6 && abs(c.z - 1.0 / 3) < 1e-6)
}

@Test
func coverageAccumulatorCountsViewsOnCoarseGrid() {
    // Display 0.1, view counts on a 0.3 grid: different display voxels in the same 0.3 cell share
    // the view count.
    var acc = CaptureCoverageAccumulator(voxelSizeMeters: 0.1, maxSplats: 100, viewVoxelSizeMeters: 0.3)

    acc.insert([SIMD3<Float>(0.05, 0, 0)], timestamp: 0, frameIndex: 1)
    acc.insert([SIMD3<Float>(0.05, 0, 0)], timestamp: 0, frameIndex: 2)
    acc.insert([SIMD3<Float>(0.15, 0, 0)], timestamp: 0, frameIndex: 3)   // Neighboring display voxel, same coarse cell

    let splats = acc.snapshot().sorted { $0.position.x < $1.position.x }
    #expect(splats.count == 2)
    #expect(splats.allSatisfy { $0.viewCount == 3 })
    #expect(splats.allSatisfy { $0.color == nil })
}

/// The view count of a coarse display voxel is the largest among the view cells inside it (not
/// dragged down by the 1-view point inserted first).
@Test
func coverageAccumulatorCoarseCellUsesBestViewCount() {
    var acc = CaptureCoverageAccumulator(voxelSizeMeters: 0.1, maxSplats: 1, viewVoxelSizeMeters: 0.1)
    acc.insert([SIMD3<Float>(0.15, 0, 0)], timestamp: 0, frameIndex: 0)        // The 1-view point comes first
    for f in 1...4 { acc.insert([SIMD3<Float>(0.05, 0, 0)], timestamp: 0, frameIndex: f) }
    let splats = acc.snapshot()
    #expect(splats.count == 1)
    #expect(splats[0].viewCount == 4)
}

/// The coarse display omits cells with few points (everything is shown while not coarsened).
@Test
func coverageAccumulatorHidesSparseCellsOnlyWhenCoarse() {
    var acc = CaptureCoverageAccumulator(voxelSizeMeters: 0.1, maxSplats: 2, minPointsWhenCoarse: 3)
    acc.insert([SIMD3<Float>(0.05, 0, 0), SIMD3<Float>(0.06, 0, 0), SIMD3<Float>(0.07, 0, 0)], timestamp: 0)
    acc.insert([SIMD3<Float>(1.05, 0, 0)], timestamp: 0)          // Isolated point
    #expect(acc.snapshot().count == 2, "within the cap, single-point voxels are emitted too")
    acc.insert([SIMD3<Float>(2.05, 0, 0)], timestamp: 0)          // Over the cap → coarse display
    // On a 1.2m grid, 2 cells: {0.05-0.07, 1.05} (4 points) and {2.05} (1 point). The 1-point one
    // is not shown.
    let splats = acc.snapshot()
    #expect(abs(acc.displayVoxelSizeMeters - 1.2) < 1e-5)
    #expect(splats.count == 1)
    #expect(splats[0].observationCount == 4)
}

@Test
func coverageAccumulatorWithoutFramesReportsZeroViews() {
    var acc = CaptureCoverageAccumulator(voxelSizeMeters: 0.1, maxSplats: 100)
    acc.insert(SIMD3<Float>(0.01, 0.02, 0.03), timestamp: 1)
    #expect(acc.snapshot()[0].viewCount == 0)
}
