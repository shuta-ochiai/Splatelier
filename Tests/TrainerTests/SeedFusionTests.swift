// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import Testing
@testable import Trainer

/// Points in the same voxel become 1 point by confidence-weighted average; points in different
/// voxels are not merged.
@Test
func seedFusionWeightsPointsInSameVoxel() {
    // voxel 0.1m: (0.01, 0.01, 0.01) and (0.05, 0.05, 0.05) share a voxel, (0.25, 0, 0) is in
    // another.
    let a = (
        positions: MLXArray([Float(0.01), 0.01, 0.01, 0.25, 0, 0], [2, 3]),
        colors: MLXArray([Float(1), 0, 0, 0, 0, 1], [2, 3]),
        confidences: MLXArray([Float(3), 1], [2])
    )
    let b = (
        positions: MLXArray([Float(0.05), 0.05, 0.05], [1, 3]),
        colors: MLXArray([Float(0), 1, 0], [1, 3]),
        confidences: MLXArray([Float(1)], [1])
    )
    let (positions, colors) = SeedFusion.fuse(clouds: [a, b], voxelSize: 0.1)
    #expect(positions.shape == [2, 3])
    let p = positions.asArray(Float.self), c = colors.asArray(Float.self)
    // Output order is unspecified, so tell them apart by x.
    let merged = p[0] < 0.2 ? 0 : 1
    let other = 1 - merged
    // (0.01·3 + 0.05·1) / 4 = 0.02
    for k in 0..<3 { #expect(abs(p[merged * 3 + k] - 0.02) < 1e-6) }
    #expect(abs(c[merged * 3 + 0] - 0.75) < 1e-6 && abs(c[merged * 3 + 1] - 0.25) < 1e-6)
    #expect(abs(p[other * 3 + 0] - 0.25) < 1e-6 && c[other * 3 + 2] == 1)
}

/// Negative coordinates and both sides of a voxel boundary separate correctly (xoring a random
/// value into the key does not change voxel identity).
@Test
func seedFusionSeparatesVoxelsAcrossOrigin() {
    let xs: [Float] = [-0.15, -0.05, 0.05, 0.15, -0.06, 0.04]   // 4 voxel (-0.2..-0.1, -0.1..0, 0..0.1, 0.1..0.2)
    var flat: [Float] = []
    for x in xs { flat += [x, -0.05, 0.05] }
    let cloud = (
        positions: MLXArray(flat, [xs.count, 3]),
        colors: MLXArray([Float](repeating: 0.5, count: xs.count * 3), [xs.count, 3]),
        confidences: MLXArray([Float](repeating: 1, count: xs.count), [xs.count])
    )
    let (positions, _) = SeedFusion.fuse(clouds: [cloud, cloud], voxelSize: 0.1)
    let p = positions.asArray(Float.self)
    let px = stride(from: 0, to: p.count, by: 3).map { p[$0] }.sorted()
    let expected: [Float] = [-0.15, -0.055, 0.045, 0.15]
    #expect(px.count == expected.count)
    for (a, e) in zip(px, expected) { #expect(abs(a - e) < 1e-6) }
}
