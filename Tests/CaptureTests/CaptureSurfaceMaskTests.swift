// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import MLX
import Testing
import Trainer
@testable import Capture

private let identity: [Float] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

private func frame(lidarSamples: [LiDARSample] = []) -> CapturedFrame {
    CapturedFrame(
        image: MLXArray.zeros([240, 320, 3], type: UInt8.self), viewmat: identity,
        intrinsics: (fx: 100, fy: 100, cx: 160, cy: 120), width: 320, height: 240, timestamp: 0,
        lidarSamples: lidarSamples)
}

/// An MVS depth map at half the frame's resolution marks the cells it covers in frame coordinates.
@Test func mvsSurfaceCellsFollowTheDepthMapIntrinsics() {
    // Depth on the left half of the map only.
    let (w, h) = (160, 120)
    let depth = (0..<(w * h)).map { $0 % w < w / 2 ? Float(1) : 0 }
    let map = CostVolumeMVS.DepthView(
        width: w, height: h, depth: depth, rgb: [UInt8](repeating: 0, count: w * h * 3),
        viewmat: identity, intrinsics: (fx: 50, fy: 50, cx: 80, cy: 60))
    let cells = CaptureTraining.mvsSurfaceCells(map, frame: frame())
    let (rows, cols) = CaptureTraining.surfaceMaskGrid
    for r in 0..<rows {
        #expect(cells[(r * cols)..<(r * cols + cols)] == ArraySlice((0..<cols).map { $0 < cols / 2 ? 1 : 0 }))
    }
}

/// The opaque cells are the surface cells away from its edges; the image border is no edge.
@Test func opaqueMaskLeavesOutTheSurfaceEdge() {
    let (rows, cols) = CaptureTraining.surfaceMaskGrid
    let mvs = (0..<(rows * cols)).map { Float($0 % cols < cols / 2 ? 1 : 0) }
    // A LiDAR sample in the middle of cell (12, 24), on the side MVS left empty.
    let sample = LiDARSample(cameraPoint: SIMD3(0.85, 0.05, 1), confidenceLevel: 2, depthSpread: 0)
    let masks = try! #require(CaptureTraining.surfaceMasks(frame(lidarSamples: [sample]), cells: mvs))
    let surface = masks.surface.asArray(Float.self), opaque = masks.opaque.asArray(Float.self)
    #expect(surface[12 * cols + 24] == 1)
    #expect(opaque[12 * cols + 24] == 0)
    for r in 0..<rows {
        for c in 0..<cols / 2 { #expect(surface[r * cols + c] == 1) }
        for c in 0..<(cols / 2 - 1) { #expect(opaque[r * cols + c] == 1, "cell (\(r), \(c))") }
        #expect(opaque[r * cols + cols / 2 - 1] == 0)
    }
    #expect(CaptureTraining.surfaceMasks(frame(), cells: nil) == nil)
}

/// A cell counts as measured once enough sparse points land in it; points behind the camera or
/// outside the image do not count.
@Test func sparseSurfaceCellsCountPointsPerCell() {
    let (rows, cols) = CaptureTraining.surfaceMaskGrid
    // The middle of cell (12, 24) at depths 1, 2 and 3 (the identity pose looks down +z).
    let inCell = (1...3).map { SIMD3<Float>(0.85, 0.05, 1) * Float($0) }
    let behind = SIMD3<Float>(-0.85, -0.05, -1)   // projects onto the same pixel from behind
    let outside = SIMD3<Float>(5, 0, 1)
    let two = CaptureTraining.sparseSurfaceCells(Array(inCell.prefix(2)) + [behind, outside], frame: frame())
    #expect(two.allSatisfy { $0 == 0 })
    let three = CaptureTraining.sparseSurfaceCells(inCell, frame: frame())
    #expect(three.count == rows * cols)
    #expect(three[12 * cols + 24] == 1)
    #expect(three.reduce(0, +) == 1)
}
