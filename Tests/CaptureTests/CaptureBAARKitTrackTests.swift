// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// CaptureBA.measureARKitTracks: builds multi-view tracks by matching on ARKit feature point IDs and
// measuring the position with NCC patch matching around the projection (GPU-independent). Synthetic
// images: frame f is texture G translated by s_f (I_f[y][x] = G(x+0.5−s_f.x, y+0.5−s_f.y)), so a
// point at G coordinate g maps to pixel g + s_f. The ARKit projection is given offset by a few px
// and the test checks that the measurement returns to the true value.

import Foundation
import simd
import Testing
import Trainer
@testable import Capture

private let intr: Intrinsics = (100, 100, 60, 60)
private let size = 120

/// OpenCV camera coordinates that project to pixel (u, v) at z=1.
private func obs(_ id: UInt64, _ p: SIMD2<Float>) -> ARKitFeatureObservation {
    ARKitFeatureObservation(trackID: id,
                            cameraPoint: SIMD3<Float>((p.x - intr.cx) / intr.fx, (p.y - intr.cy) / intr.fy, 1))
}

private func texture(_ x: Float, _ y: Float) -> Float {
    128 + 40 * sin(0.31 * x + 0.17 * y) + 30 * cos(0.23 * x - 0.41 * y)
        + 25 * sin(0.53 * x * 0.7 + 0.11 * y * 1.3) * cos(0.07 * y)
}

private func shiftedImage(_ s: SIMD2<Float>, _ g: (Float, Float) -> Float = texture) -> [Float] {
    var img = [Float](repeating: 0, count: size * size)
    for y in 0..<size {
        for x in 0..<size { img[y * size + x] = g(Float(x) + 0.5 - s.x, Float(y) + 0.5 - s.y) }
    }
    return img
}

private func measure(
    _ observations: [[ARKitFeatureObservation]], images: [[Float]]
) -> (tracks: [[CaptureBA.TrackObservation]], stats: CaptureBA.ARKitTrackStats) {
    let n = observations.count
    return CaptureBA.measureARKitTracks(
        featureObservations: observations,
        intrinsics: Array(repeating: intr, count: n),
        sizes: Array(repeating: (width: size, height: size), count: n),
        luma: { images[$0] }
    )
}

private let shifts: [SIMD2<Float>] = [[0, 0], [3.3, -2.6], [-4.7, 1.2], [6.2, 5.5]]
private let noise: [SIMD2<Float>] = [[1.5, -2], [-3, 2.5], [2, 3], [-2.5, -1]]

@Test
func arkitTracksRecoverSubpixelPositions() {
    let g = SIMD2<Float>(58.2, 61.7)   // G coordinate of the physical point
    let frames = (0..<4).map { [obs(7, g + shifts[$0] + noise[$0])] }
    let r = measure(frames, images: shifts.map { shiftedImage($0) })
    #expect(r.tracks.count == 1)
    #expect(r.stats.aligned == 3 && r.stats.rejected == 0)
    let t = r.tracks[0]
    #expect(t.map(\.frame) == [0, 1, 2, 3])
    // Whether the physical point at the reference (center = frame 2) pixel is measured correctly in
    // the other frames.
    let ref = t[2]
    let gRef = ref.pixel - shifts[2]
    for o in t {
        let expected = gRef + shifts[o.frame]
        #expect(simd_length(o.pixel - expected) < 0.25,
                "frame \(o.frame): measured \(o.pixel) expected \(expected)")
    }
}

@Test
func arkitTracksDropLowTexture() {
    let flat = [Float](repeating: 100, count: size * size)
    let frames = (0..<3).map { _ in [obs(1, SIMD2<Float>(60, 60))] }
    let r = measure(frames, images: Array(repeating: flat, count: 3))
    #expect(r.tracks.isEmpty)
    #expect(r.stats.candidates == 1 && r.stats.lowTexture == 1)
}

@Test
func arkitTracksRejectMismatchedFrame() {
    // Only frame 3 has a different texture (= ARKit's ID match is wrong) → rejected by NCC.
    let g = SIMD2<Float>(58.2, 61.7)
    let frames = (0..<4).map { [obs(3, g + shifts[$0] + noise[$0])] }
    var images = shifts.map { shiftedImage($0) }
    images[3] = shiftedImage(shifts[3]) { x, y in 128 + 60 * sin(0.9 * x * y / 40) * cos(0.37 * y + 1) }
    let r = measure(frames, images: images)
    #expect(r.stats.rejected == 1)
    #expect(r.tracks.count == 1)
    #expect(r.tracks[0].map(\.frame) == [0, 1, 2])
}

@Test
func arkitTracksIgnoreOffImageAndBehindCamera() {
    let behind = ARKitFeatureObservation(trackID: 1, cameraPoint: SIMD3<Float>(0, 0, -1))
    let nearEdge = obs(2, SIMD2<Float>(5, 60))   // Does not fit within the margin (patch+search)
    let r = measure(Array(repeating: [behind, nearEdge], count: 3),
                    images: Array(repeating: shiftedImage([0, 0]), count: 3))
    #expect(r.tracks.isEmpty)
    #expect(r.stats.candidates == 0)
}
