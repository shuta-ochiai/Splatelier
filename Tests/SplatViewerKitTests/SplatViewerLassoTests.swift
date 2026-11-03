// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Lasso crop: which splats a loop drawn on screen covers, from the front to the back, by their
// centers.

import Foundation
import Testing
import simd
@testable import SplatViewerKit

private let aspect: Float = 1000.0 / 800.0

private func camera() -> SplatViewerCamera {
    SplatViewerCamera(
        azimuth: 0, elevation: 0, distance: 5, center: .zero,
        fovyRadians: 1.0, aspect: aspect)
}

/// A square loop around the screen's center, `half` wide on each side (fraction of the view).
private func square(half: Float) -> [SIMD2<Float>] {
    let c = SIMD2<Float>(0.5, 0.5)
    return [c + SIMD2(-half, -half), c + SIMD2(half, -half), c + SIMD2(half, half), c + SIMD2(-half, half)]
}

@Test
func polygonContainment() {
    let polygon: [SIMD2<Float>] = [SIMD2(0, 0), SIMD2(10, 0), SIMD2(10, 10), SIMD2(0, 10)]
    #expect(SplatViewerLasso.contains(polygon, SIMD2(5, 5)))
    #expect(!SplatViewerLasso.contains(polygon, SIMD2(11, 5)))
    // Concave loops count only their own inside.
    let notch: [SIMD2<Float>] = [SIMD2(0, 0), SIMD2(10, 0), SIMD2(10, 10), SIMD2(5, 4), SIMD2(0, 10)]
    #expect(!SplatViewerLasso.contains(notch, SIMD2(5, 8)))
    #expect(SplatViewerLasso.contains(notch, SIMD2(5, 2)))
}

@Test
func loopCutsFromFrontToBack() {
    let test = SplatViewerLasso.test(polygon: square(half: 0.1), camera: camera())
    // On the view's axis, near the camera and far from it: both are under the loop.
    #expect(test(SIMD3(0, 0, 3)))
    #expect(test(SIMD3(0, 0, -20)))
    // Off to the side, outside the loop.
    #expect(!test(SIMD3(2, 0, 0)))
}

@Test
func splatBehindTheCameraIsOutside() {
    let test = SplatViewerLasso.test(polygon: square(half: 0.4), camera: camera())
    #expect(!test(SIMD3(0, 0, 8)))
}

@Test
func drawingKeepsPointsApartAndRejectsLines() {
    var loop: [SIMD2<Float>] = []
    for p in [SIMD2<Float>(0.5, 0.5), SIMD2(0.5005, 0.5), SIMD2(0.6, 0.5)] {
        loop = SplatViewerLasso.appending(p, to: loop, aspect: 1.25)
    }
    #expect(loop.count == 2)
    #expect(!SplatViewerLasso.isUsable(loop))
    #expect(!SplatViewerLasso.isUsable([SIMD2(0.1, 0.1), SIMD2(0.2, 0.2), SIMD2(0.3, 0.3)]))
    #expect(SplatViewerLasso.isUsable(square(half: 0.1)))
}

@Test
func lassoChangesNothingInTheRegion() {
    let box = SplatViewerCropRegion(
        shape: .box, center: SIMD3(1, 2, 3), halfExtent: SIMD3(1, 1, 1), radius: 1)
    #expect(box.converted(to: .lasso) == box)
    #expect(box.converted(to: .lasso).converted(to: .box) == box)
}

@Test
func loopIsJoinedWhenTheEndReturnsToTheStart() {
    let aspect: Float = 1.25
    // A ring of 16 points that stops short of the start, then one that gets back to it.
    let ring: [SIMD2<Float>] = (0..<16).map { i in
        let t = 2 * Float.pi * Float(i) / 16
        return SIMD2(0.5 + 0.2 * cos(t), 0.5 + 0.2 * sin(t))
    }
    #expect(!SplatViewerLasso.isJoined(Array(ring.prefix(10)), aspect: aspect))
    #expect(SplatViewerLasso.isJoined(ring + [ring[0] + SIMD2(0.01, 0)], aspect: aspect))
    // The first few points are always near the start, which is not a loop.
    #expect(!SplatViewerLasso.isJoined(Array(ring.prefix(4)), aspect: aspect))
}
