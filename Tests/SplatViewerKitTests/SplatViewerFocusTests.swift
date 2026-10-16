// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Viewer double-tap zoom: the resulting look-at point and distance.

import Testing
import simd
@testable import SplatViewerKit

/// On a hit, that point comes to screen center (= becomes the center of rotation) and the distance
/// from the camera halves. On a miss, the look-at point stays and only the distance halves.
@Test
func doubleTapFocusCentersTappedPointAndHalvesDistance() throws {
    let camera = SplatViewerCamera(
        azimuth: 0.7, elevation: -0.4, distance: 3, center: SIMD3(0.2, -0.1, 1),
        fovyRadians: 1.0, aspect: 0.46)
    let ray = camera.ray(normalized: SIMD2(0.15, 0.8))
    let hit = ray.origin + ray.direction * 2.4

    let target = camera.focused(on: hit, zoom: 2)
    var after = camera
    after.center = target.center
    after.distance = target.distance
    let projected = try #require(after.project(cameraSpace: after.cameraSpace(hit)))
    #expect(simd_distance(projected, SIMD2(0.5, 0.5)) < 1e-4)
    #expect(abs(simd_distance(after.eye, hit) - 1.2) < 1e-4)

    let miss = camera.focused(on: nil, zoom: 2)
    #expect(miss.center == camera.center)
    #expect(abs(miss.distance - 1.5) < 1e-6)

    // Don't zoom in too far (inside near, the look-at point gets clipped in rendering).
    let close = camera.focused(on: camera.eye + camera.forward * 0.06, zoom: 2)
    #expect(close.distance == SplatViewerCamera.near)
}
