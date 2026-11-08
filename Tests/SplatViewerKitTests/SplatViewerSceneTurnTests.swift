// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Orient: turning a whole scene upright in the viewer by quarter turns, without changing its file.

import Foundation
import Testing
import simd
@testable import SplatViewerKit

private let identity = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)

private func camera(azimuth: Float = 0, elevation: Float = 0, scene: simd_quatf = identity) -> SplatViewerCamera {
    SplatViewerCamera(
        azimuth: azimuth, elevation: elevation, distance: 3, center: SIMD3(0.2, -0.1, 0.4),
        fovyRadians: 1, aspect: 0.5, sceneRotation: scene)
}

/// Where a scene direction shows on screen: its components along the camera's right and up.
private func onScreen(_ v: SIMD3<Float>, _ camera: SplatViewerCamera) -> SIMD2<Float> {
    SIMD2(simd_dot(v, camera.axis(0)), simd_dot(v, camera.axis(1)))
}

/// A model made Z-up (as some apps write it) is lying on its back for a Y-up orbit, seen from the
/// front showing its top; tipping it away once stands it up, and the orbit then turns about its Z.
@Test
func tipBackStandsAZUpSceneUpright() {
    let turned = SplatViewerSceneTurn.tipBack.applied(to: identity, azimuth: 0, elevation: 0)
    let cam = camera(scene: turned)
    #expect(simd_distance(cam.up, SIMD3(0, 0, 1)) < 1e-5)
    // Scene +Z shows straight up on screen.
    #expect(simd_distance(onScreen(SIMD3(0, 0, 1), cam), SIMD2(0, 1)) < 1e-5)
}

/// Rotate left turns the scene counter-clockwise on screen, about the point the camera orbits, even
/// from an oblique view; four presses bring it back.
@Test
func rotateLeftTurnsCounterClockwiseOnScreen() {
    var q = identity
    q = SplatViewerSceneTurn.rotateLeft.applied(to: q, azimuth: 0.2, elevation: 0.1)
    let after = camera(scene: q)
    // What was up on screen now points left.
    let up = onScreen(SIMD3(0, 1, 0), after)
    #expect(up.x < -0.9)
    // The orbit point stays put on screen.
    #expect(simd_length(SIMD2(after.cameraSpace(after.center).x, after.cameraSpace(after.center).y)) < 1e-5)
    for _ in 0..<3 { q = SplatViewerSceneTurn.rotateLeft.applied(to: q, azimuth: 0.2, elevation: 0.1) }
    #expect(abs(abs(q.real) - 1) < 1e-6)
}

/// Each turn is undone by the opposite one, and the result is an exact quarter-turn rotation so
/// repeated presses don't drift.
@Test
func turnsAreExactAndUndoEachOther() {
    let pairs: [(SplatViewerSceneTurn, SplatViewerSceneTurn)] = [(.rotateLeft, .rotateRight), (.tipForward, .tipBack)]
    var q = identity
    for k in 0..<40 {
        q = SplatViewerSceneTurn.allCases[k % 4].applied(to: q, azimuth: Float(k) * 0.37, elevation: 0.3)
    }
    let m = simd_float3x3(q)
    for c in 0..<3 {
        for r in 0..<3 { #expect(m[c][r] == m[c][r].rounded()) }
    }
    for (a, b) in pairs {
        let back = b.applied(to: a.applied(to: q, azimuth: 1, elevation: -0.4), azimuth: 1, elevation: -0.4)
        #expect(abs(abs(simd_dot(back.vector, q.vector)) - 1) < 1e-6)
    }
}

/// Rays and projections stay in the scene's coordinates however it is turned, so a tap still hits
/// what is drawn under it.
@Test
func turnedCameraRayMeetsProjectedPoint() throws {
    let q = SplatViewerSceneTurn.tipForward.applied(
        to: SplatViewerSceneTurn.rotateRight.applied(to: identity, azimuth: 0.5, elevation: 0.2),
        azimuth: 0.5, elevation: 0.2)
    let cam = camera(azimuth: 0.5, elevation: 0.2, scene: q)
    let point = SIMD3<Float>(0.5, 0.3, -0.2)
    let screen = try #require(cam.project(cameraSpace: cam.cameraSpace(point)))
    let ray = cam.ray(normalized: screen)
    let along = simd_dot(point - ray.origin, ray.direction)
    #expect(simd_distance(ray.origin + ray.direction * along, point) < 1e-4)
}

/// The training view still opens the viewer from the source camera after the scene is turned.
@Test
func previewCameraFollowsTheSceneTurn() {
    let forward = simd_normalize(SIMD3<Float>(0.3, -0.4, -0.8))
    let preview = SplatViewerPreviewCamera(
        azimuth: 0, elevation: 0, forward: forward, distance: 2, fovyRadians: 1, center: .zero)
    let q = SplatViewerSceneTurn.tipBack.applied(to: identity, azimuth: 0, elevation: 0)
    let angles = preview.orbitAngles(sceneRotation: q)
    let cam = camera(azimuth: angles.azimuth, elevation: angles.elevation, scene: q)
    #expect(simd_distance(cam.forward, forward) < 1e-5)
}

/// A CAD model in a turned scene turns about the viewer's up, and Reset stands it as in its file
/// in the viewer.
@Test
func cadTurnsFollowTheViewersUp() {
    let q = SplatViewerSceneTurn.tipBack.applied(to: identity, azimuth: 0, elevation: 0)
    let cam = camera(scene: q)
    let box = SplatViewerCropRegion(shape: .box, center: .zero, halfExtent: SIMD3(0.1, 0.2, 0.3), radius: 1)
    let left = SplatViewerCADTurn.left.applied(to: box, camera: cam)
    #expect(simd_distance(left.rotation.act(cam.up), cam.up) < 1e-5)
    let reset = SplatViewerCADTurn.resetRotation.applied(to: left, camera: cam)
    // The model's own +Y points up in the viewer.
    #expect(simd_distance(q.act(reset.rotation.act(SIMD3(0, 1, 0))), SIMD3(0, 1, 0)) < 1e-5)
    let laid = SplatViewerCADTurn.layDown.applied(to: reset, camera: cam)
    #expect(abs(simd_dot(laid.rotation.act(SIMD3(0, 1, 0)), cam.up)) < 1e-5)
}

/// The saved rotation round-trips, and a broken one reads as no turn.
@Test
func sceneRotationCoding() throws {
    let q = SplatViewerSceneTurn.tipForward.applied(to: identity, azimuth: 0, elevation: 0)
    let data = try JSONEncoder().encode(SplatViewerSceneRotation(q))
    let read = try JSONDecoder().decode(SplatViewerSceneRotation.self, from: data)
    #expect(simd_distance(read.quaternion.vector, q.vector) < 1e-6)
    var broken = SplatViewerSceneRotation(q)
    broken.rotation = .zero
    #expect(broken.quaternion.vector == identity.vector)
}
