// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Viewer crop tool: containment, handle drags that follow the finger on screen, and the edit
// history (apply / undo / restore).

import Foundation
import Testing
import simd
@testable import SplatViewerKit

private let viewSize = SIMD2<Float>(1000, 800)

private func camera(azimuth: Float = 0.3, elevation: Float = -0.2) -> SplatViewerCamera {
    SplatViewerCamera(
        azimuth: azimuth, elevation: elevation, distance: 5, center: SIMD3(0, 0, 0),
        fovyRadians: 1.0, aspect: viewSize.x / viewSize.y)
}

/// Pixel position of a world point on the view.
private func screen(_ p: SIMD3<Float>, _ camera: SplatViewerCamera) -> SIMD2<Float> {
    let n = camera.project(cameraSpace: camera.cameraSpace(p)) ?? .zero
    return n * viewSize
}

/// Pixel movement on screen of moving `from` by `step` in the world.
private func pixels(moving from: SIMD3<Float>, by step: SIMD3<Float>, _ camera: SplatViewerCamera) -> SIMD2<Float> {
    screen(from + step, camera) - screen(from, camera)
}

@Test
func boxAndSphereContainment() {
    let box = SplatViewerCropRegion(
        shape: .box, center: SIMD3(1, 0, 0), halfExtent: SIMD3(1, 2, 3), radius: 1)
    #expect(box.contains(SIMD3(1.5, 1.9, -2.9)))
    #expect(!box.contains(SIMD3(2.1, 0, 0)))
    #expect(!box.contains(SIMD3(0, 0, 3.5)))

    let sphere = SplatViewerCropRegion(
        shape: .sphere, center: .zero, halfExtent: SIMD3(repeating: 1), radius: 1)
    #expect(sphere.contains(SIMD3(0.6, 0.6, 0.6)) == false)
    #expect(sphere.contains(SIMD3(0.5, 0.5, 0.5)))
}

/// Switching shape keeps the center; the sphere takes the box's largest half size, and back, the
/// box takes the radius on every side.
@Test
func shapeConversionKeepsCenter() {
    let box = SplatViewerCropRegion(
        shape: .box, center: SIMD3(1, 2, 3), halfExtent: SIMD3(1, 4, 2), radius: 1)
    let sphere = box.converted(to: .sphere)
    #expect(sphere.center == box.center)
    #expect(sphere.radius == 4)
    let back = sphere.converted(to: .box)
    #expect(back.center == box.center)
    #expect(back.halfExtent == SIMD3(repeating: 4))
}

/// Dragging a box face by the screen movement of a world step moves that face by the step and
/// leaves the opposite face where it was.
@Test
func boxFaceFollowsTheFinger() {
    let cam = camera()
    let region = SplatViewerCropRegion(
        shape: .box, center: .zero, halfExtent: SIMD3(repeating: 1), radius: 1)
    let face = SIMD3<Float>(1, 0, 0)
    let d = pixels(moving: face, by: SIMD3(0.2, 0, 0), cam)
    let moved = region.dragged(.face(axis: 0, side: 1), byPixels: d, camera: cam, viewSize: viewSize)
    #expect(abs(moved.center.x + moved.halfExtent.x - 1.2) < 0.01)
    #expect(abs(moved.center.x - moved.halfExtent.x + 1) < 0.001)
    #expect(moved.halfExtent.y == 1 && moved.center.y == 0)
}

/// The other face moves the low side. Shrinking past the minimum stops at it.
@Test
func boxFaceStopsAtMinimumSize() {
    let cam = camera()
    let region = SplatViewerCropRegion(
        shape: .box, center: .zero, halfExtent: SIMD3(repeating: 1), radius: 1)
    let low = SIMD3<Float>(-1, 0, 0)
    // Dragging the low face past the high face (x = 1).
    let d = pixels(moving: low, by: SIMD3(2.5, 0, 0), cam)
    let moved = region.dragged(.face(axis: 0, side: -1), byPixels: d, camera: cam, viewSize: viewSize)
    // The high side stays; the low side stops just short of it at the minimum size.
    #expect(abs(moved.center.x + moved.halfExtent.x - 1) < 1e-4)
    #expect(abs(moved.halfExtent.x - SplatViewerCropRegion.minimumSize) < 1e-3)
}

@Test
func sphereRadiusFollowsTheFinger() {
    let cam = camera()
    let region = SplatViewerCropRegion(
        shape: .sphere, center: .zero, halfExtent: SIMD3(repeating: 1), radius: 1)
    let right = cam.axis(0)
    let d = pixels(moving: right * region.radius, by: right * 0.3, cam)
    let moved = region.dragged(.radius, byPixels: d, camera: cam, viewSize: viewSize)
    #expect(abs(moved.radius - 1.3) < 0.01)
    #expect(moved.center == region.center)
}

@Test
func sphereCenterFollowsTheFinger() {
    let cam = camera()
    let region = SplatViewerCropRegion(
        shape: .sphere, center: .zero, halfExtent: SIMD3(repeating: 1), radius: 1)
    let right = cam.axis(0), up = cam.axis(1)
    // Drag by a screen movement equal to 0.2 along camera right and 0.1 along camera up.
    let d = pixels(moving: .zero, by: right * 0.2 + up * 0.1, cam)
    let moved = region.dragged(.center, byPixels: d, camera: cam, viewSize: viewSize)
    #expect(simd_distance(moved.center, right * 0.2 + up * 0.1) < 0.005)
    #expect(moved.radius == 1)
}

@Test
func boxCenterFollowsTheFinger() {
    let cam = camera()
    let region = SplatViewerCropRegion(
        shape: .box, center: .zero, halfExtent: SIMD3(1, 2, 3), radius: 1)
    let right = cam.axis(0), up = cam.axis(1)
    let d = pixels(moving: .zero, by: right * 0.2 + up * 0.1, cam)
    let moved = region.dragged(.center, byPixels: d, camera: cam, viewSize: viewSize)
    #expect(simd_distance(moved.center, right * 0.2 + up * 0.1) < 0.005)
    #expect(moved.halfExtent == region.halfExtent)
    #expect(moved.rotation == region.rotation)
}

/// The grabbed point of a ring follows the finger along the ring: the box turns about that axis
/// by the arc the finger covers over the radius of the ring, and nothing else changes.
@Test
func ringTurnsTheBoxAboutItsAxis() {
    let cam = camera()
    let region = SplatViewerCropRegion(
        shape: .box, center: SIMD3(1, 0, 0), halfExtent: SIMD3(1, 1, 1), radius: 1)
    for axis in 0..<3 {
        let angle: Float = 0.7
        let p = region.ringPoint(axis: axis, angle: angle)
        let around = region.rotation.act(SIMD3<Float>(Float(axis == 0 ? 1 : 0), Float(axis == 1 ? 1 : 0), Float(axis == 2 ? 1 : 0)))
        let tangent = simd_normalize(simd_cross(around, p - region.center))
        let d = pixels(moving: p, by: tangent * 0.2, cam)
        let turned = region.dragged(.ring(axis: axis, angle: angle), byPixels: d, camera: cam, viewSize: viewSize)
        #expect(turned.center == region.center)
        #expect(turned.halfExtent == region.halfExtent)
        let expected = simd_quatf(angle: 0.2 / region.ringRadius, axis: around)
        let moved = turned.ringPoint(axis: axis, angle: angle)
        let wanted = region.center + expected.act(p - region.center)
        // The grabbed point of the ring ends up where the finger took it (the arc is a little
        // shorter than the straight step along the tangent).
        #expect(simd_distance(moved, wanted) < 0.01)
        // The axis itself stays where it was.
        #expect(simd_distance(turned.rotation.act(SIMD3(Float(axis == 0 ? 1 : 0), Float(axis == 1 ? 1 : 0), Float(axis == 2 ? 1 : 0))), around) < 1e-4)
    }
}

/// Touching a ring finds it, and the angle where it was touched.
@Test
func touchingARingFindsTheRingAndWhereOnIt() {
    let cam = camera()
    let region = SplatViewerCropRegion(
        shape: .box, center: .zero, halfExtent: SIMD3(1, 2, 3), radius: 1)
    // A little off the middle of the arc.
    let angle = SplatViewerCropRegion.ringArcMiddle + 0.3
    let touch = screen(region.ringPoint(axis: 2, angle: angle), cam) / viewSize
    let found = region.nearestRing(to: touch, camera: cam, viewSize: viewSize)!
    #expect(found.distance < 1)
    guard case let .ring(axis, found) = found.handle else {
        Issue.record("not a ring")
        return
    }
    #expect(axis == 2)
    #expect(abs(found - angle) < 0.05)
    // The rest of the ring is not there to grab.
    let away = screen(region.ringPoint(axis: 2, angle: angle + .pi), cam) / viewSize
    #expect(region.nearestRing(to: away, camera: cam, viewSize: viewSize)!.distance > 20)
    #expect(SplatViewerCropRegion(shape: .sphere, center: .zero, halfExtent: .one, radius: 1)
        .nearestRing(to: touch, camera: cam, viewSize: viewSize) == nil)
}

/// A turned box contains what lies inside it, not inside the axis-aligned box around the same
/// center, and its face handles follow it.
@Test
func turnedBoxContainsAlongItsOwnAxes() {
    var region = SplatViewerCropRegion(
        shape: .box, center: SIMD3(1, 0, 0), halfExtent: SIMD3(2, 0.5, 0.5), radius: 1)
    region.rotation = simd_quatf(angle: .pi / 2, axis: SIMD3(0, 0, 1))
    // The long side now runs along world Y.
    #expect(region.contains(SIMD3(1, 1.8, 0)))
    #expect(!region.contains(SIMD3(2.8, 0, 0)))
    let face = region.handles(camera: camera()).first { $0.handle == .face(axis: 0, side: 1) }!.position
    #expect(simd_distance(face, SIMD3(1, 2, 0)) < 1e-4)
}

/// A face of a turned box moves along the box's own axis.
@Test
func turnedBoxFaceMovesAlongItsAxis() {
    let cam = camera()
    var region = SplatViewerCropRegion(
        shape: .box, center: .zero, halfExtent: SIMD3(repeating: 1), radius: 1)
    region.rotation = simd_quatf(angle: .pi / 2, axis: SIMD3(0, 0, 1))
    let face = SIMD3<Float>(0, 1, 0)
    let d = pixels(moving: face, by: SIMD3(0, 0.2, 0), cam)
    let moved = region.dragged(.face(axis: 0, side: 1), byPixels: d, camera: cam, viewSize: viewSize)
    #expect(abs(moved.halfExtent.x - 1.1) < 0.01)
    #expect(abs(moved.center.y - 0.1) < 0.01)
    #expect(abs(moved.center.x) < 1e-4)
}

@Test
func initialRegionIgnoresStrayPoints() {
    var positions = (0..<100).map { SIMD3<Float>(Float($0) * 0.01, 0, 0) }
    positions.append(SIMD3(1000, 0, 0))
    let region = SplatViewerCropRegion.initial(shape: .box, positions: positions)
    #expect(region.halfExtent.x < 1)
    #expect(region.contains(SIMD3(0.5, 0, 0)))
}

@Test
func editsApplyUndoAndRestore() {
    var edits = SplatViewerCropEdits(count: 5)
    #expect(edits.apply { $0 >= 2 } == .applied)
    #expect(edits.kept == [2, 3, 4])
    #expect(edits.apply { $0 >= 3 } == .applied)
    #expect(edits.kept == [3, 4])
    #expect(edits.apply { $0 >= 10 } == .emptied)
    #expect(edits.kept == [3, 4])
    #expect(edits.apply { _ in true } == .unchanged)
    #expect(edits.isEdited && edits.canUndo)

    edits.undo()
    #expect(edits.kept == [2, 3, 4])
    edits.restoreAll()
    #expect(edits.kept == [0, 1, 2, 3, 4])
    #expect(!edits.isEdited && !edits.canUndo)
}

/// The orange selection of a Box or Sphere is the kept splats inside the region, by their place in
/// the kept list, not in the loaded splats.
@Test
func regionSelectsTheKeptSplatsInside() {
    let positions: [SIMD3<Float>] = [SIMD3(0, 0, 0), SIMD3(5, 0, 0), SIMD3(0.5, 0.5, 0), SIMD3(9, 9, 9)]
    let box = SplatViewerCropRegion(shape: .box, center: .zero, halfExtent: SIMD3(1, 1, 1), radius: 1)
    // Splat 2 and 0 are inside; splat 1 was cropped away already, so it is not in `kept`.
    let kept: [UInt32] = [0, 2, 3]
    #expect(box.insideIndices(kept: kept) { positions[Int($0)] } == [0, 1])
    #expect(box.converted(to: .sphere).insideIndices(kept: kept) { positions[Int($0)] } == [0, 1])
}

/// The arcs belong to the box: the camera moving doesn't move them, and the box turning does.
@Test
func ringArcsStayOnTheBox() {
    let region = SplatViewerCropRegion(
        shape: .box, center: SIMD3(1, 0, 0), halfExtent: SIMD3(1, 2, 3), radius: 1)
    let a = region.ringArc(axis: 1).map(\.point)
    #expect(a.count == SplatViewerCropRegion.ringArcSteps + 1)
    // Nothing in the arcs depends on the camera; turning the box carries them along.
    var turned = region
    turned.rotation = simd_quatf(angle: 0.5, axis: SIMD3(0, 1, 0))
    let b = turned.ringArc(axis: 1).map(\.point)
    let local = region.rotation.inverse.act(a[3] - region.center)
    #expect(simd_distance(turned.center + turned.rotation.act(local), b[3]) < 1e-4)
}
