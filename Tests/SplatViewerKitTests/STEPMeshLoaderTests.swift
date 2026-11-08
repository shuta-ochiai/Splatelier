// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// STEP loading through cadmpeg. The furniture fixtures are written by scripts/make_step_fixtures.py
// (millimetres, Z-up); the loader returns metres, Y-up.

import Foundation
import Metal
import Testing
import simd
@testable import SplatViewerKit

private func fixture(_ name: String) -> URL {
    URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("Fixtures/STEP/\(name).step")
}

private func extent(_ mesh: LoadedMesh) -> (lo: SIMD3<Float>, hi: SIMD3<Float>) {
    var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
    var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
    for v in 0..<mesh.vertexCount {
        let p = SIMD3(mesh.positions[v * 3], mesh.positions[v * 3 + 1], mesh.positions[v * 3 + 2])
        lo = pointwiseMin(lo, p)
        hi = pointwiseMax(hi, p)
    }
    return (lo, hi)
}

private func colors(_ mesh: LoadedMesh) -> Set<SIMD3<Float>> {
    var result = Set<SIMD3<Float>>()
    for v in 0..<mesh.vertexCount {
        let c = SIMD3<Float>(mesh.colors[v * 3], mesh.colors[v * 3 + 1], mesh.colors[v * 3 + 2])
        // STEP colours are written with two decimals; round off float noise.
        result.insert((c * 100).rounded(.toNearestOrEven) / 100)
    }
    return result
}

private func near(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ tolerance: Float = 1e-4) -> Bool {
    simd_reduce_max(simd_abs(a - b)) < tolerance
}

/// A 1200 × 700 mm table, 720 mm tall, comes back in metres with its height on +Y and the CAD +Y
/// on −Z, and keeps the colours of its wooden top and steel legs.
@Test
func stepLoaderReadsTableInMetresYUp() throws {
    let mesh = try STEPMeshLoader.load(fixture("table"))
    let (lo, hi) = extent(mesh)
    #expect(near(lo, SIMD3(0, 0, -0.7)))
    #expect(near(hi, SIMD3(1.2, 0.72, 0)))
    #expect(colors(mesh) == [SIMD3(0.72, 0.53, 0.34), SIMD3(0.25, 0.25, 0.27)])
    #expect(mesh.normals.count == mesh.positions.count)
}

/// The bookshelf is an assembly that places one shelf part five times and one side panel twice;
/// every placement is drawn (eight boxes of 12 triangles) at its own position.
@Test
func stepLoaderExpandsAssemblyInstances() throws {
    let mesh = try STEPMeshLoader.load(fixture("bookshelf"))
    #expect(mesh.triangleCount == 8 * 12)
    let (lo, hi) = extent(mesh)
    #expect(near(lo, SIMD3(0, 0, -0.3)))
    #expect(near(hi, SIMD3(0.8, 1.8, 0)))
}

/// Cylinders (curved faces with a seam edge) are tessellated, not dropped.
@Test
func stepLoaderTessellatesCylinders() throws {
    let mesh = try STEPMeshLoader.load(fixture("side_table"))
    let (lo, hi) = extent(mesh)
    #expect(near(lo, SIMD3(-0.3, 0, -0.3), 1e-3))
    #expect(near(hi, SIMD3(0.3, 0.545, 0.3), 1e-3))
    #expect(mesh.triangleCount > 100)
}

/// A file that is not STEP fails with cadmpeg's reason instead of crashing.
@Test
func stepLoaderRejectsNonStepFile() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("not-step-\(UUID().uuidString).step")
    try "solid cube\nendsolid cube\n".write(to: url, atomically: true, encoding: .utf8)
    defer { try? FileManager.default.removeItem(at: url) }
    #expect(throws: STEPMeshLoader.LoadError.self) { try STEPMeshLoader.load(url) }
}

/// A CAD model turns about its own center, then shifts: the center lands on center + offset, and a
/// point beside it swings around it.
@MainActor
@Test
func cadPlacementTurnsAboutPivotThenShifts() {
    let model = SplatViewerModel()
    model.setCAD(pivot: SIMD3(1, 0, 0), halfExtent: SIMD3(1, 1, 1), offset: SIMD3(0, 0.5, 0))
    model.cadRotation = simd_quatf(angle: .pi / 2, axis: SIMD3(0, 1, 0))
    func place(_ p: SIMD3<Float>) -> SIMD3<Float> {
        let q = model.cadTransform * SIMD4(p, 1)
        return SIMD3(q.x, q.y, q.z)
    }
    #expect(near(place(SIMD3(1, 0, 0)), SIMD3(1, 0.5, 0)))
    // +X of the center turns to −Z (a quarter turn counter-clockwise seen from above).
    #expect(near(place(SIMD3(2, 0, 0)), SIMD3(1, 0.5, -1)))
}

/// The model is moved and turned through its bounding box, handled as a Crop box: dragging the
/// box's move handle or a ring places the model where the box goes.
@MainActor
@Test
func cadFollowsItsDraggedBox() {
    let model = SplatViewerModel()
    model.setCAD(pivot: SIMD3(0, 1, 0), halfExtent: SIMD3(0.4, 1, 0.2), offset: SIMD3(2, 0, 0))
    var box = model.cadRegion
    #expect(near(box.center, SIMD3(2, 1, 0)))
    #expect(near(box.halfExtent, SIMD3(0.4, 1, 0.2)))
    box.center += SIMD3(0, 0, -1)
    box.rotation = simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))
    model.setCADRegion(box)
    #expect(near(model.cadOffset, SIMD3(2, 0, -1)))
    // The model's center follows the box; its +X end is now on the box's −X side.
    let end = model.cadTransform * SIMD4<Float>(0.4, 1, 0, 1)
    #expect(near(SIMD3(end.x, end.y, end.z), SIMD3(1.6, 1, -1)))
}

/// A tap ray finds a moved CAD model where it is drawn, not where the file put it.
@Test
func rayCastHitsCADWhereItIsPlaced() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    // A 2 × 2 m square in the z = 0 plane around the origin.
    let positions: [Float] = [-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]
    let indices: [UInt32] = [0, 1, 2, 0, 2, 3]
    let p = try #require(device.makeBuffer(bytes: positions, length: positions.count * 4, options: .storageModeShared))
    let i = try #require(device.makeBuffer(bytes: indices, length: indices.count * 4, options: .storageModeShared))
    var square = MeshRayGeometry(positions: p, indices: i, triangleCount: 2)
    square.transform = simd_float4x4(SIMD4(1, 0, 0, 0), SIMD4(0, 1, 0, 0), SIMD4(0, 0, 1, 0), SIMD4(5, 0, -2, 1))
    let caster = SplatViewerRayCaster(splatChunks: [], mesh: nil, cad: square)
    let down = SIMD3<Float>(0, 0, -1)
    // Where the file put it there is nothing now; 5 m to the right it is 2 m further away.
    #expect(caster.cast(origin: SIMD3(0, 0, 3), direction: down, pixelAngle: 0.001) == nil)
    let t = try #require(caster.cast(origin: SIMD3(5, 0, 3), direction: down, pixelAngle: 0.001))
    #expect(abs(t - 5) < 1e-5)
}

/// A CAD model reopens where it was left.
@MainActor
@Test
func cadPlacementRestores() {
    let model = SplatViewerModel()
    let saved = SplatViewerCADPlacement(offset: SIMD3(1, 2, 3), rotation: simd_quatf(angle: 0.5, axis: SIMD3(0, 1, 0)))
    model.setCAD(pivot: .zero, halfExtent: SIMD3(1, 1, 1), offset: SIMD3(0, 0, -1), saved: saved)
    #expect(model.cadPlacement == saved)
    #expect(!model.isCADSelected)
}

/// Selecting the model is let go by another mode or by hiding the model.
@MainActor
@Test
func cadSelectionEndsWithAnotherModeOrHiding() {
    let model = SplatViewerModel()
    model.setCAD(pivot: .zero, halfExtent: SIMD3(1, 1, 1), offset: .zero)
    model.isCADSelected = true
    model.isMeasuring = true
    #expect(!model.isCADSelected)
    model.isMeasuring = false
    model.isCADSelected = true
    model.showCAD = false
    #expect(!model.isCADSelected)
}

private let gizmoCamera = SplatViewerCamera(
    azimuth: 0.5, elevation: 0.4, distance: 4, center: SIMD3(0, 0.5, 0), fovyRadians: 1.0, aspect: 1)
private let gizmoView = SIMD2<Float>(1000, 1000)
private let uprightBox = SplatViewerCropRegion(
    shape: .box, center: SIMD3(0, 0.5, 0), halfExtent: SIMD3(0.4, 0.5, 0.2), radius: 1)

private func screen(_ p: SIMD3<Float>) throws -> SIMD2<Float> {
    try #require(gizmoCamera.project(cameraSpace: gizmoCamera.cameraSpace(p)))
}

/// The rings sit up and to the right of the model on screen, off it: a touch on the model's center is
/// left to moving it, and a touch on a ring grabs that ring, from any view.
@Test
func cadGizmoRingsSitBesideTheModel() throws {
    let front = SplatViewerCamera(azimuth: 0, elevation: 0, distance: 4, center: uprightBox.center, fovyRadians: 1.0, aspect: 1)
    for camera in [gizmoCamera, front] {
        let gizmo = SplatViewerCADGizmo(box: uprightBox, camera: camera)
        func pick(_ touch: SIMD2<Float>) -> SplatViewerCADHandle? {
            gizmo.ring(at: touch, camera: camera, viewSize: gizmoView, ringReach: 15)
        }
        func onScreen(_ p: SIMD3<Float>) throws -> SIMD2<Float> {
            try #require(camera.project(cameraSpace: camera.cameraSpace(p)))
        }
        let center = try onScreen(uprightBox.center), anchor = try onScreen(gizmo.anchor)
        // Up and to the right of the model on screen, outside its box.
        #expect(anchor.x > center.x && anchor.y < center.y)
        #expect(!uprightBox.contains(gizmo.anchor))
        #expect(pick(center) == nil)
        for axis in 0..<3 {
            guard case .ring(axis: axis, _)? = pick(try onScreen(gizmo.ringPoint(axis: axis, angle: 2.0))) else {
                Issue.record("expected the ring round axis \(axis)"); continue
            }
        }
        #expect(pick(SIMD2(0.02, 0.98)) == nil)
    }
    // Held in place (while a ring is dragged), the rings stay there whatever the camera.
    let held = SplatViewerCADGizmo(box: uprightBox, camera: front, anchor: SIMD3(1, 2, 3))
    #expect(held.anchor == SIMD3(1, 2, 3))
}

/// A model that fills the view still gets its rings inside the view, over its top right.
@Test
func cadGizmoRingsStayOnScreenForABigModel() throws {
    let close = SplatViewerCamera(azimuth: 0.3, elevation: 0.2, distance: 1.2, center: uprightBox.center, fovyRadians: 1.0, aspect: 0.5)
    let gizmo = SplatViewerCADGizmo(box: uprightBox, camera: close)
    for axis in 0..<3 {
        for angle in stride(from: Float(0), to: 2 * .pi, by: 0.5) {
            let p = try #require(close.project(cameraSpace: close.cameraSpace(gizmo.ringPoint(axis: axis, angle: angle))))
            #expect(p.x > 0 && p.x < 1 && p.y > 0 && p.y < 1)
        }
    }
}

/// Dragged along its edge-on line, a ring still turns the model.
@Test
func cadGizmoEdgeOnRingStillTurns() {
    let front = SplatViewerCamera(azimuth: 0, elevation: 0, distance: 4, center: uprightBox.center, fovyRadians: 1.0, aspect: 1)
    let gizmo = SplatViewerCADGizmo(box: uprightBox, camera: front)
    // Seen from straight in front, the ring round Y is a horizontal line beside the model; dragging
    // along it turns the model about Y.
    let turned = gizmo.dragged(.ring(axis: 1, angle: 2.0), byPixels: SIMD2(60, 0), camera: front, viewSize: gizmoView)
    #expect(abs(turned.rotation.angle) > 0.05)
}

/// A clockwise twist seen from above turns the model clockwise seen from above (a negative turn
/// about +Y), keeping it upright and in place.
@Test
func cadGizmoTwistTurnsAboutTheVertical() {
    let gizmo = SplatViewerCADGizmo(box: uprightBox, camera: gizmoCamera)
    let twisted = gizmo.twisted(by: 0.3, camera: gizmoCamera)
    let up = SIMD3<Float>(0, 1, 0)
    #expect(simd_distance(twisted.rotation.act(up), up) < 1e-4)
    #expect(simd_distance(twisted.rotation.act(SIMD3(1, 0, 0)), SIMD3(cosf(0.3), 0, sinf(0.3))) < 1e-4)
    #expect(twisted.center == uprightBox.center)
}

/// Dragging a ring turns the model about that axis of its own, keeping its center; each ring
/// carries two arrows.
@Test
func cadGizmoRingTurnsAboutItsAxis() {
    let gizmo = SplatViewerCADGizmo(box: uprightBox, camera: gizmoCamera)
    let up = SIMD3<Float>(0, 1, 0)
    let turned = gizmo.dragged(.ring(axis: 1, angle: 0.3), byPixels: SIMD2(80, 0), camera: gizmoCamera, viewSize: gizmoView)
    #expect(simd_distance(turned.rotation.act(up), up) < 1e-4)
    #expect(abs(turned.rotation.angle) > 0.05)
    #expect(simd_distance(turned.center, uprightBox.center) < 1e-5)
    let tilted = gizmo.dragged(.ring(axis: 0, angle: 1.2), byPixels: SIMD2(0, 80), camera: gizmoCamera, viewSize: gizmoView)
    #expect(simd_distance(tilted.rotation.act(up), up) > 0.05)
    let frame = gizmo.frame(camera: gizmoCamera)
    #expect(frame.rings.count == 3)
    #expect(frame.ringArrows.count == 6)
}

/// The rings are small, a fixed share of the view, not the size of the model: twice as far away they
/// are twice as large in the world, the same on screen.
@Test
func cadGizmoRingsKeepTheirSizeOnScreen() throws {
    let near = SplatViewerCADGizmo(box: uprightBox, camera: gizmoCamera)
    var farCamera = gizmoCamera
    farCamera.distance *= 2
    let far = SplatViewerCADGizmo(box: uprightBox, camera: farCamera)
    #expect(near.ringRadius < simd_length(uprightBox.halfExtent))
    #expect(abs(far.ringRadius / near.ringRadius - 2) < 1e-3)
    // Measured from the ring's own center, beside the model; the share is set at the model's depth,
    // so a ring a little nearer or further than it is a little larger or smaller.
    let top = try screen(near.ringPoint(axis: 0, angle: 0)), center = try screen(near.anchor)
    #expect(abs(simd_distance(top, center) - SplatViewerCADGizmo.ringScreenShare / 2) < 0.04)
}

/// The foot is under the model's center, level with its lowest corner as it now stands: laid on its
/// side, the side becomes the foot.
@Test
func cadFootIsTheLowestLevelUnderTheCenter() {
    #expect(near(SplatViewerCADSurface.foot(of: uprightBox), SIMD3(0, 0, 0)))
    var laid = uprightBox
    laid.rotation = simd_quatf(angle: .pi / 2, axis: SIMD3(0, 0, 1))
    // Turned onto its side, its 0.4 m half width now reaches down.
    #expect(near(SplatViewerCADSurface.foot(of: laid), SIMD3(0, 0.1, 0)))
}

/// The model moves freely and settles on a floor only when its foot comes within reach of it,
/// from above or below; further away it stays where it was dragged.
@Test
func cadSettlesOnAFloorWithinReach() {
    let reach: Float = 0.05
    let level = SplatViewerCADSurface.Floor(point: .zero, normal: SIMD3(0, 1, 0))
    var hovering = uprightBox
    hovering.center.y += 0.03
    let settled = SplatViewerCADSurface.settled(hovering, on: level, reach: reach)
    #expect(abs(SplatViewerCADSurface.foot(of: settled).y) < 1e-5)
    #expect(settled.rotation == hovering.rotation)
    var sunk = uprightBox
    sunk.center.y -= 0.04
    #expect(abs(SplatViewerCADSurface.foot(of: SplatViewerCADSurface.settled(sunk, on: level, reach: reach)).y) < 1e-5)
    var lifted = uprightBox
    lifted.center.y += 0.3
    #expect(SplatViewerCADSurface.settled(lifted, on: level, reach: reach) == lifted)
    #expect(SplatViewerCADSurface.settled(hovering, on: nil, reach: reach) == hovering)
}

/// On a leaning floor the model tilts to lie flat on it, its bottom's middle on the floor; laid on
/// its side it stays on its side. A floor that reads far off level is taken as a misreading.
@Test
func cadTiltsToLieFlatOnALeaningFloor() {
    let reach: Float = 0.1
    let lean = simd_quatf(angle: 0.1, axis: SIMD3(0, 0, 1)).act(SIMD3<Float>(0, 1, 0))
    let floor = SplatViewerCADSurface.Floor(point: .zero, normal: lean)
    let settled = SplatViewerCADSurface.settled(uprightBox, on: floor, reach: reach)
    #expect(simd_distance(settled.rotation.act(SIMD3(0, 1, 0)), lean) < 1e-4)
    let bottom = settled.center - lean * uprightBox.halfExtent.y
    #expect(abs(bottom.y - floor.height(atX: bottom.x, z: bottom.z)) < 1e-4)

    let onSide = SplatViewerCADTurn.layDown.applied(to: uprightBox, camera: gizmoCamera)
    let side = SplatViewerCADSurface.upAxis(of: onSide)
    #expect(side.axis != 1)
    var e = SIMD3<Float>.zero
    e[side.axis] = side.sign
    let settledOnSide = SplatViewerCADSurface.settled(onSide, on: floor, reach: 1)
    #expect(simd_distance(settledOnSide.rotation.act(e), lean) < 1e-4)

    let farOff = SplatViewerCADSurface.Floor(
        point: .zero, normal: simd_quatf(angle: 0.3, axis: SIMD3(0, 0, 1)).act(SIMD3<Float>(0, 1, 0)))
    let upright = SplatViewerCADSurface.settled(uprightBox, on: farOff, reach: reach)
    #expect(simd_distance(upright.rotation.act(SIMD3(0, 1, 0)), SIMD3(0, 1, 0)) < 1e-4)
}

/// The probes look straight down round the foot; a level surface within reach is a floor, a steep
/// one or one out of reach is not.
@Test
func cadFindsFloorsNotSlopesUnderItsFoot() {
    let foot = SIMD3<Float>(1, 0.02, 2), reach: Float = 0.05
    let probes = SplatViewerCADSurface.probes(under: foot, reach: reach, spread: 0.1)
    #expect(probes.count == 9 && probes.allSatisfy { abs($0.y - 0.07) < 1e-6 })
    let level = probes.map { SIMD3<Float>($0.x, 0, $0.z) }
    let found = SplatViewerCADSurface.floor(hits: level, foot: foot, reach: reach)
    #expect(found?.point.y == 0)
    #expect(found.map { simd_distance($0.normal, SIMD3(0, 1, 0)) < 1e-5 } == true)
    // A 60° slope rising along +X.
    let steep = probes.map { SIMD3<Float>($0.x, ($0.x - 1) * 1.7, $0.z) }
    #expect(SplatViewerCADSurface.floor(hits: steep, foot: foot, reach: reach) == nil)
    let far = probes.map { SIMD3<Float>($0.x, -0.5, $0.z) }
    #expect(SplatViewerCADSurface.floor(hits: far, foot: foot, reach: reach) == nil)
}

/// The slope of a surface comes from hits round a point, facing the camera; bumps of a few
/// centimetres on a floor don't tip it far, and a missing hit is passed over.
@Test
func cadSurfaceNormalFacesTheCamera() {
    let eye = SIMD3<Float>(0, 5, 0)
    let flat = SplatViewerCADSurface.normal(
        around: [SIMD3(1, 0, 0), SIMD3(0, 0, -1), SIMD3(-1, 0, 0), SIMD3(0, 0, 1)], toward: eye)
    #expect(flat.map { near($0, SIMD3(0, 1, 0)) } == true)
    let bumpy = SplatViewerCADSurface.normal(
        around: [SIMD3(0.2, 0.01, 0), SIMD3(0, -0.012, -0.2), SIMD3(-0.2, 0.008, 0), SIMD3(0, -0.01, 0.2)], toward: eye)
    #expect(bumpy.map { $0.y > SplatViewerCADSurface.minimumUp } == true)
    let missing = SplatViewerCADSurface.normal(around: [SIMD3(1, 0, 0), nil, SIMD3(-1, 0, 0), SIMD3(0, 0, 1)], toward: eye)
    #expect(missing.map { near($0, SIMD3(0, 1, 0)) } == true)
    #expect(SplatViewerCADSurface.normal(around: [.zero, SIMD3(1, 0, 0), SIMD3(2, 0, 0)], toward: .one) == nil)
}

/// The turn buttons: a quarter turn about the vertical either way, laid on its side toward the
/// camera's right, and Reset rotation back to the way it was imported. None moves its center.
@MainActor
@Test
func cadTurnButtons() {
    let x = SIMD3<Float>(1, 0, 0), up = SIMD3<Float>(0, 1, 0)
    let left = SplatViewerCADTurn.left.applied(to: uprightBox, camera: gizmoCamera)
    #expect(simd_distance(left.rotation.act(x), SIMD3(0, 0, -1)) < 1e-5)
    let right = SplatViewerCADTurn.right.applied(to: uprightBox, camera: gizmoCamera)
    #expect(simd_distance(right.rotation.act(x), SIMD3(0, 0, 1)) < 1e-5)
    let laid = SplatViewerCADTurn.layDown.applied(to: uprightBox, camera: gizmoCamera)
    var cameraRight = gizmoCamera.axis(0)
    cameraRight.y = 0
    #expect(simd_distance(laid.rotation.act(up), simd_normalize(cameraRight)) < 1e-4)
    let reset = SplatViewerCADTurn.resetRotation.applied(
        to: SplatViewerCADTurn.layDown.applied(to: left, camera: gizmoCamera), camera: gizmoCamera)
    #expect(reset.rotation == simd_quatf(ix: 0, iy: 0, iz: 0, r: 1))
    for turned in [left, right, laid, reset] {
        #expect(turned.center == uprightBox.center)
    }
    // The model says whether there is a turn to reset (the button is greyed out otherwise).
    let model = SplatViewerModel()
    model.setCAD(pivot: .zero, halfExtent: SIMD3(1, 1, 1), offset: .zero)
    #expect(!model.isCADTurned)
    model.setCADRegion(left)
    #expect(model.isCADTurned)
    model.setCADRegion(reset)
    #expect(!model.isCADTurned)
}

/// Removing the CAD model takes it out of the scene and lets go of it; a new one can be set up again.
@MainActor
@Test
func cadRemovalClearsTheModel() {
    let model = SplatViewerModel()
    model.setCAD(pivot: .zero, halfExtent: SIMD3(1, 1, 1), offset: SIMD3(1, 0, 0))
    model.isCADSelected = true
    model.setCADRegion(SplatViewerCADTurn.left.applied(to: model.cadRegion, camera: gizmoCamera))
    model.clearCAD()
    #expect(!model.hasCAD && !model.isCADSelected && !model.isCADTurned)
    model.setCAD(pivot: .zero, halfExtent: SIMD3(1, 1, 1), offset: .zero)
    #expect(model.hasCAD)
}

/// From far away the model is small and hard to hit, so a touch just beside it still takes hold; up
/// close an unselected model is only taken by touching it, so the camera can move round it; once
/// selected, anywhere in its box counts.
@Test
func cadReachIsGenerousForSmallOrSelectedModels() throws {
    let view = SIMD2<Float>(1000, 1000)
    func beside(_ camera: SplatViewerCamera) throws -> SIMD2<Float> {
        let bounds = try #require(SplatViewerCADReach.screenBounds(of: uprightBox, camera: camera))
        // Just past the right edge of its box, within the margin.
        return SIMD2(bounds.hi.x + 0.02, (bounds.lo.y + bounds.hi.y) / 2)
    }
    let far = SplatViewerCamera(azimuth: 0.3, elevation: 0.2, distance: 30, center: uprightBox.center, fovyRadians: 1.0, aspect: 1)
    #expect(SplatViewerCADReach.takes(try beside(far), box: uprightBox, selected: false, onModel: false, camera: far, viewSize: view))
    #expect(!SplatViewerCADReach.takes(SIMD2(0.05, 0.05), box: uprightBox, selected: false, onModel: false, camera: far, viewSize: view))

    let close = SplatViewerCamera(azimuth: 0.3, elevation: 0.2, distance: 2.5, center: uprightBox.center, fovyRadians: 1.0, aspect: 1)
    #expect(!SplatViewerCADReach.takes(try beside(close), box: uprightBox, selected: false, onModel: false, camera: close, viewSize: view))
    #expect(SplatViewerCADReach.takes(try beside(close), box: uprightBox, selected: true, onModel: false, camera: close, viewSize: view))
    #expect(SplatViewerCADReach.takes(SIMD2(0.05, 0.05), box: uprightBox, selected: false, onModel: true, camera: close, viewSize: view))
}
