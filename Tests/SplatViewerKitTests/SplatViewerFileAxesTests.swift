// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Splats files are stood upright the way other viewers stand them: a half turn about Z, except for
// a capture PLY written Y-up before captures were turned.

import Foundation
import Testing
import simd
@testable import SplatViewerKit

private func tempDir() throws -> URL {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("file-axes-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

private func writeHeader(_ lines: [String], to url: URL) throws {
    let header = (["ply", "format binary_little_endian 1.0"] + lines
        + ["element vertex 0", "property float x", "end_header"]).joined(separator: "\n") + "\n"
    try Data(header.utf8).write(to: url)
}

private func isHalfTurn(_ q: simd_quatf) -> Bool {
    simd_distance(q.act(SIMD3(1, 2, 3)), SIMD3(-1, -2, 3)) < 1e-5
}

@Test
func filesStandUprightAsOtherViewersStandThem() throws {
    let dir = try tempDir()
    defer { try? FileManager.default.removeItem(at: dir) }

    // Turned by Splatelier: the comment says so, whatever is beside it.
    let turned = dir.appendingPathComponent("turned.ply")
    try writeHeader([SplatViewerFileAxes.turnedComment], to: turned)
    try Data("{}".utf8).write(to: dir.appendingPathComponent("turned.preview.json"))
    #expect(isHalfTurn(SplatViewerFileAxes.uprightRotation(forSplatsAt: turned)))

    // A capture from before: Y-up, left as it is.
    let legacy = dir.appendingPathComponent("legacy.ply")
    try writeHeader([], to: legacy)
    try Data("{}".utf8).write(to: dir.appendingPathComponent("legacy.preview.json"))
    #expect(SplatViewerFileAxes.uprightRotation(forSplatsAt: legacy).real == 1)

    // From elsewhere, in any format: the usual 3DGS axes.
    let imported = dir.appendingPathComponent("imported.ply")
    try writeHeader(["comment made elsewhere"], to: imported)
    #expect(isHalfTurn(SplatViewerFileAxes.uprightRotation(forSplatsAt: imported)))
    #expect(isHalfTurn(SplatViewerFileAxes.uprightRotation(forSplatsAt: dir.appendingPathComponent("x.spz"))))
    #expect(SplatViewerFileAxes.uprightRotation(forSplatsAt: nil).real == 1)
}

/// The source view saved in the capture's world still opens the viewer from the training camera
/// once the splats are in turned coordinates and stood up again.
@Test
func previewCameraOpensFromTheSourceViewInATurnedFile() throws {
    let dir = try tempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let ply = dir.appendingPathComponent("scene.ply")
    let viewmat: [Float] = [
        0.9786774, -0.0004100761, -0.2054106, -0.64733285,
        0.10346492, -0.862895, 0.4946805, 0.35901976,
        -0.17745036, -0.5053845, -0.8444519, -1.6583405,
        0, 0, 0, 1,
    ]
    let json = """
        {"version":1,"width":1080,"height":810,"viewmat":\(viewmat),"intrinsics":[757.8,757.8,540,405]}
        """
    try json.write(to: dir.appendingPathComponent("scene.preview.json"), atomically: true, encoding: .utf8)
    let m = viewmat
    let eyeWorld = SIMD3<Float>(
        -(m[0] * m[3] + m[4] * m[7] + m[8] * m[11]),
        -(m[1] * m[3] + m[5] * m[7] + m[9] * m[11]),
        -(m[2] * m[3] + m[6] * m[7] + m[10] * m[11]))
    let forwardWorld = simd_normalize(SIMD3<Float>(m[8], m[9], m[10]))

    let half = SplatViewerFileAxes.halfTurnZ
    // The scene's center, in the file's (turned) coordinates.
    let center = half.act(SIMD3<Float>(0.25, -0.4, -2.55))
    let preview = try #require(SplatViewerPreviewCamera.load(
        siblingOf: ply, center: center, defaultDistance: 3, sceneFromWorld: half.inverse))
    let angles = preview.orbitAngles(sceneRotation: half)
    let camera = SplatViewerCamera(
        azimuth: angles.azimuth, elevation: angles.elevation, distance: preview.distance,
        center: preview.center, fovyRadians: preview.fovyRadians, aspect: 4.0 / 3.0, sceneRotation: half)
    #expect(simd_distance(camera.eye, half.inverse.act(eyeWorld)) < 1e-4)
    #expect(simd_distance(camera.forward, half.inverse.act(forwardWorld)) < 1e-4)
    // Up on screen is up in the capture's world.
    #expect(simd_distance(half.act(camera.up), SIMD3(0, 1, 0)) < 1e-5)
}

/// The capture's mesh is brought into the turned coordinates with the splats.
@Test
func meshIsTurnedIntoTheSplatsCoordinates() {
    let mesh = LoadedMesh(
        positions: [0, 0, 0, 1, 0, 0, 0, 2, 1], normals: [0, 0, 1, 0, 0, 1, 0, 1, 0],
        colors: [], indices: [0, 1, 2], center: SIMD3(0.5, 1, 0.5), radius: 1.2)
    let turned = mesh.rotated(by: SplatViewerFileAxes.halfTurnZ)
    #expect(turned.positions.map { ($0 * 1e4).rounded() / 1e4 } == [0, 0, 0, -1, 0, 0, 0, -2, 1])
    #expect(simd_distance(turned.center, SIMD3(-0.5, -1, 0.5)) < 1e-5)
    #expect(abs(turned.normals[7] + 1) < 1e-5)
}
