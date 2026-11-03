// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// The viewer's initial camera reproduces the training view saved next to the PLY.

import Foundation
import Testing
import simd
@testable import SplatViewerKit

/// The orbit camera must start at the source view's eye and look along its optical axis, even when
/// the scene center is far off that axis.
@Test
func previewCameraReproducesTrainingViewPose() throws {
    // Row-major world→camera (OpenCV, +Z forward), looking down and to the side.
    let viewmat: [Float] = [
        0.9786774, -0.0004100761, -0.2054106, -0.64733285,
        0.10346492, -0.862895, 0.4946805, 0.35901976,
        -0.17745036, -0.5053845, -0.8444519, -1.6583405,
        0, 0, 0, 1,
    ]
    let m = viewmat
    let eye = SIMD3<Float>(
        -(m[0] * m[3] + m[4] * m[7] + m[8] * m[11]),
        -(m[1] * m[3] + m[5] * m[7] + m[9] * m[11]),
        -(m[2] * m[3] + m[6] * m[7] + m[10] * m[11]))
    let forward = simd_normalize(SIMD3<Float>(m[8], m[9], m[10]))

    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("preview-camera-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    let ply = dir.appendingPathComponent("scene.ply")
    let json = """
        {"version":1,"width":1080,"height":810,"viewmat":\(viewmat),"intrinsics":[757.8,757.8,540,405]}
        """
    try json.write(to: dir.appendingPathComponent("scene.preview.json"), atomically: true, encoding: .utf8)

    let sceneCenter = SIMD3<Float>(0.25, -0.4, -2.55)
    let preview = try #require(
        SplatViewerPreviewCamera.load(siblingOf: ply, center: sceneCenter, defaultDistance: 3))
    let camera = SplatViewerCamera(
        azimuth: preview.azimuth, elevation: preview.elevation, distance: preview.distance,
        center: preview.center, fovyRadians: preview.fovyRadians, aspect: 4.0 / 3.0)

    #expect(simd_distance(camera.eye, eye) < 1e-4)
    #expect(simd_distance(camera.forward, forward) < 1e-4)
}
