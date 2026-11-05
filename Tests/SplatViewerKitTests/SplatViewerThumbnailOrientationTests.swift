// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// The library thumbnail is turned so world up is at the top, whichever way the phone was held.

import Foundation
import Testing
@testable import SplatViewerKit

/// Row-major world→camera with the image's right (+X) and down (+Y) axes given in world; the
/// optical axis is their cross product.
private func viewmat(right r: SIMD3<Float>, down d: SIMD3<Float>) -> [Float] {
    let f = SIMD3<Float>(r.y * d.z - r.z * d.y, r.z * d.x - r.x * d.z, r.x * d.y - r.y * d.x)
    return [r.x, r.y, r.z, 0, d.x, d.y, d.z, 0, f.x, f.y, f.z, 0, 0, 0, 0, 1]
}

@Test(arguments: [
    // Landscape with the image top toward the sky: already upright.
    (SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, -1, 0), 0),
    // Portrait (sensor top toward the left of the image is world up): clockwise.
    (SIMD3<Float>(0, -1, 0), SIMD3<Float>(1, 0, 0), 1),
    // Landscape the other way round: upside down.
    (SIMD3<Float>(-1, 0, 0), SIMD3<Float>(0, 1, 0), 2),
    // Portrait upside down: counter-clockwise.
    (SIMD3<Float>(0, 1, 0), SIMD3<Float>(-1, 0, 0), 3),
])
func thumbnailTurnsWorldUpToTheTop(right: SIMD3<Float>, down: SIMD3<Float>, turns: Int) {
    #expect(SplatViewerThumbnailOrientation.clockwiseQuarterTurns(
        trainingViewmat: viewmat(right: right, down: down)) == turns)
}

/// A camera looking straight down has no clear up in the image, so the caller keeps its default.
@Test
func thumbnailLookingStraightDownHasNoTurn() {
    let m = viewmat(right: SIMD3<Float>(1, 0, 0), down: SIMD3<Float>(0, 0, 1))
    #expect(SplatViewerThumbnailOrientation.clockwiseQuarterTurns(trainingViewmat: m) == nil)
}

/// Read from the `.preview.json` written next to the thumbnail.
@Test
func thumbnailTurnsReadFromPreviewMetadata() throws {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("thumbnail-orientation-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    let png = dir.appendingPathComponent("scene.png")
    #expect(SplatViewerThumbnailOrientation.clockwiseQuarterTurns(siblingOf: png) == nil)

    let m = viewmat(right: SIMD3<Float>(0, -1, 0), down: SIMD3<Float>(1, 0, 0))
    let json = """
        {"version":1,"width":1080,"height":810,"viewmat":\(m),"intrinsics":[757.8,757.8,540,405]}
        """
    try json.write(to: dir.appendingPathComponent("scene.preview.json"), atomically: true, encoding: .utf8)
    #expect(SplatViewerThumbnailOrientation.clockwiseQuarterTurns(siblingOf: png) == 1)
}
