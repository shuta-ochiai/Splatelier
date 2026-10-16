// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Viewer measurement: tap ray vs screen projection consistency, splat / mesh surface points,
// measurement-line projection and display.

import Foundation
import Metal
import MetalSplatter
import SplatIO
import Testing
import simd
@testable import SplatViewerKit

private func splat(
    _ position: SIMD3<Float>, scale: SIMD3<Float>, opacity: Float,
    rotation: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
) -> SplatPoint {
    SplatPoint(
        position: position, color: .sRGBUInt8(SIMD3(128, 128, 128)),
        opacity: .linearFloat(opacity), scale: .linearFloat(scale), rotation: rotation)
}

private func caster(_ points: [SplatPoint], mesh: MeshRayGeometry? = nil) throws -> SplatViewerRayCaster {
    let device = try #require(MTLCreateSystemDefaultDevice())
    return SplatViewerRayCaster(splatChunks: points.isEmpty ? [] : [try SplatChunk(device: device, from: points)], mesh: mesh)
}

private func mesh(_ positions: [Float], _ indices: [UInt32]) throws -> MeshRayGeometry {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let p = try #require(device.makeBuffer(bytes: positions, length: positions.count * 4, options: .storageModeShared))
    let i = try #require(device.makeBuffer(bytes: indices, length: indices.count * 4, options: .storageModeShared))
    return MeshRayGeometry(positions: p, indices: i, triangleCount: indices.count / 3)
}

/// A ray straight ahead of the camera (from +Z toward -Z).
private let down = SIMD3<Float>(0, 0, -1)
/// 1 px ≈ 1 mrad.
private let pixelAngle: Float = 0.001

/// Projecting a point on the tap ray returns to the same screen position, and that matches the
/// render's view / projection matrices (measurement lines don't drift from where you tapped or from
/// the drawn splats).
@Test
func cameraRayAndProjectionMatchRendering() throws {
    let camera = SplatViewerCamera(
        azimuth: 0.7, elevation: -0.4, distance: 3, center: SIMD3(0.2, -0.1, 1),
        fovyRadians: 1.0, aspect: 0.46)
    for p in [SIMD2<Float>(0.5, 0.5), SIMD2(0.1, 0.9), SIMD2(0.93, 0.07)] {
        let ray = camera.ray(normalized: p)
        let world = ray.origin + ray.direction * 2.5
        let back = try #require(camera.project(cameraSpace: camera.cameraSpace(world)))
        #expect(simd_distance(back, p) < 1e-4)

        let clip = camera.projectionMatrix * camera.viewMatrix * SIMD4<Float>(world, 1)
        let ndc = SIMD2<Float>(clip.x, clip.y) / clip.w
        #expect(simd_distance(SIMD2((ndc.x + 1) / 2, (1 - ndc.y) / 2), p) < 1e-4)
    }
    // The ray through screen center passes through the look-at point.
    let center = camera.ray(normalized: SIMD2(0.5, 0.5))
    #expect(simd_distance(center.origin + center.direction * camera.distance, camera.center) < 1e-4)
    #expect(simd_distance(camera.eye, camera.ray(normalized: SIMD2(0.2, 0.3)).origin) < 1e-6)
}

/// A large thin splat placed at an angle returns the ray-surface intersection, not the center's
/// depth. With half-precision covariance the thin-direction variance is buried in rounding error,
/// but it still hits.
@Test
func splatPickReturnsSurfaceOfTiltedThinSplat() throws {
    let rotation = simd_quatf(angle: .pi / 4, axis: SIMD3(1, 0, 0))
    let c = try caster([splat(.zero, scale: SIMD3(0.3, 0.3, 0.0005), opacity: 0.9, rotation: rotation)])
    let origin = SIMD3<Float>(0.1, 0.05, 3)
    let t = try #require(c.cast(origin: origin, direction: down, pixelAngle: pixelAngle))
    // Intersection with the plane n·x = 0 (the center's depth would give t = 3, 5 cm off).
    let n = rotation.act(SIMD3(0, 0, 1))
    let expected = -simd_dot(n, origin) / simd_dot(n, down)
    #expect(abs(t - expected) < 0.002)
}

/// Stops at the near opaque splat and doesn't pick the one behind.
@Test
func splatPickStopsAtFrontSurface() throws {
    let disc = SIMD3<Float>(0.2, 0.2, 0.001)
    let c = try caster([
        splat(SIMD3(0, 0, -0.5), scale: disc, opacity: 0.95),
        splat(SIMD3(0, 0, 0.4), scale: disc, opacity: 0.95),
    ])
    let t = try #require(c.cast(origin: SIMD3(0.02, 0, 3), direction: down, pixelAngle: pixelAngle))
    #expect(abs(t - 2.6) < 0.002)
}

/// Empty or nearly transparent areas are not treated as a surface.
@Test
func splatPickMissesEmptyAndFaintSpace() throws {
    let disc = SIMD3<Float>(0.1, 0.1, 0.001)
    let c = try caster([
        splat(.zero, scale: disc, opacity: 0.9),
        splat(SIMD3(2, 0, 0), scale: disc, opacity: 0.1),
    ])
    #expect(c.cast(origin: SIMD3(1, 0, 3), direction: down, pixelAngle: pixelAngle) == nil)
    #expect(c.cast(origin: SIMD3(2, 0, 3), direction: down, pixelAngle: pixelAngle) == nil)
    // Splats behind the camera (in front of near) aren't drawn, so they aren't hit.
    #expect(c.cast(origin: SIMD3(0, 0, -1), direction: down, pixelAngle: pixelAngle) == nil)
}

/// A translucent stack that never reaches the median returns the depth where it reaches half the
/// composited opacity.
@Test
func surfaceDepthUsesHalfOfCoverageWhenThin() {
    typealias Hit = SplatViewerRayCaster.SplatHit
    // Transmittance drops below 0.5 at the second one.
    #expect(SplatViewerRayCaster.surfaceDepth([Hit(t: 2, alpha: 0.4), Hit(t: 1, alpha: 0.3)]) == 2)
    // Composite 0.36: half (0.18) is reached at the first one (0.2).
    #expect(SplatViewerRayCaster.surfaceDepth([Hit(t: 1, alpha: 0.2), Hit(t: 2, alpha: 0.2)]) == 1)
    #expect(SplatViewerRayCaster.surfaceDepth([Hit(t: 1, alpha: 0.1)]) == nil)
    #expect(SplatViewerRayCaster.surfaceDepth([]) == nil)
}

/// For the mesh, the first hit triangle (two-sided). With both a splat and the mesh, the nearer one
/// wins.
@Test
func meshPickAndNearestLayerWins() throws {
    // A square at z = 0 (2 triangles).
    let square = try mesh([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0], [0, 1, 2, 0, 2, 3])
    let meshOnly = try caster([], mesh: square)
    let t = try #require(meshOnly.cast(origin: SIMD3(0.3, -0.2, 2), direction: down, pixelAngle: pixelAngle))
    #expect(abs(t - 2) < 1e-5)
    // Hits from behind too.
    let back = try #require(meshOnly.cast(origin: SIMD3(0.3, -0.2, -2), direction: -down, pixelAngle: pixelAngle))
    #expect(abs(back - 2) < 1e-5)
    #expect(meshOnly.cast(origin: SIMD3(3, 0, 2), direction: down, pixelAngle: pixelAngle) == nil)

    let disc = SIMD3<Float>(0.2, 0.2, 0.001)
    let splatInFront = try caster([splat(SIMD3(0, 0, 0.5), scale: disc, opacity: 0.95)], mesh: square)
    let front = try #require(splatInFront.cast(origin: SIMD3(0, 0, 2), direction: down, pixelAngle: pixelAngle))
    #expect(abs(front - 1.5) < 0.002)
    let splatBehind = try caster([splat(SIMD3(0, 0, -0.5), scale: disc, opacity: 0.95)], mesh: square)
    let hidden = try #require(splatBehind.cast(origin: SIMD3(0, 0, 2), direction: down, pixelAngle: pixelAngle))
    #expect(abs(hidden - 2) < 1e-5)
}

/// Measurement points form segments in pairs; with an odd count the last is a point only. Parts
/// behind the camera are clipped.
@Test
func measureProjectionPairsPointsAndClipsBehindCamera() throws {
    let camera = SplatViewerCamera(
        azimuth: 0, elevation: 0, distance: 2, center: .zero, fovyRadians: 1.0, aspect: 1)
    let frame = SplatMeasureProjection.frame(
        points: [SIMD3(-0.5, 0, 0), SIMD3(0.5, 0, 0), SIMD3(0, 0.3, 0)], camera: camera)
    #expect(frame.points.count == 3)
    #expect(frame.segments.count == 1)
    #expect(frame.segments.first?.label == "1.00 m")
    let segment = try #require(frame.segments.first)
    #expect(segment.a.x < 0.5 && segment.b.x > 0.5)
    #expect(abs(segment.a.y - 0.5) < 1e-5 && abs(segment.b.y - 0.5) < 1e-5)

    // A segment passing through the camera (z = 2) and extending behind keeps only the visible
    // side.
    let through = SplatMeasureProjection.frame(points: [SIMD3(0.1, 0, 0), SIMD3(0.1, 0, 5)], camera: camera)
    #expect(through.points.count == 1)
    #expect(through.segments.count == 1)
    let behind = SplatMeasureProjection.frame(points: [SIMD3(0, 0, 3), SIMD3(0.1, 0, 5)], camera: camera)
    #expect(behind.points.isEmpty && behind.segments.isEmpty)
}

@Test
func measureDistanceFormatting() {
    #expect(SplatMeasureProjection.formatDistance(0.0042) == "4.2 mm")
    #expect(SplatMeasureProjection.formatDistance(0.123) == "12.3 cm")
    #expect(SplatMeasureProjection.formatDistance(1.234) == "1.23 m")
}
