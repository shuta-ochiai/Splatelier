// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// TSDF fusion (GPU kernel `tsdf_integrate`) → Marching Cubes, end to end.

import Foundation
import MLX
import Testing
@testable import Trainer

/// Fusing depth maps of the world plane z = 1 seen by 5 cameras at z = 0 (facing +z) puts the
/// surface at z = 1, with normals facing the cameras (-z) and colors taken from the images.
@Test
func tsdfFusionPlaneEndToEnd() {
    let w = 64, h = 48
    let intr: Intrinsics = (fx: 60, fy: 60, cx: 32, cy: 24)
    let centers: [SIMD3<Float>] = [
        SIMD3(0, 0, 0), SIMD3(0.1, 0, 0), SIMD3(-0.1, 0, 0), SIMD3(0, 0.1, 0), SIMD3(0, -0.1, 0),
    ]
    let views = centers.map { c in
        // No rotation: P_c = P_w - C (t = -C). Depth to the plane z = 1 is 1 at every pixel.
        CostVolumeMVS.DepthView(
            width: w, height: h,
            depth: [Float](repeating: 1, count: w * h),
            rgb: [UInt8]((0..<(w * h)).flatMap { _ in [UInt8(200), 100, 50] }),
            viewmat: [1, 0, 0, -c.x,
                      0, 1, 0, -c.y,
                      0, 0, 1, -c.z,
                      0, 0, 0, 1],
            intrinsics: intr)
    }
    let vs: Float = 0.01
    // Also exercise multiple dispatches with a small dispatch unit.
    let params = TSDFFusion.Parameters(voxelSize: vs, truncationVoxels: 4, voxelsPerDispatch: 100_000)
    guard let volume = TSDFFusion.integrate(views: views, parameters: params, log: { _ in }) else {
        Issue.record("integrate returned nil")
        return
    }
    #expect(volume.voxelCount > 100_000)
    let mesh = MarchingCubes.extract(volume: volume, minWeight: 3)
    #expect(mesh.triangleCount > 1000)
    #expect(mesh.positions.allSatisfy { abs($0.z - 1) < 0.1 * vs })
    #expect(mesh.normals.allSatisfy { $0.z < -0.99 })
    #expect(mesh.colors.allSatisfy {
        abs($0.x - 200.0 / 255) < 2.0 / 255 && abs($0.y - 100.0 / 255) < 2.0 / 255
            && abs($0.z - 50.0 / 255) < 2.0 / 255
    })
    // No surface outside the field of view (x ∈ [-0.53, 0.53] at z = 1)
    #expect(mesh.positions.allSatisfy { abs($0.x) < 0.7 && abs($0.y) < 0.6 })
}

/// nil if there are no valid pixels.
@Test
func tsdfFusionWithoutValidDepthReturnsNil() {
    let view = CostVolumeMVS.DepthView(
        width: 4, height: 4, depth: [Float](repeating: -1, count: 16),
        rgb: [UInt8](repeating: 0, count: 48),
        viewmat: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        intrinsics: (fx: 4, fy: 4, cx: 2, cy: 2))
    #expect(TSDFFusion.integrate(views: [view], log: { _ in }) == nil)
}

/// Five views of the plane z = 1. The primary misses |x|, |y| < 0.15 (its hole); the fill sees the
/// plane at z = 1.02 everywhere, plus a floater square at z = 0.6 around (0.3, 0.2).
private func hybridScene() -> (primary: [CostVolumeMVS.DepthView], fill: [CostVolumeMVS.DepthView]) {
    let w = 64, h = 48
    let intr: Intrinsics = (fx: 60, fy: 60, cx: 32, cy: 24)
    let centers: [SIMD3<Float>] = [
        SIMD3(0, 0, 0), SIMD3(0.1, 0, 0), SIMD3(-0.1, 0, 0), SIMD3(0, 0.1, 0), SIMD3(0, -0.1, 0),
    ]
    func viewmat(_ c: SIMD3<Float>) -> [Float] { [1, 0, 0, -c.x, 0, 1, 0, -c.y, 0, 0, 1, -c.z, 0, 0, 0, 1] }
    // World points on z = 1 with |x|, |y| < 0.15 are the primary's hole.
    let primary = centers.map { c in
        var depth = [Float](repeating: 1, count: w * h)
        for y in 0..<h {
            for x in 0..<w {
                let wx = (Float(x) + 0.5 - intr.cx) / intr.fx + c.x
                let wy = (Float(y) + 0.5 - intr.cy) / intr.fy + c.y
                if abs(wx) < 0.15 && abs(wy) < 0.15 { depth[y * w + x] = -1 }
            }
        }
        return CostVolumeMVS.DepthView(
            width: w, height: h, depth: depth, rgb: [UInt8](repeating: 200, count: w * h * 3),
            viewmat: viewmat(c), intrinsics: intr)
    }
    let fill = centers.map { c in
        var depth = [Float](repeating: 1.02, count: w * h)
        for y in 0..<h {
            for x in 0..<w {
                // Floater: a small square at z = 0.6 around world (0.3, 0.2).
                let wx = (Float(x) + 0.5 - intr.cx) / intr.fx * 0.6 + c.x
                let wy = (Float(y) + 0.5 - intr.cy) / intr.fy * 0.6 + c.y
                if abs(wx - 0.3) < 0.05 && abs(wy - 0.2) < 0.05 { depth[y * w + x] = 0.6 }
            }
        }
        return CostVolumeMVS.DepthView(
            width: w, height: h, depth: depth, rgb: [UInt8](repeating: 100, count: w * h * 3),
            viewmat: viewmat(c), intrinsics: intr)
    }
    return (primary, fill)
}

/// Hybrid fusion: the primary (MVS) planes z = 1 with a hole in the middle; the fill (Gaussian) sees
/// the plane 2 cm farther, without the hole, plus a floater at z = 0.6. The surface follows the
/// primary where it observed, the fill only inside the hole, and the floater (in front of primary
/// surfaces, i.e. in space the primary saw as empty) is dropped.
@Test
func tsdfHybridFillsOnlyPrimaryHoles() {
    let (primary, fill) = hybridScene()
    let vs: Float = 0.01
    guard let volume = TSDFFusion.integrateHybrid(
        primary: primary, fill: fill, primaryTrustObservations: 3,
        parameters: TSDFFusion.Parameters(voxelSize: vs, truncationVoxels: 4), log: { _ in }) else {
        Issue.record("integrateHybrid returned nil")
        return
    }
    let mesh = MarchingCubes.extract(volume: volume, minWeight: 3)
    #expect(mesh.triangleCount > 1000)
    let inHole = mesh.positions.filter { abs($0.x) < 0.1 && abs($0.y) < 0.1 }
    let outside = mesh.positions.filter { abs($0.x) > 0.25 || abs($0.y) > 0.25 }
    #expect(inHole.count > 100)
    #expect(inHole.allSatisfy { abs($0.z - 1.02) < 0.3 * vs }, "hole should take the fill depth")
    #expect(outside.allSatisfy { abs($0.z - 1) < 0.3 * vs }, "observed region should keep the primary depth")
    #expect(!mesh.positions.contains { $0.z < 0.8 }, "floater should be removed")
}

@Test("hybrid TSDF keeps the primary's voxel size when the fill would exceed the voxel cap")
func tsdfHybridFillDoesNotCoarsenPrimary() {
    let (primary, fill) = hybridScene()
    // A cap the primary alone already exceeds at 5 mm, so its voxels grow.
    var parameters = TSDFFusion.Parameters(voxelSize: 0.005, truncationVoxels: 4, maxVoxels: 400_000)
    parameters.fillVoxelFraction = 0.1
    guard let primaryOnly = TSDFFusion.integrate(views: primary, parameters: parameters, log: { _ in }),
          let hybrid = TSDFFusion.integrateHybrid(
              primary: primary, fill: fill, parameters: parameters, log: { _ in }) else {
        Issue.record("integration returned nil")
        return
    }
    #expect(primaryOnly.voxelSize > 0.005)
    #expect(abs(hybrid.voxelSize - primaryOnly.voxelSize) < 0.03 * primaryOnly.voxelSize)
    #expect(hybrid.voxelCount > primaryOnly.voxelCount, "the fill should add blocks in the hole")
    #expect(hybrid.voxelCount <= primaryOnly.voxelCount * 105 / 100 + 40_000, "fill stays within its budget")
}
