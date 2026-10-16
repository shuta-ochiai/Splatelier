// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Marching Cubes (CPU) and OBJ export. The TSDF is built directly from an analytic field (no GPU
// needed).

import Foundation
import Testing
@testable import Trainer

/// Fully allocated Volume of (blocks·8)³ voxels. `field` maps a voxel center's world coords and
/// voxel coords to (tsdf, weight).
private func denseVolume(
    blocks: Int, voxelSize: Float,
    field: (SIMD3<Float>, SIMD3<Int32>) -> (tsdf: Float, weight: Float)
) -> TSDFFusion.Volume {
    var blockIndex = [Int32](repeating: -1, count: blocks * blocks * blocks)
    var coords: [SIMD3<Int32>] = []
    for z in 0..<blocks {
        for y in 0..<blocks {
            for x in 0..<blocks {
                blockIndex[(z * blocks + y) * blocks + x] = Int32(coords.count)
                coords.append(SIMD3(Int32(x), Int32(y), Int32(z)))
            }
        }
    }
    let n = coords.count * 512
    var tsdf = [Float](repeating: 1, count: n)
    var weight = [Float](repeating: 0, count: n)
    for (bi, bc) in coords.enumerated() {
        for l in 0..<512 {
            let v = SIMD3<Int32>(bc.x * 8 + Int32(l & 7), bc.y * 8 + Int32((l >> 3) & 7), bc.z * 8 + Int32(l >> 6))
            let p = SIMD3<Float>(Float(v.x) + 0.5, Float(v.y) + 0.5, Float(v.z) + 0.5) * voxelSize
            let f = field(p, v)
            tsdf[bi * 512 + l] = f.tsdf
            weight[bi * 512 + l] = f.weight
        }
    }
    return TSDFFusion.Volume(
        voxelSize: voxelSize, truncation: 4 * voxelSize, origin: .zero,
        blockDims: SIMD3(repeating: Int32(blocks)), blockIndex: blockIndex, blockCoords: coords,
        tsdf: tsdf, weight: weight,
        color: [UInt32](repeating: 0x0032_64C8, count: n))   // (200, 100, 50)
}

private func length(_ v: SIMD3<Float>) -> Float { (v * v).sum().squareRoot() }

/// TSDF of a sphere: signed distance normalized by the truncation distance (outside = positive).
private func sphereTSDF(_ p: SIMD3<Float>, center: SIMD3<Float>, radius: Float, trunc: Float) -> Float {
    Swift.min(1, Swift.max(-1, (length(p - center) - radius) / trunc))
}

/// In a closed, oriented mesh, each directed edge (a→b) appears exactly once, and its reverse (b→a)
/// exactly once too.
private func isClosedAndConsistentlyOriented(_ mesh: TriangleMesh) -> Bool {
    var directed: [UInt64: Int] = [:]
    for t in stride(from: 0, to: mesh.indices.count, by: 3) {
        for k in 0..<3 {
            let a = UInt64(mesh.indices[t + k]), b = UInt64(mesh.indices[t + (k + 1) % 3])
            directed[(a << 32) | b, default: 0] += 1
        }
    }
    for (key, count) in directed {
        let reverse = ((key & 0xFFFF_FFFF) << 32) | (key >> 32)
        if count != 1 || directed[reverse] != 1 { return false }
    }
    return true
}

@Test
func marchingCubesTableHasNoSurfaceForUniformCases() {
    #expect(MarchingCubes.triangleTable.count == 256)
    #expect(MarchingCubes.triangleTable[0].isEmpty)
    #expect(MarchingCubes.triangleTable[255].isEmpty)
    // 1 corner inside = 1 triangle cutting off that corner
    for c in 0..<8 { #expect(MarchingCubes.triangleTable[1 << c].count == 3) }
    // 4 corners at z = 0 inside = a horizontal quad = 2 triangles
    #expect(MarchingCubes.triangleTable[0b0000_1111].count == 6)
}

/// The mesh from a sphere TSDF is closed, normals face outward (positive TSDF), and vertices lie on
/// the sphere.
@Test
func marchingCubesSphereIsClosedOutwardAndOnSurface() {
    let vs: Float = 1.0 / 32
    let center = SIMD3<Float>(0.5, 0.5, 0.5), radius: Float = 0.3
    let volume = denseVolume(blocks: 4, voxelSize: vs) { p, _ in
        (sphereTSDF(p, center: center, radius: radius, trunc: 4 * vs), 10)
    }
    let mesh = MarchingCubes.extract(volume: volume, minWeight: 3)
    #expect(mesh.triangleCount > 500)
    #expect(isClosedAndConsistentlyOriented(mesh))
    var maxRadialError: Float = 0
    var inward = 0
    for i in 0..<mesh.vertexCount {
        let d = mesh.positions[i] - center
        maxRadialError = Swift.max(maxRadialError, abs(length(d) - radius))
        if (mesh.normals[i] * d).sum() <= 0 { inward += 1 }
    }
    #expect(maxRadialError < 0.25 * vs)
    #expect(inward == 0)
    // Color stays the voxel color (200, 100, 50)
    #expect(mesh.colors.allSatisfy { abs($0.x - 200.0 / 255) < 1e-4 && abs($0.z - 50.0 / 255) < 1e-4 })
}

/// Even on a random field (with many ambiguous faces), cut lines of adjacent cubes agree, so no
/// holes open.
@Test(arguments: [UInt32(7), 11, 23, 101])
func marchingCubesRandomFieldIsWatertight(seed: UInt32) {
    var lcg = TestLCG(seed: seed)
    let n: Int32 = 16
    let volume = denseVolume(blocks: 2, voxelSize: 0.1) { _, v in
        let boundary = v.x == 0 || v.y == 0 || v.z == 0 || v.x == n - 1 || v.y == n - 1 || v.z == n - 1
        let value = lcg.next() * 2 - 1
        return (boundary ? 1 : value, 5)
    }
    let mesh = MarchingCubes.extract(volume: volume, minWeight: 1)
    #expect(mesh.triangleCount > 1000)
    #expect(isClosedAndConsistentlyOriented(mesh))
}

/// No surface is emitted from voxels with fewer observations than minWeight.
@Test
func marchingCubesSkipsUnderObservedVoxels() {
    let vs: Float = 1.0 / 32
    let center = SIMD3<Float>(0.5, 0.5, 0.5)
    let volume = denseVolume(blocks: 4, voxelSize: vs) { p, v in
        // Only the first half in x is sufficiently observed
        (sphereTSDF(p, center: center, radius: 0.3, trunc: 4 * vs), v.x < 16 ? 5 : 2)
    }
    let mesh = MarchingCubes.extract(volume: volume, minWeight: 3)
    #expect(mesh.triangleCount > 0)
    #expect(mesh.positions.allSatisfy { $0.x <= 16 * vs })
}

/// Only small connected components (small floating spheres) are removed.
@Test
func removingSmallComponentsDropsFloaters() {
    let vs: Float = 1.0 / 32
    let big = SIMD3<Float>(0.35, 0.5, 0.5), small = SIMD3<Float>(0.8, 0.5, 0.5)
    let volume = denseVolume(blocks: 4, voxelSize: vs) { p, _ in
        let t = Swift.min(sphereTSDF(p, center: big, radius: 0.25, trunc: 4 * vs),
                          sphereTSDF(p, center: small, radius: 0.06, trunc: 4 * vs))
        return (t, 10)
    }
    let raw = MarchingCubes.extract(volume: volume, minWeight: 3)
    #expect(raw.positions.contains { length($0 - small) < 0.1 })
    let cleaned = raw.removingSmallComponents(minTriangles: raw.triangleCount / 2)
    #expect(cleaned.triangleCount > 0 && cleaned.triangleCount < raw.triangleCount)
    #expect(cleaned.positions.allSatisfy { abs(length($0 - big) - 0.25) < 0.25 * vs })
    #expect(isClosedAndConsistentlyOriented(cleaned))
}

@Test
func meshOBJWriterFormatsVerticesNormalsAndFaces() throws {
    let mesh = TriangleMesh(
        positions: [SIMD3(0, 1.5, -2.25), SIMD3(1, 0, 0), SIMD3(0.123456, -0.000001, 3)],
        normals: [SIMD3(0, 0, 1), SIMD3(0, 0, 1), SIMD3(0, 0, -1)],
        colors: [SIMD3(1, 0.5, 0), SIMD3(0, 0, 0), SIMD3(0.25, 0.25, 0.25)],
        indices: [0, 1, 2])
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("mesh-obj-\(UUID().uuidString).obj")
    defer { try? FileManager.default.removeItem(at: url) }
    try MeshOBJWriter.write(mesh, to: url)
    let lines = try String(contentsOf: url, encoding: .utf8).split(separator: "\n").map(String.init)
    let body = lines.filter { !$0.hasPrefix("#") }
    #expect(body == [
        "v 0.00000 1.50000 -2.25000 1.0000 0.5000 0.0000",
        "v 1.00000 0.00000 0.00000 0.0000 0.0000 0.0000",
        "v 0.12346 0.00000 3.00000 0.2500 0.2500 0.2500",
        "vn 0.0000 0.0000 1.0000",
        "vn 0.0000 0.0000 1.0000",
        "vn 0.0000 0.0000 -1.0000",
        "f 1//1 2//2 3//3",
    ])
}
