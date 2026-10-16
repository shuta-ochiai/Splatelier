// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Viewer OBJ loading (`MeshOBJWriter` output and general OBJs).

import Foundation
import Testing
@testable import Trainer
@testable import SplatViewerKit

private func temporaryOBJ(_ text: String? = nil) throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("objloader-\(UUID().uuidString).obj")
    if let text { try text.write(to: url, atomically: true, encoding: .utf8) }
    return url
}

/// A mesh written by `MeshOBJWriter` reads back unchanged within the format's precision (positions
/// 1e-5 / colors 1e-4).
@Test
func objLoaderReadsMeshOBJWriterOutput() throws {
    let mesh = TriangleMesh(
        positions: [SIMD3(0, 1.5, -2.25), SIMD3(1, 0, 0), SIMD3(0.123456, -3e-6, 3), SIMD3(-1, -1, 1)],
        normals: [SIMD3(0, 0, 1), SIMD3(0, 0, 1), SIMD3(0, 0, 1), SIMD3(0, 0, 1)],
        colors: [SIMD3(1, 0.5, 0), SIMD3(0, 0, 0), SIMD3(0.25, 0.25, 0.25), SIMD3(0.1, 0.2, 0.3)],
        indices: [0, 1, 2, 2, 1, 3])
    let url = try temporaryOBJ()
    defer { try? FileManager.default.removeItem(at: url) }
    try MeshOBJWriter.write(mesh, to: url, comment: "test")

    let loaded = try OBJMeshLoader.load(url)
    #expect(loaded.vertexCount == 4)
    #expect(loaded.indices == [0, 1, 2, 2, 1, 3])
    for v in 0..<4 {
        for k in 0..<3 {
            #expect(abs(loaded.positions[v * 3 + k] - mesh.positions[v][k]) < 1e-5)
            #expect(abs(loaded.colors[v * 3 + k] - mesh.colors[v][k]) < 1e-4)
        }
    }
    #expect(loaded.normals.count == 12)
}

/// Reads OBJs without color, with `f v/vt/vn`, quads, negative (from-the-end) vertex indices,
/// exponent notation and comments.
@Test
func objLoaderReadsGenericOBJ() throws {
    let url = try temporaryOBJ("""
    # generic
    v 0 0 0
    v 1.0e0 0 0
    v 1 1 0
    v 0 1E+0 0
    vt 0 0
    vn 0 0 1
    f 1/1/1 2/1/1 3/1/1 4/1/1
    f -4//1 -2//1 -1//1
    """)
    defer { try? FileManager.default.removeItem(at: url) }
    let loaded = try OBJMeshLoader.load(url)
    #expect(loaded.vertexCount == 4)
    #expect(loaded.colors.isEmpty)
    // a quad is fanned into 2 triangles; negative indices count from the end
    #expect(loaded.indices == [0, 1, 2, 0, 2, 3, 0, 2, 3])
    #expect(abs(loaded.center.x - 0.5) < 1e-6 && abs(loaded.center.y - 0.5) < 1e-6)
    // normals are recomputed from faces (a z+ face)
    #expect(abs(loaded.normals[2] - 1) < 1e-6)
}

@Test
func objLoaderRejectsFileWithoutTriangles() throws {
    let url = try temporaryOBJ("v 0 0 0\nv 1 0 0\n")
    defer { try? FileManager.default.removeItem(at: url) }
    #expect(throws: OBJMeshLoader.LoadError.self) { try OBJMeshLoader.load(url) }
}
