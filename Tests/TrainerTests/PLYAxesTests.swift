// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// A capture's PLY is written turned 180° about Z from its Y-up world to the usual 3DGS axes, and
// read back into that world for training.

import Foundation
import MLX
import Testing
import simd
@testable import Trainer

private func tempDir() throws -> URL {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("ply-axes-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

private func smallModel() -> GaussianModel {
    let n = 3
    let rb = GaussianModel.shRestBands
    let model = GaussianModel(
        means: MLXArray([Float(0.1), 0.2, 0.3, -1.0, 0.5, 2.0, 0.4, -0.6, -0.2], [n, 3]),
        quats: MLXArray([Float(1), 0, 0, 0, 0.92388, 0, 0, 0.38268, 0.5, 0.5, 0.5, 0.5], [n, 4]),
        logScales: MLXArray([Float(-1), -2, -3, -1.5, -1.5, -1.5, -2, -1, -2], [n, 3]),
        logitOpacities: MLXArray([Float(0.5), 1.0, -0.5], [n]),
        sh0: MLXArray([Float(0.9), 0.1, 0.2, 0.0, 0.8, 0.4, 0.3, 0.3, 1.0], [n, 3]),
        shN: MLXArray((0..<(n * rb * 3)).map { Float($0) * 0.05 - 0.4 }, [n, rb, 3]))
    eval(model.means, model.quats, model.logScales, model.logitOpacities, model.sh0, model.shN)
    return model
}

/// Rotation matrix of a (w, x, y, z) quaternion.
private func rotation(_ q: [Float]) -> simd_float3x3 {
    simd_float3x3(simd_quatf(ix: q[1], iy: q[2], iz: q[3], r: q[0]).normalized)
}

/// The file holds the turned splats (x and y negated, the orientation turned, the comment in the
/// header), and reading it back gives the model as trained.
@Test
func yUpModelIsWrittenTurnedAndReadBack() throws {
    let dir = try tempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let url = dir.appendingPathComponent("scene.ply")
    let model = smallModel()
    try PLYWriter.writeGaussianPLY(model: model, to: url, cull: .init(enabled: false), yUpWorld: true)

    let header = try #require(PLYAxes.header(of: url))
    #expect(PLYAxes.isTurned(header: header))

    let file = try SplatTable.readPLY(url: url)
    let means = model.means.asArray(Float.self), quats = model.quats.asArray(Float.self)
    let half = simd_float3x3(simd_quatf(angle: .pi, axis: SIMD3(0, 0, 1)))
    for i in 0..<3 {
        #expect(abs(file.positions[i * 3] + means[i * 3]) < 1e-6)
        #expect(abs(file.positions[i * 3 + 1] + means[i * 3 + 1]) < 1e-6)
        #expect(abs(file.positions[i * 3 + 2] - means[i * 3 + 2]) < 1e-6)
        let turned = rotation(Array(file.quats[(i * 4)..<(i * 4 + 4)]))
        let want = half * rotation(Array(quats[(i * 4)..<(i * 4 + 4)]))
        for c in 0..<3 { #expect(simd_distance(turned[c], want[c]) < 1e-4) }
    }

    let back = try PLYReader.readGaussianPLY(url: url)
    let backMeans = back.means.asArray(Float.self), backQuats = back.quats.asArray(Float.self)
    let backShN = back.shN.asArray(Float.self), shN = model.shN.asArray(Float.self)
    for k in 0..<means.count { #expect(abs(backMeans[k] - means[k]) < 1e-6) }
    for k in 0..<shN.count { #expect(abs(backShN[k] - shN[k]) < 1e-6) }
    for i in 0..<3 {
        let a = rotation(Array(backQuats[(i * 4)..<(i * 4 + 4)]))
        let b = rotation(Array(quats[(i * 4)..<(i * 4 + 4)]))
        for c in 0..<3 { #expect(simd_distance(a[c], b[c]) < 1e-4) }
    }
}

/// The sign each SH coefficient takes is the one that keeps the colour seen from every direction:
/// the turned coefficients seen along the turned direction give the original colour. Checked with
/// the real SH basis of degrees 1 and 2 as 3DGS evaluates it.
@Test
func shSignsKeepTheColourSeenFromEachDirection() {
    func basis(_ d: SIMD3<Float>) -> [Float] {
        let (x, y, z) = (d.x, d.y, d.z)
        return [-y, z, -x, x * y, y * z, 2 * z * z - x * x - y * y, x * z, x * x - y * y]
    }
    let coeffs: [Float] = [0.3, -0.2, 0.5, 0.1, -0.4, 0.25, 0.6, -0.15]
    let turned = coeffs.enumerated().map { PLYAxes.restSign($0.offset) * $0.element }
    for d in [SIMD3<Float>(0.3, -0.5, 0.8), SIMD3(-0.9, 0.1, 0.4), SIMD3(0.2, 0.7, -0.68)] {
        let n = simd_normalize(d)
        let seen = zip(coeffs, basis(n)).map(*).reduce(0, +)
        let seenTurned = zip(turned, basis(SIMD3(-n.x, -n.y, n.z))).map(*).reduce(0, +)
        #expect(abs(seen - seenTurned) < 1e-5)
    }
    #expect((0..<15).map(PLYAxes.restSign) == [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
}

/// Turning twice gives the table back.
@Test
func turningTwiceIsNoTurn() throws {
    let table = SplatTable.fromModel(smallModel(), cull: .init(enabled: false)).table
    let twice = table.turnedAboutZ().turnedAboutZ()
    #expect(twice.positions == table.positions)
    #expect(twice.shRest == table.shRest)
    for k in 0..<table.quats.count { #expect(abs(abs(twice.quats[k]) - abs(table.quats[k])) < 1e-6) }
}

/// A capture PLY from before (no comment, a preview.json beside it) is read as Y-up and turned for
/// export; a turned one, or one from elsewhere, is left as it is.
@Test
func legacyCapturePLYIsTurnedForExport() throws {
    let dir = try tempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let model = smallModel()
    let legacy = dir.appendingPathComponent("legacy.ply")
    try PLYWriter.writeGaussianPLY(model: model, to: legacy, cull: .init(enabled: false))
    try Data("{}".utf8).write(to: dir.appendingPathComponent("legacy.preview.json"))
    let turned = dir.appendingPathComponent("turned.ply")
    try PLYWriter.writeGaussianPLY(model: model, to: turned, cull: .init(enabled: false), yUpWorld: true)
    try Data("{}".utf8).write(to: dir.appendingPathComponent("turned.preview.json"))
    let imported = dir.appendingPathComponent("imported.ply")
    try PLYWriter.writeGaussianPLY(model: model, to: imported, cull: .init(enabled: false))

    #expect(PLYAxes.isLegacyYUp(legacy))
    #expect(!PLYAxes.isLegacyYUp(turned))
    #expect(!PLYAxes.isLegacyYUp(imported))
    let fromLegacy = try SplatTable.readPLYInUsualAxes(url: legacy)
    let fromTurned = try SplatTable.readPLYInUsualAxes(url: turned)
    for k in 0..<fromLegacy.positions.count { #expect(abs(fromLegacy.positions[k] - fromTurned.positions[k]) < 1e-6) }
    #expect(try SplatTable.readPLYInUsualAxes(url: imported).positions == SplatTable.readPLY(url: imported).positions)
}
