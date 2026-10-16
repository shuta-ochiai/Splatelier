// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Loads vertex-colored meshes (OBJ). Expects `MeshOBJWriter` output
// (`v x y z r g b` / `f a//a b//b c//c`) but also reads OBJs without color, with `f v/vt/vn`, and
// with polygons.

import Foundation
import simd

/// A mesh loaded for display (flat arrays that can go straight into GPU buffers).
struct LoadedMesh: Sendable {
    var positions: [Float]   // [V*3]
    var normals: [Float]     // [V*3]
    /// [V*3] sRGB [0, 1]. Empty for OBJs without color.
    var colors: [Float]
    var indices: [UInt32]    // [T*3]
    var center: SIMD3<Float>
    var radius: Float

    var vertexCount: Int { positions.count / 3 }
    var triangleCount: Int { indices.count / 3 }
}

enum OBJMeshLoader {

    struct LoadError: Error, CustomStringConvertible {
        let description: String
    }

    /// Reads millions of vertices, so it scans the bytes directly. Normals are recomputed from
    /// faces (`vn` is not read).
    static func load(_ url: URL) throws -> LoadedMesh {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        var positions: [Float] = []
        var colors: [Float] = []
        var indices: [UInt32] = []
        var allColored = true

        data.withUnsafeBytes { raw in
            let p = raw.bindMemory(to: UInt8.self)
            let n = p.count
            var i = 0
            func isBlank(_ c: UInt8) -> Bool { c == 0x20 || c == 0x09 }
            func isLineEnd(_ c: UInt8) -> Bool { c == 0x0A || c == 0x0D }
            func skipBlanks() { while i < n && isBlank(p[i]) { i += 1 } }
            func isDigit(_ c: UInt8) -> Bool { c >= 0x30 && c <= 0x39 }
            func parseNumber() -> Double? {
                skipBlanks()
                var negative = false
                if i < n && (p[i] == 0x2D || p[i] == 0x2B) { negative = p[i] == 0x2D; i += 1 }
                var value: Double = 0
                var digits = 0
                while i < n && isDigit(p[i]) { value = value * 10 + Double(p[i] - 0x30); i += 1; digits += 1 }
                if i < n && p[i] == 0x2E {
                    i += 1
                    var scale = 0.1
                    while i < n && isDigit(p[i]) { value += Double(p[i] - 0x30) * scale; scale *= 0.1; i += 1; digits += 1 }
                }
                guard digits > 0 else { return nil }
                if i < n && (p[i] == 0x65 || p[i] == 0x45) {
                    i += 1
                    var expNegative = false
                    if i < n && (p[i] == 0x2D || p[i] == 0x2B) { expNegative = p[i] == 0x2D; i += 1 }
                    var e = 0
                    while i < n && isDigit(p[i]) { e = e * 10 + Int(p[i] - 0x30); i += 1 }
                    value *= pow(10, Double(expNegative ? -e : e))
                }
                return negative ? -value : value
            }
            var face: [Int] = []
            while i < n {
                let c = p[i]
                let next: UInt8 = i + 1 < n ? p[i + 1] : 0x0A
                if c == 0x76 && isBlank(next) {            // "v x y z [r g b]"
                    i += 1
                    let x = parseNumber() ?? 0, y = parseNumber() ?? 0, z = parseNumber() ?? 0
                    positions += [Float(x), Float(y), Float(z)]
                    if let r = parseNumber(), let g = parseNumber(), let b = parseNumber() {
                        colors += [Float(r), Float(g), Float(b)]
                    } else {
                        allColored = false
                    }
                } else if c == 0x66 && isBlank(next) {     // "f a[/t[/n]] b ... " (polygons are fan-triangulated)
                    i += 1
                    face.removeAll(keepingCapacity: true)
                    while true {
                        skipBlanks()
                        guard i < n, !isLineEnd(p[i]), let v = parseNumber() else { break }
                        face.append(Int(v))
                        while i < n && !isBlank(p[i]) && !isLineEnd(p[i]) { i += 1 }   // skip "/vt/vn"
                    }
                    let vertexCount = positions.count / 3
                    // 1-based; negative is relative to the end
                    let resolved = face.map { $0 < 0 ? vertexCount + $0 : $0 - 1 }
                    // drop faces with out-of-range vertex indices and keep displaying
                    if resolved.count >= 3, resolved.allSatisfy({ $0 >= 0 && $0 < vertexCount }) {
                        for k in 1..<(resolved.count - 1) {
                            indices += [UInt32(resolved[0]), UInt32(resolved[k]), UInt32(resolved[k + 1])]
                        }
                    }
                }
                while i < n && p[i] != 0x0A { i += 1 }
                i += 1
            }
        }

        let vertexCount = positions.count / 3
        guard vertexCount > 0, !indices.isEmpty else {
            throw LoadError(description: "No triangles (\(vertexCount) vertices)")
        }
        if !allColored || colors.count != positions.count {
            colors = []
        }

        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for v in 0..<vertexCount {
            let q = SIMD3(positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2])
            lo = pointwiseMin(lo, q)
            hi = pointwiseMax(hi, q)
        }
        let diagonal = hi - lo
        return LoadedMesh(
            positions: positions,
            normals: vertexNormals(positions: positions, indices: indices),
            colors: colors,
            indices: indices,
            center: (lo + hi) / 2,
            radius: Swift.max(1e-3, 0.5 * (diagonal * diagonal).sum().squareRoot()))
    }

    /// Area-weighted vertex normals.
    static func vertexNormals(positions: [Float], indices: [UInt32]) -> [Float] {
        var acc = [Float](repeating: 0, count: positions.count)
        func at(_ v: Int) -> SIMD3<Float> { SIMD3(positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2]) }
        for t in stride(from: 0, to: indices.count, by: 3) {
            let a = Int(indices[t]), b = Int(indices[t + 1]), c = Int(indices[t + 2])
            let e1 = at(b) - at(a), e2 = at(c) - at(a)
            let nrm = SIMD3(e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z, e1.x * e2.y - e1.y * e2.x)
            for v in [a, b, c] {
                acc[v * 3] += nrm.x; acc[v * 3 + 1] += nrm.y; acc[v * 3 + 2] += nrm.z
            }
        }
        for v in 0..<(acc.count / 3) {
            let x = acc[v * 3], y = acc[v * 3 + 1], z = acc[v * 3 + 2]
            let len = (x * x + y * y + z * z).squareRoot()
            if len > 0 {
                acc[v * 3] = x / len; acc[v * 3 + 1] = y / len; acc[v * 3 + 2] = z / len
            } else {
                acc[v * 3 + 1] = 1
            }
        }
        return acc
    }
}
