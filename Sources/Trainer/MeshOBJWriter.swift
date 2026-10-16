// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation

/// Writes a `TriangleMesh` as Wavefront OBJ (text). Vertex colors use the widely read extension `v
/// x y z r g b` (read by MeshLab / Blender), normals `vn`, faces `f a//a b//b c//c` (1-based).
/// Coordinates are ARKit world (m, y up). Millions of vertices are written, so numbers are
/// formatted as fixed-point directly into bytes.
public enum MeshOBJWriter {

    public static func write(_ mesh: TriangleMesh, to url: URL, comment: String? = nil) throws {
        var buf: [UInt8] = []
        buf.reserveCapacity(mesh.vertexCount * 72 + mesh.triangleCount * 40 + 256)
        func text(_ s: String) { buf.append(contentsOf: s.utf8) }

        text("# splatelier mesh: \(mesh.vertexCount) vertices, \(mesh.triangleCount) triangles\n")
        if let comment {
            for line in comment.split(separator: "\n") { text("# \(line)\n") }
        }
        for i in 0..<mesh.vertexCount {
            let p = mesh.positions[i], c = mesh.colors[i]
            text("v ")
            appendFixed(p.x, 5, &buf); buf.append(0x20)
            appendFixed(p.y, 5, &buf); buf.append(0x20)
            appendFixed(p.z, 5, &buf); buf.append(0x20)
            appendFixed(c.x, 4, &buf); buf.append(0x20)
            appendFixed(c.y, 4, &buf); buf.append(0x20)
            appendFixed(c.z, 4, &buf); buf.append(0x0A)
        }
        for n in mesh.normals {
            text("vn ")
            appendFixed(n.x, 4, &buf); buf.append(0x20)
            appendFixed(n.y, 4, &buf); buf.append(0x20)
            appendFixed(n.z, 4, &buf); buf.append(0x0A)
        }
        for t in stride(from: 0, to: mesh.indices.count, by: 3) {
            text("f")
            for k in 0..<3 {
                let v = Int(mesh.indices[t + k]) + 1
                buf.append(0x20)
                appendInt(v, &buf); buf.append(0x2F); buf.append(0x2F); appendInt(v, &buf)
            }
            buf.append(0x0A)
        }
        try Data(buf).write(to: url, options: .atomic)
    }

    /// Fixed-point (`decimals` fractional digits, rounded). Non-finite values become 0.
    static func appendFixed(_ value: Float, _ decimals: Int, _ buf: inout [UInt8]) {
        guard value.isFinite else { buf.append(0x30); return }
        var scale: Int64 = 1
        for _ in 0..<decimals { scale *= 10 }
        let scaled = (Double(abs(value)) * Double(scale)).rounded()
        guard scaled < 9e17 else { buf.append(0x30); return }
        let r = Int64(scaled)
        if value < 0 && r != 0 { buf.append(0x2D) }
        appendInt(Int(r / scale), &buf)
        buf.append(0x2E)
        // Reserve the fractional digits and fill from the lowest digit (no temporary array)
        var frac = r % scale
        let start = buf.count
        buf.append(contentsOf: repeatElement(0x30, count: decimals))
        for i in stride(from: decimals - 1, through: 0, by: -1) {
            buf[start + i] = 0x30 + UInt8(frac % 10)
            frac /= 10
        }
    }

    static func appendInt(_ value: Int, _ buf: inout [UInt8]) {
        if value < 0 { buf.append(0x2D) }
        var v = value.magnitude
        var digits = 1
        var p: UInt = 10
        while v >= p && digits < 19 { digits += 1; p *= 10 }
        let start = buf.count
        buf.append(contentsOf: repeatElement(0x30, count: digits))
        for i in stride(from: digits - 1, through: 0, by: -1) {
            buf[start + i] = 0x30 + UInt8(v % 10)
            v /= 10
        }
    }
}
