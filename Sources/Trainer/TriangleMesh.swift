// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation

/// Triangle mesh with vertex colors. Triangles are counter-clockwise seen from the front, and
/// normals point to the front (positive TSDF = camera side).
public struct TriangleMesh: Sendable {
    public var positions: [SIMD3<Float>]
    public var normals: [SIMD3<Float>]
    /// [0, 1].
    public var colors: [SIMD3<Float>]
    /// Every 3 form one triangle (0-based vertex indices).
    public var indices: [UInt32]

    public var vertexCount: Int { positions.count }
    public var triangleCount: Int { indices.count / 3 }

    public init(positions: [SIMD3<Float>], normals: [SIMD3<Float>], colors: [SIMD3<Float>],
                indices: [UInt32]) {
        precondition(normals.count == positions.count && colors.count == positions.count
                     && indices.count % 3 == 0)
        self.positions = positions
        self.normals = normals
        self.colors = colors
        self.indices = indices
    }

    /// Area-weighted vertex normals (normalized sum of triangle cross products). Vertices in no
    /// triangle get +y.
    public static func vertexNormals(positions: [SIMD3<Float>], indices: [UInt32]) -> [SIMD3<Float>] {
        var acc = [SIMD3<Float>](repeating: .zero, count: positions.count)
        for t in stride(from: 0, to: indices.count, by: 3) {
            let (a, b, c) = (Int(indices[t]), Int(indices[t + 1]), Int(indices[t + 2]))
            let n = cross3(positions[b] - positions[a], positions[c] - positions[a])
            acc[a] += n; acc[b] += n; acc[c] += n
        }
        return acc.map { n in
            let len = (n * n).sum().squareRoot()
            return len > 0 ? n / len : SIMD3<Float>(0, 1, 0)
        }
    }

    /// Removes connected components (floating fragments) with fewer than `minTriangles` triangles.
    public func removingSmallComponents(minTriangles: Int) -> TriangleMesh {
        guard minTriangles > 1, !indices.isEmpty else { return self }
        // union-find (path halving)
        var parent = [Int32](0..<Int32(positions.count))
        func find(_ x: Int32) -> Int32 {
            var x = x
            while parent[Int(x)] != x {
                parent[Int(x)] = parent[Int(parent[Int(x)])]
                x = parent[Int(x)]
            }
            return x
        }
        func union(_ a: Int32, _ b: Int32) {
            let ra = find(a), rb = find(b)
            if ra != rb { parent[Int(max(ra, rb))] = min(ra, rb) }
        }
        for t in stride(from: 0, to: indices.count, by: 3) {
            union(Int32(indices[t]), Int32(indices[t + 1]))
            union(Int32(indices[t]), Int32(indices[t + 2]))
        }
        var triCount = [Int32](repeating: 0, count: positions.count)
        for t in stride(from: 0, to: indices.count, by: 3) { triCount[Int(find(Int32(indices[t])))] += 1 }

        var remap = [Int32](repeating: -1, count: positions.count)
        var out = TriangleMesh(positions: [], normals: [], colors: [], indices: [])
        for t in stride(from: 0, to: indices.count, by: 3)
        where triCount[Int(find(Int32(indices[t])))] >= minTriangles {
            for k in 0..<3 {
                let v = Int(indices[t + k])
                if remap[v] < 0 {
                    remap[v] = Int32(out.positions.count)
                    out.positions.append(positions[v])
                    out.normals.append(normals[v])
                    out.colors.append(colors[v])
                }
                out.indices.append(UInt32(remap[v]))
            }
        }
        return out
    }
}

@inline(__always)
func cross3(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> SIMD3<Float> {
    SIMD3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
}
