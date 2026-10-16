// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Marching Cubes (Lorensen & Cline 1987) turning the TSDF zero isosurface into a triangle mesh. The
// 256-case triangle table is generated at startup: on each cube face, draw segments cutting off the
// runs of inside (TSDF < 0) corners, then triangulate the closed loops formed by joining them at
// edge crossings. On ambiguous faces, where the 4 corners alternate inside/outside diagonally,
// inside corners are always cut off individually, so the cut lines match the neighboring cube
// sharing the face and the mesh has no holes. Segments run "from the edge entering the inside run →
// the edge leaving it, in counter-clockwise order around the face seen from outside", so triangles
// are counter-clockwise seen from outside (TSDF > 0 = camera side).

import Foundation

public enum MarchingCubes {

    // Corner c has in-cube coordinates (c & 1, (c >> 1) & 1, (c >> 2) & 1). Edge e = axis * 4 + k
    // runs along axis; its start is the k-th (ascending) corner whose axis bit is 0.

    /// Edge (start corner, end corner).
    static let edgeCorners: [(Int, Int)] = (0..<12).map { e in
        let axis = e / 4, k = e % 4
        // The start is the corner given by inserting a 0 into k's bits at the axis position
        let base = ((k >> axis) << (axis + 1)) | (k & ((1 << axis) - 1))
        return (base, base | (1 << axis))
    }

    /// Index of the edge joining 2 corners that differ in exactly 1 bit.
    static func edgeIndex(_ a: Int, _ b: Int) -> Int {
        let diff = a ^ b
        precondition(diff != 0 && diff & (diff - 1) == 0, "corners \(a), \(b) are not adjacent")
        let axis = diff.trailingZeroBitCount
        let base = min(a, b)
        return axis * 4 + (((base >> (axis + 1)) << axis) | (base & ((1 << axis) - 1)))
    }

    /// case (bit c = corner c is inside) → 3 edge indices per triangle.
    static let triangleTable: [[UInt8]] = (0..<256).map { triangulate(case: $0) }

    static func triangulate(case mask: Int) -> [UInt8] {
        // Cut line: edge entering the inside run → edge leaving it. An edge with a crossing belongs
        // to exactly 2 faces, entering on one and leaving on the other, so following next gives a
        // closed loop.
        var next = [Int](repeating: -1, count: 12)
        for axis in 0..<3 {
            let b = (axis + 1) % 3, c = (axis + 2) % 3
            for side in 0..<2 {
                // (0,0)→(1,0)→(1,1)→(0,1) in the (b, c) plane is counter-clockwise about +axis.
                // Faces whose outward normal is -axis (side 0) go the other way.
                var cycle = [(0, 0), (1, 0), (1, 1), (0, 1)].map { (u, v) in
                    (side << axis) | (u << b) | (v << c)
                }
                if side == 0 { cycle.reverse() }
                let inside = cycle.map { (mask >> $0) & 1 == 1 }
                for i in 0..<4 where !inside[i] && inside[(i + 1) % 4] {
                    var j = (i + 1) % 4
                    while inside[(j + 1) % 4] { j = (j + 1) % 4 }
                    next[edgeIndex(cycle[i], cycle[(i + 1) % 4])] = edgeIndex(cycle[j], cycle[(j + 1) % 4])
                }
            }
        }
        var triangles: [UInt8] = []
        var visited = [Bool](repeating: false, count: 12)
        for start in 0..<12 where next[start] >= 0 && !visited[start] {
            var loop: [Int] = []
            var e = start
            while !visited[e] {
                visited[e] = true
                loop.append(e)
                e = next[e]
            }
            precondition(e == start && loop.count >= 3, "open cut loop in case \(mask)")
            guard let tris = triangulateLoop(loop) else {
                preconditionFailure("no face-free triangulation in case \(mask)")
            }
            for (a, b, c) in tris {
                triangles += [UInt8(loop[a]), UInt8(loop[b]), UInt8(loop[c])]
            }
        }
        return triangles
    }

    /// Whether 2 edges lie on the same cube face.
    static func edgesShareFace(_ e1: Int, _ e2: Int) -> Bool {
        let corners = [edgeCorners[e1].0, edgeCorners[e1].1, edgeCorners[e2].0, edgeCorners[e2].1]
        for axis in 0..<3 {
            for side in 0..<2 where corners.allSatisfy({ ($0 >> axis) & 1 == side }) {
                return true
            }
        }
        return false
    }

    /// Triangulates a cut-line loop (sequence of edge indices) so no diagonal lies on a cube face
    /// (returns triples of indices into the loop). A diagonal on a face would coincide with a
    /// triangle of the neighboring cube sharing that face, making an edge shared by 3+ faces
    /// (non-manifold), so it is avoided. Loop edges (cut lines on faces) are fine, since the
    /// neighbor shares them in the opposite direction.
    static func triangulateLoop(_ loop: [Int]) -> [(Int, Int, Int)]? {
        let n = loop.count
        func allowed(_ i: Int, _ j: Int) -> Bool {
            j - i == 1 || j - i == n - 1 || !edgesShareFace(loop[i], loop[j])
        }
        // Split the range i...j (sub-polygon cut off by chord i–j). Same orientation as the loop
        // ((i, k, j) with i < k < j).
        var memo: [Int: [(Int, Int, Int)]?] = [:]
        func solve(_ i: Int, _ j: Int) -> [(Int, Int, Int)]? {
            if j - i < 2 { return [] }
            if let cached = memo[i * 16 + j] { return cached }
            var result: [(Int, Int, Int)]?
            for k in (i + 1)..<j where allowed(i, k) && allowed(k, j) {
                if let left = solve(i, k), let right = solve(k, j) {
                    result = left + [(i, k, j)] + right
                    break
                }
            }
            memo[i * 16 + j] = result
            return result
        }
        return solve(0, n - 1)
    }

    /// Extracts the isosurface only from cubes whose 8 corners were all observed at least
    /// `minWeight` times (prevents false surfaces in unobserved regions). One vertex per grid edge,
    /// shared with adjacent cubes.
    public static func extract(volume: TSDFFusion.Volume, minWeight: Float = 3) -> TriangleMesh {
        extract(volume: volume, minWeight: minWeight, recordEdges: false).mesh
    }

    /// The grid edge a vertex was placed on: world positions of its two corner samples, the inside
    /// (negative) one first.
    struct VertexEdge {
        var inside: SIMD3<Float>
        var outside: SIMD3<Float>
    }

    /// `extract`, also returning each vertex's grid edge (`edges[v]` for vertex v) when
    /// `recordEdges` is set, so callers can refine vertex positions along the edge.
    static func extract(
        volume: TSDFFusion.Volume, minWeight: Float, recordEdges: Bool
    ) -> (mesh: TriangleMesh, edges: [VertexEdge]) {
        let table = triangleTable
        let corners = edgeCorners
        let vs = volume.voxelSize
        // Size of the full voxel grid, for edge keys
        let gx = Int64(volume.blockDims.x) * 8, gy = Int64(volume.blockDims.y) * 8
        var positions: [SIMD3<Float>] = []
        var colors: [SIMD3<Float>] = []
        var indices: [UInt32] = []
        var edges: [VertexEdge] = []
        var vertexOfEdge: [Int64: UInt32] = [:]
        var slots = [Int](repeating: 0, count: 8)
        var values = [Float](repeating: 0, count: 8)

        func unpack(_ c: UInt32) -> SIMD3<Float> {
            SIMD3(Float(c & 0xFF), Float((c >> 8) & 0xFF), Float((c >> 16) & 0xFF)) / 255
        }

        volume.tsdf.withUnsafeBufferPointer { tsdf in
        volume.weight.withUnsafeBufferPointer { weight in
            for (bi, bc) in volume.blockCoords.enumerated() {
                for l in 0..<TSDFFusion.Volume.voxelsPerBlock {
                    let s0 = bi * TSDFFusion.Volume.voxelsPerBlock + l
                    if weight[s0] < minWeight { continue }
                    let x = bc.x * 8 + Int32(l & 7)
                    let y = bc.y * 8 + Int32((l >> 3) & 7)
                    let z = bc.z * 8 + Int32(l >> 6)
                    var mask = 0
                    var valid = true
                    for c in 0..<8 {
                        let s: Int
                        if c == 0 {
                            s = s0
                        } else {
                            guard let sc = volume.slot(
                                x + Int32(c & 1), y + Int32((c >> 1) & 1), z + Int32((c >> 2) & 1)),
                                  weight[sc] >= minWeight else { valid = false; break }
                            s = sc
                        }
                        slots[c] = s
                        values[c] = tsdf[s]
                        if values[c] < 0 { mask |= 1 << c }
                    }
                    if !valid || mask == 0 || mask == 255 { continue }
                    for e in table[mask] {
                        let (a, b) = corners[Int(e)]
                        let axis = Int64(e / 4)
                        let ax = Int64(x) + Int64(a & 1)
                        let ay = Int64(y) + Int64((a >> 1) & 1)
                        let az = Int64(z) + Int64((a >> 2) & 1)
                        let key = ((az * gy + ay) * gx + ax) * 3 + axis
                        if let v = vertexOfEdge[key] {
                            indices.append(v)
                            continue
                        }
                        // a and b differ in sign, so the denominator is never 0
                        let t = values[a] / (values[a] - values[b])
                        let pa = SIMD3<Float>(Float(ax) + 0.5, Float(ay) + 0.5, Float(az) + 0.5)
                        var pb = pa
                        pb[Int(axis)] += 1
                        let p = volume.origin + (pa + t * (pb - pa)) * vs
                        let ca = unpack(volume.color[slots[a]]), cb = unpack(volume.color[slots[b]])
                        let v = UInt32(positions.count)
                        positions.append(p)
                        colors.append(ca + t * (cb - ca))
                        if recordEdges {
                            let wa = volume.origin + pa * vs, wb = volume.origin + pb * vs
                            edges.append(values[a] < 0
                                ? VertexEdge(inside: wa, outside: wb) : VertexEdge(inside: wb, outside: wa))
                        }
                        vertexOfEdge[key] = v
                        indices.append(v)
                    }
                }
            }
        }
        }
        let normals = TriangleMesh.vertexNormals(positions: positions, indices: indices)
        let mesh = TriangleMesh(positions: positions, normals: normals, colors: colors, indices: indices)
        return (mesh, edges)
    }
}
