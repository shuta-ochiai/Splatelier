// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Loads CAD models (STEP) for display. The stepmesh static library (native/stepmesh, built on
// cadmpeg) converts the STEP file to a USDZ package, and ModelIO reads its display meshes. Parts that
// an assembly places several times arrive as USD instances, which are expanded here.

import Foundation
import ModelIO
import StepMesh
import simd

enum STEPMeshLoader {

    struct LoadError: Error, CustomStringConvertible {
        let description: String
    }

    static let fileExtensions: Set<String> = ["step", "stp"]

    /// Grey for faces whose STEP file gives no colour, when other faces have one.
    private static let defaultColor = SIMD3<Float>(0.7, 0.7, 0.7)

    /// Positions are in metres and Y-up (CAD files are Z-up; see `yUp`).
    static func load(_ url: URL) throws -> LoadedMesh {
        let usdz = FileManager.default.temporaryDirectory
            .appendingPathComponent("step-\(UUID().uuidString).usdz")
        defer { try? FileManager.default.removeItem(at: usdz) }
        try convert(url, to: usdz)
        return try loadUSD(usdz)
    }

    static func convert(_ step: URL, to usdz: URL) throws {
        let message = step.path.withCString { input in
            usdz.path.withCString { output in stepmesh_convert_to_usdz(input, output) }
        }
        if let message {
            defer { stepmesh_free_string(message) }
            throw LoadError(description: String(cString: message))
        }
    }

    /// Reads the meshes of a USD file written by cadmpeg (Z-up, metres).
    static func loadUSD(_ url: URL) throws -> LoadedMesh {
        let asset = MDLAsset(url: url)
        var builder = Builder()
        var roots: [MDLObject] = []
        var prototypes = Set<String>()
        for i in 0..<asset.count {
            let object = asset.object(at: i)
            roots.append(object)
            collectInstances(object, into: &prototypes)
        }
        // ModelIO lists the prototypes of instanced parts as extra top-level objects; they are
        // drawn only through the instances that place them.
        for root in roots where !prototypes.contains(root.path) {
            builder.add(root, transform: yUp)
        }
        guard !builder.indices.isEmpty else {
            throw LoadError(description: "No triangles in the STEP file")
        }
        return builder.mesh()
    }

    /// CAD Z-up to the viewer's Y-up: (x, y, z) → (x, z, −y).
    private static let yUp = simd_float4x4(
        SIMD4(1, 0, 0, 0), SIMD4(0, 0, -1, 0), SIMD4(0, 1, 0, 0), SIMD4(0, 0, 0, 1))

    private static func collectInstances(_ object: MDLObject, into paths: inout Set<String>) {
        if let instance = object.instance { paths.insert(instance.path) }
        for child in object.children.objects { collectInstances(child, into: &paths) }
    }

    private struct Builder {
        var positions: [Float] = []
        var normals: [Float] = []
        var colors: [SIMD3<Float>?] = []
        var indices: [UInt32] = []
        var missingNormals = false

        mutating func add(_ object: MDLObject, transform parent: simd_float4x4) {
            let world = parent * (object.transform?.matrix ?? matrix_identity_float4x4)
            if let mesh = object as? MDLMesh { add(mesh, transform: world) }
            if let instance = object.instance { add(instance, transform: world) }
            for child in object.children.objects { add(child, transform: world) }
        }

        /// Appends each triangle submesh with its own copy of the vertices it uses, so a vertex
        /// shared by differently coloured submeshes keeps both colours.
        mutating func add(_ mesh: MDLMesh, transform: simd_float4x4) {
            guard let position = mesh.vertexAttributeData(forAttributeNamed: MDLVertexAttributePosition, as: .float3)
            else { return }
            let normal = mesh.vertexAttributeData(forAttributeNamed: MDLVertexAttributeNormal, as: .float3)
            if normal == nil { missingNormals = true }
            let linear = simd_float3x3(
                SIMD3(transform.columns.0.x, transform.columns.0.y, transform.columns.0.z),
                SIMD3(transform.columns.1.x, transform.columns.1.y, transform.columns.1.z),
                SIMD3(transform.columns.2.x, transform.columns.2.y, transform.columns.2.z))
            let normalMatrix = linear.inverse.transpose
            func read(_ data: MDLVertexAttributeData, _ v: Int) -> SIMD3<Float> {
                let p = data.dataStart.advanced(by: v * data.stride).assumingMemoryBound(to: Float.self)
                return SIMD3(p[0], p[1], p[2])
            }

            for case let submesh as MDLSubmesh in mesh.submeshes ?? [] where submesh.geometryType == .triangles {
                let color = STEPMeshLoader.baseColor(submesh.material)
                let buffer = submesh.indexBuffer(asIndexType: .uInt32)
                let map = buffer.map()
                let source = map.bytes.assumingMemoryBound(to: UInt32.self)
                var remap = [Int: UInt32]()
                for k in 0..<(submesh.indexCount - submesh.indexCount % 3) {
                    let v = Int(source[k])
                    guard v < mesh.vertexCount else { continue }
                    if let index = remap[v] {
                        indices.append(index)
                        continue
                    }
                    let index = UInt32(positions.count / 3)
                    remap[v] = index
                    let p = transform * SIMD4(read(position, v), 1)
                    positions += [p.x, p.y, p.z]
                    let n = normal.map { simd_normalize(normalMatrix * read($0, v)) } ?? .zero
                    normals += [n.x, n.y, n.z]
                    colors.append(color)
                    indices.append(index)
                }
            }
        }

        func mesh() -> LoadedMesh {
            let hasColor = colors.contains { $0 != nil }
            let (center, radius) = LoadedMesh.bounds(positions)
            return LoadedMesh(
                positions: positions,
                normals: missingNormals ? OBJMeshLoader.vertexNormals(positions: positions, indices: indices) : normals,
                colors: hasColor ? colors.flatMap { c -> [Float] in
                    let c = c ?? STEPMeshLoader.defaultColor
                    return [c.x, c.y, c.z]
                } : [],
                indices: indices,
                center: center,
                radius: radius)
        }
    }

    /// The display colour of a submesh's material (cadmpeg writes the STEP colour as the preview
    /// surface's diffuse colour).
    fileprivate static func baseColor(_ material: MDLMaterial?) -> SIMD3<Float>? {
        guard let property = material?.property(with: .baseColor) else { return nil }
        switch property.type {
        case .float3: return property.float3Value
        case .float4: let c = property.float4Value; return SIMD3(c.x, c.y, c.z)
        case .color:
            guard let c = property.color?.components, c.count >= 3 else { return nil }
            return SIMD3(Float(c[0]), Float(c[1]), Float(c[2]))
        default: return nil
        }
    }
}
