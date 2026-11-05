// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Mesh layer of the unified viewer. Draws the vertex-colored triangle mesh opaque and writes depth;
// the splats are then overlaid depth-tested with `SplatRenderer.composeOverExistingContents` (LOCAL
// PATCH 8 in Vendor/MetalSplatter). As with splats, colors are written as raw sRGB values to a
// non-sRGB target. The shader is small, so it is compiled from source at runtime (no Metal resource
// added to the SwiftPM target).

#if os(iOS) || os(macOS)

import Metal
import simd

@MainActor
final class SplatViewerMeshLayer {
    let triangleCount: Int
    let center: SIMD3<Float>
    let radius: Float
    /// For measurement tap hit tests (vertex / index buffers are read directly from the CPU).
    let rayGeometry: MeshRayGeometry

    private let positions: MTLBuffer
    private let normals: MTLBuffer
    private let colors: MTLBuffer?
    private let indices: MTLBuffer
    private let pipeline: MTLRenderPipelineState
    private let depthState: MTLDepthStencilState

    /// Same layout as the shader's `MeshUniforms` (stride 160 bytes).
    private struct Uniforms {
        var viewProjection: simd_float4x4
        /// Mesh to world (identity except for a CAD model placed by hand). Rigid.
        var model: simd_float4x4
        /// xyz: camera forward in world (light direction for shaded display).
        var viewDirection: SIMD4<Float>
        /// 0: vertex colors, 1: gray lit from the camera (shape), 2: vertex colors lit from the camera.
        var shaded: UInt32
        var hasColor: UInt32
        /// How strongly to tint the mesh with the selection colour (a selected CAD model), 0...1.
        var highlight: Float
    }

    enum Appearance: UInt32 {
        case colors = 0
        case shaded = 1
        case litColors = 2
    }

    enum SetupError: Error {
        case buffer
        case depthState
    }

    init(device: MTLDevice, mesh: LoadedMesh, colorFormat: MTLPixelFormat, depthFormat: MTLPixelFormat) throws {
        func buffer<T>(_ values: [T]) throws -> MTLBuffer {
            guard let b = values.withUnsafeBytes({
                device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            }) else { throw SetupError.buffer }
            return b
        }
        positions = try buffer(mesh.positions)
        normals = try buffer(mesh.normals)
        colors = mesh.colors.isEmpty ? nil : try buffer(mesh.colors)
        indices = try buffer(mesh.indices)
        triangleCount = mesh.triangleCount
        center = mesh.center
        radius = mesh.radius
        rayGeometry = MeshRayGeometry(positions: positions, indices: indices, triangleCount: triangleCount)

        let library = try device.makeLibrary(source: Self.shaderSource, options: nil)
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "MeshPipeline"
        descriptor.vertexFunction = library.makeFunction(name: "meshVertex")
        descriptor.fragmentFunction = library.makeFunction(name: "meshFragment")
        descriptor.colorAttachments[0].pixelFormat = colorFormat
        descriptor.depthAttachmentPixelFormat = depthFormat
        descriptor.rasterSampleCount = 1
        pipeline = try device.makeRenderPipelineState(descriptor: descriptor)

        let depthDescriptor = MTLDepthStencilDescriptor()
        depthDescriptor.depthCompareFunction = .less
        depthDescriptor.isDepthWriteEnabled = true
        guard let state = device.makeDepthStencilState(descriptor: depthDescriptor) else {
            throw SetupError.depthState
        }
        depthState = state
    }

    /// Draws the mesh, clearing color / depth first when `clear` (the first pass of the frame) or
    /// else depth-tested against what is there. Depth is standard Z (near 0 / far 1).
    func encode(
        into commandBuffer: MTLCommandBuffer, color: MTLTexture, depth: MTLTexture, clear: Bool = true,
        clearColor: MTLClearColor, viewProjection: simd_float4x4, model: simd_float4x4 = matrix_identity_float4x4,
        viewDirection: SIMD3<Float>, appearance: Appearance, highlight: Float = 0
    ) {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = clear ? .clear : .load
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = clearColor
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = clear ? .clear : .load
        pass.depthAttachment.storeAction = .store
        pass.depthAttachment.clearDepth = 1
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { return }
        encoder.label = "Mesh"
        encoder.setRenderPipelineState(pipeline)
        encoder.setDepthStencilState(depthState)
        // Draw both faces so surfaces don't disappear when seen from the side that wasn't captured.
        encoder.setCullMode(.none)
        var uniforms = Uniforms(
            viewProjection: viewProjection,
            model: model,
            viewDirection: SIMD4(viewDirection, 0),
            shaded: appearance.rawValue,
            hasColor: colors != nil ? 1 : 0,
            highlight: highlight)
        let length = MemoryLayout<Uniforms>.stride
        encoder.setVertexBuffer(positions, offset: 0, index: 0)
        encoder.setVertexBuffer(normals, offset: 0, index: 1)
        encoder.setVertexBuffer(colors ?? positions, offset: 0, index: 2)
        encoder.setVertexBytes(&uniforms, length: length, index: 3)
        encoder.setFragmentBytes(&uniforms, length: length, index: 0)
        encoder.drawIndexedPrimitives(
            type: .triangle, indexCount: triangleCount * 3, indexType: .uint32,
            indexBuffer: indices, indexBufferOffset: 0)
        encoder.endEncoding()
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct MeshUniforms {
        float4x4 viewProjection;
        float4x4 model;
        float4 viewDirection;
        uint shaded;
        uint hasColor;
        float highlight;
    };

    struct MeshVertexOut {
        float4 position [[position]];
        float3 color;
        float3 normal;
    };

    vertex MeshVertexOut meshVertex(uint vid [[vertex_id]],
                                    const device packed_float3* positions [[buffer(0)]],
                                    const device packed_float3* normals [[buffer(1)]],
                                    const device packed_float3* colors [[buffer(2)]],
                                    constant MeshUniforms& u [[buffer(3)]]) {
        MeshVertexOut out;
        out.position = u.viewProjection * (u.model * float4(float3(positions[vid]), 1.0));
        out.color = u.hasColor != 0 ? float3(colors[vid]) : float3(0.75);
        out.normal = (u.model * float4(float3(normals[vid]), 0.0)).xyz;
        return out;
    }

    fragment float4 meshFragment(MeshVertexOut in [[stage_in]],
                                 constant MeshUniforms& u [[buffer(0)]]) {
        float3 c = in.color;
        float l = abs(dot(normalize(in.normal), u.viewDirection.xyz));
        if (u.shaded == 1) {
            // For inspecting shape: gray lit from the camera direction (two-sided).
            c = float3(0.85) * (0.25 + 0.75 * l);
        } else if (u.shaded == 2) {
            // A CAD model's flat part colors, lit the same way so its faces stay apart.
            c *= 0.45 + 0.55 * l;
        }
        // Selection: a light orange wash over the whole model and a strong one toward its outline (faces
        // seen edge-on), so it reads as selected whatever its own colours, while its middle keeps them.
        float rim = (1.0 - l) * (1.0 - l);
        c = mix(c, float3(1.0, 0.55, 0.1) * (0.55 + 0.45 * l), u.highlight * (0.3 + 0.7 * rim));
        return float4(c, 1.0);
    }
    """
}

#endif // os(iOS) || os(macOS)
