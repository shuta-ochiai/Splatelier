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

    /// Same layout as the shader's `MeshUniforms` (stride 96 bytes).
    private struct Uniforms {
        var viewProjection: simd_float4x4
        /// xyz: camera forward in world (light direction for shaded display).
        var viewDirection: SIMD4<Float>
        var shaded: UInt32
        var hasColor: UInt32
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

    /// Clears color / depth and draws the mesh (the first pass of the frame). Depth is standard Z
    /// (near 0 / far 1).
    func encode(
        into commandBuffer: MTLCommandBuffer, color: MTLTexture, depth: MTLTexture,
        clearColor: MTLClearColor, viewProjection: simd_float4x4, viewDirection: SIMD3<Float>, shaded: Bool
    ) {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = clearColor
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
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
            viewDirection: SIMD4(viewDirection, 0),
            shaded: shaded ? 1 : 0,
            hasColor: colors != nil ? 1 : 0)
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
        float4 viewDirection;
        uint shaded;
        uint hasColor;
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
        out.position = u.viewProjection * float4(float3(positions[vid]), 1.0);
        out.color = u.hasColor != 0 ? float3(colors[vid]) : float3(0.75);
        out.normal = float3(normals[vid]);
        return out;
    }

    fragment float4 meshFragment(MeshVertexOut in [[stage_in]],
                                 constant MeshUniforms& u [[buffer(0)]]) {
        float3 c = in.color;
        if (u.shaded != 0) {
            // For inspecting shape: gray lit from the camera direction (two-sided).
            float l = abs(dot(normalize(in.normal), u.viewDirection.xyz));
            c = float3(0.85) * (0.25 + 0.75 * l);
        }
        return float4(c, 1.0);
    }
    """
}

#endif // os(iOS) || os(macOS)
