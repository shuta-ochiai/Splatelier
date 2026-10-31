// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Draws the capture-time display (CaptureSurfelMap) directly with Metal in SceneKit's render pass.
// Each surfel becomes a square on the plane perpendicular to its normal (center ± `extentSigmas`
// σ), and the fragment applies Gaussian falloff by distance from the center (a flat Gaussian
// attached to the surface, so it keeps its on-surface shape even when viewed obliquely).
//
// Rebuilding a SceneKit geometry (4 vertices per surfel, 18MB at 128k surfels) froze the display on
// every update, because at the moment of replacement SceneKit repacks the vertices on the render
// thread and sends them to the GPU. Here surfels are packed 32 bytes each
// into a GPU buffer on a separate queue, and the render thread only swaps the buffer and
// instance-draws the squares. Sorting is done once, back to front from the camera position at
// packing time.
//
// Shader compilation and pipeline creation are not awaited on the render thread either. The first
// compile is slow, and waiting on the render thread stalled the capture screen until the camera
// feed appeared on the first capture start. When the capture
// screen opens, `prewarm` starts compiling on another thread, and no splats are drawn until it is
// ready.

#if canImport(SceneKit) && canImport(Metal)

import Foundation
import Metal
import SceneKit
import simd

/// Mutable state is guarded by `lock` (buffer swap, in-progress marks, pipelines), hence @unchecked
/// Sendable.
public final class CaptureSurfelRenderer: @unchecked Sendable {
    /// Half width of the square (in σ). The training renderer also cuts off at 3σ.
    public static let extentSigmas: Float = 3

    /// Least cosine between a square's plane normal and the direction to the camera: a square seen
    /// more edge-on is tilted toward the camera, so it shows at least this share of its width. Flat
    /// squares seen at a grazing angle, or with the normal a little off (the normals come from a
    /// low-resolution depth map), otherwise thin into slivers that read as scattered leaves.
    public static let minFacingCosine: Float = 0.9

    /// Surfel sent to the GPU (32 bytes, same as `Splat` on the Metal side). Color is 8-bit sRGB,
    /// a = opacity.
    private struct GPUSplat {
        var px, py, pz, sigma: Float
        var nx, ny, nz: Float
        var color: UInt32
    }

    private struct Uniforms {
        var viewProjection: simd_float4x4
        var extent: Float
        /// 1: the render target has no sRGB conversion, so output colors in sRGB as is.
        var outputSRGB: UInt32
        var minFacing: Float
        var pad: Float = 0
        var cameraPosition: SIMD4<Float>
    }

    private struct PipelineKey: Hashable {
        let color: MTLPixelFormat
        let depth: MTLPixelFormat
        let stencil: MTLPixelFormat
        let samples: Int
    }

    /// Shader compile result (reused by the renderer created for each capture).
    private static let libraries = LibraryCache()

    private let device: MTLDevice
    private let depthState: MTLDepthStencilState
    private let queue = DispatchQueue(label: "live-surfel-buffer", qos: .userInitiated)
    private let lock = NSLock()
    private var current: (buffer: MTLBuffer, count: Int)?
    /// The old buffer right after a swap. A frame being drawn may still read it, so keep it for a
    /// while.
    private var retired: [MTLBuffer] = []
    private var updating = false
    private var pipelines: [PipelineKey: MTLRenderPipelineState] = [:]
    private var pipelinesInProgress: Set<PipelineKey> = []
    private var drawn = 0

    public init?(device: MTLDevice) {
        let ds = MTLDepthStencilDescriptor()
        ds.depthCompareFunction = .always
        ds.isDepthWriteEnabled = false
        guard let depthState = device.makeDepthStencilState(descriptor: ds) else { return nil }
        self.device = device
        self.depthState = depthState
        _ = Self.libraries.library(for: device)
    }

    /// Starts compiling the shader on another thread ahead of time (called when the capture screen
    /// opens; does not wait).
    public static func prewarm(device: MTLDevice? = MTLCreateSystemDefaultDevice()) {
        if let device { _ = libraries.library(for: device) }
    }

    /// Whether the previous `update` has not finished yet.
    public var isUpdating: Bool { lock.withLock { updating } }

    /// On a separate queue, sorts surfels back to front from `eye`, packs them into a buffer, and
    /// swaps the drawn buffer when done.
    public func update(_ surfels: [CaptureSurfelMap.Surfel], eye: SIMD3<Float>) {
        lock.withLock { updating = true }
        queue.async { [self] in
            let packed = pack(surfels, eye: eye)
            lock.withLock {
                replace(with: packed)
                updating = false
            }
        }
    }

    /// Runs `update` on the caller's thread (for checking on Mac).
    public func updateSynchronously(_ surfels: [CaptureSurfelMap.Surfel], eye: SIMD3<Float>) {
        let packed = pack(surfels, eye: eye)
        lock.withLock { replace(with: packed) }
    }

    /// Call inside `lock`.
    private func replace(with packed: (buffer: MTLBuffer, count: Int)?) {
        if let old = current?.buffer { retired = Array((retired + [old]).suffix(3)) }
        current = packed
    }

    /// Draws splats into SceneKit's render pass (from `renderer(_:didRenderScene:atTime:)`, on the
    /// render thread).
    public func draw(in renderer: SCNSceneRenderer) {
        let pass = renderer.currentRenderPassDescriptor
        guard let cur = lock.withLock({ current }), cur.count > 0,
              let encoder = renderer.currentRenderCommandEncoder,
              let target = pass.colorAttachments[0].texture,
              let pov = renderer.pointOfView, let camera = pov.camera else { return }
        // Do not draw until the shader is compiled (do not start creating pipelines either)
        guard Self.libraries.library(for: device) != nil else { return }
        let key = PipelineKey(
            color: target.pixelFormat,
            depth: pass.depthAttachment.texture?.pixelFormat ?? .invalid,
            stencil: pass.stencilAttachment.texture?.pixelFormat ?? .invalid,
            samples: target.sampleCount)
        guard let pipeline = pipeline(for: key) else { return }
        var uniforms = Uniforms(
            viewProjection: simd_float4x4(camera.projectionTransform) * pov.simdWorldTransform.inverse,
            extent: Self.extentSigmas,
            outputSRGB: Self.isSRGBEncoded(target.pixelFormat) ? 0 : 1,
            minFacing: Self.minFacingCosine,
            cameraPosition: SIMD4(pov.simdWorldPosition, 1))
        encoder.setRenderPipelineState(pipeline)
        encoder.setDepthStencilState(depthState)
        encoder.setCullMode(.none)
        encoder.setVertexBuffer(cur.buffer, offset: 0, index: 0)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: cur.count)
        lock.withLock { drawn += 1 }
    }

    /// Number of times splats were actually drawn (frames skipped waiting for shader compilation
    /// are not counted).
    public var drawnFrames: Int { lock.withLock { drawn } }

    private func pack(_ surfels: [CaptureSurfelMap.Surfel], eye: SIMD3<Float>) -> (buffer: MTLBuffer, count: Int)? {
        let n = surfels.count
        guard n > 0, let buffer = device.makeBuffer(
            length: n * MemoryLayout<GPUSplat>.stride, options: .storageModeShared) else { return nil }
        let order = Self.backToFrontOrder(surfels, eye: eye)
        let out = buffer.contents().bindMemory(to: GPUSplat.self, capacity: n)
        for (i, idx) in order.enumerated() {
            let s = surfels[idx]
            func byte(_ v: Float) -> UInt32 { UInt32(max(0, min(255, (v * 255).rounded()))) }
            let c = s.color
            out[i] = GPUSplat(
                px: s.position.x, py: s.position.y, pz: s.position.z, sigma: s.splatSigma,
                nx: s.normal.x, ny: s.normal.y, nz: s.normal.z,
                color: byte(c.x) | byte(c.y) << 8 | byte(c.z) << 16 | byte(s.splatOpacity) << 24)
        }
        return (buffer, n)
    }

    /// Indices in far-to-near order from `eye`. Counting sort on distance quantized to 16 bits (a
    /// fraction of a comparison sort's time). Within the same quantum (0.15mm wide in a 10m scene)
    /// the original order is kept.
    static func backToFrontOrder(_ surfels: [CaptureSurfelMap.Surfel], eye: SIMD3<Float>) -> [Int] {
        let n = surfels.count
        let dist = surfels.map { simd_distance($0.position, eye) }
        let scale = 65535 / max(dist.max() ?? 0, 1e-6)
        // Farther gets a smaller key
        let keys = dist.map { 65535 - Int(min(65535, $0 * scale)) }
        var start = [Int](repeating: 0, count: 65537)
        for k in keys { start[k + 1] += 1 }
        for b in 1...65536 { start[b] += start[b - 1] }
        var order = [Int](repeating: 0, count: n)
        for i in 0..<n {
            order[start[keys[i]]] = i
            start[keys[i]] += 1
        }
        return order
    }

    /// The created pipeline. If not yet created (and the shader is compiled), starts creating it on
    /// another thread and returns nil.
    private func pipeline(for key: PipelineKey) -> MTLRenderPipelineState? {
        let start: Bool = lock.withLock {
            if pipelines[key] != nil { return false }
            return pipelinesInProgress.insert(key).inserted
        }
        if start {
            DispatchQueue.global(qos: .userInitiated).async { [self] in
                let p = Self.libraries.library(for: device).flatMap { makePipeline(key, library: $0) }
                lock.withLock {
                    pipelines[key] = p
                    pipelinesInProgress.remove(key)   // before compile or on failure, retry on the next draw
                }
            }
        }
        return lock.withLock { pipelines[key] }
    }

    private func makePipeline(_ key: PipelineKey, library: MTLLibrary) -> MTLRenderPipelineState? {
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = library.makeFunction(name: "surfel_vertex")
        d.fragmentFunction = library.makeFunction(name: "surfel_fragment")
        d.colorAttachments[0].pixelFormat = key.color
        d.depthAttachmentPixelFormat = key.depth
        d.stencilAttachmentPixelFormat = key.stencil
        d.rasterSampleCount = key.samples
        // Premultiplied-alpha source over
        let c = d.colorAttachments[0]!
        c.isBlendingEnabled = true
        c.rgbBlendOperation = .add
        c.alphaBlendOperation = .add
        c.sourceRGBBlendFactor = .one
        c.sourceAlphaBlendFactor = .one
        c.destinationRGBBlendFactor = .oneMinusSourceAlpha
        c.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        return try? device.makeRenderPipelineState(descriptor: d)
    }

    /// Whether the format makes the GPU convert linear → sRGB on write (if not, sRGB values are
    /// written as is). Floating-point formats such as rgba16Float stay linear, so they are treated
    /// as converting.
    static func isSRGBEncoded(_ format: MTLPixelFormat) -> Bool {
        switch format {
        case .bgra8Unorm_srgb, .rgba8Unorm_srgb, .bgra10_xr_srgb, .bgr10_xr_srgb,
             .rgba16Float, .rgba32Float:
            return true
        default:
            return false
        }
    }

    /// Translucent white plane placed just in front of the camera (a child of the camera node). As
    /// part of the SceneKit scene, it is drawn before the splats from `draw`.
    public static func makeWashNode(opacity: Float) -> SCNNode {
        let plane = SCNPlane(width: 10, height: 10)
        let material = SCNMaterial()
        material.lightingModel = .constant
        material.diffuse.contents = CGColor(red: 1, green: 1, blue: 1, alpha: 1)
        material.transparency = CGFloat(opacity)
        material.isDoubleSided = true
        material.readsFromDepthBuffer = false
        material.writesToDepthBuffer = false
        plane.firstMaterial = material
        let node = SCNNode(geometry: plane)
        node.simdPosition = SIMD3<Float>(0, 0, -0.1)
        node.renderingOrder = -10
        return node
    }

    fileprivate static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct Splat {
        packed_float3 position;
        float sigma;
        packed_float3 normal;
        uint color;
    };

    struct Uniforms {
        float4x4 viewProjection;
        float extent;
        uint outputSRGB;
        float minFacing;
        float pad;
        float4 cameraPosition;
    };

    struct VOut {
        float4 position [[position]];
        float2 uv;
        float4 color;
    };

    static float3 srgb_to_linear(float3 c) {
        return select(pow((c + 0.055) / 1.055, 2.4), c / 12.92, c <= 0.04045);
    }

    vertex VOut surfel_vertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                              const device Splat* splats [[buffer(0)]],
                              constant Uniforms& u [[buffer(1)]]) {
        Splat s = splats[iid];
        float3 n = float3(s.normal);
        // Tilt a square seen more edge-on than minFacing toward the camera (keeping the side it
        // leans to), so it does not thin into a sliver.
        float3 v = normalize(u.cameraPosition.xyz - float3(s.position));
        float facing = dot(n, v);
        if (facing < 0.0) { n = -n; facing = -facing; }
        if (facing < u.minFacing) {
            float3 side = n - facing * v;
            float len = length(side);
            if (len > 1e-6) {
                n = u.minFacing * v + sqrt(1.0 - u.minFacing * u.minFacing) * (side / len);
            }
        }
        float3 helper = abs(n.x) < 0.9 ? float3(1, 0, 0) : float3(0, 1, 0);
        float3 t1 = normalize(cross(n, helper));
        float3 t2 = cross(n, t1);
        float2 corner = float2((vid & 1) ? 1.0 : -1.0, (vid & 2) ? 1.0 : -1.0);
        float3 p = float3(s.position) + u.extent * s.sigma * (corner.x * t1 + corner.y * t2);
        VOut o;
        o.position = u.viewProjection * float4(p, 1.0);
        // Depth is not used (no test, no write). z is placed at half of w, clipping only what is
        // behind the camera.
        o.position.z = 0.5 * o.position.w;
        o.uv = corner;
        float4 c = unpack_unorm4x8_to_float(s.color);
        o.color = float4(u.outputSRGB != 0 ? c.rgb : srgb_to_linear(c.rgb), c.a);
        return o;
    }

    fragment float4 surfel_fragment(VOut in [[stage_in]], constant Uniforms& u [[buffer(1)]]) {
        float g = exp(-0.5 * u.extent * u.extent * dot(in.uv, in.uv));
        float a = in.color.a * g;
        return float4(in.color.rgb * a, a);
    }
    """
}

/// Shader compile results per device (registryID). Compilation runs on another thread; callers do
/// not wait.
private final class LibraryCache: @unchecked Sendable {
    private struct DeviceBox: @unchecked Sendable { let device: MTLDevice }
    private let lock = NSLock()
    private var libraries: [UInt64: MTLLibrary] = [:]
    private var compiling: Set<UInt64> = []

    /// Returns the compiled result if ready. Otherwise starts compiling on another thread and
    /// returns nil.
    func library(for device: MTLDevice) -> MTLLibrary? {
        let id = device.registryID
        let start: Bool = lock.withLock {
            if libraries[id] != nil { return false }
            return compiling.insert(id).inserted
        }
        if start {
            let box = DeviceBox(device: device)
            DispatchQueue.global(qos: .userInitiated).async { [self] in
                let library = try? box.device.makeLibrary(source: CaptureSurfelRenderer.shaderSource, options: nil)
                lock.withLock {
                    libraries[id] = library
                    compiling.remove(id)   // on failure, retry on the next call
                }
            }
        }
        return lock.withLock { libraries[id] }
    }
}

#endif // canImport(SceneKit) && canImport(Metal)
