// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

#if os(macOS)

import AppKit
import CoreGraphics
import Metal
import MetalKit
import MetalSplatter
import Trainer
import MLX
import SplatIO
import simd

/// The render target is non-sRGB `.bgra8Unorm`. With this format MetalSplatter composites raw sRGB
/// values, matching training. An sRGB target composites linearly and looks washed out
/// (Vendor/MetalSplatter/VENDORED.md).
let viewerColorFormat: MTLPixelFormat = .bgra8Unorm

/// Writes status logs to stderr (they survive force-quitting the GUI and don't mix with piped
/// stdout).
private func vlog(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

// MARK: - Live snapshot hand-off

/// Thread-safe box that hands the latest snapshot from the training thread to the viewer on main.
/// Keeps only the latest one.
final class SnapshotBox: @unchecked Sendable {
    private let lock = NSLock()
    private var latest: [SplatPoint]?

    func publish(_ points: [SplatPoint]) {
        lock.lock(); latest = points; lock.unlock()
    }

    func take() -> [SplatPoint]? {
        lock.lock(); defer { lock.unlock() }
        let points = latest
        latest = nil
        return points
    }
}

/// Converts a `GaussianModel` to `SplatPoint`s. Same conventions as `PLYWriter`
/// (SH0 coefficients / logit opacity / log scale / (w, x, y, z) quaternion).
func modelToSplatPoints(_ model: GaussianModel) -> [SplatPoint] {
    let n = model.count
    let means = model.means.asArray(Float.self)
    let sh0Host = model.sh0.asArray(Float.self)
    let logScales = model.logScales.asArray(Float.self)
    let logitOpac = model.logitOpacities.asArray(Float.self)
    let quats = model.quats.asArray(Float.self)

    var points = [SplatPoint]()
    points.reserveCapacity(n)
    for i in 0..<n {
        let mx = means[i * 3], my = means[i * 3 + 1], mz = means[i * 3 + 2]
        // Diverged (non-finite) gaussians are passed neither to the GPU nor to camera framing.
        guard mx.isFinite, my.isFinite, mz.isFinite else { continue }
        let sh0 = SIMD3<Float>(
            sh0Host[i * 3], sh0Host[i * 3 + 1], sh0Host[i * 3 + 2]
        )
        let rot = simd_quatf(
            ix: quats[i * 4 + 1], iy: quats[i * 4 + 2], iz: quats[i * 4 + 3], r: quats[i * 4]
        )
        points.append(SplatPoint(
            position: SIMD3<Float>(means[i * 3], means[i * 3 + 1], means[i * 3 + 2]),
            color: .sphericalHarmonicFloat([sh0]),
            opacity: .logitFloat(logitOpac[i]),
            scale: .exponent(SIMD3<Float>(logScales[i * 3], logScales[i * 3 + 1], logScales[i * 3 + 2])),
            rotation: rot
        ))
    }
    return points
}

/// Carries Metal objects and framing across `captureSplatPNG`'s async Task.
/// main reads them only after the Task signals (serialized by a semaphore), hence @unchecked.
private final class CaptureContext: @unchecked Sendable {
    let device: MTLDevice
    let renderer: SplatRenderer
    var center = SIMD3<Float>(0, 0, 0)
    var distance: Float = 4
    init(device: MTLDevice, renderer: SplatRenderer) {
        self.device = device
        self.renderer = renderer
    }
}

/// Center and camera distance that fit within a 60° vertical FOV. Uses the 2–98th percentile per
/// axis so a few outliers don't inflate the bbox.
func boundingFrame(_ points: [SplatPoint]) -> (center: SIMD3<Float>, distance: Float) {
    var xs = [Float](), ys = [Float](), zs = [Float]()
    xs.reserveCapacity(points.count); ys.reserveCapacity(points.count); zs.reserveCapacity(points.count)
    for p in points {
        let q = p.position
        guard q.x.isFinite, q.y.isFinite, q.z.isFinite else { continue }
        xs.append(q.x); ys.append(q.y); zs.append(q.z)
    }
    guard !xs.isEmpty else { return (SIMD3<Float>(0, 0, 0), 4) }
    xs.sort(); ys.sort(); zs.sort()
    func pct(_ a: [Float], _ q: Float) -> Float {
        a[Swift.min(a.count - 1, Swift.max(0, Int(q * Float(a.count))))]
    }
    let lo = SIMD3<Float>(pct(xs, 0.02), pct(ys, 0.02), pct(zs, 0.02))
    let hi = SIMD3<Float>(pct(xs, 0.98), pct(ys, 0.98), pct(zs, 0.98))
    let center = (lo + hi) * 0.5
    let radius = Swift.max(simd_length(hi - lo) * 0.5, 0.01)
    return (center, radius / tanf((60 * .pi / 180) * 0.5) * 1.2)
}

/// Renders one frame of a splat file to PNG without a window.
func captureSplatPNG(url: URL, outURL: URL, width: Int, height: Int) {
    guard let device = MTLCreateSystemDefaultDevice() else { vlog("capture: no Metal device"); return }
    guard let queue = device.makeCommandQueue() else { vlog("capture: no command queue"); return }
    let renderer: SplatRenderer
    do {
        renderer = try SplatRenderer(
            device: device, colorFormat: viewerColorFormat, depthFormat: .depth32Float,
            sampleCount: 1, maxViewCount: 1, maxSimultaneousRenders: 1,
            clearColor: MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        )
    } catch { vlog("capture: \(error)"); return }

    let ctx = CaptureContext(device: device, renderer: renderer)
    let sem = DispatchSemaphore(value: 0)
    Task {
        do {
            let raw = try await AutodetectSceneReader(url).readAll()
            let pts = raw.filter {
                $0.position.x.isFinite && $0.position.y.isFinite && $0.position.z.isFinite
            }
            vlog("capture: \(pts.count)/\(raw.count) finite splats")
            if !pts.isEmpty {
                let frame = boundingFrame(pts)
                ctx.center = frame.center
                ctx.distance = frame.distance
                vlog("capture: center=(\(frame.center.x),\(frame.center.y),\(frame.center.z)) distance=\(frame.distance)")
                let chunk = try SplatChunk(device: ctx.device, from: pts)
                _ = await ctx.renderer.addChunk(chunk)
            }
        } catch { vlog("capture: load failed: \(error)") }
        sem.signal()
    }
    sem.wait()
    let center = ctx.center
    let distance = ctx.distance

    let colorDesc = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: viewerColorFormat, width: width, height: height, mipmapped: false
    )
    colorDesc.usage = [.renderTarget, .shaderRead]
    colorDesc.storageMode = .shared
    let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .depth32Float, width: width, height: height, mipmapped: false
    )
    depthDesc.usage = [.renderTarget]
    depthDesc.storageMode = .private
    guard let colorTex = device.makeTexture(descriptor: colorDesc),
          let depthTex = device.makeTexture(descriptor: depthDesc)
    else { vlog("capture: texture alloc failed"); return }

    let proj = perspectiveRH(fovyRadians: 60 * .pi / 180, aspect: Float(width) / Float(height), near: 0.05, far: 1000)
    let viewMatrix = translation(0, 0, -distance) * translation(-center.x, -center.y, -center.z)
    let vp = SplatRenderer.ViewportDescriptor(
        viewport: MTLViewport(originX: 0, originY: 0, width: Double(width), height: Double(height), znear: 0, zfar: 1),
        projectionMatrix: proj, viewMatrix: viewMatrix, screenSize: SIMD2(width, height)
    )

    // Render a few frames until the async depth sort settles.
    for _ in 0..<4 {
        guard let cb = queue.makeCommandBuffer() else { break }
        do {
            _ = try renderer.render(
                viewports: [vp], colorTexture: colorTex, colorStoreAction: .store,
                depthTexture: depthTex, rasterizationRateMap: nil, renderTargetArrayLength: 0,
                to: cb
            )
        } catch { vlog("capture: render \(error)") }
        cb.commit()
        cb.waitUntilCompleted()
    }

    let rowBytes = width * 4
    var bgra = [UInt8](repeating: 0, count: rowBytes * height)
    colorTex.getBytes(&bgra, bytesPerRow: rowBytes, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
    var rgba = [UInt8](repeating: 0, count: rowBytes * height)
    var nonBlack = 0
    for i in stride(from: 0, to: bgra.count, by: 4) {
        rgba[i] = bgra[i + 2]; rgba[i + 1] = bgra[i + 1]; rgba[i + 2] = bgra[i]; rgba[i + 3] = 255
        if bgra[i] | bgra[i + 1] | bgra[i + 2] != 0 { nonBlack += 1 }
    }
    vlog("capture: \(nonBlack) non-black pixels of \(width * height)")

    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    guard let ctx = CGContext(
        data: &rgba, width: width, height: height, bitsPerComponent: 8, bytesPerRow: rowBytes,
        space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ), let cg = ctx.makeImage() else { vlog("capture: CGImage failed"); return }
    let rep = NSBitmapImageRep(cgImage: cg)
    guard let png = rep.representation(using: .png, properties: [:]) else { vlog("capture: PNG encode failed"); return }
    do { try png.write(to: outURL); vlog("capture: wrote \(outURL.path)") }
    catch { vlog("capture: write failed: \(error)") }
}

/// Saves a `[H, W, 3]` float image (clamped to [0, 1]) as PNG.
func saveImagePNG(_ image: MLXArray, to url: URL) {
    let h = image.shape[0], w = image.shape[1]
    let host = image.reshaped([h * w * 3]).asArray(Float.self)
    let rowBytes = w * 4
    var rgba = [UInt8](repeating: 255, count: rowBytes * h)
    for i in 0..<(w * h) {
        for c in 0..<3 {
            let v = Swift.max(0, Swift.min(1, host[i * 3 + c]))
            rgba[i * 4 + c] = UInt8((v * 255).rounded())
        }
    }
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    guard let ctx = CGContext(
        data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: rowBytes,
        space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ), let cg = ctx.makeImage() else { vlog("saveImagePNG: CGImage failed"); return }
    guard let png = NSBitmapImageRep(cgImage: cg).representation(using: .png, properties: [:]) else {
        vlog("saveImagePNG: PNG encode failed"); return
    }
    do { try png.write(to: url); vlog("wrote \(url.path)") }
    catch { vlog("saveImagePNG: write failed: \(error)") }
}

// MARK: - Matrix helpers (right-handed, column-major)

func perspectiveRH(fovyRadians fovy: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
    let ys = 1 / tanf(fovy * 0.5)
    let xs = ys / aspect
    let zs = far / (near - far)
    return simd_float4x4(columns: (
        SIMD4<Float>(xs, 0, 0, 0),
        SIMD4<Float>(0, ys, 0, 0),
        SIMD4<Float>(0, 0, zs, -1),
        SIMD4<Float>(0, 0, zs * near, 0)
    ))
}

func translation(_ x: Float, _ y: Float, _ z: Float) -> simd_float4x4 {
    simd_float4x4(columns: (
        SIMD4<Float>(1, 0, 0, 0),
        SIMD4<Float>(0, 1, 0, 0),
        SIMD4<Float>(0, 0, 1, 0),
        SIMD4<Float>(x, y, z, 1)
    ))
}

func rotation(radians: Float, axis: SIMD3<Float>) -> simd_float4x4 {
    let a = normalize(axis)
    let ct = cosf(radians), st = sinf(radians), ci = 1 - ct
    let x = a.x, y = a.y, z = a.z
    return simd_float4x4(columns: (
        SIMD4<Float>(ct + x * x * ci, y * x * ci + z * st, z * x * ci - y * st, 0),
        SIMD4<Float>(x * y * ci - z * st, ct + y * y * ci, z * y * ci + x * st, 0),
        SIMD4<Float>(x * z * ci + y * st, y * z * ci - x * st, ct + z * z * ci, 0),
        SIMD4<Float>(0, 0, 0, 1)
    ))
}

// MARK: - MTKView that picks up orbit / zoom input

final class OrbitMTKView: MTKView {
    var onDrag: ((_ dx: CGFloat, _ dy: CGFloat) -> Void)?
    var onScroll: ((_ dy: CGFloat) -> Void)?

    override var acceptsFirstResponder: Bool { true }
    override func mouseDragged(with event: NSEvent) { onDrag?(event.deltaX, event.deltaY) }
    override func scrollWheel(with event: NSEvent) { onScroll?(event.deltaY) }
}

// MARK: - Viewer

/// Drives `MetalSplatter.SplatRenderer` on an `MTKView` with an orbit camera (macOS only).
/// Loading uses `SplatIO` auto-detection (.ply / .splat / .spz).
@MainActor
final class SplatViewer: NSObject, MTKViewDelegate {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let renderer: SplatRenderer
    private let maxRenders = 3
    private let inFlight: DispatchSemaphore

    /// Published by the training thread and taken by `draw`.
    let snapshotBox = SnapshotBox()
    private var hasFramed = false
    private var isApplying = false
    private var currentChunkID: ChunkID?
    private var skippedFirstFrame = false

    private var azimuth: Float = 0
    private var elevation: Float = 0
    private var distance: Float = 4
    private var center = SIMD3<Float>(0, 0, 0)
    private var drawableSize = CGSize(width: 1, height: 1)

    init(view: MTKView) throws {
        guard let device = view.device ?? MTLCreateSystemDefaultDevice() else {
            throw ViewerError.noMetalDevice
        }
        self.device = device
        view.device = device
        view.colorPixelFormat = viewerColorFormat
        view.depthStencilPixelFormat = .depth32Float
        view.sampleCount = 1
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        guard let queue = device.makeCommandQueue() else { throw ViewerError.noCommandQueue }
        self.commandQueue = queue
        self.renderer = try SplatRenderer(
            device: device,
            colorFormat: viewerColorFormat,
            depthFormat: .depth32Float,
            sampleCount: 1,
            maxViewCount: 1,
            maxSimultaneousRenders: maxRenders,
            clearColor: MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        )
        // VIEWER_TRANSPARENCY=sorted|stochastic|movement (default movement, same as the app's
        // viewer).
        switch ProcessInfo.processInfo.environment["VIEWER_TRANSPARENCY"] {
        case "sorted": renderer.transparencyMode = .sorted
        case "stochastic": renderer.transparencyMode = .stochastic
        default: renderer.transparencyMode = .movement
        }
        self.inFlight = DispatchSemaphore(value: maxRenders)
        super.init()
    }

    enum ViewerError: Error { case noMetalDevice, noCommandQueue }

    func attachInput(to view: OrbitMTKView) {
        view.onDrag = { [weak self] dx, dy in
            guard let self else { return }
            self.azimuth += Float(dx) * 0.01
            self.elevation += Float(dy) * 0.01
            // Clamp elevation to avoid flipping near the poles.
            self.elevation = max(-1.5, min(1.5, self.elevation))
        }
        view.onScroll = { [weak self] dy in
            guard let self else { return }
            self.distance *= Float(1 + dy * 0.02)
            self.distance = max(0.05, self.distance)
        }
    }

    /// Loads splats from `url` asynchronously and fits the camera to the bbox.
    func loadSplats(_ url: URL) {
        Task {
            do {
                let reader = try AutodetectSceneReader(url)
                let raw = try await reader.readAll()
                let points = raw.filter {
                    $0.position.x.isFinite && $0.position.y.isFinite && $0.position.z.isFinite
                }
                guard !points.isEmpty else {
                    vlog("viewer: no usable splats in \(url.lastPathComponent) (\(raw.count) read)")
                    return
                }
                if points.count != raw.count {
                    vlog("viewer: skipped \(raw.count - points.count) non-finite splats")
                }
                frameCamera(to: points)
                let chunk = try SplatChunk(device: device, from: points)
                await renderer.addChunk(chunk)
                vlog("viewer: loaded \(points.count) splats from \(url.lastPathComponent)")
            } catch {
                vlog("viewer: failed to load \(url.path): \(error)")
            }
        }
    }

    /// Replaces the rendered splats with a snapshot. Camera fitting happens only the first time. To
    /// avoid empty frames, the new chunk is added before the old one is removed.
    func applySnapshot(_ points: [SplatPoint]) async {
        guard !points.isEmpty else { return }
        if !hasFramed {
            frameCamera(to: points)
            hasFramed = true
            vlog("viewer: first snapshot rendered (\(points.count) splats)")
        }
        guard let chunk = try? SplatChunk(device: device, from: points) else { return }
        let newID = await renderer.addChunk(chunk)
        if let old = currentChunkID { await renderer.removeChunk(old) }
        currentChunkID = newID
    }

    private func frameCamera(to points: [SplatPoint]) {
        (center, distance) = boundingFrame(points)
    }

    private func viewportDescriptor() -> SplatRenderer.ViewportDescriptor {
        let aspect = Float(drawableSize.width / Swift.max(1, drawableSize.height))
        let proj = perspectiveRH(fovyRadians: 60 * .pi / 180, aspect: aspect, near: 0.05, far: 1000)
        // Centroid to origin → azimuth rotation → elevation rotation → pull back by `distance`.
        let viewMatrix =
            translation(0, 0, -distance)
            * rotation(radians: elevation, axis: SIMD3<Float>(1, 0, 0))
            * rotation(radians: azimuth, axis: SIMD3<Float>(0, 1, 0))
            * translation(-center.x, -center.y, -center.z)
        let viewport = MTLViewport(
            originX: 0, originY: 0,
            width: Double(drawableSize.width), height: Double(drawableSize.height),
            znear: 0, zfar: 1
        )
        return SplatRenderer.ViewportDescriptor(
            viewport: viewport,
            projectionMatrix: proj,
            viewMatrix: viewMatrix,
            screenSize: SIMD2(Int(drawableSize.width), Int(drawableSize.height))
        )
    }

    // MARK: MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        drawableSize = size
    }

    func draw(in view: MTKView) {
        // drawableSizeWillChange isn't guaranteed to arrive before the first draw, and a stale 1×1
        // size gives a black screen, so read the drawable's size directly.
        drawableSize = view.drawableSize

        // The first drawable can be composited uninitialized (magenta), so drop the first frame.
        guard skippedFirstFrame else { skippedFirstFrame = true; return }

        // Apply the latest snapshot if any (at most one at a time).
        if !isApplying, let points = snapshotBox.take() {
            isApplying = true
            Task { await applySnapshot(points); isApplying = false }
        }
        guard let drawable = view.currentDrawable else { return }
        // Clear to black before splats are uploaded (skipping this flashes magenta).
        guard renderer.isReadyToRender else {
            if let rpd = view.currentRenderPassDescriptor,
               let cb = commandQueue.makeCommandBuffer() {
                cb.makeRenderCommandEncoder(descriptor: rpd)?.endEncoding()
                cb.commit()
                cb.waitUntilScheduled()
                drawable.present()
            }
            return
        }
        _ = inFlight.wait(timeout: .distantFuture)
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            inFlight.signal()
            return
        }
        // Called from a background queue, so make it @Sendable to avoid @MainActor inference
        // (the executor check would trap).
        let sem = inFlight
        commandBuffer.addCompletedHandler { @Sendable _ in sem.signal() }
        do {
            _ = try renderer.render(
                viewports: [viewportDescriptor()],
                colorTexture: drawable.texture,
                colorStoreAction: .store,
                depthTexture: view.depthStencilTexture,
                rasterizationRateMap: nil,
                renderTargetArrayLength: 0,
                to: commandBuffer
            )
        } catch {
            vlog("viewer: render error: \(error)")
        }
        // presentsWithTransaction, so present the drawable directly after scheduling.
        commandBuffer.commit()
        commandBuffer.waitUntilScheduled()
        drawable.present()
    }
}

// MARK: - App launch

private final class ViewerAppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

/// Opens the window and blocks the main thread in the AppKit event loop until it closes.
@MainActor
enum SplatViewerApp {
    private static let delegate = ViewerAppDelegate()

    static func run(url: URL, title: String) {
        present(title: title) { viewer in viewer.loadSplats(url) }
    }

    /// Runs `train` on another thread and renders the published snapshots live.
    static func runLive(title: String, train: @escaping @Sendable (SnapshotBox) -> Void) {
        present(title: title) { viewer in
            let box = viewer.snapshotBox
            Thread.detachNewThread { train(box) }
        }
    }

    /// Sets up the window and viewer and enters the run loop. The window / layer background is
    /// black so the magenta of an uninitialized drawable never shows.
    private static func present(title: String, configure: (SplatViewer) -> Void) {
        let app = NSApplication.shared
        app.setActivationPolicy(.regular)
        app.delegate = delegate

        let frame = NSRect(x: 0, y: 0, width: 1024, height: 768)
        let window = NSWindow(
            contentRect: frame,
            styleMask: [.titled, .closable, .resizable, .miniaturizable],
            backing: .buffered, defer: false
        )
        window.title = title
        window.backgroundColor = .black
        window.center()

        let mtkView = OrbitMTKView(frame: frame, device: MTLCreateSystemDefaultDevice())
        let viewer: SplatViewer
        do {
            viewer = try SplatViewer(view: mtkView)
        } catch {
            vlog("viewer: \(error)")
            return
        }
        mtkView.delegate = viewer
        mtkView.layer?.isOpaque = true
        (mtkView.layer as? CAMetalLayer)?.backgroundColor = CGColor(gray: 0, alpha: 1)
        // Present in sync with the CoreAnimation transaction so uninitialized drawables are never
        // composited.
        mtkView.presentsWithTransaction = true
        viewer.attachInput(to: mtkView)
        configure(viewer)

        window.contentView = mtkView
        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(mtkView)
        app.activate(ignoringOtherApps: true)
        app.run()
    }
}

#endif // os(macOS)
