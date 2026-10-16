// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// MTKViewDelegate that drives MetalSplatter's SplatRenderer with an orbit camera (port of the CLI
// viewer). As in the MetalSplatter SampleApp, file IO runs off the main actor and presenting uses
// `commandBuffer.present` (the correct path on iOS; the CLI's manual present via
// presentsWithTransaction is macOS-only). Draws only when needed: once the final image
// (`lastRenderWasFinal`) is out it pauses the MTKView, and resumes when the camera, layers or
// drawable size change. No GPU use while at rest. The background at the end of the GS PLY (named by
// `<base>.background.json`) gets its own chunk so it can be hidden; it is left out of framing and of
// the tap ray tests. The mesh (OBJ) is drawn opaque before the splats,
// which are overlaid depth-tested with `composeOverExistingContents`. Taps in measure mode cast a
// ray against the visible layers to find the surface point (`SplatViewerRayCaster`), and
// measurement lines are projected with the same camera as rendering and handed to SwiftUI. Double
// tap uses the same ray test and zooms toward the hit surface as the new look-at point.

#if os(iOS) || os(macOS)

import Metal
import MetalKit
import MetalSplatter
import QuartzCore
import SplatIO
import simd

#if os(iOS)
import UIKit
#endif

/// The render target is non-sRGB `.bgra8Unorm`. With this format MetalSplatter composites raw sRGB
/// values, matching training. An sRGB target composites linearly and looks washed out
/// (Vendor/MetalSplatter/VENDORED.md).
private let viewerColorFormat: MTLPixelFormat = .bgra8Unorm

@MainActor
final class SplatViewerRenderer: NSObject, MTKViewDelegate {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let renderer: SplatRenderer?
    private let model: SplatViewerModel
    private weak var view: MTKView?

    private let maxRenders = 3
    private let inFlight: DispatchSemaphore
    private var drawableSize = CGSize(width: 1, height: 1)
    private var didLoad = false
    /// Whether adding chunks and setting the initial visibility are done. The draw loop isn't woken
    /// until then.
    private var isLoaded = false
    /// Whether anything affecting the image changed since the last draw.
    private var needsRedraw = false
    /// Whether the last draw was the final image for this state
    /// (`SplatRenderer.lastRenderWasFinal`).
    private var lastFrameWasFinal = false
    // Chunks of the splat layers (nil = no layer). `applied*` is the visibility already applied;
    // setChunkEnabled is called only when model.show* changes.
    private var gsChunkID: ChunkID?
    private var backgroundChunkID: ChunkID?
    private var seedChunkID: ChunkID?
    private var cameraChunkID: ChunkID?
    private var appliedGS = true
    private var appliedBackground = true
    private var appliedSeed = false
    private var appliedCamera = false
    private var meshLayer: SplatViewerMeshLayer?
    // Keep the chunks for measurement tap hit tests (only reads GPU buffers, no copy). The
    // background is not a surface, so it is not kept.
    private var gsChunk: SplatChunk?
    private var seedChunk: SplatChunk?
    /// A ray test is in flight. Don't stack another; taps until it finishes are dropped.
    private var isPicking = false

    init?(view: MTKView, model: SplatViewerModel) {
        guard let device = view.device ?? MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.commandQueue = queue
        self.model = model
        self.inFlight = DispatchSemaphore(value: maxRenders)

        view.device = device
        view.colorPixelFormat = viewerColorFormat
        view.depthStencilPixelFormat = .depth32Float
        view.sampleCount = 1
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        // Stay paused until loading finishes (requestRedraw resumes).
        view.isPaused = true
        view.enableSetNeedsDisplay = false

        var made: SplatRenderer?
        do {
            made = try SplatRenderer(
                device: device,
                colorFormat: viewerColorFormat,
                depthFormat: .depth32Float,
                sampleCount: 1,
                maxViewCount: 1,
                maxSimultaneousRenders: maxRenders,
                clearColor: MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
            )
            // stochastic only while moving (no sort, cheap); once stopped and the sort catches up,
            // exact compositing.
            made?.transparencyMode = .movement
        } catch {
            self.renderer = nil
            super.init()
            model.setLoadState(.failed("renderer init failed: \(error.localizedDescription)"))
            return
        }
        self.renderer = made
        self.view = view
        super.init()
        model.onNeedsRedraw = { [weak self] in self?.requestRedraw() }
        model.onMeasureTap = { [weak self] in self?.measureTap(normalized: $0) }
        model.onFocusTap = { [weak self] in self?.focusTap(normalized: $0) }
        #if os(iOS)
        // Redraw once when returning from background (if left paused, the screen can stay stale
        // after resuming).
        NotificationCenter.default.addObserver(
            self, selector: #selector(willEnterForeground),
            name: UIApplication.willEnterForegroundNotification, object: nil)
        #endif
    }

    #if os(iOS)
    @objc private func willEnterForeground() { requestRedraw() }
    #endif

    /// Resumes the draw loop. draw(in:) pauses it again once the final image is out and no changes
    /// arrive.
    func requestRedraw() {
        guard isLoaded, let view else { return }
        needsRedraw = true
        if view.isPaused { view.isPaused = false }
    }

    /// Loads a GS PLY, seed PLY and mesh OBJ (all optional) into one scene. GS / seed point cloud /
    /// camera markers go into separate chunks and the mesh into its own layer, so they can be
    /// toggled.
    func load(_ url: URL?, seedURL: URL?, meshURL: URL? = nil) {
        guard !didLoad, let renderer else { return }
        didLoad = true
        Task { [weak self] in
            do {
                // Sidecars name how many splats at the end of the PLY are camera markers
                // (seedmeta; seed PLY order is scene point cloud → camera markers) and background
                // (background; GS PLY order is foreground → background). Splits them off.
                func read(_ u: URL) async throws -> ReadSplats {
                    let raw = try await Task.detached(priority: .userInitiated) {
                        try await AutodetectSceneReader(u).readAll()
                    }.value
                    func finite(_ ps: ArraySlice<SplatPoint>) -> [SplatPoint] {
                        ps.filter { $0.position.x.isFinite && $0.position.y.isFinite && $0.position.z.isFinite }
                    }
                    let camCount = Self.trailingSplatCount(siblingOf: u, sidecar: "seedmeta", key: .camera)
                    let camSplit = (camCount > 0 && camCount < raw.count) ? (raw.count - camCount) : raw.count
                    let bgCount = Self.trailingSplatCount(siblingOf: u, sidecar: "background", key: .background)
                    let bgSplit = (bgCount > 0 && bgCount < camSplit) ? (camSplit - bgCount) : camSplit
                    return ReadSplats(
                        scene: finite(raw[0..<bgSplit]),
                        background: finite(raw[bgSplit..<camSplit]),
                        camera: finite(raw[camSplit...]))
                }

                var main = ReadSplats()
                if let url {
                    main = try await read(url)
                }
                var seed = ReadSplats()
                if let seedURL {
                    seed = try await read(seedURL)
                }
                var mesh: LoadedMesh?
                var meshError: Error?
                if let meshURL {
                    do {
                        mesh = try await Task.detached(priority: .userInitiated) {
                            try OBJMeshLoader.load(meshURL)
                        }.value
                    } catch {
                        meshError = error
                    }
                }
                guard let self else { return }

                let gsPoints = main.scene
                // Without a foreground there is nothing for it to sit behind.
                let backgroundPoints = gsPoints.isEmpty ? [] : main.background
                let seedPoints = seed.scene
                // Camera markers are normally only on the seed side, but allow both.
                let cameraPoints = main.camera + seed.camera
                if let mesh {
                    do {
                        self.meshLayer = try SplatViewerMeshLayer(
                            device: self.device, mesh: mesh,
                            colorFormat: viewerColorFormat, depthFormat: .depth32Float)
                    } catch {
                        meshError = error
                    }
                }

                guard !gsPoints.isEmpty || !seedPoints.isEmpty || !cameraPoints.isEmpty
                        || self.meshLayer != nil else {
                    if let meshError {
                        self.model.setLoadState(.failed("Couldn't load the mesh: \(meshError)"))
                    } else {
                        self.model.setLoadState(.empty)
                    }
                    return
                }

                // Framing uses whichever exists, in the order GS → seed + cameras → mesh.
                let splatFramePoints = !gsPoints.isEmpty ? gsPoints : (seedPoints + cameraPoints)
                let (center, distance) = if let layer = self.meshLayer, splatFramePoints.isEmpty {
                    (layer.center, framingDistance(radius: layer.radius))
                } else {
                    boundingFrame(splatFramePoints)
                }
                let previewCamera = url.flatMap {
                    SplatViewerPreviewCamera.load(siblingOf: $0, center: center, defaultDistance: distance)
                }
                self.model.frame(center: center, distance: distance, previewCamera: previewCamera)

                if !gsPoints.isEmpty {
                    let chunk = try SplatChunk(device: self.device, from: gsPoints)
                    self.gsChunkID = await renderer.addChunk(chunk)
                    self.gsChunk = chunk
                }
                if !backgroundPoints.isEmpty {
                    self.backgroundChunkID = await renderer.addChunk(
                        try SplatChunk(device: self.device, from: backgroundPoints))
                }
                if !seedPoints.isEmpty {
                    let chunk = try SplatChunk(device: self.device, from: seedPoints)
                    self.seedChunkID = await renderer.addChunk(chunk)
                    self.seedChunk = chunk
                }
                if !cameraPoints.isEmpty {
                    self.cameraChunkID = await renderer.addChunk(try SplatChunk(device: self.device, from: cameraPoints))
                }

                // Initially show GS only. Without GS show seed to avoid an empty view, and if
                // neither exists show the mesh.
                let hasGS = !gsPoints.isEmpty
                let hasSeed = !seedPoints.isEmpty
                let showMeshNow = self.meshLayer != nil && !hasGS && !hasSeed
                let showGS = hasGS
                let showSeed = !hasGS && hasSeed
                self.appliedGS = showGS
                self.appliedBackground = showGS && self.model.showBackground
                self.appliedSeed = showSeed
                self.appliedCamera = false
                if let id = self.gsChunkID { await renderer.setChunkEnabled(id, enabled: showGS) }
                if let id = self.backgroundChunkID {
                    await renderer.setChunkEnabled(id, enabled: self.appliedBackground)
                }
                if let id = self.seedChunkID { await renderer.setChunkEnabled(id, enabled: showSeed) }
                if let id = self.cameraChunkID { await renderer.setChunkEnabled(id, enabled: false) }
                self.model.setLayers(
                    hasGS: hasGS, hasBackground: !backgroundPoints.isEmpty,
                    hasSeed: hasSeed, hasCameras: !cameraPoints.isEmpty,
                    meshTriangleCount: self.meshLayer?.triangleCount,
                    showGS: showGS, showSeed: showSeed, showCameras: false, showMesh: showMeshNow
                )
                // The count is the trained foreground; the background shell is left out.
                let shownCount = hasGS ? gsPoints.count : seedPoints.count + cameraPoints.count
                self.model.setLoadState(.loaded(count: shownCount))
                self.isLoaded = true
                self.requestRedraw()
            } catch {
                self?.model.setLoadState(.failed(error.localizedDescription))
            }
        }
    }

    /// One PLY split into its scene and the trailing parts named by sidecars.
    private struct ReadSplats {
        var scene: [SplatPoint] = []
        var background: [SplatPoint] = []
        var camera: [SplatPoint] = []
    }

    private enum TrailingSplatKey: String {
        case camera = "cameraSplatCount"
        case background = "backgroundSplatCount"
    }

    /// Reads the number of trailing splats from `<base>.<sidecar>.json` (`{"version":1,"<key>":N}`).
    /// 0 if absent.
    nonisolated private static func trailingSplatCount(
        siblingOf url: URL, sidecar: String, key: TrailingSplatKey
    ) -> Int {
        let metaURL = url.deletingPathExtension().appendingPathExtension("\(sidecar).json")
        guard let data = try? Data(contentsOf: metaURL),
              let meta = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              meta["version"] as? Int == 1,
              let count = meta[key.rawValue] as? Int, count > 0
        else { return 0 }
        return count
    }

    /// The current camera pose (shared by rendering, tap rays and measurement-line projection).
    private func camera() -> SplatViewerCamera {
        SplatViewerCamera(
            azimuth: model.azimuth, elevation: model.elevation, distance: model.distance,
            center: model.center, fovyRadians: model.fovyRadians,
            aspect: Float(drawableSize.width / max(1, drawableSize.height)))
    }

    private func viewportDescriptor(_ camera: SplatViewerCamera) -> SplatRenderer.ViewportDescriptor {
        let viewport = MTLViewport(
            originX: 0, originY: 0,
            width: Double(drawableSize.width), height: Double(drawableSize.height),
            znear: 0, zfar: 1
        )
        return SplatRenderer.ViewportDescriptor(
            viewport: viewport,
            projectionMatrix: camera.projectionMatrix,
            viewMatrix: camera.viewMatrix,
            screenSize: SIMD2(Int(drawableSize.width), Int(drawableSize.height))
        )
    }

    // MARK: - Camera control (called from gesture handlers on the main thread)

    /// One-finger drag: rotate around the look-at point.
    func orbit(dxPixels dx: Float, dyPixels dy: Float) {
        model.azimuth += dx * 0.01
        model.elevation = min(1.5, max(-1.5, model.elevation + dy * 0.01))
    }

    /// Pinch: look-at distance. factor > 1 zooms in.
    func zoom(by factor: Float) {
        guard factor > 0 else { return }
        model.distance = max(0.05, model.distance / factor)
    }

    /// Two-finger drag: pans the look-at point within the screen plane. dy is a pixel delta,
    /// positive downward.
    func pan(dxPixels dx: Float, dyPixels dy: Float) {
        let h = Float(max(1.0, drawableSize.height))
        let fovy = model.fovyRadians
        let worldPerPixel = 2 * model.distance * tanf(fovy * 0.5) / h
        let camera = self.camera()
        // Move center the opposite way so the scene follows the finger.
        model.center -= camera.axis(0) * (dx * worldPerPixel)
        model.center += camera.axis(1) * (dy * worldPerPixel)
    }

    // MARK: - Tapped surface

    /// Ray test against the visible layers (GS, Seed, mesh). nil if none are visible.
    private func visibleRayCaster() -> SplatViewerRayCaster? {
        var chunks: [SplatChunk] = []
        if model.showGS, let gsChunk { chunks.append(gsChunk) }
        if model.showSeed, let seedChunk { chunks.append(seedChunk) }
        let mesh = model.showMesh ? meshLayer?.rayGeometry : nil
        guard !chunks.isEmpty || mesh != nil else { return nil }
        return SplatViewerRayCaster(splatChunks: chunks, mesh: mesh)
    }

    /// Finds, on another thread, the surface under a screen point (top-left 0 / bottom-right 1) and
    /// passes the world point (nil if none).
    private func pickSurface(
        normalized point: SIMD2<Float>, caster: SplatViewerRayCaster,
        completion: @escaping (SIMD3<Float>?) -> Void
    ) {
        let ray = camera().ray(normalized: point)
        let pixelAngle = 2 * tanf(model.fovyRadians * 0.5) / Float(max(1, drawableSize.height))
        isPicking = true
        Task { [weak self] in
            let t = await Task.detached(priority: .userInitiated) {
                caster.cast(origin: ray.origin, direction: ray.direction, pixelAngle: pixelAngle)
            }.value
            guard let self else { return }
            self.isPicking = false
            completion(t.map { ray.origin + ray.direction * $0 })
        }
    }

    // MARK: - Measurement

    /// Tap in measure mode. Places a measurement point on the visible surface under the screen
    /// point (top-left 0 / bottom-right 1).
    func measureTap(normalized point: SIMD2<Float>) {
        guard model.isMeasuring, isLoaded, !isPicking else { return }
        guard let caster = visibleRayCaster() else {
            model.showMeasureNotice("Show GS, Seed or Mesh first")
            return
        }
        pickSurface(normalized: point, caster: caster) { [weak self] hit in
            // Don't place it if measure mode was exited during the test.
            guard let self, self.model.isMeasuring else { return }
            if let hit {
                self.model.addMeasurePoint(hit)
            } else {
                self.model.showMeasureNotice("No surface here")
            }
        }
    }

    // MARK: - Double-tap zoom

    /// Zoom factor per double tap.
    private static let focusZoom: Float = 2
    private static let focusDuration: CFTimeInterval = 0.3

    /// Animation of look-at point and distance. draw(in:) advances it every frame.
    private struct FocusAnimation {
        var fromCenter: SIMD3<Float>
        var toCenter: SIMD3<Float>
        var fromDistance: Float
        var toDistance: Float
        var start: CFTimeInterval
        /// Values written last. If pinch, two-finger drag or Reset overwrote them, those win and
        /// the animation stops.
        var lastCenter: SIMD3<Float>
        var lastDistance: Float
    }
    private var focusAnimation: FocusAnimation?

    /// Double tap. Makes the visible surface under the screen point the new look-at point (center
    /// of rotation) and zooms to half the distance to it. With no surface (background, layers
    /// hidden) the look-at point stays and it zooms to half. Does nothing in measure mode (the
    /// first tap places a measurement point; so as not to delay measurement points, single taps
    /// don't wait for the double tap to fail).
    func focusTap(normalized point: SIMD2<Float>) {
        guard !model.isMeasuring, isLoaded, !isPicking else { return }
        guard let caster = visibleRayCaster() else {
            animateFocus(to: camera().focused(on: nil, zoom: Self.focusZoom))
            return
        }
        pickSurface(normalized: point, caster: caster) { [weak self] hit in
            guard let self, !self.model.isMeasuring else { return }
            // Even if rotated during the test, zoom from the camera at that moment.
            self.animateFocus(to: self.camera().focused(on: hit, zoom: Self.focusZoom))
        }
    }

    private func animateFocus(to target: (center: SIMD3<Float>, distance: Float)) {
        focusAnimation = FocusAnimation(
            fromCenter: model.center, toCenter: target.center,
            fromDistance: model.distance, toDistance: target.distance,
            start: CACurrentMediaTime(),
            lastCenter: model.center, lastDistance: model.distance)
        requestRedraw()
    }

    /// Advances the animation by one frame. The didSet of the written values wakes the next frame.
    private func stepFocusAnimation() {
        guard var a = focusAnimation else { return }
        guard model.center == a.lastCenter, model.distance == a.lastDistance else {
            focusAnimation = nil
            return
        }
        let s = Float(min(1, (CACurrentMediaTime() - a.start) / Self.focusDuration))
        if s < 1 {
            let e = s * s * (3 - 2 * s)
            a.lastCenter = simd_mix(a.fromCenter, a.toCenter, SIMD3(repeating: e))
            // Interpolate distance by ratio (so zoom speed looks constant).
            a.lastDistance = a.fromDistance * powf(a.toDistance / a.fromDistance, e)
        } else {
            a.lastCenter = a.toCenter
            a.lastDistance = a.toDistance
        }
        model.center = a.lastCenter
        model.distance = a.lastDistance
        focusAnimation = s < 1 ? a : nil
    }

    // MARK: MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        drawableSize = size
        requestRedraw()
    }

    /// Syncs the chunks' enabled state with the model's toggles (only on change).
    private func applyChunkToggle(
        _ renderer: SplatRenderer, id: ChunkID?, want: Bool, applied: inout Bool
    ) {
        guard let id, want != applied else { return }
        applied = want
        // The change takes effect from the next render. Until then the draw may stop as final with
        // the old visibility, so wake it once more after applying.
        Task { [weak self] in
            await renderer.setChunkEnabled(id, enabled: want)
            self?.requestRedraw()
        }
    }

    func draw(in view: MTKView) {
        drawableSize = view.drawableSize
        stepFocusAnimation()
        // After the final image is out, if the next frame comes with nothing changed, stop without
        // drawing. Stopping right after the final image would, in modes like stochastic that are
        // final every frame, stop and wake every frame even during interaction, dropping frames
        // while waiting to resume.
        if lastFrameWasFinal && !needsRedraw {
            view.isPaused = true
            return
        }
        guard isLoaded, let renderer else { return }
        let meshLayer = model.showMesh ? self.meshLayer : nil
        let showBackground = model.showGS && model.showBackground
        let splatsWanted = (gsChunkID != nil && model.showGS)
            || (backgroundChunkID != nil && showBackground)
            || (seedChunkID != nil && model.showSeed)
            || (cameraChunkID != nil && model.showCameraMarkers)
        // If drawing splats, they must be sorted (mesh-only doesn't wait).
        if splatsWanted && !renderer.isReadyToRender { return }
        // setChunkEnabled is async, so it takes effect from the sort of a later frame.
        applyChunkToggle(renderer, id: gsChunkID, want: model.showGS, applied: &appliedGS)
        applyChunkToggle(renderer, id: backgroundChunkID, want: showBackground, applied: &appliedBackground)
        applyChunkToggle(renderer, id: seedChunkID, want: model.showSeed, applied: &appliedSeed)
        applyChunkToggle(renderer, id: cameraChunkID, want: model.showCameraMarkers, applied: &appliedCamera)
        // Once the mesh is drawn first, splats are overlaid depth-tested without clearing it.
        renderer.composeOverExistingContents = meshLayer != nil
        guard let drawable = view.currentDrawable else { return }

        _ = inFlight.wait(timeout: .distantFuture)
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            inFlight.signal()
            return
        }
        // Called from a background queue, so make it @Sendable to avoid @MainActor inference.
        let sem = inFlight
        commandBuffer.addCompletedHandler { @Sendable _ in sem.signal() }

        // Exiting before this point means the change wasn't drawn, so keep needsRedraw.
        needsRedraw = false
        let camera = self.camera()
        let viewport = viewportDescriptor(camera)
        // Project measurement lines with the same camera as this frame.
        model.measureProjection.update(
            model.isMeasuring
                ? SplatMeasureProjection.frame(points: model.measurePoints, camera: camera)
                : SplatMeasureProjection.Frame())
        let colorTexture = view.multisampleColorTexture ?? drawable.texture
        let colorStoreAction: MTLStoreAction = view.multisampleColorTexture == nil ? .store : .multisampleResolve
        if let meshLayer, let depthTexture = view.depthStencilTexture {
            meshLayer.encode(
                into: commandBuffer, color: colorTexture, depth: depthTexture,
                clearColor: view.clearColor,
                viewProjection: viewport.projectionMatrix * viewport.viewMatrix,
                viewDirection: camera.forward, shaded: model.meshShaded)
        }
        var didRender = true
        var isFinal = true
        if splatsWanted {
            do {
                didRender = try renderer.render(
                    viewports: [viewport],
                    colorTexture: colorTexture,
                    colorStoreAction: colorStoreAction,
                    depthTexture: view.depthStencilTexture,
                    rasterizationRateMap: nil,
                    renderTargetArrayLength: 0,
                    to: commandBuffer
                )
            } catch {
                didRender = false
            }
            isFinal = renderer.lastRenderWasFinal
        } else if meshLayer == nil {
            // With all layers hidden, draw just the background.
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = colorTexture
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = colorStoreAction
            pass.colorAttachments[0].clearColor = view.clearColor
            commandBuffer.makeRenderCommandEncoder(descriptor: pass)?.endEncoding()
        }
        if didRender { commandBuffer.present(drawable) }
        commandBuffer.commit()
        // Keep looping while waiting for the sort (movement's stochastic, sorted's stale order).
        lastFrameWasFinal = didRender && isFinal
    }
}

// MARK: - Framing

/// Center and camera distance that fit within a 60° vertical FOV. Uses the 2–98th percentile per
/// axis so a few outliers don't inflate the bbox.
private func boundingFrame(_ points: [SplatPoint]) -> (center: SIMD3<Float>, distance: Float) {
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
        a[min(a.count - 1, max(0, Int(q * Float(a.count))))]
    }
    let lo = SIMD3<Float>(pct(xs, 0.02), pct(ys, 0.02), pct(zs, 0.02))
    let hi = SIMD3<Float>(pct(xs, 0.98), pct(ys, 0.98), pct(zs, 0.98))
    let center = (lo + hi) * 0.5
    let radius = max(simd_length(hi - lo) * 0.5, 0.01)
    return (center, framingDistance(radius: radius))
}

/// Camera distance at which a sphere of radius `radius` fits within a 60° vertical FOV with margin.
private func framingDistance(radius: Float) -> Float {
    max(radius, 0.01) / tanf((60 * .pi / 180) * 0.5) * 1.2
}

#endif // os(iOS) || os(macOS)
