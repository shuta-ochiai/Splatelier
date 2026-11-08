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
// tap uses the same ray test and zooms toward the hit surface as the new look-at point. Taps in
// floater mode cast against the foreground for the first thing along the ray and select the splats
// within the brush there (`SplatViewerFloaterBrush`); Detect adds the small clusters apart from the
// scene (`SplatViewerFloaterDetector`). The selection is baked into the foreground chunk as a
// highlight.

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
    /// A CAD model (STEP), drawn like the mesh but placed by `model.cadTransform`.
    private var cadLayer: SplatViewerMeshLayer?
    // Keep the chunks for measurement tap hit tests (only reads GPU buffers, no copy). The
    // background is not a surface, so it is not kept.
    private var gsChunk: SplatChunk?
    private var seedChunk: SplatChunk?
    /// A ray test is in flight. Don't stack another; taps until it finishes are dropped.
    private var isPicking = false
    // Crop: the foreground splats as loaded, the background (which crops leave alone, but a shared
    // PLY keeps), the edits (indices kept) and the handle being dragged. The foreground chunk is
    // rebuilt from the edits after every change; rebuilds run one after another.
    private var gsSourcePoints: [SplatPoint] = []
    private var backgroundSourcePoints: [SplatPoint] = []
    private var cropEdits = SplatViewerCropEdits(count: 0)
    private var cropRebuild: Task<Void, Never>?
    private var sourceName = "splats"
    private var activeCropHandle: SplatViewerCropHandle?
    /// The ring of the Crop box being dragged, and where its rings sit meanwhile (held, so they stay
    /// under the finger as the box turns).
    private var activeCropRing: SplatViewerCADHandle?
    private var cropRingAnchor: SIMD3<Float>?

    /// The Crop box's rings: the CAD model's small full rings, beside the box at the top right.
    private func cropRingGizmo(_ region: SplatViewerCropRegion, _ camera: SplatViewerCamera) -> SplatViewerCADGizmo {
        SplatViewerCADGizmo(box: region, camera: camera, anchor: cropRingAnchor)
    }
    /// The handle of the selected CAD model being dragged (the model itself or one of its rings).
    private var activeCADHandle: SplatViewerCADHandle?
    /// A two-finger twist of the selected CAD model is under way.
    private(set) var isTwistingCAD = false
    /// Where the CAD model's rings sit while one of them is dragged, so they stay under the finger
    /// as the model turns instead of following its outline.
    private var cadRingAnchor: SIMD3<Float>?
    /// Where the CAD model would be if nothing pulled it, while it is dragged: it follows the finger
    /// freely, and what is shown is this settled on a floor within reach.
    private var cadFreeRegion: SplatViewerCropRegion?
    /// The floor found within reach of its foot (nil: none), and how far that reach is.
    private var cadFloor: SplatViewerCADSurface.Floor?
    private var cadSnapReach: Float = 0.05
    /// A floor search is running, and another move came in meanwhile.
    private var isFindingFloor = false
    private var floorSearchPending = false

    private func cadGizmo(_ camera: SplatViewerCamera) -> SplatViewerCADGizmo {
        SplatViewerCADGizmo(box: model.cadRegion, camera: camera, anchor: cadRingAnchor)
    }
    /// Where the finger drawing the Lasso loop is now (normalized coordinates). nil when none is.
    private var lassoTouch: SIMD2<Float>?
    // Floaters: the selected splats, as indices into the kept ones; the detector for the kept splats
    // (worked out on the first Detect, dropped when they change); the scene size the brush is scaled
    // by (worked out when floater mode first opens). `keptGeneration` counts changes to the kept splats, so work
    // started before one is dropped.
    private var floaterSelection: Set<Int32> = []
    private var floaterDetector: SplatViewerFloaterDetector?
    private var floaterDetectorTask: Task<SplatViewerFloaterDetector, Never>?
    private var floaterSceneSize: Float?
    /// Where the last tap brushed, which the brush circle sits around.
    private var lastBrushCenter: SIMD3<Float>?
    /// Floater mode's Undo, newest last: the selection before each change, and whether the change
    /// was a delete (which Undo also takes back in the crop edits). Empty once the kept splats change
    /// some other way or the mode ends.
    private var floaterUndo: [(selection: Set<Int32>, wasDelete: Bool)] = []
    private var keptGeneration = 0

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
        // The stochastic temporal accumulation reprojects with the depth, and seeds its history from
        // the exact frame on screen (SplatRenderer.stochasticTemporalAccumulation).
        view.depthStencilAttachmentTextureUsage = [.renderTarget, .shaderRead]
        view.framebufferOnly = false
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
        model.onTap = { [weak self] in self?.tap(normalized: $0) }
        model.onFocusTap = { [weak self] in self?.focusTap(normalized: $0) }
        model.onLoadCAD = { [weak self] in self?.loadCAD($0) }
        model.onTwistCAD = { [weak self] in self?.twistCAD(by: $0) }
        model.onTurnCAD = { [weak self] in self?.turnCAD($0) }
        model.onRemoveCAD = { [weak self] in self?.removeCAD() }
        model.onTwistCADEnd = { [weak self] in self?.endCADTwist() }
        model.onCropHandleBegin = { [weak self] in self?.beginCropDrag(normalized: $0) ?? false }
        model.onCropHandleDrag = { [weak self] in self?.cropDrag(normalizedDelta: $0) }
        model.onCropHandleEnd = { [weak self] in self?.endCropDrag() }
        model.onCropApply = { [weak self] in self?.applyCrop() }
        model.onCropUndo = { [weak self] in self?.undoCrop() }
        model.onCropRestore = { [weak self] in self?.restoreCrop() }
        model.onCropSelectionRefresh = { [weak self] in self?.refreshCropSelection() }
        model.onColorGradeApply = { [weak self] grade in self?.applyColorGrade(grade) }
        model.onFloaterModeChange = { [weak self] in self?.floaterModeChanged(isOn: $0) }
        model.onFloaterDetect = { [weak self] in self?.detectFloaters() }
        model.onFloaterDeselectAll = { [weak self] in self?.deselectAllFloaters() }
        model.onFloaterDelete = { [weak self] in self?.deleteFloaters() }
        model.onFloaterUndo = { [weak self] in self?.undoFloaters() }
        #if os(iOS)
        // Redraw once when returning from background (if left paused, the screen can stay stale
        // after resuming).
        NotificationCenter.default.addObserver(
            self, selector: #selector(willEnterForeground),
            name: UIApplication.willEnterForegroundNotification, object: nil)
        #endif
    }

    /// Re-bakes the foreground with the current grade. Shares the crop path, so the chunk, the
    /// Share entry and the rebuild order all stay in step with the edits.
    private func applyColorGrade(_ grade: SplatViewerColorGrade) {
        guard isLoaded else { return }
        commitEdits()
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

    /// Loads a GS PLY, seed PLY, mesh OBJ and CAD model (STEP) (all optional) into one scene. GS /
    /// seed point cloud / camera markers go into separate chunks and the mesh and the CAD model
    /// into their own layers, so they can be toggled. `cadPlacement` is where the CAD model was
    /// left last time.
    func load(
        _ url: URL?, seedURL: URL?, meshURL: URL? = nil, cadURL: URL? = nil,
        cadPlacement: SplatViewerCADPlacement? = nil
    ) {
        guard !didLoad, let renderer else { return }
        didLoad = true
        let device = self.device
        let cropShape = model.cropShape
        Task { [weak self] in
            do {
                // Everything that grows with the splat count runs off the main actor: for a large
                // scene it takes seconds on a phone, and the push into the viewer froze meanwhile.
                let scene = try await Task.detached(priority: .userInitiated) {
                    try await Self.prepare(
                        url: url, seedURL: seedURL, meshURL: meshURL, cadURL: cadURL, device: device,
                        cropShape: cropShape)
                }.value
                guard let self else { return }

                let gsPoints = scene.gs.points
                let backgroundPoints = scene.background.points
                let seedPoints = scene.seed.points
                let cameraPoints = scene.camera.points
                var meshError = scene.meshError
                self.gsSourcePoints = gsPoints
                self.backgroundSourcePoints = backgroundPoints
                self.cropEdits = SplatViewerCropEdits(count: gsPoints.count)
                self.sourceName = url?.deletingPathExtension().lastPathComponent ?? "splats"
                if let cropRegion = scene.cropRegion {
                    self.model.cropRegion = cropRegion
                }
                if let mesh = scene.mesh {
                    do {
                        self.meshLayer = try SplatViewerMeshLayer(
                            device: self.device, mesh: mesh,
                            colorFormat: viewerColorFormat, depthFormat: .depth32Float)
                    } catch {
                        meshError = error
                    }
                }
                var cadError = scene.cadError
                if let cad = scene.cad {
                    do {
                        self.cadLayer = try SplatViewerMeshLayer(
                            device: self.device, mesh: cad,
                            colorFormat: viewerColorFormat, depthFormat: .depth32Float)
                    } catch {
                        cadError = error
                    }
                }
                if let cadError {
                    self.model.setCADLoadError("Couldn't load the CAD model: \(cadError)")
                }

                guard !gsPoints.isEmpty || !seedPoints.isEmpty || !cameraPoints.isEmpty
                        || self.meshLayer != nil || self.cadLayer != nil else {
                    if let cadError {
                        self.model.setLoadState(.failed("Couldn't load the CAD model: \(cadError)"))
                    } else if let meshError {
                        self.model.setLoadState(.failed("Couldn't load the mesh: \(meshError)"))
                    } else {
                        self.model.setLoadState(.empty)
                    }
                    return
                }

                // Framing uses whichever exists, in the order GS → seed + cameras → mesh.
                let (center, distance) = scene.splatFrame
                    ?? self.meshLayer.map { ($0.center, framingDistance(radius: $0.radius)) }
                    ?? self.cadLayer.map { ($0.center, framingDistance(radius: $0.radius)) }
                    ?? (SIMD3<Float>(0, 0, 0), 4)
                // Before framing and the CAD model, which stand upright in the viewer.
                self.model.setFileRotation(scene.fileUpright)
                let previewCamera = url.flatMap {
                    SplatViewerPreviewCamera.load(
                        siblingOf: $0, center: center, defaultDistance: distance,
                        sceneFromWorld: scene.fileUpright.inverse)
                }
                self.model.frame(center: center, distance: distance, previewCamera: previewCamera)
                // A CAD model starts at the middle of the scene, to be moved into place by hand.
                if let cadLayer = self.cadLayer {
                    self.model.setCAD(
                        pivot: cadLayer.center, halfExtent: scene.cadHalfExtent, offset: center - cadLayer.center,
                        saved: cadPlacement)
                }

                if let chunk = scene.gs.chunk {
                    self.gsChunkID = await renderer.addChunk(chunk)
                    self.gsChunk = chunk
                }
                if let chunk = scene.background.chunk {
                    self.backgroundChunkID = await renderer.addChunk(chunk)
                }
                if let chunk = scene.seed.chunk {
                    self.seedChunkID = await renderer.addChunk(chunk)
                    self.seedChunk = chunk
                }
                if let chunk = scene.camera.chunk {
                    self.cameraChunkID = await renderer.addChunk(chunk)
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

    /// One layer's splats and the chunk built from them (nil when there are none).
    private struct PreparedSplats: Sendable {
        let points: [SplatPoint]
        let chunk: SplatChunk?

        init(_ points: [SplatPoint], device: MTLDevice) throws {
            self.points = points
            chunk = points.isEmpty ? nil : try SplatChunk(device: device, from: points)
        }
    }

    /// What `load` needs from the files, worked out off the main actor.
    private struct PreparedScene: Sendable {
        let gs: PreparedSplats
        let background: PreparedSplats
        let seed: PreparedSplats
        let camera: PreparedSplats
        let cropRegion: SplatViewerCropRegion?
        /// Framing of the splats; nil when there are none (the mesh frames the view then).
        let splatFrame: (center: SIMD3<Float>, distance: Float)?
        let mesh: LoadedMesh?
        let meshError: Error?
        let cad: LoadedMesh?
        /// Half sizes of the CAD model's bounding box.
        let cadHalfExtent: SIMD3<Float>
        let cadError: Error?
        /// The turn that stands the splats upright in the viewer (`SplatViewerFileAxes`); the mesh
        /// has been brought into their coordinates.
        let fileUpright: simd_quatf
    }

    nonisolated private static func prepare(
        url: URL?, seedURL: URL?, meshURL: URL?, cadURL: URL?, device: MTLDevice,
        cropShape: SplatViewerCropShape
    ) async throws -> PreparedScene {
        // Sidecars name how many splats at the end of the PLY are camera markers (seedmeta; seed
        // PLY order is scene point cloud → camera markers) and background (background; GS PLY order
        // is foreground → background). Splits them off.
        func read(_ u: URL?) async throws -> ReadSplats {
            guard let u else { return ReadSplats() }
            let raw = try await AutodetectSceneReader(u).readAll()
            func finite(_ ps: ArraySlice<SplatPoint>) -> [SplatPoint] {
                ps.filter { $0.position.x.isFinite && $0.position.y.isFinite && $0.position.z.isFinite }
            }
            let camCount = trailingSplatCount(siblingOf: u, sidecar: "seedmeta", key: .camera)
            let camSplit = (camCount > 0 && camCount < raw.count) ? (raw.count - camCount) : raw.count
            let bgCount = trailingSplatCount(siblingOf: u, sidecar: "background", key: .background)
            let bgSplit = (bgCount > 0 && bgCount < camSplit) ? (camSplit - bgCount) : camSplit
            return ReadSplats(
                scene: finite(raw[0..<bgSplit]),
                background: finite(raw[bgSplit..<camSplit]),
                camera: finite(raw[camSplit...]))
        }

        let main = try await read(url)
        let seed = try await read(seedURL)
        // The seed is written like the main PLY, so the main one decides for both.
        let fileUpright = SplatViewerFileAxes.uprightRotation(forSplatsAt: url)
        var mesh: LoadedMesh?
        var meshError: Error?
        if let meshURL {
            do {
                // The mesh is in the capture's Y-up world: into the splats' coordinates.
                let loaded = try OBJMeshLoader.load(meshURL)
                mesh = fileUpright.real == 1 ? loaded : loaded.rotated(by: fileUpright.inverse)
            } catch {
                meshError = error
            }
        }
        var cad: LoadedMesh?
        var cadError: Error?
        if let cadURL {
            do {
                cad = try STEPMeshLoader.load(cadURL)
            } catch {
                cadError = error
            }
        }

        let gsPoints = main.scene
        // Without a foreground there is nothing for it to sit behind.
        let backgroundPoints = gsPoints.isEmpty ? [] : main.background
        // Camera markers are normally only on the seed side, but allow both.
        let cameraPoints = main.camera + seed.camera
        let framePoints = !gsPoints.isEmpty ? gsPoints : (seed.scene + cameraPoints)
        return PreparedScene(
            gs: try PreparedSplats(gsPoints, device: device),
            background: try PreparedSplats(backgroundPoints, device: device),
            seed: try PreparedSplats(seed.scene, device: device),
            camera: try PreparedSplats(cameraPoints, device: device),
            cropRegion: gsPoints.isEmpty
                ? nil : SplatViewerCropRegion.initial(shape: cropShape, positions: gsPoints.map(\.position)),
            splatFrame: framePoints.isEmpty ? nil : boundingFrame(framePoints),
            mesh: mesh,
            meshError: meshError,
            cad: cad,
            cadHalfExtent: cad.map { boxHalfExtent($0.positions) } ?? .zero,
            cadError: cadError,
            fileUpright: fileUpright)
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
            aspect: Float(drawableSize.width / max(1, drawableSize.height)),
            sceneRotation: model.viewRotation)
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
        var cad = model.showCAD ? cadLayer?.rayGeometry : nil
        cad?.transform = model.cadTransform
        guard !chunks.isEmpty || mesh != nil || cad != nil else { return nil }
        return SplatViewerRayCaster(splatChunks: chunks, mesh: mesh, cad: cad)
    }

    /// Finds, on another thread, the surface under a screen point (top-left 0 / bottom-right 1) and
    /// passes the world point (nil if none). `frontOpacity` finds the first thing along the ray
    /// instead (`SplatViewerRayCaster.cast`).
    private func pickSurface(
        normalized point: SIMD2<Float>, caster: SplatViewerRayCaster, frontOpacity: Float? = nil,
        completion: @escaping (SIMD3<Float>?) -> Void
    ) {
        let ray = camera().ray(normalized: point)
        let pixelAngle = 2 * tanf(model.fovyRadians * 0.5) / Float(max(1, drawableSize.height))
        isPicking = true
        Task { [weak self] in
            let t = await Task.detached(priority: .userInitiated) {
                caster.cast(
                    origin: ray.origin, direction: ray.direction, pixelAngle: pixelAngle,
                    frontOpacity: frontOpacity)
            }.value
            guard let self else { return }
            self.isPicking = false
            completion(t.map { ray.origin + ray.direction * $0 })
        }
    }

    /// A single tap: a measurement point in measure mode, a floater in floater mode. Ignored
    /// otherwise.
    func tap(normalized point: SIMD2<Float>) {
        if model.isRemovingFloaters {
            floaterTap(normalized: point)
        } else if model.isMeasuring {
            measureTap(normalized: point)
        } else if !model.isCropping && !model.isColoring {
            cadTap(normalized: point)
        }
    }

    // MARK: - CAD model

    /// Loads (or replaces) the CAD model while the scene is open. It goes to the middle of the view
    /// and is selected, so it can be moved into place straight away.
    func loadCAD(_ url: URL) {
        guard isLoaded else { return }
        model.setLoadingCAD(true)
        Task { [weak self] in
            do {
                let mesh = try await Task.detached(priority: .userInitiated) { try STEPMeshLoader.load(url) }.value
                guard let self else { return }
                let layer = try SplatViewerMeshLayer(
                    device: self.device, mesh: mesh, colorFormat: viewerColorFormat, depthFormat: .depth32Float)
                self.cadLayer = layer
                self.model.setCAD(
                    pivot: layer.center, halfExtent: boxHalfExtent(mesh.positions),
                    offset: self.model.center - layer.center)
                self.model.showCAD = true
                self.model.isCADSelected = true
                self.model.setLoadingCAD(false)
                self.model.onCADPlacementChange?(self.model.cadPlacement)
                self.requestRedraw()
            } catch {
                self?.model.setLoadingCAD(false)
                self?.model.setCADLoadError("Couldn't load the CAD model: \(error)")
            }
        }
    }

    /// A tap on the CAD model selects it (it is tinted and its rings show); anywhere else lets go.
    private func cadTap(normalized point: SIMD2<Float>) {
        guard isLoaded, model.showCAD else { return }
        model.isCADSelected = cadIsGrabbed(at: point)
    }

    /// Takes the CAD model out of the scene; Import can bring one in again.
    func removeCAD() {
        activeCADHandle = nil
        cadFreeRegion = nil
        cadRingAnchor = nil
        cadLayer = nil
        model.clearCAD()
        model.cadProjection.update(SplatCropFrame())
        requestRedraw()
    }

    /// One of the quarter-turn buttons shown while the CAD model is selected.
    func turnCAD(_ turn: SplatViewerCADTurn) {
        guard isLoaded, model.isCADSelected else { return }
        model.setCADRegion(turn.applied(to: model.cadRegion, camera: camera()))
        model.onCADPlacementChange?(model.cadPlacement)
    }

    /// A touch takes hold of the CAD model: on it, or near its box when it is small on screen or
    /// selected (`SplatViewerCADReach`).
    private func cadIsGrabbed(at point: SIMD2<Float>) -> Bool {
        SplatViewerCADReach.takes(
            point, box: model.cadRegion, selected: model.isCADSelected, onModel: cadIsUnder(point),
            camera: camera(), viewSize: pixelSize)
    }

    private func cadIsUnder(_ point: SIMD2<Float>) -> Bool {
        guard var cad = cadLayer?.rayGeometry else { return false }
        cad.transform = model.cadTransform
        let ray = camera().ray(normalized: point)
        return SplatViewerRayCaster(splatChunks: [], mesh: nil, cad: cad)
            .cast(origin: ray.origin, direction: ray.direction, pixelAngle: 0) != nil
    }

    /// Start of a drag: a ring of the selected CAD model turns it; the model itself, selected or
    /// not, is selected and moved. Only outside the other modes, which have drags of their own.
    private func beginCADDrag(normalized point: SIMD2<Float>) -> Bool {
        guard isLoaded, model.hasCAD, model.showCAD,
              !model.isCropping, !model.isMeasuring, !model.isColoring, !model.isRemovingFloaters
        else { return false }
        let size = pixelSize
        if model.isCADSelected,
           let ring = cadGizmo(camera()).ring(
            at: point, camera: camera(), viewSize: size,
            ringReach: Self.cadRingGrabFraction * min(size.x, size.y)) {
            activeCADHandle = ring
            cadRingAnchor = cadGizmo(camera()).anchor
        } else if cadIsGrabbed(at: point) {
            model.isCADSelected = true
            activeCADHandle = .move
            cadFreeRegion = model.cadRegion
            cadFloor = nil
            // The pull reaches a fixed distance on screen, so it feels the same at any zoom.
            let camera = self.camera()
            let depth = max(simd_dot(model.cadRegion.center - camera.eye, camera.forward), 1e-3)
            let worldPerPixel = 2 * depth * tanf(camera.fovyRadians / 2) / max(pixelSize.y, 1)
            cadSnapReach = SplatViewerCADSurface.snapPixels * worldPerPixel
        } else {
            return false
        }
        requestRedraw()
        return true
    }

    /// The surfaces a dragged CAD model settles on: the mesh made from the capture when there is one
    /// (it is quicker and its faces are sharp), else the splats. The model itself is left out.
    private func surfaceCaster() -> SplatViewerRayCaster? {
        if let mesh = meshLayer?.rayGeometry { return SplatViewerRayCaster(splatChunks: [], mesh: mesh) }
        if let gsChunk { return SplatViewerRayCaster(splatChunks: [gsChunk], mesh: nil) }
        return nil
    }

    /// Looks for a floor-like surface within reach under or over the dragged CAD model's foot, and
    /// settles the model on it. The search runs off the main actor, one at a time; moves that arrive
    /// meanwhile are searched again once it lands. Until then the last floor found holds, so the
    /// model doesn't flicker.
    private func findFloor() {
        guard let free = cadFreeRegion, let caster = surfaceCaster() else { return }
        if isFindingFloor {
            floorSearchPending = true
            return
        }
        isFindingFloor = true
        let reach = cadSnapReach
        // Floors are found in the viewer's coordinates, where up is +Y however the scene is turned.
        let scene = model.viewRotation
        let foot = SplatViewerCADSurface.foot(of: free.rotated(by: scene))
        // The slope is read across at least 15 cm, so splat bumps don't make the model wobble.
        let probes = SplatViewerCADSurface.probes(under: foot, reach: reach, spread: max(4 * reach, 0.15))
        let down = SIMD3<Float>(0, -1, 0)
        Task { [weak self] in
            let hits = await Task.detached(priority: .userInitiated) {
                probes.map { origin in
                    caster.cast(origin: scene.inverse.act(origin), direction: scene.inverse.act(down), pixelAngle: 0.002)
                        .flatMap { $0 <= 2 * reach ? origin + down * $0 : nil }
                }
            }.value
            guard let self else { return }
            self.isFindingFloor = false
            // The scene may have turned meanwhile: the floor goes with it.
            self.cadFloor = SplatViewerCADSurface.floor(hits: hits, foot: foot, reach: reach)?.rotated(by: scene.inverse)
            if let free = self.cadFreeRegion {
                self.model.setCADRegion(self.settledCAD(free, reach: reach))
            }
            if self.floorSearchPending, self.cadFreeRegion != nil {
                self.floorSearchPending = false
                self.findFloor()
            } else {
                self.floorSearchPending = false
                // The drag ended before this search landed: keep where it settled.
                if self.activeCADHandle == nil { self.model.onCADPlacementChange?(self.model.cadPlacement) }
            }
        }
    }

    /// `box` settled on the last floor found, worked out in the viewer's coordinates (+Y up).
    private func settledCAD(_ box: SplatViewerCropRegion, reach: Float) -> SplatViewerCropRegion {
        let scene = model.viewRotation
        return SplatViewerCADSurface.settled(
            box.rotated(by: scene), on: cadFloor?.rotated(by: scene), reach: reach
        ).rotated(by: scene.inverse)
    }

    /// Two-finger twist (or trackpad rotation) of the selected CAD model about the vertical.
    func twistCAD(by radians: Float) {
        guard isLoaded, model.isCADSelected, model.showCAD else { return }
        isTwistingCAD = true
        model.setCADRegion(
            SplatViewerCADGizmo(box: model.cadRegion, camera: camera()).twisted(by: radians, camera: camera()))
    }

    func endCADTwist() {
        guard isTwistingCAD else { return }
        isTwistingCAD = false
        model.onCADPlacementChange?(model.cadPlacement)
        requestRedraw()
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

    // MARK: - Crop

    /// How close, in pixels, a touch must be to a handle to grab it.
    /// How far a touch may be from a handle and still grab it, as a fraction of the shorter side of
    /// the drawable. A fixed pixel count shrinks to about 15 points on a Retina iPhone.
    private static let cropGrabFraction: Float = 0.09
    /// The same for the rings of a CAD model, a little tighter: they lie over the model, which is
    /// itself dragged to move it.
    private static let cadRingGrabFraction: Float = 0.035

    private var pixelSize: SIMD2<Float> {
        SIMD2(Float(drawableSize.width), Float(drawableSize.height))
    }

    /// Start of a drag (normalized coordinates). Grabs the nearest handle within reach. false leaves
    /// the drag to the camera. In Lasso every drag draws a new loop, so the camera is moved with two
    /// fingers there.
    func beginCropDrag(normalized point: SIMD2<Float>) -> Bool {
        if beginCADDrag(normalized: point) { return true }
        guard model.isCropping, isLoaded else { return false }
        if model.cropShape == .lasso {
            lassoTouch = point
            model.cropProjection.setLasso([point])
            return true
        }
        guard let region = model.cropRegion else { return false }
        let camera = self.camera()
        let size = pixelSize
        // A box turns by the same small rings as a CAD model, beside it at the top right.
        if region.shape == .box,
           let ring = cropRingGizmo(region, camera).ring(
            at: point, camera: camera, viewSize: size, ringReach: Self.cadRingGrabFraction * min(size.x, size.y)) {
            activeCropRing = ring
            cropRingAnchor = cropRingGizmo(region, camera).anchor
            return true
        }
        let nearest = region.handles(camera: camera).compactMap { item -> (SplatViewerCropHandle, Float)? in
            guard let s = camera.project(cameraSpace: camera.cameraSpace(item.position)) else { return nil }
            return (item.handle, simd_distance(s * size, point * size))
        }.min { $0.1 < $1.1 }
        let reach = Self.cropGrabFraction * min(size.x, size.y)
        if let nearest, nearest.1 <= reach {
            activeCropHandle = nearest.0
            return true
        }
        return false
    }

    /// Drag of the grabbed handle. `delta` is the touch movement normalized by the view size.
    func cropDrag(normalizedDelta delta: SIMD2<Float>) {
        if let handle = activeCADHandle {
            if handle == .move, let free = cadFreeRegion {
                // The model moves freely in the screen plane, and settles on a floor within reach.
                let moved = free.dragged(.center, byPixels: delta * pixelSize, camera: camera(), viewSize: pixelSize)
                cadFreeRegion = moved
                model.setCADRegion(settledCAD(moved, reach: cadSnapReach))
                findFloor()
            } else {
                model.setCADRegion(cadGizmo(camera()).dragged(
                    handle, byPixels: delta * pixelSize, camera: camera(), viewSize: pixelSize))
            }
            return
        }
        if let touch = lassoTouch {
            let moved = simd_clamp(touch + delta, .zero, SIMD2(repeating: 1))
            lassoTouch = moved
            let size = pixelSize
            model.cropProjection.setLasso(
                SplatViewerLasso.appending(
                    moved, to: model.cropProjection.lasso, aspect: size.x / max(size.y, 1)))
            return
        }
        if let ring = activeCropRing, let region = model.cropRegion {
            model.cropRegion = cropRingGizmo(region, camera()).dragged(
                ring, byPixels: delta * pixelSize, camera: camera(), viewSize: pixelSize)
            return
        }
        guard let handle = activeCropHandle, let region = model.cropRegion else { return }
        model.cropRegion = region.dragged(
            handle, byPixels: delta * pixelSize, camera: camera(), viewSize: pixelSize)
    }

    func endCropDrag() {
        if activeCADHandle != nil {
            activeCADHandle = nil
            cadRingAnchor = nil
            cadFreeRegion = nil
            // A floor search still running saves again when it lands.
            model.onCADPlacementChange?(model.cadPlacement)
            requestRedraw()
        }
        if activeCropHandle != nil || activeCropRing != nil {
            activeCropHandle = nil
            activeCropRing = nil
            cropRingAnchor = nil
            refreshCropSelection()
        }
        if lassoTouch != nil {
            lassoTouch = nil
            // A scribble that encloses nothing is dropped; a loop turns into a selection.
            let loop = model.cropProjection.lasso
            model.cropProjection.setLasso([])
            if SplatViewerLasso.isUsable(loop) { selectWithLasso(loop) }
        }
    }

    /// Removes what the region doesn't keep (or keeps, if "remove inside" is off).
    func applyCrop() {
        guard isLoaded else { return }
        let keepsInside = model.cropKeepsInside
        let source = gsSourcePoints
        let isInside: (UInt32) -> Bool
        if model.cropShape == .lasso {
            guard !floaterSelection.isEmpty else {
                model.showCropNotice("Draw a loop around what to crop")
                return
            }
            let kept = cropEdits.kept
            let selected = Set(floaterSelection.map { kept[Int($0)] })
            isInside = { selected.contains($0) }
        } else {
            guard let region = model.cropRegion else { return }
            isInside = { region.contains(source[Int($0)].position) }
        }
        switch cropEdits.apply({ isInside($0) == keepsInside }) {
        case .applied:
            commitCropEdits()
        case .unchanged:
            model.showCropNotice("Nothing to remove here")
        case .emptied:
            model.showCropNotice("That would remove every splat")
        }
    }

    /// Selects the kept splats whose center is inside the loop, as seen from the camera now. They
    /// turn orange, and move with the scene when the view does.
    private func selectWithLasso(_ loop: [SIMD2<Float>]) {
        let kept = cropEdits.kept, source = gsSourcePoints
        let test = SplatViewerLasso.test(polygon: loop, camera: camera())
        let selection = Set(kept.indices.lazy.filter { test(source[Int(kept[$0])].position) }.map(Int32.init))
        guard !selection.isEmpty else {
            model.showCropNotice("No splats inside the loop")
            return
        }
        floaterSelection = selection
        commitEdits()
    }

    /// The selection Crop shows in orange: for Box and Sphere, the kept splats inside the region;
    /// for Lasso none until a loop is drawn (so none when Lasso is picked, or Crop is closed).
    private func cropSelection() -> Set<Int32> {
        guard model.isCropping, model.cropShape != .lasso, let region = model.cropRegion else { return [] }
        let source = gsSourcePoints
        return region.insideIndices(kept: cropEdits.kept) { source[Int($0)].position }
    }

    /// Paints the selection again after the region was let go of, the shape changed, or Crop was
    /// opened or closed. Floater mode keeps its own selection.
    private func refreshCropSelection() {
        guard isLoaded, !model.isRemovingFloaters else { return }
        let selection = cropSelection()
        guard selection != floaterSelection else { return }
        floaterSelection = selection
        commitEdits()
    }

    func undoCrop() {
        guard isLoaded else { return }
        cropEdits.undo()
        commitCropEdits()
    }

    /// Back to every splat as loaded.
    func restoreCrop() {
        guard isLoaded else { return }
        cropEdits.restoreAll()
        commitCropEdits()
    }

    /// After the kept splats changed: the detector, the selection and floater mode's Undo were for
    /// the old ones, so they go, then the edits are committed.
    private func commitCropEdits() {
        keptChanged()
        floaterUndo = []
        floaterSelection = cropSelection()
        commitEdits()
    }

    /// Drops what was worked out for the kept splats before they changed.
    private func keptChanged() {
        keptGeneration += 1
        floaterDetector = nil
        floaterDetectorTask = nil
        floaterSelection = []
    }

    /// Updates the count, the Share entry, the floater selection and the foreground chunk after the
    /// edits changed.
    private func commitEdits() {
        let kept = cropEdits.kept
        let source = gsSourcePoints
        let background = backgroundSourcePoints
        let grade = model.colorGrade
        model.setLoadState(.loaded(count: kept.count))
        // The Edited entry covers a crop and a colour grade alike; either one changes the splats.
        let isEdited = cropEdits.isEdited || !grade.isIdentity
        let shareItem = isEdited
            ? Self.editedShareItem(
                kept: kept, source: source, background: background, grade: grade,
                export: model.cropExport, name: sourceName)
            : nil
        model.setCropState(edited: cropEdits.isEdited, canUndo: cropEdits.canUndo, shareItem: shareItem)
        var highlighted: [Bool]?
        if !floaterSelection.isEmpty {
            var mask = [Bool](repeating: false, count: kept.count)
            for i in floaterSelection { mask[Int(i)] = true }
            highlighted = mask
        }
        updateFloaterState()
        rebuildGSChunk(kept: kept, source: source, grade: grade, highlighted: highlighted)
    }

    /// Replaces the foreground chunk with the kept splats, graded, and the selected floaters
    /// highlighted (`highlighted` is in the order of `kept`). Rebuilds run one after another.
    /// The new chunk is built off the main actor and added before the old one is removed, so the
    /// Apply doesn't stall the view and the splats don't drop out while it builds.
    private func rebuildGSChunk(
        kept: [UInt32], source: [SplatPoint], grade: SplatViewerColorGrade, highlighted: [Bool]?
    ) {
        guard let renderer else { return }
        let previous = cropRebuild
        let device = self.device
        // Lasso tints the selection; floaters paint it flat.
        let tinted = model.isCropping
        cropRebuild = Task { [weak self] in
            await previous?.value
            guard let self else { return }
            let enabled = self.appliedGS
            let built = await Task.detached(priority: .userInitiated) { () -> Result<SplatChunk, Error> in
                Result {
                    try SplatChunk(device: device, from: kept.indices.map { i in
                        let point = grade.apply(to: source[Int(kept[i])])
                        guard highlighted?[i] == true else { return point }
                        return tinted ? SplatViewerFloaterHighlight.tint(point) : SplatViewerFloaterHighlight.apply(to: point)
                    })
                }
            }.value
            switch built {
            case .success(let chunk):
                let newID = await renderer.addChunk(chunk, enabled: enabled)
                let oldID = self.gsChunkID
                self.gsChunk = chunk
                self.gsChunkID = newID
                if let oldID { await renderer.removeChunk(oldID) }
            case .failure:
                if let oldID = self.gsChunkID { await renderer.removeChunk(oldID) }
                self.gsChunk = nil
                self.gsChunkID = nil
                self.model.showCropNotice("Couldn't rebuild the splats")
            }
            self.requestRedraw()
        }
    }

    /// Share entry that writes the edited splats as PLY. The arrays are values, so the file is
    /// written from the edits as they were when the entry was made.
    private static func editedShareItem(
        kept: [UInt32], source: [SplatPoint], background: [SplatPoint], grade: SplatViewerColorGrade,
        export: SplatViewerCropExport?, name: String
    ) -> SplatViewerShareItem {
        // Every option writes the edited PLY when it is picked. The PLY is the first option, and
        // each format the app offers is converted from that same PLY.
        let writePLY: @Sendable () async throws -> URL = {
            try await writeEditedPLY(kept.map { grade.apply(to: source[Int($0)]) } + background, name: name)
        }
        var options = [SplatViewerShareItem.Option(title: "PLY", prepare: writePLY)]
        for format in export?.formats ?? [] {
            options.append(SplatViewerShareItem.Option(title: format.title, prepare: {
                guard let export else { return try await writePLY() }
                return try await export.convert(try await writePLY(), format.fileExtension)
            }))
        }
        return SplatViewerShareItem(title: "Edited splats", systemImage: "slider.horizontal.3", options: options)
    }

    // MARK: - Floaters

    /// How opaque the front of the ray must get for a floater tap to hit. Floaters are faint, so
    /// this is far below the half used for a surface.
    private static let floaterPickOpacity: Float = 0.05

    /// Opening floater mode works out the scene size, so the brush circle can be drawn before the
    /// first tap. Leaving it drops the selection and its Undo steps (a delete stays undoable from the
    /// Crop panel, as a crop is).
    private func floaterModeChanged(isOn: Bool) {
        guard isLoaded else { return }
        if isOn {
            guard floaterSceneSize == nil else { return }
            let source = gsSourcePoints
            Task {
                let size = await Task.detached(priority: .userInitiated) {
                    SplatViewerFloaterBrush.sceneSize(of: source.map(\.position))
                }.value
                floaterSceneSize = size
                requestRedraw()
            }
        } else {
            floaterUndo = []
            floaterSelection = []
            commitEdits()
        }
    }

    private func updateFloaterState() {
        model.setFloaterState(
            selectionCount: floaterSelection.count, isFinding: floaterDetectorTask != nil,
            canUndo: !floaterUndo.isEmpty || cropEdits.canUndo)
    }

    /// Changes the selection as one Undo step.
    private func setFloaterSelection(_ selection: Set<Int32>) {
        guard selection != floaterSelection else { return }
        floaterUndo.append((floaterSelection, false))
        floaterSelection = selection
        commitEdits()
    }

    /// Takes back the last step: a selection change, or a delete, which brings the splats back
    /// selected. With no step left, takes back the crop before floater mode.
    func undoFloaters() {
        guard isLoaded else { return }
        guard let step = floaterUndo.popLast() else {
            undoCrop()
            return
        }
        if step.wasDelete {
            cropEdits.undo()
            keptChanged()
        }
        floaterSelection = step.selection
        commitEdits()
    }

    /// Tap in floater mode: brushes the splats around the first thing under the screen point. A tap
    /// on the selection lets the brushed splats go (`SplatViewerFloaterBrush.toggled`).
    func floaterTap(normalized point: SIMD2<Float>) {
        guard model.isRemovingFloaters, isLoaded, !isPicking else { return }
        guard model.showGS, let gsChunk else {
            model.showCropNotice("Show the splats first")
            return
        }
        let caster = SplatViewerRayCaster(splatChunks: [gsChunk], mesh: nil)
        pickSurface(normalized: point, caster: caster, frontOpacity: Self.floaterPickOpacity) { [weak self] hit in
            guard let self, self.model.isRemovingFloaters else { return }
            guard let hit else {
                self.model.showCropNotice("No splats here")
                return
            }
            self.brush(at: hit)
        }
    }

    private func brush(at center: SIMD3<Float>) {
        lastBrushCenter = center
        let kept = cropEdits.kept, source = gsSourcePoints
        let generation = keptGeneration
        let sizeFraction = model.floaterBrushSize
        let knownSceneSize = floaterSceneSize
        Task {
            let (splats, sceneSize) = await Task.detached(priority: .userInitiated) {
                // Scaled by the scene as loaded, so a brush stays the same size after deletes.
                let sceneSize = knownSceneSize ?? SplatViewerFloaterBrush.sceneSize(of: source.map(\.position))
                let positions = kept.map { source[Int($0)].position }
                return (
                    SplatViewerFloaterBrush.splats(within: sizeFraction * sceneSize, of: center, positions: positions),
                    sceneSize)
            }.value
            floaterSceneSize = sceneSize
            guard model.isRemovingFloaters, keptGeneration == generation else { return }
            guard !splats.isEmpty else {
                model.showCropNotice("No splats here")
                return
            }
            setFloaterSelection(SplatViewerFloaterBrush.toggled(floaterSelection, by: splats))
        }
    }

    /// The detector for the kept splats, worked out off the main actor on first use. nil if the kept
    /// splats changed while it was being worked out.
    private func detectorForKept() async -> SplatViewerFloaterDetector? {
        if let floaterDetector { return floaterDetector }
        let task: Task<SplatViewerFloaterDetector, Never>
        if let floaterDetectorTask {
            task = floaterDetectorTask
        } else {
            let kept = cropEdits.kept, source = gsSourcePoints
            task = Task.detached(priority: .userInitiated) {
                SplatViewerFloaterDetector(splats: source, kept: kept)
            }
            floaterDetectorTask = task
            updateFloaterState()
        }
        let detector = await task.value
        guard floaterDetectorTask == task else { return nil }
        floaterDetector = detector
        floaterDetectorTask = nil
        updateFloaterState()
        return detector
    }

    /// Adds the small clusters apart from the scene to the selection.
    func detectFloaters() {
        guard isLoaded else { return }
        Task {
            guard let detector = await detectorForKept(), model.isRemovingFloaters else { return }
            let selection = floaterSelection.union(detector.floaterSplats)
            guard selection.count > floaterSelection.count else {
                model.showCropNotice("No more floaters found")
                return
            }
            setFloaterSelection(selection)
        }
    }

    func deselectAllFloaters() {
        setFloaterSelection([])
    }

    /// Removes the selected splats. Undo brings them back, selected.
    func deleteFloaters() {
        guard isLoaded, !floaterSelection.isEmpty else { return }
        let kept = cropEdits.kept
        let removed = Set(floaterSelection.map { kept[Int($0)] })
        switch cropEdits.apply({ !removed.contains($0) }) {
        case .applied:
            let selection = floaterSelection
            keptChanged()
            floaterUndo.append((selection, true))
            commitEdits()
        case .unchanged:
            model.showCropNotice("Nothing to remove here")
        case .emptied:
            model.showCropNotice("That would remove every splat")
        }
    }

    /// The brush circle while it is shown: around the last tapped point if it is on screen, else
    /// around the look-at point.
    private func brushFrame(camera: SplatViewerCamera) -> SplatBrushFrame? {
        guard model.isRemovingFloaters, model.isShowingBrush, let sceneSize = floaterSceneSize else { return nil }
        let radius = model.floaterBrushSize * sceneSize
        if let lastBrushCenter,
           let frame = SplatBrushFrame.make(anchor: lastBrushCenter, radius: radius, camera: camera),
           frame.isOnScreen {
            return frame
        }
        return SplatBrushFrame.make(anchor: model.center, radius: radius, camera: camera)
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
    /// Floater mode ignores it too, since its taps select.
    func focusTap(normalized point: SIMD2<Float>) {
        guard !model.isMeasuring, !model.isRemovingFloaters, isLoaded, !isPicking else { return }
        guard let caster = visibleRayCaster() else {
            animateFocus(to: camera().focused(on: nil, zoom: Self.focusZoom))
            return
        }
        pickSurface(normalized: point, caster: caster) { [weak self] hit in
            guard let self, !self.model.isMeasuring, !self.model.isRemovingFloaters else { return }
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
        let cadLayer = model.showCAD ? self.cadLayer : nil
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
        renderer.composeOverExistingContents = meshLayer != nil || cadLayer != nil
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
        if model.isCropping, model.cropShape != .lasso, let region = model.cropRegion {
            var frame = SplatCropFrame.make(region: region, camera: camera)
            if region.shape == .box {
                let rings = cropRingGizmo(region, camera).frame(camera: camera)
                frame.rings = rings.rings
                frame.ringArrows = rings.ringArrows
            }
            model.cropProjection.update(frame)
        } else {
            model.cropProjection.update(SplatCropFrame())
        }
        if model.isCADSelected && model.showCAD {
            model.cadProjection.update(cadGizmo(camera).frame(camera: camera))
        } else {
            model.cadProjection.update(SplatCropFrame())
        }
        model.brushProjection.update(brushFrame(camera: camera))
        let colorTexture = view.multisampleColorTexture ?? drawable.texture
        let colorStoreAction: MTLStoreAction = view.multisampleColorTexture == nil ? .store : .multisampleResolve
        if let depthTexture = view.depthStencilTexture {
            // The opaque layers go first, each depth-tested against the last; the first clears.
            let viewProjection = viewport.projectionMatrix * viewport.viewMatrix
            meshLayer?.encode(
                into: commandBuffer, color: colorTexture, depth: depthTexture, clear: true,
                clearColor: view.clearColor, viewProjection: viewProjection, viewDirection: camera.forward,
                appearance: model.meshShaded ? .shaded : .colors)
            cadLayer?.encode(
                into: commandBuffer, color: colorTexture, depth: depthTexture, clear: meshLayer == nil,
                clearColor: view.clearColor, viewProjection: viewProjection, model: model.cadTransform,
                viewDirection: camera.forward, appearance: .litColors,
                highlight: !model.isCADSelected ? 0 : activeCADHandle != nil || isTwistingCAD ? 1 : 0.7)
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
        } else if meshLayer == nil && cadLayer == nil {
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

/// Half sizes of the bounding box of `positions` ([V*3]).
private func boxHalfExtent(_ positions: [Float]) -> SIMD3<Float> {
    var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
    var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
    for v in 0..<(positions.count / 3) {
        let p = SIMD3(positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2])
        lo = pointwiseMin(lo, p)
        hi = pointwiseMax(hi, p)
    }
    return positions.isEmpty ? .zero : (hi - lo) / 2
}

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
