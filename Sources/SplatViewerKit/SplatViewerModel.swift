// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Shared state between the SwiftUI gesture layer and the MTKView draw loop. The orbit-camera fields
// are updated every drag frame, so they are intentionally not @Published to avoid invalidating
// SwiftUI. Drawing stops while at rest, so fields that affect the image wake it via `onNeedsRedraw`
// when changed.

#if os(iOS) || os(macOS)

import simd
import SwiftUI

struct SplatViewerPreviewCamera {
    let azimuth: Float
    let elevation: Float
    /// The source view's forward in the scene's coordinates.
    let forward: SIMD3<Float>
    let distance: Float
    let fovyRadians: Float
    /// Orbit pivot on the source view's optical axis, `distance` ahead of its eye, so the orbit
    /// camera starts exactly at the training camera's position rather than on a line through the
    /// scene center.
    let center: SIMD3<Float>

    private struct Metadata: Decodable {
        let version: Int
        let width: Int
        let height: Int
        let viewmat: [Float]
        let intrinsics: [Float]
    }

    /// `sceneFromWorld` brings the source view from the capture's world into the splats'
    /// coordinates (they may be turned from it, see `SplatViewerFileAxes`); `center` is in the latter.
    static func load(
        siblingOf url: URL,
        center: SIMD3<Float>,
        defaultDistance: Float,
        sceneFromWorld: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    ) -> SplatViewerPreviewCamera? {
        let metadataURL = url.deletingPathExtension().appendingPathExtension("preview.json")
        guard let data = try? Data(contentsOf: metadataURL),
              let metadata = try? JSONDecoder().decode(Metadata.self, from: data),
              metadata.version == 1,
              metadata.width > 0,
              metadata.height > 0,
              metadata.viewmat.count == 16,
              metadata.intrinsics.count == 4,
              metadata.intrinsics[1] > 0
        else { return nil }

        let fovy = 2 * atan(Float(metadata.height) / (2 * metadata.intrinsics[1]))
        guard fovy.isFinite, fovy > 0 else { return nil }

        let forwardWorld = Self.cameraForwardWorld(fromTrainingViewmat: metadata.viewmat)
        let eyeWorld = Self.cameraCenterWorld(fromTrainingViewmat: metadata.viewmat)
        guard let forwardWorld, let eyeWorld else { return nil }
        let forward = sceneFromWorld.act(forwardWorld), eye = sceneFromWorld.act(eyeWorld)

        let projectedDistance = simd_dot(center - eye, forward)
        let distance = projectedDistance.isFinite && projectedDistance > 0.05
            ? projectedDistance
            : defaultDistance
        let angles = Self.orbitAngles(lookingToward: forward)
        return SplatViewerPreviewCamera(
            azimuth: angles.azimuth,
            elevation: angles.elevation,
            forward: forward,
            distance: distance,
            fovyRadians: fovy,
            center: eye + forward * distance
        )
    }

    private static func cameraForwardWorld(fromTrainingViewmat m: [Float]) -> SIMD3<Float>? {
        // Training cameras are row-major world→camera (OpenCV style, +Z forward). Row 3 is the
        // forward axis in world.
        let forward = SIMD3<Float>(m[8], m[9], m[10])
        let length = simd_length(forward)
        guard length.isFinite, length > 1e-6 else { return nil }
        return forward / length
    }

    private static func cameraCenterWorld(fromTrainingViewmat m: [Float]) -> SIMD3<Float>? {
        // Camera center in world C = -Rᵀ t.
        let center = SIMD3<Float>(
            -(m[0] * m[3] + m[4] * m[7] + m[8] * m[11]),
            -(m[1] * m[3] + m[5] * m[7] + m[9] * m[11]),
            -(m[2] * m[3] + m[6] * m[7] + m[10] * m[11])
        )
        guard center.x.isFinite, center.y.isFinite, center.z.isFinite else { return nil }
        return center
    }

    /// Orbit angles that look along the source view once the scene is turned by `sceneRotation`.
    func orbitAngles(sceneRotation: simd_quatf) -> (azimuth: Float, elevation: Float) {
        Self.orbitAngles(lookingToward: sceneRotation.act(forward))
    }

    private static func orbitAngles(lookingToward forward: SIMD3<Float>) -> (azimuth: Float, elevation: Float) {
        let clampedY = min(1, max(-1, forward.y))
        let elevation = asin(-clampedY)
        let azimuth = atan2(forward.x, -forward.z)
        return (azimuth, elevation)
    }
}

@MainActor
final class SplatViewerModel: ObservableObject {
    enum LoadState: Equatable {
        case loading
        case loaded(count: Int)
        case empty
        case failed(String)
    }

    @Published private(set) var loadState: LoadState = .loading

    /// Call when state affecting the image changes (the renderer resumes its draw loop).
    var onNeedsRedraw: (() -> Void)?

    // The 4 layers of the unified view: GS (main PLY) / seed point cloud / camera markers (end of
    // seed PLY) / mesh (OBJ). The background at the end of the main PLY is a sub-layer of GS.
    // `has*` = loaded (whether to show the toggle), `show*` = visibility (the renderer reads it
    // every draw).
    @Published private(set) var hasGS = false
    @Published private(set) var hasBackground = false
    @Published private(set) var hasSeed = false
    @Published private(set) var hasCameraMarkers = false
    @Published private(set) var hasMesh = false
    @Published private(set) var meshTriangleCount = 0
    @Published var showGS = true { didSet { onNeedsRedraw?() } }
    /// The far background shell (the trained SH background baked into Gaussians). Drawn only while
    /// GS is shown too.
    @Published var showBackground = true { didSet { onNeedsRedraw?() } }
    @Published var showSeed = false { didSet { onNeedsRedraw?() } }
    @Published var showCameraMarkers = false { didSet { onNeedsRedraw?() } }
    /// The mesh is drawn opaque and splats overlap only in front of it. Visible splats use sorted
    /// compositing.
    @Published var showMesh = false { didSet { onNeedsRedraw?() } }
    /// true draws the mesh gray + lit from the camera direction to show shape. false uses vertex
    /// colors.
    @Published var meshShaded = false { didSet { onNeedsRedraw?() } }

    /// After loading, settles which layers exist and their initial visibility.
    func setLayers(
        hasGS: Bool, hasBackground: Bool, hasSeed: Bool, hasCameras: Bool, meshTriangleCount: Int?,
        showGS: Bool, showSeed: Bool, showCameras: Bool, showMesh: Bool
    ) {
        self.hasGS = hasGS
        self.hasBackground = hasBackground
        self.hasSeed = hasSeed
        self.hasCameraMarkers = hasCameras
        self.hasMesh = meshTriangleCount != nil
        self.meshTriangleCount = meshTriangleCount ?? 0
        self.showGS = showGS
        self.showSeed = showSeed
        self.showCameraMarkers = showCameras
        self.showMesh = showMesh
    }

    // MARK: CAD model

    /// A CAD model (STEP) is loaded. It is its own layer, apart from the mesh reconstructed from the
    /// capture, and can be moved and turned by hand to sit it in the scene.
    @Published private(set) var hasCAD = false
    @Published var showCAD = true {
        didSet {
            if !showCAD { isCADSelected = false }
            onNeedsRedraw?()
        }
    }
    /// Why the CAD model didn't load, when the rest of the scene did.
    @Published private(set) var cadLoadError: String?
    /// Tapping or dragging the CAD model selects it: it is tinted, and a box around it shows a
    /// small ring round each of its axes that turns it (`SplatViewerCADGizmo`), and buttons for
    /// quarter turns show (`SplatViewerCADTurn`). Dragging the model moves it, settling on floors
    /// within reach (`SplatViewerCADSurface`). Tapping elsewhere, or another mode, lets go of it.
    @Published var isCADSelected = false { didSet { onNeedsRedraw?() } }
    /// Placement of the CAD model: a turn about its own center (`cadPivot`), then a shift.
    var cadOffset = SIMD3<Float>(0, 0, 0) { didSet { onNeedsRedraw?() } }
    var cadRotation = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1) {
        didSet {
            updateCADTurned()
            onNeedsRedraw?()
        }
    }
    /// The CAD model is turned from the way it was imported, upright in the viewer (what Reset
    /// rotation undoes).
    @Published private(set) var isCADTurned = false
    /// Center and half sizes of the model's bounding box in its file coordinates.
    private(set) var cadPivot = SIMD3<Float>(0, 0, 0)
    private(set) var cadHalfExtent = SIMD3<Float>(repeating: 0.5)
    /// Outline, move handle and rings of the selected CAD model on screen.
    let cadProjection = SplatCropProjection()
    /// Told the placement after each move or turn, to keep it for the next time the scene is opened.
    var onCADPlacementChange: ((SplatViewerCADPlacement) -> Void)?
    /// A CAD model picked with Import is being read.
    @Published private(set) var isLoadingCAD = false
    /// Loads a CAD model into the open scene (replacing any). Set by the renderer.
    var onLoadCAD: ((URL) -> Void)?
    /// A quarter-turn button for the selected CAD model. Set by the renderer.
    var onTurnCAD: ((SplatViewerCADTurn) -> Void)?
    /// Takes the CAD model out of the scene. Set by the renderer.
    var onRemoveCAD: (() -> Void)?
    /// A two-finger twist or trackpad rotation of the selected CAD model, in radians clockwise on
    /// screen since the last call, and its end. Set by the renderer.
    var onTwistCAD: ((Float) -> Void)?
    var onTwistCADEnd: (() -> Void)?

    /// The CAD model's rotation when it stands as in its file (Y up) in the viewer: undoing the
    /// view's turn.
    var cadUprightRotation: simd_quatf { viewRotation.inverse }

    private func updateCADTurned() {
        let turned = abs(abs(simd_dot(cadRotation.vector, cadUprightRotation.vector)) - 1) > 1e-6
        if turned != isCADTurned { isCADTurned = turned }
    }

    func setLoadingCAD(_ loading: Bool) {
        isLoadingCAD = loading
        if loading { cadLoadError = nil }
    }

    func setCADLoadError(_ message: String) {
        cadLoadError = message
    }

    var cadPlacement: SplatViewerCADPlacement {
        SplatViewerCADPlacement(offset: cadOffset, rotation: cadRotation)
    }

    /// CAD file coordinates to world.
    var cadTransform: simd_float4x4 {
        func shift(_ v: SIMD3<Float>) -> simd_float4x4 {
            simd_float4x4(SIMD4(1, 0, 0, 0), SIMD4(0, 1, 0, 0), SIMD4(0, 0, 1, 0), SIMD4(v, 1))
        }
        return shift(cadOffset + cadPivot) * simd_float4x4(cadRotation) * shift(-cadPivot)
    }

    /// The model's bounding box as a Crop box, so it is handled the same way.
    var cadRegion: SplatViewerCropRegion {
        SplatViewerCropRegion(
            shape: .box, center: cadPivot + cadOffset, halfExtent: cadHalfExtent, rotation: cadRotation, radius: 1)
    }

    /// Places the model where a dragged `cadRegion` now is.
    func setCADRegion(_ region: SplatViewerCropRegion) {
        cadOffset = region.center - cadPivot
        cadRotation = region.rotation
    }

    /// Takes the CAD model out of the scene (the renderer drops its layer).
    func clearCAD() {
        isCADSelected = false
        hasCAD = false
        cadLoadError = nil
        cadOffset = .zero
        cadRotation = cadUprightRotation
    }

    /// Sets up the CAD layer from its bounding box, starting from a saved placement or else shifted
    /// by `offset` and standing upright in the viewer.
    func setCAD(
        pivot: SIMD3<Float>, halfExtent: SIMD3<Float>, offset: SIMD3<Float>, saved: SplatViewerCADPlacement? = nil
    ) {
        hasCAD = true
        cadPivot = pivot
        cadHalfExtent = pointwiseMax(halfExtent, SIMD3(repeating: SplatViewerCropRegion.minimumSize))
        cadOffset = saved?.offset ?? offset
        cadRotation = saved.map { simd_quatf(vector: $0.rotation) } ?? cadUprightRotation
    }

    // MARK: Orient

    /// The turn that stands the splats' file upright (`SplatViewerFileAxes`): a half turn about Z for
    /// the usual 3DGS axes, none for a capture written Y-up. Set on loading.
    private(set) var fileRotation = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)

    /// Turn from the scene's coordinates to the viewer's (`SplatViewerCamera.sceneRotation`): the
    /// file's, then Orient's.
    var viewRotation: simd_quatf { sceneRotation * fileRotation }

    func setFileRotation(_ rotation: simd_quatf) {
        fileRotation = rotation
        updateCADTurned()
        onNeedsRedraw?()
    }

    /// Orient's turn, on top of the file's. Changed by Orient's buttons, which animate it.
    private(set) var sceneRotation = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1) {
        didSet {
            let turned = abs(abs(sceneRotation.real) - 1) > 1e-6
            if turned != isSceneTurned { isSceneTurned = turned }
            updateCADTurned()
            onNeedsRedraw?()
        }
    }
    /// The scene is turned from the way its file has it (what Orient's Reset undoes).
    @Published private(set) var isSceneTurned = false
    /// Orient mode: a band of buttons that turn the whole scene by quarter turns. Turning it on turns
    /// the other modes off.
    @Published var isOrienting = false {
        didSet {
            if isOrienting {
                isMeasuring = false; isCropping = false; isColoring = false; isRemovingFloaters = false
                isCADSelected = false
            }
        }
    }
    /// Told the scene rotation after each turn, to keep it for the next time the scene is opened.
    var onSceneRotationChange: ((SplatViewerSceneRotation) -> Void)?
    /// Where the turn in progress ends; the next press turns on from there.
    private var sceneRotationTarget: simd_quatf?
    private var sceneTurnID = 0
    static let sceneTurnSeconds: Double = 0.25

    /// Turns the scene a quarter turn as seen from the camera now, about the point it orbits.
    func turnScene(_ turn: SplatViewerSceneTurn) {
        let from = sceneRotationTarget ?? sceneRotation
        animateScene(to: turn.applied(to: from, azimuth: azimuth, elevation: elevation))
    }

    /// Back to the way the scene's file has it.
    func resetSceneRotation() {
        animateScene(to: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1))
    }

    /// Turns the scene to `target` over `sceneTurnSeconds`, so it is seen to turn, and tells
    /// `onSceneRotationChange` the end straight away.
    private func animateScene(to target: simd_quatf) {
        let start = sceneRotation
        sceneRotationTarget = target
        sceneTurnID += 1
        let id = sceneTurnID
        onSceneRotationChange?(SplatViewerSceneRotation(target))
        Task { [weak self] in
            let begin = ContinuousClock.now
            while true {
                guard let self, self.sceneTurnID == id else { return }
                let elapsed = (ContinuousClock.now - begin).components
                let seconds = Double(elapsed.seconds) + Double(elapsed.attoseconds) * 1e-18
                let t = Float(min(1, seconds / Self.sceneTurnSeconds))
                // Ease in and out, so the turn reads as one motion.
                let eased = t * t * (3 - 2 * t)
                self.sceneRotation = t >= 1 ? target : simd_slerp(start, target, eased)
                if t >= 1 {
                    self.sceneRotationTarget = nil
                    return
                }
                try? await Task.sleep(for: .milliseconds(16))
            }
        }
    }

    // MARK: Measurement

    /// Measure mode. While on, taps place measurement points and the placed segments are shown
    /// (rotation etc. still work). Turning it off keeps the points, and they are shown again the
    /// next time it is turned on.
    @Published var isMeasuring = false {
        didSet {
            if isMeasuring { isCropping = false; isColoring = false; isRemovingFloaters = false; isCADSelected = false; isOrienting = false }
            onNeedsRedraw?()
        }
    }
    /// Placed measurement points (world coordinates; meters for capture data). Each pair is one
    /// segment; with an odd count the last is just a start point.
    @Published private(set) var measurePoints: [SIMD3<Float>] = [] { didSet { onNeedsRedraw?() } }
    /// Transient hint, e.g. when there was no surface where you tapped.
    @Published private(set) var measureNotice: String?
    /// Positions of the measurement lines overlaid on screen (updated by the renderer on every
    /// draw).
    let measureProjection = SplatMeasureProjection()
    /// Receiver of taps (normalized coordinates): measurement points in measure mode, floater
    /// selection in floater mode. Set by the renderer.
    var onTap: ((SIMD2<Float>) -> Void)?
    private var measureNoticeID = 0

    func addMeasurePoint(_ point: SIMD3<Float>) {
        measurePoints.append(point)
        measureNotice = nil
    }

    func undoMeasurePoint() {
        _ = measurePoints.popLast()
    }

    func clearMeasurements() {
        measurePoints.removeAll()
    }

    /// Shows the hint for just 2 seconds.
    func showMeasureNotice(_ text: String) {
        measureNoticeID += 1
        let id = measureNoticeID
        measureNotice = text
        Task { [weak self] in
            try? await Task.sleep(for: .seconds(2))
            guard let self, self.measureNoticeID == id else { return }
            self.measureNotice = nil
        }
    }

    // MARK: Crop

    /// Crop mode. Handles of the region can be dragged; a drag that doesn't start on a handle still
    /// orbits. Turning it on turns measure mode off.
    @Published var isCropping = false {
        didSet {
            if isCropping { isMeasuring = false; isColoring = false; isRemovingFloaters = false; isCADSelected = false; isOrienting = false }
            if !isCropping { cropProjection.setLasso([]) }
            onCropSelectionRefresh?()
            onNeedsRedraw?()
        }
    }
    /// Shape of the region. Switching keeps its center and size (see `setCropShape`).
    @Published private(set) var cropShape: SplatViewerCropShape = .box
    /// true keeps the splats inside the region; false removes them.
    @Published var cropKeepsInside = true
    /// Colour panel open. Measure, Color, Crop and Floaters are never on together: opening one
    /// closes the others, so only one panel sits over the view.
    @Published var isColoring = false {
        didSet {
            if isColoring { isMeasuring = false; isCropping = false; isRemovingFloaters = false; isCADSelected = false; isOrienting = false }
            onNeedsRedraw?()
        }
    }
    /// Region in world coordinates. Written by the renderer while a handle is dragged.
    var cropRegion: SplatViewerCropRegion? { didSet { onNeedsRedraw?() } }
    /// Whether a crop has been applied (the cropped result differs from the loaded file).
    @Published private(set) var cropEdited = false
    @Published private(set) var cropCanUndo = false
    /// Share entry for the cropped splats (PLY). nil until a crop is applied.
    @Published private(set) var cropShareItem: SplatViewerShareItem?
    /// Transient hint, e.g. when a crop would remove nothing or everything. The Floaters panel
    /// shows it too.
    @Published private(set) var cropNotice: String?
    let cropProjection = SplatCropProjection()
    /// Drag of a crop handle (normalized coordinates). The begin returns whether a handle was grabbed.
    /// Set by the renderer.
    var onCropHandleBegin: ((SIMD2<Float>) -> Bool)?
    var onCropHandleDrag: ((SIMD2<Float>) -> Void)?
    var onCropHandleEnd: (() -> Void)?
    var onCropApply: (() -> Void)?
    /// Undo and Reset step through every edit to the kept splats, crops and floater deletes alike.
    var onCropUndo: (() -> Void)?
    var onCropRestore: (() -> Void)?
    /// The orange selection of Crop has to be worked out again: Crop opened or closed, or its shape
    /// changed.
    var onCropSelectionRefresh: (() -> Void)?
    private var cropNoticeID = 0

    /// Changes the shape of the region, keeping its center and size.
    func setCropShape(_ shape: SplatViewerCropShape) {
        guard shape != cropShape else { return }
        cropShape = shape
        cropRegion = cropRegion?.converted(to: shape)
        cropProjection.setLasso([])
        onCropSelectionRefresh?()
    }

    func setCropState(edited: Bool, canUndo: Bool, shareItem: SplatViewerShareItem?) {
        cropEdited = edited
        cropCanUndo = canUndo
        cropShareItem = shareItem
    }

    /// Shows the hint for just 2 seconds.
    func showCropNotice(_ text: String) {
        cropNoticeID += 1
        let id = cropNoticeID
        cropNotice = text
        Task { [weak self] in
            try? await Task.sleep(for: .seconds(2))
            guard let self, self.cropNoticeID == id else { return }
            self.cropNotice = nil
        }
    }

    // MARK: Floaters

    /// Floater mode. A tap selects the splats within the brush (a tap on the selection lets them
    /// go); Detect selects the small clusters apart from the scene; Delete removes the selection.
    /// The selection is drawn highlighted and is dropped when the mode ends.
    @Published var isRemovingFloaters = false {
        didSet {
            guard isRemovingFloaters != oldValue else { return }
            if isRemovingFloaters { isMeasuring = false; isCropping = false; isColoring = false; isCADSelected = false; isOrienting = false }
            onFloaterModeChange?(isRemovingFloaters)
            onNeedsRedraw?()
        }
    }
    /// Splats in the selection.
    @Published private(set) var floaterSelectionCount = 0
    /// Radius of the brush a tap selects with, as a fraction of the scene's size ("Selection size").
    @Published var floaterBrushSize = SplatViewerFloaterBrush.defaultSize { didSet { onNeedsRedraw?() } }
    /// The brush circle is on screen: while Selection size is being changed, and a moment after.
    @Published private(set) var isShowingBrush = false { didSet { onNeedsRedraw?() } }
    /// Position and size of the brush circle (updated by the renderer on every draw).
    let brushProjection = SplatBrushProjection()
    private var brushHideID = 0
    /// Detect is working out the clusters (the first Detect after the splats changed).
    @Published private(set) var isFindingFloaters = false
    /// Set by the renderer.
    var onFloaterModeChange: ((Bool) -> Void)?
    var onFloaterDetect: (() -> Void)?
    var onFloaterDeselectAll: (() -> Void)?
    var onFloaterDelete: (() -> Void)?
    /// Takes back the last selection change or delete in floater mode.
    var onFloaterUndo: (() -> Void)?
    @Published private(set) var floaterCanUndo = false

    /// Shows the brush circle while Selection size is being changed, and hides it a moment after.
    func setBrushSizeEditing(_ editing: Bool) {
        brushHideID += 1
        guard !editing else {
            isShowingBrush = true
            return
        }
        let id = brushHideID
        Task { [weak self] in
            try? await Task.sleep(for: .seconds(0.8))
            guard let self, self.brushHideID == id else { return }
            self.isShowingBrush = false
        }
    }

    func setFloaterState(selectionCount: Int, isFinding: Bool, canUndo: Bool) {
        floaterSelectionCount = selectionCount
        isFindingFloaters = isFinding
        floaterCanUndo = canUndo
    }

    // orbit camera (written by gestures, read by the renderer). `framed*` are the values at framing
    // time.
    var azimuth: Float = 0 { didSet { onNeedsRedraw?() } }
    var elevation: Float = 0 { didSet { onNeedsRedraw?() } }
    var distance: Float = 4 { didSet { onNeedsRedraw?() } }
    // Look-at point. Pan moves it within the screen plane; double tap moves it to the tapped
    // surface.
    var center = SIMD3<Float>(0, 0, 0) { didSet { onNeedsRedraw?() } }
    /// Receiver of double taps (normalized coordinates). Set by the renderer.
    var onFocusTap: ((SIMD2<Float>) -> Void)?

    /// Formats and conversions offered for a cropped result. Set by the screen that made the viewer.
    let cropExport: SplatViewerCropExport?
    init(cropExport: SplatViewerCropExport? = nil, sceneRotation: SplatViewerSceneRotation? = nil) {
        self.cropExport = cropExport
        // Observers don't run in an initializer.
        if let sceneRotation {
            self.sceneRotation = sceneRotation.quaternion
            isSceneTurned = abs(abs(self.sceneRotation.real) - 1) > 1e-6
        }
    }

    /// The colour grade on the foreground. The renderer bakes it in through `onColorGradeApply`.
    @Published private(set) var colorGrade = SplatViewerColorGrade.identity
    var onColorGradeApply: ((SplatViewerColorGrade) -> Void)?

    func applyColorGrade(_ grade: SplatViewerColorGrade) {
        guard grade != colorGrade else { return }
        colorGrade = grade
        onColorGradeApply?(grade)
    }

    private(set) var framedDistance: Float = 4
    private(set) var framedCenter = SIMD3<Float>(0, 0, 0)
    private(set) var previewCamera: SplatViewerPreviewCamera?
    private(set) var framedFovyRadians: Float = SplatViewerModel.defaultFovyRadians
    var fovyRadians: Float = SplatViewerModel.defaultFovyRadians { didSet { onNeedsRedraw?() } }

    static let defaultFovyRadians: Float = 60 * .pi / 180

    func setLoadState(_ state: LoadState) { loadState = state }

    /// Records the auto-framing computed from the loaded splats.
    func frame(center: SIMD3<Float>, distance: Float, previewCamera: SplatViewerPreviewCamera?) {
        self.previewCamera = previewCamera
        self.framedCenter = previewCamera?.center ?? center
        if let previewCamera {
            let angles = previewCamera.orbitAngles(sceneRotation: viewRotation)
            self.azimuth = angles.azimuth
            self.elevation = angles.elevation
            self.distance = previewCamera.distance
            self.center = previewCamera.center
            self.fovyRadians = previewCamera.fovyRadians
            self.framedDistance = previewCamera.distance
            self.framedFovyRadians = previewCamera.fovyRadians
        } else {
            self.azimuth = 0
            self.elevation = 0
            self.distance = distance
            self.center = center
            self.fovyRadians = Self.defaultFovyRadians
            self.framedDistance = distance
            self.framedFovyRadians = Self.defaultFovyRadians
        }
    }

    /// Resets the camera to the initial auto-framing pose.
    func resetCamera() {
        if let previewCamera {
            let angles = previewCamera.orbitAngles(sceneRotation: viewRotation)
            azimuth = angles.azimuth
            elevation = angles.elevation
        } else {
            azimuth = 0
            elevation = 0
        }
        distance = framedDistance
        center = framedCenter
        fovyRadians = framedFovyRadians
    }
}

#endif // os(iOS) || os(macOS)

/// Where a CAD model sits in the scene, relative to its file coordinates (metres, Y up): a turn
/// about its center, then a shift. Saved by the app beside the model.
public struct SplatViewerCADPlacement: Codable, Equatable, Sendable {
    public var offset: SIMD3<Float>
    /// Quaternion (ix, iy, iz, r).
    public var rotation: SIMD4<Float>

    public init(offset: SIMD3<Float>, rotation: simd_quatf) {
        self.offset = offset
        self.rotation = rotation.vector
    }
}
