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

    static func load(
        siblingOf url: URL,
        center: SIMD3<Float>,
        defaultDistance: Float
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

        let forward = Self.cameraForwardWorld(fromTrainingViewmat: metadata.viewmat)
        let eye = Self.cameraCenterWorld(fromTrainingViewmat: metadata.viewmat)
        guard let forward, let eye else { return nil }

        let projectedDistance = simd_dot(center - eye, forward)
        let distance = projectedDistance.isFinite && projectedDistance > 0.05
            ? projectedDistance
            : defaultDistance
        let angles = Self.orbitAngles(lookingToward: forward)
        return SplatViewerPreviewCamera(
            azimuth: angles.azimuth,
            elevation: angles.elevation,
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

    // MARK: Measurement

    /// Measure mode. While on, taps place measurement points and the placed segments are shown
    /// (rotation etc. still work). Turning it off keeps the points, and they are shown again the
    /// next time it is turned on.
    @Published var isMeasuring = false {
        didSet {
            if isMeasuring { isCropping = false; isRemovingFloaters = false }
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
            if isCropping { isMeasuring = false; isColoring = false; isRemovingFloaters = false }
            if !isCropping { cropProjection.setLasso([]) }
            onCropSelectionRefresh?()
            onNeedsRedraw?()
        }
    }
    /// Shape of the region. Switching keeps its center and size (see `setCropShape`).
    @Published private(set) var cropShape: SplatViewerCropShape = .box
    /// true keeps the splats inside the region; false removes them.
    @Published var cropKeepsInside = true
    /// Colour panel open. It, Crop and Floaters are never on together: opening one closes the others.
    @Published var isColoring = false {
        didSet {
            if isColoring { isCropping = false; isRemovingFloaters = false }
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
            if isRemovingFloaters { isMeasuring = false; isCropping = false; isColoring = false }
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
    init(cropExport: SplatViewerCropExport? = nil) {
        self.cropExport = cropExport
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
            self.azimuth = previewCamera.azimuth
            self.elevation = previewCamera.elevation
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
            azimuth = previewCamera.azimuth
            elevation = previewCamera.elevation
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
