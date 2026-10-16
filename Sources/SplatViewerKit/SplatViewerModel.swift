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
            fovyRadians: fovy
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
    @Published var isMeasuring = false { didSet { onNeedsRedraw?() } }
    /// Placed measurement points (world coordinates; meters for capture data). Each pair is one
    /// segment; with an odd count the last is just a start point.
    @Published private(set) var measurePoints: [SIMD3<Float>] = [] { didSet { onNeedsRedraw?() } }
    /// Transient hint, e.g. when there was no surface where you tapped.
    @Published private(set) var measureNotice: String?
    /// Positions of the measurement lines overlaid on screen (updated by the renderer on every
    /// draw).
    let measureProjection = SplatMeasureProjection()
    /// Receiver of taps (normalized coordinates) in measure mode. Set by the renderer.
    var onMeasureTap: ((SIMD2<Float>) -> Void)?
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
    private(set) var framedDistance: Float = 4
    private(set) var framedCenter = SIMD3<Float>(0, 0, 0)
    private(set) var previewCamera: SplatViewerPreviewCamera?
    private(set) var framedFovyRadians: Float = SplatViewerModel.defaultFovyRadians
    var fovyRadians: Float = SplatViewerModel.defaultFovyRadians { didSet { onNeedsRedraw?() } }

    static let defaultFovyRadians: Float = 60 * .pi / 180

    func setLoadState(_ state: LoadState) { loadState = state }

    /// Records the auto-framing computed from the loaded splats.
    func frame(center: SIMD3<Float>, distance: Float, previewCamera: SplatViewerPreviewCamera?) {
        self.framedCenter = center
        self.previewCamera = previewCamera
        if let previewCamera {
            self.azimuth = previewCamera.azimuth
            self.elevation = previewCamera.elevation
            self.distance = previewCamera.distance
            self.center = center
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
