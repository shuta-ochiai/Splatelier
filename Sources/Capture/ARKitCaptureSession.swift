// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// ARKit capture (anchor-based capture). An ARAnchor is placed for each accepted frame; stopCapture
// waits settleSeconds to pick up late anchor updates, then builds viewmats from the anchors' final
// transforms, because loop closure only updates ARAnchor.transform, not ARFrame.camera.transform.
// State is mutated only on a private serial delegate queue and exposed as a lock-guarded snapshot.
// Pixel buffer conversion runs synchronously on the delegate queue and ARFrames are never retained
// (holding them makes ARKit stop delivering frames). During capture frames are kept as [UInt8];
// they become MLXArrays only in stopCapture. Feature points are recorded in anchor-local
// coordinates and mapped back to world in stopCapture to form sparsePoints.

import Foundation
import MLX
import simd
import Trainer

#if canImport(ARKit) && os(iOS)
import ARKit
import AVFoundation
#endif

public enum CaptureError: Error, Sendable, CustomStringConvertible {
    case unavailableOnPlatform
    case alreadyRunning
    case notRunning
    case noFramesCaptured
    case sessionFailed(String)
    case pixelBufferConversion(String)

    public var description: String {
        switch self {
        case .unavailableOnPlatform: return "ARKit capture is only available on iOS"
        case .alreadyRunning: return "ARKitCaptureSession is already running"
        case .notRunning: return "ARKitCaptureSession is not running"
        case .noFramesCaptured: return "no frames captured (tracking never reached .normal?)"
        case .sessionFailed(let s): return "ARSession failed: \(s)"
        case .pixelBufferConversion(let s): return "pixel buffer conversion failed: \(s)"
        }
    }
}

/// Source of the live point cloud during capture (coverage overlay).
public enum LiveReconstructionMode: String, Sendable, CaseIterable {
    /// ARKit feature points (rawFeaturePoints). Sparse, but no extra cost.
    case featurePoints
    /// Dense points from unprojecting LiDAR sceneDepth. LiDAR devices only.
    case lidar
    /// Incremental MVS during capture (`LiveMVSReconstructor`, same method as MVS mode). Dense with
    /// the camera alone. rawValue keeps the old name so values saved by the old Mobile3DRecon
    /// implementation still load.
    case mvs = "mobile3DRecon"
}

/// Capture settings.
public struct CaptureConfiguration: Sendable {
    /// Minimum translation between accepted frames (m). Matches FrameSubsampler's τ_d.
    public var minMotionMeters: Float
    /// Minimum rotation between accepted frames (rad). ORed with the translation condition.
    public var minRotationRadians: Float
    /// Maximum frame count. Beyond it no anchors are added; only updates to existing anchors are
    /// taken.
    public var maxFrames: Int
    /// Drop candidates close in both position and angle to any existing frame. Comparing only
    /// against the previous frame lets redundant frames pile up when orbiting back to the same
    /// viewpoint, which breaks memory and training.
    public var cullRedundantViewpoints: Bool
    /// Make the acceptance spacing proportional to subject distance (constant baseline/depth). Uses
    /// clamp(keyframeDepthFactor × median feature-point distance, floor, ceil) instead of
    /// minMotionMeters.
    public var adaptiveKeyframeSpacing: Bool
    /// Coefficient of the distance-linked spacing (baseline/depth ratio).
    public var keyframeDepthFactor: Float
    /// Lower bound of the distance-linked spacing (m).
    public var keyframeSpacingFloor: Float
    /// Upper bound of the distance-linked spacing (m).
    public var keyframeSpacingCeil: Float
    /// Max long side (px) of saved/training frames. The app passes 1080.
    public var targetMaxDim: Int
    /// Time to keep the session running after stopCapture to pick up late anchor updates.
    public var settleSeconds: TimeInterval
    /// Do not accept frames while tracking is not `.normal`.
    public var requireNormalTracking: Bool
    /// Collect LiDAR sceneDepth into lidarPoints. No-op on devices without LiDAR.
    public var enableLiDARDepth: Bool
    /// Minimum LiDAR confidence (0=low / 1=medium / 2=high).
    /// An Int so ARConfidenceLevel is not exposed in the public type.
    public var lidarConfidenceMinRaw: Int
    /// Subsampling stride (px) for the LiDAR depthMap.
    public var lidarStride: Int
    /// Run the image quality gate right after conversion. Rejected frames are neither saved nor
    /// shown in coverage.
    public var enableRealtimeQualityGate: Bool
    public var realtimeQualityGate: CaptureRealtimeFrameSelector.Parameters
    /// Lock AE/AWB during capture. Per-frame brightness/color drift gets averaged by the
    /// photometric loss, dulling colors and producing floaters. iOS 16+ only (otherwise stays
    /// auto). Focus stays auto.
    public var lockExposureAndWhiteBalance: Bool
    /// Delay (s) from reaching tracking=.normal to locking AE/AWB, since AE/AWB has not converged
    /// at startup.
    public var exposureLockWarmupSeconds: TimeInterval
    /// Max exposure time (s) when locking AE/AWB. If the AE value exceeds it, switch to custom
    /// exposure and compensate brightness with ISO, since for 3DGS motion blur hurts more than
    /// noise. Blur remained at 1/120s. nil locks the AE value as is.
    public var maxShutterDuration: TimeInterval?
    public var liveReconstructionMode: LiveReconstructionMode

    public init(
        minMotionMeters: Float = 0.05,        // PocketGS τ_d
        minRotationRadians: Float = 0.087,    // ~5°
        maxFrames: Int = 300,
        cullRedundantViewpoints: Bool = true,
        adaptiveKeyframeSpacing: Bool = true,
        keyframeDepthFactor: Float = 0.07,
        keyframeSpacingFloor: Float = 0.05,
        keyframeSpacingCeil: Float = 0.4,
        targetMaxDim: Int = 768,
        settleSeconds: TimeInterval = 0.5,
        requireNormalTracking: Bool = true,
        enableLiDARDepth: Bool = true,
        lidarConfidenceMinRaw: Int = 1,        // .medium
        lidarStride: Int = 8,
        enableRealtimeQualityGate: Bool = true,
        realtimeQualityGate: CaptureRealtimeFrameSelector.Parameters = .init(),
        lockExposureAndWhiteBalance: Bool = true,
        exposureLockWarmupSeconds: TimeInterval = 1.0,
        maxShutterDuration: TimeInterval? = 1.0 / 250.0,
        liveReconstructionMode: LiveReconstructionMode = .featurePoints
    ) {
        precondition(minMotionMeters >= 0)
        precondition(minRotationRadians >= 0)
        precondition(maxFrames > 0)
        precondition(targetMaxDim > 0)
        precondition(settleSeconds >= 0)
        precondition(lidarConfidenceMinRaw >= 0 && lidarConfidenceMinRaw <= 2)
        precondition(lidarStride > 0)
        precondition(exposureLockWarmupSeconds >= 0)
        self.minMotionMeters = minMotionMeters
        self.minRotationRadians = minRotationRadians
        self.maxFrames = maxFrames
        self.cullRedundantViewpoints = cullRedundantViewpoints
        self.adaptiveKeyframeSpacing = adaptiveKeyframeSpacing
        self.keyframeDepthFactor = keyframeDepthFactor
        self.keyframeSpacingFloor = keyframeSpacingFloor
        self.keyframeSpacingCeil = keyframeSpacingCeil
        self.targetMaxDim = targetMaxDim
        self.settleSeconds = settleSeconds
        self.requireNormalTracking = requireNormalTracking
        self.enableLiDARDepth = enableLiDARDepth
        self.lidarConfidenceMinRaw = lidarConfidenceMinRaw
        self.lidarStride = lidarStride
        self.enableRealtimeQualityGate = enableRealtimeQualityGate
        var gate = realtimeQualityGate
        gate.enabled = enableRealtimeQualityGate
        self.realtimeQualityGate = gate
        self.lockExposureAndWhiteBalance = lockExposureAndWhiteBalance
        self.exposureLockWarmupSeconds = exposureLockWarmupSeconds
        self.maxShutterDuration = maxShutterDuration
        self.liveReconstructionMode = liveReconstructionMode
    }
}

/// Snapshot of capture progress and observations (polled by the UI).
public struct CaptureDiagnostics: Sendable {
    public var keptFrameCount: Int
    public var trackingStateDescription: String
    /// Whether relocalization was observed (the pose history jumps significantly).
    public var relocalizationOccurred: Bool
    /// Cumulative anchor update count (a rough gauge of loop closure strength).
    public var anchorUpdateCount: Int
    /// Number of anchors still at their capture-time transform (never updated) at stopCapture.
    public var unsettledAnchorCount: Int
    /// Whether LiDAR sceneDepth is active (device supports it and enableLiDARDepth).
    public var lidarSupported: Bool
    /// Frames that contributed LiDAR points. 0 with lidarSupported suggests confidence dropped
    /// every pixel.
    public var lidarFrameCount: Int
    /// Cumulative LiDAR world points (after filtering and stride).
    public var lidarPointCount: Int
    /// Frames evaluated by the quality gate (those that passed the motion gate and conversion).
    public var qualityCandidateFrameCount: Int
    /// Frames rejected by the quality gate.
    public var qualityRejectedFrameCount: Int
    public var qualityRejectedBlurCount: Int
    public var qualityRejectedExposureCount: Int
    /// Whether the AE/AWB lock actually fired. false with the setting on means auto exposure
    /// remained (suspect color shifts between frames).
    public var exposureLocked: Bool
    /// Color tags of the first converted capture buffer (for the record). Conversion declares the
    /// input as sRGB, so no tone conversion from the transfer tag (ITU_R_709_2 on device) is
    /// applied.
    public var pixelBufferColorTags: String?
    /// Processing stats of the `.mvs` live point cloud (frames processed, time per frame, frames
    /// skipped while busy). nil for other modes.
    public var liveMVSSummary: String?

    public init(
        keptFrameCount: Int = 0,
        trackingStateDescription: String = "idle",
        relocalizationOccurred: Bool = false,
        anchorUpdateCount: Int = 0,
        unsettledAnchorCount: Int = 0,
        lidarSupported: Bool = false,
        lidarFrameCount: Int = 0,
        lidarPointCount: Int = 0,
        qualityCandidateFrameCount: Int = 0,
        qualityRejectedFrameCount: Int = 0,
        qualityRejectedBlurCount: Int = 0,
        qualityRejectedExposureCount: Int = 0,
        exposureLocked: Bool = false,
        pixelBufferColorTags: String? = nil,
        liveMVSSummary: String? = nil
    ) {
        self.keptFrameCount = keptFrameCount
        self.trackingStateDescription = trackingStateDescription
        self.relocalizationOccurred = relocalizationOccurred
        self.anchorUpdateCount = anchorUpdateCount
        self.unsettledAnchorCount = unsettledAnchorCount
        self.lidarSupported = lidarSupported
        self.lidarFrameCount = lidarFrameCount
        self.lidarPointCount = lidarPointCount
        self.qualityCandidateFrameCount = qualityCandidateFrameCount
        self.qualityRejectedFrameCount = qualityRejectedFrameCount
        self.qualityRejectedBlurCount = qualityRejectedBlurCount
        self.qualityRejectedExposureCount = qualityRejectedExposureCount
        self.exposureLocked = exposureLocked
        self.pixelBufferColorTags = pixelBufferColorTags
        self.liveMVSSummary = liveMVSSummary
    }
}

public enum CaptureState: Sendable {
    case idle
    case running
    case stopping
    case stopped
}

/// Capture session. Symbols exist on all platforms so the ViewModel can be shared; non-iOS throws
/// `CaptureError.unavailableOnPlatform`.
public final class ARKitCaptureSession {

    private let configuration: CaptureConfiguration
    /// Source of the live point cloud during capture (used by the overlay to switch its display).
    public var liveReconstructionMode: LiveReconstructionMode { configuration.liveReconstructionMode }
    private let stateLock = NSLock()
    private var _state: CaptureState = .idle
    private var _diagnostics = CaptureDiagnostics()

    /// Whether LiDAR sceneDepth is available. Lets the UI branch without the app importing ARKit.
    public static var isLiDARAvailable: Bool {
        #if canImport(ARKit) && os(iOS)
        return ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
        #else
        return false
        #endif
    }

    public init(configuration: CaptureConfiguration = CaptureConfiguration()) {
        self.configuration = configuration
        #if canImport(ARKit) && os(iOS)
        do { self.iOSImpl = try IOSImpl(configuration: configuration) }
        catch { self.iOSImpl = nil; self.initError = error }
        #endif
    }

    public var state: CaptureState {
        stateLock.lock(); defer { stateLock.unlock() }
        return _state
    }

    public var diagnostics: CaptureDiagnostics {
        stateLock.lock(); defer { stateLock.unlock() }
        return _diagnostics
    }

    #if canImport(ARKit) && os(iOS)
    /// For the live preview (ARSCNView). ARSCNView does not take over session.delegate, so it
    /// coexists with the capture delegate.
    public var arSession: ARSession? { iOSImpl?.session }
    #endif

    /// Number of splats in the live overlay (surfels for `.mvs`, coverage splats otherwise).
    public var liveSplatCount: Int {
        #if canImport(ARKit) && os(iOS)
        return iOSImpl?.liveSplatCount ?? 0
        #else
        return 0
        #endif
    }

    /// Live coverage splats for the capture guide.
    public var liveCoverageSplats: [CaptureCoverageSplat] {
        #if canImport(ARKit) && os(iOS)
        return iOSImpl?.liveCoverageSplatsSnapshot ?? []
        #else
        return []
        #endif
    }

    public var liveCoverageRevision: Int {
        #if canImport(ARKit) && os(iOS)
        return iOSImpl?.liveCoverageRevision ?? 0
        #else
        return 0
        #endif
    }

    /// Display surfels during capture (`.mvs` only; drawn as splats that sharpen as you get
    /// closer).
    public var liveSurfels: [CaptureSurfelMap.Surfel] {
        #if canImport(ARKit) && os(iOS)
        return iOSImpl?.liveSurfelsSnapshot ?? []
        #else
        return []
        #endif
    }

    public var liveSurfelRevision: Int {
        #if canImport(ARKit) && os(iOS)
        return iOSImpl?.liveSurfelRevision ?? 0
        #else
        return 0
        #endif
    }

    public func start() throws {
        #if canImport(ARKit) && os(iOS)
        if let err = initError { throw err }
        guard let impl = iOSImpl else { throw CaptureError.unavailableOnPlatform }
        try setState(.running)
        try impl.start(onStateChange: { [weak self] update in
            self?.applyStateUpdate(update)
        })
        #else
        throw CaptureError.unavailableOnPlatform
        #endif
    }

    public func stopCapture() async throws -> CapturedScene {
        #if canImport(ARKit) && os(iOS)
        guard let impl = iOSImpl else { throw CaptureError.unavailableOnPlatform }
        try setState(.stopping)
        // Return to .stopped even on throw. Stuck in .stopping, the UI cannot leave the spinner.
        do {
            let scene = try await impl.stopCapture()
            try setState(.stopped)
            return scene
        } catch {
            try? setState(.stopped)
            throw error
        }
        #else
        throw CaptureError.unavailableOnPlatform
        #endif
    }

    public func cancel() {
        #if canImport(ARKit) && os(iOS)
        iOSImpl?.cancel()
        #endif
        try? setState(.stopped)
    }

    // MARK: - internal helpers (cross-platform)

    private func setState(_ next: CaptureState) throws {
        stateLock.lock(); defer { stateLock.unlock() }
        switch (_state, next) {
        case (.idle, .running), (.running, .stopping),
             (.stopping, .stopped), (.idle, .stopped), (.running, .stopped):
            _state = next
        case (.running, .running):
            throw CaptureError.alreadyRunning
        case (.idle, .stopping), (.stopped, .stopping):
            throw CaptureError.notRunning
        default:
            _state = next
        }
    }

    fileprivate func applyStateUpdate(_ update: DiagnosticsUpdate) {
        stateLock.lock(); defer { stateLock.unlock() }
        switch update {
        case .keptFrameCount(let n): _diagnostics.keptFrameCount = n
        case .trackingState(let s): _diagnostics.trackingStateDescription = s
        case .relocalization: _diagnostics.relocalizationOccurred = true
        case .anchorUpdate(let count): _diagnostics.anchorUpdateCount += count
        case .unsettledCount(let n): _diagnostics.unsettledAnchorCount = n
        case .lidarSupported(let b): _diagnostics.lidarSupported = b
        case .lidarFrame(let n): _diagnostics.lidarFrameCount = n
        case .lidarPoints(let n): _diagnostics.lidarPointCount = n
        case .exposureLocked(let b): _diagnostics.exposureLocked = b
        case .qualityCandidate: _diagnostics.qualityCandidateFrameCount += 1
        case .qualityRejected(let reasons):
            _diagnostics.qualityRejectedFrameCount += 1
            if reasons.contains(.blur) { _diagnostics.qualityRejectedBlurCount += 1 }
            if reasons.contains(.exposure) { _diagnostics.qualityRejectedExposureCount += 1 }
        }
    }

    // MARK: - iOS-only storage

    #if canImport(ARKit) && os(iOS)
    private var iOSImpl: IOSImpl?
    private var initError: Error?
    #endif
}

fileprivate enum DiagnosticsUpdate {
    case keptFrameCount(Int)
    case trackingState(String)
    case relocalization
    case anchorUpdate(Int)
    case unsettledCount(Int)
    case lidarSupported(Bool)
    case lidarFrame(Int)
    case lidarPoints(Int)
    case qualityCandidate
    case qualityRejected([CaptureRealtimeFrameSelector.RejectReason])
    case exposureLocked(Bool)
}

// MARK: - iOS implementation

#if canImport(ARKit) && os(iOS)

private final class IOSImpl: NSObject, ARSessionDelegate {

    private struct PendingFrame {
        let anchorID: UUID
        let rgbBytes: [UInt8]
        let width: Int
        let height: Int
        let intrinsics: Intrinsics
        let timestamp: TimeInterval
        // Anchor transform at capture time (used to count un-updated anchors and to detect
        // redundant views)
        let seedTransform: simd_float4x4
        // ARKit feature points seen in this frame (OpenCV camera coordinates at the capture-time
        // pose)
        let featureObservations: [ARKitFeatureObservation]
    }

    private let configuration: CaptureConfiguration
    // fileprivate: lent to the live preview
    fileprivate let session = ARSession()
    private let delegateQueue = DispatchQueue(label: "io.github.shuta-ochiai.splatelier.capture")
    private let pixelConverter: PixelBufferToBytes
    private var qualitySelector: CaptureRealtimeFrameSelector

    // For live display. Updated on delegateQueue, read from the UI through the lock.
    // Approximate display that does not reflect loop closure corrections.
    private let pointsLock = NSLock()
    private var _liveCoverageSplats: [CaptureCoverageSplat] = []
    private var _liveCoverageRevision = 0
    private var _liveSurfels: [CaptureSurfelMap.Surfel] = []
    private var _liveSurfelRevision = 0
    // Sized to also hold the dense LiDAR points
    private var coverageAccumulator = CaptureCoverageAccumulator(
        voxelSizeMeters: 0.03, maxSplats: 60_000)

    fileprivate var liveSplatCount: Int {
        pointsLock.lock(); defer { pointsLock.unlock() }
        return configuration.liveReconstructionMode == .mvs ? _liveSurfels.count : _liveCoverageSplats.count
    }

    fileprivate var liveCoverageSplatsSnapshot: [CaptureCoverageSplat] {
        pointsLock.lock(); defer { pointsLock.unlock() }
        return _liveCoverageSplats
    }

    fileprivate var liveCoverageRevision: Int {
        pointsLock.lock(); defer { pointsLock.unlock() }
        return _liveCoverageRevision
    }

    fileprivate var liveSurfelsSnapshot: [CaptureSurfelMap.Surfel] {
        pointsLock.lock(); defer { pointsLock.unlock() }
        return _liveSurfels
    }

    fileprivate var liveSurfelRevision: Int {
        pointsLock.lock(); defer { pointsLock.unlock() }
        return _liveSurfelRevision
    }

    /// Feed world points into coverage and update the snapshot (shared by feature points and LiDAR,
    /// on delegateQueue). `.mvs` shows surfels from `LiveMVSReconstructor` instead.
    fileprivate func updateLiveCoverage(_ points: [SIMD3<Float>], timestamp: TimeInterval) {
        guard !points.isEmpty else { return }
        coverageAccumulator.insert(points, timestamp: timestamp)
        let splats = coverageAccumulator.snapshot()
        pointsLock.lock()
        _liveCoverageSplats = splats
        _liveCoverageRevision += 1
        pointsLock.unlock()
    }

    // Everything below is accessed only on delegateQueue
    private var capturing = false
    private var pending: [PendingFrame] = []
    private var anchorTransforms: [UUID: simd_float4x4] = [:]
    private var lastKeptCameraToWorld: simd_float4x4 = matrix_identity_float4x4
    private var hasKeptFirst = false
    private var anchorUpdateCount = 0
    private var relocalizationObserved = false
    private var stateChange: ((DiagnosticsUpdate) -> Void)?
    // AE/AWB lock: arm the timer once on the first .normal frame and lock once when it fires
    private var exposureLockScheduled = false
    private var exposureLockDone = false
    private var stopContinuation: CheckedContinuation<CapturedScene, Error>?
    // Anchor-local record of rawFeaturePoints. On re-observation the first anchor is kept, so loop
    // closure corrections also propagate to the point cloud.
    private var featurePoints: [UInt64: LoopClosureResolver.AnchoredPoint] = [:]
    private var lidarWorldPoints: [SIMD3<Float>] = []
    private var lidarFrameCount = 0
    private var lidarSupportedRuntime = false
    private var lastTrackingStateDescription = "idle"
    private var pixelBufferColorTags: String?
    // For .mvs mode only (managed on delegateQueue)
    private var liveMVS: LiveMVSReconstructor?
    private var liveMVSSummary: String?

    init(configuration: CaptureConfiguration) throws {
        self.configuration = configuration
        self.pixelConverter = try PixelBufferToBytes(targetMaxDim: configuration.targetMaxDim)
        self.qualitySelector = CaptureRealtimeFrameSelector(parameters: configuration.realtimeQualityGate)
        super.init()
    }

    func start(onStateChange: @escaping (DiagnosticsUpdate) -> Void) throws {
        let cfg = ARWorldTrackingConfiguration()
        cfg.worldAlignment = .gravity
        // autoFocus on. Intrinsics are updated per frame so they stay consistent, and with it off
        // every frame blurs as soon as the capture distance changes.
        cfg.isAutoFocusEnabled = true
        // supportsFrameSemantics is called on ARWorldTrackingConfiguration
        let lidarSupported = configuration.enableLiDARDepth
            && ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
        if lidarSupported {
            cfg.frameSemantics.insert(.sceneDepth)
        }
        session.delegate = self
        session.delegateQueue = delegateQueue
        delegateQueue.sync {
            self.capturing = true
            self.pending.removeAll(keepingCapacity: true)
            self.anchorTransforms.removeAll()
            self.featurePoints.removeAll()
            self.hasKeptFirst = false
            self.anchorUpdateCount = 0
            self.relocalizationObserved = false
            self.exposureLockScheduled = false
            self.exposureLockDone = false
            self.stateChange = onStateChange
            self.lidarWorldPoints.removeAll()
            self.lidarFrameCount = 0
            self.lidarSupportedRuntime = lidarSupported
            self.lastTrackingStateDescription = "idle"
            self.qualitySelector.reset()
            self.coverageAccumulator.reset()
            // Only .mvs sets up incremental MVS. The capture view draws its surfels only, so the
            // coverage splats are not built.
            self.liveMVS?.stopAndWait()
            self.liveMVSSummary = nil
            if self.configuration.liveReconstructionMode == .mvs {
                self.liveMVS = LiveMVSReconstructor(onSurfels: { [weak self] surfels in
                    guard let self else { return }
                    self.pointsLock.lock()
                    self._liveSurfels = surfels
                    self._liveSurfelRevision += 1
                    self.pointsLock.unlock()
                })
            } else {
                self.liveMVS = nil
            }
            self.pointsLock.lock()
            self._liveCoverageSplats.removeAll()
            self._liveCoverageRevision += 1
            self._liveSurfels.removeAll()
            self._liveSurfelRevision += 1
            self.pointsLock.unlock()
        }
        onStateChange(.lidarSupported(lidarSupported))
        session.run(cfg, options: [.resetTracking, .removeExistingAnchors])
    }

    func stopCapture() async throws -> CapturedScene {
        // Stop accepting -> resolve after settleSeconds (to pick up late anchor updates)
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<CapturedScene, Error>) in
            delegateQueue.async {
                self.capturing = false
                self.stopContinuation = cont
            }
            let deadline: DispatchTime = .now() + configuration.settleSeconds
            delegateQueue.asyncAfter(deadline: deadline) { [weak self] in
                self?.finalizeCapture()
            }
        }
    }

    func cancel() {
        delegateQueue.async {
            self.capturing = false
            self.liveMVS?.stopAndWait()
            self.liveMVS = nil
            self.session.pause()
            self.stopContinuation?.resume(throwing: CaptureError.notRunning)
            self.stopContinuation = nil
        }
    }

    /// Lock AE/AWB at their current values (called once from the warm-up timer). No-op if
    /// unsupported.
    private func lockCameraExposureAndWhiteBalance() {
        // On delegateQueue
        guard capturing, !exposureLockDone else { return }
        guard #available(iOS 16.0, *),
              let device = ARWorldTrackingConfiguration.configurableCaptureDeviceForPrimaryCamera
        else { return }
        do {
            try device.lockForConfiguration()
            // If the AE exposure time exceeds the cap, clamp it to the cap and compensate with ISO;
            // otherwise lock at the AE values.
            var lockedViaCustom = false
            if let maxDur = configuration.maxShutterDuration,
               device.isExposureModeSupported(.custom) {
                let curDurSec = CMTimeGetSeconds(device.exposureDuration)
                if curDurSec.isFinite, curDurSec > maxDur, maxDur > 0 {
                    let fmt = device.activeFormat
                    let minD = CMTimeGetSeconds(fmt.minExposureDuration)
                    let cappedSec = Swift.max(minD, maxDur)
                    // Keep brightness with ISO ∝ 1/exposure time (clamped to the ISO range)
                    let scale = Float(curDurSec / cappedSec)
                    let targetISO = Swift.min(fmt.maxISO,
                                              Swift.max(fmt.minISO, device.iso * scale))
                    let cappedDur = CMTime(seconds: cappedSec, preferredTimescale: 1_000_000)
                    device.setExposureModeCustom(duration: cappedDur, iso: targetISO,
                                                  completionHandler: nil)
                    lockedViaCustom = true
                    stateChange?(.trackingState(String(
                        format: "shutter cap: %.1f→%.1f ms, ISO %.0f→%.0f",
                        curDurSec * 1000, cappedSec * 1000, device.iso, targetISO)))
                }
            }
            if !lockedViaCustom, device.isExposureModeSupported(.locked) {
                device.exposureMode = .locked
            }
            if device.isWhiteBalanceModeSupported(.locked) {
                device.whiteBalanceMode = .locked
            }
            device.unlockForConfiguration()
            exposureLockDone = true
            stateChange?(.exposureLocked(true))
        } catch {
            stateChange?(.trackingState("exposureLockError: \(error)"))
        }
    }

    private func finalizeCapture() {
        // On delegateQueue
        session.pause()
        // Drain pending incremental MVS work before creating MLXArrays (do not use MLX concurrently
        // with preprocessing/training).
        if let mvs = liveMVS {
            liveMVSSummary = mvs.stopAndWait()
            liveMVS = nil
        }
        guard let cont = stopContinuation else { return }
        stopContinuation = nil

        // Resolve with the latest anchor transforms and count anchors still at their capture-time
        // transform. Each frame's host bytes are dropped as soon as they are copied into an
        // MLXArray, so the capture is not held twice (host + MLX) after stop; pending is not used
        // after this.
        var records: [LoopClosureResolver.Record] = []
        records.reserveCapacity(pending.count)
        var unsettled = 0
        pending.reverse()
        while let p = pending.popLast() {
            let image = MLXArray(p.rgbBytes, [p.height, p.width, 3])
            records.append(LoopClosureResolver.Record(
                anchorID: p.anchorID, image: image,
                intrinsics: p.intrinsics,
                width: p.width, height: p.height, timestamp: p.timestamp,
                featureObservations: p.featureObservations
            ))
            if let cur = anchorTransforms[p.anchorID], cur == p.seedTransform {
                unsettled += 1
            }
        }
        let frames = LoopClosureResolver.resolve(
            records: records, anchorTransforms: anchorTransforms
        )
        stateChange?(.unsettledCount(unsettled))

        guard !frames.isEmpty else {
            cont.resume(throwing: CaptureError.noFramesCaptured)
            return
        }
        // Map feature points back to world with the post-loop-closure anchor transforms
        let pointArray = Array(featurePoints.values)
        let worldPoints = LoopClosureResolver.projectAnchoredPoints(
            pointArray, anchorTransforms: anchorTransforms
        )
        let diag = CaptureDiagnostics(
            keptFrameCount: frames.count,
            trackingStateDescription: lastTrackingStateDescription,
            relocalizationOccurred: relocalizationObserved,
            anchorUpdateCount: anchorUpdateCount,
            unsettledAnchorCount: unsettled,
            lidarSupported: lidarSupportedRuntime,
            lidarFrameCount: lidarFrameCount,
            lidarPointCount: lidarWorldPoints.count,
            pixelBufferColorTags: pixelBufferColorTags,
            liveMVSSummary: liveMVSSummary
        )
        cont.resume(returning: CapturedScene(
            frames: frames, sparsePoints: worldPoints,
            lidarPoints: lidarWorldPoints,
            diagnostics: diag
        ))
    }

    // MARK: ARSessionDelegate (on delegateQueue)

    /// Representative subject distance (median distance from the camera center to
    /// rawFeaturePoints). nil if there are no feature points.
    private func medianFeatureDepth(_ frame: ARFrame, cameraToWorld c2w: simd_float4x4) -> Float? {
        guard let pts = frame.rawFeaturePoints?.points, !pts.isEmpty else { return nil }
        let cx = c2w.columns.3.x, cy = c2w.columns.3.y, cz = c2w.columns.3.z
        var dists = [Float]()
        dists.reserveCapacity(pts.count)
        for p in pts {
            let dx = p.x - cx, dy = p.y - cy, dz = p.z - cz
            dists.append((dx * dx + dy * dy + dz * dz).squareRoot())
        }
        dists.sort()
        return dists[dists.count / 2]
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let stateString: String
        switch frame.camera.trackingState {
        case .normal: stateString = "normal"
        case .notAvailable: stateString = "notAvailable"
        case .limited(.relocalizing):
            stateString = "limited(.relocalizing)"
            if !relocalizationObserved {
                relocalizationObserved = true
                stateChange?(.relocalization)
            }
        case .limited(.initializing): stateString = "limited(.initializing)"
        case .limited(.insufficientFeatures): stateString = "limited(.insufficientFeatures)"
        case .limited(.excessiveMotion): stateString = "limited(.excessiveMotion)"
        case .limited: stateString = "limited(.other)"
        }
        lastTrackingStateDescription = stateString
        stateChange?(.trackingState(stateString))

        guard capturing else { return }

        if configuration.lockExposureAndWhiteBalance,
           !exposureLockScheduled,
           case .normal = frame.camera.trackingState {
            exposureLockScheduled = true
            delegateQueue.asyncAfter(deadline: .now() + configuration.exposureLockWarmupSeconds) {
                [weak self] in self?.lockCameraExposureAndWhiteBalance()
            }
        }

        guard pending.count < configuration.maxFrames else { return }
        if configuration.requireNormalTracking {
            guard case .normal = frame.camera.trackingState else { return }
        }

        // Translation / rotation gate
        let c2w = frame.camera.transform
        // Acceptance spacing: proportional to subject distance if distance-linked, fixed if no
        // feature points.
        let effMinMotion: Float = {
            guard configuration.adaptiveKeyframeSpacing,
                  let depth = medianFeatureDepth(frame, cameraToWorld: c2w)
            else { return configuration.minMotionMeters }
            return Swift.max(configuration.keyframeSpacingFloor,
                             Swift.min(configuration.keyframeSpacingCeil,
                                       configuration.keyframeDepthFactor * depth))
        }()
        if hasKeptFirst {
            let lastT = lastKeptCameraToWorld.columns.3
            let curT = c2w.columns.3
            let dx = curT.x - lastT.x
            let dy = curT.y - lastT.y
            let dz = curT.z - lastT.z
            let d2 = dx * dx + dy * dy + dz * dz
            let motionOK = d2 >= effMinMotion * effMinMotion
            // Approximate rotation by the angle between view axes (column 2)
            let lastZ = SIMD3<Float>(lastKeptCameraToWorld.columns.2.x,
                                     lastKeptCameraToWorld.columns.2.y,
                                     lastKeptCameraToWorld.columns.2.z)
            let curZ = SIMD3<Float>(c2w.columns.2.x, c2w.columns.2.y, c2w.columns.2.z)
            let cosA = max(-1, min(1, simd_dot(lastZ, curZ)))
            let angle = acos(cosA)
            let rotationOK = angle >= configuration.minRotationRadians
            guard motionOK || rotationOK else { return }

            // Redundant viewpoint: drop if close in both position and angle to an existing frame.
            // The previous frame is guaranteed above threshold by the gate above, so it never hits
            // the strict <, and normal forward capture is not rejected.
            if configuration.cullRedundantViewpoints {
                let dupDist2 = effMinMotion * effMinMotion
                let dupAngle = configuration.minRotationRadians
                for p in pending {
                    let pt = p.seedTransform.columns.3
                    let pdx = curT.x - pt.x, pdy = curT.y - pt.y, pdz = curT.z - pt.z
                    guard pdx * pdx + pdy * pdy + pdz * pdz < dupDist2 else { continue }
                    let pz = SIMD3<Float>(p.seedTransform.columns.2.x,
                                          p.seedTransform.columns.2.y,
                                          p.seedTransform.columns.2.z)
                    let cosP = max(-1, min(1, simd_dot(curZ, pz)))
                    if acos(cosP) < dupAngle { return }
                }
            }
        }

        let out: PixelBufferToBytes.Output
        do {
            out = try pixelConverter.convert(frame.capturedImage)
            if pixelBufferColorTags == nil {
                pixelBufferColorTags = PixelBufferToBytes.colorTagDescription(frame.capturedImage)
            }
        } catch {
            // A conversion failure skips just that frame
            stateChange?(.trackingState("convertError: \(error)"))
            return
        }
        let qualityDecision = qualitySelector.evaluate(
            rgbBytes: out.bytes,
            width: out.width,
            height: out.height
        )
        stateChange?(.qualityCandidate)
        guard qualityDecision.accepted else {
            stateChange?(.qualityRejected(qualityDecision.reasons))
            return
        }

        let origRes = frame.camera.imageResolution
        let intr = frame.camera.intrinsics
        let scaled = PoseConversion.scaledIntrinsics(
            fx: intr[0, 0], fy: intr[1, 1],
            cx: intr[2, 0], cy: intr[2, 1],
            originalWidth: Int(origRes.width), originalHeight: Int(origRes.height),
            scaledWidth: out.width, scaledHeight: out.height
        )

        // Register with the capture-time transform; didUpdate(anchors:) overwrites it later
        let anchor = ARAnchor(transform: c2w)
        anchorTransforms[anchor.identifier] = c2w
        session.add(anchor: anchor)

        // Record visible ARKit feature points (for CaptureBA multi-view tracks). Only points in
        // front of the camera.
        var featureObs: [ARKitFeatureObservation] = []
        if let cloud = frame.rawFeaturePoints {
            let w2c = c2w.inverse
            let n = min(cloud.identifiers.count, cloud.points.count)
            featureObs.reserveCapacity(n)
            for i in 0..<n {
                let w = cloud.points[i]
                let l = w2c * SIMD4<Float>(w.x, w.y, w.z, 1)
                // ARKit (OpenGL) camera → OpenCV camera: flip Y/Z
                let cv = SIMD3<Float>(l.x, -l.y, -l.z)
                guard cv.z > 1e-3 else { continue }
                featureObs.append(ARKitFeatureObservation(trackID: cloud.identifiers[i], cameraPoint: cv))
            }
        }

        pending.append(PendingFrame(
            anchorID: anchor.identifier,
            rgbBytes: out.bytes,
            width: out.width, height: out.height,
            intrinsics: scaled, timestamp: frame.timestamp,
            seedTransform: c2w,
            featureObservations: featureObs
        ))
        lastKeptCameraToWorld = c2w
        hasKeptFirst = true
        stateChange?(.keptFrameCount(pending.count))

        // Record feature points in anchor-local coordinates (keep the first anchor on
        // re-observation)
        var newFeatureWorld: [SIMD3<Float>] = []
        if let cloud = frame.rawFeaturePoints {
            let inv = c2w.inverse
            let ids = cloud.identifiers
            let pts = cloud.points
            let n = min(ids.count, pts.count)
            var coverageWorld: [SIMD3<Float>] = []
            coverageWorld.reserveCapacity(n)
            for i in 0..<n {
                let id = ids[i]
                let w = pts[i]
                coverageWorld.append(SIMD3<Float>(w.x, w.y, w.z))
                if featurePoints[id] != nil { continue }
                newFeatureWorld.append(SIMD3<Float>(w.x, w.y, w.z))
                let l4 = inv * SIMD4<Float>(w.x, w.y, w.z, 1)
                featurePoints[id] = LoopClosureResolver.AnchoredPoint(
                    anchorID: anchor.identifier,
                    localPos: SIMD3<Float>(l4.x, l4.y, l4.z)
                )
            }
            // Only .featurePoints mode feeds live coverage
            if configuration.liveReconstructionMode == .featurePoints {
                updateLiveCoverage(coverageWorld, timestamp: frame.timestamp)
            }
        }

        // .mvs: pass the keyframe and new feature points (for depth range and reference selection)
        // to incremental MVS
        liveMVS?.submit(
            rgb: out.bytes, width: out.width, height: out.height,
            intrinsics: (fx: scaled.fx, fy: scaled.fy, cx: scaled.cx, cy: scaled.cy),
            cameraToWorld: c2w, newSparsePoints: newFeatureWorld)

        // Accumulate LiDAR points in the same ARKit world coordinates as sparsePoints
        if lidarSupportedRuntime, let sd = frame.sceneDepth {
            let pts = DepthUnprojection.unprojectToWorld(
                depthMap: sd.depthMap,
                confidenceMap: sd.confidenceMap,
                cameraIntrinsics: intr,
                imageResolution: origRes,
                cameraToWorld: c2w,
                params: DepthUnprojection.Params(
                    stride: configuration.lidarStride,
                    minConfidenceRaw: UInt8(configuration.lidarConfidenceMinRaw)
                )
            )
            if !pts.isEmpty {
                lidarWorldPoints.append(contentsOf: pts)
                lidarFrameCount += 1
                stateChange?(.lidarFrame(lidarFrameCount))
                stateChange?(.lidarPoints(lidarWorldPoints.count))
                // In .lidar mode, also feed live coverage
                if configuration.liveReconstructionMode == .lidar {
                    updateLiveCoverage(pts, timestamp: frame.timestamp)
                }
            }
        }
    }

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        for a in anchors { anchorTransforms[a.identifier] = a.transform }
        anchorUpdateCount += anchors.count
        stateChange?(.anchorUpdate(anchors.count))
    }

    func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        for a in anchors { anchorTransforms.removeValue(forKey: a.identifier) }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        stopContinuation?.resume(throwing: CaptureError.sessionFailed("\(error)"))
        stopContinuation = nil
    }
}

#endif
