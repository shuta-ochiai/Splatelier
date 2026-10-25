// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// ViewModel that runs `CaptureTraining.train` on a background queue and streams logs and progress
// to `@Published`. MLXArray / CapturedScene are non-Sendable, so it runs on a DispatchQueue rather
// than a Task and hops back with DispatchQueue.main.async (assumes the App's
// SWIFT_STRICT_CONCURRENCY=minimal).

import Foundation
import UIKit
import MLX
@preconcurrency import Trainer
@preconcurrency import Capture

/// Snapshot of the settings at training start (`<base>.settings.json`). Shown in Library's settings
/// sheet.
struct TrainingSettingsRecord: Codable {
    let version: Int
    let frameCount: Int
    let iterations: Int
    let gaussianCap: Int
    let lambdaSSIM: Float
    let strategy: String
    let background: String
    let mcmcDensification: Bool
    let fixedCountDeadMode: String
    let coarseToFinePercent: Int
    let fineResPercent: Int
    let excludeARKitRaw: Bool
    let includeSparse: Bool
    let frameQualityFilter: Bool
    /// Present only in records from before the automatic switch (replayCacheMode) was introduced.
    let useReplayCache: Bool?
    /// rawValue of `ReplayCacheMode`, e.g. "on". nil in records from before it was introduced.
    let replayCacheMode: String?
    let replayKMax: Int
    let useMCMCReg: Bool
    let useEarlySH: Bool
    let useHighPrune: Bool
    let enableAnchorReg: Bool
    let enableMVS: Bool?
    let opticalFlowWeighting: Bool?
    let maxShDegree: Int?
    /// App version that trained it, e.g. "0.1.0 (1)". nil in records from before it was added.
    let appVersion: String?

    private var replayCacheText: String {
        // "thermal" (switching by thermal state) is no longer a mode, but older records may have it.
        if replayCacheMode == "thermal" { return "Auto (thermal, K=\(replayKMax))" }
        switch replayCacheMode.flatMap(ReplayCacheMode.init(rawValue:)) {
        case .on: return "ON (K=\(replayKMax))"
        case .off: return "OFF"
        case nil: return useReplayCache == false ? "OFF" : "ON (K=\(replayKMax))"
        }
    }

    /// List of (section heading, [(item, value)]) for display.
    var displaySections: [(title: String, rows: [(label: String, value: String)])] {
        func onOff(_ b: Bool) -> String { b ? "ON" : "OFF" }
        return [
            ("Basics", [
                ("App version", appVersion ?? "-"),
                ("Frames", "\(frameCount)"),
                ("Iterations", "\(iterations)"),
                ("Gaussian cap", gaussianCap >= 1000 ? "\(gaussianCap / 1000)k" : "\(gaussianCap)"),
                ("SSIM", String(format: "%.2f", lambdaSSIM)),
                ("Strategy", strategy),
                ("Background", background),
                ("SH degree", maxShDegree.map { "\($0)" } ?? "-"),
            ]),
            ("Seed", [
                ("MVS mode", onOff(enableMVS ?? false)),
                ("Exclude ARKit raw", onOff(excludeARKitRaw)),
                ("Exclude sparse", onOff(!includeSparse)),
                ("Frame quality filter", onOff(frameQualityFilter)),
                ("Optical flow weighting", opticalFlowWeighting.map(onOff) ?? "-"),
            ]),
            ("Advanced", [
                ("MCMC densification", onOff(mcmcDensification)),
                ("MCMC dead handling", fixedCountDeadMode),
                ("Coarse-to-fine", "\(coarseToFinePercent)%"),
                ("Fine resolution", "\(fineResPercent)%"),
                ("Replay cache", replayCacheText),
                ("MCMC regularization", onOff(useMCMCReg)),
                ("Early SH", onOff(useEarlySH)),
                ("Higher prune threshold", onOff(useHighPrune)),
                ("Anchor regularization", onOff(enableAnchorReg)),
            ]),
        ]
    }
}

/// Training result metrics (`<base>.metrics.json`). Shown on the Library card.
struct CaptureMetricsRecord: Codable {
    let version: Int
    let finalSmoothLoss: Float   // Rolling mean over the trailing window (same as "smooth" in the log)
    let iterations: Int
    let gaussianCount: Int
}

enum CaptureMetricsStore {
    static func sidecarURL(for fileURL: URL) -> URL {
        fileURL.deletingPathExtension().appendingPathExtension("metrics.json")
    }

    static func save(_ record: CaptureMetricsRecord, for fileURL: URL) {
        if let data = try? JSONEncoder().encode(record) {
            try? data.write(to: sidecarURL(for: fileURL), options: .atomic)
        }
    }

    static func load(for fileURL: URL) -> CaptureMetricsRecord? {
        guard let data = try? Data(contentsOf: sidecarURL(for: fileURL)),
              let record = try? JSONDecoder().decode(CaptureMetricsRecord.self, from: data),
              record.version == 1
        else { return nil }
        return record
    }
}

enum TrainingSettingsStore {
    static func sidecarURL(for fileURL: URL) -> URL {
        fileURL.deletingPathExtension().appendingPathExtension("settings.json")
    }

    static func save(_ record: TrainingSettingsRecord, for fileURL: URL) {
        if let data = try? JSONEncoder().encode(record) {
            try? data.write(to: sidecarURL(for: fileURL), options: .atomic)
        }
    }

    static func load(for fileURL: URL) -> TrainingSettingsRecord? {
        guard let data = try? Data(contentsOf: sidecarURL(for: fileURL)),
              let record = try? JSONDecoder().decode(TrainingSettingsRecord.self, from: data),
              record.version == 1
        else { return nil }
        return record
    }
}

/// What the viewer opens after training (Hashable so a navigation destination can key on it).
struct ViewerTarget: Identifiable, Hashable {
    let plyURL: URL
    let logURL: URL?
    var id: URL { plyURL }
}

@MainActor
final class TrainingViewModel: ObservableObject {

    enum Phase {
        case idle
        case running(iter: Int, latestLoss: Float?)
        case completed(plyURL: URL, logURL: URL?, gaussianCount: Int, finalLoss: Float?)
        case failed(String)
    }

    /// Mesh generation state (independent of the training phase; runs automatically after
    /// training and after continued training).
    enum MeshPhase {
        case idle
        case running
        case completed(objURL: URL, vertexCount: Int, triangleCount: Int)
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var meshPhase: MeshPhase = .idle
    /// Log of the last mesh generation (shared with the training log).
    @Published private(set) var meshLogURL: URL?
    @Published var iterations: Int = 3000
    @Published private(set) var elapsedSec: Double = 0
    @Published private(set) var peakMemoryBytes: Int = 0
    @Published private(set) var activeMemoryBytes: Int = 0
    /// While running, Elapsed and GPU memory are updated every second. If they only updated when a
    /// log line is emitted, the numbers would freeze during long steps that log nothing (BA, MVS,
    /// seed fusion, etc.) and the app would look hung.
    private var metricsTimer: Timer?
    /// Whether optimization is paused at thermalState=.critical. Resumes automatically once cooled.
    @Published private(set) var thermalPaused: Bool = false
    /// Whether it is running slowed down with a lower duty cycle at thermalState=.serious.
    @Published private(set) var thermalThrottled: Bool = false
    /// Iteration count of the running training (denominator of the progress bar). +N for continue
    /// training.
    @Published private(set) var runningTotalIterations: Int = 0
    /// Cumulative iteration count including continue training.
    @Published private(set) var cumulativeIterations: Int = 0

    /// Resumable session for continue training (holds the trainer and Adam state, in memory only).
    /// Discarded by a new start(). The continue training button is shown only while non-nil.
    private var resumableSession: TrainingDriver.ResumableTrainingSession?
    /// Output destination that continue training overwrites (fixed at the first training).
    private var lastOutputPLY: URL?
    private var lastThumbnailPNG: URL?
    private var lastLogURL: URL?
    /// Smooth window for metrics (= max(20, frameCount)).
    private var lastSmoothWindow: Int = 20
    static let continueIterationCount = 500

    private let scene: CapturedScene
    /// Intake, BA and MVS of the capture: training builds it and mesh generation (only after
    /// training) reuses it instead of redoing them.
    private let geometryCache = GeometryCache()
    /// Preserved dataset already on disk. If non-nil, start() skips re-saving it and deletes it on
    /// successful completion (so the next launch does not offer to resume).
    private let resumeDatasetURL: URL?
    /// Whether training from a validation dataset. If true, the dataset is kept even on successful
    /// completion.
    private let preserveDataset: Bool
    private let trainQueue = DispatchQueue(label: "io.github.shuta-ochiai.splatelier.train",
                                            qos: .userInitiated)
    /// Cap on the Gaussian count (peak), the only training setting on the screen; everything else
    /// is the app's configuration (`CaptureTraining.AppPreset`). The densification ceiling; the
    /// initialization uses 5/6 of it (`AppPreset.initialGaussianCap`).
    @Published var gaussianCap = TrainingViewModel.loadGaussianCap() {
        didSet {
            if gaussianCap != oldValue { UserDefaults.standard.set(gaussianCap, forKey: Self.gaussianCapKey) }
        }
    }
    /// Do not change the key: it carries over existing users' saved value.
    private nonisolated static let gaussianCapKey = "train.gaussianCap"

    private nonisolated static func loadGaussianCap() -> Int {
        let defaults = UserDefaults.standard
        let preset = CaptureTraining.AppPreset.defaultGaussianCap
        guard defaults.object(forKey: gaussianCapKey) != nil else { return preset }
        let saved = defaults.integer(forKey: gaussianCapKey)
        // 200k was removed from the choices; reset it to the default (prevents the picker from
        // showing no selection).
        return saved == 200_000 ? preset : saved
    }
    /// Display name (optional). From a validation dataset it inherits that name; finalized in the
    /// naming popup on completion (and with the viewer's rename button).
    @Published var captureName: String = ""
    /// Naming popup, shown when the first training's mesh is done.
    @Published var showNamePrompt = false
    /// The first training finished and has not asked for a name yet ("Train more" never asks).
    private var needsNamePrompt = false
    /// Set when training and its automatic mesh have finished and the naming popup (if any) is
    /// closed (first run or "Train more"); the screen pushes the viewer for it. Cleared when the
    /// viewer is dismissed.
    @Published var viewerTarget: ViewerTarget?
    /// The viewer to open once the naming popup is closed.
    private var pendingViewer: ViewerTarget?
    /// PLY URL of the completed capture (where the name sidecar is written).
    private var completedPlyURL: URL?
    /// Output folder capture_<T>/ and <T>. Created at the training start; mesh generation writes
    /// into it too.
    private var captureFolder: (dir: URL, timestamp: String)?

    init(scene: CapturedScene, resumeDatasetURL: URL? = nil, preserveDataset: Bool = false,
         initialName: String = "") {
        self.scene = scene
        self.resumeDatasetURL = resumeDatasetURL
        self.preserveDataset = preserveDataset
        self.captureName = initialName
        // The initial value is the MVS-mode recommendation tied to the frame count (4 iter/frame
        // clamped to [500, 2400]).
        self.iterations = CaptureTraining.recommendedIterations(
            frameCount: scene.frames.count, mvs: true)
    }

    func saveCaptureName() {
        guard let url = completedPlyURL else { return }
        CaptureMetadataName.save(captureName, for: url)
    }

    func start() {
        guard case .idle = phase, !isGeneratingMesh else { return }
        phase = .running(iter: 0, latestLoss: nil)
        elapsedSec = 0
        peakMemoryBytes = 0
        activeMemoryBytes = 0
        runningTotalIterations = iterations
        cumulativeIterations = iterations
        resumableSession = nil
        beginProgress(includesPreprocessing: true)
        previewImage = nil

        // GPU training stops if the app is suspended by screen lock, so disable auto-lock during
        // training. Always set it back to false on completion/failure.
        UIApplication.shared.isIdleTimerDisabled = true

        // Reset to measure the training peak excluding the capture-time buffers.
        MLX.Memory.peakMemory = 0
        let startTime = Date()
        startMetricsTimer(since: startTime)

        let captured = scene
        let cap = gaussianCap
        let totalIter = iterations
        let effectiveIterations = totalIter
        let captureNameSnapshot = captureName
        // For computing the smooth loss window (matches SceneTraining's max(20, views.count)).
        let frameCountSnapshot = scene.frames.count
        let densificationScheduleSnapshot = CaptureTraining.AppPreset.schedule(iterations: effectiveIterations)

        let (captureDir, timestamp) = ensureCaptureFolder(date: startTime)
        let plyURL = captureDir.appendingPathComponent("capture_\(timestamp).ply")
        // Write it before the PLY is finished so a resume after a crash can still pick up the name.
        CaptureMetadataName.save(captureNameSnapshot, for: plyURL)
        TrainingSettingsStore.save(
            TrainingSettingsRecord(
                version: 1,
                frameCount: scene.frames.count,
                iterations: effectiveIterations,
                gaussianCap: cap,
                lambdaSSIM: 0.2,
                strategy: "Hybrid",
                background: "SH background",
                // Hybrid uses no MCMC densification; the MCMC fields keep the record format.
                mcmcDensification: false,
                fixedCountDeadMode: "Relocate",
                coarseToFinePercent: Int((CaptureTraining.AppPreset.coarseToFineFraction * 100).rounded()),
                fineResPercent: Int((Float(CaptureTraining.AppPreset.fineResample.num)
                    / Float(CaptureTraining.AppPreset.fineResample.den) * 100).rounded()),
                excludeARKitRaw: true,
                includeSparse: false,
                frameQualityFilter: true,
                useReplayCache: nil,
                replayCacheMode: ReplayCacheMode.on.rawValue,
                replayKMax: CaptureTraining.AppPreset.replayKMax,
                useMCMCReg: true,
                useEarlySH: true,
                useHighPrune: false,
                enableAnchorReg: false,
                enableMVS: true,
                opticalFlowWeighting: true,
                maxShDegree: GaussianModel.maxShDegree,
                appVersion: AppVersion.plain
            ),
            for: plyURL
        )
        let logURL = captureDir.appendingPathComponent("capture_\(timestamp).log")
        let thumbURL = captureDir.appendingPathComponent("capture_\(timestamp).png")
        // Use a "_seed" suffix as a separate base so it does not collide with Library's base
        // parsing (up to the first ".").
        let seedURL = captureDir.appendingPathComponent("capture_\(timestamp)_seed.ply")
        // Preserve the capture data for recovery from a thermal crash / OOM; delete it on
        // successful completion.
        let datasetURL = resumeDatasetURL
            ?? captureDir.appendingPathComponent("capture_\(timestamp).dataset")
        let alreadyPersisted = (resumeDatasetURL != nil)
        // Local copy so the [weak self] closure can reference it without going through self.
        let preserveDataset = preserveDataset
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        let logHandle = try? FileHandle(forWritingTo: logURL)
        if let h = logHandle {
            let header = "=== splatelier \(AppVersion.full) log @\(timestamp), \(effectiveIterations) iter ===\n"
            if let d = header.data(using: .utf8) { h.write(d) }
        }

        let logHandler: (String) -> Void = { [weak self] msg in
            // jetsam (OOM kill) is SIGKILL with no chance to flush, so fsync every line to keep the
            // lines right before the crash.
            if let h = logHandle, let d = (msg + "\n").data(using: .utf8) {
                h.write(d)
                try? h.synchronize()
            }
            DispatchQueue.main.async {
                guard let self else { return }
                self.parseProgress(msg)
                let snap = MLX.Memory.snapshot()
                self.peakMemoryBytes = snap.peakMemory
                self.activeMemoryBytes = snap.activeMemory
                self.elapsedSec = Date().timeIntervalSince(startTime)
            }
        }

        let thermalHandler = makeThermalHandler()
        let previewHandler = makePreviewHandler()
        let geometryCache = geometryCache

        trainQueue.async { [weak self] in
            do {
                // The frames reload their images from it once training releases them.
                var savedDataset: CaptureDatasetFiles?
                // Preservation is best-effort. Training continues even if it fails.
                if alreadyPersisted {
                    savedDataset = captured.savedDataset
                    let kind = preserveDataset ? "validation dataset" : "persisted dataset"
                    logHandler("[dataset] training from \(kind) → \(datasetURL.lastPathComponent)")
                } else {
                    do {
                        let tSave = Date()
                        savedDataset = try CaptureDatasetStore.save(captured, to: datasetURL)
                        logHandler(String(format: "[dataset] saved %d frames for crash recovery → %@ [timing] datasetSave = %.2fs",
                                          captured.frames.count, datasetURL.lastPathComponent,
                                          Date().timeIntervalSince(tSave)))
                    } catch {
                        logHandler("[dataset] warning: could not persist dataset for recovery: \(error)")
                    }
                }
                let strategy = CaptureTraining.AppPreset.strategy(iterations: effectiveIterations, cap: cap)
                logHandler(Self.trainingStrategySettingsLine(cap: cap, schedule: densificationScheduleSnapshot))
                let initCap = CaptureTraining.AppPreset.initialGaussianCap(cap: cap)
                logHandler(Self.trainingPerformanceSettingsLine(initCap: initCap))
                logHandler("[MVS mode] MVS dense seed (cost mode m=5, 3-view consistency "
                    + "+ depth averaging, voxel 3mm, window 1), "
                    + "disc-init uncapped, coarse 1/\(CaptureTraining.AppPreset.coarseToFineFactor), strategy Hybrid")
                let geometry = CaptureTraining.prepareGeometry(scene: captured, log: logHandler)
                geometryCache.geometry = geometry
                let result = try CaptureTraining.train(
                    geometry: geometry,
                    iterations: effectiveIterations,
                    strategy: strategy,
                    // The rest is the default, the app's configuration (AppPreset / MVSMode).
                    // Start from 5/6 of the cap (actual init = min(seed point count, that)); edge
                    // splits grow it to the cap.
                    maxInitGaussians: initCap,
                    outputPLY: plyURL,
                    thumbnailPNG: thumbURL,
                    // Always save the pre-training seed point cloud + camera poses to a PLY for
                    // inspection (the viewer's Seed / camera display).
                    seedInspectionPLY: seedURL,
                    onThermalThrottle: thermalHandler,
                    onPreview: previewHandler,
                    log: logHandler
                )
                try? logHandle?.close()
                // Completed successfully, so delete the preserved data (validation datasets are
                // kept). The frames may still read their images from it (meshing, training
                // again), so it goes once they are released.
                if !preserveDataset {
                    if let savedDataset {
                        savedDataset.retire()
                    } else {
                        try? FileManager.default.removeItem(at: datasetURL)
                    }
                }
                // Final smooth loss (rolling mean over the trailing window). The window
                // approximates SceneTraining's max(20, views.count) with the frame count.
                let lossHistory = result.lossHistory
                let smoothWindow = Swift.max(20, frameCountSnapshot)
                let recentLosses = lossHistory.suffix(smoothWindow)
                if !recentLosses.isEmpty {
                    let smooth = recentLosses.reduce(0, +) / Float(recentLosses.count)
                    CaptureMetricsStore.save(
                        CaptureMetricsRecord(
                            version: 1,
                            finalSmoothLoss: smooth,
                            iterations: lossHistory.count,
                            gaussianCount: result.gaussianCount
                        ),
                        for: plyURL
                    )
                }
                let resultSession = result.session
                DispatchQueue.main.async {
                    UIApplication.shared.isIdleTimerDisabled = false
                    guard let self else { return }
                    self.stopMetricsTimer()
                    self.thermalPaused = false
                    self.thermalThrottled = false
                    let snap = MLX.Memory.snapshot()
                    self.peakMemoryBytes = snap.peakMemory
                    self.activeMemoryBytes = snap.activeMemory
                    self.elapsedSec = Date().timeIntervalSince(startTime)
                    self.resumableSession = resultSession
                    self.lastOutputPLY = plyURL
                    self.lastThumbnailPNG = thumbURL
                    self.lastLogURL = logURL
                    self.lastSmoothWindow = Swift.max(20, frameCountSnapshot)
                    self.phase = .completed(
                        plyURL: plyURL,
                        logURL: logURL,
                        gaussianCount: result.gaussianCount,
                        finalLoss: result.lossHistory.last
                    )
                    self.completedPlyURL = plyURL
                    self.needsNamePrompt = true
                    self.generateMesh()
                }
            } catch {
                try? logHandle?.close()
                DispatchQueue.main.async {
                    UIApplication.shared.isIdleTimerDisabled = false
                    guard let self else { return }
                    self.stopMetricsTimer()
                    self.thermalPaused = false
                    self.thermalThrottled = false
                    self.phase = .failed("\(error)")
                    self.endProgress(finished: false)
                }
            }
        }
    }

    /// Show the elapsed time since `startTime` (+ `base`) and GPU memory every second.
    /// `stopMetricsTimer()` on completion/failure.
    private func startMetricsTimer(since startTime: Date, base: Double = 0) {
        metricsTimer?.invalidate()
        let timer = Timer(timeInterval: 1, repeats: true) { [weak self] timer in
            guard let self else { timer.invalidate(); return }
            MainActor.assumeIsolated {
                self.elapsedSec = base + Date().timeIntervalSince(startTime)
                let snap = MLX.Memory.snapshot()
                self.peakMemoryBytes = snap.peakMemory
                self.activeMemoryBytes = snap.activeMemory
            }
        }
        // .common: keep running while the log is being scrolled.
        RunLoop.main.add(timer, forMode: .common)
        metricsTimer = timer
    }

    private func stopMetricsTimer() {
        metricsTimer?.invalidate()
        metricsTimer = nil
    }

    // MARK: - Continue training

    var canContinueTraining: Bool {
        if case .completed = phase, resumableSession != nil { return true }
        return false
    }

    /// Handler that reflects thermal state notifications from the training thread into the banner
    /// flags on main.
    private func makeThermalHandler() -> (ThermalThrottleState) -> Void {
        { [weak self] state in
            DispatchQueue.main.async {
                guard let self else { return }
                self.thermalPaused = (state == .paused)
                self.thermalThrottled = (state == .throttled)
            }
        }
    }

    /// Fine-tune the completed model for +`continueIterationCount` iters (no densification).
    /// Continues with the carried-over Adam state and overwrites the same PLY.
    func continueTraining() {
        guard case .completed = phase, !isGeneratingMesh,
              let session = resumableSession,
              let plyURL = lastOutputPLY else { return }
        let additional = Self.continueIterationCount
        phase = .running(iter: 0, latestLoss: nil)
        runningTotalIterations = additional
        beginProgress(includesPreprocessing: false)
        UIApplication.shared.isIdleTimerDisabled = true
        MLX.Memory.peakMemory = 0
        let startTime = Date()
        // Elapsed is shown as the total since the first training.
        let elapsedBase = elapsedSec
        startMetricsTimer(since: startTime, base: elapsedBase)

        let thumbURL = lastThumbnailPNG
        let logURL = lastLogURL
        let smoothWindow = lastSmoothWindow
        let logHandle = logURL.flatMap { try? FileHandle(forWritingTo: $0) }
        logHandle?.seekToEndOfFile()

        let logHandler: (String) -> Void = { msg in
            if let data = (msg + "\n").data(using: .utf8) { logHandle?.write(data) }
        }
        let progressHandler: (Int, Float) -> Void = { [weak self] localIter, loss in
            DispatchQueue.main.async {
                guard let self, case .running = self.phase else { return }
                self.phase = .running(iter: localIter, latestLoss: loss)
            }
        }

        let thermalHandler = makeThermalHandler()
        let previewHandler = makePreviewHandler()
        trainQueue.async { [weak self] in
            do {
                let result = try TrainingDriver.continueTraining(
                    session: session,
                    additionalIterations: additional,
                    outputPLY: plyURL,
                    thumbnailPNG: thumbURL,
                    onProgress: progressHandler,
                    onThermalThrottle: thermalHandler,
                    onPreview: previewHandler,
                    log: logHandler
                )
                try? logHandle?.close()
                let recent = result.lossHistory.suffix(smoothWindow)
                if !recent.isEmpty {
                    let smooth = recent.reduce(0, +) / Float(recent.count)
                    CaptureMetricsStore.save(
                        CaptureMetricsRecord(
                            version: 1,
                            finalSmoothLoss: smooth,
                            iterations: session.totalIterations,
                            gaussianCount: result.gaussianCount
                        ),
                        for: plyURL
                    )
                }
                DispatchQueue.main.async {
                    UIApplication.shared.isIdleTimerDisabled = false
                    guard let self else { return }
                    self.stopMetricsTimer()
                    let snap = MLX.Memory.snapshot()
                    self.peakMemoryBytes = snap.peakMemory
                    self.activeMemoryBytes = snap.activeMemory
                    self.elapsedSec = elapsedBase + Date().timeIntervalSince(startTime)
                    self.cumulativeIterations = session.totalIterations
                    self.phase = .completed(
                        plyURL: plyURL,
                        logURL: logURL,
                        gaussianCount: result.gaussianCount,
                        finalLoss: result.lossHistory.last
                    )
                    // The Gaussians changed, so rebuild the mesh (overwrites the same OBJ).
                    self.generateMesh()
                }
            } catch {
                try? logHandle?.close()
                DispatchQueue.main.async {
                    UIApplication.shared.isIdleTimerDisabled = false
                    guard let self else { return }
                    self.stopMetricsTimer()
                    self.phase = .failed("\(error)")
                    self.endProgress(finished: false)
                }
            }
        }
    }

    // MARK: - Mesh generation

    var isGeneratingMesh: Bool {
        if case .running = meshPhase { return true }
        return false
    }

    /// Training, continued training or mesh generation is running. The work cannot be cancelled,
    /// so the screen hides its back button meanwhile: leaving would free this view model while the
    /// work keeps running, and a second training could then be started next to it.
    var isBusy: Bool {
        if case .running = phase { return true }
        return isGeneratingMesh
    }

    /// Only after training: the mesh comes from the trained splats (their surface depth, flattened
    /// by the geometric regularization), or from training's MVS depth maps when the run had none.
    private var canGenerateMesh: Bool {
        if isGeneratingMesh { return false }
        if case .completed = phase { return true }
        return false
    }

    /// Build a vertex-colored mesh via depth → TSDF fusion → Marching Cubes and save it as
    /// capture_<T>_mesh.obj in the capture folder. The depth is the trained splats' surface when
    /// training used the geometric regularization (holes in MVS filled, walls flat), else training's
    /// MVS depth. Runs on the same queue as training so they do not compete for the GPU. Called on
    /// every training completion, so there is no button for it.
    private func generateMesh() {
        guard canGenerateMesh else { return }
        meshPhase = .running
        meshStartDate = Date()
        refreshProgress()
        UIApplication.shared.isIdleTimerDisabled = true
        let startTime = Date()
        let (dir, timestamp) = ensureCaptureFolder(date: startTime)
        // Library treats any .ply other than "_seed" as the main one, so put the mesh in a .obj
        // (separate base).
        let objURL = dir.appendingPathComponent("capture_\(timestamp)_mesh.obj")
        let logURL = dir.appendingPathComponent("capture_\(timestamp)_mesh.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        let logHandle = try? FileHandle(forWritingTo: logURL)
        let logHandler: (String) -> Void = { msg in
            if let h = logHandle, let d = (msg + "\n").data(using: .utf8) {
                h.write(d)
                try? h.synchronize()
            }
        }
        meshLogURL = logURL
        let geometryCache = geometryCache
        let session = resumableSession?.geometryRegularization != nil ? resumableSession : nil
        let sourceNote = session != nil
            ? "[mesh] source: the trained Gaussians' surface depth"
            : resumableSession == nil
                ? "[mesh] source: MVS depth (no trained Gaussians kept in memory)"
                : "[mesh] source: MVS depth (training ran without the geometric regularization)"
        let parameters = CaptureTraining.AppPreset.meshParameters
        trainQueue.async { [weak self] in
            logHandler(sourceNote)
            var built: (mesh: TriangleMesh, source: String)?
            if let session {
                do {
                    if let mesh = try CaptureMeshing.reconstruct(
                        model: session.trainer.model, cameras: session.views.map(\.camera),
                        shDegree: session.maxShDegree, parameters: parameters, log: logHandler) {
                        built = (mesh, "Gaussian surface depth")
                    }
                } catch {
                    logHandler("[mesh] Gaussian surface depth failed: \(error)")
                }
                MLX.Memory.clearCache()
                if built == nil { logHandler("[mesh] falling back to the MVS depth") }
            }
            if built == nil, let geometry = geometryCache.geometry, let mesh = CaptureMeshing.reconstruct(
                geometry: geometry, parameters: parameters, log: logHandler) {
                built = (mesh, "MVS depth")
            }
            let outcome: MeshPhase
            if let (mesh, source) = built {
                do {
                    try MeshOBJWriter.write(
                        mesh, to: objURL,
                        comment: "\(source) → TSDF (voxel \(parameters.tsdf.voxelSize) m) → marching cubes")
                    outcome = .completed(
                        objURL: objURL, vertexCount: mesh.vertexCount, triangleCount: mesh.triangleCount)
                } catch {
                    outcome = .failed("Couldn't write the OBJ: \(error)")
                }
            } else {
                outcome = .failed("Couldn't reconstruct a surface")
            }
            logHandler(String(format: "[timing] mesh(app) = %.2fs", Date().timeIntervalSince(startTime)))
            try? logHandle?.close()
            MLX.Memory.clearCache()
            DispatchQueue.main.async {
                guard let self else {
                    UIApplication.shared.isIdleTimerDisabled = false
                    return
                }
                if case .running = self.phase {} else {
                    UIApplication.shared.isIdleTimerDisabled = false
                }
                self.meshPhase = outcome
                self.endProgress(finished: true)
                // Training is only done once its mesh is: ask for a name (first run only), then open
                // the viewer with all layers (a failed mesh still opens it, without the mesh).
                if let plyURL = self.lastOutputPLY {
                    self.pendingViewer = ViewerTarget(plyURL: plyURL, logURL: self.lastLogURL)
                    if self.needsNamePrompt {
                        self.needsNamePrompt = false
                        self.showNamePrompt = true
                    } else {
                        self.openViewerIfReady()
                    }
                }
            }
        }
    }

    /// Opens the viewer once the naming popup is closed (also called directly when there is none).
    func openViewerIfReady() {
        guard !showNamePrompt, let target = pendingViewer else { return }
        pendingViewer = nil
        viewerTarget = target
    }

    /// The seed point cloud + camera marker PLY written by training (if any). Overlaid with GS and
    /// mesh in the viewer.
    var seedPLYURL: URL? {
        guard let folder = captureFolder else { return nil }
        let url = folder.dir.appendingPathComponent("capture_\(folder.timestamp)_seed.ply")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// OBJ of the generated mesh.
    var meshOBJURL: URL? {
        if case .completed(let url, _, _) = meshPhase { return url }
        return nil
    }

    /// Create the output folder capture_<T>/ (reuse it if it exists). File names inside stay
    /// capture_<T>.* (sidecars are resolved by base match).
    private func ensureCaptureFolder(date: Date) -> (dir: URL, timestamp: String) {
        if let folder = captureFolder { return folder }
        let timestamp = Self.captureTimestampString(for: date)
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("capture_\(timestamp)", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        captureFolder = (dir, timestamp)
        return (dir, timestamp)
    }

    // MARK: - Preview of the model while it trains

    /// What the model looks like now, from one fixed view; updated every few seconds while training.
    /// Kept after training so it stays on screen through the mesh generation.
    @Published private(set) var previewImage: UIImage?
    /// Counts the pictures (changes with each new one, so the screen can animate the swap).
    @Published private(set) var previewCount = 0

    /// Handler for the pictures the training loop produces (on the training thread). The frames are
    /// in sensor orientation, so the image is marked rotated for a portrait screen.
    private func makePreviewHandler() -> (TrainingPreviewFrame) -> Void {
        { [weak self] frame in
            guard let provider = CGDataProvider(data: Data(frame.rgba) as CFData),
                  let cg = CGImage(
                    width: frame.width, height: frame.height, bitsPerComponent: 8, bitsPerPixel: 32,
                    bytesPerRow: frame.width * 4, space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                    provider: provider, decode: nil, shouldInterpolate: true,
                    intent: .defaultIntent)
            else { return }
            let image = UIImage(cgImage: cg, scale: 1, orientation: .right)
            DispatchQueue.main.async {
                guard let self else { return }
                self.previewImage = image
                self.previewCount += 1
            }
        }
    }

    // MARK: - Overall progress

    /// One bar for everything after Start training: preparing the frames, training, and the mesh.
    /// The first run is split 15% / 75% / 10% by what the logs of real device runs showed
    /// (preprocessing 14-36 s, training about 0.1 s per iteration, mesh about 10 s); "Train more"
    /// has no preprocessing, so it is 90% / 10%. Training is exact (iteration / total); the other
    /// two follow the stages the log reports and creep toward the next one in between.
    @Published private(set) var overallProgress: Double = 0
    /// What is being done now ("Training", "Generating mesh", ...).
    @Published private(set) var progressLabel = ""
    /// Detail next to the bar ("iter 1200 / 3000" while training).
    @Published private(set) var progressDetail = ""

    private var includesPreprocessing = false
    private var trainingStarted = false
    /// How many preprocessing markers (see `preStages`) the log has reported.
    private var preMarkersSeen = 0
    private var preMarkerDate = Date()
    private var meshStartDate: Date?
    private var progressTimer: Timer?

    /// Preprocessing stages, in the order the log reports them. `marker` is the start of the log line
    /// that ends the stage, `fraction` how much of preprocessing is done then (from the device
    /// timings: intake 18%, poses 47%, depth 33%), `label` the stage that follows, `seconds` its
    /// usual length.
    private static let preStages: [(marker: String, fraction: Double, label: String, seconds: Double)] = [
        ("[timing] intake(", 0.18, "Estimating camera poses", 10),
        ("[timing] BA+geomrefine", 0.65, "Building depth maps", 8),
        ("[timing] MVS(total)", 0.97, "Preparing to train", 2),
        ("[timing] === pre-training total", 1.0, "Training", 1),
    ]

    private func beginProgress(includesPreprocessing: Bool) {
        self.includesPreprocessing = includesPreprocessing
        trainingStarted = !includesPreprocessing
        preMarkersSeen = 0
        preMarkerDate = Date()
        meshStartDate = nil
        refreshProgress()
        progressTimer?.invalidate()
        let timer = Timer(timeInterval: 0.5, repeats: true) { [weak self] timer in
            guard let self else { timer.invalidate(); return }
            MainActor.assumeIsolated { self.refreshProgress() }
        }
        RunLoop.main.add(timer, forMode: .common)
        progressTimer = timer
    }

    private func endProgress(finished: Bool) {
        progressTimer?.invalidate()
        progressTimer = nil
        if finished {
            overallProgress = 1
            progressLabel = ""
            progressDetail = ""
        }
    }

    /// Notes the log lines that mark a preprocessing stage as done.
    private func noteStage(_ msg: String) {
        guard includesPreprocessing, !trainingStarted else { return }
        for (i, stage) in Self.preStages.enumerated() where msg.hasPrefix(stage.marker) {
            preMarkersSeen = Swift.max(preMarkersSeen, i + 1)
            preMarkerDate = Date()
            if i == Self.preStages.count - 1 { trainingStarted = true }
        }
        if msg.hasPrefix("iter ") { trainingStarted = true }
        refreshProgress()
    }

    private func refreshProgress() {
        let preEnd = includesPreprocessing ? 0.15 : 0.0
        let trainEnd = 0.90
        if let start = meshStartDate, case .running = meshPhase {
            // Mesh: creeps toward the end over its usual ten seconds.
            let dt = Date().timeIntervalSince(start)
            overallProgress = trainEnd + (1 - trainEnd) * 0.95 * (1 - exp(-dt / 6))
            progressLabel = "Generating mesh"
            progressDetail = ""
        } else if case .running(let iter, _) = phase {
            if trainingStarted {
                let total = Double(Swift.max(1, runningTotalIterations))
                overallProgress = preEnd + (trainEnd - preEnd) * Swift.min(1, Double(iter) / total)
                progressLabel = "Training"
                progressDetail = "iter \(iter) / \(runningTotalIterations)"
            } else {
                // Preprocessing: from the last finished stage toward the next one.
                let floor = preMarkersSeen == 0 ? 0 : Self.preStages[preMarkersSeen - 1].fraction
                let ceil = preMarkersSeen < Self.preStages.count ? Self.preStages[preMarkersSeen].fraction : 1
                let usual = preMarkersSeen == 0 ? 5 : Self.preStages[preMarkersSeen - 1].seconds
                let dt = Date().timeIntervalSince(preMarkerDate)
                let inStage = 0.9 * (1 - exp(-dt / Swift.max(1, usual)))
                overallProgress = preEnd * (floor + (ceil - floor) * inStage)
                progressLabel = preMarkersSeen == 0
                    ? "Reading the frames" : Self.preStages[preMarkersSeen - 1].label
                progressDetail = ""
            }
        }
    }

    // MARK: - log handling (the log itself is the file; only the progress is read from it)

    /// Pick up progress from the "iter <N>: loss <F> (<count> gaussians) (<name>)" lines emitted by
    /// the training loop.
    private func parseProgress(_ msg: String) {
        noteStage(msg)
        guard msg.hasPrefix("iter "), case .running = phase else { return }
        let parts = msg.split(separator: " ")
        guard parts.count >= 4,
              let iter = Int(parts[1].dropLast()),        // Drop the ":"
              let loss = Float(parts[3])
        else { return }
        phase = .running(iter: iter, latestLoss: loss)
    }

    private static func captureTimestampString(for date: Date) -> String {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = .current
        formatter.dateFormat = "yyyyMMddHHmmss"
        return formatter.string(from: date)
    }

    private nonisolated static func trainingStrategySettingsLine(
        cap: Int,
        schedule: CaptureTraining.AppPreset.Schedule
    ) -> String {
        let scheduleText = "igsplus-las schedule=start \(schedule.refineStartIter), stop \(schedule.refineStopIter), every \(schedule.refineEvery), fires≈\(schedule.approximateFireCount)"
        return "[settings] strategy: Hybrid (IGS+ LAS growth + MCMC prune cleanup) cap=\(cap) \(scheduleText)"
    }

    private nonisolated static func trainingPerformanceSettingsLine(initCap: Int) -> String {
        String(format:
            "[settings] training perf: initCap=%d frameQuality=on opticalFlow=on lambdaSSIM=0.20 replayCache=on K=%d",
            initCap, CaptureTraining.AppPreset.replayKMax
        )
    }
}

/// Holder of a capture's `CaptureGeometry`, read and written only on the training queue.
private final class GeometryCache: @unchecked Sendable {
    var geometry: CaptureGeometry?
}
