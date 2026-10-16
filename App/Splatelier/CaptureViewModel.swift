// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// View model that drives `ARKitCaptureSession` from SwiftUI.
// State updated on the worker queue is mirrored to the MainActor by 100 ms Timer polling
// (to cap the UI update rate and handle state transitions in one place).

import Foundation
import SwiftUI
@preconcurrency import Capture

@MainActor
final class CaptureViewModel: ObservableObject {

    @Published private(set) var state: CaptureState = .idle
    @Published private(set) var diagnostics: CaptureDiagnostics = CaptureDiagnostics()
    @Published private(set) var capturedScene: CapturedScene?
    @Published private(set) var errorMessage: String?
    /// Number of world feature points accumulated for the live overlay.
    @Published private(set) var pointCount: Int = 0
    /// LiDAR sceneDepth intake. It also decides the live point cloud (ON = LiDAR depth / OFF =
    /// incremental MVS during capture). The LiDAR route is still experimental, so it is not exposed
    /// in the UI and stays off.
    private let enableLiDARDepth = false
    @Published private(set) var validationSaveState: ValidationSaveState = .idle
    /// Seconds since capture started (updated by the polling timer while running).
    @Published private(set) var captureElapsedSec: Int = 0

    /// State of saving the validation dataset. `saved` holds the display name or file name.
    enum ValidationSaveState: Equatable {
        case idle
        case saving
        case saved(String)
        case failed(String)
    }

    /// nil until start(). This VM is created at app launch, so it is created lazily to avoid paying
    /// the ARSession / PixelBufferToBytes construction cost at launch.
    private var session: ARKitCaptureSession?
    private var baseConfiguration: CaptureConfiguration
    private var pollTimer: Timer?
    private var captureStartDate: Date?

    /// Frame cap of the capture. Beyond it, frames are no longer kept.
    var maxFrames: Int { baseConfiguration.maxFrames }

    /// Exposed so ARCaptureView can reference the ARSession and point cloud. nil before start.
    var captureSession: ARKitCaptureSession? { session }

    // targetMaxDim 1080: cap on the long side of saved/training frames. At 1440, training with the
    // dense seed crashed from heat/OOM. The library default of 768 is left as is for test
    // compatibility.
    // maxFrames 600: the motion gate keeps about 5 frames/s, so 300 filled up in about 1 minute;
    // 600 allows about 2 minutes. Frames are 1080x810 uint8 (2.6 MB), about 1.6 GB at 600.
    init(configuration: CaptureConfiguration = CaptureConfiguration(maxFrames: 600, targetMaxDim: 1080)) {
        self.baseConfiguration = configuration
    }

    // MARK: - Actions

    func start() {
        guard state != .running else { return }
        capturedScene = nil
        errorMessage = nil
        var cfg = baseConfiguration
        // Devices without LiDAR cannot capture depth, so treat it as OFF (= MVS live point cloud).
        let useLiDAR = enableLiDARDepth && ARKitCaptureSession.isLiDARAvailable
        cfg.enableLiDARDepth = useLiDAR
        cfg.liveReconstructionMode = useLiDAR ? .lidar : .mvs
        // Quality selection during capture (drop blurred / badly exposed frames before saving) is
        // always ON.
        cfg.enableRealtimeQualityGate = true
        cfg.realtimeQualityGate.enabled = true
        let session = ARKitCaptureSession(configuration: cfg)
        self.session = session
        do {
            try session.start()
            captureStartDate = Date()
            captureElapsedSec = 0
            startPolling()
            refresh()
        } catch {
            errorMessage = "start failed: \(error)"
        }
    }

    func stop() async {
        guard state == .running, let session else { return }
        stopPolling()
        do {
            let scene = try await session.stopCapture()
            self.capturedScene = scene
        } catch {
            errorMessage = "stop failed: \(error)"
        }
        refresh()
    }

    func cancel() {
        stopPolling()
        session?.cancel()
        session = nil   // Release ARSession/Metal resources
        refresh()
    }

    func discardCapturedScene() {
        stopPolling()
        session?.cancel()
        session = nil
        capturedScene = nil
        errorMessage = nil
        validationSaveState = .idle
        refresh()
    }

    // MARK: - Validation dataset saving

    /// Save the captured scene to Documents as `validation_<datetime>.valset`. Unlike the
    /// crash-recovery `.dataset`, which is deleted when training finishes, it persists and is used
    /// to retrain on the same data. If `displayName` is non-empty, the display name is saved to a
    /// `<base>.name.json` sidecar.
    func saveValidationDataset(displayName: String = "") {
        guard let scene = capturedScene, validationSaveState != .saving else { return }
        validationSaveState = .saving
        let fileName = "validation_\(Self.timestampString()).valset"
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let url = docs.appendingPathComponent(fileName)
        let trimmedName = displayName.trimmingCharacters(in: .whitespacesAndNewlines)
        // Writing raw RGB (~200MB) is heavy, so do it in the background.
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let result: ValidationSaveState
            do {
                try CaptureDatasetStore.save(scene, to: url)
                CaptureMetadataName.save(trimmedName, for: url)
                result = .saved(trimmedName.isEmpty ? fileName : trimmedName)
            } catch {
                result = .failed("\(error)")
            }
            DispatchQueue.main.async { self?.validationSaveState = result }
        }
    }

    private static func timestampString(for date: Date = Date()) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyyMMddHHmmss"
        return f.string(from: date)
    }

    // MARK: - Sync polling

    private func startPolling() {
        pollTimer?.invalidate()
        pollTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in self?.refresh() }
        }
    }

    private func stopPolling() {
        pollTimer?.invalidate()
        pollTimer = nil
    }

    private func refresh() {
        guard let session else {
            self.state = .idle
            self.pointCount = 0
            return
        }
        self.state = session.state
        self.diagnostics = session.diagnostics
        self.pointCount = session.liveSplatCount
        if state == .running, let captureStartDate {
            let sec = Int(Date().timeIntervalSince(captureStartDate))
            if sec != captureElapsedSec { captureElapsedSec = sec }
        }
    }
}

// MARK: - Display name sidecar

/// Reads and writes the display name of a capture / .valset as a sibling `<base>.name.json`
/// sidecar. The actual files are never renamed (that would break the companion base match, and with
/// it the Library grouping and bulk delete). The sidecar also uses the base + "." prefix, so it is
/// deleted together with the capture.
enum CaptureMetadataName {
    private struct Meta: Codable { let version: Int; let name: String }

    static func sidecarURL(for fileURL: URL) -> URL {
        fileURL.deletingPathExtension().appendingPathExtension("name.json")
    }

    /// Read the display name. nil for an unset / empty / broken sidecar.
    static func load(for fileURL: URL) -> String? {
        guard let data = try? Data(contentsOf: sidecarURL(for: fileURL)),
              let meta = try? JSONDecoder().decode(Meta.self, from: data),
              meta.version == 1
        else { return nil }
        let trimmed = meta.name.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    /// Save the display name. If empty after trimming, delete the sidecar to return to "unnamed".
    static func save(_ name: String, for fileURL: URL) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let url = sidecarURL(for: fileURL)
        guard !trimmed.isEmpty else {
            try? FileManager.default.removeItem(at: url)
            return
        }
        if let data = try? JSONEncoder().encode(Meta(version: 1, name: trimmed)) {
            try? data.write(to: url, options: .atomic)
        }
    }

    /// Delete the name sidecar (for .valset, which has no companion bulk delete).
    static func delete(for fileURL: URL) {
        try? FileManager.default.removeItem(at: sidecarURL(for: fileURL))
    }
}
