// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Capture screen. Handles pre-capture settings, the live AR preview during capture, and, in the
// same screen, what follows: the result card with the training settings, the training progress,
// crash recovery and the list of validation datasets (.valset). Only the finished result is
// pushed (the viewer).

import SwiftUI
import UIKit
import Capture

struct CaptureView: View {
    @StateObject private var vm = CaptureViewModel()
    @State private var showDiscardConfirm = false
    @State private var showFrameReview = false
    /// LiDAR depth during capture (LiDAR devices only). Still experimental, so off by default.
    @AppStorage("capture.useLiDAR") private var useLiDAR = false

    // MARK: training (in this screen)
    /// Set once there is something to train: a fresh capture, a resumed dataset or a validation
    /// dataset. The training panel replaces the start screen while it is set.
    @State private var trainer: TrainingViewModel?
    /// The trainer holds the capture just made (not saved until trained), not a stored dataset.
    @State private var trainerIsFreshCapture = false
    /// Name and frame count of the stored dataset being trained (header card).
    @State private var storedName = ""
    @State private var storedFrameCount = 0

    // MARK: crash recovery
    /// Preserved dataset (*.dataset) left over from a previous crash. Detected at launch.
    @State private var pendingDatasetURL: URL?
    @State private var showResumePrompt = false
    @State private var showAbout = false
    @State private var didCheckRecovery = false
    @State private var recoveryError: String?

    // MARK: validation datasets
    @State private var validationSets: [ValidationEntry] = []
    @State private var showValidationList = false
    @State private var loadingValidationURL: URL?
    @State private var showValsetNamePrompt = false
    @State private var valsetNameText = ""
    @State private var renamingValset: ValidationEntry?
    @State private var renameValsetText = ""
    @State private var deletingValset: ValidationEntry?
    /// .valset being zipped (a spinner is shown while non-nil).
    @State private var zippingValsetURL: URL?
    /// The finished zip. Non-nil presents the share sheet.
    @State private var valsetShareItem: ShareableFile?

    /// One row of the validation dataset list (read from the manifest only).
    private struct ValidationEntry: Identifiable, Hashable {
        let url: URL
        let frameCount: Int
        let modifiedAt: Date
        let name: String?       // Display name (<base>.name.json)
        var id: URL { url }
        var displayDate: String {
            let f = DateFormatter()
            f.dateStyle = .medium
            f.timeStyle = .short
            return f.string(from: modifiedAt)
        }
        var displayTitle: String { name ?? displayDate }
    }

    var body: some View {
        Group {
            if isLive {
                liveCaptureView
            } else {
                setupView
            }
        }
        
        .navigationTitle("Capture")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            // About is on the start screen only; during capture and training the bar stays clean.
            if trainer == nil, vm.capturedScene == nil, !isLive {
                ToolbarItem(placement: .topBarTrailing) {
                    Button { showAbout = true } label: { Image(systemName: "info.circle") }
                        .accessibilityLabel("About")
                }
            }
        }
        .sheet(isPresented: $showAbout) { AboutView() }
        .onAppear {
            checkForRecoverableDataset()
            refreshValidationSets()
            // Compile the shaders for the splat display during capture ahead of time (so the first
            // capture start does not wait)
            CaptureSurfelRenderer.prewarm()
        }
        // The capture just made becomes the thing to train.
        .onChange(of: vm.capturedScene != nil) { _, hasScene in
            if hasScene, trainer == nil, let scene = vm.capturedScene {
                trainer = TrainingViewModel(scene: scene)
                trainerIsFreshCapture = true
            }
        }
        .alert("Unfinished capture found", isPresented: $showResumePrompt) {
            Button("Open") { resumePendingDataset() }
            Button("Discard", role: .destructive) { discardPendingDataset() }
            Button("Later", role: .cancel) { }
        } message: {
            Text("The last training run stopped before finishing. The capture data is still here, so you can train it again from the start.")
        }
        .alert(
            "Couldn't resume",
            isPresented: Binding(get: { recoveryError != nil },
                                 set: { if !$0 { recoveryError = nil } })
        ) {
            Button("OK", role: .cancel) { recoveryError = nil }
        } message: {
            Text(recoveryError ?? "")
        }
        .onDisappear { vm.cancel() }
        // A captured scene is not saved until it is trained, so confirm before discarding it.
        .confirmationDialog(
            "Discard this capture?",
            isPresented: $showDiscardConfirm,
            titleVisibility: .visible
        ) {
            Button("Discard and go back", role: .destructive) { endTraining() }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("If you haven't trained it or used \"Save for validation\", the capture data will be lost. This can't be undone.")
        }
    }

    private var isLive: Bool {
        switch vm.state {
        case .running, .stopping: return true
        default: return false
        }
    }

    @ViewBuilder
    private var setupView: some View {
        if let trainer {
            TrainingView(
                vm: trainer, header: trainingHeader, onClose: closeTraining,
                closeDiscards: trainerIsFreshCapture)
        } else if vm.capturedScene != nil {
            // The capture just finished; the training panel appears on the next update.
            Color.clear
        } else {
            startView
        }
    }

    private var startView: some View {
        VStack(spacing: 24) {
            // Keep the card fixed on screen and scroll only the validation dataset list inside it.
            VStack(spacing: 24) {
                if !validationSets.isEmpty { validationListButton }
                if let err = vm.errorMessage { errorRow(err) }
                // Capture tips (a different clip each time this screen appears).
                GuideClipView()
                    .frame(maxHeight: .infinity)
            }
            .frame(maxHeight: .infinity, alignment: .top)
            if ARKitCaptureSession.isLiDARAvailable {
                Toggle("LiDAR depth (experimental)", isOn: $useLiDAR)
                    .padding(.horizontal)
            }
            controlRow
        }
        .padding()
        .sheet(isPresented: $showValidationList) { validationListSheet }
        // The last dataset was deleted: nothing left to list.
        .onChange(of: validationSets.isEmpty) { _, isEmpty in
            if isEmpty { showValidationList = false }
        }
    }

    /// What is being trained, above the settings.
    private var trainingHeader: AnyView {
        if trainerIsFreshCapture, let scene = vm.capturedScene {
            return AnyView(resultCard(scene))
        }
        return AnyView(
            VStack(alignment: .leading, spacing: 8) {
                Label(storedName.isEmpty ? "Saved capture" : storedName, systemImage: "flask")
                    .font(.headline)
                row("Frames", "\(storedFrameCount)")
            }
            .padding()
            .frame(maxWidth: .infinity, alignment: .leading)
            .cardBackground()
        )
    }

    /// The button at the bottom of the training panel. A fresh capture that was not trained yet
    /// asks first; everything else just goes back to the start screen.
    private func closeTraining() {
        if trainerIsFreshCapture, let trainer, !isTrained(trainer) {
            showDiscardConfirm = true
        } else {
            endTraining()
        }
    }

    private func isTrained(_ trainer: TrainingViewModel) -> Bool {
        if case .completed = trainer.phase { return true }
        return false
    }

    /// Back to the start screen. A fresh capture is released here (it is saved by training, or
    /// the user chose to discard it).
    private func endTraining() {
        if trainerIsFreshCapture { vm.discardCapturedScene() }
        trainer = nil
        trainerIsFreshCapture = false
    }

    /// Starts the training panel for a dataset read from storage.
    private func startStoredTraining(scene: CapturedScene, url: URL, preserve: Bool, name: String) {
        storedName = name
        storedFrameCount = scene.frames.count
        trainerIsFreshCapture = false
        trainer = TrainingViewModel(
            scene: scene, resumeDatasetURL: url, preserveDataset: preserve, initialName: name)
    }

    private var liveCaptureView: some View {
        ZStack {
            #if os(iOS)
            if let captureSession = vm.captureSession {
                ARCaptureView(captureSession: captureSession)
                    .ignoresSafeArea()
            }
            #else
            Color.black.ignoresSafeArea()
            #endif
            VStack {
                liveHUD
                Spacer()
                controlRow
                    .padding()
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
                    .padding(.horizontal)
                    .padding(.bottom)
            }
        }
    }

    private var liveHUD: some View {
        HStack(spacing: 16) {
            hudItem(label: "Tracking", value: vm.diagnostics.trackingStateDescription)
            hudItem(label: "Frames", value: "\(vm.diagnostics.keptFrameCount)/\(vm.maxFrames)",
                    highlight: vm.diagnostics.keptFrameCount >= vm.maxFrames)
            hudItem(label: "Time", value: formatCaptureTime(vm.captureElapsedSec))
            hudItem(label: "Splats", value: "\(vm.pointCount)")
            if vm.diagnostics.qualityRejectedFrameCount > 0 {
                hudItem(label: "Rejected", value: "\(vm.diagnostics.qualityRejectedFrameCount)")
            }
            if vm.diagnostics.relocalizationOccurred {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial, in: Capsule())
        .padding(.top, 8)
    }

    /// `highlight` colors the value orange (e.g. the frame cap is reached and new frames are no
    /// longer kept).
    private func hudItem(label: String, value: String, highlight: Bool = false) -> some View {
        VStack(spacing: 2) {
            Text(label).font(.caption2).foregroundStyle(.secondary)
            Text(value).font(.caption.monospacedDigit().bold())
                .foregroundStyle(highlight ? AnyShapeStyle(.orange) : AnyShapeStyle(.primary))
        }
    }

    private func formatCaptureTime(_ sec: Int) -> String {
        String(format: "%d:%02d", sec / 60, sec % 60)
    }

    // MARK: - subviews

    private func resultCard(_ scene: CapturedScene) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Capture complete", systemImage: "checkmark.circle.fill")
                .font(.headline)
                .foregroundStyle(.green)
            Text("\(scene.frames.count) frames · \(scene.sparsePoints.count) points")
                .font(.subheadline.monospacedDigit())
                .foregroundStyle(.secondary)
            // Only what needs attention. The detailed ARKit counters are in the processing log.
            if vm.diagnostics.relocalizationOccurred {
                Label("Relocalization observed", systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
            if let warning = Self.trackingWarning(vm.diagnostics.trackingStateDescription) {
                Label(warning, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
            HStack(spacing: 8) {
                Button { showFrameReview = true } label: {
                    Label("Review frames", systemImage: "photo.stack")
                        .font(.subheadline)
                        .lineLimit(1).minimumScaleFactor(0.8)
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.regular)
                validationSaveButton
            }
            .padding(.top, 2)
            if case .failed(let msg) = vm.validationSaveState {
                Text("Couldn't save: \(msg)").font(.caption).foregroundStyle(.red)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .cardBackground()
        .sheet(isPresented: $showFrameReview) {
            // The frame quality filter is always ON in training, so the review verdict always
            // includes the filter.
            FrameReviewView(scene: scene, qualityFilterEnabled: true)
        }
    }

    /// A warning for the tracking state ARKit reported when capture stopped, or nil when it was
    /// fine. The state strings come from ARKitCaptureSession: "normal", "notAvailable",
    /// "limited(.excessiveMotion)", "limited(.insufficientFeatures)", "limited(.initializing)",
    /// "limited(.relocalizing)", "limited(.other)", and "idle" before any frame.
    static func trackingWarning(_ state: String) -> String? {
        switch state {
        case "normal", "idle": return nil
        case "notAvailable": return "Tracking was not available when you stopped"
        case "limited(.excessiveMotion)": return "Tracking was limited by fast motion when you stopped"
        case "limited(.insufficientFeatures)":
            return "Tracking was limited by too little detail when you stopped"
        case "limited(.initializing)": return "Tracking was still starting when you stopped"
        case "limited(.relocalizing)": return "Tracking was recovering its position when you stopped"
        default:
            // "limited(.other)", or a message the capture session reported in this field.
            return state.hasPrefix("limited") ? "Tracking was limited when you stopped" : nil
        }
    }

    /// Next to Review frames: saves the capture as a validation dataset.
    @ViewBuilder
    private var validationSaveButton: some View {
        switch vm.validationSaveState {
        case .saved:
            Label("Saved", systemImage: "checkmark.circle.fill")
                .lineLimit(1)
                .frame(maxWidth: .infinity, minHeight: 32)
                .foregroundStyle(.green)
        default:
            let saving = vm.validationSaveState == .saving
            Button { valsetNameText = ""; showValsetNamePrompt = true } label: {
                HStack(spacing: 6) {
                    if saving { ProgressView() }
                    Label(saving ? "Saving…" : "Save for validation", systemImage: "flask")
                        .font(.subheadline)
                        .lineLimit(1).minimumScaleFactor(0.8)
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .controlSize(.regular)
            .disabled(saving)
            .alert("Validation dataset name", isPresented: $showValsetNamePrompt) {
                TextField("e.g. Sofa test (optional)", text: $valsetNameText)
                Button("Save") { vm.saveValidationDataset(displayName: valsetNameText) }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("If left empty, the date and time are used.")
            }
        }
    }

    /// List of saved *.valset. Loading one proceeds to training.
    /// One row that opens the saved-dataset list, so the start screen stays just "Start capture".
    private var validationListButton: some View {
        Button { showValidationList = true } label: {
            HStack {
                Label("Validation datasets", systemImage: "flask")
                Spacer()
                Text("\(validationSets.count)").foregroundStyle(.secondary)
                Image(systemName: "chevron.right")
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(.tertiary)
            }
            .padding()
            .frame(maxWidth: .infinity)
            .cardBackground()
        }
        .buttonStyle(.plain)
    }

    private var validationListSheet: some View {
        NavigationStack {
            ScrollView {
                validationDatasetsList
                    .padding()
            }
            .appBackground()
            .navigationTitle("Validation datasets")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { showValidationList = false }
                }
            }
        }
    }

    private var validationDatasetsList: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Train on saved captures as often as you like (for comparing settings).")
                .font(.caption)
                .foregroundStyle(.secondary)
            validationRows
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .alert("Rename", isPresented: Binding(
            get: { renamingValset != nil },
            set: { if !$0 { renamingValset = nil } }
        ), presenting: renamingValset) { entry in
            TextField("Name", text: $renameValsetText)
            Button("Save") {
                CaptureMetadataName.save(renameValsetText, for: entry.url)
                renamingValset = nil
                refreshValidationSets()
            }
            Button("Cancel", role: .cancel) { renamingValset = nil }
        } message: { entry in
            Text(entry.displayDate)
        }
        .confirmationDialog(
            "Delete this validation dataset?",
            isPresented: Binding(
                get: { deletingValset != nil },
                set: { if !$0 { deletingValset = nil } }
            ),
            titleVisibility: .visible,
            presenting: deletingValset
        ) { entry in
            Button("Delete", role: .destructive) {
                deleteValidationDataset(entry.url)
                deletingValset = nil
            }
            Button("Cancel", role: .cancel) { deletingValset = nil }
        } message: { entry in
            Text("\(entry.displayTitle) will be permanently deleted. This can't be undone.")
        }
        .sheet(item: $valsetShareItem) { item in
            ShareSheet(items: [item.url])
        }
    }

    /// Each row of the validation dataset list (train button and menu).
    private var validationRows: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(validationSets) { entry in
                VStack(spacing: 8) {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(entry.displayTitle).font(.body)
                            Text(entry.name != nil
                                 ? "\(entry.displayDate) · \(entry.frameCount) frames"
                                 : "\(entry.frameCount) frames")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        if loadingValidationURL == entry.url {
                            ProgressView()
                        } else {
                            Button { loadValidationDataset(entry.url) } label: {
                                Label("Train", systemImage: "wand.and.stars")
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(Color.kikyoFill)
                            .controlSize(.small)
                            .disabled(loadingValidationURL != nil)
                            // Share / rename / delete go in a menu to prevent mis-taps next to the
                            // train button.
                            Menu {
                                Button {
                                    shareValidationDataset(entry)
                                } label: {
                                    Label("Share", systemImage: "square.and.arrow.up")
                                }
                                Button {
                                    renameValsetText = entry.name ?? ""
                                    renamingValset = entry
                                } label: {
                                    Label("Rename", systemImage: "pencil")
                                }
                                Button(role: .destructive) {
                                    deletingValset = entry
                                } label: {
                                    Label("Delete", systemImage: "trash")
                                }
                            } label: {
                                if zippingValsetURL == entry.url {
                                    ProgressView()
                                } else {
                                    Image(systemName: "ellipsis.circle")
                                        .imageScale(.large)
                                }
                            }
                            .menuStyle(.borderlessButton)
                            .disabled(loadingValidationURL != nil || zippingValsetURL != nil)
                        }
                    }
                    if entry.id != validationSets.last?.id { Divider() }
                }
            }
        }
    }

    private func errorRow(_ msg: String) -> some View {
        Label(msg, systemImage: "xmark.octagon.fill")
            .font(.caption)
            .foregroundStyle(.red)
            .padding(.horizontal)
    }

    private var controlRow: some View {
        HStack(spacing: 12) {
            switch vm.state {
            case .idle, .stopped:
                captureStartButton
            case .running:
                Button(role: .destructive, action: { Task { await vm.stop() } }) {
                    Label("Stop", systemImage: "stop.circle.fill")
                        .frame(maxWidth: .infinity)
                }
                // No tint: the destructive role makes it red, so it differs from Start capture.
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            case .stopping:
                ProgressView("Finalising…")
                    .frame(maxWidth: .infinity)
            }
        }
    }

    private var captureStartButton: some View {
        Button(action: { vm.start(useLiDAR: useLiDAR) }) {
            Label("Start capture", systemImage: "record.circle")
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(.borderedProminent)
        .tint(Color.kikyoFill)
        .controlSize(.large)
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label).foregroundStyle(.secondary)
            Spacer()
            Text(value).font(.body.monospacedDigit())
        }
    }

    // MARK: - crash recovery

    /// Once after launch, look for an unfinished preserved dataset and offer to resume it.
    /// Not shown during capture or while a scene is in memory.
    private func checkForRecoverableDataset() {
        guard !didCheckRecovery else { return }
        didCheckRecovery = true
        guard !isLive, vm.capturedScene == nil, trainer == nil,
              let url = Self.mostRecentPendingDataset() else { return }
        pendingDatasetURL = url
        showResumePrompt = true
    }

    /// Decode the preserved dataset in the background and open the training panel for it.
    private func resumePendingDataset() {
        guard let url = pendingDatasetURL else { return }
        Task.detached {
            do {
                let scene = try CaptureDatasetStore.load(from: url)
                await MainActor.run {
                    startStoredTraining(
                        scene: scene, url: url, preserve: false,
                        name: CaptureMetadataName.load(for: url) ?? "")
                }
            } catch {
                await MainActor.run { recoveryError = "\(error)" }
            }
        }
    }

    private func discardPendingDataset() {
        if let url = pendingDatasetURL { try? FileManager.default.removeItem(at: url) }
        pendingDatasetURL = nil
    }

    /// Find the latest preserved data (`*.dataset`). Scans both inside capture_<T>/ folders and
    /// directly under Documents.
    private static func mostRecentPendingDataset() -> URL? {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let entries = (try? FileManager.default.contentsOfDirectory(
            at: docs, includingPropertiesForKeys: [.contentModificationDateKey, .isDirectoryKey])) ?? []
        var datasets: [URL] = []
        for entry in entries {
            if entry.pathExtension == "dataset" {
                datasets.append(entry)
            } else if entry.lastPathComponent.hasPrefix("capture_"),
                      (try? entry.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
                let inner = (try? FileManager.default.contentsOfDirectory(
                    at: entry, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
                datasets.append(contentsOf: inner.filter { $0.pathExtension == "dataset" })
            }
        }
        func modDate(_ u: URL) -> Date {
            (try? u.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
        }
        return datasets.max(by: { modDate($0) < modDate($1) })
    }

    // MARK: - validation datasets

    /// List the `*.valset` directly under Documents by reading only their manifests (newest first).
    private func refreshValidationSets() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let entries = (try? FileManager.default.contentsOfDirectory(
            at: docs, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        var out: [ValidationEntry] = []
        for url in entries where url.pathExtension == "valset" {
            guard let info = try? CaptureDatasetStore.peek(at: url) else { continue }
            let mod = (try? url.resourceValues(forKeys: [.contentModificationDateKey])
                .contentModificationDate) ?? .distantPast
            out.append(ValidationEntry(
                url: url, frameCount: info.frameCount, modifiedAt: mod,
                name: CaptureMetadataName.load(for: url)))
        }
        validationSets = out.sorted { $0.modifiedAt > $1.modifiedAt }
    }

    /// Load a validation dataset in the background and open the training panel for it.
    private func loadValidationDataset(_ url: URL) {
        guard loadingValidationURL == nil else { return }
        loadingValidationURL = url
        Task.detached {
            do {
                let scene = try CaptureDatasetStore.load(from: url)
                await MainActor.run {
                    loadingValidationURL = nil
                    showValidationList = false
                    startStoredTraining(
                        scene: scene, url: url, preserve: true,
                        name: CaptureMetadataName.load(for: url) ?? "")
                }
            } catch {
                await MainActor.run {
                    loadingValidationURL = nil
                    showValidationList = false
                    recoveryError = "\(error)"
                }
            }
        }
    }

    private func deleteValidationDataset(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
        CaptureMetadataName.delete(for: url)
        refreshValidationSets()
    }

    /// A .valset is a directory and cannot be passed to ShareLink directly, so zip it before
    /// sharing (on a Mac, unzip it and train with `train-capture`). It is hundreds of MB, so do it
    /// in the background.
    private func shareValidationDataset(_ entry: ValidationEntry) {
        guard zippingValsetURL == nil else { return }
        zippingValsetURL = entry.url
        let srcURL = entry.url
        Task.detached {
            do {
                let zipURL = try Self.zipDirectoryForSharing(srcURL)
                await MainActor.run {
                    zippingValsetURL = nil
                    valsetShareItem = ShareableFile(url: zipURL)
                }
            } catch {
                await MainActor.run {
                    zippingValsetURL = nil
                    recoveryError = "Couldn't compress for sharing: \(error)"
                }
            }
        }
    }

    /// Zip with `NSFileCoordinator(.forUploading)`. The URL it hands over is valid only inside the
    /// closure, so copy it to a temporary directory and return that.
    nonisolated private static func zipDirectoryForSharing(_ dir: URL) throws -> URL {
        let coordinator = NSFileCoordinator()
        var coordError: NSError?
        var produced: Result<URL, Error>?
        coordinator.coordinate(readingItemAt: dir, options: [.forUploading],
                               error: &coordError) { zippedURL in
            let dest = FileManager.default.temporaryDirectory
                .appendingPathComponent(dir.lastPathComponent + ".zip")
            try? FileManager.default.removeItem(at: dest)
            do {
                try FileManager.default.copyItem(at: zippedURL, to: dest)
                produced = .success(dest)
            } catch {
                produced = .failure(error)
            }
        }
        if let coordError { throw coordError }
        switch produced {
        case .success(let url): return url
        case .failure(let err): throw err
        case .none:
            throw NSError(domain: "splatelier", code: -1,
                          userInfo: [NSLocalizedDescriptionKey: "Couldn't create the zip file"])
        }
    }

}

/// zip file reference for `.sheet(item:)`.
private struct ShareableFile: Identifiable {
    let id = UUID()
    let url: URL
}

/// Wrapper for `UIActivityViewController`. Needed because the shared item only exists after zipping
/// finishes, so ShareLink cannot be used.
private struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}

#Preview {
    NavigationStack { CaptureView() }
}
