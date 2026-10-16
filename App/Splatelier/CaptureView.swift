// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Capture screen. Handles pre-capture settings, the live AR preview during capture, the result card
// after capture, crash recovery and the list of validation datasets (.valset).

import SwiftUI
import UIKit
import Capture

struct CaptureView: View {
    @StateObject private var vm = CaptureViewModel()
    @State private var showDiscardConfirm = false
    @State private var showFrameReview = false

    // MARK: crash recovery
    /// Preserved dataset (*.dataset) left over from a previous crash. Detected at launch.
    @State private var pendingDatasetURL: URL?
    @State private var showResumePrompt = false
    @State private var didCheckRecovery = false
    @State private var resumeTarget: ResumeTarget?
    @State private var recoveryError: String?

    // MARK: validation datasets
    @State private var validationSets: [ValidationEntry] = []
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

    /// Restored result used to push TrainingView. With `preserve` (validation dataset), the dataset
    /// is not deleted after training finishes.
    private struct ResumeTarget: Identifiable, Hashable {
        let id = UUID()
        let scene: CapturedScene
        let datasetURL: URL
        let preserve: Bool
        let name: String   // Display name to carry over; "" if none

        // CapturedScene is not Hashable, so compare equality by id.
        static func == (lhs: ResumeTarget, rhs: ResumeTarget) -> Bool { lhs.id == rhs.id }
        func hash(into hasher: inout Hasher) { hasher.combine(id) }
    }

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
        .onAppear {
            checkForRecoverableDataset()
            refreshValidationSets()
            // Compile the shaders for the splat display during capture ahead of time (so the first
            // capture start does not wait)
            CaptureSurfelRenderer.prewarm()
        }
        .navigationDestination(item: $resumeTarget) { target in
            TrainingView(scene: target.scene, resumeDatasetURL: target.datasetURL,
                         preserveDataset: target.preserve, initialName: target.name)
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
        // A captured scene is not saved until it is trained, so confirm before recapturing.
        .confirmationDialog(
            "Discard this capture?",
            isPresented: $showDiscardConfirm,
            titleVisibility: .visible
        ) {
            Button("Discard and go back", role: .destructive) { vm.discardCapturedScene() }
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

    private var setupView: some View {
        VStack(spacing: 24) {
            if let scene = vm.capturedScene {
                resultCard(scene)
                if let err = vm.errorMessage { errorRow(err) }
                Spacer()
            } else {
                // Keep the card fixed on screen and scroll only the validation dataset list inside
                // it.
                VStack(spacing: 24) {
                    if !validationSets.isEmpty { validationDatasetsCard }
                    if let err = vm.errorMessage { errorRow(err) }
                }
                .frame(maxHeight: .infinity, alignment: .top)
            }
            controlRow
        }
        .padding()
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
        VStack(alignment: .leading, spacing: 8) {
            Label("Capture complete", systemImage: "checkmark.circle.fill")
                .font(.headline)
                .foregroundStyle(.green)
            row("Frames", "\(scene.frames.count)")
            row("Sparse points", "\(scene.sparsePoints.count)")
            row("Tracking", vm.diagnostics.trackingStateDescription)
            row("Anchor updates", "\(vm.diagnostics.anchorUpdateCount)")
            row("Unsettled anchors", "\(vm.diagnostics.unsettledAnchorCount)")
            if vm.diagnostics.relocalizationOccurred {
                Label("Relocalization observed",
                      systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
            Button { showFrameReview = true } label: {
                Label("Review frames", systemImage: "photo.stack")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .controlSize(.large)
            .padding(.top, 4)
            validationSaveRow
            NavigationLink(destination: TrainingView(scene: scene)) {
                Label("Train this capture", systemImage: "wand.and.stars")
                    .fontWeight(.semibold)
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .padding(.top, 4)
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

    @ViewBuilder
    private var validationSaveRow: some View {
        switch vm.validationSaveState {
        case .saved(let name):
            Label("Saved for validation (\(name))", systemImage: "checkmark.circle.fill")
                .font(.caption)
                .foregroundStyle(.green)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 4)
        default:
            let saving = vm.validationSaveState == .saving
            Button { valsetNameText = ""; showValsetNamePrompt = true } label: {
                HStack(spacing: 6) {
                    if saving { ProgressView() }
                    Label(saving ? "Saving…" : "Save for validation", systemImage: "flask")
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .controlSize(.large)
            .disabled(saving)
            .padding(.top, 4)
            .alert("Validation dataset name", isPresented: $showValsetNamePrompt) {
                TextField("e.g. Sofa test (optional)", text: $valsetNameText)
                Button("Save") { vm.saveValidationDataset(displayName: valsetNameText) }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("If left empty, the date and time are used.")
            }
            if case .failed(let msg) = vm.validationSaveState {
                Text("Couldn't save: \(msg)").font(.caption).foregroundStyle(.red)
            }
        }
    }

    /// List of saved *.valset. Loading one proceeds to training.
    private var validationDatasetsCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Validation datasets", systemImage: "flask")
                .font(.headline)
            Text("Train on saved captures as often as you like (for comparing settings).")
                .font(.caption)
                .foregroundStyle(.secondary)
            // While it fits, shrink the card to its content height; only when it overflows, scroll
            // within the remaining height.
            ViewThatFits(in: .vertical) {
                validationRows
                ScrollView {
                    validationRows
                }
                .scrollIndicatorsFlash(onAppear: true)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .cardBackground()
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
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            case .stopping:
                ProgressView("Finalising…")
                    .frame(maxWidth: .infinity)
            }
        }
    }

    /// After capture, the primary action goes to "Train this capture" and this becomes the
    /// secondary action (discard the captured scene and return to the initial screen).
    @ViewBuilder
    private var captureStartButton: some View {
        let hasScene = vm.capturedScene != nil
        let button = Button(action: { if hasScene { showDiscardConfirm = true } else { vm.start() } }) {
            Label(hasScene ? "Capture again" : "Start capture",
                  systemImage: hasScene ? "arrow.counterclockwise" : "record.circle")
                .frame(maxWidth: .infinity)
        }
        .controlSize(.large)
        if hasScene {
            button.buttonStyle(.bordered)
        } else {
            button.buttonStyle(.borderedProminent)
        }
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
        guard !isLive, vm.capturedScene == nil,
              let url = Self.mostRecentPendingDataset() else { return }
        pendingDatasetURL = url
        showResumePrompt = true
    }

    /// Decode the preserved dataset in the background and navigate to TrainingView.
    private func resumePendingDataset() {
        guard let url = pendingDatasetURL else { return }
        Task.detached {
            do {
                let scene = try CaptureDatasetStore.load(from: url)
                await MainActor.run {
                    resumeTarget = ResumeTarget(
                        scene: scene, datasetURL: url, preserve: false,
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

    /// Load a validation dataset in the background and navigate to TrainingView.
    private func loadValidationDataset(_ url: URL) {
        guard loadingValidationURL == nil else { return }
        loadingValidationURL = url
        Task.detached {
            do {
                let scene = try CaptureDatasetStore.load(from: url)
                await MainActor.run {
                    loadingValidationURL = nil
                    resumeTarget = ResumeTarget(
                        scene: scene, datasetURL: url, preserve: true,
                        name: CaptureMetadataName.load(for: url) ?? "")
                }
            } catch {
                await MainActor.run {
                    loadingValidationURL = nil
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
