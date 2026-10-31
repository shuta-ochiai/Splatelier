// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// The training part of the Capture screen: before training it shows what is being trained and the
// settings; then progress and, when done, the way into the viewer. It lives inside
// the Capture tab (no screen of its own); only the viewer is pushed on top.

import SwiftUI
import Trainer
import Capture
import SplatViewerKit

private struct TrainingHelp: Identifiable {
    let id: String
    let title: String
    let message: String

    static let cap = TrainingHelp(
        id: "cap",
        title: "Gaussian cap",
        message: "Maximum number of Gaussians. Training starts from 5/6 of it (initial points beyond that are thinned out) and grows to it by splitting along edges. Lower is lighter and runs cooler; higher keeps more detail."
    )
}

struct TrainingView: View {
    @ObservedObject var vm: TrainingViewModel
    /// What is being trained (the capture result or the stored dataset), shown above the settings
    /// before training starts. Type-erased like the cards below (see decoratedScroll).
    let header: AnyView
    /// Leaves the training flow and goes back to the start screen.
    let onClose: () -> Void
    /// A fresh capture is not saved until it is trained, so closing before that discards it. A
    /// stored dataset stays where it is, so closing is harmless.
    let closeDiscards: Bool
    @State private var activeHelp: TrainingHelp?

    /// Thermal banner. critical (thermalPaused) means paused, serious (thermalThrottled) means
    /// slowed down. critical takes display priority; the banner disappears once the device cools
    /// down.
    @ViewBuilder private var thermalPausedBanner: some View {
        if vm.thermalPaused {
            thermalBanner(
                icon: "thermometer.high",
                text: "Paused until the device cools down",
                tint: .orange
            )
        } else if vm.thermalThrottled {
            thermalBanner(
                icon: "thermometer.medium",
                text: "Slowed down: the device is hot",
                tint: .yellow
            )
        }
    }

    /// Whether Low Power Mode is on; it follows the setting if it is switched meanwhile.
    @State private var lowPowerMode = ProcessInfo.processInfo.isLowPowerModeEnabled

    /// Low Power Mode slows the CPU and GPU down, and training with them. Shown before it starts and
    /// while it runs.
    private var lowPowerNotice: some View {
        Label("Low Power Mode is on, which slows training down. Turn it off in Settings > Battery.",
              systemImage: "battery.25percent")
            .font(.footnote.weight(.semibold))
            .foregroundStyle(.yellow)
            .padding()
            .frame(maxWidth: .infinity, alignment: .leading)
            .cardBackground()
    }

    private func thermalBanner(icon: String, text: String, tint: Color) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
            Text(text)
                .font(.footnote.weight(.semibold))
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(.thinMaterial, in: Capsule())
        .foregroundStyle(tint)
        .shadow(radius: 4)
        .padding(.top, 8)
        .transition(.move(edge: .top).combined(with: .opacity))
    }

    var body: some View {
        decoratedScroll
    }

    private var decoratedScroll: some View {
        Group {
            // Without type erasure via AnyView the generic metadata gets deep, and on device the
            // SubstGenericParametersFromMetadata recursion overflows the stack.
            if isIdle {
                ScrollView {
                    VStack(spacing: 16) {
                        header
                        AnyView(trainingSettingsCard)
                        if lowPowerMode { AnyView(lowPowerNotice) }
                        AnyView(startButton)
                    }
                    .padding(.horizontal)
                    .padding(.top, 12)
                    .padding(.bottom, 12)
                }
                .scrollIndicators(.hidden)
            } else {
                // What the model looks like now (it sharpens as it trains), then the progress and
                // the measurements.
                VStack(spacing: 16) {
                    AnyView(previewCard).frame(maxHeight: .infinity)
                    if lowPowerMode && vm.isBusy { AnyView(lowPowerNotice) }
                    AnyView(phaseCard)
                    AnyView(metricsCard)
                }
                .padding(.horizontal)
                .padding(.top, 12)
                .padding(.bottom, 12)
            }
        }
        .overlay(alignment: .top) { thermalPausedBanner }
        .onReceive(NotificationCenter.default.publisher(for: .NSProcessInfoPowerStateDidChange)) { _ in
            lowPowerMode = ProcessInfo.processInfo.isLowPowerModeEnabled
        }
        .animation(.easeInOut(duration: 0.25), value: vm.thermalPaused)
        .animation(.easeInOut(duration: 0.25), value: vm.thermalThrottled)
        .safeAreaInset(edge: .bottom, spacing: 0) {
            actionRow
                .padding(.horizontal)
                .padding(.top, 10)
                .padding(.bottom, 10)
                .frame(maxWidth: .infinity)
                .background(.ultraThinMaterial)
        }
        // Do not touch the parent TabView's tab bar. It triggers a known bug where the layout
        // change at training start jumps to the Library tab on its own (paired with RootView's
        // explicit selection).
        .alert(item: $activeHelp) { help in
            Alert(
                title: Text(help.title),
                message: Text(help.message),
                dismissButton: .default(Text("OK"))
            )
        }
        // Naming at the end of training. When training from a validation dataset, this holds the
        // valset name.
        .alert("Name this capture", isPresented: $vm.showNamePrompt) {
            TextField("Name (optional)", text: $vm.captureName)
            Button("Set name") { vm.saveCaptureName() }
            Button("Default name", role: .cancel) {}
        } message: {
            Text("The result is already saved. Name it now, or keep the date and time as its name. You can rename it any time in the viewer or the Library.")
        }
        .onChange(of: vm.showNamePrompt) { _, shown in
            if !shown { vm.openViewerIfReady() }
        }
        // The viewer opens on its own once training and its mesh have finished.
        .navigationDestination(item: $vm.viewerTarget) { target in
            viewerScreen(target)
        }
    }

    /// Viewer for the trained capture, with sharing and "Train more" wired to this model. Naming
    /// can also be done with its rename button.
    private func viewerScreen(_ target: ViewerTarget) -> some View {
        var shareItems = [SplatExportShare.item(for: target.plyURL)]
        if let mesh = vm.meshOBJURL {
            shareItems.append(SplatViewerShareItem(
                title: "Mesh (OBJ)", systemImage: "triangle", url: mesh))
        }
        // The training log and the mesh log go out together as "Log".
        let logs = [target.logURL, vm.meshLogURL].compactMap { $0 }
        if !logs.isEmpty {
            shareItems.append(SplatViewerShareItem(title: "Log", systemImage: "doc.text", urls: logs))
        }
        return SplatViewerScreen(
            url: target.plyURL,
            seedURL: vm.seedPLYURL,
            meshURL: vm.meshOBJURL,
            name: CaptureMetadataName.load(for: target.plyURL),
            shareItems: shareItems,
            cropExport: SplatExportShare.cropExport,
            onTrainMore: vm.canContinueTraining && !vm.isGeneratingMesh ? {
                // Back to this screen's log, then continue; the viewer opens again when it ends.
                vm.viewerTarget = nil
                vm.continueTraining()
            } : nil,
            trainMoreMessage: "Trains \(TrainingViewModel.continueIterationCount) more iterations on this result, then rebuilds the mesh. It takes a minute or two, and the result is replaced.",
            onRename: { newName in
                vm.captureName = newName
                vm.saveCaptureName()
            }
        )
    }

    /// The model as it is now, from one view of the capture. Before the first picture there is a
    /// quiet placeholder, so the layout does not jump.
    private var previewCard: some View {
        ZStack {
            RoundedRectangle(cornerRadius: DesignSystem.cardRadius, style: .continuous)
                .fill(Color.black.opacity(0.35))
            if let image = vm.previewImage {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .clipShape(RoundedRectangle(cornerRadius: DesignSystem.cardRadius, style: .continuous))
                    .id(vm.previewCount)
                    .transition(.opacity)
            } else {
                VStack(spacing: 10) {
                    Image(systemName: "sparkles").font(.title2)
                    Text("The picture appears when training starts").font(.footnote)
                }
                .foregroundStyle(.secondary)
            }
        }
        .animation(.easeInOut(duration: 0.5), value: vm.previewCount)
        .accessibilityLabel("Preview of the model")
    }

    /// Time and GPU memory in one small line under the progress. The peak turns red above the 3 GB
    /// the PocketGS paper aims under.
    private var metricsCard: some View {
        HStack(spacing: 12) {
            metricItem("Elapsed", formatElapsed(vm.elapsedSec))
            metricItem("GPU peak", formatBytes(vm.peakMemoryBytes),
                       isMet: vm.peakMemoryBytes < 3 * 1024 * 1024 * 1024)
            metricItem("GPU active", formatBytes(vm.activeMemoryBytes))
        }
        .font(.caption2.monospacedDigit())
        .lineLimit(1)
        .minimumScaleFactor(0.8)
        .frame(maxWidth: .infinity)
    }

    private func metricItem(_ title: String, _ value: String, isMet: Bool = true) -> some View {
        HStack(spacing: 4) {
            Text(title).foregroundStyle(.secondary)
            // Ternary inference between HierarchicalShapeStyle and Color fails, so make Color
            // explicit.
            Text(value).foregroundStyle(isMet ? Color.primary : Color.red)
        }
    }

    private func formatElapsed(_ sec: Double) -> String {
        let m = Int(sec) / 60
        let s = Int(sec) % 60
        return String(format: "%d:%02d", m, s)
    }

    private func formatBytes(_ b: Int) -> String {
        let mb = Double(b) / (1024 * 1024)
        if mb >= 1024 { return String(format: "%.2f GB", mb / 1024) }
        return String(format: "%.0f MB", mb)
    }

    /// Iterations and the Gaussian cap in one card.
    private var trainingSettingsCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Training settings").font(.headline)
            iterationControls
            Divider()
            gaussianCapControls
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .cardBackground()
    }

    private var iterationControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            Stepper(value: $vm.iterations, in: 50...5000, step: 50) {
                HStack {
                    Text("Iterations").foregroundStyle(.secondary)
                    Spacer()
                    Text("\(vm.iterations)").font(.body.monospacedDigit())
                }
            }
            .disabled(!isIdle)
            // More than the paper's §III-C 500 iters, because convergence needs it when the seed is
            // sparser than the paper assumes.
            HStack(spacing: 6) {
                presetButton(label: "Draft", value: 1000)
                presetButton(label: "Standard", value: 3000)
                presetButton(label: "High", value: 5000)
            }
        }
    }

    private var gaussianCapControls: some View {
        VStack(alignment: .leading, spacing: 10) {
            settingHeader("Gaussian cap", help: .cap)
            Picker("Gaussian cap", selection: $vm.gaussianCap) {
                Text("100k").tag(100_000)
                Text("150k").tag(150_000)
                Text("300k").tag(300_000)
                Text("600k").tag(600_000)
            }
            .pickerStyle(.segmented)
            .disabled(!isIdle)
        }
    }

    /// Right under the settings, so the two read as one step.
    private var startButton: some View {
        Button(action: { vm.start() }) {
            Label("Start training", systemImage: "play.fill")
                .fontWeight(.semibold)
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(.borderedProminent)
        .tint(Color.kikyoFill)
        .controlSize(.large)
    }

    private func settingHeader(_ title: String, help: TrainingHelp) -> some View {
        HStack(spacing: 6) {
            Text(title).foregroundStyle(.secondary)
            helpButton(help)
            Spacer()
        }
    }

    private func helpButton(_ help: TrainingHelp) -> some View {
        Button {
            activeHelp = help
        } label: {
            Image(systemName: "info.circle")
                .font(.body)
                .foregroundStyle(.secondary)
                .frame(width: 28, height: 28)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("About \(help.title)")
        .simultaneousGesture(
            LongPressGesture(minimumDuration: 0.45).onEnded { _ in
                activeHelp = help
            }
        )
    }

    @ViewBuilder
    private func presetButton(label: String, value: Int) -> some View {
        let button = Button(label) { vm.iterations = value }
            .controlSize(.small)
            .disabled(!isIdle)
            .frame(maxWidth: .infinity)
        if vm.iterations == value {
            button.buttonStyle(.borderedProminent).tint(Color.kikyoFill)
        } else {
            button.buttonStyle(.bordered)
        }
    }

    private var phaseCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            if vm.isBusy {
                // One bar for the preprocessing, the training and the mesh.
                Label(vm.progressLabel.isEmpty ? "Working" : vm.progressLabel, systemImage: "hourglass")
                    .font(.headline).foregroundStyle(.orange)
                ProgressView(value: vm.overallProgress)
                HStack {
                    if !vm.progressDetail.isEmpty {
                        Text(vm.progressDetail).font(.body.monospacedDigit())
                    }
                    Spacer()
                    if case .running(_, let loss?) = vm.phase {
                        Text("loss \(String(format: "%.4f", loss))")
                            .font(.body.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                    Text("\(Int(vm.overallProgress * 100))%").font(.body.monospacedDigit())
                }
                // Training runs on the GPU, which iOS does not give to an app that is not on screen.
                Label("Keep Splatelier open on screen until it finishes.", systemImage: "iphone")
                    .font(.caption).foregroundStyle(.secondary)
            } else { switch vm.phase {
            case .idle:
                EmptyView()
            case .running:
                EmptyView()
            case .completed(let url, _, let count, let finalLoss):
                Label("Training complete", systemImage: "checkmark.seal.fill")
                    .font(.headline).foregroundStyle(.green)
                Text("\(count) gaussians").font(.body.monospacedDigit())
                Text("Total \(vm.cumulativeIterations) iter").font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                if let finalLoss {
                    Text("final loss \(String(format: "%.4f", finalLoss))")
                        .font(.body.monospacedDigit())
                }
                Text(url.lastPathComponent).font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .lineLimit(1).truncationMode(.middle)
            case .failed(let msg):
                Label("Training failed", systemImage: "xmark.octagon.fill")
                    .font(.headline).foregroundStyle(.red)
                Text(msg).font(.caption).foregroundStyle(.secondary)
            } }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .cardBackground()
    }

    private var actionRow: some View {
        VStack(spacing: 8) {
            switch vm.phase {
            case .idle:
                // Start training sits under the settings; only the way out is down here.
                closeButton
            case .running:
                ProgressView().frame(maxWidth: .infinity)
            case .completed(let url, let logURL, _, _):
                // Only View is left: sharing and "Train more" are in the viewer, and the mesh is
                // generated automatically after training.
                Button {
                    vm.viewerTarget = ViewerTarget(plyURL: url, logURL: logURL)
                } label: {
                    Label("View", systemImage: "eye")
                        .fontWeight(.semibold)
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(Color.kikyoFill)
                .controlSize(.large)
                .disabled(vm.isGeneratingMesh)
                meshCaption
                closeButton
            case .failed:
                Button(action: { vm.start() }) {
                    Label("Retry", systemImage: "arrow.clockwise")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                closeButton.padding(.top, 14)
            }
        }
    }

    /// Bottom center: discards the capture / closes the dataset before training or after a failure,
    /// and starts over with a new capture once the result is saved.
    private var closeButton: some View {
        let finished: Bool = { if case .completed = vm.phase { return true }; return false }()
        let title = finished ? "New capture" : (closeDiscards ? "Discard capture" : "Close")
        let symbol = finished ? "camera.viewfinder" : (closeDiscards ? "trash" : "xmark")
        let destructive = !finished && closeDiscards
        return Group {
            if destructive {
                // Plain red text, set apart from Start training so the two are not mixed up.
                Button(role: .destructive, action: onClose) {
                    Label(title, systemImage: symbol)
                        .font(.subheadline)
                        .padding(.horizontal, 12).padding(.vertical, 8)
                }
                .buttonStyle(.plain)
                .foregroundStyle(.red)
            } else {
                Button(action: onClose) {
                    Label(title, systemImage: symbol)
                        .padding(.horizontal, 8)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
            }
        }
        .disabled(vm.isBusy)
    }

    /// State of the automatic mesh generation (progress / failure reason / notice that it was
    /// generated). Display is left to the viewer's mesh layer and sharing to its Share menu.
    @ViewBuilder private var meshCaption: some View {
        switch vm.meshPhase {
        case .running:
            // Shown by the progress bar instead.
            EmptyView()
        case .failed(let message):
            Text("Mesh generation failed: \(message)")
                .font(.caption).foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
        case .completed(_, _, let triangleCount):
            Text("Mesh: \(triangleCount) triangles (open View to see it or share it)")
                .font(.caption).foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
        case .idle:
            EmptyView()
        }
    }

    private var isIdle: Bool {
        if case .idle = vm.phase { return true }
        return false
    }



}
