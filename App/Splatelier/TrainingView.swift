// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Screen that trains a CapturedScene. Shows the settings card, progress, the log tail and
// post-training actions.

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
    @StateObject private var vm: TrainingViewModel
    @State private var activeHelp: TrainingHelp?

    init(scene: CapturedScene, resumeDatasetURL: URL? = nil, preserveDataset: Bool = false,
         initialName: String = "") {
        _vm = StateObject(wrappedValue: TrainingViewModel(
            scene: scene, resumeDatasetURL: resumeDatasetURL, preserveDataset: preserveDataset,
            initialName: initialName))
    }

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
        ScrollView {
            // Without type erasure via AnyView the generic metadata gets deep, and on device the
            // SubstGenericParametersFromMetadata recursion overflows the stack.
            VStack(spacing: 16) {
                AnyView(iterationCountStepper)
                AnyView(gaussianCapCard)
                if !isIdle { AnyView(phaseCard) }
                if !isIdle { AnyView(metricsCard) }
                if !vm.logTail.isEmpty { AnyView(logTailCard) }
            }
            .padding(.horizontal)
            .padding(.top, 12)
            .padding(.bottom, 12)
        }
        .scrollIndicators(.hidden)
        .overlay(alignment: .top) { thermalPausedBanner }
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
        .navigationTitle("Train")
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(.visible, for: .navigationBar)
        // Also disables the swipe back. See vm.isBusy.
        .navigationBarBackButtonHidden(vm.isBusy)
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
            Button("Save") { vm.saveCaptureName() }
            Button("Skip", role: .cancel) {}
        } message: {
            Text("If left empty, the date and time name is kept. You can also rename it later in the Library.")
        }
    }

    /// Measurement display compared against the PocketGS paper's targets (4 min / <3 GB peak).
    private var metricsCard: some View {
        HStack(spacing: 16) {
            metricLabel(title: "Elapsed", value: formatElapsed(vm.elapsedSec),
                        target: "≤ 4 min",
                        isMet: vm.elapsedSec <= 240)
            Divider()
            metricLabel(title: "GPU peak", value: formatBytes(vm.peakMemoryBytes),
                        target: "< 3 GB",
                        isMet: vm.peakMemoryBytes < 3 * 1024 * 1024 * 1024)
            Divider()
            metricLabel(title: "GPU active", value: formatBytes(vm.activeMemoryBytes),
                        target: nil, isMet: true)
        }
        .padding()
        .cardBackground()
    }

    @ViewBuilder
    private func metricLabel(title: String, value: String, target: String?, isMet: Bool) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption2).foregroundStyle(.secondary)
            // Ternary inference between HierarchicalShapeStyle and Color fails, so make Color
            // explicit.
            Text(value).font(.body.monospacedDigit())
                .foregroundStyle(isMet ? Color.primary : Color.red)
            if let target {
                Text(target).font(.caption2).foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
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

    private var iterationCountStepper: some View {
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
        .padding()
        .cardBackground()
    }

    private var gaussianCapCard: some View {
        VStack(alignment: .leading, spacing: 12) {
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
        .padding()
        .cardBackground()
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
            button.buttonStyle(.borderedProminent)
        } else {
            button.buttonStyle(.bordered)
        }
    }

    private var phaseCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            switch vm.phase {
            case .idle:
                EmptyView()
            case .running(let iter, let loss):
                Label("Training", systemImage: "hourglass")
                    .font(.headline).foregroundStyle(.orange)
                ProgressView(value: Double(iter),
                             total: Double(max(1, vm.runningTotalIterations)))
                HStack {
                    Text("iter").foregroundStyle(.secondary)
                    Text("\(iter) / \(vm.runningTotalIterations)").font(.body.monospacedDigit())
                    Spacer()
                    if let loss {
                        Text("loss \(String(format: "%.4f", loss))")
                            .font(.body.monospacedDigit())
                    }
                }
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
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .cardBackground()
    }

    private var logTailCard: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(Array(vm.logTail.enumerated()), id: \.offset) { idx, line in
                        Text(line)
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)
                            .id(idx)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .onChange(of: vm.logTail.count) { _, _ in
                if let last = vm.logTail.indices.last {
                    withAnimation { proxy.scrollTo(last, anchor: .bottom) }
                }
            }
        }
        .frame(maxHeight: 180)
        .padding(8)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8))
    }

    private var actionRow: some View {
        HStack {
            switch vm.phase {
            case .idle:
                Button(action: { vm.start() }) {
                    Label("Start training", systemImage: "play.fill")
                        .fontWeight(.semibold)
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            case .running:
                ProgressView().frame(maxWidth: .infinity)
            case .completed(let url, let logURL, _, _):
                // Put View on the top row; secondary actions are at most 2 per row (with 3, "Train more"
                // (continue training) gets cut off). The middle row is processing (continue
                // training, mesh generation); the bottom row is sharing (Export, Log).
                VStack(spacing: 8) {
                    // Show the display name and rename too, same as when opened from Library.
                    NavigationLink {
                        SplatViewerScreen(
                            url: url,
                            seedURL: vm.seedPLYURL,
                            meshURL: vm.meshOBJURL,
                            name: CaptureMetadataName.load(for: url),
                            onRename: { newName in
                                vm.captureName = newName
                                vm.saveCaptureName()
                            }
                        )
                    } label: {
                        Label("View", systemImage: "eye")
                            .fontWeight(.semibold)
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    HStack(spacing: 8) {
                        if vm.canContinueTraining {
                            Button(action: { vm.continueTraining() }) {
                                secondaryLabel("Train more", systemImage: "plus.circle")
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.large)
                            .disabled(vm.isGeneratingMesh)
                        }
                        meshButton
                    }
                    HStack(spacing: 8) {
                        // Share the generated mesh (OBJ) together with the GS PLY.
                        ShareLink(items: [url] + (vm.meshOBJURL.map { [$0] } ?? [])) {
                            secondaryLabel("Export", systemImage: "square.and.arrow.up")
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.large)
                        if let logURL {
                            // With the mesh generation's log once there is one.
                            ShareLink(items: [logURL] + (vm.meshLogURL.map { [$0] } ?? [])) {
                                secondaryLabel("Log", systemImage: "doc.text")
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.large)
                        }
                    }
                    meshCaption
                }
            case .failed:
                Button(action: { vm.start() }) {
                    Label("Retry", systemImage: "arrow.clockwise")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
            }
        }
    }

    private func secondaryLabel(_ title: String, systemImage: String) -> some View {
        Label(title, systemImage: systemImage)
            .lineLimit(1)
            .minimumScaleFactor(0.8)
            .frame(maxWidth: .infinity)
    }

    /// Mesh generation (after training, from the trained Gaussians or its MVS). After generating,
    /// hide the button; display is left to View's mesh layer and sharing to Export.
    @ViewBuilder private var meshButton: some View {
        switch vm.meshPhase {
        case .idle, .failed:
            Button(action: { vm.generateMesh() }) {
                secondaryLabel("Generate mesh", systemImage: "cube")
            }
            .buttonStyle(.bordered)
            .controlSize(.large)
            .disabled(!vm.canGenerateMesh)
        case .running:
            Button(action: {}) {
                HStack(spacing: 6) {
                    ProgressView()
                    Text("Generating")
                }
                .lineLimit(1)
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .controlSize(.large)
            .disabled(true)
        case .completed:
            EmptyView()
        }
    }

    /// Result of mesh generation (failure reason / notice that it was generated).
    @ViewBuilder private var meshCaption: some View {
        switch vm.meshPhase {
        case .failed(let message):
            Text("Mesh generation failed: \(message)")
                .font(.caption).foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
        case .completed(_, _, let triangleCount):
            Text("Mesh: \(triangleCount) triangles (View · Export to share)")
                .font(.caption).foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
        case .idle, .running:
            EmptyView()
        }
    }

    private var isIdle: Bool {
        if case .idle = vm.phase { return true }
        return false
    }



}
