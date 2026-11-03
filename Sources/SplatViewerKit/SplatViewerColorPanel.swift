// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Colour grade panel for the foreground splats. It sits over the bottom of the viewer rather than
// covering it, so the model stays in view while the sliders move. Each change is applied after the
// slider settles, so the splats update as they are adjusted without a rebuild on every frame.

#if os(iOS) || os(macOS)

import Foundation
import SwiftUI
import simd

/// Buttons at the foot of the Color and Crop panels. Reset, Undo and Restore sit on the left as
/// plain buttons; the one that finishes the panel, Apply, is the filled button on the right.
struct SplatViewerPanelButtonStyle: ButtonStyle {
    static let fill = Color(red: 0x7C / 255, green: 0x5C / 255, blue: 0xE6 / 255)
    var prominent = false

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            // One line at its full width: a tight row must not break a label mid-word.
            .lineLimit(1)
            .fixedSize()
            .font(.caption.weight(prominent ? .semibold : .regular))
            .foregroundStyle(prominent ? Color.white : Color.primary)
            .padding(.horizontal, 10).padding(.vertical, 6)
            .background(
                prominent ? AnyShapeStyle(Self.fill) : AnyShapeStyle(.ultraThinMaterial),
                in: Capsule())
            .opacity(configuration.isPressed ? 0.7 : 1)
    }
}

struct SplatViewerColorPanel: View {
    @ObservedObject var model: SplatViewerModel
    let onClose: () -> Void
    @Environment(\.self) private var environment
    @State private var draft: SplatViewerColorGrade
    /// Pending apply. A new change cancels it, so only the value the slider settles on is baked.
    @State private var pendingApply: Task<Void, Never>?
    /// The grade this panel last handed to the model. The model's value comes back through
    /// onChange; only a value that differs from this (Reset in the viewer) is copied into the sliders,
    /// so a move made while an apply was in flight is not overwritten.
    @State private var lastApplied: SplatViewerColorGrade
    /// The panel fades while the sliders are moving, so the splats behind it can be seen change.
    @State private var dimmed = false
    @State private var dimRestore: Task<Void, Never>?

    init(model: SplatViewerModel, onClose: @escaping () -> Void) {
        self.model = model
        self.onClose = onClose
        _draft = State(initialValue: model.colorGrade)
        _lastApplied = State(initialValue: model.colorGrade)
    }

    var body: some View {
        VStack(spacing: 8) {
            Text("Color")
                .font(.caption.weight(.semibold))
                .frame(maxWidth: .infinity, alignment: .leading)
            ScrollView {
                VStack(spacing: 10) {
                    ColorPicker("Tint", selection: tintBinding, supportsOpacity: false)
                        .font(.caption)
                    slider("Temperature", $draft.temperature, -0.5...0.5, format: signed)
                    slider("Saturation", $draft.saturation, 0...2, format: percent)
                    slider("Brightness", $draft.brightness, -1...1, format: signed)
                    slider("Black point", $draft.blackPoint, 0...1, format: percent)
                    slider("White point", $draft.whitePoint, 0...1, format: percent)
                    slider("Opacity", $draft.transparency, -6...6, format: opacityFactor)
                    if !draft.isValid {
                        Text("White point must be above black point. The grade is not applied until it is.")
                            .font(.caption)
                            .foregroundStyle(.yellow)
                    }
                }
            }
            .frame(maxHeight: 280)
            HStack(spacing: 6) {
                Button {
                    draft = .identity
                } label: {
                    Label("Reset", systemImage: "arrow.counterclockwise")
                }
                .buttonStyle(SplatViewerPanelButtonStyle())
                .disabled(draft.isIdentity)
                Spacer(minLength: 0)
                // The grade is already applied as the sliders move. Finish settles any pending change and leaves Color.
                Button {
                    pendingApply?.cancel()
                    lastApplied = draft
                    model.applyColorGrade(draft)
                    onClose()
                } label: {
                    Label("Finish", systemImage: "flag.checkered")
                }
                .buttonStyle(SplatViewerPanelButtonStyle(prominent: true))
                .disabled(!draft.isValid)
            }
        }
        .padding(14)
        .frame(maxWidth: 360)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .padding(.horizontal, 12)
        .opacity(dimmed ? 0.2 : 1)
        .animation(.easeOut(duration: 0.15), value: dimmed)
        .onChange(of: draft) { _, newValue in
            scheduleApply(newValue)
            fadeWhileChanging()
        }
        // Reset on the viewer puts the grade back to none; keep the sliders in step with it.
        .onChange(of: model.colorGrade) { _, newValue in
            guard newValue != lastApplied else { return }
            lastApplied = newValue
            draft = newValue
        }
    }

    /// Dims the panel while the values move, and brings it back once they have been still briefly.
    /// Slider's drag-end callback is not reliable here, so this goes by the values themselves.
    private func fadeWhileChanging() {
        dimmed = true
        dimRestore?.cancel()
        dimRestore = Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(400))
            guard !Task.isCancelled else { return }
            dimmed = false
        }
    }

    /// Applies the grade once the slider has been still for a moment.
    private func scheduleApply(_ grade: SplatViewerColorGrade) {
        pendingApply?.cancel()
        pendingApply = Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(150))
            guard !Task.isCancelled, grade.isValid else { return }
            lastApplied = grade
            model.applyColorGrade(grade)
        }
    }

    private var tintBinding: Binding<Color> {
        Binding(
            get: { Color(red: Double(draft.tint.x), green: Double(draft.tint.y), blue: Double(draft.tint.z)) },
            set: { color in
                let resolved = color.resolve(in: environment)
                draft.tint = SIMD3(resolved.red, resolved.green, resolved.blue)
            })
    }

    private func slider(
        _ title: LocalizedStringKey, _ value: Binding<Float>, _ range: ClosedRange<Float>,
        format: @escaping (Float) -> String
    ) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title)
                Spacer()
                Text(format(value.wrappedValue))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            .font(.caption)
            Slider(value: value, in: range)
        }
    }

    private func signed(_ v: Float) -> String { String(format: "%+.2f", v) }
    private func percent(_ v: Float) -> String { String(format: "%.0f%%", v * 100) }
    private func opacityFactor(_ v: Float) -> String { String(format: "×%.2f", exp(v)) }
}

#endif
