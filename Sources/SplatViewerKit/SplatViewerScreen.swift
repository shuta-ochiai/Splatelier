// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Public entry point. SwiftUI screen that shows a trained PLY (any format SplatIO can read) with an
// orbit camera. The seed point cloud, capture cameras and mesh (OBJ) can be overlaid in the same
// scene and toggled from the "Display" row, where the trained background can be hidden and the mesh
// switched to shaded. The "Measure" button at the bottom enters measure mode to measure the distance
// between 2 tapped points. Double tap (double click on macOS) zooms in, making that point the center
// of rotation.

import SwiftUI

#if os(iOS) || os(macOS)

import MetalKit

#if os(iOS)
import UIKit
#endif

#if os(macOS)
private typealias SplatViewRepresentable = NSViewRepresentable
#else
private typealias SplatViewRepresentable = UIViewRepresentable
#endif

public struct SplatViewerScreen: View {
    private let url: URL?
    private let seedURL: URL?
    private let meshURL: URL?
    private let shareItems: [SplatViewerShareItem]
    /// Handler for the "Train more" button. nil hides the button (the screen has no training
    /// session to continue).
    private let onTrainMore: (() -> Void)?
    /// What the confirmation before "Train more" tells the user.
    private let trainMoreMessage: String
    /// Receiver of rename notifications. nil hides the editing UI (saving the sidecar is the app's
    /// job).
    private let onRename: ((String) -> Void)?
    /// Held in @State so edits show immediately.
    @State private var displayName: String
    @State private var showRename = false
    @State private var showHint = false
    @State private var showTrainMoreConfirm = false
    @State private var showEditOptions = false
    @State private var showDisplayOptions = false
    @State private var showColorHelp = false
    @State private var showCropHelp = false
    @State private var showFloaterHelp = false
    @State private var renameText = ""
    @StateObject private var model: SplatViewerModel
    #if os(iOS)
    @StateObject private var sharer = SplatViewerSharePreparer()
    #endif

#if os(macOS)
    // Values at gesture start. drag / pinch are handled as absolute amounts from the start point
    // (macOS only).
    @State private var dragBase: SIMD2<Float>?
    @State private var zoomBase: Float?
    /// The drag in progress moves a crop handle, and the translation applied so far.
    @State private var cropDragging = false
    @State private var lastDragTranslation = CGSize.zero
    /// Size of the MTKView, for normalizing click positions.
    @State private var metalViewSize: CGSize = .zero
#endif

    /// - Parameters:
    ///   - url: The GS PLY. If nil, only the mesh (and seed) are shown (e.g. checking the mesh
    ///     before training).
    ///   - seedURL: PLY of the seed point cloud + capture-camera markers. If given, merged into
    ///     the same scene and toggleable.
    ///   - meshURL: OBJ of a vertex-colored mesh. If given, overlaid in the same scene and
    ///     toggleable.
    ///   - name: Display name (file name if nil).
    ///   - shareItems: Files offered by the Share button, one entry each. Empty hides the button.
    ///   - onTrainMore: Called when the user confirms "Train more". nil hides the button.
    ///   - trainMoreMessage: Shown in the confirmation (what will happen, how long it takes).
    public init(
        url: URL?,
        seedURL: URL? = nil,
        meshURL: URL? = nil,
        name: String? = nil,
        shareItems: [SplatViewerShareItem] = [],
        cropExport: SplatViewerCropExport? = nil,
        onTrainMore: (() -> Void)? = nil,
        trainMoreMessage: String = "Training continues from this result.",
        onRename: ((String) -> Void)? = nil
    ) {
        _model = StateObject(wrappedValue: SplatViewerModel(cropExport: cropExport))
        self.url = url
        self.seedURL = seedURL
        self.meshURL = meshURL
        self.shareItems = shareItems
        self.onTrainMore = onTrainMore
        self.trainMoreMessage = trainMoreMessage
        self.onRename = onRename
        _displayName = State(initialValue: name ?? Self.fileName(url ?? meshURL))
    }

    private static func fileName(_ url: URL?) -> String {
        url?.deletingPathExtension().lastPathComponent ?? "Viewer"
    }

    public var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            SplatMetalView(url: url, seedURL: seedURL, meshURL: meshURL, model: model)
                .ignoresSafeArea()
            #if os(macOS)
                .gesture(orbitGesture)
                .simultaneousGesture(zoomGesture)
                .simultaneousGesture(tapGesture)
                .simultaneousGesture(focusTapGesture)
                .onGeometryChange(for: CGSize.self) { $0.size } action: { metalViewSize = $0 }
            #endif

            SplatMeasureOverlay(projection: model.measureProjection)
                .ignoresSafeArea()
            SplatCropOverlay(projection: model.cropProjection)
                .ignoresSafeArea()
            SplatBrushOverlay(projection: model.brushProjection)
                .ignoresSafeArea()

            overlay
            #if os(iOS)
            preparingOverlay
            #endif
        }
        #if os(iOS)
        .sheet(item: $sharer.ready) { file in SplatViewerShareSheet(url: file.url) }
        .alert("Couldn't share", isPresented: Binding(
            get: { sharer.errorMessage != nil }, set: { if !$0 { sharer.errorMessage = nil } })
        ) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(sharer.errorMessage ?? "")
        }
        #endif
        .navigationTitle(displayName)
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(.visible, for: .navigationBar)
        #endif
        .toolbar {
            ToolbarItemGroup(placement: toolbarPlacement) {
                hintButton
                if !allShareItems.isEmpty { shareMenu }
                if onTrainMore != nil {
                    // Training is long and replaces this result, so ask before starting it.
                    Button { showTrainMoreConfirm = true } label: {
                        Label("Train more", systemImage: "plus.circle")
                    }
                }
                if onRename != nil {
                    Button {
                        renameText = displayName
                        showRename = true
                    } label: {
                        Label("Rename", systemImage: "pencil")
                    }
                }
            }
        }
        .alert("Train more?", isPresented: $showTrainMoreConfirm) {
            Button("Train more") { onTrainMore?() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(trainMoreMessage)
        }
        .alert("Rename", isPresented: $showRename) {
            TextField("Name", text: $renameText)
            Button("Save") {
                let trimmed = renameText.trimmingCharacters(in: .whitespacesAndNewlines)
                displayName = trimmed.isEmpty ? Self.fileName(url ?? meshURL) : trimmed
                onRename?(trimmed)
            }
            Button("Cancel", role: .cancel) {}
        }
    }

    private var toolbarPlacement: ToolbarItemPlacement {
        #if os(iOS)
        .topBarTrailing
        #else
        .automatic
        #endif
    }

    /// One entry per thing to share (GS, mesh, logs). An entry with several options (the GS in
    /// different formats) is a submenu. Options that are made when picked show a progress overlay
    /// and then the share sheet.
    /// The cropped PLY comes first while a crop is applied, then the files the app passed in.
    private var allShareItems: [SplatViewerShareItem] {
        (model.cropShareItem.map { [$0] } ?? []) + shareItems
    }

    private var shareMenu: some View {
        Menu {
            ForEach(allShareItems) { item in
                if item.options.count == 1, let option = item.options.first {
                    optionControl(option, label: Label(item.title, systemImage: item.systemImage))
                } else {
                    Menu {
                        ForEach(item.options) { option in
                            optionControl(option, label: Text(option.title))
                        }
                    } label: {
                        Label(item.title, systemImage: item.systemImage)
                    }
                }
            }
        } label: {
            Label("Share", systemImage: "square.and.arrow.up")
        }
    }

    @ViewBuilder
    private func optionControl<Content: View>(
        _ option: SplatViewerShareItem.Option, label: Content
    ) -> some View {
        switch option.source {
        case .file(let url):
            ShareLink(item: url) { label }
        case .files(let urls):
            ShareLink(items: urls) { label }
        case .prepared(let prepare):
            #if os(iOS)
            Button { sharer.prepare(title: option.title, prepare) } label: { label }
            #endif
        }
    }

    /// Opens a popover that lists the camera controls.
    private var hintButton: some View {
        Button {
            showHint = true
        } label: {
            Label("Hint", systemImage: "questionmark.circle")
        }
        .popover(isPresented: $showHint) {
            hintContent
                .presentationCompactAdaptation(.popover)
        }
    }

    private var hintContent: some View {
        VStack(alignment: .leading, spacing: 10) {
            #if os(iOS)
            hintRow("hand.point.up.left", "Rotate", "Drag with 1 finger")
            hintRow("hand.draw", "Pan", "Drag with 2 fingers")
            hintRow("arrow.up.left.and.arrow.down.right", "Zoom", "Pinch")
            hintRow("scope", "Focus", "Double-tap a point to zoom in and orbit around it")
            hintRow("ruler", "Measure", "Turn on Measure, then tap 2 points")
            hintRow("crop", "Crop", "Edit, then Crop, then drag a handle of the box or sphere")
            hintRow("aqi.medium", "Floaters", "Edit, then Floaters, then tap a floater or Auto Detect")
            #else
            hintRow("cursorarrow.motionlines", "Rotate", "Drag")
            hintRow("arrow.up.left.and.arrow.down.right", "Zoom", "Pinch or scroll")
            hintRow("scope", "Focus", "Double-click a point to zoom in and orbit around it")
            hintRow("ruler", "Measure", "Turn on Measure, then click 2 points")
            hintRow("crop", "Crop", "Edit, then Crop, then drag a handle of the box or sphere")
            hintRow("aqi.medium", "Floaters", "Edit, then Floaters, then click a floater or Auto Detect")
            #endif
        }
        .padding(16)
        .frame(minWidth: 260)
    }

    private func hintRow(_ symbol: String, _ title: String, _ detail: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: symbol)
                .frame(width: 22)
                .foregroundStyle(Color.accentColor)
            VStack(alignment: .leading, spacing: 1) {
                Text(title).font(.subheadline.weight(.semibold))
                Text(detail).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    #if os(iOS)
    /// Shown while a share option is being prepared (a format conversion can take a while).
    @ViewBuilder
    private var preparingOverlay: some View {
        if let title = sharer.preparing {
            ZStack {
                Color.black.opacity(0.45).ignoresSafeArea()
                VStack(spacing: 14) {
                    ProgressView()
                    Text("Preparing \(title)…").font(.subheadline)
                    Button("Cancel") { sharer.cancel() }
                        .buttonStyle(.bordered)
                }
                .padding(24)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
            }
            .transition(.opacity)
        }
    }
    #endif

    // MARK: Overlay

    @ViewBuilder
    private var overlay: some View {
        switch model.loadState {
        case .loading:
            ProgressView("Loading…")
                .padding(16)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        case .empty:
            status(symbol: "cube.transparent", title: "No splats", detail: "This file has no Gaussians to show")
        case .failed(let msg):
            status(symbol: "exclamationmark.triangle", title: "Couldn't load", detail: msg)
        case .loaded:
            // Top to bottom: measurement hint (while measuring) → Display / Measure / Reset.
            VStack {
                Spacer()
                if model.isMeasuring {
                    measureControls
                }
                if model.hasGS && showEditOptions {
                    editOptions
                }
                if model.isCropping {
                    cropControls
                }
                if model.isRemovingFloaters {
                    floaterControls
                }
                if model.isColoring {
                    SplatViewerColorPanel(model: model) { model.isColoring = false }
                        .padding(.bottom, 4)
                }
                if hasDisplayOptions && showDisplayOptions {
                    displayOptions
                }
                bottomBar
                    .padding(.horizontal, 12).padding(.vertical)
            }
        }
    }

    /// Display reveals a row of layer switches just above the bottom bar rather than a Menu: a Menu
    /// covers the view, so a switch couldn't be seen taking effect.
    private var displayButton: some View {
        toggleCapsule("Display", systemImage: "eye", isOn: showDisplayOptions) {
            showDisplayOptions.toggle()
        }
    }

    /// One line of layer switches, so the model stays in view while they are flipped. It scrolls
    /// sideways when it doesn't fit and is centered when it does. Background and Shaded are disabled
    /// (not hidden) while their layer is off, so the switches don't move.
    private var displayOptions: some View {
        let row = HStack(spacing: 6) {
            if model.hasGS {
                toggleCapsule("Splats", systemImage: "cube.fill", isOn: model.showGS) {
                    model.showGS.toggle()
                }
                if model.hasBackground {
                    toggleCapsule("Background", systemImage: "globe", isOn: model.showBackground) {
                        model.showBackground.toggle()
                    }
                    .disabled(!model.showGS)
                    .opacity(model.showGS ? 1 : 0.4)
                }
            }
            if model.hasMesh {
                if model.hasGS { displayDivider }
                toggleCapsule("Mesh", systemImage: "triangle", isOn: model.showMesh) {
                    model.showMesh.toggle()
                }
                toggleCapsule("Shaded", systemImage: "light.max", isOn: model.meshShaded) {
                    model.meshShaded.toggle()
                }
                .disabled(!model.showMesh)
                .opacity(model.showMesh ? 1 : 0.4)
            }
            if model.hasSeed || model.hasCameraMarkers {
                if model.hasGS || model.hasMesh { displayDivider }
                if model.hasSeed {
                    toggleCapsule("Seed points", systemImage: "circle.grid.3x3.fill", isOn: model.showSeed) {
                        model.showSeed.toggle()
                    }
                }
                if model.hasCameraMarkers {
                    toggleCapsule("Cameras", systemImage: "camera.fill", isOn: model.showCameraMarkers) {
                        model.showCameraMarkers.toggle()
                    }
                }
            }
        }
        .padding(.horizontal, 12)
        return ViewThatFits(in: .horizontal) {
            row
            ScrollView(.horizontal, showsIndicators: false) { row }
        }
    }

    /// Separates the splat, mesh and capture groups in the Display row.
    private var displayDivider: some View {
        Divider().frame(height: 16)
    }

    /// Bottom edge: Display, Edit, Measure and Reset, centered. The splat and triangle counts are
    /// no longer shown here.
    private var bottomBar: some View {
        trailingControls
            .frame(maxWidth: .infinity)
    }

    private var hasDisplayOptions: Bool {
        model.hasGS || model.hasSeed || model.hasCameraMarkers || model.hasMesh
    }

    /// Display and Edit buttons, measure-mode toggle and Reset.
    private var trailingControls: some View {
        HStack(spacing: 6) {
            if hasDisplayOptions {
                displayButton
            }
            toggleCapsule("Measure", systemImage: "ruler", isOn: model.isMeasuring) {
                model.isMeasuring.toggle()
            }
            if model.hasGS {
                editButton
            }
            Button {
                model.resetCamera()
                resetEdits()
            } label: {
                Label("Reset", systemImage: "arrow.counterclockwise")
                    .font(.caption)
                    .padding(.horizontal, 10).padding(.vertical, 6)
                    .background(.ultraThinMaterial, in: Capsule())
            }
            .buttonStyle(.plain)
        }
    }

    /// Edits to the foreground splats. Edit reveals a row with Color and Crop just above the bottom
    /// bar. A plain button rather than a Menu: the menu's list opened offset from the button.
    private var editButton: some View {
        toggleCapsule("Edit", systemImage: "slider.horizontal.3", isOn: showEditOptions) {
            showEditOptions.toggle()
            // Closing Edit closes what it opened, so no panel is left behind without its row.
            if !showEditOptions {
                model.isCropping = false
                model.isColoring = false
                model.isRemovingFloaters = false
            }
        }
    }

    /// Color, Crop and Floaters, shown once Edit is on. Color opens its panel; Crop turns on the
    /// handles; Floaters makes taps select floaters.
    /// Each has a ? beside it that explains what it does, so the explanations stay out of the way.
    private var editOptions: some View {
        HStack(spacing: 6) {
            HStack(spacing: 2) {
                toggleCapsule("Color", systemImage: "paintpalette", isOn: model.isColoring) {
                    model.isColoring.toggle()
                }
                helpButton(isPresented: $showColorHelp,
                           text: "Change the tint, temperature, saturation, brightness, levels and opacity. The splats update as you move a slider. Reset returns to the original colours.")
            }
            HStack(spacing: 2) {
                toggleCapsule("Crop", systemImage: "crop", isOn: model.isCropping) {
                    model.isCropping.toggle()
                }
                helpButton(isPresented: $showCropHelp,
                           text: "Drag a handle to resize the box or sphere. Drag anywhere else to rotate the view. Keep inside or remove inside decides which splats are kept.")
            }
            HStack(spacing: 2) {
                toggleCapsule("Floaters", systemImage: "aqi.medium", isOn: model.isRemovingFloaters) {
                    model.isRemovingFloaters.toggle()
                }
                helpButton(isPresented: $showFloaterHelp,
                           text: "Tap a floater to select the splats around it (orange); Selection size sets how far. Tap orange to unselect, or ⊗ to clear. Auto Detect selects the small floaters apart from the scene for you. Delete removes the orange splats. Undo takes back the last step, a selection or a delete; Reset brings back every splat.")
            }
        }
    }

    /// A ? that shows a short explanation in a popover.
    private func helpButton(isPresented: Binding<Bool>, text: LocalizedStringKey) -> some View {
        Button {
            isPresented.wrappedValue = true
        } label: {
            Image(systemName: "questionmark.circle")
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(6)
                .background(.ultraThinMaterial, in: Circle())
        }
        .buttonStyle(.plain)
        .popover(isPresented: isPresented) {
            // Size the text to its full height. A popover on iPhone otherwise clips it to one line.
            Text(text)
                .font(.callout)
                .multilineTextAlignment(.leading)
                .fixedSize(horizontal: false, vertical: true)
                .frame(width: 260, alignment: .leading)
                .padding()
                .presentationCompactAdaptation(.popover)
        }
    }

    /// Reset also undoes the edits: the colour grade goes back to none, and a crop is restored to
    /// every splat as loaded. Edits are not undone one step at a time; that is what Undo is for.
    private func resetEdits() {
        model.applyColorGrade(.identity)
        if model.cropEdited {
            model.onCropRestore?()
        }
    }

    /// Measure-mode hint, plus undo and clear-all for measurement points.
    private var measureControls: some View {
        HStack(spacing: 8) {
            Text(model.measureNotice ?? measureHint)
                .font(.caption)
                .foregroundStyle(model.measureNotice == nil ? Color.primary : Color.yellow)
                .padding(.horizontal, 10).padding(.vertical, 6)
                .background(.ultraThinMaterial, in: Capsule())
            if !model.measurePoints.isEmpty {
                Button {
                    model.undoMeasurePoint()
                } label: {
                    Label("Undo", systemImage: "arrow.uturn.backward")
                        .font(.caption)
                        .padding(.horizontal, 10).padding(.vertical, 6)
                        .background(.ultraThinMaterial, in: Capsule())
                }
                .buttonStyle(.plain)
                Button {
                    model.clearMeasurements()
                } label: {
                    Label("Clear all", systemImage: "trash")
                        .font(.caption)
                        .padding(.horizontal, 10).padding(.vertical, 6)
                        .background(.ultraThinMaterial, in: Capsule())
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.bottom, 4)
    }

    private var measureHint: String {
        #if os(iOS)
        let action = "Tap"
        #else
        let action = "Click"
        #endif
        return model.measurePoints.count.isMultiple(of: 2)
            ? "\(action) to place the start point"
            : "\(action) to place the end point"
    }

    /// Crop-mode controls: the hint, the shape and keep/remove pickers, and Apply with Undo and
    /// Restore (the last only once something is cropped).
    /// The crop panel. It has the same card and foot row as the Color panel: a title, the two
    /// settings, then Undo and Restore on the left and Apply on the right. Undo and Restore stay in
    /// place and are disabled when there is nothing to undo or restore, so the layout doesn't jump.
    private var cropControls: some View {
        VStack(spacing: 10) {
            Text("Crop")
                .font(.caption.weight(.semibold))
                .frame(maxWidth: .infinity, alignment: .leading)
            Picker("Shape", selection: Binding(
                get: { model.cropShape }, set: { model.setCropShape($0) })
            ) {
                ForEach(SplatViewerCropShape.allCases) { Text($0.title).tag($0) }
            }
            .pickerStyle(.segmented)
            Picker("Mode", selection: $model.cropKeepsInside) {
                Text("Keep inside").tag(true)
                Text("Remove inside").tag(false)
            }
            .pickerStyle(.segmented)
            if let notice = model.cropNotice {
                Text(notice)
                    .font(.caption)
                    .foregroundStyle(Color.yellow)
            }
            HStack(spacing: 6) {
                Button {
                    model.onCropUndo?()
                } label: {
                    Label("Undo", systemImage: "arrow.uturn.backward")
                }
                .buttonStyle(SplatViewerPanelButtonStyle())
                .disabled(!model.cropCanUndo)
                Button {
                    model.onCropRestore?()
                } label: {
                    Label("Restore", systemImage: "arrow.counterclockwise")
                }
                .buttonStyle(SplatViewerPanelButtonStyle())
                .disabled(!model.cropEdited)
                Spacer(minLength: 0)
                // Apply cuts the region and stays in Crop, so more cuts can follow. Finish leaves Crop.
                Button {
                    model.onCropApply?()
                } label: {
                    Label("Apply", systemImage: "checkmark")
                }
                .buttonStyle(SplatViewerPanelButtonStyle())
                Button {
                    model.isCropping = false
                    showEditOptions = false
                } label: {
                    Label("Finish", systemImage: "flag.checkered")
                }
                .buttonStyle(SplatViewerPanelButtonStyle(prominent: true))
            }
        }
        .padding(14)
        .frame(maxWidth: 360)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .padding(.horizontal, 12)
        .padding(.bottom, 4)
    }

    /// The floater panel, in the same card as Crop. The line under the title is about the
    /// selection: what a tap does or how many splats are selected (with an ⊗ that lets them all go),
    /// and Detect, which adds to it. Then Selection size, the radius a tap selects within, and the
    /// actions: Undo takes back the last step (a selection change or a delete), Reset brings back
    /// every splat as loaded (as Crop's does), Delete removes the selection, Finish leaves. Delete
    /// leaves the count to the line above, so the row fits a phone.
    private var floaterControls: some View {
        VStack(spacing: 10) {
            HStack {
                Text("Floaters")
                    .font(.caption.weight(.semibold))
                Spacer(minLength: 0)
                if model.isFindingFloaters {
                    ProgressView().controlSize(.small)
                }
            }
            floaterSelectionLine
            if let notice = model.cropNotice {
                Text(notice)
                    .font(.caption)
                    .foregroundStyle(Color.yellow)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            HStack(spacing: 8) {
                Text("Selection size")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                // While it moves, a circle on the view shows how far a tap reaches.
                Slider(value: $model.floaterBrushSize, in: SplatViewerFloaterBrush.sizeRange) {
                    model.setBrushSizeEditing($0)
                }
                .tint(SplatViewerPanelButtonStyle.fill)
            }
            HStack(spacing: 6) {
                Button {
                    model.onFloaterUndo?()
                } label: {
                    Label("Undo", systemImage: "arrow.uturn.backward")
                }
                .buttonStyle(SplatViewerPanelButtonStyle())
                .disabled(!model.floaterCanUndo)
                Button {
                    model.onCropRestore?()
                } label: {
                    Label("Reset", systemImage: "arrow.counterclockwise")
                }
                .buttonStyle(SplatViewerPanelButtonStyle())
                .disabled(!model.cropEdited)
                Spacer(minLength: 0)
                Button {
                    model.onFloaterDelete?()
                } label: {
                    Label("Delete", systemImage: "trash")
                }
                .buttonStyle(SplatViewerPanelButtonStyle())
                .disabled(model.floaterSelectionCount == 0)
                Button {
                    model.isRemovingFloaters = false
                    showEditOptions = false
                } label: {
                    Label("Finish", systemImage: "flag.checkered")
                }
                .buttonStyle(SplatViewerPanelButtonStyle(prominent: true))
            }
        }
        .padding(14)
        .frame(maxWidth: 360)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .padding(.horizontal, 12)
        .padding(.bottom, 4)
    }

    /// The colour selected splats are drawn in.
    private static let highlightColor: Color = {
        let c = SplatViewerFloaterHighlight.color
        return Color(red: Double(c.x), green: Double(c.y), blue: Double(c.z))
    }()

    /// What is selected, with an ⊗ that lets it all go, or with nothing selected what a tap does;
    /// then Detect on the right.
    private var floaterSelectionLine: some View {
        HStack(spacing: 6) {
            if model.floaterSelectionCount > 0 {
                Text("\(model.floaterSelectionCount.formatted()) splats selected")
                    .foregroundStyle(Self.highlightColor)
                Button {
                    model.onFloaterDeselectAll?()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Clear selection")
            } else {
                Text(floaterHint)
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
            Button {
                model.onFloaterDetect?()
            } label: {
                Label("Auto Detect", systemImage: "wand.and.stars")
            }
            .buttonStyle(SplatViewerPanelButtonStyle())
            .disabled(model.isFindingFloaters)
        }
        .font(.caption)
        .lineLimit(1)
    }

    private var floaterHint: String {
        #if os(iOS)
        "Tap a floater to select it"
        #else
        "Click a floater to select it"
        #endif
    }

    /// Fill of a button that is on: the kikyo purple, with white text on it.
    private static let onFill = SplatViewerPanelButtonStyle.fill

    /// On / off capsule button (filled = on). Used for measure mode.
    private func toggleCapsule(
        _ title: String, systemImage: String, isOn: Bool, action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Label(title, systemImage: systemImage)
                .font(.caption.weight(isOn ? .semibold : .regular))
                .foregroundStyle(isOn ? Color.white : Color.secondary)
                .padding(.horizontal, 10).padding(.vertical, 6)
                .background(
                    isOn ? AnyShapeStyle(Self.onFill) : AnyShapeStyle(.ultraThinMaterial),
                    in: Capsule()
                )
        }
        .buttonStyle(.plain)
    }

    private func status(symbol: String, title: String, detail: String) -> some View {
        VStack(spacing: 8) {
            Image(systemName: symbol).font(.largeTitle).foregroundStyle(.secondary)
            Text(title).font(.headline)
            Text(detail).font(.caption).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(24)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .padding(32)
    }

    // MARK: Gestures (macOS only; iOS uses UIKit recognizers on the MTKView)

    #if os(macOS)
    private var orbitGesture: some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                if dragBase == nil {
                    // A drag that starts on a crop handle resizes the region instead of orbiting.
                    dragBase = SIMD2(model.azimuth, model.elevation)
                    lastDragTranslation = .zero
                    cropDragging = model.isCropping
                        && (normalized(value.startLocation).map { model.onCropHandleBegin?($0) ?? false } ?? false)
                }
                if cropDragging {
                    let size = metalViewSize
                    if size.width > 0, size.height > 0 {
                        let dx = Float(value.translation.width - lastDragTranslation.width)
                        let dy = Float(value.translation.height - lastDragTranslation.height)
                        model.onCropHandleDrag?(SIMD2(dx / Float(size.width), dy / Float(size.height)))
                    }
                    lastDragTranslation = value.translation
                    return
                }
                let base = dragBase ?? SIMD2(model.azimuth, model.elevation)
                model.azimuth = base.x + Float(value.translation.width) * 0.01
                model.elevation = min(1.5, max(-1.5, base.y + Float(value.translation.height) * 0.01))
            }
            .onEnded { _ in
                dragBase = nil
                if cropDragging {
                    cropDragging = false
                    model.onCropHandleEnd?()
                }
            }
    }

    private var zoomGesture: some Gesture {
        MagnifyGesture()
            .onChanged { value in
                let base = zoomBase ?? model.distance
                if zoomBase == nil { zoomBase = base }
                model.distance = max(0.05, base / Float(value.magnification))
            }
            .onEnded { _ in zoomBase = nil }
    }

    /// Click in measure or floater mode (the renderer ignores it otherwise).
    private var tapGesture: some Gesture {
        SpatialTapGesture()
            .onEnded { value in
                if let p = normalized(value.location) { model.onTap?(p) }
            }
    }

    /// Double click zooms to that point (the renderer ignores it in measure and floater mode).
    private var focusTapGesture: some Gesture {
        SpatialTapGesture(count: 2)
            .onEnded { value in
                if let p = normalized(value.location) { model.onFocusTap?(p) }
            }
    }

    /// Converts a position on the MTKView to normalized coordinates (top-left 0 / bottom-right 1).
    private func normalized(_ location: CGPoint) -> SIMD2<Float>? {
        guard metalViewSize.width > 0, metalViewSize.height > 0 else { return nil }
        return SIMD2(Float(location.x / metalViewSize.width), Float(location.y / metalViewSize.height))
    }
    #endif
}

// MARK: - MTKView host

private struct SplatMetalView: SplatViewRepresentable {
    let url: URL?
    let seedURL: URL?
    let meshURL: URL?
    let model: SplatViewerModel

    func makeCoordinator() -> Coordinator { Coordinator() }

    private func makeView(_ coordinator: Coordinator) -> MTKView {
        let view = MTKView()
        view.device = MTLCreateSystemDefaultDevice()
        let renderer = SplatViewerRenderer(view: view, model: model)
        coordinator.renderer = renderer
        view.delegate = renderer
        #if os(iOS)
        coordinator.installGestures(on: view)
        #endif
        renderer?.load(url, seedURL: seedURL, meshURL: meshURL)
        return view
    }

#if os(macOS)
    func makeNSView(context: Context) -> MTKView { makeView(context.coordinator) }
    func updateNSView(_ nsView: MTKView, context: Context) {}
#else
    func makeUIView(context: Context) -> MTKView { makeView(context.coordinator) }
    func updateUIView(_ uiView: MTKView, context: Context) {}
#endif

    @MainActor
    final class Coordinator: NSObject {
        var renderer: SplatViewerRenderer?

        #if os(iOS)
        /// 1 finger = rotate / 2 fingers = pan / pinch = zoom / tap = measurement point / double
        /// tap = zoom to that point. SwiftUI's DragGesture can't tell finger counts apart, so UIKit
        /// recognizers are used. Taps don't wait for the double tap to fail (measurement points are
        /// placed immediately). The renderer ignores one or the other depending on measure mode.
        func installGestures(on view: UIView) {
            let orbit = UIPanGestureRecognizer(target: self, action: #selector(handleOrbit(_:)))
            orbit.maximumNumberOfTouches = 1
            let pan = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
            pan.minimumNumberOfTouches = 2
            pan.maximumNumberOfTouches = 2
            let pinch = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
            let tap = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
            let doubleTap = UITapGestureRecognizer(target: self, action: #selector(handleDoubleTap(_:)))
            doubleTap.numberOfTapsRequired = 2
            for recognizer in [orbit, pan, pinch, tap, doubleTap] as [UIGestureRecognizer] {
                recognizer.delegate = self
                view.addGestureRecognizer(recognizer)
            }
        }

        /// The drag in progress moves a crop handle (it started on one).
        private var draggingCropHandle = false

        /// A 1-finger drag that starts on a crop handle moves it; any other one orbits.
        @objc private func handleOrbit(_ g: UIPanGestureRecognizer) {
            guard let view = g.view, view.bounds.width > 0, view.bounds.height > 0 else { return }
            let t = g.translation(in: view)
            g.setTranslation(.zero, in: view)
            if g.state == .began {
                // Until now the translation counted from where the touch began, so that is the start.
                let p = g.location(in: view)
                let start = CGPoint(x: p.x - t.x, y: p.y - t.y)
                draggingCropHandle = renderer?.beginCropDrag(normalized: SIMD2(
                    Float(start.x / view.bounds.width), Float(start.y / view.bounds.height))) ?? false
            }
            if draggingCropHandle {
                renderer?.cropDrag(normalizedDelta: SIMD2(
                    Float(t.x / view.bounds.width), Float(t.y / view.bounds.height)))
                if g.state == .ended || g.state == .cancelled {
                    draggingCropHandle = false
                    renderer?.endCropDrag()
                }
                return
            }
            renderer?.orbit(dxPixels: Float(t.x), dyPixels: Float(t.y))
        }

        @objc private func handlePan(_ g: UIPanGestureRecognizer) {
            let t = g.translation(in: g.view)
            g.setTranslation(.zero, in: g.view)
            renderer?.pan(dxPixels: Float(t.x), dyPixels: Float(t.y))
        }

        @objc private func handlePinch(_ g: UIPinchGestureRecognizer) {
            renderer?.zoom(by: Float(g.scale))
            g.scale = 1
        }

        /// The renderer ignores it outside measure and floater mode.
        @objc private func handleTap(_ g: UITapGestureRecognizer) {
            if let p = normalized(g) { renderer?.tap(normalized: p) }
        }

        /// The renderer ignores it in measure and floater mode.
        @objc private func handleDoubleTap(_ g: UITapGestureRecognizer) {
            if let p = normalized(g) { renderer?.focusTap(normalized: p) }
        }

        /// Converts a tap position to normalized coordinates (top-left 0 / bottom-right 1).
        private func normalized(_ g: UITapGestureRecognizer) -> SIMD2<Float>? {
            guard let view = g.view, view.bounds.width > 0, view.bounds.height > 0 else { return nil }
            let p = g.location(in: view)
            return SIMD2(Float(p.x / view.bounds.width), Float(p.y / view.bounds.height))
        }
        #endif
    }
}

#if os(iOS)
extension SplatMetalView.Coordinator: UIGestureRecognizerDelegate {
    // Let pinch (2 fingers) and pan (2 fingers) work simultaneously. Orbit is 1 finger, so no
    // conflict. Simultaneous recognition is limited to our own recognizers. Allowing outside ones
    // (navigation swipe-back etc.) would start popping the whole screen at the same time as a
    // rotate drag.
    func gestureRecognizer(
        _ gestureRecognizer: UIGestureRecognizer,
        shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer
    ) -> Bool { other.view === gestureRecognizer.view }

    // iOS 26's full-screen swipe-back is the same motion as a 1-finger drag (rotate), so let it run
    // only when our recognizer fails = effectively disabled in the viewer (go back with the back
    // button).
    func gestureRecognizer(
        _ gestureRecognizer: UIGestureRecognizer,
        shouldBeRequiredToFailBy other: UIGestureRecognizer
    ) -> Bool {
        guard #available(iOS 26.0, *) else { return false }
        return other === navigationController(of: gestureRecognizer)?.interactiveContentPopGestureRecognizer
    }

    // Keep swipe-back from the screen's left edge, and give it priority for touches that start at
    // the left edge (touches elsewhere fail it immediately, so rotation isn't delayed).
    func gestureRecognizer(
        _ gestureRecognizer: UIGestureRecognizer,
        shouldRequireFailureOf other: UIGestureRecognizer
    ) -> Bool {
        other === navigationController(of: gestureRecognizer)?.interactivePopGestureRecognizer
    }

    /// The UINavigationController of the screen hosting the MTKView (what backs SwiftUI's
    /// NavigationStack).
    private func navigationController(of recognizer: UIGestureRecognizer) -> UINavigationController? {
        var responder: UIResponder? = recognizer.view
        while let r = responder {
            if let vc = r as? UIViewController { return vc.navigationController }
            responder = r.next
        }
        return nil
    }
}
#endif

#else // platforms without the MetalKit render path (visionOS etc.)

public struct SplatViewerScreen: View {
    public init(
        url: URL?,
        seedURL: URL? = nil,
        meshURL: URL? = nil,
        name: String? = nil,
        shareItems: [SplatViewerShareItem] = [],
        onTrainMore: (() -> Void)? = nil,
        trainMoreMessage: String = "",
        onRename: ((String) -> Void)? = nil
    ) {}
    public var body: some View {
        ContentUnavailableView(
            "Viewer unavailable",
            systemImage: "eye.slash",
            description: Text("The PLY viewer isn't available on this platform")
        )
    }
}

#endif
