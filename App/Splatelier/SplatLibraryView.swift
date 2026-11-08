// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Library screen. Lists the capture_<T>/ folders in Documents (training results) and imported PLYs
// directly under it, and opens them in the viewer. .ply/.splat/.spz are imported by copying from
// Files into Documents. 1 capture = 1 card (groups the GS PLY and the seed PLY `_seed`). Delete
// removes the whole capture set.

import SwiftUI
import SplatViewerKit
import UniformTypeIdentifiers
import UIKit

struct SplatLibraryView: View {
    /// 1 capture = GS (or standalone imported) PLY + optional seed PLY.
    private struct CaptureItem: Identifiable {
        let id: String          // capture base (e.g. `capture_<yyyyMMddHHmmss>`)
        let mainURL: URL        // GS / imported PLY (the subject of thumbnail and sharing)
        let seedURL: URL?       // seed point cloud + camera marker PLY
        let meshURL: URL?       // Mesh OBJ (`_mesh.obj`, if a mesh was generated)
        let cadURL: URL?        // CAD model added by hand (`<base>.cad.step`), shown in the scene
        let name: String?       // Display name (<base>.name.json)
        let settings: TrainingSettingsRecord?  // <base>.settings.json
        let metrics: CaptureMetricsRecord?     // <base>.metrics.json
        /// capture folder. If non-nil, delete removes the whole folder; nil means a standalone PLY
        /// directly under Documents.
        let containerDir: URL?
        /// The display name, or the datetime file name if none.
        var displayName: String { name ?? mainURL.deletingPathExtension().lastPathComponent }
    }

    private final class RowFrames {
        var frames: [String: CGRect] = [:]
    }

    private struct SettingsSheetData: Identifiable {
        let id: String          // capture base
        let title: String
        let record: TrainingSettingsRecord
    }

    @State private var items: [CaptureItem] = []
    /// Width of the list, to center its column in a wide window.
    @State private var listWidth: CGFloat = 0
    @State private var importing = false
    /// The capture a CAD model is being picked for ("Add CAD model").
    @State private var cadTarget: CaptureItem?
    @State private var importingCAD = false
    @State private var errorMessage: String?
    @State private var selectMode = false
    @State private var selection = Set<String>()
    @State private var confirmingBatchDelete = false
    /// The capture a single Delete (swipe or menu) asks about.
    @State private var deletingItem: CaptureItem?
    /// Where each row is on screen (not observed: it changes on every scroll frame), and the frame
    /// of the row Delete was tapped on, which the dialog points at.
    @State private var rowFrames = RowFrames()
    @State private var deleteAnchorFrame: CGRect = .zero
    @State private var renamingItem: CaptureItem?
    @State private var renameText = ""
    @State private var settingsSheet: SettingsSheetData?
    @State private var shareRequest: ShareRequest?
    private let previewHeight: CGFloat = 156

    var body: some View {
        Group {
            if items.isEmpty {
                ContentUnavailableView {
                    Label("No captures yet", systemImage: "cube.transparent")
                } description: {
                    Text("Trained scenes appear here.\nTap Import at the top right to add .ply / .splat / .spz files.")
                }
            } else {
                List {
                    ForEach(items) { item in
                        rowContainer(item)
                            .onGeometryChange(for: CGRect.self) {
                                $0.frame(in: .global)
                            } action: {
                                rowFrames.frames[item.id] = $0
                            }
                            // The rows would draw a black group box behind each card; let the
                            // indigo show through and space the cards instead of ruling them off.
                            .listRowBackground(Color.clear)
                            .listRowSeparator(.hidden)
                            .listRowInsets(EdgeInsets(top: 6, leading: 16, bottom: 6, trailing: 16))
                            // Not `.onDelete`, and no `.destructive` role: both make the row play
                            // its removal animation, which scrolls a list that is at its end up by
                            // a card, and the dialog then pointed at the card above. The row stays
                            // because the dialog decides.
                            .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                                Button {
                                    askDelete(item)
                                } label: {
                                    Label("Delete", systemImage: "trash")
                                }
                                .tint(.red)
                            }
                    }
                }
                .listStyle(.plain)
                // In a wide iPad window the cards keep a readable width, centered.
                .onGeometryChange(for: CGFloat.self) { $0.size.width } action: { listWidth = $0 }
                .contentMargins(
                    .horizontal, max(0, (listWidth - DesignSystem.readableWidth) / 2),
                    for: .scrollContent)
            }
        }
        .navigationTitle("Library")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar { toolbarContent }
        .safeAreaInset(edge: .bottom) { selectionBar }
        .overlay { deleteDialogAnchor }
        .onAppear(perform: reload)
        .refreshable { reload() }
        .sheet(item: $shareRequest) { request in
            ShareOptionsSheet(entries: request.entries)
        }
        // A second importer can't sit on the same view, so the CAD picker hangs off a background.
        .background {
            Color.clear.fileImporter(
                isPresented: $importingCAD,
                allowedContentTypes: [CADModelFiles.stepType],
                allowsMultipleSelection: false
            ) { result in
                defer { cadTarget = nil }
                switch result {
                case .success(let urls):
                    guard let src = urls.first, let item = cadTarget else { return }
                    do {
                        try CADModelFiles.attach(src, to: item.mainURL)
                        reload()
                    } catch {
                        errorMessage = error.localizedDescription
                    }
                case .failure(let err): errorMessage = err.localizedDescription
                }
            }
        }
        .fileImporter(
            isPresented: $importing,
            allowedContentTypes: importableTypes,
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls): if let src = urls.first { importCopy(src) }
            case .failure(let err): errorMessage = err.localizedDescription
            }
        }
        .alert("Couldn't load", isPresented: Binding(
            get: { errorMessage != nil },
            set: { if !$0 { errorMessage = nil } }
        )) {
            Button("OK", role: .cancel) { errorMessage = nil }
        } message: {
            Text(errorMessage ?? "")
        }
        .alert("Rename", isPresented: Binding(
            get: { renamingItem != nil },
            set: { if !$0 { renamingItem = nil } }
        ), presenting: renamingItem) { item in
            TextField("Name", text: $renameText)
            Button("Save") {
                CaptureMetadataName.save(renameText, for: item.mainURL)
                renamingItem = nil
                reload()
            }
            Button("Cancel", role: .cancel) { renamingItem = nil }
        } message: { item in
            Text(item.mainURL.deletingPathExtension().lastPathComponent)
        }
        .sheet(item: $settingsSheet) { data in
            TrainingSettingsSheet(title: data.title, record: data.record)
        }
    }

    private func beginRename(_ item: CaptureItem) {
        renameText = item.name ?? ""
        renamingItem = item
    }

    // MARK: - Toolbar

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItemGroup(placement: .topBarLeading) {
            if !items.isEmpty {
                Button(selectModeTitle) { toggleSelectMode() }
            }
        }
        ToolbarItemGroup(placement: .topBarTrailing) {
            if selectMode {
                Button(selectAllTitle) { toggleSelectAll() }
            } else {
                // Text, like Select on the left: the bar shows only the icon of a Label.
                Button("Import") { importing = true }
            }
        }
    }

    private var selectModeTitle: String { selectMode ? "Done" : "Select" }
    private var selectAllTitle: String { selection.count == items.count ? "Deselect all" : "Select all" }

    /// What the viewer's Share menu offers: the GS, the mesh, and the training log if the capture
    /// folder has one.
    private func shareItems(for item: CaptureItem) -> [SplatViewerShareItem] {
        var items = [SplatExportShare.item(for: item.mainURL)]
        if let mesh = item.meshURL {
            items.append(SplatViewerShareItem(title: "Mesh (OBJ)", systemImage: "triangle", url: mesh))
        }
        // The training log and the mesh log go out together as "Log".
        guard AppFeatures.developerInfo else { return items }
        let base = item.mainURL.deletingPathExtension()
        let logs = [base.appendingPathExtension("log"), URL(fileURLWithPath: base.path + "_mesh.log")]
            .filter { FileManager.default.fileExists(atPath: $0.path) }
        if !logs.isEmpty {
            items.append(SplatViewerShareItem(title: "Log", systemImage: "doc.text", urls: logs))
        }
        return items
    }

    // MARK: - Rows

    @ViewBuilder
    private func rowContainer(_ item: CaptureItem) -> some View {
        if selectMode {
            Button {
                toggleSelection(item)
            } label: {
                card(item, selected: selection.contains(item.id))
            }
            .buttonStyle(.plain)
        } else {
            // The link sits invisibly over the card: a visible NavigationLink adds a ">" to the
            // right of the row, which pushed the card's "…" onto its edge.
            ZStack {
                card(item, selected: false)
                NavigationLink {
                    SplatViewerScreen(
                        url: item.mainURL,
                        // The seed points and capture cameras are for development.
                        seedURL: AppFeatures.developerInfo ? item.seedURL : nil,
                        meshURL: item.meshURL,
                        // Without these the viewer has no CAD layer and no Import button.
                        cadURL: item.cadURL,
                        cadPlacement: AppFeatures.cadModels
                            ? CADModelFiles.loadPlacement(for: item.mainURL) : nil,
                        onCADPlacementChange: AppFeatures.cadModels
                            ? { CADModelFiles.savePlacement($0, for: item.mainURL) } : nil,
                        onImportCAD: AppFeatures.cadModels
                            ? { try CADModelFiles.attach($0, to: item.mainURL) } : nil,
                        onRemoveCAD: AppFeatures.cadModels
                            ? { CADModelFiles.remove(for: item.mainURL) } : nil,
                        name: item.name,
                        shareItems: shareItems(for: item),
                        cropExport: SplatExportShare.cropExport,
                        onRename: { newName in
                            CaptureMetadataName.save(newName, for: item.mainURL)
                            reload()
                        }
                    )
                } label: {
                    EmptyView()
                }
                .opacity(0)
            }
            .contextMenu { itemActions(item) }
            // Same actions as the long press, on a visible button inside the card's bottom right.
            .overlay(alignment: .bottomTrailing) {
                Menu {
                    itemActions(item)
                } label: {
                    Image(systemName: "ellipsis")
                        .font(.body.weight(.semibold))
                        .foregroundStyle(.white)
                        .frame(width: 44, height: 44)
                        .contentShape(Rectangle())
                        .shadow(color: .black.opacity(0.5), radius: 2)
                }
                .accessibilityLabel("More actions")
                .padding(.trailing, 4)
            }
        }
    }

    /// Per-capture actions, shared by the long-press menu and the "…" button.
    @ViewBuilder
    private func itemActions(_ item: CaptureItem) -> some View {
        Button {
            beginRename(item)
        } label: {
            Label("Rename", systemImage: "pencil")
        }
        // Opens the sheet where the splats (and their format) and the mesh are chosen.
        Button {
            shareRequest = ShareRequest(entries: [shareEntry(for: item)])
        } label: {
            Label("Share", systemImage: "square.and.arrow.up")
        }
        // A CAD model (STEP) shown in the scene, to be moved into place in the viewer.
        if AppFeatures.cadModels {
            Button {
                cadTarget = item
                importingCAD = true
            } label: {
                Label(item.cadURL == nil ? "Add CAD model" : "Replace CAD model", systemImage: "cube.transparent")
            }
            if item.cadURL != nil {
                Button(role: .destructive) {
                    CADModelFiles.remove(for: item.mainURL)
                    reload()
                } label: {
                    Label("Remove CAD model", systemImage: "cube.transparent")
                }
            }
        }
        if AppFeatures.developerInfo, let record = item.settings {
            Button {
                settingsSheet = SettingsSheetData(
                    id: item.id, title: item.displayName, record: record)
            } label: {
                Label("View training settings", systemImage: "slider.horizontal.3")
            }
        }
        Button(role: .destructive) {
            askDelete(item)
        } label: {
            Label("Delete", systemImage: "trash")
        }
    }

    private func card(_ item: CaptureItem, selected: Bool) -> some View {
        let url = item.mainURL
        let values = try? url.resourceValues(forKeys: [.contentModificationDateKey])
        let dateText = values?.contentModificationDate?.formatted(date: .abbreviated, time: .shortened)
        return ZStack(alignment: .bottomLeading) {
            previewImage(for: url)
            LinearGradient(
                colors: [.clear, Color.appBackground.opacity(0.85)],
                startPoint: .center,
                endPoint: .bottom
            )
            // The name and the date, nothing else. Without a name the date is the title.
            VStack(alignment: .leading, spacing: 2) {
                if let name = item.name {
                    Text(name)
                        .font(.subheadline.weight(.semibold))
                        .lineLimit(1)
                        .truncationMode(.middle)
                    if let dateText { Text(dateText).font(.caption) }
                } else {
                    Text(dateText ?? item.displayName)
                        .font(.subheadline.weight(.semibold))
                        .lineLimit(1)
                }
            }
            .foregroundStyle(.white)
            .shadow(color: .black.opacity(0.5), radius: 2, x: 0, y: 1)
            .padding(10)
        }
        .frame(maxWidth: .infinity)
        .frame(height: previewHeight)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(alignment: .topTrailing) {
            if selectMode {
                Image(systemName: selected ? "checkmark.circle.fill" : "circle")
                    .font(.title2)
                    .symbolRenderingMode(.palette)
                    .foregroundStyle(.white, selected ? Color.kikyoFill : Color.black.opacity(0.35))
                    .padding(8)
                    .shadow(color: .black.opacity(0.5), radius: 2)
            }
        }
        .overlay {
            if selected {
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(Color.accentColor, lineWidth: 3)
            }
        }
    }

    /// Show the sibling `.png` as a thumbnail (saved in sensor orientation, so turned upright).
    /// If none (imported PLY etc.), a cube placeholder.
    @ViewBuilder
    private func previewImage(for url: URL) -> some View {
        let pngURL = url.deletingPathExtension().appendingPathExtension("png")
        if let image = LibraryThumbnailCache.image(for: pngURL) {
            GeometryReader { proxy in
                Image(uiImage: image)
                    .resizable()
                    .scaledToFill()
                    .frame(width: proxy.size.width, height: previewHeight)
                    .clipped()
            }
        } else {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(.thinMaterial)
                .frame(maxWidth: .infinity)
                .frame(height: previewHeight)
                .overlay {
                    Image(systemName: "cube.transparent")
                        .foregroundStyle(.secondary)
                }
        }
    }

    // MARK: - Delete confirmation

    private func askDelete(_ item: CaptureItem) {
        deleteAnchorFrame = rowFrames.frames[item.id] ?? .zero
        // Present after the anchor has moved to this card, so the dialog does not open on the
        // previous one.
        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(100))
            deletingItem = item
        }
    }

    /// Holds the single-delete dialog, placed over the card Delete was tapped on so the dialog
    /// points at it. It cannot sit on the row: the swipe resets the row, which closed the dialog at
    /// once. This view is outside the list, so it survives.
    private var deleteDialogAnchor: some View {
        GeometryReader { proxy in
            let origin = proxy.frame(in: .global).origin
            // The dialog goes on this view, before `.position`: position wraps it in a full-screen
            // view, and the dialog would then point at the top of the screen. The source is a
            // strip across the card's middle: with the whole card the arrow sat on the card's edge
            // and the dialog covered the card before, and with a single point it faced sideways.
            Color.clear
                .frame(width: deleteAnchorFrame.width, height: 40)
                .confirmationDialog(
                    "Delete this capture?",
                    isPresented: Binding(
                        get: { deletingItem != nil },
                        set: { if !$0 { deletingItem = nil } }
                    ),
                    titleVisibility: .visible,
                    presenting: deletingItem
                ) { item in
                    Button("Delete", role: .destructive) {
                        delete(item)
                        deletingItem = nil
                    }
                    Button("Cancel", role: .cancel) { deletingItem = nil }
                } message: { item in
                    Text("\(item.displayName) and all its files (\(Self.relatedFiles)) will be deleted. This can't be undone.")
                }
                .position(x: deleteAnchorFrame.midX - origin.x, y: deleteAnchorFrame.midY - origin.y)
        }
        .allowsHitTesting(false)
    }

    /// What deleting a capture removes, as the delete confirmations list it.
    private static let relatedFiles = AppFeatures.meshGeneration ? "PLY, mesh, logs, etc." : "PLY, logs, etc."

    // MARK: - Selection bar

    @ViewBuilder
    private var selectionBar: some View {
        if selectMode {
            HStack(spacing: 16) {
                Text(selection.isEmpty ? "None selected" : "\(selection.count) selected")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                Spacer()
                Button {
                    shareRequest = ShareRequest(entries: selectedItems.map(shareEntry(for:)))
                } label: {
                    Label("Share", systemImage: "square.and.arrow.up")
                }
                .disabled(selection.isEmpty)
                Button(role: .destructive) {
                    confirmingBatchDelete = true
                } label: {
                    Label("Delete", systemImage: "trash")
                }
                .disabled(selection.isEmpty)
                // On the button, so the dialog appears next to it at the bottom.
                .confirmationDialog(
                    "Delete \(selection.count) captures?",
                    isPresented: $confirmingBatchDelete,
                    titleVisibility: .visible
                ) {
                    Button("Delete", role: .destructive) { batchDelete() }
                    Button("Cancel", role: .cancel) {}
                } message: {
                    Text("All related files (\(Self.relatedFiles)) will also be deleted. This can't be undone.")
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 10)
            .background(.bar)
        }
    }

    private var selectedItems: [CaptureItem] {
        items.filter { selection.contains($0.id) }
    }

    private func shareEntry(for item: CaptureItem) -> ShareEntry {
        ShareEntry(name: item.displayName, splatURL: item.mainURL, meshURL: item.meshURL)
    }

    private func toggleSelectMode() {
        selectMode.toggle()
        if !selectMode { selection.removeAll() }
    }

    private func toggleSelectAll() {
        if selection.count == items.count {
            selection.removeAll()
        } else {
            selection = Set(items.map(\.id))
        }
    }

    private func toggleSelection(_ item: CaptureItem) {
        if selection.contains(item.id) {
            selection.remove(item.id)
        } else {
            selection.insert(item.id)
        }
    }

    // MARK: - File ops

    private var importableTypes: [UTType] {
        ["ply", "splat", "spz"].map { UTType(filenameExtension: $0) ?? .data }
    }

    private func documentsDirectory() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    /// Migrate bare capture_* files into folders, then scan directly under Documents and create one
    /// item per capture folder and one item per standalone PLY directly under it (imports etc.).
    private func reload() {
        let docs = documentsDirectory()
        migrateFlatCapturesToFolders(in: docs)

        let keys: [URLResourceKey] = [.contentModificationDateKey, .fileSizeKey, .isDirectoryKey]
        let entries = (try? FileManager.default.contentsOfDirectory(
            at: docs, includingPropertiesForKeys: keys)) ?? []

        var built: [CaptureItem] = []
        for entry in entries {
            let isDir = (try? entry.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
            if isDir {
                if let item = captureItem(fromFolder: entry) { built.append(item) }
            } else if ["ply", "splat", "spz"].contains(entry.pathExtension.lowercased()) {
                built.append(flatCaptureItem(for: entry))
            }
        }
        items = built.sorted { modDate($0.mainURL) > modDate($1.mainURL) }

        selection.formIntersection(Set(items.map(\.id)))
    }

    /// Build an item from the PLYs in a folder (the `_seed` suffix marks the seed). nil if there is
    /// no PLY (e.g. only a .dataset after a crash).
    private func captureItem(fromFolder folder: URL) -> CaptureItem? {
        let inner = (try? FileManager.default.contentsOfDirectory(
            at: folder, includingPropertiesForKeys: nil)) ?? []
        let plys = inner.filter { ["ply", "splat", "spz"].contains($0.pathExtension.lowercased()) }
        guard !plys.isEmpty else { return nil }
        var mainURL: URL?
        var seedURL: URL?
        for u in plys {
            if u.deletingPathExtension().lastPathComponent.hasSuffix("_seed") {
                seedURL = u
            } else {
                mainURL = u
            }
        }
        let primary = mainURL ?? seedURL!   // With no main (seed only), the seed becomes the primary
        let meshURL = inner.first {
            $0.pathExtension.lowercased() == "obj"
                && $0.deletingPathExtension().lastPathComponent.hasSuffix("_mesh")
        }
        return CaptureItem(
            id: folder.lastPathComponent,
            mainURL: primary,
            seedURL: mainURL != nil ? seedURL : nil,
            meshURL: AppFeatures.meshGeneration ? meshURL : nil,
            cadURL: AppFeatures.cadModels ? CADModelFiles.model(for: primary) : nil,
            name: CaptureMetadataName.load(for: primary),
            settings: TrainingSettingsStore.load(for: primary),
            metrics: CaptureMetricsStore.load(for: primary),
            containerDir: folder)
    }

    private func flatCaptureItem(for url: URL) -> CaptureItem {
        CaptureItem(
            id: url.deletingPathExtension().lastPathComponent,
            mainURL: url,
            seedURL: nil,
            meshURL: nil,
            cadURL: AppFeatures.cadModels ? CADModelFiles.model(for: url) : nil,
            name: CaptureMetadataName.load(for: url),
            settings: TrainingSettingsStore.load(for: url),
            metrics: CaptureMetricsStore.load(for: url),
            containerDir: nil)
    }

    /// Extract the base from a name starting with `capture_<digits>`. nil otherwise.
    private func captureBase(_ name: String) -> String? {
        guard name.hasPrefix("capture_") else { return nil }
        let digits = name.dropFirst("capture_".count).prefix { $0.isNumber }
        guard digits.count >= 8 else { return nil }
        return "capture_" + digits
    }

    /// Move bare `capture_<T>.*` files directly under Documents into a `capture_<T>/` folder
    /// (idempotent). Leaves imports with arbitrary names and the folder itself (name == base)
    /// untouched.
    private func migrateFlatCapturesToFolders(in docs: URL) {
        let entries = (try? FileManager.default.contentsOfDirectory(
            at: docs, includingPropertiesForKeys: nil)) ?? []
        for entry in entries {
            let name = entry.lastPathComponent
            guard let base = captureBase(name), name != base else { continue }
            let folder = docs.appendingPathComponent(base, isDirectory: true)
            try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let dest = folder.appendingPathComponent(name)
            if !FileManager.default.fileExists(atPath: dest.path) {
                try? FileManager.default.moveItem(at: entry, to: dest)
            }
        }
    }

    private func modDate(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
    }

    private func importCopy(_ src: URL) {
        let scoped = src.startAccessingSecurityScopedResource()
        defer { if scoped { src.stopAccessingSecurityScopedResource() } }
        let dest = documentsDirectory().appendingPathComponent(src.lastPathComponent)
        do {
            if FileManager.default.fileExists(atPath: dest.path) {
                try FileManager.default.removeItem(at: dest)
            }
            try FileManager.default.copyItem(at: src, to: dest)
            reload()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func delete(_ item: CaptureItem) {
        removeCapture(item)
        reload()
    }

    private func batchDelete() {
        let byId = Dictionary(items.map { ($0.id, $0) }, uniquingKeysWith: { a, _ in a })
        for id in selection {
            if let item = byId[id] { removeCapture(item) }
        }
        selection.removeAll()
        selectMode = false
        reload()
    }

    /// Folder-managed: delete the whole folder. Bare files: bulk-delete sibling files with a
    /// matching base.
    private func removeCapture(_ item: CaptureItem) {
        if let dir = item.containerDir {
            try? FileManager.default.removeItem(at: dir)
        } else {
            deleteFiles(base: item.id, in: documentsDirectory())
        }
    }

    /// Delete all sibling files with the same base. Matches are found by scanning rather than
    /// listing extensions, so new companions are covered automatically.
    private func deleteFiles(base: String, in dir: URL) {
        let siblings = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil)) ?? []
        for f in siblings {
            let name = f.lastPathComponent
            // base contains the full timestamp, so a prefix match cannot hit a different capture by
            // mistake.
            if name == base || name.hasPrefix(base + ".") || name.hasPrefix(base + "_seed") {
                try? FileManager.default.removeItem(at: f)
            }
        }
    }

}

private struct TrainingSettingsSheet: View {
    let title: String
    let record: TrainingSettingsRecord
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                ForEach(record.displaySections, id: \.title) { section in
                    Section(section.title) {
                        ForEach(section.rows, id: \.label) { row in
                            HStack {
                                Text(row.label).foregroundStyle(.secondary)
                                Spacer()
                                Text(row.value).font(.body.monospacedDigit())
                            }
                        }
                    }
                }
            }
            .appBackground()
            .navigationTitle("Training settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .principal) {
                    VStack(spacing: 0) {
                        Text("Training settings").font(.headline)
                        Text(title).font(.caption).foregroundStyle(.secondary)
                            .lineLimit(1).truncationMode(.middle)
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

private extension UIImage {
    func rotatedClockwise(quarterTurns: Int) -> UIImage {
        let turns = ((quarterTurns % 4) + 4) % 4
        guard turns != 0 else { return self }
        let format = UIGraphicsImageRendererFormat()
        format.scale = scale
        format.opaque = false
        let rotatedSize = turns == 2 ? size : CGSize(width: size.height, height: size.width)
        return UIGraphicsImageRenderer(size: rotatedSize, format: format).image { context in
            let cgContext = context.cgContext
            cgContext.translateBy(x: rotatedSize.width / 2, y: rotatedSize.height / 2)
            cgContext.rotate(by: CGFloat(turns) * .pi / 2)
            draw(in: CGRect(x: -size.width / 2, y: -size.height / 2, width: size.width, height: size.height))
        }
    }
}

/// Cache of rotated thumbnails, to avoid PNG decode + rotated drawing on every redraw. The key is
/// path + modification date (rebuilt when continue training overwrites the thumbnail).
private enum LibraryThumbnailCache {
    nonisolated(unsafe) static let cache = NSCache<NSString, UIImage>()   // NSCache is thread-safe

    static func image(for pngURL: URL) -> UIImage? {
        guard let modified = (try? pngURL.resourceValues(forKeys: [.contentModificationDateKey]))?
            .contentModificationDate else { return nil }
        let key = "\(pngURL.path)#\(modified.timeIntervalSince1970)" as NSString
        if let hit = cache.object(forKey: key) { return hit }
        // Turned by the source view's pose so world up is at the top (a portrait capture lies on
        // its side in sensor orientation, a landscape one does not). Without the pose, the
        // portrait default.
        let turns = SplatViewerThumbnailOrientation.clockwiseQuarterTurns(siblingOf: pngURL) ?? 1
        guard let image = UIImage(contentsOfFile: pngURL.path)?.rotatedClockwise(quarterTurns: turns)
        else { return nil }
        cache.setObject(image, forKey: key)
        return image
    }
}

/// A CAD model added to a capture, kept beside its splats as `<base>.cad.step` (or `.stp`), and
/// where it was placed in the viewer as `<base>.cadplacement.json`. Both start with `<base>.`, so
/// deleting the capture removes them too.
private enum CADModelFiles {
    static let stepType = UTType(importedAs: "org.iso.step", conformingTo: .data)

    private static func base(_ main: URL) -> URL { main.deletingPathExtension() }

    static func model(for main: URL) -> URL? {
        ["step", "stp"].lazy
            .map { base(main).appendingPathExtension("cad.\($0)") }
            .first { FileManager.default.fileExists(atPath: $0.path) }
    }

    /// Copies the picked file in, replacing any earlier model and its placement, and returns the copy.
    @discardableResult
    static func attach(_ src: URL, to main: URL) throws -> URL {
        let scoped = src.startAccessingSecurityScopedResource()
        defer { if scoped { src.stopAccessingSecurityScopedResource() } }
        remove(for: main)
        let ext = src.pathExtension.lowercased() == "stp" ? "stp" : "step"
        let dest = base(main).appendingPathExtension("cad.\(ext)")
        try FileManager.default.copyItem(at: src, to: dest)
        return dest
    }

    static func remove(for main: URL) {
        if let model = model(for: main) { try? FileManager.default.removeItem(at: model) }
        try? FileManager.default.removeItem(at: placementURL(for: main))
    }

    private static func placementURL(for main: URL) -> URL {
        base(main).appendingPathExtension("cadplacement.json")
    }

    static func loadPlacement(for main: URL) -> SplatViewerCADPlacement? {
        guard let data = try? Data(contentsOf: placementURL(for: main)) else { return nil }
        return try? JSONDecoder().decode(SplatViewerCADPlacement.self, from: data)
    }

    static func savePlacement(_ placement: SplatViewerCADPlacement, for main: URL) {
        guard let data = try? JSONEncoder().encode(placement) else { return }
        try? data.write(to: placementURL(for: main), options: .atomic)
    }
}
