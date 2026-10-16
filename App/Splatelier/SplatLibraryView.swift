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
        let name: String?       // Display name (<base>.name.json)
        let settings: TrainingSettingsRecord?  // <base>.settings.json
        let metrics: CaptureMetricsRecord?     // <base>.metrics.json
        /// capture folder. If non-nil, delete removes the whole folder; nil means a standalone PLY
        /// directly under Documents.
        let containerDir: URL?
        /// The display name, or the datetime file name if none.
        var displayName: String { name ?? mainURL.deletingPathExtension().lastPathComponent }
    }

    private struct SettingsSheetData: Identifiable {
        let id: String          // capture base
        let title: String
        let record: TrainingSettingsRecord
    }

    @State private var items: [CaptureItem] = []
    @State private var importing = false
    @State private var errorMessage: String?
    @State private var selectMode = false
    @State private var selection = Set<String>()
    @State private var confirmingBatchDelete = false
    @State private var renamingItem: CaptureItem?
    @State private var renameText = ""
    @State private var settingsSheet: SettingsSheetData?
    private let previewHeight: CGFloat = 156

    var body: some View {
        Group {
            if items.isEmpty {
                ContentUnavailableView {
                    Label("No captures yet", systemImage: "cube.transparent")
                } description: {
                    Text("Trained scenes appear here.\nTap + at the top right to import .ply / .splat / .spz files.")
                }
            } else {
                List {
                    ForEach(items) { item in
                        rowContainer(item)
                    }
                    .onDelete(perform: deleteAt)
                }
            }
        }
        .navigationTitle("Library")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar { toolbarContent }
        .safeAreaInset(edge: .bottom) { selectionBar }
        .onAppear(perform: reload)
        .refreshable { reload() }
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
        .confirmationDialog(
            "Delete \(selection.count) captures?",
            isPresented: $confirmingBatchDelete,
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) { batchDelete() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("All related files (PLY, mesh, logs, etc.) will also be deleted. This can't be undone.")
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
                Button { importing = true } label: {
                    Image(systemName: "plus")
                }
            }
        }
    }

    private var selectModeTitle: String { selectMode ? "Done" : "Select" }
    private var selectAllTitle: String { selection.count == items.count ? "Deselect all" : "Select all" }

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
            NavigationLink {
                SplatViewerScreen(
                    url: item.mainURL,
                    seedURL: item.seedURL,
                    meshURL: item.meshURL,
                    name: item.name,
                    onRename: { newName in
                        CaptureMetadataName.save(newName, for: item.mainURL)
                        reload()
                    }
                )
            } label: {
                card(item, selected: false)
            }
            .contextMenu {
                Button {
                    beginRename(item)
                } label: {
                    Label("Rename", systemImage: "pencil")
                }
                if let record = item.settings {
                    Button {
                        settingsSheet = SettingsSheetData(
                            id: item.id, title: item.displayName, record: record)
                    } label: {
                        Label("View training settings", systemImage: "slider.horizontal.3")
                    }
                }
                // Only the deliverable GS (= main) PLY is shared; the mesh has its own entry.
                ShareLink(item: item.mainURL) {
                    Label("Share", systemImage: "square.and.arrow.up")
                }
                if let meshURL = item.meshURL {
                    ShareLink(item: meshURL) {
                        Label("Share mesh", systemImage: "cube")
                    }
                }
                Button(role: .destructive) {
                    delete(item)
                } label: {
                    Label("Delete", systemImage: "trash")
                }
            }
        }
    }

    private func card(_ item: CaptureItem, selected: Bool) -> some View {
        let url = item.mainURL
        let values = try? url.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
        return ZStack(alignment: .bottomLeading) {
            previewImage(for: url)
            LinearGradient(
                colors: [.clear, .black.opacity(0.62)],
                startPoint: .center,
                endPoint: .bottom
            )
            VStack(alignment: .leading, spacing: 2) {
                Text(item.displayName)
                    .font(.caption.weight(.semibold).monospaced())
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(metadataText(values, hasSeed: item.seedURL != nil))
                    .font(.caption2.monospacedDigit())
                    .lineLimit(1)
                if let metrics = item.metrics {
                    // Smooth loss (rolling mean) at the final iter.
                    Label(String(format: "smooth loss %.5f", metrics.finalSmoothLoss),
                          systemImage: "chart.line.downtrend.xyaxis")
                        .font(.caption2.monospacedDigit())
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
                    .foregroundStyle(.white, selected ? Color.accentColor : Color.black.opacity(0.35))
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

    /// Show the sibling `.png` as a thumbnail (saved in sensor orientation, so rotate clockwise).
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

    // MARK: - Selection bar

    @ViewBuilder
    private var selectionBar: some View {
        if selectMode {
            HStack(spacing: 16) {
                Text(selection.isEmpty ? "None selected" : "\(selection.count) selected")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                Spacer()
                ShareLink(items: selectedMainURLs) {
                    Label("Share", systemImage: "square.and.arrow.up")
                }
                .disabled(selection.isEmpty)
                Button(role: .destructive) {
                    confirmingBatchDelete = true
                } label: {
                    Label("Delete", systemImage: "trash")
                }
                .disabled(selection.isEmpty)
            }
            .padding(.horizontal)
            .padding(.vertical, 10)
            .background(.bar)
        }
    }

    private var selectedMainURLs: [URL] {
        items.filter { selection.contains($0.id) }.map(\.mainURL)
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
            meshURL: meshURL,
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

    private func metadataText(_ values: URLResourceValues?, hasSeed: Bool) -> String {
        var parts: [String] = []
        if let date = values?.contentModificationDate {
            parts.append(date.formatted(date: .abbreviated, time: .shortened))
        }
        if let size = values?.fileSize {
            parts.append(byteString(size))
        }
        if hasSeed {
            parts.append("seed+cameras")
        }
        return parts.joined(separator: "  ")
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

    private func deleteAt(_ offsets: IndexSet) {
        for index in offsets {
            removeCapture(items[index])
        }
        reload()
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

    private func byteString(_ bytes: Int) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
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
    func rotatedClockwise() -> UIImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = scale
        format.opaque = false
        let rotatedSize = CGSize(width: size.height, height: size.width)
        return UIGraphicsImageRenderer(size: rotatedSize, format: format).image { context in
            let cgContext = context.cgContext
            cgContext.translateBy(x: rotatedSize.width, y: 0)
            cgContext.rotate(by: .pi / 2)
            draw(in: CGRect(origin: .zero, size: size))
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
        guard let image = UIImage(contentsOfFile: pngURL.path)?.rotatedClockwise() else { return nil }
        cache.setObject(image, forKey: key)
        return image
    }
}
