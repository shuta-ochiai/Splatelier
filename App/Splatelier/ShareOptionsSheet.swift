// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// The Library's Share: choose what to send, the splats (in a format of your choice), the mesh, or
// both, then hand the files to the system share sheet. Used for one capture and for a selection.

import SwiftUI
import UIKit
import Trainer

/// One capture to share.
struct ShareEntry {
    let name: String
    let splatURL: URL
    let meshURL: URL?
}

/// A request to share, for `.sheet(item:)`.
struct ShareRequest: Identifiable {
    let id = UUID()
    let entries: [ShareEntry]
}

private struct PreparedFiles: Identifiable {
    let id = UUID()
    let urls: [URL]
}

struct ShareOptionsSheet: View {
    let entries: [ShareEntry]
    @Environment(\.dismiss) private var dismiss
    @AppStorage("share.splatFormat") private var formatRaw = SplatExportFormat.ply.rawValue
    @State private var includeSplats = true
    @State private var includeMesh = false
    /// Progress text while files are being made; non-nil while busy.
    @State private var preparing: String?
    @State private var prepared: PreparedFiles?
    @State private var errorMessage: String?
    @State private var task: Task<Void, Never>?

    private var format: SplatExportFormat { SplatExportFormat(rawValue: formatRaw) ?? .ply }
    private var meshCount: Int { entries.filter { $0.meshURL != nil }.count }
    /// Only a PLY can be converted; imported .spz / .splat files go out as they are.
    private var convertibleCount: Int { entries.filter { $0.splatURL.pathExtension.lowercased() == "ply" }.count }
    private var canShare: Bool { (includeSplats || (includeMesh && meshCount > 0)) && preparing == nil }

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Toggle("Splats", isOn: $includeSplats)
                        .tint(Color.kikyoFill)
                    if includeSplats {
                        Picker("Format", selection: $formatRaw) {
                            ForEach(SplatExportShare.formats) { f in
                                Text(SplatExportShare.label(for: f)).tag(f.rawValue)
                            }
                        }
                        .disabled(convertibleCount == 0)
                    }
                } footer: {
                    if convertibleCount < entries.count {
                        Text("Imported files that are not a PLY are shared as they are.")
                    }
                }
                if AppFeatures.meshGeneration {
                    Section {
                        Toggle("Mesh (OBJ)", isOn: $includeMesh)
                            .tint(Color.kikyoFill)
                            .disabled(meshCount == 0)
                    } footer: {
                        if meshCount == 0 {
                            Text(entries.count == 1 ? "This capture has no mesh." : "None of these captures has a mesh.")
                        } else if meshCount < entries.count {
                            Text("\(meshCount) of \(entries.count) captures have a mesh.")
                        }
                    }
                }
                Section {
                    Button(action: start) {
                        HStack(spacing: 8) {
                            if preparing != nil { ProgressView() }
                            Text(preparing ?? "Share")
                                .fontWeight(.semibold)
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(Color.kikyoFill)
                    .controlSize(.large)
                    .disabled(!canShare)
                    .listRowBackground(Color.clear)
                    .listRowInsets(EdgeInsets())
                }
            }
            .appBackground()
            .navigationTitle(entries.count == 1 ? entries[0].name : "Share \(entries.count) captures")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button(preparing == nil ? "Close" : "Cancel") {
                        task?.cancel()
                        dismiss()
                    }
                }
            }
        }
        .presentationDetents([.medium])
        .sheet(item: $prepared, onDismiss: { dismiss() }) { files in
            SystemShareSheet(urls: files.urls)
        }
        .alert("Couldn't share", isPresented: Binding(
            get: { errorMessage != nil }, set: { if !$0 { errorMessage = nil } })
        ) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(errorMessage ?? "")
        }
    }

    /// Makes the files (converting the splats if a format other than the PLY was chosen), then
    /// hands them to the share sheet. Cancelling drops whatever is still being made.
    private func start() {
        guard canShare else { return }
        let format = format, withSplats = includeSplats, withMesh = includeMesh
        let entries = entries
        preparing = "Preparing…"
        task = Task {
            do {
                let dir = try SplatExportShare.makeExportDirectory()
                var urls: [URL] = []
                for (i, entry) in entries.enumerated() {
                    try Task.checkCancellation()
                    if entries.count > 1 { preparing = "Preparing \(i + 1) of \(entries.count)…" }
                    if withSplats {
                        urls.append(try await SplatExportShare.file(for: entry.splatURL, as: format, in: dir))
                    }
                    if withMesh, let mesh = entry.meshURL { urls.append(mesh) }
                }
                try Task.checkCancellation()
                preparing = nil
                if urls.isEmpty {
                    errorMessage = "There is nothing to share."
                } else {
                    prepared = PreparedFiles(urls: urls)
                }
            } catch is CancellationError {
                preparing = nil
            } catch {
                preparing = nil
                errorMessage = error.localizedDescription
            }
        }
    }
}

/// `UIActivityViewController` for the prepared files.
private struct SystemShareSheet: UIViewControllerRepresentable {
    let urls: [URL]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: urls, applicationActivities: nil)
    }
    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}
