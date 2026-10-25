// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// About sheet: version, links and the open source licenses the app ships with.

import SwiftUI
import Trainer

/// The version the app shows, read from Info.plist (the release version and the build number).
enum AppVersion {
    static var short: String {
        (Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String)
            ?? Splatelier.version
    }

    static var build: String? {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String
    }

    /// "0.1.0 (1)"
    static var plain: String { short + (build.map { " (\($0))" } ?? "") }

    /// "v0.1.0 (1)"
    static var full: String { "v" + plain }
}

private struct LicenseEntry: Identifiable {
    let name: String
    let license: String
    let file: String  // bundled `Licenses/<file>.txt`
    var id: String { file }
}

private let licenseEntries: [LicenseEntry] = [
    LicenseEntry(name: "Splatelier", license: "Apache-2.0", file: "Splatelier"),
    LicenseEntry(name: "MLX Swift", license: "MIT", file: "mlx-swift"),
    LicenseEntry(name: "MLX", license: "MIT", file: "mlx"),
    LicenseEntry(name: "MLX C", license: "MIT", file: "mlx-c"),
    LicenseEntry(name: "{fmt}", license: "MIT", file: "fmt"),
    LicenseEntry(name: "JSON for Modern C++", license: "MIT", file: "nlohmann-json"),
    LicenseEntry(name: "metal-cpp", license: "Apache-2.0", file: "metal-cpp"),
    LicenseEntry(name: "Swift Numerics", license: "Apache-2.0", file: "swift-numerics"),
    LicenseEntry(name: "spz-swift", license: "MIT", file: "spz-swift"),
    LicenseEntry(name: "MetalSplatter", license: "MIT", file: "MetalSplatter"),
]

struct AboutView: View {
    @Environment(\.dismiss) private var dismiss

    private let repoURL = URL(string: "https://github.com/shuta-ochiai/Splatelier")!
    private let issuesURL = URL(string: "https://github.com/shuta-ochiai/Splatelier/issues/new/choose")!

    var body: some View {
        NavigationStack {
            List {
                header
                    .listRowBackground(Color.clear)
                Section {
                    Link(destination: repoURL) {
                        Label("Source code on GitHub", systemImage: "chevron.left.forwardslash.chevron.right")
                    }
                    Link(destination: issuesURL) {
                        Label("Report a problem", systemImage: "exclamationmark.bubble")
                    }
                }
                Section("Open source licenses") {
                    ForEach(licenseEntries) { entry in
                        NavigationLink {
                            LicenseTextView(entry: entry)
                        } label: {
                            HStack {
                                Text(entry.name)
                                Spacer()
                                Text(entry.license).foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }
            .scrollContentBackground(.hidden)
            .appBackground()
            .navigationTitle("About")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private var header: some View {
        VStack(spacing: 6) {
            Text("Splatelier")
                .font(.system(size: 30, weight: .semibold, design: .serif))
            Text(AppVersion.full)
                .font(.subheadline.monospacedDigit())
                .foregroundStyle(.secondary)
            Text("Capture, train and view 3D Gaussian splats on your iPhone.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.top, 4)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
    }
}

private struct LicenseTextView: View {
    let entry: LicenseEntry

    private var text: String {
        Bundle.main.url(forResource: entry.file, withExtension: "txt", subdirectory: "Licenses")
            .flatMap { try? String(contentsOf: $0, encoding: .utf8) }
            ?? Bundle.main.url(forResource: entry.file, withExtension: "txt")
                .flatMap { try? String(contentsOf: $0, encoding: .utf8) }
            ?? "License text not found."
    }

    var body: some View {
        ScrollView {
            Text(text)
                .font(.system(.footnote, design: .monospaced))
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                .textSelection(.enabled)
        }
        .appBackground()
        .navigationTitle(entry.name)
        .navigationBarTitleDisplayMode(.inline)
    }
}
