// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// What the viewer's Share menu offers, and the machinery that prepares a file on demand (a format
// conversion can take a while) before handing it to the system share sheet.

import SwiftUI

/// One entry of the Share menu: a file, or a group of ways to share the same thing (e.g. the
/// Gaussian splats as PLY, SPZ, ...).
public struct SplatViewerShareItem: Identifiable {
    /// One way to share it: an existing file, or a file made when it is picked.
    public struct Option: Identifiable {
        public enum Source {
            case file(URL)
            /// Several files shared together in one share sheet (e.g. the training and mesh logs).
            case files([URL])
            /// Runs when the option is picked; returns the file to share.
            case prepared(@Sendable () async throws -> URL)
        }

        public let title: String
        public let source: Source
        public var id: String { title }

        public init(title: String, url: URL) {
            self.title = title
            source = .file(url)
        }

        public init(title: String, urls: [URL]) {
            self.title = title
            source = .files(urls)
        }

        public init(title: String, prepare: @escaping @Sendable () async throws -> URL) {
            self.title = title
            source = .prepared(prepare)
        }
    }

    public let title: String
    public let systemImage: String
    public let options: [Option]
    public var id: String { title }

    /// A single file.
    public init(title: String, systemImage: String, url: URL) {
        self.init(title: title, systemImage: systemImage, options: [Option(title: title, url: url)])
    }

    /// Several files shared together as one entry.
    public init(title: String, systemImage: String, urls: [URL]) {
        self.init(title: title, systemImage: systemImage, options: [Option(title: title, urls: urls)])
    }

    /// Several ways to share the same thing, shown as a submenu.
    public init(title: String, systemImage: String, options: [Option]) {
        self.title = title
        self.systemImage = systemImage
        self.options = options
    }
}

#if os(iOS)
import UIKit

/// A file prepared for sharing, presented in the share sheet.
struct PreparedShareFile: Identifiable {
    let id = UUID()
    let url: URL
}

/// Runs the preparation of a share option, shows that it is busy, and hands the result to the share
/// sheet. Picking is ignored while one preparation runs.
@MainActor
final class SplatViewerSharePreparer: ObservableObject {
    /// Title of the option being prepared; non-nil while busy.
    @Published private(set) var preparing: String?
    /// Set when a file is ready; the screen presents the share sheet for it.
    @Published var ready: PreparedShareFile?
    @Published var errorMessage: String?
    private var task: Task<Void, Never>?

    func prepare(title: String, _ work: @escaping @Sendable () async throws -> URL) {
        guard preparing == nil else { return }
        preparing = title
        task = Task { [weak self] in
            do {
                let url = try await work()
                try Task.checkCancellation()
                self?.ready = PreparedShareFile(url: url)
            } catch is CancellationError {
                // The user cancelled; the result, if any, is dropped.
            } catch {
                self?.errorMessage = String(localized: "Couldn't prepare \(title): \(error.localizedDescription)")
            }
            self?.preparing = nil
        }
    }

    /// Stops waiting. The conversion itself cannot be interrupted, so it finishes in the
    /// background and its file is ignored.
    func cancel() {
        task?.cancel()
        preparing = nil
    }
}

/// `UIActivityViewController` for a prepared file.
struct SplatViewerShareSheet: UIViewControllerRepresentable {
    let url: URL
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: [url], applicationActivities: nil)
    }
    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}
#endif
