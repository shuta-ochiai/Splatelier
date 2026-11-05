// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// `view-cad`: the app's unified viewer (SplatViewerKit) in a Mac window, with a CAD model (STEP) and
// optionally splats in the same scene.

#if os(macOS)

import AppKit
import SplatViewerKit
import SwiftUI

private final class CADViewerAppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

@MainActor
enum CADViewerApp {
    private static let delegate = CADViewerAppDelegate()

    /// Opens the window and blocks the main thread in the AppKit event loop until it closes.
    static func run(cadURL: URL, splatURL: URL?) {
        let app = NSApplication.shared
        app.setActivationPolicy(.regular)
        app.delegate = delegate

        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1280, height: 860),
            styleMask: [.titled, .closable, .resizable, .miniaturizable],
            backing: .buffered, defer: false
        )
        window.title = "splatelier — \(cadURL.lastPathComponent)"
        window.backgroundColor = .black
        window.contentView = NSHostingView(rootView: SplatViewerScreen(url: splatURL, cadURL: cadURL, onImportCAD: { $0 }))
        window.center()
        window.makeKeyAndOrderFront(nil)
        app.activate(ignoringOtherApps: true)
        app.run()
    }
}

#endif // os(macOS)
