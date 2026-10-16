// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation

/// Temporary files the capture pipeline keeps while something still reads them (retired
/// crash-recovery datasets, MVS depth maps for meshing). Each is deleted when its owner is
/// released; a run killed before that leaves them here.
public enum CaptureScratch {
    static func directory(_ name: String) -> URL {
        root.appendingPathComponent(name, isDirectory: true)
    }

    /// Deletes what a previous run left behind. Call at launch, before anything is created here.
    public static func removeLeftovers() {
        try? FileManager.default.removeItem(at: root)
    }

    private static var root: URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("capture-scratch", isDirectory: true)
    }
}
