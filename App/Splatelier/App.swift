// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// SwiftUI app entry point. Provides capture (ARKit) → training (PocketGS) → viewing in one app.

import Capture
import SwiftUI
import Trainer

@main
struct SplatelierApp: App {
    init() {
        // Temporary capture files (retired crash-recovery datasets, MVS depth maps) not yet
        // deleted when the app was last killed.
        CaptureScratch.removeLeftovers()
        // SPLATELIER_SH_DEGREE=1|2|3 (developer builds): foreground SH degree for on-device
        // experiments (default 2). Set before any model is made, since it fixes the shape of shN.
        if let degree = AppFeatures.developerEnvironment["SPLATELIER_SH_DEGREE"].flatMap({ Int($0) }) {
            GaussianModel.configuredShDegree = degree
        }
        let env = AppFeatures.developerEnvironment
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        func append(_ text: String, to name: String) {
            let url = documents.appendingPathComponent(name)
            let line = Data(text.utf8)
            if let h = try? FileHandle(forWritingTo: url) { h.seekToEndOfFile(); h.write(line); try? h.close() }
            else { try? line.write(to: url) }
        }
        // SPLATELIER_THERMAL_PROBE=1: append the thermal state to Documents/thermal_probe.log and
        // quit (for waiting until the device has cooled between unattended runs).
        if env["SPLATELIER_THERMAL_PROBE"] != nil {
            append("\(ISO8601DateFormatter().string(from: Date())) thermal=\(ProcessInfo.processInfo.thermalState.rawValue)\n",
                   to: "thermal_probe.log")
            exit(0)
        }
        // Unattended runs keep the screen on for the whole run, not only during training.
        if env["SPLATELIER_AUTOTRAIN"] != nil {
            DispatchQueue.main.async { UIApplication.shared.isIdleTimerDisabled = true }
        }
        // SPLATELIER_THERMAL_LOG=1: append the thermal state every 10 s to Documents/thermal.log.
        if env["SPLATELIER_THERMAL_LOG"] != nil {
            Thread.detachNewThread {
                let start = Date()
                while true {
                    append(String(format: "%@ t=%.0f thermal=%d\n", ISO8601DateFormatter().string(from: Date()),
                                  Date().timeIntervalSince(start), ProcessInfo.processInfo.thermalState.rawValue),
                           to: "thermal.log")
                    Thread.sleep(forTimeInterval: 10)
                }
            }
        }
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                // Fixed to dark to avoid a brightness jump when transitioning to the
                // black-background viewer.
                .preferredColorScheme(.dark)
        }
    }
}
