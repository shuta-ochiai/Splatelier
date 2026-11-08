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
