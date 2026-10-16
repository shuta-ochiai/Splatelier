// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// SwiftUI app entry point. Provides capture (ARKit) → training (PocketGS) → viewing in one app.

import Capture
import SwiftUI

@main
struct SplatelierApp: App {
    init() {
        // Temporary capture files (retired crash-recovery datasets, MVS depth maps) not yet
        // deleted when the app was last killed.
        CaptureScratch.removeLeftovers()
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
