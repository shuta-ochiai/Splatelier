// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// App root. Two tabs: Capture (capture → training) and Library (playback of trained PLYs).
// Each tab has its own NavigationStack; pushes stay within the tab.

import SwiftUI

struct RootView: View {
    // A TabView without selection has a known bug where it jumps to the Library tab on its own when
    // the layout changes at training start, so pin the selected tab with explicit state.
    @State private var selectedTab = 0

    var body: some View {
        TabView(selection: $selectedTab) {
            NavigationStack {
                CaptureView()
            }
            .tabItem { Label("Capture", systemImage: "camera.viewfinder") }
            .tag(0)

            NavigationStack {
                SplatLibraryView()
            }
            .tabItem { Label("Library", systemImage: "square.stack.3d.up") }
            .tag(1)
        }
    }
}

#Preview {
    RootView()
}
