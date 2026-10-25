// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// App root. Two tabs: Capture (capture → training) and Library (playback of trained PLYs).
// Each tab has its own NavigationStack; pushes stay within the tab. The launch animation plays
// on top at start; once the app is ready, a tap skips it, and it ends by itself otherwise.

import Capture
import SwiftUI

struct RootView: View {
    // A TabView without selection has a known bug where it jumps to the Library tab on its own when
    // the layout changes at training start, so pin the selected tab with explicit state.
    @State private var selectedTab = 0

    /// The launch animation is on screen.
    @State private var showLaunch = true
    /// The tabs are built only when the animation goes away, so what they do on appearing (the
    /// unfinished-capture prompt) does not happen behind it.
    @State private var showContent = false
    /// What the app prepares at launch is done (the animation can be skipped).
    @State private var launchReady = false
    @State private var launchVideoEnded = false
    @State private var showLaunchVersion = false

    var body: some View {
        ZStack {
            if showContent { tabs }
            if showLaunch { launchOverlay }
        }
        .task {
            // Compile the shaders of the live capture preview off the main thread, so the first
            // capture does not wait for them.
            await Task.detached(priority: .userInitiated) {
                CaptureSurfelRenderer.prewarm()
            }.value
            launchReady = true
            if launchVideoEnded { endLaunch() }
        }
    }

    private var launchOverlay: some View {
        ZStack {
            // Same color as the OS launch screen (LaunchBackground) and the video's background.
            Color(red: 0x16 / 255, green: 0x1B / 255, blue: 0x46 / 255)
            LaunchVideoView {
                launchVideoEnded = true
                if launchReady { endLaunch() }
            }
            launchVersionLabel
        }
        .ignoresSafeArea()
        .contentShape(Rectangle())
        // Before the app is ready the tap does nothing: the animation covers the wait.
        .onTapGesture { if launchReady { endLaunch() } }
        .transition(.opacity)
    }

    /// The version under the wordmark, fading in as the wordmark finishes. The video is shown aspect
    /// fill (1080×1920), and the wordmark's baseline sits at y≈1474 px in it.
    private var launchVersionLabel: some View {
        GeometryReader { geo in
            let scale = max(geo.size.width / 1080, geo.size.height / 1920)
            let top = (geo.size.height - 1920 * scale) / 2
            Text(AppVersion.full)
                .font(.system(size: 15, design: .serif))
                .tracking(1.5)
                .foregroundStyle(Color(red: 0.96, green: 0.93, blue: 0.86).opacity(0.7))
                .position(x: geo.size.width / 2, y: top + (1474 + 110) * scale)
                .opacity(showLaunchVersion ? 1 : 0)
                .animation(.easeIn(duration: 0.8), value: showLaunchVersion)
        }
        .allowsHitTesting(false)
        .task {
            try? await Task.sleep(for: .seconds(2.8))
            showLaunchVersion = true
        }
    }

    /// Builds the tabs under the animation, then fades the animation out.
    private func endLaunch() {
        guard showLaunch else { return }
        showContent = true
        withAnimation(.easeOut(duration: 0.3)) { showLaunch = false }
    }

    private var tabs: some View {
        TabView(selection: $selectedTab) {
            NavigationStack {
                CaptureView().appBackground()
            }
            .tabItem { Label("Capture", systemImage: "camera.viewfinder") }
            .tag(0)

            NavigationStack {
                SplatLibraryView().appBackground()
            }
            .tabItem { Label("Library", systemImage: "square.stack.3d.up") }
            .tag(1)
        }
    }
}

#Preview {
    RootView()
}
