// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Launch animation. The OS launch screen is a flat color (LaunchBackground, the same color as the
// video's background), so this plays right after it to cover the wait while the app prepares.
// Muted and without an audio track, so it never interrupts the user's music.

import AVFoundation
import SwiftUI

/// Plays the bundled launch.mp4 once, filling the screen. Calls `onEnd` when it has played to the
/// end (or at once if the file can't be played).
struct LaunchVideoView: UIViewRepresentable {
    /// Fill the view (portrait windows), or fit inside it (landscape ones, where filling would crop
    /// the portrait video's wordmark off the bottom). The background around it is the same color.
    let fills: Bool
    let onEnd: () -> Void

    /// How the 1080×1920 video is scaled into a view of `size`, matching `fills`.
    static func scale(for size: CGSize, fills: Bool) -> CGFloat {
        let sx = size.width / 1080, sy = size.height / 1920
        return fills ? max(sx, sy) : min(sx, sy)
    }

    /// Fill a window at least as tall as it is wide; fit inside a wider one.
    static func fills(_ size: CGSize) -> Bool { size.height >= size.width }

    func makeUIView(context: Context) -> LaunchPlayerView {
        let view = LaunchPlayerView()
        guard let url = Bundle.main.url(forResource: "launch", withExtension: "mp4") else {
            DispatchQueue.main.async(execute: onEnd)
            return view
        }
        let player = AVPlayer(url: url)
        player.isMuted = true
        player.actionAtItemEnd = .pause
        view.playerLayer.player = player
        view.playerLayer.videoGravity = fills ? .resizeAspectFill : .resizeAspect
        context.coordinator.observe(player, onEnd: onEnd)
        player.play()
        return view
    }

    func updateUIView(_ uiView: LaunchPlayerView, context: Context) {
        uiView.playerLayer.videoGravity = fills ? .resizeAspectFill : .resizeAspect
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        private var observers: [NSObjectProtocol] = []

        func observe(_ player: AVPlayer, onEnd: @escaping () -> Void) {
            let center = NotificationCenter.default
            // A failed item (corrupt file) never reaches the end, so treat it as ended too.
            for name in [AVPlayerItem.didPlayToEndTimeNotification,
                         AVPlayerItem.failedToPlayToEndTimeNotification] {
                observers.append(center.addObserver(
                    forName: name, object: player.currentItem, queue: .main
                ) { _ in onEnd() })
            }
        }

        deinit { observers.forEach(NotificationCenter.default.removeObserver) }
    }
}

final class LaunchPlayerView: UIView {
    override class var layerClass: AnyClass { AVPlayerLayer.self }
    var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }
}
