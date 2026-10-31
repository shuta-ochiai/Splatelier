// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Capture tips shown on the start screen: muted clips played one after another (the captions are
// part of the videos). The clips come in fixed-order sets; a set is picked at random, played
// through, and then another set is picked.

import AVFoundation
import SwiftUI

/// Decides which clip plays next.
///
/// Each set is played in its written order. When a set ends, the next one is picked at random:
/// never the set that just ran, and never one that would start with the clip that just played, so
/// the same clip never comes twice in a row.
struct GuideSequence {
    /// File names in the bundle: guide_<name>.mp4. Every set holds all the clips (so each one is
    /// shown equally often), in a different order. Keep the first clips and the last clips of the
    /// sets apart (here {slow, light, loop} and {subjects, turn, pivot}), so that no set can follow
    /// another with the same clip.
    static let sets: [[String]] = [
        ["slow", "turn", "angles", "cool", "pivot", "loop", "light", "subjects"],
        ["light", "subjects", "pivot", "slow", "angles", "cool", "loop", "turn"],
        ["loop", "cool", "angles", "subjects", "turn", "light", "slow", "pivot"],
    ]

    private static let lastSetKey = "guideLastSet"
    private static let lastClipKey = "guideLastClip"

    private var setIndex: Int
    private var upcoming: [String] = []
    private var last: String?
    private let defaults: UserDefaults

    /// Starts in a random set, remembering across launches what ran last.
    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        setIndex = defaults.object(forKey: Self.lastSetKey) as? Int ?? -1
        last = defaults.string(forKey: Self.lastClipKey)
        startNextSet()
    }

    mutating func next() -> String {
        if upcoming.isEmpty { startNextSet() }
        let clip = upcoming.removeFirst()
        last = clip
        defaults.set(clip, forKey: Self.lastClipKey)
        return clip
    }

    private mutating func startNextSet() {
        let all = Array(Self.sets.indices)
        let others = all.filter { $0 != setIndex }
        // Prefer a set that does not start with the clip that just played.
        let fresh = others.filter { Self.sets[$0].first != last }
        setIndex = (fresh.isEmpty ? (others.isEmpty ? all : others) : fresh).randomElement()!
        upcoming = Self.sets[setIndex]
        defaults.set(setIndex, forKey: Self.lastSetKey)
    }
}

/// A request to move to the next or the previous clip. `id` makes repeated requests distinct.
private struct GuideNavigation: Equatable {
    var id = 0
    var forward = true
}

/// Square tip animation with rounded corners. Plays only while it is on screen and the app is
/// active; with Reduce Motion on it stays on its first frame. A tap pauses it (and resumes it); a
/// swipe to the left goes to the next clip, a swipe to the right to the previous one.
struct GuideClipView: View {
    /// Changes on every appearance, which starts a fresh random set.
    @State private var session = 0
    @State private var isOnScreen = false
    /// Paused by a tap.
    @State private var userPaused = false
    @State private var navigation = GuideNavigation()
    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        ZStack {
            if session > 0 {
                GuideSequencePlayer(
                    isPlaying: isOnScreen && scenePhase == .active && !reduceMotion && !userPaused,
                    navigation: navigation)
                    .id(session)
            }
            if userPaused {
                Image(systemName: "play.fill")
                    .font(.title)
                    .foregroundStyle(.white)
                    .frame(width: 64, height: 64)
                    .background(.black.opacity(0.45), in: Circle())
                    .transition(.opacity)
            }
        }
        .aspectRatio(1, contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .contentShape(Rectangle())
        // A mostly horizontal drag; a tap (which pauses) is too short to count as one.
        .gesture(DragGesture(minimumDistance: 30).onEnded { drag in
            let dx = drag.translation.width
            guard abs(dx) > abs(drag.translation.height) else { return }
            userPaused = false
            navigation = GuideNavigation(id: navigation.id + 1, forward: dx < 0)
        })
        .onTapGesture {
            withAnimation(.easeInOut(duration: 0.15)) { userPaused.toggle() }
        }
        .accessibilityHidden(true)
        .onAppear {
            session += 1
            userPaused = false
            isOnScreen = true
        }
        .onDisappear { isOnScreen = false }
    }
}

private struct GuideSequencePlayer: UIViewRepresentable {
    let isPlaying: Bool
    let navigation: GuideNavigation

    func makeUIView(context: Context) -> GuidePlayerView {
        let view = GuidePlayerView()
        view.playerLayer.videoGravity = .resizeAspectFill
        view.playerLayer.player = context.coordinator.player
        return view
    }

    func updateUIView(_ uiView: GuidePlayerView, context: Context) {
        let coordinator = context.coordinator
        coordinator.wantsPlaying = isPlaying
        if navigation.id != coordinator.handledNavigation {
            coordinator.handledNavigation = navigation.id
            if navigation.forward { coordinator.goNext() } else { coordinator.goPrevious() }
        }
        if isPlaying { coordinator.player.play() } else { coordinator.player.pause() }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    /// Plays the clips in the order GuideSequence chooses them and remembers that order, so it can
    /// go back. Two clips are queued ahead of the current one, so the next starts without a gap.
    @MainActor
    final class Coordinator {
        let player = AVQueuePlayer()
        var wantsPlaying = false
        /// Navigation requests already acted on.
        var handledNavigation = 0
        private var sequence = GuideSequence()
        /// Every clip chosen so far, in order; `index` is the one playing.
        private var timeline: [String] = []
        private var index = 0
        /// Which timeline entry each queued item plays.
        private var entry: [ObjectIdentifier: Int] = [:]
        private var observation: NSKeyValueObservation?

        init() {
            player.isMuted = true
            // The player drops each clip as it finishes (or on a skip); follow it and top up.
            observation = player.observe(\.currentItem, options: [.new]) { [weak self] _, _ in
                Task { @MainActor in self?.sync() }
            }
            sync()
        }

        func goNext() { player.advanceToNextItem() }

        /// Back to the previous clip, from its start. On the first clip it replays that clip.
        func goPrevious() {
            let target = max(0, index - 1)
            player.removeAllItems()
            entry.removeAll()
            index = target
            sync()
            if wantsPlaying { player.play() }
        }

        private func clip(at i: Int) -> String {
            while timeline.count <= i { timeline.append(sequence.next()) }
            return timeline[i]
        }

        private func makeItem(at i: Int) -> AVPlayerItem? {
            guard let url = Bundle.main.url(forResource: "guide_\(clip(at: i))", withExtension: "mp4")
            else { return nil }
            let item = AVPlayerItem(url: url)
            entry[ObjectIdentifier(item)] = i
            return item
        }

        /// Takes the playing entry from the player, and queues up to two more after it.
        private func sync() {
            if let current = player.currentItem, let i = entry[ObjectIdentifier(current)] {
                index = i
            }
            var next = (player.items().last.flatMap { entry[ObjectIdentifier($0)] } ?? index - 1) + 1
            // Bounded, so a missing file cannot spin this forever.
            for _ in 0..<10 where next <= index + 2 {
                if let item = makeItem(at: next) { player.insert(item, after: nil) }
                next += 1
            }
        }
    }
}

private final class GuidePlayerView: UIView {
    override class var layerClass: AnyClass { AVPlayerLayer.self }
    var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }
}
