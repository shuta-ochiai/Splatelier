// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Splatelier's shared design system (colors, corner radius, card surfaces).

import SwiftUI

enum DesignSystem {
    /// Unified corner radius for content cards.
    static let cardRadius: CGFloat = 16
    /// Widest a column of cards and buttons grows. On an iPhone the screen is narrower, so they
    /// span it; in a wide iPad window they stay this wide, centered, instead of stretching across.
    static let readableWidth: CGFloat = 640

    /// Brand cyan / magenta (measured from the logo). The tint lives in the AccentColor asset;
    /// these are for decoration.
    static let brandCyan = Color(red: 0x2B / 255, green: 0xD9 / 255, blue: 0xD2 / 255)
    static let brandMagenta = Color(red: 0xD1 / 255, green: 0x4E / 255, blue: 0xD8 / 255)

    /// Decorative cyan→magenta gradient.
    static let brandGradient = LinearGradient(
        colors: [brandCyan, brandMagenta],
        startPoint: .topLeading, endPoint: .bottomTrailing
    )
}

extension View {
    /// Standard content card background (material + unified corner radius). The caller adds padding
    /// / frame.
    /// The app's indigo behind a navigation screen (and behind a List's rows, which would
    /// otherwise draw their own background).
    func appBackground() -> some View {
        scrollContentBackground(.hidden)
            .containerBackground(Color.appBackground, for: .navigation)
    }

    func cardBackground() -> some View {
        background(
            .thinMaterial,
            in: RoundedRectangle(cornerRadius: DesignSystem.cardRadius, style: .continuous)
        )
    }
}

extension Color {
    /// The kikyo purple of the icon and the animations, for filled buttons (white text on it reads
    /// at 4.6:1). Text and icons use the lighter AccentColor (#9A82F2), which reads on the dark UI.
    static let kikyoFill = Color(red: 0x7C / 255, green: 0x5C / 255, blue: 0xE6 / 255)

    /// The deep indigo of the launch screen and the animations, behind every screen except the
    /// ones that show photos or the 3D scene (those stay black).
    static let appBackground = Color(red: 0x16 / 255, green: 0x1B / 255, blue: 0x46 / 255)
}
