// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Splatelier's shared design system (colors, corner radius, card surfaces).

import SwiftUI

enum DesignSystem {
    /// Unified corner radius for content cards.
    static let cardRadius: CGFloat = 16

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
    func cardBackground() -> some View {
        background(
            .thinMaterial,
            in: RoundedRectangle(cornerRadius: DesignSystem.cardRadius, style: .continuous)
        )
    }
}
