// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Viewer tools: Measure, Color, Crop and Floaters are never on together, so only one panel sits
// over the view.

import Testing
@testable import SplatViewerKit

@MainActor
@Suite struct SplatViewerEditModeTests {
    private func activeTools(_ m: SplatViewerModel) -> [String] {
        [("measure", m.isMeasuring), ("color", m.isColoring),
         ("crop", m.isCropping), ("floaters", m.isRemovingFloaters)]
            .filter(\.1).map(\.0)
    }

    @Test func turningAToolOnTurnsTheOthersOff() {
        let m = SplatViewerModel()
        let turnOn: [(String, () -> Void)] = [
            ("measure", { m.isMeasuring = true }),
            ("color", { m.isColoring = true }),
            ("crop", { m.isCropping = true }),
            ("floaters", { m.isRemovingFloaters = true }),
        ]
        // Every ordered pair: the second one switched on is the only one left on.
        for (first, onFirst) in turnOn {
            for (second, onSecond) in turnOn where second != first {
                onFirst()
                #expect(activeTools(m) == [first])
                onSecond()
                #expect(activeTools(m) == [second], "\(first) then \(second)")
            }
        }
    }
}
