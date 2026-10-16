// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// os_signposts at training-stage completion points. View them in Instruments' Points of Interest
// next to the GPU track and thermal state. MLX evaluates lazily, so CPU-side intervals only measure
// enqueue time; hence point events rather than intervals. Without Instruments attached,
// signpostsEnabled is false and the cost is 0.

import os

enum TrainingSignpost {
    static let log = OSLog(subsystem: "io.github.shuta-ochiai.splatelier", category: .pointsOfInterest)

    @inline(__always)
    static func event(_ stage: String, ms: Double) {
        guard log.signpostsEnabled else { return }
        os_signpost(.event, log: log, name: "stage", "%{public}s %.3fms", stage, ms)
    }
}
