// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

/// Deterministic uniform random numbers for tests (Numerical Recipes LCG). Values in [0, 1].
struct TestLCG {
    var state: UInt32

    init(seed: UInt32) { state = seed }

    mutating func next() -> Float {
        state = state &* 1664525 &+ 1013904223
        return Float(state) / Float(UInt32.max)
    }
}
