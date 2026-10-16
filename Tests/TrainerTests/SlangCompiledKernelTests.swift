// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import Testing
@testable import Trainer

/// Unit test for the parser of the generated `.metal`. `adam_step` has param/grad/m/v/hyper as
/// inputs (buffers 0-4) and param/m/v as outputs (buffers 5-7). Dispatch is covered by
/// AdamOptimizerTests.
@Test
func slangCompiledKernelParsesAdamStep() throws {
    let url = try #require(
        Bundle.module.url(forResource: "adam_step", withExtension: "metal", subdirectory: "Slang"))
    let parsed = try SlangCompiledKernel.parseMetal(String(contentsOf: url, encoding: .utf8))

    #expect(parsed.entry == "adam_step")
    #expect(parsed.bufferParams.map(\.bufferIndex).sorted() == Array(0..<8))
    #expect(parsed.semanticParams.contains { $0.semantic == "thread_position_in_grid" })
    #expect(parsed.bufferParams.map(\.name).contains { parsed.body.contains($0) })
}
