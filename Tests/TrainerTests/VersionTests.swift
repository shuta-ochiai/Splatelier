// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Testing
@testable import Trainer

@Test
func versionStringFormat() {
    let version = Splatelier.version
    #expect(!version.isEmpty)
    let parts = version.split(separator: ".")
    #expect(parts.count == 3, "version should be in MAJOR.MINOR.PATCH form")
    for p in parts {
        #expect(Int(p) != nil, "version part \(p) should be an integer")
    }
}
