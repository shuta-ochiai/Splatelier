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

/// The app's Info.plist gives the build number and commit; without the commit key (the CLI, the
/// tests, an app built without git) there is no build and the description is the version alone.
@Test
func buildComesFromTheStampedInfoKeys() {
    let stamped: [String: Any] = ["CFBundleVersion": "412", "SplatelierCommit": "1a2b3c4d5e6f"]
    let build = Splatelier.Build(info: stamped)
    #expect(build?.number == "412")
    #expect(build?.commit == "1a2b3c4d5e6f")
    #expect(Splatelier.describe(version: "0.6.0", build: build) == "0.6.0 (build 412, 1a2b3c4d5e6f)")

    #expect(Splatelier.Build(info: ["CFBundleVersion": "1"]) == nil)
    #expect(Splatelier.Build(info: nil) == nil)
    #expect(Splatelier.describe(version: "0.6.0", build: nil) == "0.6.0")
}
