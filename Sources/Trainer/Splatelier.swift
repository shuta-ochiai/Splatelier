// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation

/// Namespace for the splatelier library.
public enum Splatelier {
    /// The release version. `scripts/version.sh set` writes it here and in App/Config/Version.xcconfig
    /// (the app's CFBundleShortVersionString), and CI checks that the two and the release tag agree.
    public static let version = "0.6.0"

    /// The app build this runs in. The app's "Stamp the build" script writes both into its
    /// Info.plist; the CLI and the tests have neither.
    public struct Build: Sendable, Equatable {
        /// CFBundleVersion: the number of commits up to the commit built.
        public let number: String
        /// The commit built (12 hex digits), with "+dirty" when tracked files had changes.
        public let commit: String

        /// nil unless `info` has both keys: without `SplatelierCommit` the script did not run (no
        /// git), and CFBundleVersion is the placeholder 1.
        public init?(info: [String: Any]?) {
            guard let number = info?["CFBundleVersion"] as? String,
                  let commit = info?["SplatelierCommit"] as? String
            else { return nil }
            self.number = number
            self.commit = commit
        }
    }

    public static let build = Build(info: Bundle.main.infoDictionary)

    /// "0.6.0 (build 412, 1a2b3c4d5e6f)" in the app, "0.6.0" elsewhere.
    public static var versionDescription: String { describe(version: version, build: build) }

    static func describe(version: String, build: Build?) -> String {
        guard let build else { return version }
        return "\(version) (build \(build.number), \(build.commit))"
    }
}
