// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Features the App Store version leaves out for now. A normal build (Debug / Release) has them all;
// the AppStore configuration (the "Splatelier App Store" scheme, see App/project.yml) builds with
// APP_STORE and turns them off. The App Store version stops at the trained splats, and keeps out
// what is experimental or only there for development.

import Foundation

enum AppFeatures {
    #if APP_STORE
    /// The mesh (OBJ) made from the trained splats after training, shown in the viewer and shared.
    static let meshGeneration = false
    /// CAD models (STEP) added to a capture and placed in the viewer.
    static let cadModels = false
    /// The experimental "LiDAR depth" switch on the capture screen (off: captures never record it).
    static let lidarDepth = false
    /// Saving a capture as a validation dataset, and the list of them to train again from.
    static let validationDatasets = false
    /// Review frames after a capture: every frame with the quality check's verdict. The capture
    /// card's line of how many frames pass the check stays.
    static let frameReview = false
    /// What is there for development: GPU memory and loss while training, the training settings
    /// of a capture, the seed points and capture cameras in the viewer, the relocalization notice
    /// after a capture, and sharing the training logs. Off, the files behind them aren't written
    /// either (the log, the seed PLY, the settings and metrics sidecars).
    static let developerInfo = false
    /// The iterations and Gaussian cap before training (off: the defaults, the recommendation for
    /// the frame count and `AppPreset.defaultGaussianCap`).
    static let trainingSettings = false
    /// The links to the GitHub repository and its issues in About. The licenses stay.
    static let projectLinks = false
    /// The experimental wide-angle loop closure with FlexGlue. The AppStore configuration doesn't
    /// bundle its models either (App/project.yml), so the switch never shows there.
    static let wideLoopClosure = false
    #else
    static let meshGeneration = true
    static let cadModels = true
    static let lidarDepth = true
    static let validationDatasets = true
    static let frameReview = true
    static let developerInfo = true
    static let trainingSettings = true
    static let projectLinks = true
    static let wideLoopClosure = true
    #endif

    /// The environment variables unattended A/B runs are launched with (`devicectl …
    /// --environment-variables`, `SPLATELIER_AUTOTRAIN` and the like). The App Store version reads
    /// none of them.
    static var developerEnvironment: [String: String] {
        #if APP_STORE
        [:]
        #else
        ProcessInfo.processInfo.environment
        #endif
    }
}
