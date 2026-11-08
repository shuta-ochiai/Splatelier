// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Whether this device can train. Training's kernels sum and scan across a SIMD group (simd_sum,
// simd_prefix_*), which Apple GPUs have from the Apple7 family on (A14, M1). An A12 or A13 device
// (iPhone XS, XR, 11, SE 2nd and 3rd generation… iPad 9th generation) meets the app's other
// requirements (iOS 18, ARKit), so it can install the app but not train; capture and training say
// so there, and the Library still opens splat files.

import Metal

enum DeviceSupport {
    static let canTrain: Bool = {
        // Lets a development build show the unsupported state on a supported device.
        if AppFeatures.developerEnvironment["SPLATELIER_UNSUPPORTED_GPU"] != nil { return false }
        return MTLCreateSystemDefaultDevice()?.supportsFamily(.apple7) ?? false
    }()

    static let unsupportedMessage =
        "Training needs an A14 chip or newer: iPhone 12 or later, or iPad Air (4th generation), "
        + "iPad (10th generation), iPad mini (6th generation), an iPad with an M chip, or later. "
        + "You can still open splat files in the Library."
}
