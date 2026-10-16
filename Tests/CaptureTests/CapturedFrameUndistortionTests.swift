// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// CapturedFrame's lens undistortion spec (the image is built from the captured image on each use).

import Foundation
import MLX
import Testing
@testable import Trainer
@testable import Capture

@Suite struct CapturedFrameUndistortionTests {

    static func frame() -> CapturedFrame {
        let h = 24, w = 32
        let bytes = (0..<(h * w * 3)).map { UInt8(($0 * 37) % 251) }
        return CapturedFrame(
            image: MLXArray(bytes, [h, w, 3]),
            viewmat: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            intrinsics: (fx: 30, fy: 30, cx: 16, cy: 12),
            width: w, height: h, timestamp: 1)
    }

    /// A frame with the spec has the undistorted captured image as its image, and its intrinsics
    /// become the target.
    @Test func undistortedFrameResamplesOnAccess() {
        let f = Self.frame()
        let d = RadialDistortion(k1: -0.08, k2: 0.01)
        let target: Intrinsics = (fx: 29, fy: 29, cx: 16, cy: 12)
        let u = f.undistorted(target: target, distortion: d)
        #expect(f.lensUndistortion == nil)
        #expect(u.lensUndistortion?.source.fx == 30)
        #expect(u.intrinsics.fx == 29)
        let expected = LensUndistortion.undistortImage(f.image, source: f.intrinsics, target: target, distortion: d)
        #expect(u.image.dtype == .uint8)
        #expect(u.image.asArray(UInt8.self) == expected.asArray(UInt8.self))
        #expect(u.image.asArray(UInt8.self) != f.image.asArray(UInt8.self))
    }

    /// The undistortion spec survives replacing the pose.
    @Test func withViewmatKeepsUndistortion() {
        let f = Self.frame()
        let u = f.undistorted(target: (fx: 29, fy: 29, cx: 16, cy: 12), distortion: RadialDistortion(k1: -0.08, k2: 0))
        var vm = u.viewmat
        vm[3] = 0.5
        let moved = u.withViewmat(vm)
        #expect(moved.viewmat[3] == 0.5)
        #expect(moved.lensUndistortion != nil)
        #expect(moved.intrinsics.fx == 29)
        #expect(moved.image.asArray(UInt8.self) == u.image.asArray(UInt8.self))
    }
}
