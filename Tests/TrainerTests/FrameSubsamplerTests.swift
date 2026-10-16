// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Frame selection (PocketGS Operator G, Sec. III-A 1). Tests the helpers and `selectFrames` on a
// synthetic trajectory + known sharpness patterns.

import Foundation
import MLX
import Testing
@testable import Trainer

// MARK: - cameraCenter

/// C = -Rᵀt from a row-major w2c `[R | t]`. With R = I, t = (3, 4, 5), C = -(3, 4, 5).
@Test
func cameraCenterIdentityRotation() {
    let vm: [Float] = [
        1, 0, 0, 3,
        0, 1, 0, 4,
        0, 0, 1, 5,
        0, 0, 0, 1,
    ]
    let c = FrameSubsampler.cameraCenter(viewmat: vm)
    #expect(abs(c.x - (-3)) < 1e-6)
    #expect(abs(c.y - (-4)) < 1e-6)
    #expect(abs(c.z - (-5)) < 1e-6)
}

/// 90° about z (R = [[0,-1,0],[1,0,0],[0,0,1]]), t = (1, 0, 0) gives Rᵀt = (0, -1, 0), so C = (0,
/// 1, 0).
@Test
func cameraCenterRotated() {
    let vm: [Float] = [
        0, -1, 0, 1,
        1,  0, 0, 0,
        0,  0, 1, 0,
        0,  0, 0, 1,
    ]
    let c = FrameSubsampler.cameraCenter(viewmat: vm)
    #expect(abs(c.x - 0) < 1e-6)
    #expect(abs(c.y - 1) < 1e-6)
    #expect(abs(c.z - 0) < 1e-6)
}

// MARK: - displacementGate

@Test
func displacementGateLinearTrajectory() {
    // 10 frames 0.03 m apart. With τ_d = 0.05 m every other frame is kept.
    var centers: [SIMD3<Float>] = []
    for i in 0..<10 {
        centers.append(SIMD3(Float(i) * 0.03, 0, 0))
    }
    let kept = FrameSubsampler.displacementGate(centers: centers, tauD: 0.05)
    // The first is kept unconditionally; 0.03 < 0.05 from the previous kept → drop, 0.06 ≥ 0.05 →
    // kept, ...
    let expected: [Bool] = [
        true, false, true, false, true, false, true, false, true, false,
    ]
    #expect(kept == expected, "displacementGate produced \(kept), expected \(expected)")
}

@Test
func displacementGateEmpty() {
    let kept = FrameSubsampler.displacementGate(centers: [], tauD: 0.05)
    #expect(kept == [])
}

@Test
func displacementGateSingleFrame() {
    let kept = FrameSubsampler.displacementGate(
        centers: [SIMD3<Float>(0, 0, 0)], tauD: 0.05
    )
    #expect(kept == [true], "single frame should always be kept")
}

// MARK: - sharpnessScore

/// A uniform gray image has zero difference everywhere, sharpness S = 0.
@Test
func sharpnessScoreConstantImageIsZero() {
    let h = 32, w = 32
    let rgb = MLXArray(
        [Float](repeating: 0.5, count: h * w * 3), [h, w, 3]
    )
    let s = FrameSubsampler.sharpnessScore(rgb: rgb, stride: 4)
    #expect(s < 1e-6, "constant image got non-zero sharpness: \(s)")
}

/// A high-frequency checker is sharp; a smooth linear ramp is dull.
@Test
func sharpnessScoreCheckerHigherThanBlur() {
    let h = 32, w = 32
    var checker: [Float] = []
    for y in 0..<h {
        for x in 0..<w {
            let v: Float = ((x + y) % 2 == 0) ? 1.0 : 0.0
            checker += [v, v, v]
        }
    }
    let sharp = MLXArray(checker, [h, w, 3])
    // Blurry: linear ramp. Adjacent pixel differences are small, ~1/w
    var ramp: [Float] = []
    for y in 0..<h {
        for x in 0..<w {
            let v: Float = Float(x) / Float(w)
            ramp += [v, v, v]
            _ = y
        }
    }
    let blur = MLXArray(ramp, [h, w, 3])
    let sSharp = FrameSubsampler.sharpnessScore(rgb: sharp, stride: 1)
    let sBlur = FrameSubsampler.sharpnessScore(rgb: blur, stride: 1)
    #expect(sSharp > sBlur,
        "checker sharpness \(sSharp) should exceed blur \(sBlur)")
    #expect(sSharp > 0.3, "checker should be clearly sharp; got \(sSharp)")
    #expect(sBlur < 0.1, "blur should be low; got \(sBlur)")
}

// MARK: - selectFrames

/// selectFrames keeps every frame that passes the displacement gate, in order.
@Test
func selectFramesKeepsAllDisplacementSurvivors() {
    let centersX: [Float] = [0, 0.01, 0.10, 0.11, 0.20, 0.21, 0.30, 0.31]
    let viewmats: [[Float]] = centersX.map { x in
        [1, 0, 0, x, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    }
    #expect(FrameSubsampler.selectFrames(viewmats: viewmats, tauD: 0.05) == [0, 2, 4, 6])
    #expect(FrameSubsampler.selectFrames(viewmats: []) == [])
}
