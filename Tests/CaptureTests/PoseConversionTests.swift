// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// PoseConversion (ARKit ↔ trainer coordinate conventions). ARKit / MLX-independent.
// viewmat(fromCameraToWorld:) converts the ARKit OpenGL camera (+Z toward the user) to the OpenCV
// camera (+Z forward), so it left-multiplies w2c by diag(1,-1,-1,1). Expected values are the
// converted values with the Y/Z component signs flipped.

import Foundation
import simd
import Testing
@testable import Capture

// MARK: - Identity

@Test
func poseConversionIdentityCamera2World() {
    // c2w = I → w2c = I (ARKit) → flip * I = diag(1,-1,-1,1) (trainer)
    let v = PoseConversion.viewmat(fromCameraToWorld: matrix_identity_float4x4)
    let expected: [Float] = [
        1,  0,  0, 0,
        0, -1,  0, 0,
        0,  0, -1, 0,
        0,  0,  0, 1,
    ]
    #expect(v.count == 16)
    for i in 0..<16 {
        #expect(abs(v[i] - expected[i]) < 1e-6, "idx \(i): got \(v[i]) expected \(expected[i])")
    }
}

// MARK: - Forward-direction invariant

/// A point in front in ARKit (z < 0) gets cam.z > 0 in the trainer convention. If this breaks,
/// project_to_screen culls every splat and training becomes a no-op.
@Test
func poseConversionARKitForwardPointMapsToPositiveCameraZ() {
    let c2w = matrix_identity_float4x4
    let viewmat = PoseConversion.viewmat(fromCameraToWorld: c2w)
    // 5 m from the camera origin along -Z
    let pWorld = SIMD3<Float>(0, 0, -5)
    func apply(_ p: SIMD3<Float>) -> SIMD3<Float> {
        let x = viewmat[0]*p.x + viewmat[1]*p.y + viewmat[2]*p.z  + viewmat[3]
        let y = viewmat[4]*p.x + viewmat[5]*p.y + viewmat[6]*p.z  + viewmat[7]
        let z = viewmat[8]*p.x + viewmat[9]*p.y + viewmat[10]*p.z + viewmat[11]
        return SIMD3<Float>(x, y, z)
    }
    let pCam = apply(pWorld)
    // The trainer convention is forward = +Z
    #expect(abs(pCam.x) < 1e-5, "expected x≈0, got \(pCam.x)")
    #expect(abs(pCam.y) < 1e-5, "expected y≈0, got \(pCam.y)")
    #expect(pCam.z > 0, "ARKit forward point should map to positive cam.z, got \(pCam.z)")
    #expect(abs(pCam.z - 5) < 1e-5, "expected z=5, got \(pCam.z)")
}

// MARK: - Translation only

/// Camera center world (3, 4, 5), rotation I. The w2c_arkit translation (-3, -4, -5) becomes (-3,
/// 4, 5) after the flip.
@Test
func poseConversionPureTranslation() {
    // column-major: columns.3 is the translation
    let c2w = simd_float4x4(columns: (
        SIMD4<Float>(1, 0, 0, 0),
        SIMD4<Float>(0, 1, 0, 0),
        SIMD4<Float>(0, 0, 1, 0),
        SIMD4<Float>(3, 4, 5, 1)
    ))
    let v = PoseConversion.viewmat(fromCameraToWorld: c2w)
    // Rotation part: flip * I = diag(1,-1,-1)
    #expect(abs(v[0] - 1) < 1e-6 && abs(v[5] - (-1)) < 1e-6 && abs(v[10] - (-1)) < 1e-6)
    #expect(abs(v[3]  - (-3)) < 1e-5, "t.x = \(v[3])")
    #expect(abs(v[7]  -  4)   < 1e-5, "t.y = \(v[7])")
    #expect(abs(v[11] -  5)   < 1e-5, "t.z = \(v[11])")
    #expect(v[12] == 0 && v[13] == 0 && v[14] == 0 && abs(v[15] - 1) < 1e-6)
}

// MARK: - Rotation only

/// With a c2w of 90° yaw about the Y axis, the world point (1, 0, 0) lands on the camera z axis
/// (|z| = 1).
@Test
func poseConversionPureYawRotation() {
    let q = simd_quatf(angle: .pi / 2, axis: SIMD3<Float>(0, 1, 0))
    let c2w = simd_float4x4(q)
    let viewmat = PoseConversion.viewmat(fromCameraToWorld: c2w)
    func apply(_ p: SIMD3<Float>) -> SIMD3<Float> {
        let x = viewmat[0]*p.x + viewmat[1]*p.y + viewmat[2]*p.z  + viewmat[3]
        let y = viewmat[4]*p.x + viewmat[5]*p.y + viewmat[6]*p.z  + viewmat[7]
        let z = viewmat[8]*p.x + viewmat[9]*p.y + viewmat[10]*p.z + viewmat[11]
        return SIMD3<Float>(x, y, z)
    }
    let cam = apply(SIMD3<Float>(1, 0, 0))
    #expect(abs(cam.x) < 1e-5, "expected x=0, got \(cam.x)")
    #expect(abs(cam.y) < 1e-5)
    #expect(abs(abs(cam.z) - 1) < 1e-5, "expected |z|=1, got \(cam.z)")
}

// MARK: - Round trip

/// `cameraToWorld(fromViewmat:)` and `viewmat(fromCameraToWorld:)` are inverses of each other for
/// 20 random SE(3) (checks the flip is handled correctly on both sides).
@Test
func poseConversionRoundTripRandomSE3() {
    var seed: UInt64 = 0xC0FFEE
    func nextFloat() -> Float {
        seed = seed &* 6364136223846793005 &+ 1442695040888963407
        return Float(seed >> 32) / Float(UInt32.max) * 2 - 1  // [-1, 1]
    }
    for _ in 0..<20 {
        let axis = simd_normalize(SIMD3<Float>(nextFloat(), nextFloat(), nextFloat()))
        let angle = nextFloat() * .pi
        let q = simd_quatf(angle: angle, axis: axis)
        let t = SIMD3<Float>(nextFloat() * 3, nextFloat() * 3, nextFloat() * 3)
        var m = simd_float4x4(q)
        m.columns.3 = SIMD4<Float>(t.x, t.y, t.z, 1)

        let vm = PoseConversion.viewmat(fromCameraToWorld: m)
        let back = PoseConversion.cameraToWorld(fromViewmat: vm)
        // columns is a tuple, so repack into arrays and compare every element
        let mCols = [m.columns.0, m.columns.1, m.columns.2, m.columns.3]
        let bCols = [back.columns.0, back.columns.1, back.columns.2, back.columns.3]
        for c in 0..<4 {
            let a = mCols[c]
            let b = bCols[c]
            for r in 0..<4 {
                let aComp = (r == 0) ? a.x : (r == 1) ? a.y : (r == 2) ? a.z : a.w
                let bComp = (r == 0) ? b.x : (r == 1) ? b.y : (r == 2) ? b.z : b.w
                #expect(abs(aComp - bComp) < 1e-4,
                    "round-trip diff at [\(c),\(r)]: \(aComp) vs \(bComp)")
            }
        }
    }
}

// MARK: - world-to-camera direct path

/// `viewmat(fromWorldToCamera: c2w.inverse)` matches `viewmat(fromCameraToWorld: c2w)`.
@Test
func poseConversionWorldToCameraDirectMatchesInverse() {
    let q = simd_quatf(angle: .pi / 3, axis: SIMD3<Float>(0.4, 0.6, 0.7))
    var c2w = simd_float4x4(q)
    c2w.columns.3 = SIMD4<Float>(2.5, -1.0, 4.2, 1)

    let viaInverse = PoseConversion.viewmat(fromCameraToWorld: c2w)
    let viaDirect  = PoseConversion.viewmat(fromWorldToCamera: c2w.inverse)
    for i in 0..<16 {
        #expect(abs(viaInverse[i] - viaDirect[i]) < 1e-5,
            "idx \(i): \(viaInverse[i]) vs \(viaDirect[i])")
    }
}

// MARK: - Intrinsics scaling

@Test
func intrinsicsScalingIdentity() {
    let (fx, fy, cx, cy) = PoseConversion.scaledIntrinsics(
        fx: 1500, fy: 1500, cx: 960, cy: 720,
        originalWidth: 1920, originalHeight: 1440,
        scaledWidth: 1920, scaledHeight: 1440
    )
    #expect(fx == 1500 && fy == 1500 && cx == 960 && cy == 720)
}

@Test
func intrinsicsScalingHalfResolution() {
    let (fx, fy, cx, cy) = PoseConversion.scaledIntrinsics(
        fx: 1500, fy: 1500, cx: 960, cy: 720,
        originalWidth: 1920, originalHeight: 1440,
        scaledWidth: 960, scaledHeight: 720
    )
    #expect(abs(fx - 750) < 1e-5)
    #expect(abs(fy - 750) < 1e-5)
    #expect(abs(cx - 480) < 1e-5)
    #expect(abs(cy - 360) < 1e-5)
}

@Test
func intrinsicsScalingAsymmetric() {
    // 1920×1440 → 512×384 (1/3.75 in both dimensions)
    let (fx, fy, cx, cy) = PoseConversion.scaledIntrinsics(
        fx: 1500, fy: 1500, cx: 960, cy: 720,
        originalWidth: 1920, originalHeight: 1440,
        scaledWidth: 512, scaledHeight: 384
    )
    let sx: Float = 512.0 / 1920.0
    let sy: Float = 384.0 / 1440.0
    #expect(abs(fx - 1500 * sx) < 1e-5)
    #expect(abs(fy - 1500 * sy) < 1e-5)
    #expect(abs(cx - 960 * sx) < 1e-5)
    #expect(abs(cy - 720 * sy) < 1e-5)
}
