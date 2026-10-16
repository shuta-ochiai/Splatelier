// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import simd
import Testing
@testable import Trainer

/// From a depth map of the tilted plane z = 1 + 0.5x (camera coords), the plane normal (facing the
/// camera) comes out. The side past a depth step (beyond `maxRelativeJump`) is not used; only the
/// opposite side is.
@Test
func depthNormalRecoversTiltedPlane() {
    let w = 64, h = 48
    let intr: Intrinsics = (fx: 60, fy: 60, cx: 32, cy: 24)
    var depth = [Float](repeating: 0, count: w * h)
    for y in 0..<h {
        for x in 0..<w {
            let u = (Float(x) + 0.5 - intr.cx) / intr.fx
            depth[y * w + x] = 1 / (1 - 0.5 * u)   // depth at which the ray (u, v, 1)·d lies on z - 0.5x = 1
        }
    }
    let expected = simd_normalize(SIMD3<Float>(0.5, 0, -1))
    let n = StreamingMVS.depthNormal(
        x: 20, y: 20, z: depth[20 * w + 20], w: w, h: h, intrinsics: intr, radius: 2,
        maxRelativeJump: 0.05, depth: { depth[$0] })
    #expect(n.map { simd_distance($0, expected) < 1e-4 } == true)

    // Even with a far wall on the right, the left side alone gives the same normal.
    var stepped = depth
    for y in 0..<h { for x in 21..<w { stepped[y * w + x] *= 2 } }
    let m = StreamingMVS.depthNormal(
        x: 20, y: 20, z: stepped[20 * w + 20], w: w, h: h, intrinsics: intr, radius: 2,
        maxRelativeJump: 0.05, depth: { stepped[$0] })
    #expect(m.map { simd_distance($0, expected) < 1e-4 } == true)

    // nil if neither side is usable.
    let none = StreamingMVS.depthNormal(
        x: 20, y: 20, z: 1, w: w, h: h, intrinsics: intr, radius: 2,
        maxRelativeJump: 0.05, depth: { $0 == 20 * w + 20 ? 1 : 5 })
    #expect(none == nil)
}
