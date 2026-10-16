// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Geometry logic of CaptureBA.loopClosureCandidates. On a synthetic orbit facing the origin, the
// start↔end become loop candidates and facing frames (180° opposite) are excluded.
// Image-independent.

import Foundation
import simd
import Testing
@testable import Capture

/// w2c viewmat (row-major [16]) of an orbit camera looking at the origin, at radius radius and
/// angle angleDeg. R rows = camera axes (world), t = -R·C. Then cameraCenter=-Rᵀt=C, forward = 3rd
/// row of R = view direction.
private func orbitViewmat(angleDeg: Float, radius: Float) -> [Float] {
    let a = angleDeg * .pi / 180
    let c = SIMD3<Float>(radius * cos(a), 0, radius * sin(a))   // Camera center
    let z = simd_normalize(SIMD3<Float>(0, 0, 0) - c)            // forward (toward the origin)
    let x = simd_normalize(simd_cross(SIMD3<Float>(0, 1, 0), z))
    let y = simd_cross(z, x)
    let t = SIMD3<Float>(-simd_dot(x, c), -simd_dot(y, c), -simd_dot(z, c))
    return [
        x.x, x.y, x.z, t.x,
        y.x, y.y, y.z, t.y,
        z.x, z.y, z.z, t.z,
        0, 0, 0, 1,
    ]
}

@Test
func loopClosureDetectsOrbitWraparound() {
    let n = 30
    let vms = (0..<n).map { orbitViewmat(angleDeg: Float($0) / Float(n) * 360, radius: 1) }
    let cands = CaptureBA.loopClosureCandidates(viewmats: vms)   // Default minGap=10

    #expect(!cands.isEmpty, "orbit should yield loop candidates near the wrap")
    // The orbit's start↔end (0, n-1) are spatially close + aligned in view direction, so they
    // should be candidates.
    #expect(cands.contains { $0.a == 0 && $0.b == n - 1 },
            "wrap-around pair (0, \(n - 1)) expected; got \(cands)")
    // Facing frames (0, n/2) are far apart + opposite in view direction, so they must not be
    // candidates.
    #expect(!cands.contains { $0.a == 0 && $0.b == n / 2 },
            "opposite-side pair (0, \(n / 2)) must be rejected")
    // All candidates satisfy minGap and a<b.
    #expect(cands.allSatisfy { $0.b - $0.a >= 10 && $0.a < $0.b },
            "all candidates must satisfy minGap and a<b")
}

/// A straight pan (a capture with no loop) yields zero candidates.
@Test
func loopClosureEmptyForLinearPan() {
    let n = 30
    // Evenly spaced translation along +x, all view directions +z (R = I, t = -C).
    let vms = (0..<n).map { i -> [Float] in
        let cx = Float(i) * 0.1
        return [
            1, 0, 0, -cx,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ]
    }
    let cands = CaptureBA.loopClosureCandidates(viewmats: vms)
    // The view directions are all identical, but the two ends are far apart, so the proximity
    // condition drops them.
    #expect(!cands.contains { $0.a == 0 && $0.b == n - 1 },
            "linear pan endpoints are far apart; (0, n-1) must not be a candidate")
}
