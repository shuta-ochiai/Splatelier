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

/// Bitset of the point indices in `indices`, for `covisibleLoopClosureCandidates`.
private func bits(_ indices: Range<Int>, total: Int) -> [UInt64] {
    var b = [UInt64](repeating: 0, count: (total + 63) / 64)
    for i in indices { b[i / 64] |= 1 << UInt64(i % 64) }
    return b
}

/// A capture that starts with frames seeing one part of the scene from far back, looks elsewhere,
/// and comes back to it from close up: the camera-distance candidates miss the return, the
/// co-visibility candidates find it, and a frame's partners are spread over the passes.
@Test
func covisibleLoopClosurePairsFramesThatSeeTheSamePoints() {
    let n = 120, total = 300
    // Frames 0-19 stand 2 m back and see points 0-99; 20-79 see 100-299; 80-119 stand close and
    // see points 0-99 again. All face +z.
    let vms = (0..<n).map { i -> [Float] in
        let z: Float = i < 20 ? -2 : (i >= 80 ? 0 : -1)
        return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -z, 0, 0, 0, 1]
    }
    let visible = (0..<n).map { i in i < 20 || i >= 80 ? bits(0..<100, total: total) : bits(100..<300, total: total) }

    let byDistance = CaptureBA.loopClosureCandidates(viewmats: vms)
    #expect(!byDistance.contains { $0.a < 20 && $0.b >= 80 }, "camera distance should miss the return from close up")

    let cands = CaptureBA.covisibleLoopClosureCandidates(viewmats: vms, visible: visible)
    #expect(cands.contains { $0.a < 20 && $0.b >= 80 }, "frames seeing the same points should pair")
    #expect(!cands.contains { ($0.a < 20 || $0.a >= 80) && ($0.b >= 20 && $0.b < 80) },
            "frames that share no points must not pair")
    #expect(cands.allSatisfy { $0.b - $0.a >= 30 }, "partners are at least minGap frames away")
    for a in Set(cands.map(\.a)) {
        let partners = cands.filter { $0.a == a }.map(\.b).sorted()
        #expect(partners.count <= 6)
        #expect(zip(partners, partners.dropFirst()).allSatisfy { $1 - $0 >= 30 },
                "frame \(a)'s partners \(partners) should be spread over the capture")
    }
}

/// Frames facing away from each other do not pair even when they share points.
@Test
func covisibleLoopClosureSkipsOppositeViews() {
    let n = 80, total = 64
    let vms = (0..<n).map { i -> [Float] in
        let s: Float = i < 40 ? 1 : -1   // the second half faces -z
        return [s, 0, 0, 0, 0, 1, 0, 0, 0, 0, s, 0, 0, 0, 0, 1]
    }
    let visible = (0..<n).map { _ in bits(0..<64, total: total) }
    let cands = CaptureBA.covisibleLoopClosureCandidates(viewmats: vms, visible: visible)
    #expect(!cands.isEmpty, "frames facing the same way should pair")
    #expect(cands.allSatisfy { ($0.a < 40) == ($0.b < 40) }, "opposite views must not pair: \(cands)")
}
