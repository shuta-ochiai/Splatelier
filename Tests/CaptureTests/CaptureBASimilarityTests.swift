// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Tests for the similarity transform (Umeyama) that maps post-BA poses back into the ARKit frame.

import Foundation
import Testing
import simd
@testable import Trainer
@testable import Capture

/// Umeyama recovers a known similarity transform.
@Test
func umeyamaRecoversKnownSimilarity() {
    let src: [SIMD3<Double>] = [
        SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(0, 2, 0), SIMD3(0, 0, 3), SIMD3(1, 1, 1), SIMD3(-1, 0.5, 2),
    ]
    let q = simd_quatd(angle: 0.7, axis: simd_normalize(SIMD3(0.3, -1, 0.2)))
    let R = simd_double3x3(q)
    let s = 1.7, t = SIMD3(0.4, -2.0, 5.0)
    let dst = src.map { s * (R * $0) + t }
    let sim = try! #require(CaptureBA.umeyamaSimilarity(src: src, dst: dst))
    #expect(abs(sim.s - s) < 1e-9)
    #expect(simd_length(sim.t - t) < 1e-9)
    for c in 0..<3 { #expect(simd_length(sim.R[c] - R[c]) < 1e-9) }
    #expect(sim.rms < 1e-9)
}

/// Applying the similarity transform does not change the projection (normalized coordinates) into
/// each camera.
@Test
func applySimilarityPreservesProjections() {
    let vm: [Float] = [
        0.9950042, 0, 0.0998334, -0.05,
        0, 1, 0, 0.02,
        -0.0998334, 0, 0.9950042, 0.1,
        0, 0, 0, 1,
    ]
    let pts: [SIMD3<Float>] = [SIMD3(0.1, 0.2, 1.0), SIMD3(-0.3, 0.1, 0.8), SIMD3(0.2, -0.2, 1.4)]
    let q = simd_quatd(angle: 0.4, axis: simd_normalize(SIMD3(1, 0.2, -0.5)))
    let sim = CaptureBA.Similarity(s: 2.5, R: simd_double3x3(q), t: SIMD3(1, -1, 0.5), rms: 0)
    let out = CaptureBA.applySimilarity(sim, to: .init(viewmats: [vm], points: pts))
    func proj(_ v: [Float], _ p: SIMD3<Float>) -> SIMD2<Float> {
        let x = v[0] * p.x + v[1] * p.y + v[2] * p.z + v[3]
        let y = v[4] * p.x + v[5] * p.y + v[6] * p.z + v[7]
        let z = v[8] * p.x + v[9] * p.y + v[10] * p.z + v[11]
        return SIMD2(x / z, y / z)
    }
    for (p, p2) in zip(pts, out.points) {
        #expect(simd_length(proj(vm, p) - proj(out.viewmats[0], p2)) < 1e-5)
    }
}
