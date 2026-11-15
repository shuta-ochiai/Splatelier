// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import XCTest
import simd
@testable import Trainer

final class MeshPlanarityTests: XCTestCase {

    /// A surface swept along x (0...length) from a profile in (y, z), resampled at `step`;
    /// `offset(x, s)` moves each vertex along +y (s: arc length along the profile).
    private func extrusion(profile: [SIMD2<Float>], length: Float = 1, step: Float = 0.01,
                           offset: (Float, Float) -> Float = { _, _ in 0 })
        -> (positions: [SIMD3<Float>], indices: [UInt32]) {
        var samples: [SIMD2<Float>] = [profile[0]]
        var arcs: [Float] = [0]
        var arc: Float = 0
        for k in 1 ..< profile.count {
            let a = profile[k - 1], b = profile[k]
            let n = max(1, Int((simd_length(b - a) / step).rounded()))
            for i in 1 ... n {
                samples.append(a + (b - a) * Float(i) / Float(n))
                arc += simd_length(b - a) / Float(n)
                arcs.append(arc)
            }
        }
        let nx = Int((length / step).rounded()) + 1
        var positions: [SIMD3<Float>] = []
        for i in 0 ..< nx {
            let x = Float(i) * step
            for (k, s) in samples.enumerated() {
                positions.append(SIMD3(x, s.x + offset(x, arcs[k]), s.y))
            }
        }
        var indices: [UInt32] = []
        let ns = samples.count
        for i in 0 ..< nx - 1 {
            for k in 0 ..< ns - 1 {
                let a = UInt32(i * ns + k), b = UInt32((i + 1) * ns + k)
                indices += [a, b, a + 1, a + 1, b, b + 1]
            }
        }
        return (positions, indices)
    }

    /// Deterministic pseudo-noise in [-1, 1].
    private func noise(_ x: Float, _ s: Float) -> Float {
        let h = sin(x * 12.9898 * 100 + s * 78.233 * 100) * 43758.5453
        return 2 * (h - h.rounded(.down)) - 1
    }

    func testSymmetricEigen() {
        let (values, vectors) = symmetricEigen3([4, 1, 0, 1, 3, 0, 0, 0, 1])
        XCTAssertEqual(values[0], 1, accuracy: 1e-9)
        XCTAssertEqual(values[1], (7 - 5.0.squareRoot()) / 2, accuracy: 1e-9)
        XCTAssertEqual(values[2], (7 + 5.0.squareRoot()) / 2, accuracy: 1e-9)
        XCTAssertEqual(abs(vectors[0].z), 1, accuracy: 1e-9)
    }

    func testNoisyPlaneRMS() {
        // A flat 1 m × 1 m sheet with uniform noise of ±2√3 mm (σ = 2 mm).
        let sigma: Float = 0.002
        let m = extrusion(profile: [SIMD2(0, 0), SIMD2(0, 1)],
                          offset: { x, s in sigma * Float(3).squareRoot() * self.noise(x, s) })
        let planes = MeshPlanarity.detectPlanes(positions: m.positions, indices: m.indices)
        XCTAssertEqual(planes.count, 1)
        let areas = MeshPlanarity.regionAreas(positions: m.positions, indices: m.indices, planes: planes)
        let r = MeshPlanarity.evaluate(positions: m.positions, indices: m.indices, planes: planes,
                                       corners: [], referenceAreas: areas)
        // Face centroids average three vertices, so their spread is σ/√3 or a little more.
        XCTAssertEqual(r.planeRMS, sigma / Float(3).squareRoot(), accuracy: 0.4 * sigma)
        XCTAssertEqual(r.coverage, 1, accuracy: 1e-4)
    }

    func testChamferedCornerScoresWorseThanSharp() {
        let sharp = extrusion(profile: [SIMD2(0, 1), SIMD2(0, 0), SIMD2(1, 0)])
        let chamfered = extrusion(profile: [SIMD2(0, 1), SIMD2(0, 0.04), SIMD2(0.04, 0), SIMD2(1, 0)])
        let planes = MeshPlanarity.detectPlanes(positions: sharp.positions, indices: sharp.indices)
        XCTAssertEqual(planes.count, 2)
        let corners = MeshPlanarity.detectCorners(planes: planes, positions: sharp.positions, indices: sharp.indices)
        XCTAssertEqual(corners.count, 1)
        let areas = MeshPlanarity.regionAreas(positions: sharp.positions, indices: sharp.indices, planes: planes)
        let a = MeshPlanarity.evaluate(positions: sharp.positions, indices: sharp.indices, planes: planes,
                                       corners: corners, referenceAreas: areas)
        let b = MeshPlanarity.evaluate(positions: chamfered.positions, indices: chamfered.indices, planes: planes,
                                       corners: corners, referenceAreas: areas)
        XCTAssertLessThan(a.cornerRMS, 0.0005)
        XCTAssertGreaterThan(b.cornerRMS, 0.005)
        XCTAssertEqual(a.cornerFill, 1, accuracy: 0.05)
        XCTAssertLessThan(b.planeRMS, 0.0005)
    }

    func testCylinderIsNotAPlane() {
        // A post of radius 15 cm: its strips within the plane tolerances are narrower than minExtent.
        let profile = (0 ... 94).map { k -> SIMD2<Float> in
            let a = Float(k) / 94 * 2 * .pi
            return SIMD2(0.15 * cos(a), 0.15 * sin(a))
        }
        let m = extrusion(profile: profile)
        XCTAssertTrue(MeshPlanarity.detectPlanes(positions: m.positions, indices: m.indices).isEmpty)
    }
}
