// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import simd
import Testing
@testable import Capture

/// 4 levels with cells of 0.01 / 0.02 / 0.04 / 0.08 m. Looks at the z = 0 plane from straight above
/// (z = 1).
private func makeMap() -> CaptureSurfelMap {
    CaptureSurfelMap(finestCellMeters: 0.01, levelCount: 4, cellsPerPixel: 2,
                     minCosine: 0.3, hideWhenRefinedChildren: 3, minPoints: 1)
}

private func insertGrid(_ map: inout CaptureSurfelMap, xMax: Float, pixelSize: Float, frame: Int) {
    for ix in stride(from: Float(0.005), to: xMax, by: 0.01) {
        for iy in stride(from: Float(0.005), to: 0.08, by: 0.01) {
            map.insert(position: SIMD3(ix, iy, 0.001), normal: SIMD3(0, 0, 1), color: SIMD3(1, 0, 0),
                       pixelSize: pixelSize, cameraCenter: SIMD3(ix, iy, 1), frame: frame)
        }
    }
}

/// Points captured from far away (3.5cm pixels) merge into one 8cm cell and become one large splat.
@Test
func surfelMapPutsDistantObservationsInCoarseCell() {
    var map = makeMap()
    insertGrid(&map, xMax: 0.08, pixelSize: 0.035, frame: 0)
    let s = map.snapshot()
    #expect(s.count == 1)
    #expect(s[0].level == 3)
    #expect(s[0].cellMeters == 0.08)
    #expect(s[0].pointCount == 64)
    #expect(abs(s[0].position.x - 0.04) < 1e-5)
}

/// When points captured up close (4.5mm pixels) cover the whole surface, the coarse splats from
/// afar are hidden and only fine splats remain.
@Test
func surfelMapHidesCoarseCellOnceRefined() {
    var map = makeMap()
    insertGrid(&map, xMax: 0.08, pixelSize: 0.035, frame: 0)
    insertGrid(&map, xMax: 0.08, pixelSize: 0.0045, frame: 1)
    let s = map.snapshot()
    #expect(s.count == 64)
    #expect(s.allSatisfy { $0.level == 0 && $0.cellMeters == 0.01 })
}

/// If only part was captured up close, the coarse splats are kept (no holes outside the fine area).
@Test
func surfelMapKeepsCoarseCellWhenPartlyRefined() {
    var map = makeMap()
    insertGrid(&map, xMax: 0.08, pixelSize: 0.035, frame: 0)
    insertGrid(&map, xMax: 0.04, pixelSize: 0.0045, frame: 1)
    let s = map.snapshot()
    #expect(s.filter { $0.level == 0 }.count == 32)
    #expect(s.filter { $0.level == 3 }.count == 1)
}

/// A pixel viewing the surface obliquely stretches across the surface, so the same pixel size falls
/// into a coarser level.
@Test
func surfelMapUsesCoarserCellForObliqueView() {
    var map = makeMap()
    let p = SIMD3<Float>(0.005, 0.005, 0.001)
    // Facing the surface: 2 × 4.5mm = 9mm → level 0 (1cm)
    map.insert(position: p, normal: SIMD3(0, 0, 1), color: .zero, pixelSize: 0.0045,
               cameraCenter: p + SIMD3(0, 0, 1), frame: 0)
    // ~11° to the surface (cos 0.2 → floor 0.3): 2 × 4.5mm / 0.3 = 3cm → level 2 (4cm)
    let q = SIMD3<Float>(0.045, 0.045, 0.001)
    map.insert(position: q, normal: SIMD3(0, 0, 1), color: .zero, pixelSize: 0.0045,
               cameraCenter: q + simd_normalize(SIMD3<Float>(1, 0, 0.2041)), frame: 0)
    #expect(Set(map.snapshot().map(\.level)) == [0, 2])
}

/// Coarse cells floating in front of a surface captured up close (1m depth, 1cm pixels) are
/// removed; coarse cells on the surface and cells finer than the observation are kept.
@Test
func surfelMapCarvesCoarseCellsInFrontOfObservedSurface() {
    var map = makeMap()
    let eye = SIMD3<Float>(0, 0, -5)
    func add(_ p: SIMD3<Float>, pixelSize: Float) {
        for _ in 0..<2 {
            map.insert(position: p, normal: SIMD3(0, 0, -1), color: .zero, pixelSize: pixelSize,
                       cameraCenter: eye, frame: 0)
        }
    }
    add(SIMD3(0.04, 0.04, 0.30), pixelSize: 0.035)   // 8cm cell, 70cm in front of the surface → removed
    add(SIMD3(0.20, 0.04, 1.00), pixelSize: 0.035)   // 8cm cell, on the surface → kept
    add(SIMD3(0.04, 0.20, 0.50), pixelSize: 0.009)   // 2cm cell = the observation's cell (2 × 1cm) → kept
    #expect(map.snapshot().count == 3)

    let w = 100, h = 100
    let identity: [Float] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    let removed = map.carve(depth: [Float](repeating: 1, count: w * h), width: w, height: h,
                            intrinsics: (fx: 100, fy: 100, cx: 50, cy: 50), viewmat: identity)
    #expect(removed == 1)
    let left = map.snapshot()
    #expect(left.count == 2)
    #expect(!left.contains { abs($0.position.z - 0.30) < 1e-4 })
}

/// Splats are drawn in order of distance from the eye, farthest first (back → front).
@Test
func surfelRendererOrdersBackToFront() {
    func surfel(_ z: Float) -> CaptureSurfelMap.Surfel {
        CaptureSurfelMap.Surfel(position: SIMD3(0, 0, z), normal: SIMD3(0, 0, 1), color: .zero,
                                cellMeters: 0.01, level: 0, pointCount: 1, viewCount: 1)
    }
    let surfels = [2, 9, 0.5, 4, 9.5, 0.01].map { surfel(Float($0)) }
    let order = CaptureSurfelRenderer.backToFrontOrder(surfels, eye: .zero)
    #expect(order == [4, 1, 3, 0, 2, 5])
}
