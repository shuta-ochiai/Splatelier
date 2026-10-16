// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// LoopClosureResolver invariants: the viewmat reflects the anchor's latest transform (ARKit's later
// updates), frames whose anchor disappeared are dropped, and it is a pure function.
// ARKit-independent, so it runs on macOS.

import Foundation
import MLX
import simd
import Testing
@testable import Capture

/// Record with a dummy image / intrinsics.
private func makeRecord(
    id: UUID = UUID(), timestamp: TimeInterval = 0
) -> LoopClosureResolver.Record {
    LoopClosureResolver.Record(
        anchorID: id,
        image: MLXArray([Float](repeating: 0.5, count: 2 * 2 * 3), [2, 2, 3]),
        intrinsics: (fx: 100, fy: 100, cx: 1, cy: 1),
        width: 2, height: 2, timestamp: timestamp
    )
}

/// When the anchor is updated after capture (loop closure), the viewmat is built from the latest
/// transform.
@Test
func loopClosureResolverUsesLatestAnchorTransform() {
    let anchorID = UUID()
    let record = makeRecord(id: anchorID)

    // Latest transform after loop closure: camera center (10, 0, 0).
    var finalC2W = matrix_identity_float4x4
    finalC2W.columns.3 = SIMD4<Float>(10, 0, 0, 1)
    let resolved = LoopClosureResolver.resolve(
        records: [record],
        anchorTransforms: [anchorID: finalC2W]
    )
    #expect(resolved.count == 1)
    // The w2c translation is -10.
    let v = resolved[0].viewmat
    #expect(abs(v[3] - (-10)) < 1e-5, "expected w2c t.x = -10, got \(v[3])")
    #expect(abs(v[7]) < 1e-5)
    #expect(abs(v[11]) < 1e-5)
    #expect(resolved[0].anchorIdentifier == anchorID)
}

/// Records whose anchor was removed (didRemove) are skipped.
@Test
func loopClosureResolverDropsRecordsWithoutAnchor() {
    let keepID = UUID()
    let dropID = UUID()
    let r1 = makeRecord(id: keepID, timestamp: 1.0)
    let r2 = makeRecord(id: dropID, timestamp: 2.0)
    // Only keepID exists in the dictionary
    let resolved = LoopClosureResolver.resolve(
        records: [r1, r2],
        anchorTransforms: [keepID: matrix_identity_float4x4]
    )
    #expect(resolved.count == 1)
    #expect(resolved[0].anchorIdentifier == keepID)
    #expect(resolved[0].timestamp == 1.0)
}

/// Pure function: same input, same output.
@Test
func loopClosureResolverDeterministic() {
    let id1 = UUID()
    let id2 = UUID()
    let records = [makeRecord(id: id1), makeRecord(id: id2)]
    var c2w1 = matrix_identity_float4x4; c2w1.columns.3 = SIMD4<Float>(1, 0, 0, 1)
    var c2w2 = matrix_identity_float4x4; c2w2.columns.3 = SIMD4<Float>(0, 2, 0, 1)
    let dict: [UUID: simd_float4x4] = [id1: c2w1, id2: c2w2]

    let a = LoopClosureResolver.resolve(records: records, anchorTransforms: dict)
    let b = LoopClosureResolver.resolve(records: records, anchorTransforms: dict)
    #expect(a.count == 2 && b.count == 2)
    for (x, y) in zip(a, b) {
        for i in 0..<16 {
            #expect(x.viewmat[i] == y.viewmat[i])
        }
        #expect(x.anchorIdentifier == y.anchorIdentifier)
    }
}

/// Two records sharing one anchor both get the same viewmat (normally 1:1, but checked
/// defensively).
@Test
func loopClosureResolverHandlesAnchorSharedByMultipleRecords() {
    let id = UUID()
    let r1 = makeRecord(id: id, timestamp: 1.0)
    let r2 = makeRecord(id: id, timestamp: 2.0)
    var c2w = matrix_identity_float4x4
    c2w.columns.3 = SIMD4<Float>(5, 0, 0, 1)
    let resolved = LoopClosureResolver.resolve(
        records: [r1, r2], anchorTransforms: [id: c2w]
    )
    #expect(resolved.count == 2)
    #expect(abs(resolved[0].viewmat[3] - (-5)) < 1e-5)
    #expect(abs(resolved[1].viewmat[3] - (-5)) < 1e-5)
    #expect(resolved[0].timestamp == 1.0)
    #expect(resolved[1].timestamp == 2.0)
}

/// Empty input.
@Test
func loopClosureResolverEmpty() {
    let resolved = LoopClosureResolver.resolve(
        records: [], anchorTransforms: [:]
    )
    #expect(resolved.isEmpty)
}

// MARK: - projectAnchoredPoints

/// With an identity anchor, local coordinates become world coordinates as is.
@Test
func projectAnchoredPointsIdentity() {
    let id = UUID()
    let local = SIMD3<Float>(1, 2, 3)
    let world = LoopClosureResolver.projectAnchoredPoints(
        [LoopClosureResolver.AnchoredPoint(anchorID: id, localPos: local)],
        anchorTransforms: [id: matrix_identity_float4x4]
    )
    #expect(world.count == 1)
    #expect(abs(world[0].x - 1) < 1e-6)
    #expect(abs(world[0].y - 2) < 1e-6)
    #expect(abs(world[0].z - 3) < 1e-6)
}

/// When loop closure corrects the anchor by 10 m along +x, the point appears at the corrected world
/// position (11, 2, 3).
@Test
func projectAnchoredPointsTranslated() {
    let id = UUID()
    var t = matrix_identity_float4x4
    t.columns.3 = SIMD4<Float>(10, 0, 0, 1)
    let world = LoopClosureResolver.projectAnchoredPoints(
        [LoopClosureResolver.AnchoredPoint(anchorID: id, localPos: SIMD3<Float>(1, 2, 3))],
        anchorTransforms: [id: t]
    )
    #expect(world.count == 1)
    #expect(abs(world[0].x - 11) < 1e-5)
    #expect(abs(world[0].y - 2) < 1e-5)
    #expect(abs(world[0].z - 3) < 1e-5)
}

/// Points whose anchor is not in the dictionary (dropped by ARKit) are skipped.
@Test
func projectAnchoredPointsDropsOrphanedAnchors() {
    let keepID = UUID()
    let dropID = UUID()
    let world = LoopClosureResolver.projectAnchoredPoints(
        [
            LoopClosureResolver.AnchoredPoint(anchorID: keepID, localPos: SIMD3<Float>(0, 0, 1)),
            LoopClosureResolver.AnchoredPoint(anchorID: dropID, localPos: SIMD3<Float>(0, 0, 2)),
        ],
        anchorTransforms: [keepID: matrix_identity_float4x4]
    )
    #expect(world.count == 1)
    #expect(abs(world[0].z - 1) < 1e-6)
}

/// Even when multiple points share one anchor (typical for ARKit feature points), each local
/// coordinate becomes a separate world point.
@Test
func projectAnchoredPointsSharedAnchor() {
    let id = UUID()
    var anchor = matrix_identity_float4x4
    anchor.columns.3 = SIMD4<Float>(5, 0, 0, 1)
    let world = LoopClosureResolver.projectAnchoredPoints(
        [
            LoopClosureResolver.AnchoredPoint(anchorID: id, localPos: SIMD3<Float>(0, 0, 0)),
            LoopClosureResolver.AnchoredPoint(anchorID: id, localPos: SIMD3<Float>(1, 0, 0)),
            LoopClosureResolver.AnchoredPoint(anchorID: id, localPos: SIMD3<Float>(0, 0, 1)),
        ],
        anchorTransforms: [id: anchor]
    )
    #expect(world.count == 3)
    #expect(abs(world[0].x - 5) < 1e-6)
    #expect(abs(world[1].x - 6) < 1e-6)
    #expect(abs(world[2].x - 5) < 1e-6 && abs(world[2].z - 1) < 1e-6)
}
