// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Wide-angle loop closure with FlexGlue: the candidate pairs, the model file check and reading Core
// ML's padded output arrays. Model-independent.

import CoreML
import Foundation
import simd
import Testing
@testable import Capture

/// w2c viewmat of a camera at angleDeg on a circle of `radius`, looking at the origin.
private func orbitViewmat(angleDeg: Float, radius: Float) -> [Float] {
    let a = angleDeg * .pi / 180
    let c = SIMD3<Float>(radius * cos(a), 0, radius * sin(a))
    let z = simd_normalize(-c)
    let x = simd_normalize(simd_cross(SIMD3<Float>(0, 1, 0), z))
    let y = simd_cross(z, x)
    return [x.x, x.y, x.z, -simd_dot(x, c), y.x, y.y, y.z, -simd_dot(y, c), z.x, z.y, z.z, -simd_dot(z, c), 0, 0, 0, 1]
}

/// Two passes around a post (points on a small cylinder at the origin): the first from 0° to 39°,
/// the second from 120° to 159°. Only pairs across the passes are 45° or more apart.
@Test
func flexGlueCandidatesPairWideAngleViewsOfTheSamePoints() {
    let angles = (0..<40).map { Float($0) } + (0..<40).map { 120 + Float($0) }
    let vms = angles.map { orbitViewmat(angleDeg: $0, radius: 1.5) }
    var points: [SIMD3<Float>] = []
    for k in 0..<400 {
        let a = Float(k) * 0.37, h = Float(k % 20) / 20 - 0.5
        points.append(SIMD3(0.1 * cos(a), h, 0.1 * sin(a)))
    }
    let n = vms.count
    var p = CaptureBA.FlexGlueLoopParameters(directory: URL(fileURLWithPath: "/nonexistent"))
    p.maxPairs = 30
    let pairs = CaptureBA.flexGlueLoopCandidates(
        viewmats: vms, intrinsics: Array(repeating: (fx: 500, fy: 500, cx: 320, cy: 240), count: n),
        sizes: Array(repeating: (width: 640, height: 480), count: n), points: points, parameters: p)

    #expect(!pairs.isEmpty)
    #expect(pairs.count <= p.maxPairs)
    var used = [Int](repeating: 0, count: n)
    for (k, q) in pairs.enumerated() {
        #expect(q.b - q.a >= p.minGap)
        let fa = SIMD3(vms[q.a][8], vms[q.a][9], vms[q.a][10]), fb = SIMD3(vms[q.b][8], vms[q.b][9], vms[q.b][10])
        #expect(simd_dot(fa, fb) <= cos(p.minAngleDeg * .pi / 180) + 1e-5, "pair \(q) looks the same way")
        // From different passes: the first pass ends at frame 39
        #expect(q.a < 40 && q.b >= 40)
        used[q.a] += 1; used[q.b] += 1
        for r in pairs[..<k] {
            #expect(abs(r.a - q.a) >= p.spacing || abs(r.b - q.b) >= p.spacing, "pairs \(r) and \(q) are neighbors")
        }
    }
    #expect(used.max()! <= p.perFrame)
}

/// Views that see different points are not paired, however far apart their directions are.
@Test
func flexGlueCandidatesSkipViewsWithoutSharedPoints() {
    let vms = (0..<80).map { orbitViewmat(angleDeg: $0 < 40 ? 0 : 180, radius: 1.5) }
    // A point on each side of a wall at x = 0, each seen only by the cameras on its side
    let points = (0..<200).map { k in SIMD3<Float>(k % 2 == 0 ? 1.0 : -1.0, Float(k / 2) / 200 - 0.25, 0) }
    var p = CaptureBA.FlexGlueLoopParameters(directory: URL(fileURLWithPath: "/nonexistent"))
    p.maxDepth = 0.75   // each camera (1.5 m out) sees only the points on its own side
    let pairs = CaptureBA.flexGlueLoopCandidates(
        viewmats: vms, intrinsics: Array(repeating: (fx: 500, fy: 500, cx: 320, cy: 240), count: 80),
        sizes: Array(repeating: (width: 640, height: 480), count: 80), points: points, parameters: p)
    #expect(pairs.isEmpty)
}

@Test
func flexGlueModelsPresentNeedsEveryFile() throws {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("flexglue-models-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    #expect(!FlexGlueMatcher.modelsPresent(in: dir))
    for name in ["flexglue_descriptor.mlmodelc", "flexglue_transformer_a.mlpackage", "flexglue_transformer_b.mlmodelc"] {
        try FileManager.default.createDirectory(at: dir.appendingPathComponent(name), withIntermediateDirectories: true)
    }
    #expect(!FlexGlueMatcher.modelsPresent(in: dir))
    try Data([0]).write(to: dir.appendingPathComponent("flexglue_head.safetensors"))
    #expect(FlexGlueMatcher.modelsPresent(in: dir))
}

/// Core ML pads rows: a [2, 3] array laid out with a row stride of 4 reads back in row-major order.
@Test
func flexGlueReadsPaddedCoreMLArrays() throws {
    let storage = UnsafeMutablePointer<Float16>.allocate(capacity: 8)
    for i in 0..<8 { storage[i] = Float16(i) }
    let a = try MLMultiArray(dataPointer: storage, shape: [2, 3], dataType: .float16, strides: [4, 1],
                             deallocator: { $0.deallocate() })
    #expect(FlexGlueMatcher.contiguous(a) == [0, 1, 2, 4, 5, 6])
    let packed = try MLMultiArray(shape: [2, 3], dataType: .float32)
    for i in 0..<6 { packed[i] = NSNumber(value: Float(i) * 0.5) }
    #expect(FlexGlueMatcher.contiguous(packed) == [0, 0.5, 1, 1.5, 2, 2.5])
}
