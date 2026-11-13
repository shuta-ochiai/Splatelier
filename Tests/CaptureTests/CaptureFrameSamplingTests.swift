// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import simd
import Testing
@testable import Capture

/// Camera-to-world pose at `position`, turned `yaw` radians about the vertical axis.
private func pose(_ position: SIMD3<Float>, yaw: Float = 0) -> simd_float4x4 {
    let c = cos(yaw), s = sin(yaw)
    return simd_float4x4(columns: (
        SIMD4<Float>(c, 0, -s, 0),
        SIMD4<Float>(0, 1, 0, 0),
        SIMD4<Float>(s, 0, c, 0),
        SIMD4<Float>(position.x, position.y, position.z, 1)))
}

@Test
func keyframeGateNeedsMotionOrRotation() {
    let gate = CaptureKeyframeGate(minMotion: 0.1, minRotation: 0.1)
    let last = pose(.zero)
    #expect(gate.isNewViewpoint(last, last: nil))
    #expect(!gate.isNewViewpoint(pose(SIMD3(0.05, 0, 0)), last: last))
    #expect(gate.isNewViewpoint(pose(SIMD3(0.15, 0, 0)), last: last))
    #expect(!gate.isNewViewpoint(pose(.zero, yaw: 0.05), last: last))
    #expect(gate.isNewViewpoint(pose(.zero, yaw: 0.15), last: last))
}

@Test
func keyframeGateFindsRedundantViewpoints() {
    let gate = CaptureKeyframeGate(minMotion: 0.1, minRotation: 0.1)
    let existing = [pose(.zero), pose(SIMD3(1, 0, 0))]
    // Back near the first keyframe, facing the same way: redundant.
    #expect(gate.isRedundant(pose(SIMD3(0.02, 0, 0)), existing: existing))
    // Same place but turned: a new view.
    #expect(!gate.isRedundant(pose(SIMD3(0.02, 0, 0), yaw: 0.3), existing: existing))
    // Away from both.
    #expect(!gate.isRedundant(pose(SIMD3(0.5, 0, 0)), existing: existing))
}

/// The timestamps the sampler keeps from a 60 Hz stream between `start` and `end`.
private func keptTimestamps(_ sampler: inout CaptureTimedSampler, from start: TimeInterval,
                            to end: TimeInterval, skipping skip: (TimeInterval) -> Bool = { _ in false })
    -> [TimeInterval] {
    var kept: [TimeInterval] = []
    var t = start
    while t <= end {
        if !skip(t), sampler.isDue(t) {
            sampler.recordKept(t)
            kept.append(t)
        }
        t += 1.0 / 60.0
    }
    return kept
}

@Test
func timedSamplerKeepsTheRateWithoutDrift() {
    var sampler = CaptureTimedSampler(interval: 0.1, maxDuration: 60)
    let kept = keptTimestamps(&sampler, from: 100, to: 130)
    // 0.1 s at 60 Hz is exactly 6 frame periods; 30 s gives 300 frames plus the first.
    #expect(kept.count == 301)
    let gaps = zip(kept.dropFirst(), kept).map { $0 - $1 }
    let allOK1 = gaps.allSatisfy { abs($0 - 0.1) < 0.02 }
    #expect(allOK1)
}

@Test
func timedSamplerKeepsTheAverageRateAtAnOddFrameRate() {
    // 0.15 s is 4.5 frame periods at 30 Hz: the schedule must not round every interval up to 5
    // periods (6 frames/s instead of 6.67).
    var sampler = CaptureTimedSampler(interval: 0.15, maxDuration: 60)
    var kept = 0
    var t: TimeInterval = 0
    while t < 30 {
        if sampler.isDue(t) { sampler.recordKept(t); kept += 1 }
        t += 1.0 / 30.0
    }
    #expect(abs(Double(kept) - 30 / 0.15) <= 2)
}

@Test
func timedSamplerStopsAfterItsDuration() {
    var sampler = CaptureTimedSampler(interval: 0.1, maxDuration: 2)
    let kept = keptTimestamps(&sampler, from: 0, to: 5)
    #expect(kept.last! <= 2)
    #expect(sampler.isFinished(at: 2.1))
    #expect(!sampler.isDue(2.1))
}

@Test
func timedSamplerRestartsAfterAGap() {
    // Frames between 1 s and 2 s are rejected (blur): the next good frame is kept at once, and the
    // schedule continues from it instead of catching up with a burst.
    var sampler = CaptureTimedSampler(interval: 0.1, maxDuration: 60)
    let kept = keptTimestamps(&sampler, from: 0, to: 3, skipping: { $0 > 1.0 && $0 < 2.0 })
    let afterGap = kept.filter { $0 >= 2.0 }
    #expect(afterGap.first! < 2.02)
    let gaps = zip(afterGap.dropFirst(), afterGap).map { $0 - $1 }
    let allOK2 = gaps.allSatisfy { $0 > 0.08 }
    #expect(allOK2)
}

private func frame(_ index: Int, keyframe: Bool) -> CapturedFrame {
    CapturedFrame(
        image: MLXArray([UInt8](repeating: UInt8(index), count: 2 * 2 * 3), [2, 2, 3]),
        viewmat: [1, 0, 0, Float(index), 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        intrinsics: (fx: 2, fy: 2, cx: 1, cy: 1), width: 2, height: 2,
        timestamp: TimeInterval(index) * 0.1, isKeyframe: keyframe)
}

@Test
func keyframeSceneKeepsOnlyKeyframes() {
    let scene = CapturedScene(
        frames: [frame(0, keyframe: true), frame(1, keyframe: false), frame(2, keyframe: false),
                 frame(3, keyframe: true)],
        sparsePoints: [], kind: .timed)
    let keys = scene.keyframeScene
    #expect(keys.frames.map { Int(($0.timestamp * 10).rounded()) } == [0, 3])
    #expect(keys.kind == .timed)
    let fromGeometry = CaptureTraining.staticSceneFrames(scene, log: { _ in })
    #expect(fromGeometry.frames.count == 2)
    // A keyframe capture is left as it is.
    let plain = CapturedScene(frames: [frame(0, keyframe: true)], sparsePoints: [])
    #expect(CaptureTraining.staticSceneFrames(plain, log: { _ in }).frames.count == 1)
}

@Test
func timedCaptureRoundTripsThroughTheDataset() throws {
    let scene = CapturedScene(
        frames: [frame(0, keyframe: true), frame(1, keyframe: false), frame(2, keyframe: true)],
        sparsePoints: [SIMD3<Float>(1, 2, 3)], kind: .timed)
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("timed_rt_\(UUID().uuidString)")
        .appendingPathComponent("capture.valset")
    defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }

    try CaptureDatasetStore.save(scene, to: dir)
    let loaded = try CaptureDatasetStore.load(from: dir)
    #expect(loaded.kind == .timed)
    #expect(loaded.frames.map(\.isKeyframe) == [true, false, true])

    // A keyframe capture writes neither field, so the manifest stays what older builds read.
    let plain = CapturedScene(frames: [frame(0, keyframe: true)], sparsePoints: [])
    try CaptureDatasetStore.save(plain, to: dir)
    let manifest = try String(contentsOf: dir.appendingPathComponent("manifest.json"), encoding: .utf8)
    #expect(!manifest.contains("isKeyframe") && !manifest.contains("\"kind\""))
    let plainLoaded = try CaptureDatasetStore.load(from: dir)
    #expect(plainLoaded.kind == .keyframes)
    let allOK3 = plainLoaded.frames.allSatisfy(\.isKeyframe)
    #expect(allOK3)
}
