// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import simd

/// How a capture decides which camera frames to keep.
public enum CaptureFrameSampling: Sendable, Equatable {
    /// Keep a frame only when the camera has moved or turned enough since the last kept one
    /// (`CaptureKeyframeGate`). Standing still adds nothing. For static scenes.
    case keyframes
    /// Keep a frame every `interval` seconds whether the camera moved or not, for up to
    /// `maxDuration` seconds after the first kept frame. For scenes with a moving subject, where
    /// the timeline matters as much as the viewpoints. Each kept frame still records whether it
    /// would have passed the keyframe gate (`CapturedFrame.isKeyframe`), so the static scene can
    /// be built from those alone.
    case timed(interval: TimeInterval, maxDuration: TimeInterval)
}

/// What kind of capture a scene came from (saved with the dataset).
public enum CaptureKind: String, Codable, Sendable {
    /// Keyframes only (`CaptureFrameSampling.keyframes`).
    case keyframes
    /// Frames at a fixed interval (`CaptureFrameSampling.timed`); `CapturedFrame.isKeyframe` marks
    /// the ones the keyframe gate would also have kept.
    case timed
}

/// The keyframe test of `ARKitCaptureSession`: a candidate is a new viewpoint if it moved at least
/// `minMotion` metres or turned its view axis at least `minRotation` radians from the last
/// keyframe, and it is redundant if some existing keyframe is closer than both.
struct CaptureKeyframeGate {
    var minMotion: Float
    var minRotation: Float

    /// Whether `cameraToWorld` moved or turned enough from `last` (always true without a `last`).
    func isNewViewpoint(_ cameraToWorld: simd_float4x4, last: simd_float4x4?) -> Bool {
        guard let last else { return true }
        let d = Self.position(cameraToWorld) - Self.position(last)
        if simd_length_squared(d) >= minMotion * minMotion { return true }
        return Self.viewAngle(cameraToWorld, last) >= minRotation
    }

    /// Whether some pose in `existing` is closer than `minMotion` and turned less than
    /// `minRotation`. A pose exactly at the thresholds is not redundant, so the previous keyframe,
    /// which `isNewViewpoint` has already cleared, never counts.
    func isRedundant<S: Sequence>(_ cameraToWorld: simd_float4x4, existing: S) -> Bool
    where S.Element == simd_float4x4 {
        let p = Self.position(cameraToWorld)
        for e in existing {
            guard simd_length_squared(p - Self.position(e)) < minMotion * minMotion else { continue }
            if Self.viewAngle(cameraToWorld, e) < minRotation { return true }
        }
        return false
    }

    static func position(_ m: simd_float4x4) -> SIMD3<Float> {
        SIMD3<Float>(m.columns.3.x, m.columns.3.y, m.columns.3.z)
    }

    /// Angle between the view axes (column 2) of two camera-to-world poses.
    static func viewAngle(_ a: simd_float4x4, _ b: simd_float4x4) -> Float {
        let za = SIMD3<Float>(a.columns.2.x, a.columns.2.y, a.columns.2.z)
        let zb = SIMD3<Float>(b.columns.2.x, b.columns.2.y, b.columns.2.z)
        return acos(Swift.max(-1, Swift.min(1, simd_dot(za, zb))))
    }
}

/// The clock of `CaptureFrameSampling.timed`: which frame timestamps to keep.
struct CaptureTimedSampler {
    let interval: TimeInterval
    let maxDuration: TimeInterval
    private(set) var firstKept: TimeInterval?
    private(set) var lastKept: TimeInterval?

    init(interval: TimeInterval, maxDuration: TimeInterval) {
        precondition(interval > 0 && maxDuration > 0)
        self.interval = interval
        self.maxDuration = maxDuration
    }

    /// Whether a frame at `timestamp` is due. Frames keep arriving at about 60 Hz, so a kept frame
    /// is up to one frame period late; the next one is due `interval` after the previous due time,
    /// not after the late frame, so the average rate stays at 1/interval. A frame more than one
    /// interval late (a gap, such as rejected frames) restarts the schedule at itself.
    func isDue(_ timestamp: TimeInterval) -> Bool {
        if isFinished(at: timestamp) { return false }
        guard let lastKept else { return true }
        return timestamp - lastKept >= interval - Self.tolerance
    }

    /// Whether `maxDuration` has passed since the first kept frame.
    func isFinished(at timestamp: TimeInterval) -> Bool {
        guard let firstKept else { return false }
        return timestamp - firstKept > maxDuration
    }

    /// Records that the frame at `timestamp` was kept.
    mutating func recordKept(_ timestamp: TimeInterval) {
        if firstKept == nil { firstKept = timestamp }
        if let lastKept, timestamp - lastKept < 2 * interval {
            // On schedule: advance by one interval so frame lateness does not accumulate.
            self.lastKept = lastKept + interval
        } else {
            self.lastKept = timestamp
        }
    }

    /// Slack for timestamps that land a hair before the due time (ARKit timestamps jitter).
    static let tolerance: TimeInterval = 0.002
}
