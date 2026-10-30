// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import simd

/// Detects that the person capturing has stopped moving. Standing still adds no new viewpoints, so
/// the capture screen asks them to keep moving.
///
/// Keeps a reference pose and the time it was set. A pose that moved or turned past the thresholds
/// from the reference becomes the new reference; the camera is still once the reference has stayed
/// for `stillSeconds`. Slow drift therefore still counts as still until it adds up to a threshold.
public struct CaptureStillnessDetector: Sendable {

    public struct Parameters: Sendable, Equatable {
        /// Translation from the reference pose that counts as moving (m).
        public var minTranslationMeters: Float
        /// Rotation of the view axis from the reference pose that counts as moving (rad).
        public var minRotationRadians: Float
        /// Time without moving before the camera counts as still (s).
        public var stillSeconds: TimeInterval

        public init(
            minTranslationMeters: Float = 0.03,
            minRotationRadians: Float = 0.052,    // ~3°
            stillSeconds: TimeInterval = 2.0
        ) {
            precondition(minTranslationMeters > 0)
            precondition(minRotationRadians > 0)
            precondition(stillSeconds > 0)
            self.minTranslationMeters = minTranslationMeters
            self.minRotationRadians = minRotationRadians
            self.stillSeconds = stillSeconds
        }
    }

    public let parameters: Parameters
    private var referenceCameraToWorld: simd_float4x4?
    private var referenceTimestamp: TimeInterval = 0

    public init(parameters: Parameters = .init()) {
        self.parameters = parameters
    }

    /// Forget the reference pose (e.g. while tracking is not reliable). The next update starts over.
    public mutating func reset() {
        referenceCameraToWorld = nil
    }

    /// Feed one camera pose. Returns whether the camera is still at this time.
    public mutating func update(cameraToWorld c2w: simd_float4x4, timestamp: TimeInterval) -> Bool {
        guard let ref = referenceCameraToWorld else {
            referenceCameraToWorld = c2w
            referenceTimestamp = timestamp
            return false
        }
        let d = SIMD3<Float>(c2w.columns.3.x - ref.columns.3.x,
                             c2w.columns.3.y - ref.columns.3.y,
                             c2w.columns.3.z - ref.columns.3.z)
        // Rotation approximated by the angle between view axes (column 2), as in the frame gate.
        let refZ = SIMD3<Float>(ref.columns.2.x, ref.columns.2.y, ref.columns.2.z)
        let curZ = SIMD3<Float>(c2w.columns.2.x, c2w.columns.2.y, c2w.columns.2.z)
        let angle = acos(max(-1, min(1, simd_dot(refZ, curZ))))
        let minT = parameters.minTranslationMeters
        if simd_length_squared(d) >= minT * minT || angle >= parameters.minRotationRadians {
            referenceCameraToWorld = c2w
            referenceTimestamp = timestamp
            return false
        }
        return timestamp - referenceTimestamp >= parameters.stillSeconds
    }
}
