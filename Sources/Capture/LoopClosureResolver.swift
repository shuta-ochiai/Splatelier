// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Pose resolution after loop closure (pure functions independent of ARKit, testable on macOS). One
// ARAnchor is placed per frame, and in stopCapture viewmats are rebuilt from the anchors' final
// transforms, because loop closure only updates ARAnchor.transform, not ARFrame.camera.transform.

import Foundation
import simd
import MLX
import Trainer

public enum LoopClosureResolver {

    /// Input for one frame accumulated during capture. The viewmat is built from the anchor at
    /// resolve time.
    public struct Record {
        public let anchorID: UUID
        public let image: MLXArray
        public let intrinsics: Intrinsics
        public let width: Int
        public let height: Int
        public let timestamp: TimeInterval
        public let featureObservations: [ARKitFeatureObservation]

        public init(
            anchorID: UUID, image: MLXArray,
            intrinsics: Intrinsics,
            width: Int, height: Int, timestamp: TimeInterval,
            featureObservations: [ARKitFeatureObservation] = []
        ) {
            self.anchorID = anchorID
            self.image = image
            self.intrinsics = intrinsics
            self.width = width
            self.height = height
            self.timestamp = timestamp
            self.featureObservations = featureObservations
        }
    }

    /// Feature point held in anchor-local coordinates. Mapped back to post-loop-closure world
    /// coordinates in stopCapture. Re-observing the same feature keeps it bound to the first anchor
    /// without updating.
    public struct AnchoredPoint {
        public let anchorID: UUID
        public let localPos: SIMD3<Float>
        public init(anchorID: UUID, localPos: SIMD3<Float>) {
            self.anchorID = anchorID
            self.localPos = localPos
        }
    }

    /// Maps back to world coordinates with the latest anchor transforms. Points whose anchor is
    /// gone are dropped.
    public static func projectAnchoredPoints(
        _ points: [AnchoredPoint],
        anchorTransforms: [UUID: simd_float4x4]
    ) -> [SIMD3<Float>] {
        points.compactMap { p in
            guard let c2w = anchorTransforms[p.anchorID] else { return nil }
            let local4 = SIMD4<Float>(p.localPos.x, p.localPos.y, p.localPos.z, 1)
            let world4 = c2w * local4
            return SIMD3<Float>(world4.x, world4.y, world4.z)
        }
    }

    /// Builds the `CapturedFrame` list from the latest anchor transforms (c2w in the ARKit
    /// convention). Frames whose anchor is gone have no settled pose and are not output.
    public static func resolve(
        records: [Record],
        anchorTransforms: [UUID: simd_float4x4]
    ) -> [CapturedFrame] {
        var out: [CapturedFrame] = []
        out.reserveCapacity(records.count)
        for r in records {
            guard let c2w = anchorTransforms[r.anchorID] else { continue }
            let viewmat = PoseConversion.viewmat(fromCameraToWorld: c2w)
            out.append(CapturedFrame(
                image: r.image,
                viewmat: viewmat,
                intrinsics: r.intrinsics,
                width: r.width, height: r.height,
                timestamp: r.timestamp,
                anchorIdentifier: r.anchorID,
                featureObservations: r.featureObservations
            ))
        }
        return out
    }
}
