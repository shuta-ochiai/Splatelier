// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Unprojects ARFrame.sceneDepth (LiDAR) into a point cloud in ARKit world coordinates. depthMap is
// DepthFloat32 with distance along the optical axis (m, not ray length). confidence is
// ARConfidenceLevel raw. The ARKit camera is +Y up / −Z forward, so P_cam = (x, −y, −z) (following
// the WWDC20 ARKit 4 sample). Output is in the same ARKit world coordinates as sparsePoints (the
// flip to OpenCV happens on the viewmat side).

#if canImport(ARKit) && os(iOS)

import Foundation
import CoreVideo
import simd
import ARKit

public enum DepthUnprojection {

    public struct Params: Sendable {
        public let stride: Int
        public let minConfidenceRaw: UInt8        // 0=low / 1=medium / 2=high
        public let minDepth: Float                // m. Lower trust bound for close-range LiDAR
        public let maxDepth: Float                // m. Nominal LiDAR range

        public init(
            stride: Int = 8,
            minConfidenceRaw: UInt8 = 1,
            minDepth: Float = 0.2,
            maxDepth: Float = 5.0
        ) {
            precondition(stride > 0)
            precondition(minDepth > 0)
            precondition(maxDepth > minDepth)
            self.stride = stride
            self.minConfidenceRaw = minConfidenceRaw
            self.minDepth = minDepth
            self.maxDepth = maxDepth
        }
    }

    /// Turns `ARFrame.sceneDepth` into points in ARKit world coordinates.
    public static func unprojectToWorld(
        depthMap: CVPixelBuffer,
        confidenceMap: CVPixelBuffer?,
        cameraIntrinsics: matrix_float3x3,
        imageResolution: CGSize,
        cameraToWorld c2w: simd_float4x4,
        params: Params = .init()
    ) -> [SIMD3<Float>] {
        precondition(
            CVPixelBufferGetPixelFormatType(depthMap) == kCVPixelFormatType_DepthFloat32,
            "ARKit depthMap format must be kCVPixelFormatType_DepthFloat32"
        )
        let depthW = CVPixelBufferGetWidth(depthMap)
        let depthH = CVPixelBufferGetHeight(depthMap)

        // Intrinsics are relative to the capturedImage resolution, so rescale them to the depthMap
        // resolution.
        let K = PoseConversion.scaledIntrinsics(
            fx: cameraIntrinsics[0, 0], fy: cameraIntrinsics[1, 1],
            cx: cameraIntrinsics[2, 0], cy: cameraIntrinsics[2, 1],
            originalWidth: Int(imageResolution.width),
            originalHeight: Int(imageResolution.height),
            scaledWidth: depthW, scaledHeight: depthH
        )

        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
        guard let depthBaseRaw = CVPixelBufferGetBaseAddress(depthMap) else { return [] }
        let depthBase = depthBaseRaw.assumingMemoryBound(to: Float32.self)
        let depthRowFloats = CVPixelBufferGetBytesPerRow(depthMap) / MemoryLayout<Float32>.stride

        var confBase: UnsafePointer<UInt8>? = nil
        var confRowBytes: Int = 0
        if let conf = confidenceMap {
            CVPixelBufferLockBaseAddress(conf, .readOnly)
            if let confBaseRaw = CVPixelBufferGetBaseAddress(conf) {
                confBase = UnsafePointer(confBaseRaw.assumingMemoryBound(to: UInt8.self))
                confRowBytes = CVPixelBufferGetBytesPerRow(conf)
            }
        }
        defer {
            if let conf = confidenceMap {
                CVPixelBufferUnlockBaseAddress(conf, .readOnly)
            }
        }

        let stride = Swift.max(1, params.stride)
        let nCols = (depthW + stride - 1) / stride
        let nRows = (depthH + stride - 1) / stride
        var out: [SIMD3<Float>] = []
        out.reserveCapacity(nCols * nRows)

        var y = 0
        while y < depthH {
            var x = 0
            while x < depthW {
                let z = depthBase[y * depthRowFloats + x]
                if z >= params.minDepth, z <= params.maxDepth {
                    var keep = true
                    if let cb = confBase {
                        let c = cb[y * confRowBytes + x]
                        if c < params.minConfidenceRaw { keep = false }
                    }
                    if keep {
                        // pixel center +0.5
                        let xp = (Float(x) + 0.5 - K.cx) / K.fx * z
                        let yp = (Float(y) + 0.5 - K.cy) / K.fy * z
                        // Image y points down and the ARKit camera is −Z forward, so negate y/z.
                        let pCam = SIMD4<Float>(xp, -yp, -z, 1)
                        let pWorld = c2w * pCam
                        out.append(SIMD3<Float>(pWorld.x, pWorld.y, pWorld.z))
                    }
                }
                x += stride
            }
            y += stride
        }
        return out
    }
}

#endif
