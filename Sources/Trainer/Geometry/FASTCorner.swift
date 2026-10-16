// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// FAST-9 corner detector (first stage of the ORB pipeline in PocketGS Appendix A.b). Score map on
// the GPU, NMS on the host.

import Foundation
import MLX
import MLXFast

public enum FASTCorner {

    private static let kernel = SlangCompiledKernel.required("fast_corner")

    /// FAST-9 score map (score > 0 is a corner).
    /// - Parameter threshold: Intensity-difference threshold (normalized 0..1). Typically 0.02-0.1.
    /// - Returns: `[H, W]` int32 (sum of q8 fixed-point values)
    public static func score(luma: MLXArray, threshold: Float = 0.05) -> MLXArray {
        precondition(luma.shape.count == 2, "luma must be [H, W]")
        let h = luma.shape[0]
        let w = luma.shape[1]
        let thrQ8 = Int32((threshold * 256.0).rounded())
        let dims = MLXArray([Int32(w), Int32(h), thrQ8], [3])
        let lumaFlat = luma.reshaped([h * w])
        let outputs = kernel(
            [lumaFlat, dims],
            grid: (((w + 15) / 16) * 16, ((h + 15) / 16) * 16, 1),
            threadGroup: (16, 16, 1),
            outputShapes: [[h * w]],
            outputDTypes: [.int32]
        )
        return outputs[0].reshaped([h, w])
    }

    /// 3×3 NMS + keep the top `maxKeypoints` by score (host implementation).
    public static func extractKeypoints(
        scoreMap: MLXArray,
        maxKeypoints: Int = 1500
    ) -> (coords: [(x: Int, y: Int)], scores: [Int32]) {
        let h = scoreMap.shape[0]
        let w = scoreMap.shape[1]
        let s = scoreMap.asArray(Int32.self)
        // Keep if no 8-neighbor has a larger score (equal-valued neighbors both survive).
        var keep: [(x: Int, y: Int, score: Int32)] = []
        keep.reserveCapacity(min(maxKeypoints * 2, h * w / 4))
        for y in 1..<(h - 1) {
            for x in 1..<(w - 1) {
                let sv = s[y * w + x]
                if sv <= 0 { continue }
                var isLocalMax = true
                outer: for dy in -1...1 {
                    for dx in -1...1 where !(dx == 0 && dy == 0) {
                        if s[(y + dy) * w + (x + dx)] > sv {
                            isLocalMax = false
                            break outer
                        }
                    }
                }
                if isLocalMax {
                    keep.append((x: x, y: y, score: sv))
                }
            }
        }
        keep.sort { $0.score > $1.score }
        let n = min(keep.count, maxKeypoints)
        return (
            coords: keep.prefix(n).map { (x: $0.x, y: $0.y) },
            scores: keep.prefix(n).map { $0.score }
        )
    }
}
