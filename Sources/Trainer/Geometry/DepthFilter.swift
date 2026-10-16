// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Depth filter (PocketGS Operator G 3b-4). Builds a confidence-filtered depth map from an
// aggregated cost volume.

import Foundation
import MLX
import MLXFast

public enum DepthFilter {

    private static let kernel = SlangCompiledKernel.required("depth_filter")

    /// `depthMap` is -1 at pixels with `confidence < threshold` (valid depths are always positive).
    public struct Result {
        public let depthMap: MLXArray     // [H, W] float32
        public let confidence: MLXArray   // [H, W] float32
    }

    /// WTA + peak-ratio confidence `1 - best/second` (PKRN from Hu & Mordohai 2012 TPAMI). The 0.4
    /// threshold is PocketGS Sec. III-A 3's value (the paper gives no formula).
    /// - Parameters:
    ///   - aggregatedCost: `[H, W, K]` int32 (output of `SGMAggregation.aggregate`)
    ///   - excludeRadius: Number of bins around the winner excluded from the second-peak search.
    ///     Without
    ///     it, the shallow valley SGM forms around the winner collapses confidence to nearly 0.
    public static func filter(
        aggregatedCost: MLXArray, depths: [Float], threshold: Float = 0.4,
        excludeRadius: Int = 2
    ) -> Result {
        precondition(aggregatedCost.shape.count == 3, "aggregatedCost must be [H, W, K]")
        precondition(!depths.isEmpty)
        precondition(excludeRadius >= 0)
        let h = aggregatedCost.shape[0]
        let w = aggregatedCost.shape[1]
        let k = aggregatedCost.shape[2]
        precondition(k == depths.count, "depths.count must equal K")

        let dims = MLXArray([Int32(w), Int32(h), Int32(k)], [3])
        let depthsArr = MLXArray(depths, [k])
        let threshBuf = MLXArray([threshold], [1])
        let excludeRadiusBuf = MLXArray([Int32(excludeRadius)], [1])

        let outputs = kernel(
            [
                aggregatedCost.reshaped([h * w * k]),
                depthsArr, dims, threshBuf, excludeRadiusBuf,
            ],
            grid: (w, h, 1),
            threadGroup: (16, 16, 1),
            outputShapes: [[h * w], [h * w]],
            outputDTypes: [.float32, .float32]
        )
        return Result(
            depthMap: outputs[0].reshaped([h, w]),
            confidence: outputs[1].reshaped([h, w])
        )
    }
}
