// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// SGM aggregation (PocketGS Operator G 3b-3). Dispatches the 1-path kernel in 8 directions and
// sums. The default is the SIMD-group parallel `sgm_aggregate_simd` (modeled on libSGM's parallel
// path aggregation; 1 scanline = 32 lanes, with depth indices distributed across lanes). The
// sequential `sgm_aggregate` is the exactly matching reference.

import Foundation
import MLX
import MLXFast

public enum SGMAggregation {

    private static let scalarKernel = SlangCompiledKernel.required("sgm_aggregate")

    private static let simdKernel = SlangCompiledKernel.required("sgm_aggregate_simd")

    /// Kernel implementation. Both give exactly the same result (verified by tests).
    public enum Implementation: Sendable {
        /// One scanline per SIMD group (32 lanes), depth indices distributed across lanes
        /// (default).
        case simd
        /// Sequential version, 1 thread per scanline (reference for the tests).
        case scalar
    }

    /// Upper bound on K the kernel allocates on the stack. Must match `SGM_K_MAX` in
    /// `slang/sgm_aggregate.slang`.
    public static let kMax: Int = 64

    /// SGM penalties (Hirschmüller 2008 TPAMI §II-B; PocketGS does not disclose the values). P2 is
    /// about twice the max cost 25 (same ratio as libSGM's defaults P1=10 / P2=120 @ max ~62). Too
    /// small and smoothing has no effect.
    public struct Penalties: Sendable, Equatable {
        public var p1: Int32
        public var p2: Int32
        public init(p1: Int32 = 2, p2: Int32 = 48) {
            precondition(p2 >= p1, "P2 must be >= P1 (Hirschmüller 2005)")
            self.p1 = p1
            self.p2 = p2
        }
    }

    /// Single-direction SGM aggregation (public for tests). Input and output are `[H, W, K]` int32.
    /// - Parameter direction: 0=L→R, 1=R→L, 2=T→B, 3=B→T, 4=TL→BR, 5=BR→TL, 6=TR→BL, 7=BL→TR
    public static func aggregateSinglePath(
        costVolume: MLXArray, direction: Int, penalties: Penalties,
        implementation: Implementation = .simd
    ) -> MLXArray {
        precondition(costVolume.shape.count == 3, "costVolume must be [H, W, K]")
        precondition(0 <= direction && direction < 8, "direction must be 0..7")
        let h = costVolume.shape[0]
        let w = costVolume.shape[1]
        let k = costVolume.shape[2]
        precondition(k <= kMax, "K=\(k) exceeds SGM_K_MAX=\(kMax)")

        let dims = MLXArray(
            [Int32(w), Int32(h), Int32(k), penalties.p1, penalties.p2, Int32(direction)],
            [6]
        )
        // 1 thread = 1 scanline: H rows horizontally, W columns vertically, W+H-1 lines diagonally.
        let gridSize: Int
        switch direction {
        case 0, 1: gridSize = h
        case 2, 3: gridSize = w
        default:   gridSize = w + h - 1
        }
        // In the simd version 1 scanline = 32 threads (= 1 SIMD group). threadgroup 64 = 2
        // scanlines.
        let kernel = implementation == .simd ? simdKernel : scalarKernel
        let threads = implementation == .simd ? gridSize * 32 : gridSize
        let outputs = kernel(
            [costVolume.reshaped([h * w * k]), dims],
            grid: (threads, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[h * w * k]],
            outputDTypes: [.int32]
        )
        return outputs[0].reshaped([h, w, k])
    }

    /// Sums L_r over 8 directions (4 axis-aligned + 4 diagonal). Hirschmüller 2008 §II-A recommends
    /// 8+ paths.
    public static func aggregate(
        costVolume: MLXArray, penalties: Penalties = Penalties(),
        implementation: Implementation = .simd
    ) -> MLXArray {
        let paths = (0..<8).map {
            aggregateSinglePath(costVolume: costVolume, direction: $0, penalties: penalties,
                                implementation: implementation)
        }
        return paths[0] + paths[1] + paths[2] + paths[3] + paths[4] + paths[5] + paths[6] + paths[7]
    }
}
