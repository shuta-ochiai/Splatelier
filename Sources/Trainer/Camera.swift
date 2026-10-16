// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX

/// Pinhole intrinsics (px).
public typealias Intrinsics = (fx: Float, fy: Float, cx: Float, cy: Float)

/// Pinhole camera and image size consumed by the renderer. `viewmat` is a flat row-major [16]
/// (`[r00, r01, r02, tx, …, 0, 0, 0, 1]`); `background` `[r, g, b]` is composited into pixels with
/// remaining transmittance.
public struct Camera {
    public let viewmat: MLXArray      // [16]
    public let intrinsics: MLXArray   // [4] = [fx, fy, cx, cy]
    public let width: Int
    public let height: Int
    public let tileSize: Int
    public let background: MLXArray   // [3]

    public init(
        viewmat: MLXArray,
        intrinsics: MLXArray,
        width: Int,
        height: Int,
        tileSize: Int = 16,
        background: MLXArray
    ) {
        precondition(viewmat.shape == [16], "viewmat must be a flat [16] row-major matrix")
        precondition(intrinsics.shape == [4], "intrinsics must be [fx, fy, cx, cy]")
        precondition(background.shape == [3], "background must be [r, g, b]")
        precondition(width > 0 && height > 0, "image dimensions must be positive")
        precondition(tileSize > 0, "tileSize must be positive")
        self.viewmat = viewmat
        self.intrinsics = intrinsics
        self.width = width
        self.height = height
        self.tileSize = tileSize
        self.background = background
    }

    public var numTilesX: Int { (width + tileSize - 1) / tileSize }
    public var numTilesY: Int { (height + tileSize - 1) / tileSize }
}
