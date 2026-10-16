// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// PNG export of render results. Depends only on CoreGraphics + ImageIO, so it works on iOS too
// (the CLI's `saveImagePNG` depends on AppKit and is macOS-only).

import Foundation
import MLX
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

public enum ImagePNG {
    enum WriteError: Error { case cgImageFailed, destinationFailed, finalizeFailed }

    /// Writes a `[H, W, 3]` float image (range 0..1) as PNG to `url`.
    /// With `maxDim`, downscales so the long side is at most that (for thumbnails).
    public static func write(_ image: MLXArray, to url: URL, maxDim: Int? = nil) throws {
        precondition(image.shape.count == 3 && image.shape[2] == 3,
                     "image must be [H, W, 3], got \(image.shape)")
        let h = image.shape[0], w = image.shape[1]
        let host = image.reshaped([h * w * 3]).asArray(Float.self)

        let rowBytes = w * 4
        var rgba = [UInt8](repeating: 255, count: rowBytes * h)
        for i in 0..<(w * h) {
            for c in 0..<3 {
                let v = Swift.max(0, Swift.min(1, host[i * 3 + c]))
                rgba[i * 4 + c] = UInt8((v * 255).rounded())
            }
        }

        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let ctx = CGContext(
            data: &rgba, width: w, height: h, bitsPerComponent: 8,
            bytesPerRow: rowBytes, space: cs, bitmapInfo: bitmapInfo
        ), var cg = ctx.makeImage() else {
            throw WriteError.cgImageFailed
        }

        if let maxDim, Swift.max(w, h) > maxDim {
            let scale = Double(maxDim) / Double(Swift.max(w, h))
            let nw = Swift.max(1, Int((Double(w) * scale).rounded()))
            let nh = Swift.max(1, Int((Double(h) * scale).rounded()))
            if let sctx = CGContext(
                data: nil, width: nw, height: nh, bitsPerComponent: 8,
                bytesPerRow: 0, space: cs, bitmapInfo: bitmapInfo
            ) {
                sctx.interpolationQuality = .high
                sctx.draw(cg, in: CGRect(x: 0, y: 0, width: nw, height: nh))
                if let scaled = sctx.makeImage() { cg = scaled }
            }
        }

        guard let dest = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil
        ) else { throw WriteError.destinationFailed }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { throw WriteError.finalizeFailed }
    }
}
