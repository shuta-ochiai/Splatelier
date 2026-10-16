// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import CoreGraphics
import Foundation
import ImageIO
import MLX

/// Reads an image via ImageIO into `[H, W, 3]` Float32 `[0, 1]`. Row 0 = top of the image (matches
/// the rasterizer pixel convention).
public enum ImageLoader {

    public enum Error: Swift.Error, CustomStringConvertible {
        case decodeFailed(String)
        case contextFailed
        public var description: String {
            switch self {
            case .decodeFailed(let p): return "Image decode failed: \(p)"
            case .contextFailed: return "Failed to create CGContext for image conversion"
            }
        }
    }

    /// Downscales, keeping aspect ratio, when the long side exceeds `maxDimension`.
    public static func loadRGB(url: URL, maxDimension: Int? = nil) throws -> MLXArray {
        guard
            let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw Error.decodeFailed(url.path)
        }
        var targetW = cgImage.width
        var targetH = cgImage.height
        if let maxDimension, max(targetW, targetH) > maxDimension, maxDimension > 0 {
            let scale = Double(maxDimension) / Double(max(targetW, targetH))
            targetW = max(1, Int((Double(cgImage.width) * scale).rounded()))
            targetH = max(1, Int((Double(cgImage.height) * scale).rounded()))
        }
        return try toRGBArray(cgImage, targetWidth: targetW, targetHeight: targetH)
    }

    /// Redraws into an RGBA8 context to normalize color space and bit depth, resampling if needed.
    static func toRGBArray(
        _ cgImage: CGImage, targetWidth: Int? = nil, targetHeight: Int? = nil
    ) throws -> MLXArray {
        let width = targetWidth ?? cgImage.width
        let height = targetHeight ?? cgImage.height
        let bytesPerRow = width * 4
        var rgba = [UInt8](repeating: 0, count: height * bytesPerRow)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        // noneSkipLast = RGBX (non-premultiplied). Keeps color even for images with alpha.
        let bitmapInfo = CGImageAlphaInfo.noneSkipLast.rawValue
        let drawn: Bool = rgba.withUnsafeMutableBytes { buf -> Bool in
            guard let ctx = CGContext(
                data: buf.baseAddress,
                width: width, height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: bitmapInfo
            ) else {
                return false
            }
            // Buffer first row = top of the image without flipping the CTM (orientation pinned by
            // tests).
            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { throw Error.contextFailed }

        var rgb = [Float](repeating: 0, count: width * height * 3)
        for i in 0..<(width * height) {
            rgb[i * 3 + 0] = Float(rgba[i * 4 + 0]) / 255.0
            rgb[i * 3 + 1] = Float(rgba[i * 4 + 1]) / 255.0
            rgb[i * 3 + 2] = Float(rgba[i * 4 + 2]) / 255.0
        }
        return MLXArray(rgb, [height, width, 3])
    }
}
