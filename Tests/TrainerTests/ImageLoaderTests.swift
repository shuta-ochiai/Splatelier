// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import CoreGraphics
import Foundation
import ImageIO
import MLX
import Testing
import UniformTypeIdentifiers
@testable import Trainer

/// Writes a top-left-origin RGB float buffer to PNG.
private func writePNG(rgb: [Float], width: Int, height: Int, to url: URL) throws {
    var rgba = [UInt8](repeating: 255, count: width * height * 4)
    for i in 0..<(width * height) {
        func q(_ v: Float) -> UInt8 { UInt8(max(0, min(255, (v * 255).rounded()))) }
        rgba[i * 4 + 0] = q(rgb[i * 3 + 0])
        rgba[i * 4 + 1] = q(rgb[i * 3 + 1])
        rgba[i * 4 + 2] = q(rgb[i * 3 + 2])
        rgba[i * 4 + 3] = 255
    }
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let cgImage: CGImage = rgba.withUnsafeMutableBytes { buf in
        let ctx = CGContext(
            data: buf.baseAddress,
            width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        )!
        return ctx.makeImage()!
    }
    let dest = CGImageDestinationCreateWithURL(
        url as CFURL, UTType.png.identifier as CFString, 1, nil
    )!
    CGImageDestinationAddImage(dest, cgImage, nil)
    guard CGImageDestinationFinalize(dest) else {
        throw NSError(domain: "test", code: 1)
    }
}

@Test
func imageLoaderRoundTripsColors() throws {
    let width = 4, height = 3
    // A different color per pixel so channel swaps and transposes are detectable.
    var rgb = [Float](repeating: 0, count: width * height * 3)
    for y in 0..<height {
        for x in 0..<width {
            let i = y * width + x
            rgb[i * 3 + 0] = Float(x) / Float(width - 1)
            rgb[i * 3 + 1] = Float(y) / Float(height - 1)
            rgb[i * 3 + 2] = 0.5
        }
    }

    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("imgload_\(UUID().uuidString).png")
    defer { try? FileManager.default.removeItem(at: url) }
    try writePNG(rgb: rgb, width: width, height: height, to: url)

    let loaded = try ImageLoader.loadRGB(url: url)
    #expect(loaded.shape == [height, width, 3])
    let got = loaded.asArray(Float.self)

    var maxDiff: Float = 0
    for i in 0..<(width * height * 3) {
        maxDiff = max(maxDiff, abs(got[i] - rgb[i]))
    }
    // Tolerance for 8-bit quantization.
    #expect(maxDiff < 1.0 / 255.0 + 1e-4, "round-trip diff \(maxDiff) too large")
}

@Test
func imageLoaderPreservesTopLeftOrigin() throws {
    // Top half red, bottom half blue; checks that row 0 is the top edge.
    let width = 2, height = 4
    var rgb = [Float](repeating: 0, count: width * height * 3)
    for y in 0..<height {
        for x in 0..<width {
            let i = y * width + x
            if y < height / 2 {
                rgb[i * 3 + 0] = 1.0
            } else {
                rgb[i * 3 + 2] = 1.0
            }
        }
    }

    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("imgorient_\(UUID().uuidString).png")
    defer { try? FileManager.default.removeItem(at: url) }
    try writePNG(rgb: rgb, width: width, height: height, to: url)

    let loaded = try ImageLoader.loadRGB(url: url).asArray(Float.self)
    #expect(loaded[0] > 0.9, "top-left R should be ~1 (got \(loaded[0]))")
    #expect(loaded[2] < 0.1, "top-left B should be ~0 (got \(loaded[2]))")
    let lastRowStart = (height - 1) * width * 3
    #expect(loaded[lastRowStart + 0] < 0.1, "bottom-left R should be ~0")
    #expect(loaded[lastRowStart + 2] > 0.9, "bottom-left B should be ~1")
}
