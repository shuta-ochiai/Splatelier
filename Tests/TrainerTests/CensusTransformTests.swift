// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Census Transform (PocketGS Operator G). Bit layout follows slang/census_transform.slang
// (LSB-first, row-major, bit = neighbor >= center).

import Foundation
import MLX
import Testing
@testable import Trainer

/// A uniform image has n >= c for all 24 neighbors, so 0xFFFFFF.
@Test
func censusTransformConstantImageAllOnes() {
    let h = 8, w = 8
    let luma = MLXArray([Float](repeating: 0.5, count: h * w), [h, w])
    let census = CensusTransform.compute(luma: luma)
    let arr = census.asArray(Int32.self)
    let expected: Int32 = (1 << 24) - 1   // 0x00FFFFFF
    for v in arr { #expect(v == expected, "got \(v), expected \(expected)") }
}

/// If only the center is the maximum, all 24 neighbors are below it, so 0.
@Test
func censusTransformCenterIsMaxYieldsZero() {
    let h = 8, w = 8
    var pixels = [Float](repeating: 0.0, count: h * w)
    pixels[(h / 2) * w + (w / 2)] = 1.0
    let luma = MLXArray(pixels, [h, w])
    let census = CensusTransform.compute(luma: luma)
    let arr = census.asArray(Int32.self)
    let centerIdx = (h / 2) * w + (w / 2)
    #expect(arr[centerIdx] == 0,
        "center pixel (max) should have all-zero census; got \(arr[centerIdx])")
}

/// If only the center is the minimum, all bits are set: 0xFFFFFF.
@Test
func censusTransformCenterIsMinYieldsAllOnes() {
    let h = 8, w = 8
    var pixels = [Float](repeating: 1.0, count: h * w)
    pixels[(h / 2) * w + (w / 2)] = 0.0
    let luma = MLXArray(pixels, [h, w])
    let census = CensusTransform.compute(luma: luma)
    let arr = census.asArray(Int32.self)
    let centerIdx = (h / 2) * w + (w / 2)
    let expected: Int32 = (1 << 24) - 1
    #expect(arr[centerIdx] == expected,
        "center (min) should have all-ones census; got \(arr[centerIdx])")
}

/// Exact bit-level check at the center (3, 3) = 0.5 of a 6×6 image. Neighbors: row dy=-2 all 1.0,
/// row dy=-1 all 0.0, row dy=0 at x=1,2,4,5 is 1,0,0,1, row dy=+1 all 1.0, row dy=+2 all 0.0.
@Test
func censusTransformExactPattern() {
    let h = 6, w = 6
    var pixels = [Float](repeating: 0.0, count: h * w)
    pixels[3 * w + 3] = 0.5
    for x in 0..<w { pixels[1 * w + x] = 1.0 }
    pixels[3 * w + 1] = 1.0
    pixels[3 * w + 2] = 0.0
    pixels[3 * w + 4] = 0.0
    pixels[3 * w + 5] = 1.0
    for x in 0..<w { pixels[4 * w + x] = 1.0 }
    let luma = MLXArray(pixels, [h, w])
    let census = CensusTransform.compute(luma: luma)
    let arr = census.asArray(Int32.self)

    // bits 0-4 = 0x1F, 5-9 = 0, 10-13 = 0b1001 << 10 = 0x2400, 14-18 = 0x1F << 14 = 0x7C000, 19-23
    // = 0
    let expected: Int32 = 0x1F | 0x2400 | 0x7C000
    let actual = arr[3 * w + 3]
    let actualHex = String(actual, radix: 16, uppercase: true)
    let expectedHex = String(expected, radix: 16, uppercase: true)
    #expect(actual == expected,
        "got 0x\(actualHex), expected 0x\(expectedHex)")
}

// MARK: - Improved Census Transform (CMCT, Fife & Archibald 2013)

/// A uniform image has all 25 pixels >= mean, so all 25 bits are set.
@Test
func improvedCensusConstantImageAllOnes() {
    let h = 8, w = 8
    let luma = MLXArray([Float](repeating: 0.5, count: h * w), [h, w])
    let census = CensusTransform.compute(luma: luma, variant: .improved)
    let arr = census.asArray(Int32.self)
    let expected: Int32 = (1 << 25) - 1
    for v in arr { #expect(v == expected, "got \(v), expected \(expected)") }
}

/// An image with only the center at 1 has mean = 1/25; only the center itself (bit 12 = row 2·5 +
/// col 2) is >= mean.
@Test
func improvedCensusCenterSpikeIsolatesBit() {
    let h = 8, w = 8
    var pixels = [Float](repeating: 0.0, count: h * w)
    pixels[(h / 2) * w + (w / 2)] = 1.0
    let luma = MLXArray(pixels, [h, w])
    let census = CensusTransform.compute(luma: luma, variant: .improved)
    let arr = census.asArray(Int32.self)
    let centerIdx = (h / 2) * w + (w / 2)
    #expect(arr[centerIdx] == (1 << 12),
        "got \(String(arr[centerIdx], radix: 2)), expected bit 12 only")
}

/// With a horizontal gradient (left 0 → right 1), the center signature is "left 2 columns 0, right
/// 3 columns 1" in every row.
@Test
func improvedCensusHorizontalGradient() {
    let h = 8, w = 8
    var pixels = [Float](repeating: 0.0, count: h * w)
    for y in 0..<h {
        for x in 0..<w { pixels[y * w + x] = Float(x) / Float(w - 1) }
    }
    let luma = MLXArray(pixels, [h, w])
    let census = CensusTransform.compute(luma: luma, variant: .improved)
    let arr = census.asArray(Int32.self)
    // The window at center (4, 4) spans x ∈ [2, 6] with values 2/7..6/7, mean = 4/7. Each row
    // dx=-2..2 gives 0,0,1,1,1 → LSB-first 0b11100, for 5 rows.
    var expected: Int32 = 0
    let rowPattern: Int32 = 0b11100   // = 28
    for k in 0..<5 { expected |= rowPattern << (5 * k) }
    let actual = arr[4 * w + 4]
    #expect(actual == expected,
        "got 0x\(String(actual, radix: 16)), expected 0x\(String(expected, radix: 16))")
}

/// CMCT compares against the mean rather than the center, so textured input gives a signature
/// different from classic.
@Test
func improvedCensusDiffersFromClassicOnGradient() {
    let h = 16, w = 16
    var pixels = [Float](repeating: 0.0, count: h * w)
    for y in 0..<h {
        for x in 0..<w { pixels[y * w + x] = Float(x) / Float(w - 1) }
    }
    let luma = MLXArray(pixels, [h, w])
    let classic = CensusTransform.compute(luma: luma, variant: .classic).asArray(Int32.self)
    let improved = CensusTransform.compute(luma: luma, variant: .improved).asArray(Int32.self)
    var diffPixels = 0
    for i in 0..<(h * w) { if classic[i] != improved[i] { diffPixels += 1 } }
    #expect(diffPixels > 0, "CMCT must differ from classic on textured input")
}

/// `hammingDistance` (Swift-side popcount).
@Test
func hammingDistanceBasics() {
    #expect(CensusTransform.hammingDistance(0, 0) == 0)
    #expect(CensusTransform.hammingDistance(0xFF, 0) == 8)
    #expect(CensusTransform.hammingDistance(0x0F, 0xF0) == 8)
    #expect(CensusTransform.hammingDistance(0x1234, 0x1235) == 1)
}

/// `rgbToLuma` coefficients (R 0.299, G 0.587, B 0.114).
@Test
func rgbToLumaCoefficients() {
    let rgb = MLXArray(
        [Float(1), 0, 0,
         0, 1, 0,
         0, 0, 1,
         0.5, 0.5, 0.5],
        [1, 4, 3]
    )
    let luma = CensusTransform.rgbToLuma(rgb)
    let arr = luma.asArray(Float.self)
    #expect(abs(arr[0] - 0.299) < 1e-5)
    #expect(abs(arr[1] - 0.587) < 1e-5)
    #expect(abs(arr[2] - 0.114) < 1e-5)
    #expect(abs(arr[3] - 0.5) < 1e-5)   // coefficients sum to 1, so gray is unchanged
}
