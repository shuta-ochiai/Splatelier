// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// FAST-9 corner detector.

import Foundation
import MLX
import Testing
@testable import Trainer

/// A uniform image should have no corners.
@Test
func fastCornerUniformImageEmpty() {
    let h = 32, w = 32
    let luma = MLXArray([Float](repeating: 0.5, count: h * w), [h, w])
    let sm = FASTCorner.score(luma: luma, threshold: 0.05)
    let arr = sm.asArray(Int32.self)
    for v in arr { #expect(v == 0, "uniform image should have no corners; got score \(v)") }
}

/// A bright 5x5 square on black gives non-zero scores near its corners.
@Test
func fastCornerDetectsClearCorner() {
    let h = 24, w = 24
    var pixels = [Float](repeating: 0.0, count: h * w)
    let cx = 10, cy = 10
    for dy in 0..<5 {
        for dx in 0..<5 {
            pixels[(cy + dy) * w + (cx + dx)] = 1.0
        }
    }
    let luma = MLXArray(pixels, [h, w])
    let sm = FASTCorner.score(luma: luma, threshold: 0.1)
    let arr = sm.asArray(Int32.self)
    // Corners should appear within ±2 of the square's 4 corners.
    let corners: [(Int, Int)] = [(cx, cy), (cx + 4, cy), (cx, cy + 4), (cx + 4, cy + 4)]
    var detected = 0
    for (x, y) in corners {
        for dy in -2...2 {
            for dx in -2...2 {
                let nx = x + dx, ny = y + dy
                if nx >= 0 && nx < w && ny >= 0 && ny < h && arr[ny * w + nx] > 0 {
                    detected += 1
                    break
                }
            }
            if detected > 0 { break }
        }
    }
    #expect(detected >= 1, "expected at least 1 detected corner near the bright square")
    // Keypoint extraction after NMS (cap 100).
    let (coords, scores) = FASTCorner.extractKeypoints(scoreMap: sm, maxKeypoints: 100)
    #expect(coords.count > 0, "extractKeypoints should return at least 1 keypoint")
    #expect(coords.count == scores.count)
    #expect(scores.allSatisfy { $0 > 0 })
}

/// NMS alone: when several pixels share the same score, only the local max is kept.
@Test
func fastCornerNMSReducesDuplicates() {
    let h = 32, w = 32
    var pixels = [Float](repeating: 0.0, count: h * w)
    let centers: [(Int, Int)] = [(7, 7), (7, 22), (22, 7), (22, 22)]
    for c in centers {
        for dy in 0..<4 {
            for dx in 0..<4 {
                pixels[(c.1 + dy) * w + (c.0 + dx)] = 1.0
            }
        }
    }
    let luma = MLXArray(pixels, [h, w])
    let sm = FASTCorner.score(luma: luma, threshold: 0.1)
    let (coords, _) = FASTCorner.extractKeypoints(scoreMap: sm, maxKeypoints: 50)
    // 4 squares × 4 corners = about 16. Same-score duplicates on adjacent pixels collapse to one
    // under NMS.
    #expect(coords.count >= 4 && coords.count <= 30,
        "expected 4-30 corners after NMS for 4 squares; got \(coords.count)")
}
