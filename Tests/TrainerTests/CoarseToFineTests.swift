// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// coarse-to-fine helpers (TrainingDriver.coarseCamera / coarseDownsample / resampleRational).
// Pins down the math of the downscaled camera + downscaled target used early in training.

import Foundation
import MLX
import Testing
@testable import Trainer

@Test
func coarseDownsampleBoxAveragesFactor2() {
    // 4×4 → 2×2 box mean: [0,1,4,5]=2.5, [2,3,6,7]=4.5, [8,9,12,13]=10.5, [10,11,14,15]=12.5
    let img = MLXArray((0..<16).map { Float($0) }, [4, 4, 1])
    let out = TrainingDriver.coarseDownsample(img, factor: 2)
    eval(out)
    #expect(out.shape == [2, 2, 1], "shape \(out.shape)")
    let v = out.reshaped([4]).asArray(Float.self)
    #expect(abs(v[0] - 2.5) < 1e-5)
    #expect(abs(v[1] - 4.5) < 1e-5)
    #expect(abs(v[2] - 10.5) < 1e-5)
    #expect(abs(v[3] - 12.5) < 1e-5)
}

@Test
func coarseDownsampleCropsIndivisibleEdge() {
    // 5×5 → crop the last row/column to get 2×2. Constant image, so values are unchanged.
    let img = MLXArray([Float](repeating: 0.7, count: 5 * 5 * 3), [5, 5, 3])
    let out = TrainingDriver.coarseDownsample(img, factor: 2)
    eval(out)
    #expect(out.shape == [2, 2, 3], "shape \(out.shape)")
    let v = out.reshaped([12]).asArray(Float.self)
    for x in v { #expect(abs(x - 0.7) < 1e-5) }
}

@Test
func coarseDownsampleFactor1IsCrop() {
    // A uniform image keeps its value after downscaling.
    let img = MLXArray([Float](repeating: 0.25, count: 8 * 8 * 3), [8, 8, 3])
    let half = TrainingDriver.coarseDownsample(img, factor: 2)
    eval(half)
    #expect(half.shape == [4, 4, 3])
    #expect(abs(half.mean().item(Float.self) - 0.25) < 1e-5)
}

@Test
func coarseDownsampleUInt8MatchesFloatPath() {
    // Downscaling uint8 directly matches the float [0,1] path (mean(x/255)=mean(x)/255).
    // Guarantees the uint8 direct read is correct without building a full-res float.
    var px = [UInt8](repeating: 0, count: 8 * 8 * 3)
    for i in 0..<px.count { px[i] = UInt8((i * 37 + 13) % 256) }
    let u8 = MLXArray(px, [8, 8, 3])
    let f01 = u8.asType(.float32) / 255.0

    let fromU8 = TrainingDriver.coarseDownsample(u8, factor: 2)     // uint8 path (includes /255)
    let fromFloat = TrainingDriver.coarseDownsample(f01, factor: 2) // float[0,1] path (passthrough)
    eval(fromU8, fromFloat)
    #expect(fromU8.shape == [4, 4, 3])
    let a = fromU8.reshaped([4 * 4 * 3]).asArray(Float.self)
    let b = fromFloat.reshaped([4 * 4 * 3]).asArray(Float.self)
    var maxDiff: Float = 0
    for i in 0..<a.count { maxDiff = max(maxDiff, abs(a[i] - b[i])) }
    #expect(maxDiff < 1e-6, "uint8 path diverged from float path by \(maxDiff)")
}

@Test
func coarseCameraScalesIntrinsicsAndDims() {
    let cam = Camera(
        viewmat: MLXArray([
            Float(1), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1,
        ], [16]),
        intrinsics: MLXArray([Float(1000), 1100, 720, 540], [4]),
        width: 1440, height: 1080,
        background: MLXArray([Float(0), 0, 0], [3])
    )
    let coarse = TrainingDriver.coarseCamera(cam, factor: 2)
    #expect(coarse.width == 720 && coarse.height == 540)
    eval(coarse.intrinsics, coarse.viewmat)
    let intr = coarse.intrinsics.asArray(Float.self)
    #expect(abs(intr[0] - 500) < 1e-3, "fx")
    #expect(abs(intr[1] - 550) < 1e-3, "fy")
    #expect(abs(intr[2] - 360) < 1e-3, "cx")
    #expect(abs(intr[3] - 270) < 1e-3, "cy")
    // extrinsics are unchanged.
    let vm = coarse.viewmat.asArray(Float.self)
    #expect(vm == cam.viewmat.asArray(Float.self), "viewmat must be unchanged")
    #expect(coarse.tileSize == cam.tileSize)
}

// MARK: - fine-resample (rational-ratio area, for 720×540)

@Test
func areaWeight3to2IsCorrect() {
    // 3 inputs → 2 outputs (cell=1.5). Column sums = 1, out0=[2/3,1/3,0], out1=[0,1/3,2/3].
    let w = TrainingDriver.areaWeight(den: 3, num: 2).reshaped([6]).asArray(Float.self)
    // Storage order w[i*num+j]: [ (0,0),(0,1), (1,0),(1,1), (2,0),(2,1) ]
    #expect(abs(w[0] - 2.0/3) < 1e-6)   // i0,j0
    #expect(abs(w[1] - 0) < 1e-6)       // i0,j1
    #expect(abs(w[2] - 1.0/3) < 1e-6)   // i1,j0
    #expect(abs(w[3] - 1.0/3) < 1e-6)   // i1,j1
    #expect(abs(w[4] - 0) < 1e-6)       // i2,j0
    #expect(abs(w[5] - 2.0/3) < 1e-6)   // i2,j1
}

@Test
func resampleRationalMatchesBoxForInteger() {
    // The area weights for num=1,den=2 are uniform [0.5,0.5] = box mean → matches
    // coarseDownsample(factor 2).
    let img = MLXArray((0..<16).map { Float($0) }, [4, 4, 1])
    let r = TrainingDriver.resampleRational(img, num: 1, den: 2)
    let box = TrainingDriver.coarseDownsample(img, factor: 2)
    eval(r, box)
    #expect(r.shape == [2, 2, 1], "shape \(r.shape)")
    let a = r.reshaped([4]).asArray(Float.self), b = box.reshaped([4]).asArray(Float.self)
    for i in 0..<4 { #expect(abs(a[i] - b[i]) < 1e-6, "resample vs box @\(i): \(a[i]) vs \(b[i])") }
}

@Test
func resampleRational3to2HandChecked() {
    // 3×6×1, value = column index (0..5), constant along rows. W:6→4 (3:2), H:3→2 (3:2). Columns
    // [0,1,2,3,4,5] → blocks [0,1,2]→[(0+0.5·1)/1.5=1/3, (0.5·1+2)/1.5=5/3],
    //                                [3,4,5]→[(3+0.5·4)/1.5=10/3, (0.5·4+5)/1.5=14/3].
    // H is constant along columns, so each column value is unchanged after 3→2 → both output rows
    // are identical.
    let row: [Float] = [0, 1, 2, 3, 4, 5]
    let img = MLXArray(row + row + row, [3, 6, 1])
    let out = TrainingDriver.resampleRational(img, num: 2, den: 3)
    eval(out)
    #expect(out.shape == [2, 4, 1], "shape \(out.shape)")
    let v = out.reshaped([8]).asArray(Float.self)
    let expected: [Float] = [1.0/3, 5.0/3, 10.0/3, 14.0/3]
    for r in 0..<2 {
        for cidx in 0..<4 {
            #expect(abs(v[r*4 + cidx] - expected[cidx]) < 1e-5,
                    "row\(r) col\(cidx): \(v[r*4+cidx]) vs \(expected[cidx])")
        }
    }
}

@Test
func resampleRationalUInt8MatchesFloatPath() {
    // The uint8 path (includes /255) matches the float[0,1] path (weight column sums = 1 preserve
    // the value range).
    var px = [UInt8](repeating: 0, count: 9 * 9 * 3)
    for i in 0..<px.count { px[i] = UInt8((i * 53 + 7) % 256) }
    let u8 = MLXArray(px, [9, 9, 3])
    let f01 = u8.asType(.float32) / 255.0
    let fromU8 = TrainingDriver.resampleRational(u8, num: 2, den: 3)
    let fromFloat = TrainingDriver.resampleRational(f01, num: 2, den: 3)
    eval(fromU8, fromFloat)
    #expect(fromU8.shape == [6, 6, 3], "shape \(fromU8.shape)")   // 9/3*2=6
    let a = fromU8.reshaped([6*6*3]).asArray(Float.self)
    let b = fromFloat.reshaped([6*6*3]).asArray(Float.self)
    var maxDiff: Float = 0
    for i in 0..<a.count { maxDiff = max(maxDiff, abs(a[i] - b[i])) }
    #expect(maxDiff < 1e-6, "uint8 path diverged by \(maxDiff)")
}

/// The 2/3 resample of a uint8 capture is the exact area average up to float rounding (the earlier
/// float32 matmul formulation was off by up to ~1e-3 on the GPU).
@Test
func resampleRational3to2MatchesExactAreaAverage() {
    let h = 30, w = 42
    var rng = TestLCG(seed: 5)
    let px = (0..<(h * w * 3)).map { _ in UInt8(rng.next() * 255) }
    let out = TrainingDriver.resampleRational(MLXArray(px, [h, w, 3]), num: 2, den: 3)
    eval(out)
    #expect(out.shape == [20, 28, 3])
    let v = out.asArray(Float.self)
    let weights: [[Double]] = [[2.0 / 3, 1.0 / 3, 0], [0, 1.0 / 3, 2.0 / 3]]
    var maxErr = 0.0
    for oy in 0..<20 { for ox in 0..<28 { for ch in 0..<3 {
        var exact = 0.0
        for i in 0..<3 { for j in 0..<3 {
            exact += weights[oy % 2][i] * weights[ox % 2][j]
                * Double(px[(((oy / 2) * 3 + i) * w + (ox / 2) * 3 + j) * 3 + ch])
        } }
        maxErr = max(maxErr, abs(Double(v[(oy * 28 + ox) * 3 + ch]) - exact / 255))
    } } }
    #expect(maxErr < 1e-6, "max error vs exact area average: \(maxErr)")
}

@Test
func coarseCameraScaled2over3() {
    // 1080×810 → 720×540, intrinsics ×2/3.
    let cam = Camera(
        viewmat: MLXArray((0..<16).map { Float($0) }, [16]),
        intrinsics: MLXArray([Float(900), 900, 540, 405], [4]),
        width: 1080, height: 810, tileSize: 16,
        background: MLXArray([Float(0), 0, 0], [3]))
    let s = TrainingDriver.coarseCameraScaled(cam, num: 2, den: 3)
    #expect(s.width == 720 && s.height == 540, "size \(s.width)x\(s.height)")
    let k = s.intrinsics.asArray(Float.self)
    #expect(abs(k[0] - 600) < 1e-3 && abs(k[2] - 360) < 1e-3, "intrinsics \(k)")  // ×2/3
}
