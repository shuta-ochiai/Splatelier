// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// CostVolumeMVS orchestrator (PocketGS Operator G).

import Foundation
import MLX
import Testing
@testable import Trainer

/// The app's MVS parameters (the same values as `CaptureTraining.MVSMode.mvsParameters`, which this
/// target cannot import).
private let appMVSParameters = CostVolumeMVS.Parameters.mobile3DRecon(
    m: 5, voxelDedupSize: 0.003,
    fusionConsistency: CostVolumeMVS.FusionConsistency(vMin: 3),
    refSelection: ReferenceSelection.Parameters(minCovisibility: 0.6))

/// With the app's parameters, 5 cameras in a row looking at a textured fronto-parallel plane at a
/// known depth give a cloud whose world z is within ±5% of the true depth (the depth hypotheses are
/// ~2% apart here).
@Test
func costVolumeMVSMultiViewPlane() {
    let w = 48, h = 24
    let fx: Float = 50, fy: Float = 50, cx: Float = Float(w) / 2, cy: Float = Float(h) / 2
    let spacing: Float = 0.5
    let trueDepth: Float = 5.0
    let n = 5
    let shiftPixels = Int((fx * spacing / trueDepth).rounded())   // = 5 per camera

    // Slice each camera's image out of a shared textured canvas: camera i stands at world x =
    // spacing·i, so a point appears shift·i px further left in its image than in camera 0's.
    var lcg = TestLCG(seed: 31)
    let extW = w + shiftPixels * (n - 1)
    var rgbPixels: [Float] = []
    for _ in 0..<(h * extW) {
        let v = lcg.next()
        rgbPixels += [v, v, v]
    }
    let extRgb = MLXArray(rgbPixels, [h, extW, 3])
    let images = (0..<n).map { i -> MLXArray in
        let img = extRgb[0..., (shiftPixels * i)..<(shiftPixels * i + w), 0...]
        eval(img)
        return img
    }
    // viewmat = [I | -C]
    let viewmats: [[Float]] = (0..<n).map { i in
        [1, 0, 0, -spacing * Float(i), 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    }
    // Sparse points on the plane inside every camera's view (they set the depth range and the
    // covisibility).
    var sparse: [SIMD3<Float>] = []
    for xi in 0...8 {
        for y: Float in [-0.5, 0, 0.5] {
            sparse.append(SIMD3<Float>(Float(xi) * 0.25, y, trueDepth))
        }
    }
    let inputs = CostVolumeMVS.Inputs(
        viewmats: viewmats,
        intrinsics: [Intrinsics](repeating: (fx: fx, fy: fy, cx: cx, cy: cy), count: n),
        rgbImages: images,
        sparsePoints: sparse,
        sceneCenter: SIMD3<Float>(spacing * Float(n - 1) / 2, 0, trueDepth)
    )
    let result = CostVolumeMVS.build(inputs: inputs, parameters: appMVSParameters)

    #expect(result.positions.shape[0] > 0,
        "expected non-empty cloud; got shape \(result.positions.shape)")
    let positions = result.positions.asArray(Float.self)
    let nPoints = positions.count / 3
    var nearTrueDepth = 0
    for i in 0..<nPoints {
        let wz = positions[i * 3 + 2]
        if abs(wz - trueDepth) < trueDepth * 0.05 { nearTrueDepth += 1 }
    }
    let ratio = Double(nearTrueDepth) / Double(max(nPoints, 1))
    #expect(ratio >= 0.95,
        "fraction near true depth = \(ratio) (\(nearTrueDepth)/\(nPoints))")
}

/// No crash on hard input (uniform image, 2 cameras), and nothing passes: the uniform image has no
/// confident depth, and 2 cameras cannot reach 3 agreeing views.
@Test
func costVolumeMVSDoesNotCrashOnDifficultScene() {
    let w = 16, h = 8
    let rgb = MLXArray([Float](repeating: 0.5, count: h * w * 3), [h, w, 3])   // uniform image
    eval(rgb)
    let vm: [Float] = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
    let vm2: [Float] = [
        1, 0, 0, 0.5,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
    let inputs = CostVolumeMVS.Inputs(
        viewmats: [vm, vm2],
        intrinsics: [(30, 30, 8, 4), (30, 30, 8, 4)],
        rgbImages: [rgb, rgb],
        sparsePoints: [SIMD3<Float>(0, 0, 5)],
        sceneCenter: SIMD3<Float>(0, 0, 5)
    )
    let result = CostVolumeMVS.build(inputs: inputs, parameters: appMVSParameters)
    #expect(result.positions.shape == [0, 3])
    #expect(result.colors.shape == [0, 3])
}

/// The GPU consistency check outputs the same points in the same order as the CPU version
/// (`fuseWithConsistencyCPU`). Depth maps of 6 side-by-side cameras viewing a tilted plane, mixed
/// with outliers and invalid pixels.
@Test(arguments: [true, false])
func fusionConsistencyGPUMatchesCPU(averageDepth: Bool) {
    let w = 40, h = 30
    let intr: Intrinsics = (fx: 40, fy: 40, cx: 20, cy: 15)
    let n = 6
    // camera i stands at x = 0.3·i looking along +z (viewmat = [I | -C]). Plane z = 4 +
    // 0.1·x_world.
    let viewmats: [[Float]] = (0..<n).map { i in
        [1, 0, 0, -0.3 * Float(i), 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    }
    var lcg = TestLCG(seed: 7)
    var depths: [Int: [Float]] = [:]
    var dims: [Int: (w: Int, h: Int)] = [:]
    for i in 0..<n {
        var d = [Float](repeating: -1, count: w * h)
        for y in 0..<h {
            for x in 0..<w {
                // depth at which ray (rx, ry, 1)·z hits the plane z = 4 + 0.1·(cx_world + rx·z)
                let rx = (Float(x) + 0.5 - intr.cx) / intr.fx
                let z = (4 + 0.1 * 0.3 * Float(i)) / (1 - 0.1 * rx)
                let r = lcg.next()
                if r < 0.1 { continue }                      // invalid pixel
                d[y * w + x] = r < 0.2 ? z * 1.3 : z * (1 + 0.004 * (lcg.next() - 0.5))   // outlier / small noise
            }
        }
        depths[i] = d
        dims[i] = (w: w, h: h)
    }
    let centers = viewmats.map { FrameSubsampler.cameraCenter(viewmat: $0) }
    let params = ReferenceSelection.Parameters(bTarget: 0.6, sigmaB: 0.5, alphaMin: 0.01, baselineToDepth: nil)
    let rgb: (Int) -> MLXArray = { i in MLXArray((0..<(w * h * 3)).map { Float(($0 + i) % 7) / 7 }, [h, w, 3]) }
    let fusion = CostVolumeMVS.FusionConsistency(neighborCount: 4, vMin: 2, relTol: 0.02, averageDepth: averageDepth)
    func run(_ f: (([Int: [Float]], [Int: (w: Int, h: Int)]) -> (positions: MLXArray, colors: MLXArray, kept: Int, total: Int)))
        -> (pos: [Float], col: [Float], kept: Int, total: Int) {
        let r = f(depths, dims)
        return (r.positions.asArray(Float.self), r.colors.asArray(Float.self), r.kept, r.total)
    }
    let sel = [SIMD3<Float>](repeating: SIMD3<Float>(0.75, 0, 4), count: n)
    let selParams = [ReferenceSelection.Parameters](repeating: params, count: n)
    let gpu = run { d, dm in
        CostVolumeMVS.fuseWithConsistency(
            depths: d, dims: dm, viewmats: viewmats, intrinsics: Array(repeating: intr, count: n), rgb: rgb,
            cameraCenters: centers, selCenter: sel, selParams: selParams, fusion: fusion)
    }
    let cpu = run { d, dm in
        CostVolumeMVS.fuseWithConsistencyCPU(
            depths: d, dims: dm, viewmats: viewmats, intrinsics: Array(repeating: intr, count: n), rgb: rgb,
            cameraCenters: centers, selCenter: sel, selParams: selParams, fusion: fusion)
    }
    #expect(cpu.kept > 0 && cpu.kept < cpu.total, "outliers are dropped and some points remain")
    #expect(gpu.kept == cpu.kept && gpu.total == cpu.total)
    #expect(gpu.col == cpu.col, "the same pixels remain in the same order")
    let maxDiff = zip(gpu.pos, cpu.pos).map { abs($0 - $1) }.max() ?? 0
    #expect(maxDiff < 1e-5, "max |Δpos| = \(maxDiff)")
}
