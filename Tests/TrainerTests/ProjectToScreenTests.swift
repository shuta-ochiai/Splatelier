// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXRandom
import Testing
@testable import Trainer

private func slangProjectToScreenFwd(
    means3dWorld: MLXArray, quats: MLXArray, scales: MLXArray,
    viewmat: MLXArray, intrinsics: MLXArray, imageDims: MLXArray,
    opacities: MLXArray
) throws -> (means2d: MLXArray, conics: MLXArray, depths: MLXArray, radii: MLXArray) {
    let n = means3dWorld.shape[0]
    let kernel = try SlangCompiledKernel.load(name: "project_to_screen_fwd")
    let nBuf = MLXArray([Int32(n)], [1])
    let outputs = kernel(
        [means3dWorld, quats, scales, viewmat, intrinsics, imageDims, opacities, nBuf],
        grid: (((n + 63) / 64) * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[n * 2], [n * 3], [n], [n * 2]],
        outputDTypes: [.float32, .float32, .float32, .int32]
    )
    return (
        outputs[0].reshaped([n, 2]),
        outputs[1].reshaped([n, 3]),
        outputs[2],
        outputs[3].reshaped([n, 2])
    )
}

private func slangProjectToScreenBwd(
    means3dWorld: MLXArray, quats: MLXArray, scales: MLXArray,
    viewmat: MLXArray, intrinsics: MLXArray, imageDims: MLXArray,
    cotMeans2d: MLXArray, cotConics: MLXArray
) throws -> (gradMeans3dWorld: MLXArray, gradQuats: MLXArray, gradScales: MLXArray) {
    let n = means3dWorld.shape[0]
    let kernel = try SlangCompiledKernel.load(name: "project_to_screen_bwd")
    let nBuf = MLXArray([Int32(n)], [1])
    let outputs = kernel(
        [means3dWorld, quats, scales, viewmat, intrinsics, imageDims, nBuf,
         cotMeans2d, cotConics],
        grid: (((n + 63) / 64) * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[n * 3], [n * 4], [n * 3]],
        outputDTypes: [.float32, .float32, .float32]
    )
    return (
        outputs[0].reshaped([n, 3]),
        outputs[1].reshaped([n, 4]),
        outputs[2].reshaped([n, 3])
    )
}

private struct ProjectScene {
    let means3dWorld: MLXArray
    let quats: MLXArray
    let scales: MLXArray
    let viewmat: MLXArray
    let viewmatHost: [Float]
    let intrinsics: MLXArray
    let intrinsicsHost: [Float]
    let imageDims: MLXArray
    let imageWidth: Float
    let imageHeight: Float
    let opacities: MLXArray
}

private func sampleProjectScene(seed: UInt64, n: Int) -> ProjectScene {
    MLXRandom.seed(seed)
    // Place random points near the origin in front of the camera (tz ≳ 3.7) via viewmat so depth is
    // positive.
    let means3dWorld = MLXRandom.normal([n, 3]) * 0.5
    let quats = MLXRandom.normal([n, 4]) * 0.8 + 0.3
    let scales = MLX.exp(MLXRandom.normal([n, 3]) * 0.2)
    let view = makeViewMatrix(seed: seed &+ 1000)
    let intrinsicsHost: [Float] = [500.0, 500.0, 256.0, 256.0]
    let intrinsics = MLXArray(intrinsicsHost, [4])
    let imageWidth: Float = 512
    let imageHeight: Float = 512
    let imageDims = MLXArray([imageWidth, imageHeight], [2])
    // All above 1/255 so no opacity cull happens.
    let opacities = MLXArray([Float](repeating: 0.5, count: n), [n])
    eval(means3dWorld, quats, scales, view.mlx, intrinsics, imageDims, opacities)
    return ProjectScene(
        means3dWorld: means3dWorld, quats: quats, scales: scales,
        viewmat: view.mlx, viewmatHost: view.host,
        intrinsics: intrinsics, intrinsicsHost: intrinsicsHost,
        imageDims: imageDims,
        imageWidth: imageWidth, imageHeight: imageHeight,
        opacities: opacities
    )
}

@Test
func projectToScreenForwardMatchesChainedReference() throws {
    let n = 6
    let scene = sampleProjectScene(seed: 0, n: n)

    let (slangM2, slangC, slangD, slangR) = try slangProjectToScreenFwd(
        means3dWorld: scene.means3dWorld,
        quats: scene.quats, scales: scene.scales,
        viewmat: scene.viewmat, intrinsics: scene.intrinsics,
        imageDims: scene.imageDims, opacities: scene.opacities
    )
    let (refM2, refC, refD, refR) = referenceProjectToScreen(
        means3dWorld: scene.means3dWorld,
        quats: scene.quats, scales: scene.scales,
        viewmat: scene.viewmatHost,
        fx: scene.intrinsicsHost[0], fy: scene.intrinsicsHost[1],
        cx: scene.intrinsicsHost[2], cy: scene.intrinsicsHost[3],
        imageWidth: scene.imageWidth, imageHeight: scene.imageHeight,
        opacities: scene.opacities
    )
    eval(slangM2, slangC, slangD, slangR, refM2, refC, refD, refR)

    let diffM2 = MLX.abs(slangM2 - refM2).max().item(Float.self)
    #expect(diffM2 < 1e-3, "means2d diverged by \(diffM2)")

    let scaleC = MLX.abs(refC).max().item(Float.self)
    let diffC = MLX.abs(slangC - refC).max().item(Float.self)
    #expect(diffC / max(scaleC, 1e-6) < 1e-4, "conic rel diff \(diffC / max(scaleC, 1e-6))")

    let diffD = MLX.abs(slangD - refD).max().item(Float.self)
    #expect(diffD < 1e-5, "depths diverged by \(diffD)")

    let diffR = MLX.abs(slangR - refR).max().item(Int32.self)
    #expect(diffR == 0, "radii differ by \(diffR)")
}

@Test
func projectToScreenCullsDegenerateNearCameraGaussian() throws {
    // Two on the optical axis: #0 at cam z ≈ 0.02 (near the near plane), #1 at z ≈ 5.02. #0's
    // radius balloons to ~80k px through the EWA Jacobian (j00=fx/z), so it must be culled (radii
    // 0) to avoid full-screen cost. #1 must survive.
    let means3dWorld = MLXArray([
        Float(0), 0, 0,      // #0 → cam z ≈ 0.02
        0, 0, 5,             // #1 → cam z ≈ 5.02
    ], [2, 3])
    let quats = MLXArray([Float(1), 0, 0, 0, 1, 0, 0, 0], [2, 4])   // identity quat
    let scales = MLXArray([Float](repeating: 1.0, count: 6), [2, 3])
    // w2c (row-major): rotation I, tz = 0.02
    let viewmat = MLXArray([
        Float(1), 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0.02,
        0, 0, 0, 1,
    ], [16])
    let intrinsics = MLXArray([Float(500), 500, 256, 256], [4])
    let imageDims = MLXArray([Float(512), 512], [2])
    let opacities = MLXArray([Float(0.5), 0.5], [2])
    eval(means3dWorld, quats, scales, viewmat, intrinsics, imageDims, opacities)

    let (_, _, depths, radii) = try slangProjectToScreenFwd(
        means3dWorld: means3dWorld, quats: quats, scales: scales,
        viewmat: viewmat, intrinsics: intrinsics,
        imageDims: imageDims, opacities: opacities
    )
    eval(depths, radii)
    let r = radii.asArray(Int32.self)
    let d = depths.asArray(Float.self)

    // #0 is culled by the max-radius guard even though it is in front (depth > 0).
    #expect(d[0] > 0, "near gaussian should be in front of camera, depth=\(d[0])")
    #expect(r[0] == 0 && r[1] == 0,
        "degenerate near-camera gaussian must be culled, radii=(\(r[0]),\(r[1]))")
    // #1 survives with a finite radius.
    #expect(r[2] > 0 && r[3] > 0, "far gaussian wrongly culled, radii=(\(r[2]),\(r[3]))")
    #expect(r[2] <= 4 * 512 && r[3] <= 4 * 512, "far gaussian radius unexpectedly huge")
}

@Test
func projectToScreenBackwardMatchesFiniteDifferences() throws {
    let n = 3
    let scene = sampleProjectScene(seed: 1, n: n)
    MLXRandom.seed(31)
    // Pixel-scale cotangents for means2d / conic.
    let cotM2 = MLXRandom.normal([n, 2]) * 0.01
    let cotC  = MLXRandom.normal([n, 3]) * 0.001
    eval(cotM2, cotC)

    let (gM, gQ, gS) = try slangProjectToScreenBwd(
        means3dWorld: scene.means3dWorld,
        quats: scene.quats, scales: scene.scales,
        viewmat: scene.viewmat, intrinsics: scene.intrinsics,
        imageDims: scene.imageDims,
        cotMeans2d: cotM2, cotConics: cotC
    )
    eval(gM, gQ, gS)
    let gMH = gM.asArray(Float.self)
    let gQH = gQ.asArray(Float.self)
    let gSH = gS.asArray(Float.self)

    let mHost = scene.means3dWorld.asArray(Float.self)
    let qHost = scene.quats.asArray(Float.self)
    let sHost = scene.scales.asArray(Float.self)
    let cotM2H = cotM2.asArray(Float.self)
    let cotCH = cotC.asArray(Float.self)
    let fx = scene.intrinsicsHost[0]
    let fy = scene.intrinsicsHost[1]
    let cx = scene.intrinsicsHost[2]
    let cy = scene.intrinsicsHost[3]

    func loss(means: [Float], quats: [Float], scales: [Float]) -> Float {
        let m = MLXArray(means, [n, 3])
        let q = MLXArray(quats, [n, 4])
        let s = MLXArray(scales, [n, 3])
        let (m2, c, _, _) = referenceProjectToScreen(
            means3dWorld: m, quats: q, scales: s,
            viewmat: scene.viewmatHost,
            fx: fx, fy: fy, cx: cx, cy: cy,
            imageWidth: scene.imageWidth, imageHeight: scene.imageHeight
        )
        let m2h = m2.asArray(Float.self)
        let ch  = c.asArray(Float.self)
        var sum: Float = 0
        for i in 0..<n {
            sum += cotM2H[i * 2 + 0] * m2h[i * 2 + 0]
            sum += cotM2H[i * 2 + 1] * m2h[i * 2 + 1]
            sum += cotCH[i * 3 + 0] * ch[i * 3 + 0]
            sum += cotCH[i * 3 + 1] * ch[i * 3 + 1]
            sum += cotCH[i * 3 + 2] * ch[i * 3 + 2]
        }
        return sum
    }

    let eps: Float = 1e-3

    func checkGrad(
        primalCount: Int, primalHost: [Float], gradHost: [Float],
        loss perturbedLoss: (_ idx: Int, _ delta: Float) -> Float,
        label: String
    ) {
        var maxAbs: Float = 0, maxRel: Float = 0
        for idx in 0..<primalCount {
            let lp = perturbedLoss(idx, +eps)
            let lm = perturbedLoss(idx, -eps)
            let numerical = (lp - lm) / (2 * eps)
            let analytical = gradHost[idx]
            let absDiff = abs(numerical - analytical)
            let scale = max(abs(numerical), abs(analytical), 1e-4)
            maxAbs = max(maxAbs, absDiff)
            maxRel = max(maxRel, absDiff / scale)
        }
        #expect(
            maxAbs < 5e-3 || maxRel < 5e-2,
            "\(label) grad off: maxAbs=\(maxAbs) maxRel=\(maxRel)"
        )
    }

    // gradient of means3d_world
    var pm = mHost
    checkGrad(
        primalCount: n * 3, primalHost: mHost, gradHost: gMH,
        loss: { idx, delta in
            pm[idx] = mHost[idx] + delta
            defer { pm[idx] = mHost[idx] }
            return loss(means: pm, quats: qHost, scales: sHost)
        },
        label: "project_to_screen means3d_world"
    )

    // gradient of quats
    var pq = qHost
    checkGrad(
        primalCount: n * 4, primalHost: qHost, gradHost: gQH,
        loss: { idx, delta in
            pq[idx] = qHost[idx] + delta
            defer { pq[idx] = qHost[idx] }
            return loss(means: mHost, quats: pq, scales: sHost)
        },
        label: "project_to_screen quats"
    )

    // gradient of scales
    var ps = sHost
    checkGrad(
        primalCount: n * 3, primalHost: sHost, gradHost: gSH,
        loss: { idx, delta in
            ps[idx] = sHost[idx] + delta
            defer { ps[idx] = sHost[idx] }
            return loss(means: mHost, quats: qHost, scales: ps)
        },
        label: "project_to_screen scales"
    )
}
