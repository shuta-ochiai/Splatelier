// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXRandom
import Testing
@testable import Trainer

private func slangWorldToCamFwd(
    means3dWorld: MLXArray, cov3dsWorld: MLXArray, viewmat: MLXArray
) throws -> (means3dCam: MLXArray, cov3dsCam: MLXArray) {
    let n = means3dWorld.shape[0]
    let kernel = try SlangCompiledKernel.load(name: "world_to_cam_fwd")
    let outputs = kernel(
        [means3dWorld, cov3dsWorld, viewmat],
        grid: (n, 1, 1),
        threadGroup: (1, 1, 1),
        outputShapes: [[n * 3], [n * 6]],
        outputDTypes: [.float32, .float32]
    )
    return (outputs[0].reshaped([n, 3]), outputs[1].reshaped([n, 6]))
}

private func slangWorldToCamBwd(
    means3dWorld: MLXArray, cov3dsWorld: MLXArray, viewmat: MLXArray,
    cotMeans3dCam: MLXArray, cotCov3dsCam: MLXArray
) throws -> (gradMeans3dWorld: MLXArray, gradCov3dsWorld: MLXArray) {
    let n = means3dWorld.shape[0]
    let kernel = try SlangCompiledKernel.load(name: "world_to_cam_bwd")
    let outputs = kernel(
        [means3dWorld, cov3dsWorld, viewmat, cotMeans3dCam, cotCov3dsCam],
        grid: (n, 1, 1),
        threadGroup: (1, 1, 1),
        outputShapes: [[n * 3], [n * 6]],
        outputDTypes: [.float32, .float32]
    )
    return (outputs[0].reshaped([n, 3]), outputs[1].reshaped([n, 6]))
}

private func sampleScene(seed: UInt64, n: Int) -> (
    means3dWorld: MLXArray, cov3dsWorld: MLXArray, viewmat: MLXArray, viewmatHost: [Float]
) {
    MLXRandom.seed(seed)
    let pts = MLXRandom.normal([n, 3]) * 0.7
    let quats = MLXRandom.normal([n, 4]) * 0.8 + 0.3
    let scales = MLX.exp(MLXRandom.normal([n, 3]) * 0.2)
    let cov3ds = referenceQuatScaleToCov3d(quats, scales)
    let view = makeViewMatrix(seed: seed &+ 1000)
    eval(pts, cov3ds, view.mlx)
    return (pts, cov3ds, view.mlx, view.host)
}

@Test
func worldToCamForwardMatchesMLXReference() throws {
    let n = 6
    let scene = sampleScene(seed: 0, n: n)

    let (slangM, slangC) = try slangWorldToCamFwd(
        means3dWorld: scene.means3dWorld,
        cov3dsWorld: scene.cov3dsWorld,
        viewmat: scene.viewmat
    )
    let (refM, refC) = referenceWorldToCam(
        means3dWorld: scene.means3dWorld,
        cov3dsWorld: scene.cov3dsWorld,
        viewmat: scene.viewmatHost
    )
    eval(slangM, slangC, refM, refC)

    let diffM = MLX.abs(slangM - refM).max().item(Float.self)
    let diffC = MLX.abs(slangC - refC).max().item(Float.self)
    let scaleC = MLX.abs(refC).max().item(Float.self)
    #expect(diffM < 1e-5, "world_to_cam mean3d diverged by \(diffM)")
    #expect(diffC / max(scaleC, 1e-6) < 1e-5, "world_to_cam cov3d rel diff \(diffC / max(scaleC, 1e-6))")
}

@Test
func worldToCamBackwardMatchesFiniteDifferences() throws {
    let n = 3
    let scene = sampleScene(seed: 1, n: n)
    MLXRandom.seed(13)
    let cotM = MLXRandom.normal([n, 3])
    let cotC = MLXRandom.normal([n, 6])
    eval(cotM, cotC)

    let (gM, gC) = try slangWorldToCamBwd(
        means3dWorld: scene.means3dWorld,
        cov3dsWorld: scene.cov3dsWorld,
        viewmat: scene.viewmat,
        cotMeans3dCam: cotM,
        cotCov3dsCam: cotC
    )
    eval(gM, gC)
    let gMHost = gM.asArray(Float.self)
    let gCHost = gC.asArray(Float.self)

    let mHost = scene.means3dWorld.asArray(Float.self)
    let cHost = scene.cov3dsWorld.asArray(Float.self)
    let cotMH = cotM.asArray(Float.self)
    let cotCH = cotC.asArray(Float.self)

    func loss(meansW: [Float], covW: [Float]) -> Float {
        let m = MLXArray(meansW, [n, 3])
        let c = MLXArray(covW, [n, 6])
        let (mc, cc) = referenceWorldToCam(means3dWorld: m, cov3dsWorld: c, viewmat: scene.viewmatHost)
        let mh = mc.asArray(Float.self)
        let ch = cc.asArray(Float.self)
        var sum: Float = 0
        for i in 0..<n {
            for k in 0..<3 { sum += cotMH[i * 3 + k] * mh[i * 3 + k] }
            for k in 0..<6 { sum += cotCH[i * 6 + k] * ch[i * 6 + k] }
        }
        return sum
    }

    let eps: Float = 1e-3
    var perturbedM = mHost
    var maxAbsM: Float = 0, maxRelM: Float = 0
    for i in 0..<n {
        for k in 0..<3 {
            perturbedM[i * 3 + k] = mHost[i * 3 + k] + eps
            let lp = loss(meansW: perturbedM, covW: cHost)
            perturbedM[i * 3 + k] = mHost[i * 3 + k] - eps
            let lm = loss(meansW: perturbedM, covW: cHost)
            perturbedM[i * 3 + k] = mHost[i * 3 + k]
            let numerical = (lp - lm) / (2 * eps)
            let analytical = gMHost[i * 3 + k]
            let absDiff = abs(numerical - analytical)
            let scale = max(abs(numerical), abs(analytical), 1e-6)
            maxAbsM = max(maxAbsM, absDiff)
            maxRelM = max(maxRelM, absDiff / scale)
        }
    }
    #expect(
        maxAbsM < 1e-3 || maxRelM < 1e-2,
        "world_to_cam grad w.r.t. means3d_world off: maxAbs=\(maxAbsM) maxRel=\(maxRelM)"
    )

    var perturbedC = cHost
    var maxAbsC: Float = 0, maxRelC: Float = 0
    for i in 0..<n {
        for k in 0..<6 {
            perturbedC[i * 6 + k] = cHost[i * 6 + k] + eps
            let lp = loss(meansW: mHost, covW: perturbedC)
            perturbedC[i * 6 + k] = cHost[i * 6 + k] - eps
            let lm = loss(meansW: mHost, covW: perturbedC)
            perturbedC[i * 6 + k] = cHost[i * 6 + k]
            let numerical = (lp - lm) / (2 * eps)
            let analytical = gCHost[i * 6 + k]
            let absDiff = abs(numerical - analytical)
            let scale = max(abs(numerical), abs(analytical), 1e-6)
            maxAbsC = max(maxAbsC, absDiff)
            maxRelC = max(maxRelC, absDiff / scale)
        }
    }
    #expect(
        maxAbsC < 1e-3 || maxRelC < 1e-2,
        "world_to_cam grad w.r.t. cov3d_world off: maxAbs=\(maxAbsC) maxRel=\(maxRelC)"
    )
}
