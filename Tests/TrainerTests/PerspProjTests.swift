// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXRandom
import Testing
@testable import Trainer

private func slangPerspProjFwd(
    means3dCam: MLXArray, cov3dsCam: MLXArray,
    intrinsics: MLXArray, imageDims: MLXArray
) throws -> (means2d: MLXArray, cov2ds: MLXArray) {
    let n = means3dCam.shape[0]
    let kernel = try SlangCompiledKernel.load(name: "persp_proj_fwd")
    let outputs = kernel(
        [means3dCam, cov3dsCam, intrinsics, imageDims],
        grid: (n, 1, 1),
        threadGroup: (1, 1, 1),
        outputShapes: [[n * 2], [n * 3]],
        outputDTypes: [.float32, .float32]
    )
    return (outputs[0].reshaped([n, 2]), outputs[1].reshaped([n, 3]))
}

private func slangPerspProjBwd(
    means3dCam: MLXArray, cov3dsCam: MLXArray, intrinsics: MLXArray,
    imageDims: MLXArray,
    cotMean2d: MLXArray, cotCov2d: MLXArray
) throws -> (gradMean3d: MLXArray, gradCov3d: MLXArray) {
    let n = means3dCam.shape[0]
    let kernel = try SlangCompiledKernel.load(name: "persp_proj_bwd")
    let outputs = kernel(
        [means3dCam, cov3dsCam, intrinsics, imageDims, cotMean2d, cotCov2d],
        grid: (n, 1, 1),
        threadGroup: (1, 1, 1),
        outputShapes: [[n * 3], [n * 6]],
        outputDTypes: [.float32, .float32]
    )
    return (outputs[0].reshaped([n, 3]), outputs[1].reshaped([n, 6]))
}

/// Inputs with Gaussians placed well in front of the camera so the projection is numerically
/// stable.
private struct PerspProjScene {
    let means3d: MLXArray
    let cov3ds: MLXArray
    let intrinsics: MLXArray
    let intrinsicsHost: [Float]
    let imageDims: MLXArray
    let imageWidth: Float
    let imageHeight: Float
}

private func sampleScene(seed: UInt64, n: Int) -> PerspProjScene {
    MLXRandom.seed(seed)
    let x = MLXRandom.normal([n, 1]) * 0.5
    let y = MLXRandom.normal([n, 1]) * 0.5
    let z = MLX.exp(MLXRandom.normal([n, 1]) * 0.2) * 3.0   // z ≳ 2, away from the pinhole
    let means3d = MLX.concatenated([x, y, z], axis: 1)

    // Build cov3d from (quats, scales) so it is PSD.
    let quats = MLXRandom.normal([n, 4]) * 0.8 + 0.3
    let scales = MLX.exp(MLXRandom.normal([n, 3]) * 0.2)
    let cov3ds = referenceQuatScaleToCov3d(quats, scales)

    let intrinsicsHost: [Float] = [500.0, 500.0, 250.0, 250.0]
    let intrinsics = MLXArray(intrinsicsHost, [4])
    let imageWidth: Float = 500
    let imageHeight: Float = 500
    let imageDims = MLXArray([imageWidth, imageHeight], [2])
    eval(means3d, cov3ds, intrinsics, imageDims)
    return PerspProjScene(
        means3d: means3d, cov3ds: cov3ds,
        intrinsics: intrinsics, intrinsicsHost: intrinsicsHost,
        imageDims: imageDims,
        imageWidth: imageWidth, imageHeight: imageHeight
    )
}

// MARK: - Forward match

@Test
func perspProjForwardMatchesMLXReference() throws {
    let n = 6
    let scene = sampleScene(seed: 0, n: n)

    let (slangMean2d, slangCov2d) = try slangPerspProjFwd(
        means3dCam: scene.means3d,
        cov3dsCam: scene.cov3ds,
        intrinsics: scene.intrinsics,
        imageDims: scene.imageDims
    )
    let (refMean2d, refCov2d) = referencePerspProj(
        means3dCam: scene.means3d,
        cov3dsCam: scene.cov3ds,
        fx: scene.intrinsicsHost[0], fy: scene.intrinsicsHost[1],
        cx: scene.intrinsicsHost[2], cy: scene.intrinsicsHost[3],
        imageWidth: scene.imageWidth, imageHeight: scene.imageHeight
    )
    eval(slangMean2d, slangCov2d, refMean2d, refCov2d)

    let diffMean = MLX.abs(slangMean2d - refMean2d).max().item(Float.self)
    let diffCov  = MLX.abs(slangCov2d - refCov2d).max().item(Float.self)
    // cov2d is in pixel² units (~focal²/depth²), around 1e3-1e4, so compare by relative error.
    let refScaleCov = MLX.abs(refCov2d).max().item(Float.self)
    let relCov = diffCov / max(refScaleCov, 1e-6)
    #expect(diffMean < 1e-3, "persp_proj mean2d diverged by \(diffMean)")
    #expect(relCov   < 1e-5, "persp_proj cov2d relative diff \(relCov) (abs \(diffCov), scale \(refScaleCov))")
}

// MARK: - Backward match (finite differences)

@Test
func perspProjBackwardMatchesFiniteDifferences() throws {
    let n = 3
    let scene = sampleScene(seed: 1, n: n)
    MLXRandom.seed(11)
    let cotMean2d = MLXRandom.normal([n, 2]) * 0.01      // pixel scale
    let cotCov2d  = MLXRandom.normal([n, 3]) * 0.0001    // pixel² scale
    eval(cotMean2d, cotCov2d)

    let (gradMean3d, gradCov3d) = try slangPerspProjBwd(
        means3dCam: scene.means3d, cov3dsCam: scene.cov3ds,
        intrinsics: scene.intrinsics, imageDims: scene.imageDims,
        cotMean2d: cotMean2d, cotCov2d: cotCov2d
    )
    eval(gradMean3d, gradCov3d)
    let gM = gradMean3d.asArray(Float.self)
    let gC = gradCov3d.asArray(Float.self)

    let m3dHost = scene.means3d.asArray(Float.self)
    let c3dHost = scene.cov3ds.asArray(Float.self)
    let cotM = cotMean2d.asArray(Float.self)
    let cotC = cotCov2d.asArray(Float.self)
    let fx = scene.intrinsicsHost[0]
    let fy = scene.intrinsicsHost[1]
    let cx = scene.intrinsicsHost[2]
    let cy = scene.intrinsicsHost[3]

    func loss(means3d: [Float], cov3d: [Float]) -> Float {
        let m = MLXArray(means3d, [n, 3])
        let c = MLXArray(cov3d, [n, 6])
        let (m2, c2) = referencePerspProj(means3dCam: m, cov3dsCam: c,
                                          fx: fx, fy: fy, cx: cx, cy: cy,
                                          imageWidth: scene.imageWidth,
                                          imageHeight: scene.imageHeight)
        let m2h = m2.asArray(Float.self)
        let c2h = c2.asArray(Float.self)
        var sum: Float = 0
        for i in 0..<n {
            sum += cotM[i * 2 + 0] * m2h[i * 2 + 0]
            sum += cotM[i * 2 + 1] * m2h[i * 2 + 1]
            sum += cotC[i * 3 + 0] * c2h[i * 3 + 0]
            sum += cotC[i * 3 + 1] * c2h[i * 3 + 1]
            sum += cotC[i * 3 + 2] * c2h[i * 3 + 2]
        }
        return sum
    }

    let epsM: Float = 1e-3
    var perturbedM = m3dHost
    var maxAbsM: Float = 0, maxRelM: Float = 0
    for i in 0..<n {
        for k in 0..<3 {
            perturbedM[i * 3 + k] = m3dHost[i * 3 + k] + epsM
            let lp = loss(means3d: perturbedM, cov3d: c3dHost)
            perturbedM[i * 3 + k] = m3dHost[i * 3 + k] - epsM
            let lm = loss(means3d: perturbedM, cov3d: c3dHost)
            perturbedM[i * 3 + k] = m3dHost[i * 3 + k]
            let numerical = (lp - lm) / (2 * epsM)
            let analytical = gM[i * 3 + k]
            let absDiff = abs(numerical - analytical)
            let scale = max(abs(numerical), abs(analytical), 1e-6)
            maxAbsM = max(maxAbsM, absDiff)
            maxRelM = max(maxRelM, absDiff / scale)
        }
    }
    #expect(
        maxAbsM < 5e-3 || maxRelM < 5e-2,
        "persp_proj grad w.r.t. mean3d off: maxAbs=\(maxAbsM) maxRel=\(maxRelM)"
    )

    let epsC: Float = 1e-3
    var perturbedC = c3dHost
    var maxAbsC: Float = 0, maxRelC: Float = 0
    for i in 0..<n {
        for k in 0..<6 {
            perturbedC[i * 6 + k] = c3dHost[i * 6 + k] + epsC
            let lp = loss(means3d: m3dHost, cov3d: perturbedC)
            perturbedC[i * 6 + k] = c3dHost[i * 6 + k] - epsC
            let lm = loss(means3d: m3dHost, cov3d: perturbedC)
            perturbedC[i * 6 + k] = c3dHost[i * 6 + k]
            let numerical = (lp - lm) / (2 * epsC)
            let analytical = gC[i * 6 + k]
            let absDiff = abs(numerical - analytical)
            let scale = max(abs(numerical), abs(analytical), 1e-6)
            maxAbsC = max(maxAbsC, absDiff)
            maxRelC = max(maxRelC, absDiff / scale)
        }
    }
    #expect(
        maxAbsC < 5e-3 || maxRelC < 5e-2,
        "persp_proj grad w.r.t. cov3d off: maxAbs=\(maxAbsC) maxRel=\(maxRelC)"
    )
}
