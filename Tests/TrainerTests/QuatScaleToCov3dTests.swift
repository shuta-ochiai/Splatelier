// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXRandom
import Testing
@testable import Trainer

// MARK: - Kernel dispatch helpers

private func slangQuatScaleToCov3dFwd(quats: MLXArray, scales: MLXArray) throws -> MLXArray {
    let n = quats.shape[0]
    let kernel = try SlangCompiledKernel.load(name: "quat_scale_to_cov3d_fwd")
    let outputs = kernel(
        [quats, scales],
        grid: (n, 1, 1),
        threadGroup: (1, 1, 1),
        outputShapes: [[n * 6]],
        outputDTypes: [.float32]
    )
    return outputs[0].reshaped([n, 6])
}

private func slangQuatScaleToCov3dBwd(
    quats: MLXArray, scales: MLXArray, cotCov: MLXArray
) throws -> (gradQuats: MLXArray, gradScales: MLXArray) {
    let n = quats.shape[0]
    let kernel = try SlangCompiledKernel.load(name: "quat_scale_to_cov3d_bwd")
    let outputs = kernel(
        [quats, scales, cotCov],
        grid: (n, 1, 1),
        threadGroup: (1, 1, 1),
        outputShapes: [[n * 4], [n * 3]],
        outputDTypes: [.float32, .float32]
    )
    return (outputs[0].reshaped([n, 4]), outputs[1].reshaped([n, 3]))
}

// MARK: - Forward match

@Test
func quatScaleToCov3dForwardMatchesMLXReference() throws {
    MLXRandom.seed(0)
    let n = 8
    let quats = MLXRandom.normal([n, 4]) * 1.4 + 0.2
    // Positive scales, as in the model (parameterized in log space).
    let scales = MLX.exp(MLXRandom.normal([n, 3]) * 0.3)
    eval(quats, scales)

    let slangC = try slangQuatScaleToCov3dFwd(quats: quats, scales: scales)
    let refC = referenceQuatScaleToCov3d(quats, scales)
    eval(slangC, refC)

    let diff = MLX.abs(slangC - refC).max().item(Float.self)
    #expect(diff < 1e-5, "Slang quat_scale_to_cov3d diverged from MLX reference by \(diff)")
}

// MARK: - Backward match (finite differences)

@Test
func quatScaleToCov3dBackwardMatchesFiniteDifferences() throws {
    MLXRandom.seed(2)
    let n = 4
    let quats = MLXRandom.normal([n, 4]) * 0.8 + 0.3
    let scales = MLX.exp(MLXRandom.normal([n, 3]) * 0.3)
    let cotCov = MLXRandom.normal([n, 6])
    eval(quats, scales, cotCov)

    let (gradQ, gradS) = try slangQuatScaleToCov3dBwd(
        quats: quats, scales: scales, cotCov: cotCov
    )
    eval(gradQ, gradS)
    let gradQHost = gradQ.asArray(Float.self)
    let gradSHost = gradS.asArray(Float.self)

    let quatsHost = quats.asArray(Float.self)
    let scalesHost = scales.asArray(Float.self)
    let cotHost = cotCov.asArray(Float.self)

    let epsValue: Float = 1e-3

    // L = Σ cot[i,k]·C(q, s)[i,k]
    func lossFor(quatsArr: [Float], scalesArr: [Float]) -> Float {
        let qMLX = MLXArray(quatsArr, [n, 4])
        let sMLX = MLXArray(scalesArr, [n, 3])
        let c = referenceQuatScaleToCov3d(qMLX, sMLX).asArray(Float.self)
        var sum: Float = 0
        for idx in 0..<(n * 6) {
            sum += cotHost[idx] * c[idx]
        }
        return sum
    }

    var maxAbsDiffQ: Float = 0
    var maxRelDiffQ: Float = 0
    var perturbed = quatsHost
    for i in 0..<n {
        for k in 0..<4 {
            perturbed[i * 4 + k] = quatsHost[i * 4 + k] + epsValue
            let plus = lossFor(quatsArr: perturbed, scalesArr: scalesHost)
            perturbed[i * 4 + k] = quatsHost[i * 4 + k] - epsValue
            let minus = lossFor(quatsArr: perturbed, scalesArr: scalesHost)
            perturbed[i * 4 + k] = quatsHost[i * 4 + k]
            let numerical = (plus - minus) / (2 * epsValue)
            let analytical = gradQHost[i * 4 + k]
            let absDiff = abs(numerical - analytical)
            let scale = max(abs(numerical), abs(analytical), 1e-3)
            maxAbsDiffQ = max(maxAbsDiffQ, absDiff)
            maxRelDiffQ = max(maxRelDiffQ, absDiff / scale)
        }
    }
    #expect(
        maxAbsDiffQ < 1e-2 || maxRelDiffQ < 1e-2,
        "cov3d grad w.r.t. quats off: maxAbs=\(maxAbsDiffQ) maxRel=\(maxRelDiffQ)"
    )

    var maxAbsDiffS: Float = 0
    var maxRelDiffS: Float = 0
    var perturbedS = scalesHost
    for i in 0..<n {
        for k in 0..<3 {
            perturbedS[i * 3 + k] = scalesHost[i * 3 + k] + epsValue
            let plus = lossFor(quatsArr: quatsHost, scalesArr: perturbedS)
            perturbedS[i * 3 + k] = scalesHost[i * 3 + k] - epsValue
            let minus = lossFor(quatsArr: quatsHost, scalesArr: perturbedS)
            perturbedS[i * 3 + k] = scalesHost[i * 3 + k]
            let numerical = (plus - minus) / (2 * epsValue)
            let analytical = gradSHost[i * 3 + k]
            let absDiff = abs(numerical - analytical)
            let scale = max(abs(numerical), abs(analytical), 1e-3)
            maxAbsDiffS = max(maxAbsDiffS, absDiff)
            maxRelDiffS = max(maxRelDiffS, absDiff / scale)
        }
    }
    #expect(
        maxAbsDiffS < 1e-2 || maxRelDiffS < 1e-2,
        "cov3d grad w.r.t. scales off: maxAbs=\(maxAbsDiffS) maxRel=\(maxRelDiffS)"
    )
}
