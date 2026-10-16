// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXRandom
import Testing
@testable import Trainer

// MARK: - Kernel dispatch helpers

private func slangQuatToRotmatFwd(_ quats: MLXArray) throws -> MLXArray {
    let n = quats.shape[0]
    let kernel = try SlangCompiledKernel.load(name: "quat_to_rotmat_fwd")
    let outputs = kernel(
        [quats],
        grid: (n, 1, 1),
        threadGroup: (1, 1, 1),
        outputShapes: [[n * 9]],
        outputDTypes: [.float32]
    )
    return outputs[0].reshaped([n, 3, 3])
}

private func slangQuatToRotmatBwd(quats: MLXArray, cotR: MLXArray) throws -> MLXArray {
    let n = quats.shape[0]
    let kernel = try SlangCompiledKernel.load(name: "quat_to_rotmat_bwd")
    let outputs = kernel(
        [quats, cotR.reshaped([n * 9])],
        grid: (n, 1, 1),
        threadGroup: (1, 1, 1),
        outputShapes: [[n * 4]],
        outputDTypes: [.float32]
    )
    return outputs[0].reshaped([n, 4])
}

// MARK: - Forward match

@Test
func quatToRotmatForwardMatchesMLXReference() throws {
    MLXRandom.seed(0)
    let n = 8
    // Unnormalized quaternions so the rsqrt path is exercised.
    let quats = MLXRandom.normal([n, 4]) * 1.5 + 0.2
    eval(quats)

    let slangR = try slangQuatToRotmatFwd(quats)
    let refR = referenceQuatToRotmat(quats)
    eval(slangR, refR)

    let diff = MLX.abs(slangR - refR).max().item(Float.self)
    #expect(diff < 1e-5, "Slang quat_to_rotmat diverged from MLX reference by \(diff)")
}

// MARK: - Backward match (finite differences)

@Test
func quatToRotmatBackwardMatchesFiniteDifferences() throws {
    MLXRandom.seed(1)
    let n = 4
    let quats = MLXRandom.normal([n, 4]) * 0.8 + 0.3   // avoid zero norm
    let cotR = MLXRandom.normal([n, 3, 3])
    eval(quats, cotR)

    let analytical = try slangQuatToRotmatBwd(quats: quats, cotR: cotR)
    eval(analytical)

    // Central differences: dL/dq[i,k] of L = Σ_ab cotR[i,a,b]·R(q[i])[a,b].
    let epsValue: Float = 1e-3
    let quatsHost = quats.asArray(Float.self)
    let cotHost = cotR.asArray(Float.self)
    var numerical = [Float](repeating: 0, count: n * 4)
    var perturbed = quatsHost

    for i in 0..<n {
        for k in 0..<4 {
            perturbed[i * 4 + k] = quatsHost[i * 4 + k] + epsValue
            let plusR = referenceQuatToRotmat(MLXArray(perturbed, [n, 4])).asArray(Float.self)
            perturbed[i * 4 + k] = quatsHost[i * 4 + k] - epsValue
            let minusR = referenceQuatToRotmat(MLXArray(perturbed, [n, 4])).asArray(Float.self)
            perturbed[i * 4 + k] = quatsHost[i * 4 + k]

            var lossPlus: Float = 0
            var lossMinus: Float = 0
            for a in 0..<3 {
                for b in 0..<3 {
                    let cotVal = cotHost[i * 9 + a * 3 + b]
                    lossPlus  += cotVal * plusR[i * 9 + a * 3 + b]
                    lossMinus += cotVal * minusR[i * 9 + a * 3 + b]
                }
            }
            numerical[i * 4 + k] = (lossPlus - lossMinus) / (2 * epsValue)
        }
    }

    let analyticalHost = analytical.asArray(Float.self)
    var maxAbsDiff: Float = 0
    var maxRelDiff: Float = 0
    for idx in 0..<(n * 4) {
        let a = analyticalHost[idx]
        let num = numerical[idx]
        let absDiff = abs(a - num)
        let scale = max(abs(a), abs(num), 1e-3)
        maxAbsDiff = max(maxAbsDiff, absDiff)
        maxRelDiff = max(maxRelDiff, absDiff / scale)
    }
    #expect(
        maxAbsDiff < 1e-2 || maxRelDiff < 1e-2,
        "quat_to_rotmat backward grads diverged: maxAbs=\(maxAbsDiff), maxRel=\(maxRelDiff)"
    )
}
