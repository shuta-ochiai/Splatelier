// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXRandom
import Testing
@testable import Trainer

// shN holds R rest bands per Gaussian (R = 3, 8 or 15 for degree 1, 2 or 3); the kernel gets
// (degree, R) in degree_buf. The reference (referenceShEvalDeg3) takes all 16 bands and ignores
// the ones above `degree`, so bands past R never matter as long as degree ≤ the degree R holds.

/// Rest bands for each foreground SH degree the model can carry.
private let restBandsByDegree = [1: 3, 2: 8, 3: 15]

// Split coeffs [n,16,3] into the kernel's sh0 [n*3] (band 0) + shN [n*R*3] (bands 1...R).
private func splitSH(_ coeffs: MLXArray, n: Int, rest: Int) -> (sh0: MLXArray, shN: MLXArray) {
    let flat = coeffs.reshaped([n, 48])
    return (flat[0..., 0..<3].reshaped([n * 3]),
            flat[0..., 3..<(3 + rest * 3)].reshaped([n * rest * 3]))
}

// Put the kernel's grad_sh0 [n,3] + grad_shN [n,R*3] back into [n,16,3] for comparison with the
// reference (bands past R are 0).
private func padGradTo16Bands(gradSh0: MLXArray, gradShN: MLXArray, n: Int, rest: Int) -> MLXArray {
    var parts = [gradSh0.reshaped([n, 3]), gradShN.reshaped([n, rest * 3])]
    if rest < 15 { parts.append(MLXArray.zeros([n, (15 - rest) * 3])) }
    return MLX.concatenated(parts, axis: 1).reshaped([n, 16, 3])
}

private func slangShEvalFwd(
    degree: Int, dirs: MLXArray, coeffs: MLXArray, rest: Int = 3
) throws -> MLXArray {
    let n = dirs.shape[0]
    let kernel = try SlangCompiledKernel.load(name: "sh_eval_deg3_fwd")
    let degreeBuf = MLXArray([Int32(degree), Int32(rest)], [2])
    let nBuf = MLXArray([Int32(n)], [1])
    // All Gaussians visible (radii > 0), no cull.
    let radii = MLXArray([Int32](repeating: 1, count: n * 2), [n * 2])
    let (sh0, shN) = splitSH(coeffs, n: n, rest: rest)
    let outputs = kernel(
        [dirs, sh0, shN, degreeBuf, nBuf, radii],
        grid: (((n + 63) / 64) * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[n * 3]],
        outputDTypes: [.float32]
    )
    return outputs[0].reshaped([n, 3])
}

private func slangShEvalBwd(
    degree: Int, dirs: MLXArray, coeffs: MLXArray, cotColor: MLXArray, rest: Int = 3
) throws -> (gradDirs: MLXArray, gradCoeffs: MLXArray) {
    let n = dirs.shape[0]
    let kernel = try SlangCompiledKernel.load(name: "sh_eval_deg3_bwd")
    let degreeBuf = MLXArray([Int32(degree), Int32(rest)], [2])
    let nBuf = MLXArray([Int32(n)], [1])
    // All Gaussians visible (radii > 0), no cull.
    let radii = MLXArray([Int32](repeating: 1, count: n * 2), [n * 2])
    let (sh0, shN) = splitSH(coeffs, n: n, rest: rest)
    let outputs = kernel(
        [dirs, sh0, shN, degreeBuf, nBuf, cotColor, radii],
        grid: (((n + 63) / 64) * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[n * 3], [n * 3], [n * rest * 3]],
        outputDTypes: [.float32, .float32, .float32]
    )
    let gradCoeffs = padGradTo16Bands(gradSh0: outputs[1], gradShN: outputs[2], n: n, rest: rest)
    return (outputs[0].reshaped([n, 3]), gradCoeffs)
}

private func sampleShInputs(seed: UInt64, n: Int) -> (dirs: MLXArray, coeffs: MLXArray) {
    MLXRandom.seed(seed)
    // Keep directions away from the origin so rsqrt is stable.
    let dirs = MLXRandom.normal([n, 3]) * 1.5 + 0.4
    // Small coefficients so finite differences keep the post-+0.5 color above the clamp threshold.
    let c = MLXRandom.normal([n, 16, 3]) * 0.05
    eval(dirs, c)
    return (dirs, c)
}

@Test(arguments: [1, 2, 3])
func shEvalForwardMatchesMLXReference(degree: Int) throws {
    let n = 6
    let inputs = sampleShInputs(seed: 0, n: n)

    let slangColor = try slangShEvalFwd(
        degree: degree, dirs: inputs.dirs, coeffs: inputs.coeffs, rest: restBandsByDegree[degree]!)
    // The kernel evaluates in fp32 and stores in half, so round the reference to fp16 before
    // comparing (tol 1e-3 tolerates rounding boundaries yet catches SH formula bugs (>>1e-2)).
    let refColor = referenceShEvalDeg3(degree: degree, dirs: inputs.dirs, coeffs: inputs.coeffs)
        .asType(.float16).asType(.float32)
    eval(slangColor, refColor)

    let diff = MLX.abs(slangColor - refColor).max().item(Float.self)
    #expect(diff < 1e-3, "sh_eval forward diverged at degree \(degree) by \(diff)")
}

/// The active degree can be below what shN holds (the warm-up raises it during training).
@Test(arguments: [3, 8, 15])
func shEvalForwardRespectsLowerDegrees(rest: Int) throws {
    let n = 5
    let inputs = sampleShInputs(seed: 5, n: n)
    let held = rest == 15 ? 3 : rest == 8 ? 2 : 1

    for degree in 0...held {
        let slangColor = try slangShEvalFwd(
            degree: degree, dirs: inputs.dirs, coeffs: inputs.coeffs, rest: rest)
        let refColor = referenceShEvalDeg3(degree: degree, dirs: inputs.dirs, coeffs: inputs.coeffs)
            .asType(.float16).asType(.float32)
        eval(slangColor, refColor)
        let diff = MLX.abs(slangColor - refColor).max().item(Float.self)
        #expect(diff < 1e-3, "sh_eval forward diverged for degree \(degree), R \(rest) by \(diff)")
    }
}

@Test(arguments: [1, 2, 3])
func shEvalBackwardMatchesFiniteDifferences(degree: Int) throws {
    let n = 3
    let rest = restBandsByDegree[degree]!
    let inputs = sampleShInputs(seed: 7, n: n)
    MLXRandom.seed(17)
    let cotColor = MLXRandom.normal([n, 3]) * 0.5
    eval(cotColor)

    let (gDirs, gCoeffs) = try slangShEvalBwd(
        degree: degree, dirs: inputs.dirs, coeffs: inputs.coeffs, cotColor: cotColor, rest: rest
    )
    eval(gDirs, gCoeffs)
    let gD = gDirs.asArray(Float.self)
    let gC = gCoeffs.asArray(Float.self)

    let dHost = inputs.dirs.asArray(Float.self)
    let cHost = inputs.coeffs.asArray(Float.self)
    let cotHost = cotColor.asArray(Float.self)

    func loss(dirs: [Float], coeffs: [Float]) -> Float {
        let d = MLXArray(dirs, [n, 3])
        let c = MLXArray(coeffs, [n, 16, 3])
        let color = referenceShEvalDeg3(degree: degree, dirs: d, coeffs: c).asArray(Float.self)
        var sum: Float = 0
        for i in 0..<n {
            for k in 0..<3 { sum += cotHost[i * 3 + k] * color[i * 3 + k] }
        }
        return sum
    }

    let eps: Float = 1e-3

    var perturbedD = dHost
    var maxAbsD: Float = 0, maxRelD: Float = 0
    for i in 0..<n {
        for k in 0..<3 {
            perturbedD[i * 3 + k] = dHost[i * 3 + k] + eps
            let lp = loss(dirs: perturbedD, coeffs: cHost)
            perturbedD[i * 3 + k] = dHost[i * 3 + k] - eps
            let lm = loss(dirs: perturbedD, coeffs: cHost)
            perturbedD[i * 3 + k] = dHost[i * 3 + k]
            let numerical = (lp - lm) / (2 * eps)
            let analytical = gD[i * 3 + k]
            let absDiff = abs(numerical - analytical)
            let scale = max(abs(numerical), abs(analytical), 1e-4)
            maxAbsD = max(maxAbsD, absDiff)
            maxRelD = max(maxRelD, absDiff / scale)
        }
    }
    #expect(
        maxAbsD < 5e-3 || maxRelD < 5e-2,
        "sh_eval grad w.r.t. dirs off: maxAbs=\(maxAbsD) maxRel=\(maxRelD)"
    )

    // FD-check bands 0...R (= the first (R+1)*3 coeffs), which contribute at this degree.
    var perturbedC = cHost
    var maxAbsC: Float = 0, maxRelC: Float = 0
    for i in 0..<n {
        for k in 0..<((rest + 1) * 3) {
            perturbedC[i * 48 + k] = cHost[i * 48 + k] + eps
            let lp = loss(dirs: dHost, coeffs: perturbedC)
            perturbedC[i * 48 + k] = cHost[i * 48 + k] - eps
            let lm = loss(dirs: dHost, coeffs: perturbedC)
            perturbedC[i * 48 + k] = cHost[i * 48 + k]
            let numerical = (lp - lm) / (2 * eps)
            let analytical = gC[i * 48 + k]
            let absDiff = abs(numerical - analytical)
            let scale = max(abs(numerical), abs(analytical), 1e-4)
            maxAbsC = max(maxAbsC, absDiff)
            maxRelC = max(maxRelC, absDiff / scale)
        }
    }
    #expect(
        maxAbsC < 5e-3 || maxRelC < 5e-2,
        "sh_eval grad w.r.t. coeffs off: maxAbs=\(maxAbsC) maxRel=\(maxRelC)"
    )
}

@Test(arguments: [1, 3])
func shEvalSkipsCulledGaussians(degree: Int) throws {
    // #0 visible, #1 culled (radii 0). #1's color and gradient are 0; #0 matches the no-cull
    // result.
    let n = 2
    let rest = restBandsByDegree[degree]!
    let per = (rest + 1) * 3   // coefficients per Gaussian
    let dirs = MLXArray([Float(0.3), 0.5, 1.2, -0.4, 0.8, 0.9], [n, 3])
    let coeffs = MLXRandom.normal([n, 16, 3]) * 0.1
    eval(dirs, coeffs)
    let radii = MLXArray([Int32(1), 1, 0, 0], [n * 2])   // #0 visible, #1 culled
    let cot = MLXArray([Float](repeating: 0.5, count: n * 3), [n * 3])

    let (sh0, shN) = splitSH(coeffs, n: n, rest: rest)
    let fwdKernel = try SlangCompiledKernel.load(name: "sh_eval_deg3_fwd")
    let fwd = fwdKernel(
        [dirs, sh0, shN, MLXArray([Int32(degree), Int32(rest)], [2]),
         MLXArray([Int32(n)], [1]), radii],
        grid: (64, 1, 1), threadGroup: (64, 1, 1),
        outputShapes: [[n * 3]], outputDTypes: [.float32]
    )
    let color = fwd[0].reshaped([n, 3])
    eval(color)
    let cHost = color.asArray(Float.self)
    // Matches #0's color computed without cull.
    let refColor = try slangShEvalFwd(degree: degree, dirs: dirs, coeffs: coeffs, rest: rest)
    eval(refColor)
    let refHost = refColor.asArray(Float.self)
    #expect(abs(cHost[0] - refHost[0]) < 1e-5 && abs(cHost[1] - refHost[1]) < 1e-5
            && abs(cHost[2] - refHost[2]) < 1e-5, "visible gaussian #0 color changed by cull gating")
    #expect(cHost[3] == 0 && cHost[4] == 0 && cHost[5] == 0, "culled gaussian #1 must be 0")

    let bwdKernel = try SlangCompiledKernel.load(name: "sh_eval_deg3_bwd")
    let bwd = bwdKernel(
        [dirs, sh0, shN, MLXArray([Int32(degree), Int32(rest)], [2]),
         MLXArray([Int32(n)], [1]), cot, radii],
        grid: (64, 1, 1), threadGroup: (64, 1, 1),
        outputShapes: [[n * 3], [n * 3], [n * rest * 3]], outputDTypes: [.float32, .float32, .float32]
    )
    let gradDirs = bwd[0].reshaped([n, 3])
    // Concatenate grad_sh0 [n,3] + grad_shN [n,R*3] into [n,(R+1)*3].
    let gradCoeffs = MLX.concatenated(
        [bwd[1].reshaped([n, 3]), bwd[2].reshaped([n, rest * 3])], axis: 1
    )
    eval(gradDirs, gradCoeffs)
    let gd = gradDirs.asArray(Float.self)
    let gc = gradCoeffs.asArray(Float.self)
    // #1 (culled) gradients are all 0.
    for k in 0..<3 { #expect(gd[3 + k] == 0, "culled gaussian #1 gradDirs must be 0") }
    for k in 0..<per { #expect(gc[per + k] == 0, "culled gaussian #1 gradCoeffs must be 0") }
    // #0 (visible) has non-zero values up to its last band (evidence it was actually computed).
    let g0nonzero = (0..<3).contains { abs(gc[per - 3 + $0]) > 1e-7 }
    #expect(g0nonzero, "visible gaussian #0 gradCoeffs should be computed (nonzero)")
}

/// At degree 0 the backward writes exact zeros (+0) for shN even with nonzero rest bands, which
/// lets GaussianTrainer skip shN's Adam step while SH is at degree 0 (zero grad and zero moments
/// leave shN unchanged; see fusedAdamZeroGradLeavesParamExactlyUnchanged).
@Test(arguments: [3, 8, 15])
func shEvalBwdDegree0WritesExactZeroForShN(rest: Int) throws {
    let n = 257
    let inputs = sampleShInputs(seed: 11, n: n)
    MLXRandom.seed(12)
    let cotColor = MLXRandom.normal([n, 3])
    let (_, gradCoeffs) = try slangShEvalBwd(
        degree: 0, dirs: inputs.dirs, coeffs: inputs.coeffs, cotColor: cotColor, rest: rest)
    let gradShN = gradCoeffs[0..., 1..<(1 + rest), 0...]
    eval(gradShN)
    let values = gradShN.asArray(Float.self)
    #expect(values.count == n * rest * 3)
    #expect(values.allSatisfy { $0.bitPattern == 0 }, "degree-0 shN gradient must be +0")
}
