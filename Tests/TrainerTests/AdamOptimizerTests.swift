// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import Testing
@testable import Trainer

// Fused Slang Adam kernel (`adam_step.slang`). Same formula as PyTorch Adam (with bias correction):
//   m ← β1·m + (1-β1)·g, v ← β2·v + (1-β2)·g², p ← p - lr·m̂ / (√v̂ + ε)
//   (m̂ = m / (1 - β1^t), v̂ = v / (1 - β2^t))
// Compared element-wise against a pure Swift reference; 1-2 ULP GPU sqrt differences are tolerated
// (bit-exact only for grad=0).

/// One pure Swift Adam step, written in the kernel's evaluation order so the numbers match.
private func referenceAdamStep(
    param: inout [Float], grad: [Float],
    m: inout [Float], v: inout [Float],
    lr: Float, beta1: Float, beta2: Float, eps: Float,
    stepCount: Int
) {
    let t = Float(stepCount)
    let bc1 = Float(1 - pow(Double(beta1), Double(t)))
    let bc2 = Float(1 - pow(Double(beta2), Double(t)))
    for i in 0..<param.count {
        let g = grad[i]
        let mNew = beta1 * m[i] + (1 - beta1) * g
        let vNew = beta2 * v[i] + (1 - beta2) * g * g
        let mHat = mNew / bc1
        let vHat = vNew / bc2
        let step = -lr * mHat / (sqrt(vHat) + eps)
        param[i] = param[i] + step
        m[i] = mNew
        v[i] = vNew
    }
}

/// One step from zero initial state. At t=1, m̂ = g and v̂ = g², so step ≈ -lr·sign(g).
@Test func fusedAdamSingleStepMatchesReferenceFloat() {
    let n = 128
    let lr: Float = 1e-3
    let beta1: Float = 0.9
    let beta2: Float = 0.999
    let eps: Float = 1e-15

    let paramHost: [Float] = (0..<n).map { Float($0) * 0.01 - 0.5 }
    let gradHost: [Float] = (0..<n).map { Float($0) * 0.001 - 0.1 }

    var paramRef = paramHost
    var mRef = [Float](repeating: 0, count: n)
    var vRef = [Float](repeating: 0, count: n)
    referenceAdamStep(
        param: &paramRef, grad: gradHost, m: &mRef, v: &vRef,
        lr: lr, beta1: beta1, beta2: beta2, eps: eps, stepCount: 1
    )

    var paramArr = MLXArray(paramHost, [n])
    let opt = AdamOptimizer(lr: lr, beta1: beta1, beta2: beta2, eps: eps)
    var state = opt.newState(like: paramArr)
    opt.step(param: &paramArr, grad: MLXArray(gradHost, [n]), state: &state)

    let paramOut = paramArr.asArray(Float.self)
    let mOut = state.m.asArray(Float.self)
    let vOut = state.v.asArray(Float.self)

    for i in 0..<n {
        // Values are ~±1e-3; tol 1e-5 (~1%) absorbs sqrt ULP differences.
        #expect(abs(paramOut[i] - paramRef[i]) < 1e-5,
                "param[\(i)] mismatch: kernel=\(paramOut[i]) ref=\(paramRef[i])")
        #expect(abs(mOut[i] - mRef[i]) < 1e-7)
        #expect(abs(vOut[i] - vRef[i]) < 1e-12)
    }
    #expect(state.stepCount == 1)
}

/// Kernel and reference state do not diverge over 100 steps.
@Test func fusedAdamHundredStepsStaysCloseToReference() {
    let n = 256
    let lr: Float = 5e-3
    let beta1: Float = 0.9
    let beta2: Float = 0.999
    let eps: Float = 1e-8

    var paramHost: [Float] = (0..<n).map { Float(($0 % 17) - 8) * 0.1 }
    var mHost = [Float](repeating: 0, count: n)
    var vHost = [Float](repeating: 0, count: n)

    var paramArr = MLXArray(paramHost, [n])
    let opt = AdamOptimizer(lr: lr, beta1: beta1, beta2: beta2, eps: eps)
    var state = opt.newState(like: paramArr)

    for s in 1...100 {
        // grad = 2(x - 5) + deterministic step-dependent noise. Expanded because the one-line
        // expression times out Swift type inference.
        var gradHost = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let mainGrad = 2 * (paramHost[i] - 5)
            let noise = Float((i + s) % 7) * 0.001
            gradHost[i] = mainGrad + noise
        }

        referenceAdamStep(
            param: &paramHost, grad: gradHost, m: &mHost, v: &vHost,
            lr: lr, beta1: beta1, beta2: beta2, eps: eps, stepCount: s
        )
        opt.step(param: &paramArr, grad: MLXArray(gradHost, [n]), state: &state)
    }

    let paramKernel = paramArr.asArray(Float.self)
    var maxAbsDiff: Float = 0
    var maxRelDiff: Float = 0
    for i in 0..<n {
        let abs_ = abs(paramKernel[i] - paramHost[i])
        let rel = abs_ / max(abs(paramHost[i]), 1e-6)
        if abs_ > maxAbsDiff { maxAbsDiff = abs_ }
        if rel > maxRelDiff { maxRelDiff = rel }
    }
    // Tolerate GPU sqrt ULP drift accumulated over 100 steps.
    #expect(maxAbsDiff < 1e-3, "max abs diff after 100 steps: \(maxAbsDiff)")
    #expect(maxRelDiff < 1e-3, "max rel diff after 100 steps: \(maxRelDiff)")
}

/// With grad = 0, m, v and step are exactly 0 and param is bit-exact unchanged.
@Test func fusedAdamZeroGradLeavesParamExactlyUnchanged() {
    let n = 64
    let paramHost: [Float] = (0..<n).map { Float($0) * 0.137 - 1.5 }
    let gradHost = [Float](repeating: 0, count: n)

    var paramArr = MLXArray(paramHost, [n])
    let opt = AdamOptimizer(lr: 1e-2, beta1: 0.9, beta2: 0.999, eps: 1e-15)
    var state = opt.newState(like: paramArr)

    opt.step(param: &paramArr, grad: MLXArray(gradHost, [n]), state: &state)

    let paramOut = paramArr.asArray(Float.self)
    let mOut = state.m.asArray(Float.self)
    let vOut = state.v.asArray(Float.self)
    for i in 0..<n {
        #expect(paramOut[i] == paramHost[i],
                "zero-grad must leave param bit-exact: kernel=\(paramOut[i]) input=\(paramHost[i])")
        #expect(mOut[i] == 0)
        #expect(vOut[i] == 0)
    }
}

/// A 2D param ([N, 3]) is processed flat and updated correctly with its shape preserved.
@Test func fusedAdamPreservesMultiDimShape() {
    let n = 32
    let d = 3
    let total = n * d
    let paramHost: [Float] = (0..<total).map { Float($0) * 0.01 }
    let gradHost: [Float] = (0..<total).map { Float($0 % 5) * 0.001 - 0.002 }

    var paramArr = MLXArray(paramHost, [n, d])
    let opt = AdamOptimizer(lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-15)
    var state = opt.newState(like: paramArr)

    opt.step(param: &paramArr, grad: MLXArray(gradHost, [n, d]), state: &state)

    #expect(paramArr.shape == [n, d], "shape must be preserved: got \(paramArr.shape)")
    #expect(state.m.shape == [n, d])
    #expect(state.v.shape == [n, d])

    // Compare against the flat reference.
    var paramRef = paramHost
    var mRef = [Float](repeating: 0, count: total)
    var vRef = [Float](repeating: 0, count: total)
    referenceAdamStep(
        param: &paramRef, grad: gradHost, m: &mRef, v: &vRef,
        lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-15, stepCount: 1
    )

    let paramOut = paramArr.reshaped([total]).asArray(Float.self)
    for i in 0..<total {
        #expect(abs(paramOut[i] - paramRef[i]) < 1e-5,
                "param[\(i)] mismatch in 2D path")
    }
}

/// Minimizing f(x) = (x - 5)² for 1000 steps converges to 5 (a kernel bug diverges / gives NaN).
@Test func fusedAdamConvergesOnQuadratic() {
    let n = 32
    let target: Float = 5.0
    let lr: Float = 1e-1   // fast enough to converge reliably in 1000 steps
    var paramArr = MLXArray([Float](repeating: 0, count: n), [n])
    let opt = AdamOptimizer(lr: lr, beta1: 0.9, beta2: 0.999, eps: 1e-15)
    var state = opt.newState(like: paramArr)

    for _ in 1...1000 {
        let p = paramArr.asArray(Float.self)
        let grad = MLXArray(p.map { 2 * ($0 - target) }, [n])
        opt.step(param: &paramArr, grad: grad, state: &state)
    }

    let final = paramArr.asArray(Float.self)
    for i in 0..<n {
        #expect(abs(final[i] - target) < 1e-2,
                "param[\(i)] failed to converge: got \(final[i]), expected ~\(target)")
        #expect(final[i].isFinite, "param[\(i)] = \(final[i]) (NaN/Inf detected)")
    }
}

/// 1M elements do not crash and give sane results.
@Test func fusedAdamHandlesLargeN() {
    let n = 1_000_000
    var paramArr = MLXArray.zeros([n])
    let gradArr = MLXArray.ones([n])
    let opt = AdamOptimizer(lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-15)
    var state = opt.newState(like: paramArr)

    opt.step(param: &paramArr, grad: gradArr, state: &state)

    // Step 1 is ≈ -lr·sign(g) = -1e-3. Comparing all 1M is slow, so check a few points.
    let final = paramArr.asArray(Float.self)
    let probes = [0, n / 4, n / 2, 3 * n / 4, n - 1]
    for i in probes {
        #expect(abs(final[i] - (-1e-3)) < 1e-5,
                "param[\(i)] = \(final[i]) (expected ~-1e-3)")
    }
}
