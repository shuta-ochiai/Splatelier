// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXFast

/// Adam with bias correction. Each update is a single launch of the fused Slang kernel
/// `adam_step.slang`. lr and stepCount differ per group, so each parameter group holds its own
/// `(optimiser, state)`.
public final class AdamOptimizer {
    public struct State {
        public var m: MLXArray
        public var v: MLXArray
        public var stepCount: Int

        public init(m: MLXArray, v: MLXArray, stepCount: Int = 0) {
            self.m = m
            self.v = v
            self.stepCount = stepCount
        }
    }

    public var lr: Float
    public let beta1: Float
    public let beta2: Float
    public let eps: Float

    public init(
        lr: Float,
        beta1: Float = 0.9,
        beta2: Float = 0.999,
        eps: Float = 1e-15
    ) {
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
    }

    /// Zero moment state with the same shape as `array`.
    public func newState(like array: MLXArray) -> State {
        State(
            m: MLXArray.zeros(array.shape, dtype: array.dtype),
            v: MLXArray.zeros(array.shape, dtype: array.dtype),
            stepCount: 0
        )
    }

    private static let adamKernel = SlangCompiledKernel.required("adam_step")

    /// Advances `state.stepCount` and returns `[lr, β1, β2, ε, 1/bc1, 1/bc2, n]`, in the same order
    /// as hyper[0..6] of `adam_step.slang`. Bias correction is computed in Double.
    private func makeHyper(state: inout State, elementCount n: Int) -> MLXArray {
        state.stepCount += 1
        let t = Double(state.stepCount)
        let invBc1 = Float(1.0 / (1 - pow(Double(beta1), t)))
        let invBc2 = Float(1.0 / (1 - pow(Double(beta2), t)))
        return MLXArray([lr, beta1, beta2, eps, invBc1, invBc2, Float(n)], [7])
    }

    /// Enqueues one Adam step on the GPU (no eval). stepCount advances even for an empty param.
    public func step(param: inout MLXArray, grad: MLXArray, state: inout State) {
        assert(param.shape == grad.shape, "grad shape must match param")
        assert(param.shape == state.m.shape, "state.m shape must match param")
        assert(param.shape == state.v.shape, "state.v shape must match param")

        let shape = param.shape
        let n = shape.reduce(1, *)
        let hyper = makeHyper(state: &state, elementCount: n)
        if n == 0 { return }

        let tgSize = 256
        let out = Self.adamKernel(
            [param, grad, state.m, state.v, hyper],
            grid: (((n + tgSize - 1) / tgSize) * tgSize, 1, 1),
            threadGroup: (tgSize, 1, 1),
            outputShapes: [shape, shape, shape],
            outputDTypes: [.float32, .float32, .float32]
        )
        param = out[0]
        state.m = out[1]
        state.v = out[2]
    }
}
