// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// One Adam step (1 thread / element). All buffers are flat float32.
// The structure (1 thread/element, register hoisting) follows gsplat AdamCUDA.cu (Apache-2.0).
// Unlike gsplat, applies bias correction (to match the Swift AdamOptimizer).
// MLX cannot work in place, so this writes to an out buffer and the caller rebinds it.

StructuredBuffer<float> param_in;
StructuredBuffer<float> grad;
StructuredBuffer<float> m_in;
StructuredBuffer<float> v_in;
// [lr, β1, β2, ε, 1/(1-β1^t), 1/(1-β2^t), n]. Order must match Swift AdamOptimizer.makeHyper.
StructuredBuffer<float> hyper;

RWStructuredBuffer<float> param_out;
RWStructuredBuffer<float> m_out;
RWStructuredBuffer<float> v_out;

[shader("compute")]
[numthreads(256, 1, 1)]
void adam_step(uint3 tid : SV_DispatchThreadID)
{
    int i = int(tid.x);
    int n = int(hyper[6]);
    if (i >= n) { return; }

    float g       = grad[i];
    float m_prev  = m_in[i];
    float v_prev  = v_in[i];
    float p_prev  = param_in[i];

    float lr      = hyper[0];
    float b1      = hyper[1];
    float b2      = hyper[2];
    float eps     = hyper[3];
    float inv_bc1 = hyper[4];
    float inv_bc2 = hyper[5];

    float m_new = b1 * m_prev + (1.0 - b1) * g;
    float v_new = b2 * v_prev + (1.0 - b2) * g * g;

    // Bias correction multiplies by reciprocals computed on the host (avoids division).
    float m_hat = m_new * inv_bc1;
    float v_hat = v_new * inv_bc2;

    float step = -lr * m_hat / (sqrt(v_hat) + eps);

    param_out[i] = p_prev + step;
    m_out[i]     = m_new;
    v_out[i]     = v_new;
}
