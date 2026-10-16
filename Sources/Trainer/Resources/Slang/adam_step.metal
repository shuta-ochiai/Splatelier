#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    float device* hyper_0;
    float device* grad_0;
    float device* m_in_0;
    float device* v_in_0;
    float device* param_in_0;
    float device* param_out_0;
    float device* m_out_0;
    float device* v_out_0;
};

[[kernel]] void adam_step(uint3 tid_0 [[thread_position_in_grid]], float device* hyper_1 [[buffer(4)]], float device* grad_1 [[buffer(1)]], float device* m_in_1 [[buffer(2)]], float device* v_in_1 [[buffer(3)]], float device* param_in_1 [[buffer(0)]], float device* param_out_1 [[buffer(5)]], float device* m_out_1 [[buffer(6)]], float device* v_out_1 [[buffer(7)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->hyper_0 = hyper_1;
    (&kernelContext_0)->grad_0 = grad_1;
    (&kernelContext_0)->m_in_0 = m_in_1;
    (&kernelContext_0)->v_in_0 = v_in_1;
    (&kernelContext_0)->param_in_0 = param_in_1;
    (&kernelContext_0)->param_out_0 = param_out_1;
    (&kernelContext_0)->m_out_0 = m_out_1;
    (&kernelContext_0)->v_out_0 = v_out_1;
    int i_0 = int(tid_0.x);
    if(i_0 >= int(hyper_1[int(6)]))
    {
        return;
    }
    float g_0 = (&kernelContext_0)->grad_0[i_0];
    float b1_0 = (&kernelContext_0)->hyper_0[int(1)];
    float b2_0 = (&kernelContext_0)->hyper_0[int(2)];
    float m_new_0 = b1_0 * (&kernelContext_0)->m_in_0[i_0] + (1.0 - b1_0) * g_0;
    float v_new_0 = b2_0 * (&kernelContext_0)->v_in_0[i_0] + (1.0 - b2_0) * g_0 * g_0;
    *((&kernelContext_0)->param_out_0+i_0) = (&kernelContext_0)->param_in_0[i_0] + - (&kernelContext_0)->hyper_0[int(0)] * (m_new_0 * (&kernelContext_0)->hyper_0[int(4)]) / (sqrt(v_new_0 * (&kernelContext_0)->hyper_0[int(5)]) + (&kernelContext_0)->hyper_0[int(3)]);
    *((&kernelContext_0)->m_out_0+i_0) = m_new_0;
    *((&kernelContext_0)->v_out_0+i_0) = v_new_0;
    return;
}

