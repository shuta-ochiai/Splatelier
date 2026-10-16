#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
void bgBasis_0(float3 dir_0, int deg_0, array<float, int(25)> thread* bs_0)
{
    uint k_0 = 0U;
    for(;;)
    {
        if(k_0 < 25U)
        {
        }
        else
        {
            break;
        }
        (*bs_0)[k_0] = 0.0;
        k_0 = k_0 + 1U;
    }
    float x_0 = dir_0.x;
    float y_0 = dir_0.y;
    float z_0 = dir_0.z;
    (*bs_0)[int(0)] = 0.282094806432724;
    if(deg_0 >= int(1))
    {
        (*bs_0)[int(1)] = -0.48860251903533936 * y_0;
        (*bs_0)[int(2)] = 0.48860251903533936 * z_0;
        (*bs_0)[int(3)] = -0.48860251903533936 * x_0;
    }
    if(deg_0 >= int(2))
    {
        float xx_0 = x_0 * x_0;
        float yy_0 = y_0 * y_0;
        float zz_0 = z_0 * z_0;
        (*bs_0)[int(4)] = 1.09254848957061768 * (x_0 * y_0);
        (*bs_0)[int(5)] = -1.09254848957061768 * (y_0 * z_0);
        (*bs_0)[int(6)] = 0.31539157032966614 * (2.0 * zz_0 - xx_0 - yy_0);
        (*bs_0)[int(7)] = -1.09254848957061768 * (x_0 * z_0);
        (*bs_0)[int(8)] = 0.54627424478530884 * (xx_0 - yy_0);
    }
    if(deg_0 >= int(3))
    {
        float xx_1 = x_0 * x_0;
        float yy_1 = y_0 * y_0;
        float zz_1 = z_0 * z_0;
        float _S1 = 3.0 * xx_1;
        (*bs_0)[int(9)] = -0.59004360437393188 * y_0 * (_S1 - yy_1);
        (*bs_0)[int(10)] = 2.89061141014099121 * (x_0 * y_0) * z_0;
        float _S2 = 4.0 * zz_1 - xx_1 - yy_1;
        (*bs_0)[int(11)] = -0.4570457935333252 * y_0 * _S2;
        float _S3 = 3.0 * yy_1;
        (*bs_0)[int(12)] = 0.37317633628845215 * z_0 * (2.0 * zz_1 - _S1 - _S3);
        (*bs_0)[int(13)] = -0.4570457935333252 * x_0 * _S2;
        (*bs_0)[int(14)] = 1.44530570507049561 * z_0 * (xx_1 - yy_1);
        (*bs_0)[int(15)] = -0.59004360437393188 * x_0 * (xx_1 - _S3);
    }
    if(deg_0 >= int(4))
    {
        float xx_2 = x_0 * x_0;
        float yy_2 = y_0 * y_0;
        float zz_2 = z_0 * z_0;
        float _S4 = x_0 * y_0;
        float _S5 = xx_2 - yy_2;
        (*bs_0)[int(16)] = 2.50334286689758301 * _S4 * _S5;
        float _S6 = y_0 * z_0;
        float _S7 = 3.0 * xx_2 - yy_2;
        (*bs_0)[int(17)] = -1.77013075351715088 * _S6 * _S7;
        float _S8 = 7.0 * zz_2;
        float _S9 = _S8 - 1.0;
        (*bs_0)[int(18)] = 0.94617468118667603 * _S4 * _S9;
        float _S10 = _S8 - 3.0;
        (*bs_0)[int(19)] = -0.66904652118682861 * _S6 * _S10;
        (*bs_0)[int(20)] = 0.1057855486869812 * (35.0 * zz_2 * zz_2 - 30.0 * zz_2 + 3.0);
        float _S11 = x_0 * z_0;
        (*bs_0)[int(21)] = -0.66904652118682861 * _S11 * _S10;
        (*bs_0)[int(22)] = 0.47308734059333801 * _S5 * _S9;
        float _S12 = xx_2 - 3.0 * yy_2;
        (*bs_0)[int(23)] = -1.77013075351715088 * _S11 * _S12;
        (*bs_0)[int(24)] = 0.62583571672439575 * (xx_2 * _S12 - yy_2 * _S7);
    }
    return;
}

struct KernelContext_0
{
    int device* dims_bwd_0;
    float device* intr_bwd_0;
    float device* viewmat_bwd_0;
    float device* cot_color_bwd_0;
    float device* render_alpha_bwd_0;
    float device* partials_0;
    uint _S13;
    uint _S14;
    array<float, int(300)> threadgroup* gsm_0;
};

[[kernel]] void background_sh_bwd(uint3 sv_groupthreadid_0 [[thread_position_in_threadgroup]], uint3 tid_0 [[thread_position_in_grid]], uint _builtinWaveLaneCount_0 [[threads_per_simdgroup]], uint _builtinWaveLaneIndex_0 [[thread_index_in_simdgroup]], int device* dims_bwd_1 [[buffer(12)]], float device* intr_bwd_1 [[buffer(11)]], float device* viewmat_bwd_1 [[buffer(10)]], float device* cot_color_bwd_1 [[buffer(8)]], float device* render_alpha_bwd_1 [[buffer(9)]], float device* partials_1 [[buffer(13)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->dims_bwd_0 = dims_bwd_1;
    (&kernelContext_0)->intr_bwd_0 = intr_bwd_1;
    (&kernelContext_0)->viewmat_bwd_0 = viewmat_bwd_1;
    (&kernelContext_0)->cot_color_bwd_0 = cot_color_bwd_1;
    (&kernelContext_0)->render_alpha_bwd_0 = render_alpha_bwd_1;
    (&kernelContext_0)->partials_0 = partials_1;
    threadgroup array<float, int(300)> gsm_1;
    (&kernelContext_0)->gsm_0 = &gsm_1;
    uint sv_groupindex_0 = (sv_groupthreadid_0[int(2)] + sv_groupthreadid_0[int(1)]) * 64U + sv_groupthreadid_0[int(0)];
    (&kernelContext_0)->_S13 = _builtinWaveLaneCount_0;
    (&kernelContext_0)->_S14 = _builtinWaveLaneIndex_0;
    uint W_0 = uint(dims_bwd_1[int(0)]);
    int deg_1 = dims_bwd_1[int(2)];
    uint p_0 = tid_0.x;
    bool active_0 = p_0 < (W_0 * uint(dims_bwd_1[int(1)]));
    thread array<float, int(75)> gsh_0;
    uint k_1 = 0U;
    for(;;)
    {
        if(k_1 < 75U)
        {
        }
        else
        {
            break;
        }
        gsh_0[k_1] = 0.0;
        k_1 = k_1 + 1U;
    }
    uint s_0;
    if(active_0)
    {
        uint x_1 = p_0 % W_0;
        uint y_1 = p_0 / W_0;
        float cxd_0 = (float(x_1) + 0.5 - (&kernelContext_0)->intr_bwd_0[int(2)]) / (&kernelContext_0)->intr_bwd_0[int(0)];
        float cyd_0 = (float(y_1) + 0.5 - (&kernelContext_0)->intr_bwd_0[int(3)]) / (&kernelContext_0)->intr_bwd_0[int(1)];
        float3 w3_0 = float3(cxd_0 * (&kernelContext_0)->viewmat_bwd_0[int(0)] + cyd_0 * (&kernelContext_0)->viewmat_bwd_0[int(4)] + (&kernelContext_0)->viewmat_bwd_0[int(8)], cxd_0 * (&kernelContext_0)->viewmat_bwd_0[int(1)] + cyd_0 * (&kernelContext_0)->viewmat_bwd_0[int(5)] + (&kernelContext_0)->viewmat_bwd_0[int(9)], cxd_0 * (&kernelContext_0)->viewmat_bwd_0[int(2)] + cyd_0 * (&kernelContext_0)->viewmat_bwd_0[int(6)] + (&kernelContext_0)->viewmat_bwd_0[int(10)]);
        thread array<float, int(25)> bs_1;
        bgBasis_0(w3_0 * float3(rsqrt(max(dot(w3_0, w3_0), 9.99999968265522539e-21))) , deg_1, &bs_1);
        uint o_0 = p_0 * 3U;
        float3 _S15 = float3((&kernelContext_0)->cot_color_bwd_0[o_0], (&kernelContext_0)->cot_color_bwd_0[o_0 + 1U], (&kernelContext_0)->cot_color_bwd_0[o_0 + 2U]) * float3((1.0 - (&kernelContext_0)->render_alpha_bwd_0[p_0])) ;
        s_0 = 0U;
        for(;;)
        {
            if(s_0 < 25U)
            {
            }
            else
            {
                break;
            }
            uint _S16 = s_0 * 3U;
            gsh_0[_S16] = bs_1[s_0] * _S15.x;
            gsh_0[_S16 + 1U] = bs_1[s_0] * _S15.y;
            gsh_0[_S16 + 2U] = bs_1[s_0] * _S15.z;
            s_0 = s_0 + 1U;
        }
    }
    uint _S17 = (&kernelContext_0)->_S13;
    uint _S18 = sv_groupindex_0 / (&kernelContext_0)->_S13;
    uint _S19 = (64U + _S17 - 1U) / _S17;
    k_1 = 0U;
    for(;;)
    {
        if(k_1 < 75U)
        {
        }
        else
        {
            break;
        }
        float ws_0 = simd_sum(gsh_0[k_1]);
        if(((&kernelContext_0)->_S14) == 0U)
        {
            (*(&kernelContext_0)->gsm_0)[_S18 * 75U + k_1] = ws_0;
        }
        k_1 = k_1 + 1U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sv_groupindex_0 == 0U)
    {
        uint _S20 = p_0 / 64U;
        k_1 = 0U;
        for(;;)
        {
            if(k_1 < 75U)
            {
            }
            else
            {
                break;
            }
            s_0 = 0U;
            float t_0 = 0.0;
            for(;;)
            {
                if(s_0 < _S19)
                {
                }
                else
                {
                    break;
                }
                float t_1 = t_0 + (*(&kernelContext_0)->gsm_0)[s_0 * 75U + k_1];
                s_0 = s_0 + 1U;
                t_0 = t_1;
            }
            *((&kernelContext_0)->partials_0+(_S20 * 75U + k_1)) = t_0;
            k_1 = k_1 + 1U;
        }
    }
    return;
}

