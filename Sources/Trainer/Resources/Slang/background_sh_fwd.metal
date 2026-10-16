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
    int device* dims_fwd_0;
    float device* intr_fwd_0;
    float device* viewmat_fwd_0;
    float device* coeffs_fwd_0;
    float device* render_alpha_0;
    float device* composited_0;
    float device* render_0;
    float device* bg_out_0;
};

[[kernel]] void background_sh_fwd(uint3 tid_0 [[thread_position_in_grid]], int device* dims_fwd_1 [[buffer(5)]], float device* intr_fwd_1 [[buffer(4)]], float device* viewmat_fwd_1 [[buffer(3)]], float device* coeffs_fwd_1 [[buffer(2)]], float device* render_alpha_1 [[buffer(1)]], float device* composited_1 [[buffer(6)]], float device* render_1 [[buffer(0)]], float device* bg_out_1 [[buffer(7)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->dims_fwd_0 = dims_fwd_1;
    (&kernelContext_0)->intr_fwd_0 = intr_fwd_1;
    (&kernelContext_0)->viewmat_fwd_0 = viewmat_fwd_1;
    (&kernelContext_0)->coeffs_fwd_0 = coeffs_fwd_1;
    (&kernelContext_0)->render_alpha_0 = render_alpha_1;
    (&kernelContext_0)->composited_0 = composited_1;
    (&kernelContext_0)->render_0 = render_1;
    (&kernelContext_0)->bg_out_0 = bg_out_1;
    uint W_0 = uint(dims_fwd_1[int(0)]);
    int deg_1 = dims_fwd_1[int(2)];
    uint p_0 = tid_0.x;
    if(p_0 >= (W_0 * uint(dims_fwd_1[int(1)])))
    {
        return;
    }
    uint x_1 = p_0 % W_0;
    uint y_1 = p_0 / W_0;
    float cxd_0 = (float(x_1) + 0.5 - (&kernelContext_0)->intr_fwd_0[int(2)]) / (&kernelContext_0)->intr_fwd_0[int(0)];
    float cyd_0 = (float(y_1) + 0.5 - (&kernelContext_0)->intr_fwd_0[int(3)]) / (&kernelContext_0)->intr_fwd_0[int(1)];
    float3 w3_0 = float3(cxd_0 * (&kernelContext_0)->viewmat_fwd_0[int(0)] + cyd_0 * (&kernelContext_0)->viewmat_fwd_0[int(4)] + (&kernelContext_0)->viewmat_fwd_0[int(8)], cxd_0 * (&kernelContext_0)->viewmat_fwd_0[int(1)] + cyd_0 * (&kernelContext_0)->viewmat_fwd_0[int(5)] + (&kernelContext_0)->viewmat_fwd_0[int(9)], cxd_0 * (&kernelContext_0)->viewmat_fwd_0[int(2)] + cyd_0 * (&kernelContext_0)->viewmat_fwd_0[int(6)] + (&kernelContext_0)->viewmat_fwd_0[int(10)]);
    thread array<float, int(25)> bs_1;
    bgBasis_0(w3_0 * float3(rsqrt(max(dot(w3_0, w3_0), 9.99999968265522539e-21))) , deg_1, &bs_1);
    int _S13 = deg_1 + int(1);
    uint nb_0 = uint(_S13 * _S13);
    uint nb_1;
    if(nb_0 > 25U)
    {
        nb_1 = 25U;
    }
    else
    {
        nb_1 = nb_0;
    }
    float3 _S14 = float3(0.0, 0.0, 0.0);
    uint b_0 = 0U;
    float3 bg_0 = _S14;
    for(;;)
    {
        if(b_0 < nb_1)
        {
        }
        else
        {
            break;
        }
        uint _S15 = b_0 * 3U;
        float3 bg_1 = bg_0 + float3(bs_1[b_0])  * float3((&kernelContext_0)->coeffs_fwd_0[_S15], (&kernelContext_0)->coeffs_fwd_0[_S15 + 1U], (&kernelContext_0)->coeffs_fwd_0[_S15 + 2U]);
        b_0 = b_0 + 1U;
        bg_0 = bg_1;
    }
    float oma_0 = 1.0 - (&kernelContext_0)->render_alpha_0[p_0];
    uint o_0 = p_0 * 3U;
    float _S16 = bg_0.x;
    *((&kernelContext_0)->composited_0+o_0) = (&kernelContext_0)->render_0[o_0] + oma_0 * _S16;
    uint _S17 = o_0 + 1U;
    float _S18 = bg_0.y;
    *((&kernelContext_0)->composited_0+_S17) = (&kernelContext_0)->render_0[_S17] + oma_0 * _S18;
    uint _S19 = o_0 + 2U;
    float _S20 = bg_0.z;
    *((&kernelContext_0)->composited_0+_S19) = (&kernelContext_0)->render_0[_S19] + oma_0 * _S20;
    *((&kernelContext_0)->bg_out_0+o_0) = _S16;
    *((&kernelContext_0)->bg_out_0+_S17) = _S18;
    *((&kernelContext_0)->bg_out_0+_S19) = _S20;
    return;
}

