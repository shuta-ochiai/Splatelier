#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct DiffPair_float_0
{
    float primal_0;
    float differential_0;
};

struct s_paramCtx_s_bwdCallableCtx_ssim_pixel_0
{
    float value0_0;
    float value1_0;
    float value2_0;
    float value3_0;
    float value4_0;
    float value5_0;
    float value6_0;
};

struct s_bwdCallableCtx_ssim_pixel_0
{
    s_paramCtx_s_bwdCallableCtx_ssim_pixel_0 value0_1;
};

void s_bwdProp_ssim_pixel_0(const s_bwdCallableCtx_ssim_pixel_0 thread* _S1, float thread* _S2, float thread* _S3, float thread* _S4, float _S5)
{
    float mu1sq_0 = (&_S1->value0_1)->value0_0 * (&_S1->value0_1)->value0_0;
    float mu2sq_0 = (&_S1->value0_1)->value3_0 * (&_S1->value0_1)->value3_0;
    float A_0 = mu1sq_0 + mu2sq_0 + (&_S1->value0_1)->value5_0;
    float B_0 = (&_S1->value0_1)->value1_0 - mu1sq_0 + ((&_S1->value0_1)->value4_0 - mu2sq_0) + (&_S1->value0_1)->value6_0;
    float Cn_0 = 2.0 * (&_S1->value0_1)->value0_0 * (&_S1->value0_1)->value3_0 + (&_S1->value0_1)->value5_0;
    float Dn_0 = 2.0 * ((&_S1->value0_1)->value2_0 - (&_S1->value0_1)->value0_0 * (&_S1->value0_1)->value3_0) + (&_S1->value0_1)->value6_0;
    float _S6 = A_0 * B_0;
    float _S7 = _S5 / (_S6 * _S6);
    float _S8 = Cn_0 * Dn_0 * - _S7;
    float _S9 = _S6 * _S7;
    float s_diff_sigma1_sq_T_0 = A_0 * _S8;
    float s_diff_sigma12_T_0 = 2.0 * (Cn_0 * _S9);
    float _S10 = (&_S1->value0_1)->value0_0 * (B_0 * _S8 + - s_diff_sigma1_sq_T_0);
    *_S2 = 2.0 * ((&_S1->value0_1)->value3_0 * (Dn_0 * _S9)) + (&_S1->value0_1)->value3_0 * - s_diff_sigma12_T_0 + _S10 + _S10;
    *_S3 = s_diff_sigma1_sq_T_0;
    *_S4 = s_diff_sigma12_T_0;
    return;
}

void _S11(DiffPair_float_0 thread* _S12, DiffPair_float_0 thread* _S13, DiffPair_float_0 thread* _S14, float _S15, float _S16, float _S17, float _S18, float _S19)
{
    thread s_paramCtx_s_bwdCallableCtx_ssim_pixel_0 _S20;
    (&_S20)->value0_0 = _S12->primal_0;
    (&_S20)->value1_0 = _S13->primal_0;
    (&_S20)->value2_0 = _S14->primal_0;
    (&_S20)->value3_0 = _S15;
    (&_S20)->value4_0 = _S16;
    (&_S20)->value5_0 = _S17;
    (&_S20)->value6_0 = _S18;
    thread s_bwdCallableCtx_ssim_pixel_0 _S21;
    (&_S21)->value0_1 = _S20;
    s_bwdProp_ssim_pixel_0(&_S21, &_S12->differential_0, &_S13->differential_0, &_S14->differential_0, _S19);
    return;
}

float ssim_pixel_0(float mu1_0, float m1x2_0, float m1xy_0, float mu2_0, float m2x2_0, float c1_0, float c2_0)
{
    float mu1sq_1 = mu1_0 * mu1_0;
    float mu2sq_1 = mu2_0 * mu2_0;
    return (2.0 * mu1_0 * mu2_0 + c1_0) * (2.0 * (m1xy_0 - mu1_0 * mu2_0) + c2_0) / ((mu1sq_1 + mu2sq_1 + c1_0) * (m1x2_0 - mu1sq_1 + (m2x2_0 - mu2sq_1) + c2_0));
}

struct SsimGrad_0
{
    float d_mu1_0;
    float d_m1x2_0;
    float d_m1xy_0;
};

SsimGrad_0 ssim_pixel_grad_0(float mu1_1, float m1x2_1, float m1xy_1, float mu2_1, float m2x2_1, float c1_1, float c2_1)
{
    thread DiffPair_float_0 dpMu1_0;
    (&dpMu1_0)->primal_0 = mu1_1;
    (&dpMu1_0)->differential_0 = 0.0;
    thread DiffPair_float_0 dpM1x2_0;
    (&dpM1x2_0)->primal_0 = m1x2_1;
    (&dpM1x2_0)->differential_0 = 0.0;
    thread DiffPair_float_0 dpM1xy_0;
    (&dpM1xy_0)->primal_0 = m1xy_1;
    (&dpM1xy_0)->differential_0 = 0.0;
    _S11(&dpMu1_0, &dpM1x2_0, &dpM1xy_0, mu2_1, m2x2_1, c1_1, c2_1, 1.0);
    thread SsimGrad_0 g_0;
    (&g_0)->d_mu1_0 = dpMu1_0.differential_0;
    (&g_0)->d_m1x2_0 = dpM1x2_0.differential_0;
    (&g_0)->d_m1xy_0 = dpM1xy_0.differential_0;
    return g_0;
}

struct KernelContext_0
{
    int device* dims_0;
    float device* img1_0;
    float device* img2_0;
    float device* gauss_0;
    half device* ssim_map_0;
    half device* dm_dmu1_0;
    half device* dm_dsigma1_sq_0;
    half device* dm_dsigma12_0;
    float device* tilePartialL1_0;
    float device* tilePartialSSIM_0;
    array<float, int(676)> threadgroup* s_img1_0;
    array<float, int(676)> threadgroup* s_img2_0;
    array<float, int(416)> threadgroup* s_hX_0;
    array<float, int(416)> threadgroup* s_hX2_0;
    array<float, int(416)> threadgroup* s_hY_0;
    array<float, int(416)> threadgroup* s_hY2_0;
    array<float, int(416)> threadgroup* s_hXY_0;
    array<float, int(256)> threadgroup* s_redL1_0;
    array<float, int(256)> threadgroup* s_redSSIM_0;
};

[[kernel]] void ssim_fwd_tiled(uint3 gid_0 [[thread_position_in_grid]], uint3 ltid_0 [[thread_position_in_threadgroup]], uint3 grp_0 [[threadgroup_position_in_grid]], int device* dims_1 [[buffer(3)]], float device* img1_1 [[buffer(0)]], float device* img2_1 [[buffer(1)]], float device* gauss_1 [[buffer(2)]], half device* ssim_map_1 [[buffer(4)]], half device* dm_dmu1_1 [[buffer(5)]], half device* dm_dsigma1_sq_1 [[buffer(6)]], half device* dm_dsigma12_1 [[buffer(7)]], float device* tilePartialL1_1 [[buffer(8)]], float device* tilePartialSSIM_1 [[buffer(9)]])
{
    int i_0;
    float aX_0;
    float aX2_0;
    bool _S22;
    bool _S23;
    bool inb_0;
    int k_0;
    float aY_0;
    float aY2_0;
    float aXY_0;
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->dims_0 = dims_1;
    (&kernelContext_0)->img1_0 = img1_1;
    (&kernelContext_0)->img2_0 = img2_1;
    (&kernelContext_0)->gauss_0 = gauss_1;
    (&kernelContext_0)->ssim_map_0 = ssim_map_1;
    (&kernelContext_0)->dm_dmu1_0 = dm_dmu1_1;
    (&kernelContext_0)->dm_dsigma1_sq_0 = dm_dsigma1_sq_1;
    (&kernelContext_0)->dm_dsigma12_0 = dm_dsigma12_1;
    (&kernelContext_0)->tilePartialL1_0 = tilePartialL1_1;
    (&kernelContext_0)->tilePartialSSIM_0 = tilePartialSSIM_1;
    threadgroup array<float, int(676)> s_img1_1;
    (&kernelContext_0)->s_img1_0 = &s_img1_1;
    threadgroup array<float, int(676)> s_img2_1;
    (&kernelContext_0)->s_img2_0 = &s_img2_1;
    threadgroup array<float, int(416)> s_hX_1;
    (&kernelContext_0)->s_hX_0 = &s_hX_1;
    threadgroup array<float, int(416)> s_hX2_1;
    (&kernelContext_0)->s_hX2_0 = &s_hX2_1;
    threadgroup array<float, int(416)> s_hY_1;
    (&kernelContext_0)->s_hY_0 = &s_hY_1;
    threadgroup array<float, int(416)> s_hY2_1;
    (&kernelContext_0)->s_hY2_0 = &s_hY2_1;
    threadgroup array<float, int(416)> s_hXY_1;
    (&kernelContext_0)->s_hXY_0 = &s_hXY_1;
    threadgroup array<float, int(256)> s_redL1_1;
    (&kernelContext_0)->s_redL1_0 = &s_redL1_1;
    threadgroup array<float, int(256)> s_redSSIM_1;
    (&kernelContext_0)->s_redSSIM_0 = &s_redSSIM_1;
    int _S24 = dims_1[int(0)];
    int W_0 = dims_1[int(1)];
    int _S25 = dims_1[int(2)];
    int rad_0 = dims_1[int(3)];
    int lx_0 = int(ltid_0.x);
    int ly_0 = int(ltid_0.y);
    int lidx_0 = ly_0 * int(16) + lx_0;
    int _S26 = int(grp_0.x);
    int _S27 = _S26 * int(16);
    int _S28 = int(grp_0.y);
    int _S29 = _S28 * int(16);
    int _S30 = int(gid_0.x);
    int _S31 = int(gid_0.y);
    int _S32 = int(2) * rad_0 + int(1);
    int c_0 = int(0);
    float l1acc_0 = 0.0;
    float ssimacc_0 = 0.0;
    for(;;)
    {
        if(c_0 < _S25)
        {
        }
        else
        {
            break;
        }
        i_0 = lidx_0;
        for(;;)
        {
            if(i_0 < int(676))
            {
            }
            else
            {
                break;
            }
            int hx_0 = _S27 - rad_0 + i_0 % int(26);
            int hy_0 = _S29 - rad_0 + i_0 / int(26);
            if(hx_0 >= int(0))
            {
                _S22 = hx_0 < W_0;
            }
            else
            {
                _S22 = false;
            }
            if(_S22)
            {
                _S23 = hy_0 >= int(0);
            }
            else
            {
                _S23 = false;
            }
            if(_S23)
            {
                inb_0 = hy_0 < _S24;
            }
            else
            {
                inb_0 = false;
            }
            int gidx_0 = (hy_0 * W_0 + hx_0) * _S25 + c_0;
            if(inb_0)
            {
                aX_0 = (&kernelContext_0)->img1_0[gidx_0];
            }
            else
            {
                aX_0 = 0.0;
            }
            (*(&kernelContext_0)->s_img1_0)[i_0] = aX_0;
            if(inb_0)
            {
                aX2_0 = (&kernelContext_0)->img2_0[gidx_0];
            }
            else
            {
                aX2_0 = 0.0;
            }
            (*(&kernelContext_0)->s_img2_0)[i_0] = aX2_0;
            i_0 = i_0 + int(256);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int j_0 = lidx_0;
        for(;;)
        {
            if(j_0 < int(416))
            {
            }
            else
            {
                break;
            }
            int _S33 = j_0 / int(16) * int(26) + j_0 % int(16);
            k_0 = int(0);
            aX_0 = 0.0;
            aX2_0 = 0.0;
            aY_0 = 0.0;
            aY2_0 = 0.0;
            aXY_0 = 0.0;
            for(;;)
            {
                if(k_0 < _S32)
                {
                }
                else
                {
                    break;
                }
                float w_0 = (&kernelContext_0)->gauss_0[k_0];
                int _S34 = _S33 + k_0;
                float _S35 = w_0 * (*(&kernelContext_0)->s_img1_0)[_S34];
                float aX_1 = aX_0 + _S35;
                float aX2_1 = aX2_0 + _S35 * (*(&kernelContext_0)->s_img1_0)[_S34];
                float _S36 = w_0 * (*(&kernelContext_0)->s_img2_0)[_S34];
                float aY_1 = aY_0 + _S36;
                float aY2_1 = aY2_0 + _S36 * (*(&kernelContext_0)->s_img2_0)[_S34];
                float aXY_1 = aXY_0 + _S35 * (*(&kernelContext_0)->s_img2_0)[_S34];
                k_0 = k_0 + int(1);
                aX_0 = aX_1;
                aX2_0 = aX2_1;
                aY_0 = aY_1;
                aY2_0 = aY2_1;
                aXY_0 = aXY_1;
            }
            (*(&kernelContext_0)->s_hX_0)[j_0] = aX_0;
            (*(&kernelContext_0)->s_hX2_0)[j_0] = aX2_0;
            (*(&kernelContext_0)->s_hY_0)[j_0] = aY_0;
            (*(&kernelContext_0)->s_hY2_0)[j_0] = aY2_0;
            (*(&kernelContext_0)->s_hXY_0)[j_0] = aXY_0;
            j_0 = j_0 + int(256);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S30 < W_0)
        {
            _S22 = _S31 < _S24;
        }
        else
        {
            _S22 = false;
        }
        if(_S22)
        {
            k_0 = int(0);
            aX_0 = 0.0;
            aX2_0 = 0.0;
            aY_0 = 0.0;
            aY2_0 = 0.0;
            aXY_0 = 0.0;
            for(;;)
            {
                if(k_0 < _S32)
                {
                }
                else
                {
                    break;
                }
                float w_1 = (&kernelContext_0)->gauss_0[k_0];
                int pidx_0 = (ly_0 + k_0) * int(16) + lx_0;
                float sX_0 = aX_0 + w_1 * (*(&kernelContext_0)->s_hX_0)[pidx_0];
                float sX2_0 = aX2_0 + w_1 * (*(&kernelContext_0)->s_hX2_0)[pidx_0];
                float sY_0 = aY_0 + w_1 * (*(&kernelContext_0)->s_hY_0)[pidx_0];
                float sY2_0 = aY2_0 + w_1 * (*(&kernelContext_0)->s_hY2_0)[pidx_0];
                float sXY_0 = aXY_0 + w_1 * (*(&kernelContext_0)->s_hXY_0)[pidx_0];
                k_0 = k_0 + int(1);
                aX_0 = sX_0;
                aX2_0 = sX2_0;
                aY_0 = sY_0;
                aY2_0 = sY2_0;
                aXY_0 = sXY_0;
            }
            float val_0 = ssim_pixel_0(aX_0, aX2_0, aXY_0, aY_0, aY2_0, 0.00009999999747379, 0.00089999998454005);
            SsimGrad_0 g_1 = ssim_pixel_grad_0(aX_0, aX2_0, aXY_0, aY_0, aY2_0, 0.00009999999747379, 0.00089999998454005);
            int o_0 = (_S31 * W_0 + _S30) * _S25 + c_0;
            *((&kernelContext_0)->ssim_map_0+o_0) = half(val_0);
            *((&kernelContext_0)->dm_dmu1_0+o_0) = half(g_1.d_mu1_0);
            *((&kernelContext_0)->dm_dsigma1_sq_0+o_0) = half(g_1.d_m1x2_0);
            *((&kernelContext_0)->dm_dsigma12_0+o_0) = half(g_1.d_m1xy_0);
            int _S37 = (ly_0 + rad_0) * int(26) + (lx_0 + rad_0);
            float l1acc_1 = l1acc_0 + abs((*(&kernelContext_0)->s_img1_0)[_S37] - (*(&kernelContext_0)->s_img2_0)[_S37]);
            if(_S30 >= rad_0)
            {
                _S23 = _S30 < (W_0 - rad_0);
            }
            else
            {
                _S23 = false;
            }
            if(_S23)
            {
                inb_0 = _S31 >= rad_0;
            }
            else
            {
                inb_0 = false;
            }
            bool _S38;
            if(inb_0)
            {
                _S38 = _S31 < (_S24 - rad_0);
            }
            else
            {
                _S38 = false;
            }
            float ssimacc_1;
            if(_S38)
            {
                ssimacc_1 = ssimacc_0 + val_0;
            }
            else
            {
                ssimacc_1 = ssimacc_0;
            }
            l1acc_0 = l1acc_1;
            ssimacc_0 = ssimacc_1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        c_0 = c_0 + int(1);
    }
    (*(&kernelContext_0)->s_redL1_0)[lidx_0] = l1acc_0;
    (*(&kernelContext_0)->s_redSSIM_0)[lidx_0] = ssimacc_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    i_0 = int(128);
    for(;;)
    {
        if(i_0 > int(0))
        {
        }
        else
        {
            break;
        }
        if(lidx_0 < i_0)
        {
            int _S39 = lidx_0 + i_0;
            (*(&kernelContext_0)->s_redL1_0)[lidx_0] = (*(&kernelContext_0)->s_redL1_0)[lidx_0] + (*(&kernelContext_0)->s_redL1_0)[_S39];
            (*(&kernelContext_0)->s_redSSIM_0)[lidx_0] = (*(&kernelContext_0)->s_redSSIM_0)[lidx_0] + (*(&kernelContext_0)->s_redSSIM_0)[_S39];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        i_0 = i_0 >> 1U;
    }
    if(lidx_0 == int(0))
    {
        int tileIdx_0 = _S28 * ((W_0 + int(16) - int(1)) / int(16)) + _S26;
        *((&kernelContext_0)->tilePartialL1_0+tileIdx_0) = (*(&kernelContext_0)->s_redL1_0)[int(0)];
        *((&kernelContext_0)->tilePartialSSIM_0+tileIdx_0) = (*(&kernelContext_0)->s_redSSIM_0)[int(0)];
    }
    return;
}

