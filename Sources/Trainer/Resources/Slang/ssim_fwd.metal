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
    float device* gauss_0;
    float device* img1_0;
    float device* img2_0;
    half device* ssim_map_0;
    half device* dm_dmu1_0;
    half device* dm_dsigma1_sq_0;
    half device* dm_dsigma12_0;
};

[[kernel]] void ssim_fwd(uint3 tid_0 [[thread_position_in_grid]], int device* dims_1 [[buffer(3)]], float device* gauss_1 [[buffer(2)]], float device* img1_1 [[buffer(0)]], float device* img2_1 [[buffer(1)]], half device* ssim_map_1 [[buffer(4)]], half device* dm_dmu1_1 [[buffer(5)]], half device* dm_dsigma1_sq_1 [[buffer(6)]], half device* dm_dsigma12_1 [[buffer(7)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->dims_0 = dims_1;
    (&kernelContext_0)->gauss_0 = gauss_1;
    (&kernelContext_0)->img1_0 = img1_1;
    (&kernelContext_0)->img2_0 = img2_1;
    (&kernelContext_0)->ssim_map_0 = ssim_map_1;
    (&kernelContext_0)->dm_dmu1_0 = dm_dmu1_1;
    (&kernelContext_0)->dm_dsigma1_sq_0 = dm_dsigma1_sq_1;
    (&kernelContext_0)->dm_dsigma12_0 = dm_dsigma12_1;
    int H_0 = dims_1[int(0)];
    int W_0 = dims_1[int(1)];
    int _S22 = dims_1[int(2)];
    int _S23 = dims_1[int(3)];
    int x_0 = int(tid_0.x);
    int y_0 = int(tid_0.y);
    bool _S24;
    if(x_0 >= W_0)
    {
        _S24 = true;
    }
    else
    {
        _S24 = y_0 >= H_0;
    }
    if(_S24)
    {
        return;
    }
    int c_0 = int(0);
    for(;;)
    {
        if(c_0 < _S22)
        {
        }
        else
        {
            break;
        }
        int _S25 = - _S23;
        int dy_0 = _S25;
        float sumX_0 = 0.0;
        float sumX2_0 = 0.0;
        float sumY_0 = 0.0;
        float sumY2_0 = 0.0;
        float sumXY_0 = 0.0;
        for(;;)
        {
            if(dy_0 <= _S23)
            {
            }
            else
            {
                break;
            }
            int yy_0 = y_0 + dy_0;
            if(yy_0 < int(0))
            {
                _S24 = true;
            }
            else
            {
                _S24 = yy_0 >= H_0;
            }
            float sumX_1;
            float sumX2_1;
            float sumY_1;
            float sumY2_1;
            float sumXY_1;
            if(_S24)
            {
                sumX_1 = sumX_0;
                sumX2_1 = sumX2_0;
                sumY_1 = sumY_0;
                sumY2_1 = sumY2_0;
                sumXY_1 = sumXY_0;
                dy_0 = dy_0 + int(1);
                sumX_0 = sumX_1;
                sumX2_0 = sumX2_1;
                sumY_0 = sumY_1;
                sumY2_0 = sumY2_1;
                sumXY_0 = sumXY_1;
                continue;
            }
            float _S26 = (&kernelContext_0)->gauss_0[dy_0 + _S23];
            int dx_0 = _S25;
            sumX_1 = sumX_0;
            sumX2_1 = sumX2_0;
            sumY_1 = sumY_0;
            sumY2_1 = sumY2_0;
            sumXY_1 = sumXY_0;
            for(;;)
            {
                if(dx_0 <= _S23)
                {
                }
                else
                {
                    break;
                }
                int xx_0 = x_0 + dx_0;
                bool _S27;
                if(xx_0 < int(0))
                {
                    _S27 = true;
                }
                else
                {
                    _S27 = xx_0 >= W_0;
                }
                if(_S27)
                {
                    dx_0 = dx_0 + int(1);
                    continue;
                }
                float w_0 = _S26 * (&kernelContext_0)->gauss_0[dx_0 + _S23];
                int idx_0 = (yy_0 * W_0 + xx_0) * _S22 + c_0;
                float a_0 = (&kernelContext_0)->img1_0[idx_0];
                float b_0 = (&kernelContext_0)->img2_0[idx_0];
                float _S28 = w_0 * a_0;
                float sumX2_2 = sumX2_1 + _S28 * a_0;
                float _S29 = w_0 * b_0;
                float sumY_2 = sumY_1 + _S29;
                float sumY2_2 = sumY2_1 + _S29 * b_0;
                float sumXY_2 = sumXY_1 + _S28 * b_0;
                sumX_1 = sumX_1 + _S28;
                sumX2_1 = sumX2_2;
                sumY_1 = sumY_2;
                sumY2_1 = sumY2_2;
                sumXY_1 = sumXY_2;
                dx_0 = dx_0 + int(1);
            }
            dy_0 = dy_0 + int(1);
            sumX_0 = sumX_1;
            sumX2_0 = sumX2_1;
            sumY_0 = sumY_1;
            sumY2_0 = sumY2_1;
            sumXY_0 = sumXY_1;
        }
        SsimGrad_0 g_1 = ssim_pixel_grad_0(sumX_0, sumX2_0, sumXY_0, sumY_0, sumY2_0, 0.00009999999747379, 0.00089999998454005);
        int o_0 = (y_0 * W_0 + x_0) * _S22 + c_0;
        *((&kernelContext_0)->ssim_map_0+o_0) = half(ssim_pixel_0(sumX_0, sumX2_0, sumXY_0, sumY_0, sumY2_0, 0.00009999999747379, 0.00089999998454005));
        *((&kernelContext_0)->dm_dmu1_0+o_0) = half(g_1.d_mu1_0);
        *((&kernelContext_0)->dm_dsigma1_sq_0+o_0) = half(g_1.d_m1x2_0);
        *((&kernelContext_0)->dm_dsigma12_0+o_0) = half(g_1.d_m1xy_0);
        c_0 = c_0 + int(1);
    }
    return;
}

