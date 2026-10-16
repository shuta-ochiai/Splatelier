#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* dims_0;
    float device* gauss_0;
    float device* dL_dmap_0;
    half device* dm_dmu1_0;
    half device* dm_dsigma1_sq_0;
    half device* dm_dsigma12_0;
    float device* img1_0;
    float device* img2_0;
    float device* dL_dimg1_0;
    float device* lossParams_0;
};

[[kernel]] void ssim_bwd(uint3 tid_0 [[thread_position_in_grid]], int device* dims_1 [[buffer(7)]], float device* gauss_1 [[buffer(6)]], float device* dL_dmap_1 [[buffer(2)]], half device* dm_dmu1_1 [[buffer(3)]], half device* dm_dsigma1_sq_1 [[buffer(4)]], half device* dm_dsigma12_1 [[buffer(5)]], float device* img1_1 [[buffer(0)]], float device* img2_1 [[buffer(1)]], float device* dL_dimg1_1 [[buffer(9)]], float device* lossParams_1 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->dims_0 = dims_1;
    (&kernelContext_0)->gauss_0 = gauss_1;
    (&kernelContext_0)->dL_dmap_0 = dL_dmap_1;
    (&kernelContext_0)->dm_dmu1_0 = dm_dmu1_1;
    (&kernelContext_0)->dm_dsigma1_sq_0 = dm_dsigma1_sq_1;
    (&kernelContext_0)->dm_dsigma12_0 = dm_dsigma12_1;
    (&kernelContext_0)->img1_0 = img1_1;
    (&kernelContext_0)->img2_0 = img2_1;
    (&kernelContext_0)->dL_dimg1_0 = dL_dimg1_1;
    (&kernelContext_0)->lossParams_0 = lossParams_1;
    int H_0 = dims_1[int(0)];
    int W_0 = dims_1[int(1)];
    int _S1 = dims_1[int(2)];
    int _S2 = dims_1[int(3)];
    int x_0 = int(tid_0.x);
    int y_0 = int(tid_0.y);
    bool _S3;
    if(x_0 >= W_0)
    {
        _S3 = true;
    }
    else
    {
        _S3 = y_0 >= H_0;
    }
    if(_S3)
    {
        return;
    }
    int c_0 = int(0);
    for(;;)
    {
        if(c_0 < _S1)
        {
        }
        else
        {
            break;
        }
        int _S4 = - _S2;
        int dy_0 = _S4;
        float sum0_0 = 0.0;
        float sum1_0 = 0.0;
        float sum2_0 = 0.0;
        for(;;)
        {
            if(dy_0 <= _S2)
            {
            }
            else
            {
                break;
            }
            int yy_0 = y_0 + dy_0;
            if(yy_0 < int(0))
            {
                _S3 = true;
            }
            else
            {
                _S3 = yy_0 >= H_0;
            }
            float sum0_1;
            float sum1_1;
            float sum2_1;
            if(_S3)
            {
                sum0_1 = sum0_0;
                sum1_1 = sum1_0;
                sum2_1 = sum2_0;
                dy_0 = dy_0 + int(1);
                sum0_0 = sum0_1;
                sum1_0 = sum1_1;
                sum2_0 = sum2_1;
                continue;
            }
            float _S5 = (&kernelContext_0)->gauss_0[dy_0 + _S2];
            int dx_0 = _S4;
            sum0_1 = sum0_0;
            sum1_1 = sum1_0;
            sum2_1 = sum2_0;
            for(;;)
            {
                if(dx_0 <= _S2)
                {
                }
                else
                {
                    break;
                }
                int xx_0 = x_0 + dx_0;
                bool _S6;
                if(xx_0 < int(0))
                {
                    _S6 = true;
                }
                else
                {
                    _S6 = xx_0 >= W_0;
                }
                if(_S6)
                {
                    dx_0 = dx_0 + int(1);
                    continue;
                }
                float w_0 = _S5 * (&kernelContext_0)->gauss_0[dx_0 + _S2];
                int idx_0 = (yy_0 * W_0 + xx_0) * _S1 + c_0;
                float dL_0 = (&kernelContext_0)->dL_dmap_0[idx_0];
                float sum1_2 = sum1_1 + w_0 * float((&kernelContext_0)->dm_dsigma1_sq_0[idx_0]) * dL_0;
                float sum2_2 = sum2_1 + w_0 * float((&kernelContext_0)->dm_dsigma12_0[idx_0]) * dL_0;
                sum0_1 = sum0_1 + w_0 * float((&kernelContext_0)->dm_dmu1_0[idx_0]) * dL_0;
                sum1_1 = sum1_2;
                sum2_1 = sum2_2;
                dx_0 = dx_0 + int(1);
            }
            dy_0 = dy_0 + int(1);
            sum0_0 = sum0_1;
            sum1_0 = sum1_1;
            sum2_0 = sum2_1;
        }
        int o_0 = (y_0 * W_0 + x_0) * _S1 + c_0;
        float p1_0 = (&kernelContext_0)->img1_0[o_0];
        float p2_0 = (&kernelContext_0)->img2_0[o_0];
        *((&kernelContext_0)->dL_dimg1_0+o_0) = sum0_0 + 2.0 * p1_0 * sum1_0 + p2_0 * sum2_0 + (&kernelContext_0)->lossParams_0[int(0)] * float((int(sign((p1_0 - p2_0))))) * (&kernelContext_0)->lossParams_0[int(1)];
        c_0 = c_0 + int(1);
    }
    return;
}

