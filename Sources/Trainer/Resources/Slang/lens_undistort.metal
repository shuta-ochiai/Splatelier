#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
float4 catmullRomWeights_0(float t_0)
{
    float t2_0 = t_0 * t_0;
    float t3_0 = t2_0 * t_0;
    return float4(0.5 * (- t3_0 + 2.0 * t2_0 - t_0), 0.5 * (3.0 * t3_0 - 5.0 * t2_0 + 2.0), 0.5 * (-3.0 * t3_0 + 4.0 * t2_0 + t_0), 0.5 * (t3_0 - t2_0));
}

struct KernelContext_0
{
    int device* dims_0;
    float device* params_0;
    float device* image_0;
    float device* undistorted_0;
};

[[kernel]] void lens_undistort(uint3 tid_0 [[thread_position_in_grid]], int device* dims_1 [[buffer(2)]], float device* params_1 [[buffer(1)]], float device* image_1 [[buffer(0)]], float device* undistorted_1 [[buffer(3)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->dims_0 = dims_1;
    (&kernelContext_0)->params_0 = params_1;
    (&kernelContext_0)->image_0 = image_1;
    (&kernelContext_0)->undistorted_0 = undistorted_1;
    int imageWidth_0 = dims_1[int(0)];
    int imageHeight_0 = dims_1[int(1)];
    int x_0 = int(tid_0.x);
    int y_0 = int(tid_0.y);
    bool _S1;
    if(x_0 >= imageWidth_0)
    {
        _S1 = true;
    }
    else
    {
        _S1 = y_0 >= imageHeight_0;
    }
    if(_S1)
    {
        return;
    }
    float xu_0 = (float(x_0) + 0.5 - (&kernelContext_0)->params_0[int(6)]) / (&kernelContext_0)->params_0[int(4)];
    float yu_0 = (float(y_0) + 0.5 - (&kernelContext_0)->params_0[int(7)]) / (&kernelContext_0)->params_0[int(5)];
    float r2_0 = xu_0 * xu_0 + yu_0 * yu_0;
    float f_0 = 1.0 + (&kernelContext_0)->params_0[int(8)] * r2_0 + (&kernelContext_0)->params_0[int(9)] * r2_0 * r2_0;
    float ui_0 = (&kernelContext_0)->params_0[int(0)] * xu_0 * f_0 + (&kernelContext_0)->params_0[int(2)] - 0.5;
    float vi_0 = (&kernelContext_0)->params_0[int(1)] * yu_0 * f_0 + (&kernelContext_0)->params_0[int(3)] - 0.5;
    float x0_0 = floor(ui_0);
    float y0_0 = floor(vi_0);
    float4 _S2 = catmullRomWeights_0(ui_0 - x0_0);
    float4 _S3 = catmullRomWeights_0(vi_0 - y0_0);
    float3 _S4 = float3(0.0, 0.0, 0.0);
    int j_0 = int(0);
    float3 acc_0 = _S4;
    for(;;)
    {
        if(j_0 < int(4))
        {
        }
        else
        {
            break;
        }
        int _S5 = clamp(int(y0_0) + j_0 - int(1), int(0), imageHeight_0 - int(1));
        int i_0 = int(0);
        float3 row_0 = _S4;
        for(;;)
        {
            if(i_0 < int(4))
            {
            }
            else
            {
                break;
            }
            int p_0 = (_S5 * imageWidth_0 + clamp(int(x0_0) + i_0 - int(1), int(0), imageWidth_0 - int(1))) * int(3);
            float3 row_1 = row_0 + float3(_S2[i_0])  * float3((&kernelContext_0)->image_0[p_0], (&kernelContext_0)->image_0[p_0 + int(1)], (&kernelContext_0)->image_0[p_0 + int(2)]);
            i_0 = i_0 + int(1);
            row_0 = row_1;
        }
        float3 acc_1 = acc_0 + float3(_S3[j_0])  * row_0;
        j_0 = j_0 + int(1);
        acc_0 = acc_1;
    }
    int o_0 = (y_0 * imageWidth_0 + x_0) * int(3);
    *((&kernelContext_0)->undistorted_0+o_0) = acc_0.x;
    *((&kernelContext_0)->undistorted_0+(o_0 + int(1))) = acc_0.y;
    *((&kernelContext_0)->undistorted_0+(o_0 + int(2))) = acc_0.z;
    return;
}

