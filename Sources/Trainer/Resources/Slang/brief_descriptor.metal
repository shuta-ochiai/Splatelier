#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* dims_0;
    int device* kp_xy_0;
    int device* descriptors_0;
    float device* luma_0;
    int device* pattern_0;
};

[[kernel]] void brief_descriptor(uint3 tid_0 [[thread_position_in_grid]], int device* dims_1 [[buffer(3)]], int device* kp_xy_1 [[buffer(1)]], int device* descriptors_1 [[buffer(4)]], float device* luma_1 [[buffer(0)]], int device* pattern_1 [[buffer(2)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->dims_0 = dims_1;
    (&kernelContext_0)->kp_xy_0 = kp_xy_1;
    (&kernelContext_0)->descriptors_0 = descriptors_1;
    (&kernelContext_0)->luma_0 = luma_1;
    (&kernelContext_0)->pattern_0 = pattern_1;
    int idx_0 = int(tid_0.x);
    int W_0 = dims_1[int(0)];
    int H_0 = dims_1[int(1)];
    if(idx_0 >= dims_1[int(2)])
    {
        return;
    }
    int _S1 = idx_0 * int(2);
    int kx_0 = (&kernelContext_0)->kp_xy_0[_S1];
    int ky_0 = (&kernelContext_0)->kp_xy_0[_S1 + int(1)];
    bool _S2;
    if(kx_0 < int(15))
    {
        _S2 = true;
    }
    else
    {
        _S2 = kx_0 >= (W_0 - int(15));
    }
    if(_S2)
    {
        _S2 = true;
    }
    else
    {
        _S2 = ky_0 < int(15);
    }
    if(_S2)
    {
        _S2 = true;
    }
    else
    {
        _S2 = ky_0 >= (H_0 - int(15));
    }
    int w_0;
    if(_S2)
    {
        w_0 = int(0);
        for(;;)
        {
            if(w_0 < int(8))
            {
            }
            else
            {
                break;
            }
            *((&kernelContext_0)->descriptors_0+(idx_0 * int(8) + w_0)) = int(0);
            w_0 = w_0 + int(1);
        }
        return;
    }
    int dy_0 = int(-15);
    float m10_0 = 0.0;
    float m01_0 = 0.0;
    for(;;)
    {
        if(dy_0 <= int(15))
        {
        }
        else
        {
            break;
        }
        w_0 = int(-15);
        float m10_1 = m10_0;
        float m01_1 = m01_0;
        for(;;)
        {
            if(w_0 <= int(15))
            {
            }
            else
            {
                break;
            }
            if((w_0 * w_0 + dy_0 * dy_0) > int(225))
            {
                w_0 = w_0 + int(1);
                continue;
            }
            float v_0 = (&kernelContext_0)->luma_0[(ky_0 + dy_0) * W_0 + (kx_0 + w_0)];
            float m01_2 = m01_1 + float(dy_0) * v_0;
            m10_1 = m10_1 + float(w_0) * v_0;
            m01_1 = m01_2;
            w_0 = w_0 + int(1);
        }
        dy_0 = dy_0 + int(1);
        m10_0 = m10_1;
        m01_0 = m01_1;
    }
    float theta_0 = atan2(m01_0, m10_0);
    float _S3 = cos(theta_0);
    float _S4 = sin(theta_0);
    int _S5 = idx_0 * int(8);
    w_0 = int(0);
    for(;;)
    {
        if(w_0 < int(8))
        {
        }
        else
        {
            break;
        }
        int bits_0 = int(0);
        int b_0 = int(0);
        for(;;)
        {
            if(b_0 < int(32))
            {
            }
            else
            {
                break;
            }
            int _S6 = (w_0 * int(32) + b_0) * int(4);
            float pxa_0 = float((&kernelContext_0)->pattern_0[_S6]);
            float pya_0 = float((&kernelContext_0)->pattern_0[_S6 + int(1)]);
            float pxb_0 = float((&kernelContext_0)->pattern_0[_S6 + int(2)]);
            float pyb_0 = float((&kernelContext_0)->pattern_0[_S6 + int(3)]);
            int _S7 = W_0 - int(1);
            int _S8 = H_0 - int(1);
            if((&kernelContext_0)->luma_0[clamp(ky_0 + int(round(pxa_0 * _S4 + pya_0 * _S3)), int(0), _S8) * W_0 + clamp(kx_0 + int(round(pxa_0 * _S3 - pya_0 * _S4)), int(0), _S7)] > ((&kernelContext_0)->luma_0)[clamp(ky_0 + int(round(pxb_0 * _S4 + pyb_0 * _S3)), int(0), _S8) * W_0 + clamp(kx_0 + int(round(pxb_0 * _S3 - pyb_0 * _S4)), int(0), _S7)])
            {
                bits_0 = bits_0 | (int(1) << uint(b_0));
            }
            b_0 = b_0 + int(1);
        }
        *((&kernelContext_0)->descriptors_0+(_S5 + w_0)) = bits_0;
        w_0 = w_0 + int(1);
    }
    return;
}

