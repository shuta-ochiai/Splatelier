#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
constant array<int, int(16)> CIRCLE_DY_0 = { int(-3), int(-3), int(-2), int(-1), int(0), int(1), int(2), int(3), int(3), int(3), int(2), int(1), int(0), int(-1), int(-2), int(-3) };
constant array<int, int(16)> CIRCLE_DX_0 = { int(0), int(1), int(2), int(3), int(3), int(3), int(2), int(1), int(0), int(-1), int(-2), int(-3), int(-3), int(-3), int(-2), int(-1) };
struct KernelContext_0
{
    int device* dims_0;
    int device* score_map_0;
    float device* luma_0;
};

[[kernel]] void fast_corner(uint3 tid_0 [[thread_position_in_grid]], int device* dims_1 [[buffer(1)]], int device* score_map_1 [[buffer(2)]], float device* luma_1 [[buffer(0)]])
{
    int maxDarkRun_0;
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->dims_0 = dims_1;
    (&kernelContext_0)->score_map_0 = score_map_1;
    (&kernelContext_0)->luma_0 = luma_1;
    int W_0 = dims_1[int(0)];
    int H_0 = dims_1[int(1)];
    int threshold_q8_0 = dims_1[int(2)];
    int x_0 = int(tid_0.x);
    int y_0 = int(tid_0.y);
    bool _S1;
    if(x_0 < int(3))
    {
        _S1 = true;
    }
    else
    {
        _S1 = x_0 >= (W_0 - int(3));
    }
    if(_S1)
    {
        _S1 = true;
    }
    else
    {
        _S1 = y_0 < int(3);
    }
    if(_S1)
    {
        _S1 = true;
    }
    else
    {
        _S1 = y_0 >= (H_0 - int(3));
    }
    if(_S1)
    {
        if(x_0 < W_0)
        {
            _S1 = y_0 < H_0;
        }
        else
        {
            _S1 = false;
        }
        if(_S1)
        {
            *((&kernelContext_0)->score_map_0+(y_0 * W_0 + x_0)) = int(0);
        }
        return;
    }
    int pix_0 = y_0 * W_0 + x_0;
    int ip_0 = int((&kernelContext_0)->luma_0[pix_0] * 256.0);
    int hi_0 = ip_0 + threshold_q8_0;
    int lo_0 = ip_0 - threshold_q8_0;
    thread array<int, int(16)> v_0;
    int k_0 = int(0);
    for(;;)
    {
        if(k_0 < int(16))
        {
        }
        else
        {
            break;
        }
        v_0[k_0] = int((&kernelContext_0)->luma_0[(y_0 + CIRCLE_DY_0[k_0]) * W_0 + (x_0 + CIRCLE_DX_0[k_0])] * 256.0);
        k_0 = k_0 + int(1);
    }
    if((v_0[int(0)]) > hi_0)
    {
        k_0 = int(1);
    }
    else
    {
        k_0 = int(0);
    }
    int maxBrightRun_0;
    if((v_0[int(4)]) > hi_0)
    {
        maxBrightRun_0 = int(1);
    }
    else
    {
        maxBrightRun_0 = int(0);
    }
    int _S2 = k_0 + maxBrightRun_0;
    if((v_0[int(8)]) > hi_0)
    {
        k_0 = int(1);
    }
    else
    {
        k_0 = int(0);
    }
    int _S3 = _S2 + k_0;
    if((v_0[int(12)]) > hi_0)
    {
        k_0 = int(1);
    }
    else
    {
        k_0 = int(0);
    }
    int hiCount_0 = _S3 + k_0;
    if((v_0[int(0)]) < lo_0)
    {
        k_0 = int(1);
    }
    else
    {
        k_0 = int(0);
    }
    if((v_0[int(4)]) < lo_0)
    {
        maxBrightRun_0 = int(1);
    }
    else
    {
        maxBrightRun_0 = int(0);
    }
    int _S4 = k_0 + maxBrightRun_0;
    if((v_0[int(8)]) < lo_0)
    {
        k_0 = int(1);
    }
    else
    {
        k_0 = int(0);
    }
    int _S5 = _S4 + k_0;
    if((v_0[int(12)]) < lo_0)
    {
        k_0 = int(1);
    }
    else
    {
        k_0 = int(0);
    }
    int loCount_0 = _S5 + k_0;
    if(hiCount_0 < int(3))
    {
        _S1 = loCount_0 < int(3);
    }
    else
    {
        _S1 = false;
    }
    if(_S1)
    {
        *((&kernelContext_0)->score_map_0+pix_0) = int(0);
        return;
    }
    maxBrightRun_0 = int(0);
    int maxDarkRun_1 = int(0);
    int j_0 = int(0);
    int curBright_0 = int(0);
    int curDark_0 = int(0);
    for(;;)
    {
        if(j_0 < int(32))
        {
        }
        else
        {
            break;
        }
        int vj_0 = v_0[j_0 & int(15)];
        if((v_0[j_0 & int(15)]) > hi_0)
        {
            int curBright_1 = curBright_0 + int(1);
            if(curBright_1 > maxBrightRun_0)
            {
                maxDarkRun_0 = curBright_1;
            }
            else
            {
                maxDarkRun_0 = maxBrightRun_0;
            }
            maxBrightRun_0 = maxDarkRun_0;
            maxDarkRun_0 = maxDarkRun_1;
            curBright_0 = curBright_1;
            curDark_0 = int(0);
        }
        else
        {
            int curDark_1;
            if(vj_0 < lo_0)
            {
                int curDark_2 = curDark_0 + int(1);
                if(curDark_2 > maxDarkRun_1)
                {
                    maxDarkRun_0 = curDark_2;
                }
                else
                {
                    maxDarkRun_0 = maxDarkRun_1;
                }
                curDark_1 = curDark_2;
            }
            else
            {
                maxDarkRun_0 = maxDarkRun_1;
                curDark_1 = int(0);
            }
            curBright_0 = int(0);
            curDark_0 = curDark_1;
        }
        int _S6 = j_0 + int(1);
        maxDarkRun_1 = maxDarkRun_0;
        j_0 = _S6;
    }
    if(maxBrightRun_0 < int(9))
    {
        _S1 = maxDarkRun_1 < int(9);
    }
    else
    {
        _S1 = false;
    }
    if(_S1)
    {
        *((&kernelContext_0)->score_map_0+pix_0) = int(0);
        return;
    }
    if(maxBrightRun_0 >= int(9))
    {
        k_0 = int(0);
        maxDarkRun_0 = int(0);
        for(;;)
        {
            if(k_0 < int(16))
            {
            }
            else
            {
                break;
            }
            if((v_0[k_0]) > hi_0)
            {
                maxDarkRun_0 = maxDarkRun_0 + (v_0[k_0] - ip_0);
            }
            k_0 = k_0 + int(1);
        }
    }
    else
    {
        k_0 = int(0);
        maxDarkRun_0 = int(0);
        for(;;)
        {
            if(k_0 < int(16))
            {
            }
            else
            {
                break;
            }
            if((v_0[k_0]) < lo_0)
            {
                maxDarkRun_0 = maxDarkRun_0 + (ip_0 - v_0[k_0]);
            }
            k_0 = k_0 + int(1);
        }
    }
    *((&kernelContext_0)->score_map_0+pix_0) = maxDarkRun_0;
    return;
}

