#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* dims_0;
    float device* luma_0;
    int device* census_0;
};

[[kernel]] void census_transform(uint3 tid_0 [[thread_position_in_grid]], int device* dims_1 [[buffer(1)]], float device* luma_1 [[buffer(0)]], int device* census_1 [[buffer(2)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->dims_0 = dims_1;
    (&kernelContext_0)->luma_0 = luma_1;
    (&kernelContext_0)->census_0 = census_1;
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
    int pix_0 = y_0 * imageWidth_0 + x_0;
    float _S2 = (&kernelContext_0)->luma_0[pix_0];
    int bits_0 = int(0);
    int bit_0 = int(0);
    int dy_0 = int(-2);
    for(;;)
    {
        if(dy_0 <= int(2))
        {
        }
        else
        {
            break;
        }
        int bits_1 = bits_0;
        int bit_1 = bit_0;
        int dx_0 = int(-2);
        for(;;)
        {
            if(dx_0 <= int(2))
            {
            }
            else
            {
                break;
            }
            if(dx_0 == int(0))
            {
                _S1 = dy_0 == int(0);
            }
            else
            {
                _S1 = false;
            }
            if(_S1)
            {
                dx_0 = dx_0 + int(1);
                continue;
            }
            int bits_2;
            if((&kernelContext_0)->luma_0[clamp(y_0 + dy_0, int(0), imageHeight_0 - int(1)) * imageWidth_0 + clamp(x_0 + dx_0, int(0), imageWidth_0 - int(1))] >= _S2)
            {
                bits_2 = bits_1 | (int(1) << uint(bit_1));
            }
            else
            {
                bits_2 = bits_1;
            }
            int _S3 = bit_1 + int(1);
            bits_1 = bits_2;
            bit_1 = _S3;
            dx_0 = dx_0 + int(1);
        }
        int _S4 = dy_0 + int(1);
        bits_0 = bits_1;
        bit_0 = bit_1;
        dy_0 = _S4;
    }
    *((&kernelContext_0)->census_0+pix_0) = bits_0;
    return;
}

