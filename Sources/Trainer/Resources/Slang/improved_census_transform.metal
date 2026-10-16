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

[[kernel]] void improved_census_transform(uint3 tid_0 [[thread_position_in_grid]], int device* dims_1 [[buffer(1)]], float device* luma_1 [[buffer(0)]], int device* census_1 [[buffer(2)]])
{
    int dx_0;
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
    thread array<float, int(25)> win_0;
    int dy_0 = int(-2);
    float sum_0 = 0.0;
    for(;;)
    {
        if(dy_0 <= int(2))
        {
        }
        else
        {
            break;
        }
        dx_0 = int(-2);
        for(;;)
        {
            if(dx_0 <= int(2))
            {
            }
            else
            {
                break;
            }
            float v_0 = (&kernelContext_0)->luma_0[clamp(y_0 + dy_0, int(0), imageHeight_0 - int(1)) * imageWidth_0 + clamp(x_0 + dx_0, int(0), imageWidth_0 - int(1))];
            win_0[(dy_0 + int(2)) * int(5) + (dx_0 + int(2))] = v_0;
            float sum_1 = sum_0 + v_0;
            dx_0 = dx_0 + int(1);
            sum_0 = sum_1;
        }
        dy_0 = dy_0 + int(1);
    }
    float _S2 = sum_0 * 0.03999999910593033;
    dx_0 = int(0);
    int slot_0 = int(0);
    for(;;)
    {
        if(slot_0 < int(25))
        {
        }
        else
        {
            break;
        }
        if((win_0[slot_0]) >= _S2)
        {
            dx_0 = dx_0 | (int(1) << uint(slot_0));
        }
        slot_0 = slot_0 + int(1);
    }
    *((&kernelContext_0)->census_0+(y_0 * imageWidth_0 + x_0)) = dx_0;
    return;
}

