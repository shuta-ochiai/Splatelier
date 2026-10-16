#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* dims_0;
    int device* cache_count_0;
    int device* cache_id_0;
    float device* means2d_0;
    float device* conics_0;
    float device* opacities_0;
    half device* features_0;
    float device* rendered_0;
};

[[kernel]] void features_replay(uint3 tid_0 [[thread_position_in_grid]], int device* dims_1 [[buffer(4)]], int device* cache_count_1 [[buffer(6)]], int device* cache_id_1 [[buffer(5)]], float device* means2d_1 [[buffer(0)]], float device* conics_1 [[buffer(1)]], float device* opacities_1 [[buffer(3)]], half device* features_1 [[buffer(2)]], float device* rendered_1 [[buffer(7)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->dims_0 = dims_1;
    (&kernelContext_0)->cache_count_0 = cache_count_1;
    (&kernelContext_0)->cache_id_0 = cache_id_1;
    (&kernelContext_0)->means2d_0 = means2d_1;
    (&kernelContext_0)->conics_0 = conics_1;
    (&kernelContext_0)->opacities_0 = opacities_1;
    (&kernelContext_0)->features_0 = features_1;
    (&kernelContext_0)->rendered_0 = rendered_1;
    int imageWidth_0 = dims_1[int(0)];
    int imageHeight_0 = dims_1[int(1)];
    int stride_0 = dims_1[int(2)];
    int _S1 = int(tid_0.x);
    int _S2 = imageWidth_0 / stride_0;
    bool _S3;
    if(_S1 >= _S2)
    {
        _S3 = true;
    }
    else
    {
        int _S4 = int(tid_0.y);
        int _S5 = imageHeight_0 / stride_0;
        _S3 = _S4 >= _S5;
    }
    if(_S3)
    {
        return;
    }
    int pixel_x_0 = _S1 * stride_0 + stride_0 / int(2);
    int pixel_y_0 = int(tid_0.y) * stride_0 + stride_0 / int(2);
    int pix_idx_0 = pixel_y_0 * imageWidth_0 + pixel_x_0;
    int _S6 = imageWidth_0 * imageHeight_0;
    float _S7 = float(pixel_x_0) + 0.5;
    float _S8 = float(pixel_y_0) + 0.5;
    int _S9 = (&kernelContext_0)->cache_count_0[pix_idx_0] & int(1073741823);
    float3 _S10 = float3(0.0, 0.0, 0.0);
    int k_0 = int(0);
    float T_0 = 1.0;
    float3 accum_0 = _S10;
    for(;;)
    {
        if(k_0 < _S9)
        {
        }
        else
        {
            break;
        }
        int gi_0 = (&kernelContext_0)->cache_id_0[k_0 * _S6 + pix_idx_0];
        int _S11 = gi_0 * int(2);
        float dx_0 = _S7 - (&kernelContext_0)->means2d_0[_S11];
        float dy_0 = _S8 - (&kernelContext_0)->means2d_0[_S11 + int(1)];
        int _S12 = gi_0 * int(3);
        int _S13 = _S12 + int(2);
        int _S14 = _S12 + int(1);
        float _S15 = min(0.99000000953674316, (&kernelContext_0)->opacities_0[gi_0] * exp(- (0.5 * ((&kernelContext_0)->conics_0[_S12] * dx_0 * dx_0 + (&kernelContext_0)->conics_0[_S13] * dy_0 * dy_0) + (&kernelContext_0)->conics_0[_S14] * dx_0 * dy_0)));
        float3 accum_1 = accum_0 + float3((T_0 * _S15))  * float3(float((&kernelContext_0)->features_0[_S12]), float((&kernelContext_0)->features_0[_S14]), float((&kernelContext_0)->features_0[_S13]));
        float T_1 = T_0 * (1.0 - _S15);
        k_0 = k_0 + int(1);
        T_0 = T_1;
        accum_0 = accum_1;
    }
    int _S16 = pix_idx_0 * int(3);
    *((&kernelContext_0)->rendered_0+_S16) = accum_0.x;
    *((&kernelContext_0)->rendered_0+(_S16 + int(1))) = accum_0.y;
    *((&kernelContext_0)->rendered_0+(_S16 + int(2))) = accum_0.z;
    return;
}

