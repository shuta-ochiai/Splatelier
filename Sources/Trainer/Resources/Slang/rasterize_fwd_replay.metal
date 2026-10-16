#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* dims_0;
    int device* tile_offsets_0;
    int device* flatten_ids_0;
    float device* means2d_0;
    float device* conics_0;
    float device* opacities_0;
    int device* cache_id_0;
    half device* colors_0;
    float device* render_color_0;
    float device* background_0;
    float device* render_alpha_0;
    int device* cache_count_0;
};

[[kernel]] void rasterize_fwd_replay(uint3 tid_0 [[thread_position_in_grid]], int device* dims_1 [[buffer(6)]], int device* tile_offsets_1 [[buffer(5)]], int device* flatten_ids_1 [[buffer(4)]], float device* means2d_1 [[buffer(0)]], float device* conics_1 [[buffer(1)]], float device* opacities_1 [[buffer(3)]], int device* cache_id_1 [[buffer(10)]], half device* colors_1 [[buffer(2)]], float device* render_color_1 [[buffer(8)]], float device* background_1 [[buffer(7)]], float device* render_alpha_1 [[buffer(9)]], int device* cache_count_1 [[buffer(11)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->dims_0 = dims_1;
    (&kernelContext_0)->tile_offsets_0 = tile_offsets_1;
    (&kernelContext_0)->flatten_ids_0 = flatten_ids_1;
    (&kernelContext_0)->means2d_0 = means2d_1;
    (&kernelContext_0)->conics_0 = conics_1;
    (&kernelContext_0)->opacities_0 = opacities_1;
    (&kernelContext_0)->cache_id_0 = cache_id_1;
    (&kernelContext_0)->colors_0 = colors_1;
    (&kernelContext_0)->render_color_0 = render_color_1;
    (&kernelContext_0)->background_0 = background_1;
    (&kernelContext_0)->render_alpha_0 = render_alpha_1;
    (&kernelContext_0)->cache_count_0 = cache_count_1;
    int imageWidth_0 = dims_1[int(0)];
    int imageHeight_0 = dims_1[int(1)];
    int tileSize_0 = dims_1[int(2)];
    int numTilesX_0 = dims_1[int(3)];
    int _S1 = dims_1[int(4)];
    int pixel_x_0 = int(tid_0.x);
    int pixel_y_0 = int(tid_0.y);
    bool overflowed_0;
    if(pixel_x_0 >= imageWidth_0)
    {
        overflowed_0 = true;
    }
    else
    {
        overflowed_0 = pixel_y_0 >= imageHeight_0;
    }
    if(overflowed_0)
    {
        return;
    }
    int tile_x_0 = pixel_x_0 / tileSize_0;
    int tile_y_0 = pixel_y_0 / tileSize_0;
    int tile_id_0 = tile_y_0 * numTilesX_0 + tile_x_0;
    int isect_start_0 = (&kernelContext_0)->tile_offsets_0[tile_id_0];
    int _S2 = (&kernelContext_0)->tile_offsets_0[tile_id_0 + int(1)];
    int pix_idx_0 = pixel_y_0 * imageWidth_0 + pixel_x_0;
    int _S3 = imageWidth_0 * imageHeight_0;
    float _S4 = float(pixel_x_0) + 0.5;
    float _S5 = float(pixel_y_0) + 0.5;
    float T_0 = 1.0;
    int k_0 = isect_start_0;
    int local_count_0 = int(0);
    float accum_r_0 = 0.0;
    float accum_g_0 = 0.0;
    float accum_b_0 = 0.0;
    for(;;)
    {
        if(k_0 < _S2)
        {
        }
        else
        {
            overflowed_0 = false;
            break;
        }
        int gi_0 = (&kernelContext_0)->flatten_ids_0[k_0];
        int _S6 = gi_0 * int(2);
        int _S7 = gi_0 * int(3);
        int _S8 = _S7 + int(1);
        int _S9 = _S7 + int(2);
        float dx_0 = _S4 - (&kernelContext_0)->means2d_0[_S6];
        float dy_0 = _S5 - (&kernelContext_0)->means2d_0[_S6 + int(1)];
        float sigma_0 = 0.5 * ((&kernelContext_0)->conics_0[_S7] * dx_0 * dx_0 + (&kernelContext_0)->conics_0[_S9] * dy_0 * dy_0) + (&kernelContext_0)->conics_0[_S8] * dx_0 * dy_0;
        if(sigma_0 < 0.0)
        {
            k_0 = k_0 + int(1);
            continue;
        }
        float _S10 = min(0.99000000953674316, (&kernelContext_0)->opacities_0[gi_0] * exp(- sigma_0));
        if(_S10 < 0.00392156979069114)
        {
            k_0 = k_0 + int(1);
            continue;
        }
        float next_T_0 = T_0 * (1.0 - _S10);
        if(next_T_0 < 0.00009999999747379)
        {
            overflowed_0 = false;
            break;
        }
        if(local_count_0 >= _S1)
        {
            overflowed_0 = true;
            break;
        }
        *((&kernelContext_0)->cache_id_0+(local_count_0 * _S3 + pix_idx_0)) = gi_0;
        int _S11 = local_count_0 + int(1);
        float weight_0 = T_0 * _S10;
        float accum_r_1 = accum_r_0 + weight_0 * float((&kernelContext_0)->colors_0[_S7]);
        float accum_g_1 = accum_g_0 + weight_0 * float((&kernelContext_0)->colors_0[_S8]);
        float accum_b_1 = accum_b_0 + weight_0 * float((&kernelContext_0)->colors_0[_S9]);
        T_0 = next_T_0;
        local_count_0 = _S11;
        accum_r_0 = accum_r_1;
        accum_g_0 = accum_g_1;
        accum_b_0 = accum_b_1;
        k_0 = k_0 + int(1);
    }
    int _S12 = pix_idx_0 * int(3);
    *((&kernelContext_0)->render_color_0+_S12) = accum_r_0 + T_0 * (&kernelContext_0)->background_0[int(0)];
    *((&kernelContext_0)->render_color_0+(_S12 + int(1))) = accum_g_0 + T_0 * (&kernelContext_0)->background_0[int(1)];
    *((&kernelContext_0)->render_color_0+(_S12 + int(2))) = accum_b_0 + T_0 * (&kernelContext_0)->background_0[int(2)];
    *((&kernelContext_0)->render_alpha_0+pix_idx_0) = 1.0 - T_0;
    int device* _S13 = (&kernelContext_0)->cache_count_0+pix_idx_0;
    if(overflowed_0)
    {
        k_0 = int(1073741824);
    }
    else
    {
        k_0 = int(0);
    }
    *_S13 = local_count_0 | k_0;
    return;
}

