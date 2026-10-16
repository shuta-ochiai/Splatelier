#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* dims_0;
    int device* tile_offsets_0;
    int device* last_ids_0;
    float device* render_alpha_0;
    float device* v_render_color_0;
    float device* v_render_alpha_0;
    float device* background_0;
    int device* flatten_ids_0;
    float device* means2d_0;
    float device* conics_0;
    float device* opacities_0;
    half device* colors_0;
    float device* v_colors_0;
    float device* v_opacities_0;
    float device* v_conics_0;
    float device* v_means2d_0;
};

[[kernel]] void rasterize_bwd(uint3 tid_0 [[thread_position_in_grid]], int device* dims_1 [[buffer(6)]], int device* tile_offsets_1 [[buffer(5)]], int device* last_ids_1 [[buffer(9)]], float device* render_alpha_1 [[buffer(8)]], float device* v_render_color_1 [[buffer(10)]], float device* v_render_alpha_1 [[buffer(11)]], float device* background_1 [[buffer(7)]], int device* flatten_ids_1 [[buffer(4)]], float device* means2d_1 [[buffer(0)]], float device* conics_1 [[buffer(1)]], float device* opacities_1 [[buffer(3)]], half device* colors_1 [[buffer(2)]], float device* v_colors_1 [[buffer(14)]], float device* v_opacities_1 [[buffer(15)]], float device* v_conics_1 [[buffer(13)]], float device* v_means2d_1 [[buffer(12)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->dims_0 = dims_1;
    (&kernelContext_0)->tile_offsets_0 = tile_offsets_1;
    (&kernelContext_0)->last_ids_0 = last_ids_1;
    (&kernelContext_0)->render_alpha_0 = render_alpha_1;
    (&kernelContext_0)->v_render_color_0 = v_render_color_1;
    (&kernelContext_0)->v_render_alpha_0 = v_render_alpha_1;
    (&kernelContext_0)->background_0 = background_1;
    (&kernelContext_0)->flatten_ids_0 = flatten_ids_1;
    (&kernelContext_0)->means2d_0 = means2d_1;
    (&kernelContext_0)->conics_0 = conics_1;
    (&kernelContext_0)->opacities_0 = opacities_1;
    (&kernelContext_0)->colors_0 = colors_1;
    (&kernelContext_0)->v_colors_0 = v_colors_1;
    (&kernelContext_0)->v_opacities_0 = v_opacities_1;
    (&kernelContext_0)->v_conics_0 = v_conics_1;
    (&kernelContext_0)->v_means2d_0 = v_means2d_1;
    int imageWidth_0 = dims_1[int(0)];
    int imageHeight_0 = dims_1[int(1)];
    int tileSize_0 = dims_1[int(2)];
    int numTilesX_0 = dims_1[int(3)];
    int pixel_x_0 = int(tid_0.x);
    int pixel_y_0 = int(tid_0.y);
    bool _S1;
    if(pixel_x_0 >= imageWidth_0)
    {
        _S1 = true;
    }
    else
    {
        _S1 = pixel_y_0 >= imageHeight_0;
    }
    if(_S1)
    {
        return;
    }
    int tile_x_0 = pixel_x_0 / tileSize_0;
    int tile_y_0 = pixel_y_0 / tileSize_0;
    int tile_id_0 = tile_y_0 * numTilesX_0 + tile_x_0;
    int isect_start_0 = (&kernelContext_0)->tile_offsets_0[tile_id_0];
    int isect_end_0 = (&kernelContext_0)->tile_offsets_0[tile_id_0 + int(1)];
    int pix_idx_0 = pixel_y_0 * imageWidth_0 + pixel_x_0;
    int last_local_idx_0 = (&kernelContext_0)->last_ids_0[pix_idx_0];
    float _S2 = float(pixel_x_0) + 0.5;
    float _S3 = float(pixel_y_0) + 0.5;
    float T_final_0 = 1.0 - (&kernelContext_0)->render_alpha_0[pix_idx_0];
    int _S4 = pix_idx_0 * int(3);
    float v_render_r_0 = (&kernelContext_0)->v_render_color_0[_S4];
    float v_render_g_0 = (&kernelContext_0)->v_render_color_0[_S4 + int(1)];
    float v_render_b_0 = (&kernelContext_0)->v_render_color_0[_S4 + int(2)];
    float _S5 = (&kernelContext_0)->v_render_alpha_0[pix_idx_0];
    float _S6 = (&kernelContext_0)->background_0[int(0)] * v_render_r_0 + (&kernelContext_0)->background_0[int(1)] * v_render_g_0 + (&kernelContext_0)->background_0[int(2)] * v_render_b_0;
    if(last_local_idx_0 < int(0))
    {
        _S1 = true;
    }
    else
    {
        _S1 = isect_end_0 <= isect_start_0;
    }
    if(_S1)
    {
        return;
    }
    int k_0 = last_local_idx_0;
    float T_0 = T_final_0;
    float buf_r_0 = 0.0;
    float buf_g_0 = 0.0;
    float buf_b_0 = 0.0;
    for(;;)
    {
        if(k_0 >= int(0))
        {
        }
        else
        {
            break;
        }
        int gi_0 = (&kernelContext_0)->flatten_ids_0[isect_start_0 + k_0];
        int _S7 = gi_0 * int(2);
        int _S8 = _S7 + int(1);
        int _S9 = gi_0 * int(3);
        int _S10 = _S9 + int(1);
        float cb_0 = (&kernelContext_0)->conics_0[_S10];
        int _S11 = _S9 + int(2);
        float opac_0 = (&kernelContext_0)->opacities_0[gi_0];
        float dx_0 = _S2 - (&kernelContext_0)->means2d_0[_S7];
        float dy_0 = _S3 - (&kernelContext_0)->means2d_0[_S8];
        float _S12 = (&kernelContext_0)->conics_0[_S9] * dx_0;
        float _S13 = (&kernelContext_0)->conics_0[_S11] * dy_0;
        float _S14 = cb_0 * dx_0;
        float sigma_0 = 0.5 * (_S12 * dx_0 + _S13 * dy_0) + _S14 * dy_0;
        if(sigma_0 < 0.0)
        {
            k_0 = k_0 - int(1);
            continue;
        }
        float vis_0 = exp(- sigma_0);
        float raw_alpha_0 = opac_0 * vis_0;
        float _S15 = min(0.99000000953674316, raw_alpha_0);
        if(_S15 < 0.00392156979069114)
        {
            k_0 = k_0 - int(1);
            continue;
        }
        float ra_0 = 1.0 / max(9.99999997475242708e-07, 1.0 - _S15);
        float T_1 = T_0 * ra_0;
        float weight_0 = T_1 * _S15;
        float _S16 = float((&kernelContext_0)->colors_0[_S9]);
        float _S17 = float((&kernelContext_0)->colors_0[_S10]);
        float _S18 = float((&kernelContext_0)->colors_0[_S11]);
        float vc_g_0 = weight_0 * v_render_g_0;
        float vc_b_0 = weight_0 * v_render_b_0;
        float _S19 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_colors_0+_S9)), weight_0 * v_render_r_0, memory_order_relaxed);
        float _S20 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_colors_0+_S10)), vc_g_0, memory_order_relaxed);
        float _S21 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_colors_0+_S11)), vc_b_0, memory_order_relaxed);
        float _S22 = T_final_0 * ra_0;
        float v_alpha_0 = (_S16 * T_1 - buf_r_0 * ra_0) * v_render_r_0 + (_S17 * T_1 - buf_g_0 * ra_0) * v_render_g_0 + (_S18 * T_1 - buf_b_0 * ra_0) * v_render_b_0 + _S22 * _S5 - _S22 * _S6;
        if(raw_alpha_0 <= 0.99000000953674316)
        {
            float v_sigma_0 = - opac_0 * vis_0 * v_alpha_0;
            float _S23 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_opacities_0+gi_0)), vis_0 * v_alpha_0, memory_order_relaxed);
            float _S24 = 0.5 * v_sigma_0;
            float _S25 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_conics_0+_S9)), _S24 * dx_0 * dx_0, memory_order_relaxed);
            float _S26 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_conics_0+_S10)), v_sigma_0 * dx_0 * dy_0, memory_order_relaxed);
            float _S27 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_conics_0+_S11)), _S24 * dy_0 * dy_0, memory_order_relaxed);
            float _S28 = - v_sigma_0;
            float _S29 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_means2d_0+_S7)), _S28 * (_S12 + cb_0 * dy_0), memory_order_relaxed);
            float _S30 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_means2d_0+_S8)), _S28 * (_S14 + _S13), memory_order_relaxed);
        }
        float buf_r_1 = buf_r_0 + weight_0 * _S16;
        float buf_g_1 = buf_g_0 + weight_0 * _S17;
        float buf_b_1 = buf_b_0 + weight_0 * _S18;
        T_0 = T_1;
        buf_r_0 = buf_r_1;
        buf_g_0 = buf_g_1;
        buf_b_0 = buf_b_1;
        k_0 = k_0 - int(1);
    }
    return;
}

