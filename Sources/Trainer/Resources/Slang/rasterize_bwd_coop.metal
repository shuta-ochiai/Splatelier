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
    float device* v_conics_0;
    float device* v_means2d_0;
    float device* v_opacities_0;
    uint _S1;
};

[[kernel]] void rasterize_bwd_coop(uint3 tid_0 [[thread_position_in_grid]], uint _builtinWaveLaneIndex_0 [[thread_index_in_simdgroup]], int device* dims_1 [[buffer(6)]], int device* tile_offsets_1 [[buffer(5)]], int device* last_ids_1 [[buffer(9)]], float device* render_alpha_1 [[buffer(8)]], float device* v_render_color_1 [[buffer(10)]], float device* v_render_alpha_1 [[buffer(11)]], float device* background_1 [[buffer(7)]], int device* flatten_ids_1 [[buffer(4)]], float device* means2d_1 [[buffer(0)]], float device* conics_1 [[buffer(1)]], float device* opacities_1 [[buffer(3)]], half device* colors_1 [[buffer(2)]], float device* v_colors_1 [[buffer(14)]], float device* v_conics_1 [[buffer(13)]], float device* v_means2d_1 [[buffer(12)]], float device* v_opacities_1 [[buffer(15)]])
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
    (&kernelContext_0)->v_conics_0 = v_conics_1;
    (&kernelContext_0)->v_means2d_0 = v_means2d_1;
    (&kernelContext_0)->v_opacities_0 = v_opacities_1;
    (&kernelContext_0)->_S1 = _builtinWaveLaneIndex_0;
    int imageWidth_0 = dims_1[int(0)];
    int imageHeight_0 = dims_1[int(1)];
    int tileSize_0 = dims_1[int(2)];
    int numTilesX_0 = dims_1[int(3)];
    int pixel_x_0 = int(tid_0.x);
    int pixel_y_0 = int(tid_0.y);
    bool inside_0;
    if(pixel_x_0 < imageWidth_0)
    {
        inside_0 = pixel_y_0 < imageHeight_0;
    }
    else
    {
        inside_0 = false;
    }
    int tile_x_0 = pixel_x_0 / tileSize_0;
    int tile_y_0 = pixel_y_0 / tileSize_0;
    int tile_id_0 = tile_y_0 * numTilesX_0 + tile_x_0;
    int isect_start_0 = (&kernelContext_0)->tile_offsets_0[tile_id_0];
    int isect_count_0 = (&kernelContext_0)->tile_offsets_0[tile_id_0 + int(1)] - isect_start_0;
    int pix_idx_0;
    if(inside_0)
    {
        pix_idx_0 = pixel_y_0 * imageWidth_0 + pixel_x_0;
    }
    else
    {
        pix_idx_0 = int(0);
    }
    int last_local_idx_0;
    if(inside_0)
    {
        last_local_idx_0 = (&kernelContext_0)->last_ids_0[pix_idx_0];
    }
    else
    {
        last_local_idx_0 = int(-1);
    }
    if(isect_count_0 <= int(0))
    {
        last_local_idx_0 = int(-1);
    }
    float _S2 = float(pixel_x_0) + 0.5;
    float _S3 = float(pixel_y_0) + 0.5;
    float T_final_0;
    if(inside_0)
    {
        T_final_0 = 1.0 - (&kernelContext_0)->render_alpha_0[pix_idx_0];
    }
    else
    {
        T_final_0 = 1.0;
    }
    float v_render_r_0;
    if(inside_0)
    {
        v_render_r_0 = (&kernelContext_0)->v_render_color_0[pix_idx_0 * int(3)];
    }
    else
    {
        v_render_r_0 = 0.0;
    }
    float v_render_g_0;
    if(inside_0)
    {
        v_render_g_0 = (&kernelContext_0)->v_render_color_0[pix_idx_0 * int(3) + int(1)];
    }
    else
    {
        v_render_g_0 = 0.0;
    }
    float v_render_b_0;
    if(inside_0)
    {
        v_render_b_0 = (&kernelContext_0)->v_render_color_0[pix_idx_0 * int(3) + int(2)];
    }
    else
    {
        v_render_b_0 = 0.0;
    }
    float _S4;
    if(inside_0)
    {
        _S4 = (&kernelContext_0)->v_render_alpha_0[pix_idx_0];
    }
    else
    {
        _S4 = 0.0;
    }
    float _S5 = (&kernelContext_0)->background_0[int(0)] * v_render_r_0 + (&kernelContext_0)->background_0[int(1)] * v_render_g_0 + (&kernelContext_0)->background_0[int(2)] * v_render_b_0;
    int warp_last_0 = simd_max(last_local_idx_0);
    int k_0 = warp_last_0;
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
        bool inRange_0 = k_0 < isect_count_0;
        int gi_0;
        if(inRange_0)
        {
            gi_0 = (&kernelContext_0)->flatten_ids_0[isect_start_0 + k_0];
        }
        else
        {
            gi_0 = int(0);
        }
        bool _S6;
        if(inside_0)
        {
            _S6 = inRange_0;
        }
        else
        {
            _S6 = false;
        }
        bool valid_0;
        if(_S6)
        {
            valid_0 = k_0 <= last_local_idx_0;
        }
        else
        {
            valid_0 = false;
        }
        bool valid_1;
        float alpha_0;
        float raw_alpha_0;
        float opac_0;
        float vis_0;
        float dx_0;
        float dy_0;
        float ca_0;
        float cb_0;
        float cc_0;
        if(valid_0)
        {
            int _S7 = gi_0 * int(2);
            int _S8 = gi_0 * int(3);
            float _S9 = (&kernelContext_0)->conics_0[_S8];
            float _S10 = (&kernelContext_0)->conics_0[_S8 + int(1)];
            float _S11 = (&kernelContext_0)->conics_0[_S8 + int(2)];
            float opac_1 = (&kernelContext_0)->opacities_0[gi_0];
            float dx_1 = _S2 - (&kernelContext_0)->means2d_0[_S7];
            float dy_1 = _S3 - (&kernelContext_0)->means2d_0[_S7 + int(1)];
            float sigma_0 = 0.5 * (_S9 * dx_1 * dx_1 + _S11 * dy_1 * dy_1) + _S10 * dx_1 * dy_1;
            if(sigma_0 < 0.0)
            {
                valid_1 = false;
                alpha_0 = 0.0;
                raw_alpha_0 = 0.0;
                opac_0 = 0.0;
            }
            else
            {
                float vis_1 = exp(- sigma_0);
                float raw_alpha_1 = opac_1 * vis_1;
                float _S12 = min(0.99000000953674316, raw_alpha_1);
                if(_S12 < 0.00392156979069114)
                {
                    valid_1 = false;
                }
                else
                {
                    valid_1 = valid_0;
                }
                alpha_0 = _S12;
                raw_alpha_0 = raw_alpha_1;
                opac_0 = vis_1;
            }
            float _S13 = opac_0;
            opac_0 = opac_1;
            vis_0 = _S13;
            dx_0 = dx_1;
            dy_0 = dy_1;
            ca_0 = _S9;
            cb_0 = _S10;
            cc_0 = _S11;
        }
        else
        {
            valid_1 = valid_0;
            alpha_0 = 0.0;
            raw_alpha_0 = 0.0;
            opac_0 = 0.0;
            vis_0 = 0.0;
            dx_0 = 0.0;
            dy_0 = 0.0;
            ca_0 = 0.0;
            cb_0 = 0.0;
            cc_0 = 0.0;
        }
        bool _S14 = simd_any(valid_1);
        if(!_S14)
        {
            k_0 = k_0 - int(1);
            continue;
        }
        float vc_r_0;
        float vc_g_0;
        float vc_b_0;
        float vcon_x_0;
        float vcon_y_0;
        float vcon_z_0;
        float vxy_x_0;
        float vxy_y_0;
        float vop_0;
        float T_1;
        float buf_r_1;
        float buf_g_1;
        float buf_b_1;
        if(valid_1)
        {
            int _S15 = gi_0 * int(3);
            float gr_0 = float((&kernelContext_0)->colors_0[_S15]);
            float gg_0 = float((&kernelContext_0)->colors_0[_S15 + int(1)]);
            float gb_0 = float((&kernelContext_0)->colors_0[_S15 + int(2)]);
            float ra_0 = 1.0 / max(9.99999997475242708e-07, 1.0 - alpha_0);
            float T_2 = T_0 * ra_0;
            float weight_0 = T_2 * alpha_0;
            float _S16 = weight_0 * v_render_r_0;
            float _S17 = weight_0 * v_render_g_0;
            float _S18 = weight_0 * v_render_b_0;
            float _S19 = T_final_0 * ra_0;
            float v_alpha_0 = (gr_0 * T_2 - buf_r_0 * ra_0) * v_render_r_0 + (gg_0 * T_2 - buf_g_0 * ra_0) * v_render_g_0 + (gb_0 * T_2 - buf_b_0 * ra_0) * v_render_b_0 + _S19 * _S4 - _S19 * _S5;
            if(raw_alpha_0 <= 0.99000000953674316)
            {
                float v_sigma_0 = - opac_0 * vis_0 * v_alpha_0;
                float _S20 = vis_0 * v_alpha_0;
                float _S21 = 0.5 * v_sigma_0;
                float _S22 = v_sigma_0 * dx_0 * dy_0;
                float _S23 = _S21 * dy_0 * dy_0;
                float _S24 = - v_sigma_0;
                float _S25 = _S24 * (ca_0 * dx_0 + cb_0 * dy_0);
                float _S26 = _S24 * (cb_0 * dx_0 + cc_0 * dy_0);
                vc_r_0 = _S21 * dx_0 * dx_0;
                vc_g_0 = _S22;
                vc_b_0 = _S23;
                vcon_x_0 = _S25;
                vcon_y_0 = _S26;
                vcon_z_0 = _S20;
            }
            else
            {
                vc_r_0 = 0.0;
                vc_g_0 = 0.0;
                vc_b_0 = 0.0;
                vcon_x_0 = 0.0;
                vcon_y_0 = 0.0;
                vcon_z_0 = 0.0;
            }
            float buf_r_2 = buf_r_0 + weight_0 * gr_0;
            float buf_g_2 = buf_g_0 + weight_0 * gg_0;
            float buf_b_2 = buf_b_0 + weight_0 * gb_0;
            float _S27 = vc_r_0;
            float _S28 = vc_g_0;
            float _S29 = vc_b_0;
            float _S30 = vcon_x_0;
            float _S31 = vcon_y_0;
            float _S32 = vcon_z_0;
            vc_r_0 = _S16;
            vc_g_0 = _S17;
            vc_b_0 = _S18;
            vcon_x_0 = _S27;
            vcon_y_0 = _S28;
            vcon_z_0 = _S29;
            vxy_x_0 = _S30;
            vxy_y_0 = _S31;
            vop_0 = _S32;
            T_1 = T_2;
            buf_r_1 = buf_r_2;
            buf_g_1 = buf_g_2;
            buf_b_1 = buf_b_2;
        }
        else
        {
            vc_r_0 = 0.0;
            vc_g_0 = 0.0;
            vc_b_0 = 0.0;
            vcon_x_0 = 0.0;
            vcon_y_0 = 0.0;
            vcon_z_0 = 0.0;
            vxy_x_0 = 0.0;
            vxy_y_0 = 0.0;
            vop_0 = 0.0;
            T_1 = T_0;
            buf_r_1 = buf_r_0;
            buf_g_1 = buf_g_0;
            buf_b_1 = buf_b_0;
        }
        float vc_r_1 = simd_sum(vc_r_0);
        float vc_g_1 = simd_sum(vc_g_0);
        float vc_b_1 = simd_sum(vc_b_0);
        float vcon_x_1 = simd_sum(vcon_x_0);
        float vcon_y_1 = simd_sum(vcon_y_0);
        float vcon_z_1 = simd_sum(vcon_z_0);
        float vxy_x_1 = simd_sum(vxy_x_0);
        float vxy_y_1 = simd_sum(vxy_y_0);
        float vop_1 = simd_sum(vop_0);
        if(((&kernelContext_0)->_S1) == 0U)
        {
            int _S33 = gi_0 * int(3);
            float _S34 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_colors_0+_S33)), vc_r_1, memory_order_relaxed);
            int _S35 = _S33 + int(1);
            float _S36 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_colors_0+_S35)), vc_g_1, memory_order_relaxed);
            int _S37 = _S33 + int(2);
            float _S38 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_colors_0+_S37)), vc_b_1, memory_order_relaxed);
            float _S39 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_conics_0+_S33)), vcon_x_1, memory_order_relaxed);
            float _S40 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_conics_0+_S35)), vcon_y_1, memory_order_relaxed);
            float _S41 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_conics_0+_S37)), vcon_z_1, memory_order_relaxed);
            int _S42 = gi_0 * int(2);
            float _S43 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_means2d_0+_S42)), vxy_x_1, memory_order_relaxed);
            float _S44 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_means2d_0+(_S42 + int(1)))), vxy_y_1, memory_order_relaxed);
            float _S45 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_opacities_0+gi_0)), vop_1, memory_order_relaxed);
        }
        T_0 = T_1;
        buf_r_0 = buf_r_1;
        buf_g_0 = buf_g_1;
        buf_b_0 = buf_b_1;
        k_0 = k_0 - int(1);
    }
    return;
}

