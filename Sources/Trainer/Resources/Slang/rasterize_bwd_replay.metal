#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* dims_0;
    int device* cache_count_0;
    float device* render_alpha_0;
    float device* v_render_alpha_0;
    float device* v_render_color_0;
    float device* render_color_0;
    int device* cache_id_0;
    float device* means2d_0;
    float device* conics_0;
    float device* opacities_0;
    half device* colors_0;
    float device* v_colors_0;
    float device* v_opacities_0;
    float device* v_conics_0;
    float device* v_means2d_0;
};

[[kernel]] void rasterize_bwd_replay(uint3 tid_0 [[thread_position_in_grid]], int device* dims_1 [[buffer(4)]], int device* cache_count_1 [[buffer(8)]], float device* render_alpha_1 [[buffer(6)]], float device* v_render_alpha_1 [[buffer(10)]], float device* v_render_color_1 [[buffer(9)]], float device* render_color_1 [[buffer(5)]], int device* cache_id_1 [[buffer(7)]], float device* means2d_1 [[buffer(0)]], float device* conics_1 [[buffer(1)]], float device* opacities_1 [[buffer(3)]], half device* colors_1 [[buffer(2)]], float device* v_colors_1 [[buffer(13)]], float device* v_opacities_1 [[buffer(14)]], float device* v_conics_1 [[buffer(12)]], float device* v_means2d_1 [[buffer(11)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->dims_0 = dims_1;
    (&kernelContext_0)->cache_count_0 = cache_count_1;
    (&kernelContext_0)->render_alpha_0 = render_alpha_1;
    (&kernelContext_0)->v_render_alpha_0 = v_render_alpha_1;
    (&kernelContext_0)->v_render_color_0 = v_render_color_1;
    (&kernelContext_0)->render_color_0 = render_color_1;
    (&kernelContext_0)->cache_id_0 = cache_id_1;
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
    int pix_idx_0 = pixel_y_0 * imageWidth_0 + pixel_x_0;
    int _S2 = imageWidth_0 * imageHeight_0;
    int K_0 = (&kernelContext_0)->cache_count_0[pix_idx_0] & int(1073741823);
    if(K_0 <= int(0))
    {
        return;
    }
    float _S3 = float(pixel_x_0) + 0.5;
    float _S4 = float(pixel_y_0) + 0.5;
    float _S5 = 1.0 - (&kernelContext_0)->render_alpha_0[pix_idx_0];
    float vra_pixel_0 = (&kernelContext_0)->v_render_alpha_0[pix_idx_0];
    if(vra_pixel_0 == 0.0)
    {
        _S1 = (&kernelContext_0)->v_render_color_0[pix_idx_0 * int(3)] == 0.0;
    }
    else
    {
        _S1 = false;
    }
    if(_S1)
    {
        _S1 = (&kernelContext_0)->v_render_color_0[pix_idx_0 * int(3) + int(1)] == 0.0;
    }
    else
    {
        _S1 = false;
    }
    if(_S1)
    {
        _S1 = (&kernelContext_0)->v_render_color_0[pix_idx_0 * int(3) + int(2)] == 0.0;
    }
    else
    {
        _S1 = false;
    }
    if(_S1)
    {
        return;
    }
    int _S6 = pix_idx_0 * int(3);
    float _S7 = (&kernelContext_0)->render_color_0[_S6];
    int _S8 = _S6 + int(1);
    float _S9 = (&kernelContext_0)->render_color_0[_S8];
    int _S10 = _S6 + int(2);
    float _S11 = (&kernelContext_0)->render_color_0[_S10];
    float _S12 = (&kernelContext_0)->v_render_color_0[_S6];
    float _S13 = (&kernelContext_0)->v_render_color_0[_S8];
    float _S14 = (&kernelContext_0)->v_render_color_0[_S10];
    float T_0 = 1.0;
    float accum_r_0 = 0.0;
    float accum_g_0 = 0.0;
    float accum_b_0 = 0.0;
    int k_0 = int(0);
    float v_R_r_0 = _S12;
    float v_R_g_0 = _S13;
    float v_R_b_0 = _S14;
    for(;;)
    {
        if(k_0 < K_0)
        {
        }
        else
        {
            break;
        }
        int gi_0 = (&kernelContext_0)->cache_id_0[k_0 * _S2 + pix_idx_0];
        int _S15 = gi_0 * int(2);
        int _S16 = _S15 + int(1);
        int _S17 = gi_0 * int(3);
        int _S18 = _S17 + int(1);
        float _S19 = (&kernelContext_0)->conics_0[_S18];
        int _S20 = _S17 + int(2);
        float opac_0 = (&kernelContext_0)->opacities_0[gi_0];
        float _S21 = float((&kernelContext_0)->colors_0[_S17]);
        float _S22 = float((&kernelContext_0)->colors_0[_S18]);
        float _S23 = float((&kernelContext_0)->colors_0[_S20]);
        float dx_0 = _S3 - (&kernelContext_0)->means2d_0[_S15];
        float dy_0 = _S4 - (&kernelContext_0)->means2d_0[_S16];
        float _S24 = (&kernelContext_0)->conics_0[_S17] * dx_0;
        float _S25 = (&kernelContext_0)->conics_0[_S20] * dy_0;
        float _S26 = _S19 * dx_0;
        float vis_0 = exp(- (0.5 * (_S24 * dx_0 + _S25 * dy_0) + _S26 * dy_0));
        float raw_alpha_0 = opac_0 * vis_0;
        float _S27 = min(0.99000000953674316, raw_alpha_0);
        float weight_0 = T_0 * _S27;
        float accum_after_r_0 = accum_r_0 + weight_0 * _S21;
        float accum_after_g_0 = accum_g_0 + weight_0 * _S22;
        float accum_after_b_0 = accum_b_0 + weight_0 * _S23;
        float _S28 = 1.0 - _S27;
        float T_next_0 = T_0 * _S28;
        float _S29 = max(T_next_0, 9.99999997475242708e-07);
        float v_alpha_0 = v_R_r_0 * (_S21 - (_S7 - accum_after_r_0) / _S29) + v_R_g_0 * (_S22 - (_S9 - accum_after_g_0) / _S29) + v_R_b_0 * (_S23 - (_S11 - accum_after_b_0) / _S29) + _S5 * (1.0 / max(9.99999997475242708e-07, _S28)) * vra_pixel_0;
        float v_color_g_0 = _S27 * v_R_g_0;
        float v_color_b_0 = _S27 * v_R_b_0;
        float _S30 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_colors_0+_S17)), _S27 * v_R_r_0, memory_order_relaxed);
        float _S31 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_colors_0+_S18)), v_color_g_0, memory_order_relaxed);
        float _S32 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_colors_0+_S20)), v_color_b_0, memory_order_relaxed);
        if(raw_alpha_0 <= 0.99000000953674316)
        {
            float v_sigma_0 = - opac_0 * vis_0 * v_alpha_0;
            float _S33 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_opacities_0+gi_0)), vis_0 * v_alpha_0, memory_order_relaxed);
            float _S34 = 0.5 * v_sigma_0;
            float _S35 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_conics_0+_S17)), _S34 * dx_0 * dx_0, memory_order_relaxed);
            float _S36 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_conics_0+_S18)), v_sigma_0 * dx_0 * dy_0, memory_order_relaxed);
            float _S37 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_conics_0+_S20)), _S34 * dy_0 * dy_0, memory_order_relaxed);
            float _S38 = - v_sigma_0;
            float _S39 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_means2d_0+_S15)), _S38 * (_S24 + _S19 * dy_0), memory_order_relaxed);
            float _S40 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_means2d_0+_S16)), _S38 * (_S26 + _S25), memory_order_relaxed);
        }
        float v_R_r_1 = v_R_r_0 * _S28;
        float v_R_g_1 = v_R_g_0 * _S28;
        float v_R_b_1 = v_R_b_0 * _S28;
        int _S41 = k_0 + int(1);
        T_0 = T_next_0;
        accum_r_0 = accum_after_r_0;
        accum_g_0 = accum_after_g_0;
        accum_b_0 = accum_after_b_0;
        k_0 = _S41;
        v_R_r_0 = v_R_r_1;
        v_R_g_0 = v_R_g_1;
        v_R_b_0 = v_R_b_1;
    }
    return;
}

