#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
bool farFromFootprint_0(float dx_0, float dy_0, float ca_0, float cb_0, float cc_0)
{
    return (0.5 * (ca_0 * dx_0 * dx_0 + cc_0 * dy_0 * dy_0) + cb_0 * dx_0 * dy_0) > 12.5;
}

struct RayGaussian_0
{
    float A_0;
    float B_0;
    float C_0;
};

RayGaussian_0 rayGaussian_0(float3 m0_0, float3 m1_0, float3 m2_0, float3 og_0, float3 d_0)
{
    float3 dg_0 = float3(dot(m0_0, d_0), dot(m1_0, d_0), dot(m2_0, d_0));
    thread RayGaussian_0 g_0;
    (&g_0)->A_0 = dot(dg_0, dg_0);
    (&g_0)->B_0 = dot(og_0, dg_0);
    (&g_0)->C_0 = dot(og_0, og_0);
    return g_0;
}

float peakAlpha_0(const RayGaussian_0 thread* g_1, float opacity_0, float nearPlane_0, float thread* tStar_0)
{
    *tStar_0 = 0.0;
    float _S1 = g_1->A_0;
    if((g_1->A_0) <= 9.99999968265522539e-21)
    {
        return 0.0;
    }
    float _S2 = g_1->B_0;
    float _S3 = - g_1->B_0 / _S1;
    *tStar_0 = _S3;
    if(_S3 < nearPlane_0)
    {
        return 0.0;
    }
    float _S4 = min(0.99000000953674316, opacity_0 * exp(-0.5 * max(g_1->C_0 - _S2 * _S2 / _S1, 0.0)));
    float _S5;
    if(_S4 < 0.00392156979069114)
    {
        _S5 = 0.0;
    }
    else
    {
        _S5 = _S4;
    }
    return _S5;
}

bool negligibleAhead_0(float peak_0, float A_1, float u_0)
{
    bool _S6;
    if(u_0 < 0.0)
    {
        _S6 = (peak_0 * exp(-0.5 * A_1 * u_0 * u_0)) < 0.00009999999747379;
    }
    else
    {
        _S6 = false;
    }
    return _S6;
}

struct LogTransmittanceGrad_0
{
    float dt_0;
    float dA_0;
    float dB_0;
    float dC_0;
    float dO_0;
};

LogTransmittanceGrad_0 logTransmittanceGrad_0(float A_2, float B_1, float C_1, float opacity_1, float peak_1, bool clamped_0, float tStar_1, float t_0, int profile_0)
{
    float u_1 = t_0 - tStar_1;
    float e_0 = exp(-0.5 * A_2 * u_1 * u_1);
    float gt_0 = peak_1 * e_0;
    float pA_0;
    float pB_0;
    float pC_0;
    float pO_0;
    if(!clamped_0)
    {
        float _S7 = -0.5 * peak_1;
        float _S8 = peak_1 * B_1 / A_2;
        float _S9 = peak_1 / max(opacity_1, 9.999999960041972e-13);
        pA_0 = _S7 * B_1 * B_1 / (A_2 * A_2);
        pB_0 = _S8;
        pC_0 = _S7;
        pO_0 = _S9;
    }
    else
    {
        pA_0 = 0.0;
        pB_0 = 0.0;
        pC_0 = 0.0;
        pO_0 = 0.0;
    }
    float gA_0 = e_0 * pA_0 + gt_0 * (-0.5 * u_1 * u_1 + u_1 * B_1 / A_2);
    float gB_0 = e_0 * pB_0 + gt_0 * - u_1;
    float gC_0 = e_0 * pC_0;
    float gO_0 = e_0 * pO_0;
    float gT_0 = - gt_0 * A_2 * u_1;
    float inv1g_0 = 1.0 / max(1.0 - gt_0, 9.99999997475242708e-07);
    thread LogTransmittanceGrad_0 r_0;
    float k_0;
    if(u_1 < 0.0)
    {
        if(profile_0 == int(0))
        {
            k_0 = 1.0;
        }
        else
        {
            k_0 = 0.5;
        }
        float _S10 = - k_0;
        (&r_0)->dt_0 = _S10 * gT_0 * inv1g_0;
        (&r_0)->dA_0 = _S10 * gA_0 * inv1g_0;
        (&r_0)->dB_0 = _S10 * gB_0 * inv1g_0;
        (&r_0)->dC_0 = _S10 * gC_0 * inv1g_0;
        (&r_0)->dO_0 = _S10 * gO_0 * inv1g_0;
    }
    else
    {
        if(profile_0 == int(0))
        {
            k_0 = 0.0;
        }
        else
        {
            k_0 = 0.5;
        }
        float inv1p_0 = 1.0 / max(1.0 - peak_1, 9.99999997475242708e-07);
        (&r_0)->dt_0 = k_0 * gT_0 * inv1g_0;
        (&r_0)->dA_0 = - pA_0 * inv1p_0 + k_0 * gA_0 * inv1g_0;
        (&r_0)->dB_0 = - pB_0 * inv1p_0 + k_0 * gB_0 * inv1g_0;
        (&r_0)->dC_0 = - pC_0 * inv1p_0 + k_0 * gC_0 * inv1g_0;
        (&r_0)->dO_0 = - pO_0 * inv1p_0 + k_0 * gO_0 * inv1g_0;
    }
    return r_0;
}

struct KernelContext_0
{
    int device* dims_0;
    float device* fwd_depth_0;
    float device* v_depth_0;
    int device* tile_offsets_0;
    float device* depth_params_0;
    float device* camera_0;
    float device* fwd_slope_0;
    int device* flatten_ids_0;
    float device* center_depths_0;
    float device* extents_0;
    float device* means2d_0;
    float device* conics_0;
    float device* frames_0;
    float device* opacities_0;
    float device* v_frames_0;
    float device* v_frame_opacities_0;
    uint _S11;
};

[[kernel]] void gof_depth_bwd_coop(uint3 tid_0 [[thread_position_in_grid]], uint _builtinWaveLaneIndex_0 [[thread_index_in_simdgroup]], int device* dims_1 [[buffer(15)]], float device* fwd_depth_1 [[buffer(20)]], float device* v_depth_1 [[buffer(22)]], int device* tile_offsets_1 [[buffer(14)]], float device* depth_params_1 [[buffer(17)]], float device* camera_1 [[buffer(16)]], float device* fwd_slope_1 [[buffer(21)]], int device* flatten_ids_1 [[buffer(13)]], float device* center_depths_1 [[buffer(9)]], float device* extents_1 [[buffer(8)]], float device* means2d_1 [[buffer(11)]], float device* conics_1 [[buffer(12)]], float device* frames_1 [[buffer(7)]], float device* opacities_1 [[buffer(10)]], float device* v_frames_1 [[buffer(23)]], float device* v_frame_opacities_1 [[buffer(24)]])
{
    float part_0;
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->dims_0 = dims_1;
    (&kernelContext_0)->fwd_depth_0 = fwd_depth_1;
    (&kernelContext_0)->v_depth_0 = v_depth_1;
    (&kernelContext_0)->tile_offsets_0 = tile_offsets_1;
    (&kernelContext_0)->depth_params_0 = depth_params_1;
    (&kernelContext_0)->camera_0 = camera_1;
    (&kernelContext_0)->fwd_slope_0 = fwd_slope_1;
    (&kernelContext_0)->flatten_ids_0 = flatten_ids_1;
    (&kernelContext_0)->center_depths_0 = center_depths_1;
    (&kernelContext_0)->extents_0 = extents_1;
    (&kernelContext_0)->means2d_0 = means2d_1;
    (&kernelContext_0)->conics_0 = conics_1;
    (&kernelContext_0)->frames_0 = frames_1;
    (&kernelContext_0)->opacities_0 = opacities_1;
    (&kernelContext_0)->v_frames_0 = v_frames_1;
    (&kernelContext_0)->v_frame_opacities_0 = v_frame_opacities_1;
    (&kernelContext_0)->_S11 = _builtinWaveLaneIndex_0;
    int imageWidth_0 = dims_1[int(0)];
    int imageHeight_0 = dims_1[int(1)];
    int tileSize_0 = dims_1[int(2)];
    int numTilesX_0 = dims_1[int(3)];
    int stride_0 = dims_1[int(5)];
    int _S12 = int(_builtinWaveLaneIndex_0);
    int sx_0 = int(tid_0.x) / int(32);
    int sy_0 = int(tid_0.y);
    int _S13 = imageWidth_0 / stride_0;
    bool beyond_0;
    if(sx_0 >= _S13)
    {
        beyond_0 = true;
    }
    else
    {
        int _S14 = imageHeight_0 / stride_0;
        beyond_0 = sy_0 >= _S14;
    }
    if(beyond_0)
    {
        return;
    }
    int _S15 = imageWidth_0 / stride_0;
    int pix_idx_0 = sy_0 * _S15 + sx_0;
    float t_1 = (&kernelContext_0)->fwd_depth_0[pix_idx_0];
    float vt_0 = (&kernelContext_0)->v_depth_0[pix_idx_0];
    if(t_1 <= 0.0)
    {
        beyond_0 = true;
    }
    else
    {
        beyond_0 = vt_0 == 0.0;
    }
    if(beyond_0)
    {
        return;
    }
    int pixel_x_0 = sx_0 * stride_0 + stride_0 / int(2);
    int pixel_y_0 = sy_0 * stride_0 + stride_0 / int(2);
    int _S16 = pixel_y_0 / tileSize_0;
    int _S17 = _S16 * numTilesX_0;
    int _S18 = pixel_x_0 / tileSize_0;
    int tile_id_0 = _S17 + _S18;
    int isect_start_0 = (&kernelContext_0)->tile_offsets_0[tile_id_0];
    int _S19 = (&kernelContext_0)->tile_offsets_0[tile_id_0 + int(1)];
    float px_0 = float(pixel_x_0) + 0.5;
    float py_0 = float(pixel_y_0) + 0.5;
    float _S20 = (&kernelContext_0)->depth_params_0[int(0)];
    float _S21 = (&kernelContext_0)->depth_params_0[int(1)];
    int _S22 = int((&kernelContext_0)->depth_params_0[int(2)]);
    float xn_0 = (px_0 - (&kernelContext_0)->camera_0[int(18)]) / (&kernelContext_0)->camera_0[int(16)];
    float yn_0 = (py_0 - (&kernelContext_0)->camera_0[int(19)]) / (&kernelContext_0)->camera_0[int(17)];
    float _S23 = (&kernelContext_0)->camera_0[int(0)] * xn_0 + (&kernelContext_0)->camera_0[int(4)] * yn_0 + (&kernelContext_0)->camera_0[int(8)];
    float _S24 = (&kernelContext_0)->camera_0[int(1)] * xn_0 + (&kernelContext_0)->camera_0[int(5)] * yn_0 + (&kernelContext_0)->camera_0[int(9)];
    float _S25 = (&kernelContext_0)->camera_0[int(2)] * xn_0 + (&kernelContext_0)->camera_0[int(6)] * yn_0 + (&kernelContext_0)->camera_0[int(10)];
    float3 _S26 = float3(_S23, _S24, _S25);
    float slope_0 = (&kernelContext_0)->fwd_slope_0[pix_idx_0];
    bool _S27;
    bool _S28;
    int base_0;
    float slope_1;
    if(slope_0 == 0.0)
    {
        base_0 = isect_start_0;
        slope_1 = 0.0;
        for(;;)
        {
            if(base_0 < _S19)
            {
            }
            else
            {
                part_0 = slope_1;
                break;
            }
            int k_1 = base_0 + _S12;
            if(k_1 < _S19)
            {
                int gi_0 = (&kernelContext_0)->flatten_ids_0[k_1];
                bool beyond_1 = ((&kernelContext_0)->center_depths_0[gi_0] - _S21) > t_1;
                if(!beyond_1)
                {
                    beyond_0 = ((&kernelContext_0)->center_depths_0[gi_0] - (&kernelContext_0)->extents_0[gi_0]) <= t_1;
                }
                else
                {
                    beyond_0 = false;
                }
                if(beyond_0)
                {
                    int _S29 = gi_0 * int(2);
                    int _S30 = gi_0 * int(3);
                    _S27 = !farFromFootprint_0(px_0 - (&kernelContext_0)->means2d_0[_S29], py_0 - (&kernelContext_0)->means2d_0[_S29 + int(1)], (&kernelContext_0)->conics_0[_S30], (&kernelContext_0)->conics_0[_S30 + int(1)], (&kernelContext_0)->conics_0[_S30 + int(2)]);
                }
                else
                {
                    _S27 = false;
                }
                if(_S27)
                {
                    int o_0 = gi_0 * int(12);
                    RayGaussian_0 g_2 = rayGaussian_0(float3((&kernelContext_0)->frames_0[o_0], (&kernelContext_0)->frames_0[o_0 + int(1)], (&kernelContext_0)->frames_0[o_0 + int(2)]), float3((&kernelContext_0)->frames_0[o_0 + int(3)], (&kernelContext_0)->frames_0[o_0 + int(4)], (&kernelContext_0)->frames_0[o_0 + int(5)]), float3((&kernelContext_0)->frames_0[o_0 + int(6)], (&kernelContext_0)->frames_0[o_0 + int(7)], (&kernelContext_0)->frames_0[o_0 + int(8)]), float3((&kernelContext_0)->frames_0[o_0 + int(9)], (&kernelContext_0)->frames_0[o_0 + int(10)], (&kernelContext_0)->frames_0[o_0 + int(11)]), _S26);
                    float _S31 = (&kernelContext_0)->opacities_0[gi_0];
                    thread RayGaussian_0 _S32 = g_2;
                    thread float tStar_2;
                    float _S33 = peakAlpha_0(&_S32, _S31, _S20, &tStar_2);
                    if(_S33 > 0.0)
                    {
                        _S28 = !negligibleAhead_0(_S33, g_2.A_0, t_1 - tStar_2);
                    }
                    else
                    {
                        _S28 = false;
                    }
                    if(_S28)
                    {
                        part_0 = slope_1 + logTransmittanceGrad_0(g_2.A_0, g_2.B_0, g_2.C_0, (&kernelContext_0)->opacities_0[gi_0], _S33, _S33 >= 0.99000000953674316, tStar_2, t_1, _S22).dt_0;
                    }
                    else
                    {
                        part_0 = slope_1;
                    }
                }
                else
                {
                    part_0 = slope_1;
                }
                beyond_0 = beyond_1;
            }
            else
            {
                beyond_0 = true;
                part_0 = slope_1;
            }
            bool _S34 = simd_all(beyond_0);
            if(_S34)
            {
                break;
            }
            base_0 = base_0 + int(32);
            slope_1 = part_0;
        }
        float _S35 = simd_sum(part_0);
        slope_1 = _S35;
    }
    else
    {
        slope_1 = slope_0;
    }
    if(slope_1 > -9.99999993922529029e-09)
    {
        return;
    }
    float _S36 = - vt_0 / slope_1;
    base_0 = isect_start_0;
    for(;;)
    {
        if(base_0 < _S19)
        {
        }
        else
        {
            break;
        }
        int k_2 = base_0 + _S12;
        if(k_2 < _S19)
        {
            int gi_1 = (&kernelContext_0)->flatten_ids_0[k_2];
            bool beyond_2 = ((&kernelContext_0)->center_depths_0[gi_1] - _S21) > t_1;
            if(!beyond_2)
            {
                beyond_0 = ((&kernelContext_0)->center_depths_0[gi_1] - (&kernelContext_0)->extents_0[gi_1]) <= t_1;
            }
            else
            {
                beyond_0 = false;
            }
            if(beyond_0)
            {
                int _S37 = gi_1 * int(2);
                int _S38 = gi_1 * int(3);
                _S27 = !farFromFootprint_0(px_0 - (&kernelContext_0)->means2d_0[_S37], py_0 - (&kernelContext_0)->means2d_0[_S37 + int(1)], (&kernelContext_0)->conics_0[_S38], (&kernelContext_0)->conics_0[_S38 + int(1)], (&kernelContext_0)->conics_0[_S38 + int(2)]);
            }
            else
            {
                _S27 = false;
            }
            if(_S27)
            {
                int o_1 = gi_1 * int(12);
                int _S39 = o_1 + int(1);
                int _S40 = o_1 + int(2);
                float3 m0_1 = float3((&kernelContext_0)->frames_0[o_1], (&kernelContext_0)->frames_0[_S39], (&kernelContext_0)->frames_0[_S40]);
                int _S41 = o_1 + int(3);
                int _S42 = o_1 + int(4);
                int _S43 = o_1 + int(5);
                float3 m1_1 = float3((&kernelContext_0)->frames_0[_S41], (&kernelContext_0)->frames_0[_S42], (&kernelContext_0)->frames_0[_S43]);
                int _S44 = o_1 + int(6);
                int _S45 = o_1 + int(7);
                int _S46 = o_1 + int(8);
                float3 m2_1 = float3((&kernelContext_0)->frames_0[_S44], (&kernelContext_0)->frames_0[_S45], (&kernelContext_0)->frames_0[_S46]);
                int _S47 = o_1 + int(9);
                int _S48 = o_1 + int(10);
                int _S49 = o_1 + int(11);
                float3 og_1 = float3((&kernelContext_0)->frames_0[_S47], (&kernelContext_0)->frames_0[_S48], (&kernelContext_0)->frames_0[_S49]);
                RayGaussian_0 g_3 = rayGaussian_0(m0_1, m1_1, m2_1, og_1, _S26);
                float _S50 = (&kernelContext_0)->opacities_0[gi_1];
                thread RayGaussian_0 _S51 = g_3;
                thread float tStar_3;
                float _S52 = peakAlpha_0(&_S51, _S50, _S20, &tStar_3);
                if(_S52 > 0.0)
                {
                    _S28 = !negligibleAhead_0(_S52, g_3.A_0, t_1 - tStar_3);
                }
                else
                {
                    _S28 = false;
                }
                if(_S28)
                {
                    LogTransmittanceGrad_0 gr_0 = logTransmittanceGrad_0(g_3.A_0, g_3.B_0, g_3.C_0, (&kernelContext_0)->opacities_0[gi_1], _S52, _S52 >= 0.99000000953674316, tStar_3, t_1, _S22);
                    float lC_0 = _S36 * gr_0.dC_0;
                    float3 dg_1 = float3(dot(m0_1, _S26), dot(m1_1, _S26), dot(m2_1, _S26));
                    float3 _S53 = float3((_S36 * gr_0.dB_0)) ;
                    float3 w_0 = float3((2.0 * (_S36 * gr_0.dA_0)))  * dg_1 + _S53 * og_1;
                    float _S54 = w_0.x;
                    float _S55 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_frames_0+o_1)), _S54 * _S23, memory_order_relaxed);
                    float _S56 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_frames_0+_S39)), _S54 * _S24, memory_order_relaxed);
                    float _S57 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_frames_0+_S40)), _S54 * _S25, memory_order_relaxed);
                    float _S58 = w_0.y;
                    float _S59 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_frames_0+_S41)), _S58 * _S23, memory_order_relaxed);
                    float _S60 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_frames_0+_S42)), _S58 * _S24, memory_order_relaxed);
                    float _S61 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_frames_0+_S43)), _S58 * _S25, memory_order_relaxed);
                    float _S62 = w_0.z;
                    float _S63 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_frames_0+_S44)), _S62 * _S23, memory_order_relaxed);
                    float _S64 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_frames_0+_S45)), _S62 * _S24, memory_order_relaxed);
                    float _S65 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_frames_0+_S46)), _S62 * _S25, memory_order_relaxed);
                    float3 vog_0 = _S53 * dg_1 + float3((2.0 * lC_0))  * og_1;
                    float _S66 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_frames_0+_S47)), vog_0.x, memory_order_relaxed);
                    float _S67 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_frames_0+_S48)), vog_0.y, memory_order_relaxed);
                    float _S68 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_frames_0+_S49)), vog_0.z, memory_order_relaxed);
                    float _S69 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->v_frame_opacities_0+gi_1)), _S36 * gr_0.dO_0, memory_order_relaxed);
                }
            }
            beyond_0 = beyond_2;
        }
        else
        {
            beyond_0 = true;
        }
        bool _S70 = simd_all(beyond_0);
        if(_S70)
        {
            break;
        }
        base_0 = base_0 + int(32);
    }
    return;
}

