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

float transmittanceAt_0(float peak_0, float tStar_1, float A_1, float t_0, int profile_0)
{
    float dt_0 = t_0 - tStar_1;
    float gt_0 = peak_0 * exp(-0.5 * A_1 * dt_0 * dt_0);
    float _S6;
    if(profile_0 == int(0))
    {
        if(dt_0 >= 0.0)
        {
            _S6 = 1.0 - peak_0;
        }
        else
        {
            _S6 = 1.0 - gt_0;
        }
        return _S6;
    }
    float v_0 = sqrt(1.0 - gt_0);
    if(dt_0 <= 0.0)
    {
        _S6 = v_0;
    }
    else
    {
        _S6 = (1.0 - peak_0) / v_0;
    }
    return _S6;
}

struct KernelContext_0
{
    int device* dims_0;
    int device* tile_offsets_0;
    float device* depth_params_0;
    float device* camera_0;
    int device* flatten_ids_0;
    float device* means2d_0;
    float device* conics_0;
    float device* frames_0;
    float device* opacities_0;
    float device* surface_depth_0;
    float device* surface_slope_0;
    float device* center_depths_0;
    float device* extents_0;
    uint _S7;
};

[[kernel]] void gof_depth_coop(uint3 tid_0 [[thread_position_in_grid]], uint _builtinWaveLaneIndex_0 [[thread_index_in_simdgroup]], int device* dims_1 [[buffer(15)]], int device* tile_offsets_1 [[buffer(14)]], float device* depth_params_1 [[buffer(17)]], float device* camera_1 [[buffer(16)]], int device* flatten_ids_1 [[buffer(13)]], float device* means2d_1 [[buffer(11)]], float device* conics_1 [[buffer(12)]], float device* frames_1 [[buffer(7)]], float device* opacities_1 [[buffer(10)]], float device* surface_depth_1 [[buffer(18)]], float device* surface_slope_1 [[buffer(19)]], float device* center_depths_1 [[buffer(9)]], float device* extents_1 [[buffer(8)]])
{
    bool hiExact_0;
    bool crossed_0;
    float lf_0;
    float tS_0;
    float sg_0;
    float ext_0;
    int pass_0;
    int j_0;
    int seg_0;
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->dims_0 = dims_1;
    (&kernelContext_0)->tile_offsets_0 = tile_offsets_1;
    (&kernelContext_0)->depth_params_0 = depth_params_1;
    (&kernelContext_0)->camera_0 = camera_1;
    (&kernelContext_0)->flatten_ids_0 = flatten_ids_1;
    (&kernelContext_0)->means2d_0 = means2d_1;
    (&kernelContext_0)->conics_0 = conics_1;
    (&kernelContext_0)->frames_0 = frames_1;
    (&kernelContext_0)->opacities_0 = opacities_1;
    (&kernelContext_0)->surface_depth_0 = surface_depth_1;
    (&kernelContext_0)->surface_slope_0 = surface_slope_1;
    (&kernelContext_0)->center_depths_0 = center_depths_1;
    (&kernelContext_0)->extents_0 = extents_1;
    (&kernelContext_0)->_S7 = _builtinWaveLaneIndex_0;
    int imageWidth_0 = dims_1[int(0)];
    int imageHeight_0 = dims_1[int(1)];
    int tileSize_0 = dims_1[int(2)];
    int numTilesX_0 = dims_1[int(3)];
    int iterations_0 = dims_1[int(4)];
    int stride_0 = dims_1[int(5)];
    int lane_0 = int(_builtinWaveLaneIndex_0);
    int sx_0 = int(tid_0.x) / int(32);
    int sy_0 = int(tid_0.y);
    int _S8 = imageWidth_0 / stride_0;
    bool reached_0;
    if(sx_0 >= _S8)
    {
        reached_0 = true;
    }
    else
    {
        int _S9 = imageHeight_0 / stride_0;
        reached_0 = sy_0 >= _S9;
    }
    if(reached_0)
    {
        return;
    }
    int pixel_x_0 = sx_0 * stride_0 + stride_0 / int(2);
    int pixel_y_0 = sy_0 * stride_0 + stride_0 / int(2);
    int _S10 = pixel_y_0 / tileSize_0;
    int _S11 = _S10 * numTilesX_0;
    int _S12 = pixel_x_0 / tileSize_0;
    int tile_id_0 = _S11 + _S12;
    int isect_start_0 = (&kernelContext_0)->tile_offsets_0[tile_id_0];
    int _S13 = (&kernelContext_0)->tile_offsets_0[tile_id_0 + int(1)];
    int _S14 = imageWidth_0 / stride_0;
    int pix_idx_0 = sy_0 * _S14 + sx_0;
    float px_0 = float(pixel_x_0) + 0.5;
    float py_0 = float(pixel_y_0) + 0.5;
    float nearPlane_1 = (&kernelContext_0)->depth_params_0[int(0)];
    float _S15 = (&kernelContext_0)->depth_params_0[int(1)];
    int _S16 = int((&kernelContext_0)->depth_params_0[int(2)]);
    float xn_0 = (px_0 - (&kernelContext_0)->camera_0[int(18)]) / (&kernelContext_0)->camera_0[int(16)];
    float yn_0 = (py_0 - (&kernelContext_0)->camera_0[int(19)]) / (&kernelContext_0)->camera_0[int(17)];
    float3 _S17 = float3((&kernelContext_0)->camera_0[int(0)] * xn_0 + (&kernelContext_0)->camera_0[int(4)] * yn_0 + (&kernelContext_0)->camera_0[int(8)], (&kernelContext_0)->camera_0[int(1)] * xn_0 + (&kernelContext_0)->camera_0[int(5)] * yn_0 + (&kernelContext_0)->camera_0[int(9)], (&kernelContext_0)->camera_0[int(2)] * xn_0 + (&kernelContext_0)->camera_0[int(6)] * yn_0 + (&kernelContext_0)->camera_0[int(10)]);
    reached_0 = false;
    float tHi_0 = nearPlane_1;
    float tCross_0 = nearPlane_1;
    float sCross_0 = 0.0;
    int base_0 = isect_start_0;
    float logT_0 = 0.0;
    for(;;)
    {
        if(base_0 < _S13)
        {
            hiExact_0 = !reached_0;
        }
        else
        {
            hiExact_0 = false;
        }
        if(hiExact_0)
        {
        }
        else
        {
            break;
        }
        int k_0 = base_0 + lane_0;
        if(k_0 < _S13)
        {
            int gi_0 = (&kernelContext_0)->flatten_ids_0[k_0];
            int _S18 = gi_0 * int(2);
            int _S19 = gi_0 * int(3);
            if(!farFromFootprint_0(px_0 - (&kernelContext_0)->means2d_0[_S18], py_0 - (&kernelContext_0)->means2d_0[_S18 + int(1)], (&kernelContext_0)->conics_0[_S19], (&kernelContext_0)->conics_0[_S19 + int(1)], (&kernelContext_0)->conics_0[_S19 + int(2)]))
            {
                int o_0 = gi_0 * int(12);
                RayGaussian_0 g_2 = rayGaussian_0(float3((&kernelContext_0)->frames_0[o_0], (&kernelContext_0)->frames_0[o_0 + int(1)], (&kernelContext_0)->frames_0[o_0 + int(2)]), float3((&kernelContext_0)->frames_0[o_0 + int(3)], (&kernelContext_0)->frames_0[o_0 + int(4)], (&kernelContext_0)->frames_0[o_0 + int(5)]), float3((&kernelContext_0)->frames_0[o_0 + int(6)], (&kernelContext_0)->frames_0[o_0 + int(7)], (&kernelContext_0)->frames_0[o_0 + int(8)]), float3((&kernelContext_0)->frames_0[o_0 + int(9)], (&kernelContext_0)->frames_0[o_0 + int(10)], (&kernelContext_0)->frames_0[o_0 + int(11)]), _S17);
                float _S20 = (&kernelContext_0)->opacities_0[gi_0];
                thread RayGaussian_0 _S21 = g_2;
                thread float tStar_2;
                float _S22 = peakAlpha_0(&_S21, _S20, nearPlane_1, &tStar_2);
                if(_S22 > 0.0)
                {
                    float _S23 = log(1.0 - _S22);
                    float _S24 = tStar_2;
                    float sg_1 = 1.0 / sqrt(g_2.A_0);
                    if(_S16 == int(0))
                    {
                        lf_0 = tStar_2;
                    }
                    else
                    {
                        lf_0 = tStar_2 + 4.0 * sg_1;
                    }
                    float _S25 = lf_0;
                    lf_0 = _S23;
                    tS_0 = _S24;
                    sg_0 = sg_1;
                    ext_0 = _S25;
                }
                else
                {
                    lf_0 = 0.0;
                    tS_0 = nearPlane_1;
                    sg_0 = 0.0;
                    ext_0 = nearPlane_1;
                }
            }
            else
            {
                lf_0 = 0.0;
                tS_0 = nearPlane_1;
                sg_0 = 0.0;
                ext_0 = nearPlane_1;
            }
        }
        else
        {
            lf_0 = 0.0;
            tS_0 = nearPlane_1;
            sg_0 = 0.0;
            ext_0 = nearPlane_1;
        }
        float _S26 = simd_prefix_exclusive_sum(lf_0);
        float incl_0 = _S26 + lf_0;
        if(lf_0 < 0.0)
        {
            crossed_0 = (logT_0 + incl_0) < (log(0.5));
        }
        else
        {
            crossed_0 = false;
        }
        bool _S27 = simd_any(crossed_0);
        if(_S27)
        {
            if(crossed_0)
            {
                pass_0 = lane_0;
            }
            else
            {
                pass_0 = int(32);
            }
            int first_0 = simd_min(pass_0);
            float _S28 = (simd_shuffle((tS_0), ushort((first_0))));
            float _S29 = (simd_shuffle((sg_0), ushort((first_0))));
            float _S30;
            if(lane_0 <= first_0)
            {
                _S30 = ext_0;
            }
            else
            {
                _S30 = nearPlane_1;
            }
            float _S31 = simd_max(_S30);
            float _S32 = max(tHi_0, _S31);
            reached_0 = true;
            tHi_0 = _S32;
            tCross_0 = _S28;
            sCross_0 = _S29;
        }
        else
        {
            float _S33 = simd_sum(lf_0);
            float logT_1 = logT_0 + _S33;
            float _S34 = simd_max(ext_0);
            tHi_0 = max(tHi_0, _S34);
            logT_0 = logT_1;
        }
        base_0 = base_0 + int(32);
    }
    if(!reached_0)
    {
        if(lane_0 == int(0))
        {
            *((&kernelContext_0)->surface_depth_0+pix_idx_0) = -1.0;
            *((&kernelContext_0)->surface_slope_0+pix_idx_0) = 0.0;
        }
        return;
    }
    int _S35 = (iterations_0 + int(2)) / int(3);
    lf_0 = nearPlane_1;
    tS_0 = tHi_0;
    hiExact_0 = false;
    sg_0 = 1.0;
    ext_0 = 0.0;
    pass_0 = int(0);
    for(;;)
    {
        if(pass_0 < _S35)
        {
        }
        else
        {
            break;
        }
        thread array<float, int(9)> pts_0;
        float _S36 = 3.0 * sCross_0;
        float _S37 = max(lf_0, tCross_0 - _S36);
        float _S38 = min(tS_0, tCross_0 + _S36);
        if(pass_0 == int(0))
        {
            reached_0 = _S37 < _S38;
        }
        else
        {
            reached_0 = false;
        }
        int count_0;
        if(reached_0)
        {
            count_0 = int(0);
            for(;;)
            {
                if(count_0 < int(9))
                {
                }
                else
                {
                    break;
                }
                pts_0[count_0] = _S37 + (_S38 - _S37) * float(count_0) / 8.0;
                count_0 = count_0 + int(1);
            }
            count_0 = int(9);
        }
        else
        {
            float _S39 = (tS_0 - lf_0) / 8.0;
            count_0 = int(0);
            for(;;)
            {
                if(count_0 < int(7))
                {
                }
                else
                {
                    break;
                }
                int _S40 = count_0 + int(1);
                pts_0[count_0] = lf_0 + float(_S40) * _S39;
                count_0 = _S40;
            }
            count_0 = int(7);
        }
        float _S41 = pts_0[count_0 - int(1)];
        thread array<float, int(9)> ls_0;
        int j_1 = int(0);
        for(;;)
        {
            if(j_1 < int(9))
            {
            }
            else
            {
                break;
            }
            ls_0[j_1] = 0.0;
            j_1 = j_1 + int(1);
        }
        base_0 = isect_start_0;
        for(;;)
        {
            if(base_0 < _S13)
            {
            }
            else
            {
                break;
            }
            int k_1 = base_0 + lane_0;
            if(k_1 < _S13)
            {
                int gi_1 = (&kernelContext_0)->flatten_ids_0[k_1];
                bool beyond_0 = ((&kernelContext_0)->center_depths_0[gi_1] - _S15) > _S41;
                if(!beyond_0)
                {
                    crossed_0 = ((&kernelContext_0)->center_depths_0[gi_1] - (&kernelContext_0)->extents_0[gi_1]) <= _S41;
                }
                else
                {
                    crossed_0 = false;
                }
                bool _S42;
                if(crossed_0)
                {
                    int _S43 = gi_1 * int(2);
                    int _S44 = gi_1 * int(3);
                    _S42 = !farFromFootprint_0(px_0 - (&kernelContext_0)->means2d_0[_S43], py_0 - (&kernelContext_0)->means2d_0[_S43 + int(1)], (&kernelContext_0)->conics_0[_S44], (&kernelContext_0)->conics_0[_S44 + int(1)], (&kernelContext_0)->conics_0[_S44 + int(2)]);
                }
                else
                {
                    _S42 = false;
                }
                if(_S42)
                {
                    int o_1 = gi_1 * int(12);
                    RayGaussian_0 g_3 = rayGaussian_0(float3((&kernelContext_0)->frames_0[o_1], (&kernelContext_0)->frames_0[o_1 + int(1)], (&kernelContext_0)->frames_0[o_1 + int(2)]), float3((&kernelContext_0)->frames_0[o_1 + int(3)], (&kernelContext_0)->frames_0[o_1 + int(4)], (&kernelContext_0)->frames_0[o_1 + int(5)]), float3((&kernelContext_0)->frames_0[o_1 + int(6)], (&kernelContext_0)->frames_0[o_1 + int(7)], (&kernelContext_0)->frames_0[o_1 + int(8)]), float3((&kernelContext_0)->frames_0[o_1 + int(9)], (&kernelContext_0)->frames_0[o_1 + int(10)], (&kernelContext_0)->frames_0[o_1 + int(11)]), _S17);
                    float _S45 = (&kernelContext_0)->opacities_0[gi_1];
                    thread RayGaussian_0 _S46 = g_3;
                    thread float tStar_3;
                    float _S47 = peakAlpha_0(&_S46, _S45, nearPlane_1, &tStar_3);
                    if(_S47 > 0.0)
                    {
                        float sigma4_0 = 4.0 / sqrt(g_3.A_0);
                        if((pts_0[int(0)] - tStar_3) > sigma4_0)
                        {
                            float _S48 = log(1.0 - _S47);
                            j_0 = int(0);
                            for(;;)
                            {
                                if(j_0 < count_0)
                                {
                                }
                                else
                                {
                                    break;
                                }
                                ls_0[j_0] = ls_0[j_0] + _S48;
                                j_0 = j_0 + int(1);
                            }
                        }
                        else
                        {
                            if((pts_0[count_0 - int(1)] - tStar_3) >= (- sigma4_0))
                            {
                                j_0 = int(0);
                                for(;;)
                                {
                                    if(j_0 < count_0)
                                    {
                                    }
                                    else
                                    {
                                        break;
                                    }
                                    ls_0[j_0] = ls_0[j_0] + log(transmittanceAt_0(_S47, tStar_3, g_3.A_0, pts_0[j_0], _S16));
                                    j_0 = j_0 + int(1);
                                }
                            }
                        }
                    }
                }
                crossed_0 = beyond_0;
            }
            else
            {
                crossed_0 = true;
            }
            bool _S49 = simd_all(crossed_0);
            if(_S49)
            {
                break;
            }
            base_0 = base_0 + int(32);
        }
        thread array<float, int(9)> Ts_0;
        j_0 = int(0);
        for(;;)
        {
            if(j_0 < int(9))
            {
            }
            else
            {
                break;
            }
            float _S50 = simd_sum(ls_0[j_0]);
            Ts_0[j_0] = exp(_S50);
            j_0 = j_0 + int(1);
        }
        int j_2 = int(0);
        for(;;)
        {
            if(j_2 < count_0)
            {
            }
            else
            {
                seg_0 = count_0;
                break;
            }
            if((Ts_0[j_2]) < 0.5)
            {
                seg_0 = j_2;
                break;
            }
            j_2 = j_2 + int(1);
        }
        if(seg_0 > int(0))
        {
            sg_0 = Ts_0[seg_0 - int(1)];
        }
        if(seg_0 < count_0)
        {
            hiExact_0 = true;
            ext_0 = Ts_0[seg_0];
        }
        if(seg_0 == int(0))
        {
        }
        else
        {
            lf_0 = pts_0[seg_0 - int(1)];
        }
        if(seg_0 == count_0)
        {
        }
        else
        {
            tS_0 = pts_0[seg_0];
        }
        pass_0 = pass_0 + int(1);
    }
    if(lane_0 != int(0))
    {
        return;
    }
    if(hiExact_0)
    {
        reached_0 = sg_0 > 0.5;
    }
    else
    {
        reached_0 = false;
    }
    if(reached_0)
    {
        reached_0 = ext_0 < 0.5;
    }
    else
    {
        reached_0 = false;
    }
    if(reached_0)
    {
        reached_0 = tS_0 > lf_0;
    }
    else
    {
        reached_0 = false;
    }
    if(reached_0)
    {
        float llo_0 = log(sg_0);
        float lhi_0 = log(max(ext_0, 1.00000000317107685e-30));
        float _S51 = tS_0 - lf_0;
        float _S52 = llo_0 - lhi_0;
        *((&kernelContext_0)->surface_depth_0+pix_idx_0) = lf_0 + _S51 * (llo_0 - log(0.5)) / _S52;
        float device* _S53 = (&kernelContext_0)->surface_slope_0+pix_idx_0;
        if(_S52 > 0.00100000004749745)
        {
            tHi_0 = (lhi_0 - llo_0) / _S51;
        }
        else
        {
            tHi_0 = 0.0;
        }
        *_S53 = tHi_0;
    }
    else
    {
        *((&kernelContext_0)->surface_depth_0+pix_idx_0) = 0.5 * (lf_0 + tS_0);
        *((&kernelContext_0)->surface_slope_0+pix_idx_0) = 0.0;
    }
    return;
}

