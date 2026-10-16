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
};

[[kernel]] void gof_depth(uint3 tid_0 [[thread_position_in_grid]], int device* dims_1 [[buffer(15)]], int device* tile_offsets_1 [[buffer(14)]], float device* depth_params_1 [[buffer(17)]], float device* camera_1 [[buffer(16)]], int device* flatten_ids_1 [[buffer(13)]], float device* means2d_1 [[buffer(11)]], float device* conics_1 [[buffer(12)]], float device* frames_1 [[buffer(7)]], float device* opacities_1 [[buffer(10)]], float device* surface_depth_1 [[buffer(18)]], float device* surface_slope_1 [[buffer(19)]], float device* center_depths_1 [[buffer(9)]], float device* extents_1 [[buffer(8)]])
{
    float tCross_0;
    float sCross_0;
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
    int imageWidth_0 = dims_1[int(0)];
    int imageHeight_0 = dims_1[int(1)];
    int tileSize_0 = dims_1[int(2)];
    int numTilesX_0 = dims_1[int(3)];
    int iterations_0 = dims_1[int(4)];
    int stride_0 = dims_1[int(5)];
    int _S7 = int(tid_0.x);
    int _S8 = imageWidth_0 / stride_0;
    bool reached_0;
    if(_S7 >= _S8)
    {
        reached_0 = true;
    }
    else
    {
        int _S9 = int(tid_0.y);
        int _S10 = imageHeight_0 / stride_0;
        reached_0 = _S9 >= _S10;
    }
    if(reached_0)
    {
        return;
    }
    int pixel_x_0 = _S7 * stride_0 + stride_0 / int(2);
    int _S11 = int(tid_0.y);
    int pixel_y_0 = _S11 * stride_0 + stride_0 / int(2);
    int _S12 = pixel_y_0 / tileSize_0;
    int _S13 = _S12 * numTilesX_0;
    int _S14 = pixel_x_0 / tileSize_0;
    int tile_id_0 = _S13 + _S14;
    int isect_start_0 = (&kernelContext_0)->tile_offsets_0[tile_id_0];
    int _S15 = (&kernelContext_0)->tile_offsets_0[tile_id_0 + int(1)];
    int _S16 = imageWidth_0 / stride_0;
    int pix_idx_0 = _S11 * _S16 + _S7;
    float px_0 = float(pixel_x_0) + 0.5;
    float py_0 = float(pixel_y_0) + 0.5;
    float nearPlane_1 = (&kernelContext_0)->depth_params_0[int(0)];
    float _S17 = (&kernelContext_0)->depth_params_0[int(1)];
    int _S18 = int((&kernelContext_0)->depth_params_0[int(2)]);
    float xn_0 = (px_0 - (&kernelContext_0)->camera_0[int(18)]) / (&kernelContext_0)->camera_0[int(16)];
    float yn_0 = (py_0 - (&kernelContext_0)->camera_0[int(19)]) / (&kernelContext_0)->camera_0[int(17)];
    float3 _S19 = float3((&kernelContext_0)->camera_0[int(0)] * xn_0 + (&kernelContext_0)->camera_0[int(4)] * yn_0 + (&kernelContext_0)->camera_0[int(8)], (&kernelContext_0)->camera_0[int(1)] * xn_0 + (&kernelContext_0)->camera_0[int(5)] * yn_0 + (&kernelContext_0)->camera_0[int(9)], (&kernelContext_0)->camera_0[int(2)] * xn_0 + (&kernelContext_0)->camera_0[int(6)] * yn_0 + (&kernelContext_0)->camera_0[int(10)]);
    float tHi_0 = nearPlane_1;
    int k_0 = isect_start_0;
    float T_0 = 1.0;
    for(;;)
    {
        if(k_0 < _S15)
        {
        }
        else
        {
            reached_0 = false;
            tCross_0 = nearPlane_1;
            sCross_0 = 0.0;
            break;
        }
        int gi_0 = (&kernelContext_0)->flatten_ids_0[k_0];
        int _S20 = gi_0 * int(2);
        int _S21 = gi_0 * int(3);
        if(farFromFootprint_0(px_0 - (&kernelContext_0)->means2d_0[_S20], py_0 - (&kernelContext_0)->means2d_0[_S20 + int(1)], (&kernelContext_0)->conics_0[_S21], (&kernelContext_0)->conics_0[_S21 + int(1)], (&kernelContext_0)->conics_0[_S21 + int(2)]))
        {
            k_0 = k_0 + int(1);
            continue;
        }
        int o_0 = gi_0 * int(12);
        RayGaussian_0 g_2 = rayGaussian_0(float3((&kernelContext_0)->frames_0[o_0], (&kernelContext_0)->frames_0[o_0 + int(1)], (&kernelContext_0)->frames_0[o_0 + int(2)]), float3((&kernelContext_0)->frames_0[o_0 + int(3)], (&kernelContext_0)->frames_0[o_0 + int(4)], (&kernelContext_0)->frames_0[o_0 + int(5)]), float3((&kernelContext_0)->frames_0[o_0 + int(6)], (&kernelContext_0)->frames_0[o_0 + int(7)], (&kernelContext_0)->frames_0[o_0 + int(8)]), float3((&kernelContext_0)->frames_0[o_0 + int(9)], (&kernelContext_0)->frames_0[o_0 + int(10)], (&kernelContext_0)->frames_0[o_0 + int(11)]), _S19);
        float _S22 = (&kernelContext_0)->opacities_0[gi_0];
        thread RayGaussian_0 _S23 = g_2;
        thread float tStar_2;
        float _S24 = peakAlpha_0(&_S23, _S22, nearPlane_1, &tStar_2);
        if(_S24 <= 0.0)
        {
            k_0 = k_0 + int(1);
            continue;
        }
        float T_1 = T_0 * (1.0 - _S24);
        if(_S18 == int(0))
        {
            tCross_0 = tStar_2;
        }
        else
        {
            tCross_0 = tStar_2 + 4.0 / sqrt(g_2.A_0);
        }
        float _S25 = max(tHi_0, tCross_0);
        if(T_1 < 0.5)
        {
            float _S26 = 1.0 / sqrt(g_2.A_0);
            reached_0 = true;
            tHi_0 = _S25;
            tCross_0 = tStar_2;
            sCross_0 = _S26;
            break;
        }
        tHi_0 = _S25;
        T_0 = T_1;
        k_0 = k_0 + int(1);
    }
    if(!reached_0)
    {
        *((&kernelContext_0)->surface_depth_0+pix_idx_0) = -1.0;
        *((&kernelContext_0)->surface_slope_0+pix_idx_0) = 0.0;
        return;
    }
    int _S27 = (iterations_0 + int(2)) / int(3);
    float newLo_0 = nearPlane_1;
    float newHi_0 = tHi_0;
    bool hiExact_0 = false;
    float Tlo_0 = 1.0;
    float Thi_0 = 0.0;
    int pass_0 = int(0);
    for(;;)
    {
        if(pass_0 < _S27)
        {
        }
        else
        {
            break;
        }
        thread array<float, int(9)> pts_0;
        float _S28 = 3.0 * sCross_0;
        float _S29 = max(newLo_0, tCross_0 - _S28);
        float _S30 = min(newHi_0, tCross_0 + _S28);
        if(pass_0 == int(0))
        {
            reached_0 = _S29 < _S30;
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
                pts_0[count_0] = _S29 + (_S30 - _S29) * float(count_0) / 8.0;
                count_0 = count_0 + int(1);
            }
            count_0 = int(9);
        }
        else
        {
            float _S31 = (newHi_0 - newLo_0) / 8.0;
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
                int _S32 = count_0 + int(1);
                pts_0[count_0] = newLo_0 + float(_S32) * _S31;
                count_0 = _S32;
            }
            count_0 = int(7);
        }
        float _S33 = pts_0[count_0 - int(1)];
        thread array<float, int(9)> Ts_0;
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
            Ts_0[j_1] = 1.0;
            j_1 = j_1 + int(1);
        }
        k_0 = isect_start_0;
        for(;;)
        {
            if(k_0 < _S15)
            {
            }
            else
            {
                break;
            }
            int gi_1 = (&kernelContext_0)->flatten_ids_0[k_0];
            if(((&kernelContext_0)->center_depths_0[gi_1] - _S17) > _S33)
            {
                break;
            }
            if(((&kernelContext_0)->center_depths_0[gi_1] - (&kernelContext_0)->extents_0[gi_1]) > _S33)
            {
                k_0 = k_0 + int(1);
                continue;
            }
            int _S34 = gi_1 * int(2);
            int _S35 = gi_1 * int(3);
            if(farFromFootprint_0(px_0 - (&kernelContext_0)->means2d_0[_S34], py_0 - (&kernelContext_0)->means2d_0[_S34 + int(1)], (&kernelContext_0)->conics_0[_S35], (&kernelContext_0)->conics_0[_S35 + int(1)], (&kernelContext_0)->conics_0[_S35 + int(2)]))
            {
                k_0 = k_0 + int(1);
                continue;
            }
            int o_1 = gi_1 * int(12);
            RayGaussian_0 g_3 = rayGaussian_0(float3((&kernelContext_0)->frames_0[o_1], (&kernelContext_0)->frames_0[o_1 + int(1)], (&kernelContext_0)->frames_0[o_1 + int(2)]), float3((&kernelContext_0)->frames_0[o_1 + int(3)], (&kernelContext_0)->frames_0[o_1 + int(4)], (&kernelContext_0)->frames_0[o_1 + int(5)]), float3((&kernelContext_0)->frames_0[o_1 + int(6)], (&kernelContext_0)->frames_0[o_1 + int(7)], (&kernelContext_0)->frames_0[o_1 + int(8)]), float3((&kernelContext_0)->frames_0[o_1 + int(9)], (&kernelContext_0)->frames_0[o_1 + int(10)], (&kernelContext_0)->frames_0[o_1 + int(11)]), _S19);
            float _S36 = (&kernelContext_0)->opacities_0[gi_1];
            thread RayGaussian_0 _S37 = g_3;
            thread float tStar_3;
            float _S38 = peakAlpha_0(&_S37, _S36, nearPlane_1, &tStar_3);
            if(_S38 <= 0.0)
            {
                k_0 = k_0 + int(1);
                continue;
            }
            float sigma4_0 = 4.0 / sqrt(g_3.A_0);
            if((pts_0[int(0)] - tStar_3) > sigma4_0)
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
                    Ts_0[j_0] = Ts_0[j_0] * (1.0 - _S38);
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
                        Ts_0[j_0] = Ts_0[j_0] * transmittanceAt_0(_S38, tStar_3, g_3.A_0, pts_0[j_0], _S18);
                        j_0 = j_0 + int(1);
                    }
                }
            }
            if((Ts_0[int(0)]) < 0.5)
            {
                break;
            }
            k_0 = k_0 + int(1);
        }
        j_0 = int(0);
        for(;;)
        {
            if(j_0 < count_0)
            {
            }
            else
            {
                seg_0 = count_0;
                break;
            }
            if((Ts_0[j_0]) < 0.5)
            {
                seg_0 = j_0;
                break;
            }
            j_0 = j_0 + int(1);
        }
        if(seg_0 > int(0))
        {
            Tlo_0 = Ts_0[seg_0 - int(1)];
        }
        bool _S39 = seg_0 == int(0);
        if(_S39)
        {
            hiExact_0 = false;
        }
        else
        {
            bool hiExact_1;
            float Thi_1;
            if(seg_0 < count_0)
            {
                hiExact_1 = true;
                Thi_1 = Ts_0[seg_0];
            }
            else
            {
                hiExact_1 = hiExact_0;
                Thi_1 = Thi_0;
            }
            hiExact_0 = hiExact_1;
            Thi_0 = Thi_1;
        }
        if(_S39)
        {
        }
        else
        {
            newLo_0 = pts_0[seg_0 - int(1)];
        }
        if(seg_0 == count_0)
        {
        }
        else
        {
            newHi_0 = pts_0[seg_0];
        }
        pass_0 = pass_0 + int(1);
    }
    if(hiExact_0)
    {
        reached_0 = Tlo_0 > 0.5;
    }
    else
    {
        reached_0 = false;
    }
    if(reached_0)
    {
        reached_0 = Thi_0 < 0.5;
    }
    else
    {
        reached_0 = false;
    }
    if(reached_0)
    {
        reached_0 = newHi_0 > newLo_0;
    }
    else
    {
        reached_0 = false;
    }
    if(reached_0)
    {
        float llo_0 = log(Tlo_0);
        float lhi_0 = log(max(Thi_0, 1.00000000317107685e-30));
        float _S40 = newHi_0 - newLo_0;
        float _S41 = llo_0 - lhi_0;
        *((&kernelContext_0)->surface_depth_0+pix_idx_0) = newLo_0 + _S40 * (llo_0 - log(0.5)) / _S41;
        float device* _S42 = (&kernelContext_0)->surface_slope_0+pix_idx_0;
        if(_S41 > 0.00100000004749745)
        {
            tHi_0 = (lhi_0 - llo_0) / _S40;
        }
        else
        {
            tHi_0 = 0.0;
        }
        *_S42 = tHi_0;
    }
    else
    {
        *((&kernelContext_0)->surface_depth_0+pix_idx_0) = 0.5 * (newLo_0 + newHi_0);
        *((&kernelContext_0)->surface_slope_0+pix_idx_0) = 0.0;
    }
    return;
}

