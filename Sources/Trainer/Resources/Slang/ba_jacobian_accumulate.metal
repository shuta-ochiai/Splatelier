#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* jac_dims_0;
    int device* obs_cam_idx_0;
    int device* obs_pt_idx_0;
    float device* intrinsics_buf_0;
    float device* points_0;
    float device* viewmats_0;
    int device* Htp_valid_0;
    float device* Htp_blocks_0;
    float device* obs_pixels_0;
    float device* jac_params_0;
    float device* Htt_0;
    float device* Hpp_0;
    float device* bt_0;
    float device* bp_0;
};

[[kernel]] void ba_jacobian_accumulate(uint3 tid_0 [[thread_position_in_grid]], int device* jac_dims_1 [[buffer(7)]], int device* obs_cam_idx_1 [[buffer(3)]], int device* obs_pt_idx_1 [[buffer(4)]], float device* intrinsics_buf_1 [[buffer(1)]], float device* points_1 [[buffer(2)]], float device* viewmats_1 [[buffer(0)]], int device* Htp_valid_1 [[buffer(13)]], float device* Htp_blocks_1 [[buffer(12)]], float device* obs_pixels_1 [[buffer(5)]], float device* jac_params_1 [[buffer(6)]], float device* Htt_1 [[buffer(8)]], float device* Hpp_1 [[buffer(9)]], float device* bt_1 [[buffer(10)]], float device* bp_1 [[buffer(11)]])
{
    int jj_0;
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->jac_dims_0 = jac_dims_1;
    (&kernelContext_0)->obs_cam_idx_0 = obs_cam_idx_1;
    (&kernelContext_0)->obs_pt_idx_0 = obs_pt_idx_1;
    (&kernelContext_0)->intrinsics_buf_0 = intrinsics_buf_1;
    (&kernelContext_0)->points_0 = points_1;
    (&kernelContext_0)->viewmats_0 = viewmats_1;
    (&kernelContext_0)->Htp_valid_0 = Htp_valid_1;
    (&kernelContext_0)->Htp_blocks_0 = Htp_blocks_1;
    (&kernelContext_0)->obs_pixels_0 = obs_pixels_1;
    (&kernelContext_0)->jac_params_0 = jac_params_1;
    (&kernelContext_0)->Htt_0 = Htt_1;
    (&kernelContext_0)->Hpp_0 = Hpp_1;
    (&kernelContext_0)->bt_0 = bt_1;
    (&kernelContext_0)->bp_0 = bp_1;
    int i_0 = int(tid_0.x);
    if(i_0 >= jac_dims_1[int(0)])
    {
        return;
    }
    int cam_0 = (&kernelContext_0)->obs_cam_idx_0[i_0];
    int pt_0 = (&kernelContext_0)->obs_pt_idx_0[i_0];
    int vmBase_0 = cam_0 * int(16);
    int inBase_0 = cam_0 * int(4);
    int ptBase_0 = pt_0 * int(3);
    float fx_0 = (&kernelContext_0)->intrinsics_buf_0[inBase_0];
    float fy_0 = (&kernelContext_0)->intrinsics_buf_0[inBase_0 + int(1)];
    float cx_0 = (&kernelContext_0)->intrinsics_buf_0[inBase_0 + int(2)];
    float cy_0 = (&kernelContext_0)->intrinsics_buf_0[inBase_0 + int(3)];
    float Px_0 = (&kernelContext_0)->points_0[ptBase_0];
    float Py_0 = (&kernelContext_0)->points_0[ptBase_0 + int(1)];
    float Pz_0 = (&kernelContext_0)->points_0[ptBase_0 + int(2)];
    int _S1 = vmBase_0 + int(1);
    int _S2 = vmBase_0 + int(2);
    float pcx_0 = (&kernelContext_0)->viewmats_0[vmBase_0] * Px_0 + (&kernelContext_0)->viewmats_0[_S1] * Py_0 + (&kernelContext_0)->viewmats_0[_S2] * Pz_0 + (&kernelContext_0)->viewmats_0[vmBase_0 + int(3)];
    int _S3 = vmBase_0 + int(4);
    int _S4 = vmBase_0 + int(5);
    int _S5 = vmBase_0 + int(6);
    float pcy_0 = (&kernelContext_0)->viewmats_0[_S3] * Px_0 + (&kernelContext_0)->viewmats_0[_S4] * Py_0 + (&kernelContext_0)->viewmats_0[_S5] * Pz_0 + (&kernelContext_0)->viewmats_0[vmBase_0 + int(7)];
    int _S6 = vmBase_0 + int(8);
    int _S7 = vmBase_0 + int(9);
    int _S8 = vmBase_0 + int(10);
    float pcz_0 = (&kernelContext_0)->viewmats_0[_S6] * Px_0 + (&kernelContext_0)->viewmats_0[_S7] * Py_0 + (&kernelContext_0)->viewmats_0[_S8] * Pz_0 + (&kernelContext_0)->viewmats_0[vmBase_0 + int(11)];
    int k_0;
    if(pcz_0 <= 0.00100000004749745)
    {
        *((&kernelContext_0)->Htp_valid_0+i_0) = int(0);
        k_0 = int(0);
        for(;;)
        {
            if(k_0 < int(18))
            {
            }
            else
            {
                break;
            }
            *((&kernelContext_0)->Htp_blocks_0+(i_0 * int(18) + k_0)) = 0.0;
            k_0 = k_0 + int(1);
        }
        return;
    }
    float invZ_0 = 1.0 / pcz_0;
    int _S9 = i_0 * int(2);
    float rU_0 = fx_0 * pcx_0 * invZ_0 + cx_0 - (&kernelContext_0)->obs_pixels_0[_S9];
    float rV_0 = fy_0 * pcy_0 * invZ_0 + cy_0 - (&kernelContext_0)->obs_pixels_0[_S9 + int(1)];
    float dpi_00_0 = fx_0 * invZ_0;
    float dpi_02_0 = - fx_0 * pcx_0 * invZ_0 * invZ_0;
    float dpi_11_0 = fy_0 * invZ_0;
    float dpi_12_0 = - fy_0 * pcy_0 * invZ_0 * invZ_0;
    float R01_0 = (&kernelContext_0)->viewmats_0[_S1];
    float R02_0 = (&kernelContext_0)->viewmats_0[_S2];
    float R10_0 = (&kernelContext_0)->viewmats_0[_S3];
    float R11_0 = (&kernelContext_0)->viewmats_0[_S4];
    float R12_0 = (&kernelContext_0)->viewmats_0[_S5];
    float R20_0 = (&kernelContext_0)->viewmats_0[_S6];
    float R21_0 = (&kernelContext_0)->viewmats_0[_S7];
    float R22_0 = (&kernelContext_0)->viewmats_0[_S8];
    thread array<float, int(6)> Jp_0;
    Jp_0[int(0)] = dpi_00_0 * (&kernelContext_0)->viewmats_0[vmBase_0] + dpi_02_0 * R20_0;
    Jp_0[int(1)] = dpi_00_0 * R01_0 + dpi_02_0 * R21_0;
    Jp_0[int(2)] = dpi_00_0 * R02_0 + dpi_02_0 * R22_0;
    Jp_0[int(3)] = dpi_11_0 * R10_0 + dpi_12_0 * R20_0;
    Jp_0[int(4)] = dpi_11_0 * R11_0 + dpi_12_0 * R21_0;
    Jp_0[int(5)] = dpi_11_0 * R12_0 + dpi_12_0 * R22_0;
    thread array<float, int(12)> Jc_0;
    Jc_0[int(0)] = dpi_00_0 * 0.0 + dpi_02_0 * pcy_0;
    float _S10 = - pcx_0;
    Jc_0[int(1)] = dpi_00_0 * pcz_0 + dpi_02_0 * _S10;
    Jc_0[int(2)] = dpi_00_0 * - pcy_0 + dpi_02_0 * 0.0;
    Jc_0[int(3)] = dpi_00_0;
    Jc_0[int(4)] = 0.0;
    Jc_0[int(5)] = dpi_02_0;
    Jc_0[int(6)] = dpi_11_0 * - pcz_0 + dpi_12_0 * pcy_0;
    Jc_0[int(7)] = dpi_11_0 * 0.0 + dpi_12_0 * _S10;
    Jc_0[int(8)] = dpi_11_0 * pcx_0 + dpi_12_0 * 0.0;
    Jc_0[int(9)] = 0.0;
    Jc_0[int(10)] = dpi_11_0;
    Jc_0[int(11)] = dpi_12_0;
    float huberK_0 = (&kernelContext_0)->jac_params_0[int(0)];
    float w_0;
    if(huberK_0 > 0.0)
    {
        float absR_0 = sqrt(rU_0 * rU_0 + rV_0 * rV_0);
        if(absR_0 > huberK_0)
        {
            w_0 = huberK_0 / max(absR_0, 9.99999971718068537e-10);
        }
        else
        {
            w_0 = 1.0;
        }
    }
    else
    {
        w_0 = 1.0;
    }
    *((&kernelContext_0)->Htp_valid_0+i_0) = int(1);
    int _S11 = cam_0 * int(36);
    k_0 = int(0);
    for(;;)
    {
        if(k_0 < int(6))
        {
        }
        else
        {
            break;
        }
        jj_0 = int(0);
        for(;;)
        {
            if(jj_0 < int(6))
            {
            }
            else
            {
                break;
            }
            float _S12 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->Htt_0+(_S11 + k_0 * int(6) + jj_0))), w_0 * (Jc_0[k_0] * Jc_0[jj_0] + Jc_0[int(6) + k_0] * Jc_0[int(6) + jj_0]), memory_order_relaxed);
            jj_0 = jj_0 + int(1);
        }
        k_0 = k_0 + int(1);
    }
    int _S13 = pt_0 * int(9);
    k_0 = int(0);
    for(;;)
    {
        if(k_0 < int(3))
        {
        }
        else
        {
            break;
        }
        jj_0 = int(0);
        for(;;)
        {
            if(jj_0 < int(3))
            {
            }
            else
            {
                break;
            }
            float _S14 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->Hpp_0+(_S13 + k_0 * int(3) + jj_0))), w_0 * (Jp_0[k_0] * Jp_0[jj_0] + Jp_0[int(3) + k_0] * Jp_0[int(3) + jj_0]), memory_order_relaxed);
            jj_0 = jj_0 + int(1);
        }
        k_0 = k_0 + int(1);
    }
    int _S15 = cam_0 * int(6);
    k_0 = int(0);
    for(;;)
    {
        if(k_0 < int(6))
        {
        }
        else
        {
            break;
        }
        float _S16 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->bt_0+(_S15 + k_0))), - w_0 * (Jc_0[k_0] * rU_0 + Jc_0[int(6) + k_0] * rV_0), memory_order_relaxed);
        k_0 = k_0 + int(1);
    }
    k_0 = int(0);
    for(;;)
    {
        if(k_0 < int(3))
        {
        }
        else
        {
            break;
        }
        float _S17 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->bp_0+(ptBase_0 + k_0))), - w_0 * (Jp_0[k_0] * rU_0 + Jp_0[int(3) + k_0] * rV_0), memory_order_relaxed);
        k_0 = k_0 + int(1);
    }
    int _S18 = i_0 * int(18);
    k_0 = int(0);
    for(;;)
    {
        if(k_0 < int(6))
        {
        }
        else
        {
            break;
        }
        jj_0 = int(0);
        for(;;)
        {
            if(jj_0 < int(3))
            {
            }
            else
            {
                break;
            }
            *((&kernelContext_0)->Htp_blocks_0+(_S18 + k_0 * int(3) + jj_0)) = w_0 * (Jc_0[k_0] * Jp_0[jj_0] + Jc_0[int(6) + k_0] * Jp_0[int(3) + jj_0]);
            jj_0 = jj_0 + int(1);
        }
        k_0 = k_0 + int(1);
    }
    return;
}

