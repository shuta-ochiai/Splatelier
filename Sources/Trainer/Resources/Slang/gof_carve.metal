#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* carve_range_0;
    float device* grid_params_0;
    int device* block_coords_0;
    float device* points_0;
    float device* carve_params_0;
    float device* viewmats_0;
    float device* intrinsics_buf_0;
    int device* frame_meta_0;
    float device* depth_0;
    uint device* rgb_packed_0;
    float device* carve_out_0;
    float device* observed_out_0;
    float device* support_out_0;
    uint device* color_out_0;
};

[[kernel]] void gof_carve(uint3 tid_0 [[thread_position_in_grid]], int device* carve_range_1 [[buffer(9)]], float device* grid_params_1 [[buffer(7)]], int device* block_coords_1 [[buffer(6)]], float device* points_1 [[buffer(8)]], float device* carve_params_1 [[buffer(5)]], float device* viewmats_1 [[buffer(2)]], float device* intrinsics_buf_1 [[buffer(3)]], int device* frame_meta_1 [[buffer(4)]], float device* depth_1 [[buffer(0)]], uint device* rgb_packed_1 [[buffer(1)]], float device* carve_out_1 [[buffer(10)]], float device* observed_out_1 [[buffer(11)]], float device* support_out_1 [[buffer(13)]], uint device* color_out_1 [[buffer(12)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->carve_range_0 = carve_range_1;
    (&kernelContext_0)->grid_params_0 = grid_params_1;
    (&kernelContext_0)->block_coords_0 = block_coords_1;
    (&kernelContext_0)->points_0 = points_1;
    (&kernelContext_0)->carve_params_0 = carve_params_1;
    (&kernelContext_0)->viewmats_0 = viewmats_1;
    (&kernelContext_0)->intrinsics_buf_0 = intrinsics_buf_1;
    (&kernelContext_0)->frame_meta_0 = frame_meta_1;
    (&kernelContext_0)->depth_0 = depth_1;
    (&kernelContext_0)->rgb_packed_0 = rgb_packed_1;
    (&kernelContext_0)->carve_out_0 = carve_out_1;
    (&kernelContext_0)->observed_out_0 = observed_out_1;
    (&kernelContext_0)->support_out_0 = support_out_1;
    (&kernelContext_0)->color_out_0 = color_out_1;
    int local_0 = int(tid_0.x);
    if(local_0 >= carve_range_1[int(2)])
    {
        return;
    }
    int g_0 = (&kernelContext_0)->carve_range_0[int(1)] + local_0;
    float wx_0;
    float wy_0;
    float wz_0;
    if((&kernelContext_0)->carve_range_0[int(0)] == int(0))
    {
        int l_0 = g_0 & int(511);
        float voxelSize_0 = (&kernelContext_0)->grid_params_0[int(0)];
        int _S1 = (g_0 >> 9U) * int(3);
        float _S2 = (&kernelContext_0)->grid_params_0[int(2)] + (float((&kernelContext_0)->block_coords_0[_S1 + int(1)] * int(8) + ((l_0 >> 3U) & int(7))) + 0.5) * voxelSize_0;
        float _S3 = (&kernelContext_0)->grid_params_0[int(3)] + (float((&kernelContext_0)->block_coords_0[_S1 + int(2)] * int(8) + (l_0 >> 6U)) + 0.5) * voxelSize_0;
        wx_0 = (&kernelContext_0)->grid_params_0[int(1)] + (float((&kernelContext_0)->block_coords_0[_S1] * int(8) + (l_0 & int(7))) + 0.5) * voxelSize_0;
        wy_0 = _S2;
        wz_0 = _S3;
    }
    else
    {
        int _S4 = g_0 * int(3);
        float _S5 = (&kernelContext_0)->points_0[_S4 + int(1)];
        float _S6 = (&kernelContext_0)->points_0[_S4 + int(2)];
        wx_0 = (&kernelContext_0)->points_0[_S4];
        wy_0 = _S5;
        wz_0 = _S6;
    }
    float _S7 = (&kernelContext_0)->carve_params_0[int(0)];
    float _S8 = (&kernelContext_0)->carve_params_0[int(1)];
    float _S9 = (&kernelContext_0)->carve_params_0[int(2)];
    int _S10 = (&kernelContext_0)->carve_range_0[int(3)];
    int f_0 = int(0);
    float observed_0 = 0.0;
    float carve_0 = 0.0;
    float support_0 = 0.0;
    float cr_0 = 0.0;
    float cg_0 = 0.0;
    float cb_0 = 0.0;
    float cw_0 = 0.0;
    for(;;)
    {
        if(f_0 < _S10)
        {
        }
        else
        {
            break;
        }
        int vo_0 = f_0 * int(16);
        float z_0 = (&kernelContext_0)->viewmats_0[vo_0 + int(8)] * wx_0 + (&kernelContext_0)->viewmats_0[vo_0 + int(9)] * wy_0 + (&kernelContext_0)->viewmats_0[vo_0 + int(10)] * wz_0 + (&kernelContext_0)->viewmats_0[vo_0 + int(11)];
        if(z_0 <= 0.00009999999747379)
        {
            f_0 = f_0 + int(1);
            continue;
        }
        int _S11 = f_0 * int(4);
        float u_0 = (&kernelContext_0)->intrinsics_buf_0[_S11] * ((&kernelContext_0)->viewmats_0[vo_0] * wx_0 + (&kernelContext_0)->viewmats_0[vo_0 + int(1)] * wy_0 + (&kernelContext_0)->viewmats_0[vo_0 + int(2)] * wz_0 + (&kernelContext_0)->viewmats_0[vo_0 + int(3)]) / z_0 + (&kernelContext_0)->intrinsics_buf_0[_S11 + int(2)];
        float v_0 = (&kernelContext_0)->intrinsics_buf_0[_S11 + int(1)] * ((&kernelContext_0)->viewmats_0[vo_0 + int(4)] * wx_0 + (&kernelContext_0)->viewmats_0[vo_0 + int(5)] * wy_0 + (&kernelContext_0)->viewmats_0[vo_0 + int(6)] * wz_0 + (&kernelContext_0)->viewmats_0[vo_0 + int(7)]) / z_0 + (&kernelContext_0)->intrinsics_buf_0[_S11 + int(3)];
        bool _S12;
        if(u_0 < 0.0)
        {
            _S12 = true;
        }
        else
        {
            _S12 = v_0 < 0.0;
        }
        if(_S12)
        {
            f_0 = f_0 + int(1);
            continue;
        }
        int _S13 = f_0 * int(3);
        int W_0 = (&kernelContext_0)->frame_meta_0[_S13];
        int H_0 = (&kernelContext_0)->frame_meta_0[_S13 + int(1)];
        int ix_0 = int(u_0);
        int iy_0 = int(v_0);
        bool _S14;
        if(ix_0 >= W_0)
        {
            _S14 = true;
        }
        else
        {
            _S14 = iy_0 >= H_0;
        }
        if(_S14)
        {
            f_0 = f_0 + int(1);
            continue;
        }
        float observed_1 = observed_0 + 1.0;
        int base_0 = (&kernelContext_0)->frame_meta_0[_S13 + int(2)];
        int pix_0 = base_0 + iy_0 * W_0 + ix_0;
        float fu_0 = u_0 - 0.5;
        float fv_0 = v_0 - 0.5;
        int x0_0 = int(floor(fu_0));
        int y0_0 = int(floor(fv_0));
        float ax_0 = fu_0 - float(x0_0);
        float ay_0 = fv_0 - float(y0_0);
        int _S15 = W_0 - int(1);
        int xa_0 = clamp(x0_0, int(0), _S15);
        int xb_0 = clamp(x0_0 + int(1), int(0), _S15);
        int _S16 = H_0 - int(1);
        int _S17 = base_0 + clamp(y0_0, int(0), _S16) * W_0;
        float d00_0 = (&kernelContext_0)->depth_0[_S17 + xa_0];
        float d10_0 = (&kernelContext_0)->depth_0[_S17 + xb_0];
        int _S18 = base_0 + clamp(y0_0 + int(1), int(0), _S16) * W_0;
        float d01_0 = (&kernelContext_0)->depth_0[_S18 + xa_0];
        float d11_0 = (&kernelContext_0)->depth_0[_S18 + xb_0];
        bool _S19 = d00_0 > 0.0;
        float dmin_0;
        float dmax_0;
        if(_S19)
        {
            float _S20 = max(0.0, d00_0);
            dmin_0 = min(1.00000001504746622e+30, d00_0);
            dmax_0 = _S20;
        }
        else
        {
            dmin_0 = 1.00000001504746622e+30;
            dmax_0 = 0.0;
        }
        bool _S21 = d10_0 > 0.0;
        float dmin_1;
        float dmax_1;
        if(_S21)
        {
            float _S22 = max(dmax_0, d10_0);
            dmin_1 = min(dmin_0, d10_0);
            dmax_1 = _S22;
        }
        else
        {
            dmin_1 = dmin_0;
            dmax_1 = dmax_0;
        }
        bool _S23 = d01_0 > 0.0;
        float dmin_2;
        float dmax_2;
        if(_S23)
        {
            float _S24 = max(dmax_1, d01_0);
            dmin_2 = min(dmin_1, d01_0);
            dmax_2 = _S24;
        }
        else
        {
            dmin_2 = dmin_1;
            dmax_2 = dmax_1;
        }
        bool _S25 = d11_0 > 0.0;
        float dmax_3;
        float dmin_3;
        if(_S25)
        {
            float _S26 = min(dmin_2, d11_0);
            dmax_3 = max(dmax_2, d11_0);
            dmin_3 = _S26;
        }
        else
        {
            dmax_3 = dmax_2;
            dmin_3 = dmin_2;
        }
        if(dmax_3 <= 0.0)
        {
            float carve_1 = carve_0 + 1.0;
            observed_0 = observed_1;
            carve_0 = carve_1;
            f_0 = f_0 + int(1);
            continue;
        }
        bool _S27;
        if(_S19)
        {
            _S27 = _S21;
        }
        else
        {
            _S27 = false;
        }
        bool _S28;
        if(_S27)
        {
            _S28 = _S23;
        }
        else
        {
            _S28 = false;
        }
        bool allValid_0;
        if(_S28)
        {
            allValid_0 = _S25;
        }
        else
        {
            allValid_0 = false;
        }
        bool _S29;
        if(allValid_0)
        {
            _S29 = (dmax_3 - dmin_3) <= ((&kernelContext_0)->carve_params_0[int(3)] + (&kernelContext_0)->carve_params_0[int(4)] * dmin_3);
        }
        else
        {
            _S29 = false;
        }
        float d_0;
        if(_S29)
        {
            float _S30 = 1.0 - ax_0;
            d_0 = (1.0 - ay_0) * (_S30 * d00_0 + ax_0 * d10_0) + ay_0 * (_S30 * d01_0 + ax_0 * d11_0);
        }
        else
        {
            d_0 = dmin_3;
        }
        if(z_0 < (d_0 - (_S7 + _S8 * d_0)))
        {
            float carve_2 = carve_0 + 1.0;
            observed_0 = observed_1;
            carve_0 = carve_2;
            f_0 = f_0 + int(1);
            continue;
        }
        float off_0 = abs(z_0 - d_0);
        float support_1;
        float cr_1;
        float cg_1;
        float cb_1;
        float cw_1;
        if(off_0 < _S9)
        {
            float wc_0 = 1.0 - off_0 / _S9;
            uint p_0 = (&kernelContext_0)->rgb_packed_0[pix_0];
            float cr_2 = cr_0 + wc_0 * float(p_0 & 255U);
            float cg_2 = cg_0 + wc_0 * float((p_0 >> 8U) & 255U);
            float cb_2 = cb_0 + wc_0 * float((p_0 >> 16U) & 255U);
            float cw_2 = cw_0 + wc_0;
            support_1 = support_0 + 1.0;
            cr_1 = cr_2;
            cg_1 = cg_2;
            cb_1 = cb_2;
            cw_1 = cw_2;
        }
        else
        {
            support_1 = support_0;
            cr_1 = cr_0;
            cg_1 = cg_0;
            cb_1 = cb_0;
            cw_1 = cw_0;
        }
        observed_0 = observed_1;
        support_0 = support_1;
        cr_0 = cr_1;
        cg_0 = cg_1;
        cb_0 = cb_1;
        cw_0 = cw_1;
        f_0 = f_0 + int(1);
    }
    *((&kernelContext_0)->carve_out_0+local_0) = carve_0;
    *((&kernelContext_0)->observed_out_0+local_0) = observed_0;
    *((&kernelContext_0)->support_out_0+local_0) = support_0;
    uint packed_0;
    if(cw_0 > 0.0)
    {
        packed_0 = (uint(min(255.0, cr_0 / cw_0 + 0.5)) | (uint(min(255.0, cg_0 / cw_0 + 0.5)) << 8U)) | (uint(min(255.0, cb_0 / cw_0 + 0.5)) << 16U);
    }
    else
    {
        packed_0 = 0U;
    }
    *((&kernelContext_0)->color_out_0+local_0) = packed_0;
    return;
}

