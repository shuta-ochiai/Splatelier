#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* params_i_0;
    float device* params_f_0;
    int device* block_coords_0;
    float device* viewmats_0;
    float device* intrinsics_buf_0;
    int device* frame_meta_0;
    float device* depth_0;
    uint device* rgb_packed_0;
    float device* tsdf_out_0;
    float device* weight_out_0;
    uint device* color_out_0;
};

[[kernel]] void tsdf_integrate(uint3 tid_0 [[thread_position_in_grid]], int device* params_i_1 [[buffer(7)]], float device* params_f_1 [[buffer(6)]], int device* block_coords_1 [[buffer(0)]], float device* viewmats_1 [[buffer(3)]], float device* intrinsics_buf_1 [[buffer(4)]], int device* frame_meta_1 [[buffer(5)]], float device* depth_1 [[buffer(1)]], uint device* rgb_packed_1 [[buffer(2)]], float device* tsdf_out_1 [[buffer(8)]], float device* weight_out_1 [[buffer(9)]], uint device* color_out_1 [[buffer(10)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->params_i_0 = params_i_1;
    (&kernelContext_0)->params_f_0 = params_f_1;
    (&kernelContext_0)->block_coords_0 = block_coords_1;
    (&kernelContext_0)->viewmats_0 = viewmats_1;
    (&kernelContext_0)->intrinsics_buf_0 = intrinsics_buf_1;
    (&kernelContext_0)->frame_meta_0 = frame_meta_1;
    (&kernelContext_0)->depth_0 = depth_1;
    (&kernelContext_0)->rgb_packed_0 = rgb_packed_1;
    (&kernelContext_0)->tsdf_out_0 = tsdf_out_1;
    (&kernelContext_0)->weight_out_0 = weight_out_1;
    (&kernelContext_0)->color_out_0 = color_out_1;
    int local_0 = int(tid_0.x);
    if(local_0 >= params_i_1[int(1)])
    {
        return;
    }
    int g_0 = (&kernelContext_0)->params_i_0[int(0)] + local_0;
    int l_0 = g_0 & int(511);
    float voxelSize_0 = (&kernelContext_0)->params_f_0[int(0)];
    float _S1 = (&kernelContext_0)->params_f_0[int(1)];
    int _S2 = (g_0 >> 9U) * int(3);
    float _S3 = (&kernelContext_0)->params_f_0[int(2)] + (float((&kernelContext_0)->block_coords_0[_S2] * int(8) + (l_0 & int(7))) + 0.5) * voxelSize_0;
    float _S4 = (&kernelContext_0)->params_f_0[int(3)] + (float((&kernelContext_0)->block_coords_0[_S2 + int(1)] * int(8) + ((l_0 >> 3U) & int(7))) + 0.5) * voxelSize_0;
    float _S5 = (&kernelContext_0)->params_f_0[int(4)] + (float((&kernelContext_0)->block_coords_0[_S2 + int(2)] * int(8) + (l_0 >> 6U)) + 0.5) * voxelSize_0;
    int _S6 = (&kernelContext_0)->params_i_0[int(2)];
    int f_0 = int(0);
    float sdfSum_0 = 0.0;
    float wSum_0 = 0.0;
    float cr_0 = 0.0;
    float cg_0 = 0.0;
    float cb_0 = 0.0;
    float cw_0 = 0.0;
    for(;;)
    {
        if(f_0 < _S6)
        {
        }
        else
        {
            break;
        }
        int vo_0 = f_0 * int(16);
        float z_0 = (&kernelContext_0)->viewmats_0[vo_0 + int(8)] * _S3 + (&kernelContext_0)->viewmats_0[vo_0 + int(9)] * _S4 + (&kernelContext_0)->viewmats_0[vo_0 + int(10)] * _S5 + (&kernelContext_0)->viewmats_0[vo_0 + int(11)];
        if(z_0 <= 0.00009999999747379)
        {
            f_0 = f_0 + int(1);
            continue;
        }
        int _S7 = f_0 * int(4);
        float u_0 = (&kernelContext_0)->intrinsics_buf_0[_S7] * ((&kernelContext_0)->viewmats_0[vo_0] * _S3 + (&kernelContext_0)->viewmats_0[vo_0 + int(1)] * _S4 + (&kernelContext_0)->viewmats_0[vo_0 + int(2)] * _S5 + (&kernelContext_0)->viewmats_0[vo_0 + int(3)]) / z_0 + (&kernelContext_0)->intrinsics_buf_0[_S7 + int(2)];
        float v_0 = (&kernelContext_0)->intrinsics_buf_0[_S7 + int(1)] * ((&kernelContext_0)->viewmats_0[vo_0 + int(4)] * _S3 + (&kernelContext_0)->viewmats_0[vo_0 + int(5)] * _S4 + (&kernelContext_0)->viewmats_0[vo_0 + int(6)] * _S5 + (&kernelContext_0)->viewmats_0[vo_0 + int(7)]) / z_0 + (&kernelContext_0)->intrinsics_buf_0[_S7 + int(3)];
        bool _S8;
        if(u_0 < 0.0)
        {
            _S8 = true;
        }
        else
        {
            _S8 = v_0 < 0.0;
        }
        if(_S8)
        {
            f_0 = f_0 + int(1);
            continue;
        }
        int _S9 = f_0 * int(3);
        int W_0 = (&kernelContext_0)->frame_meta_0[_S9];
        int H_0 = (&kernelContext_0)->frame_meta_0[_S9 + int(1)];
        int ix_0 = int(u_0);
        int iy_0 = int(v_0);
        bool _S10;
        if(ix_0 >= W_0)
        {
            _S10 = true;
        }
        else
        {
            _S10 = iy_0 >= H_0;
        }
        if(_S10)
        {
            f_0 = f_0 + int(1);
            continue;
        }
        int pix_0 = (&kernelContext_0)->frame_meta_0[_S9 + int(2)] + iy_0 * W_0 + ix_0;
        float d_0 = (&kernelContext_0)->depth_0[pix_0];
        if(d_0 <= 0.0)
        {
            f_0 = f_0 + int(1);
            continue;
        }
        float sdf_0 = d_0 - z_0;
        if(sdf_0 < (- _S1))
        {
            f_0 = f_0 + int(1);
            continue;
        }
        float sdfSum_1 = sdfSum_0 + min(1.0, sdf_0 / _S1);
        float wSum_1 = wSum_0 + 1.0;
        float _S11 = abs(sdf_0);
        float cr_1;
        float cg_1;
        float cb_1;
        float cw_1;
        if(_S11 < _S1)
        {
            float wc_0 = 1.0 - _S11 / _S1;
            uint p_0 = (&kernelContext_0)->rgb_packed_0[pix_0];
            float cg_2 = cg_0 + wc_0 * float((p_0 >> 8U) & 255U);
            float cb_2 = cb_0 + wc_0 * float((p_0 >> 16U) & 255U);
            float cw_2 = cw_0 + wc_0;
            cr_1 = cr_0 + wc_0 * float(p_0 & 255U);
            cg_1 = cg_2;
            cb_1 = cb_2;
            cw_1 = cw_2;
        }
        else
        {
            cr_1 = cr_0;
            cg_1 = cg_0;
            cb_1 = cb_0;
            cw_1 = cw_0;
        }
        sdfSum_0 = sdfSum_1;
        wSum_0 = wSum_1;
        cr_0 = cr_1;
        cg_0 = cg_1;
        cb_0 = cb_1;
        cw_0 = cw_1;
        f_0 = f_0 + int(1);
    }
    float device* _S12 = (&kernelContext_0)->tsdf_out_0+local_0;
    if(wSum_0 > 0.0)
    {
        sdfSum_0 = sdfSum_0 / wSum_0;
    }
    else
    {
        sdfSum_0 = 1.0;
    }
    *_S12 = sdfSum_0;
    *((&kernelContext_0)->weight_out_0+local_0) = wSum_0;
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

