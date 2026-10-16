#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* res_dims_0;
    int device* obs_cam_idx_0;
    int device* obs_pt_idx_0;
    float device* intrinsics_buf_0;
    float device* points_0;
    float device* viewmats_0;
    float device* z_cam_0;
    float device* residuals_0;
    float device* obs_pixels_0;
};

[[kernel]] void ba_residual(uint3 tid_0 [[thread_position_in_grid]], int device* res_dims_1 [[buffer(6)]], int device* obs_cam_idx_1 [[buffer(3)]], int device* obs_pt_idx_1 [[buffer(4)]], float device* intrinsics_buf_1 [[buffer(1)]], float device* points_1 [[buffer(2)]], float device* viewmats_1 [[buffer(0)]], float device* z_cam_1 [[buffer(8)]], float device* residuals_1 [[buffer(7)]], float device* obs_pixels_1 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->res_dims_0 = res_dims_1;
    (&kernelContext_0)->obs_cam_idx_0 = obs_cam_idx_1;
    (&kernelContext_0)->obs_pt_idx_0 = obs_pt_idx_1;
    (&kernelContext_0)->intrinsics_buf_0 = intrinsics_buf_1;
    (&kernelContext_0)->points_0 = points_1;
    (&kernelContext_0)->viewmats_0 = viewmats_1;
    (&kernelContext_0)->z_cam_0 = z_cam_1;
    (&kernelContext_0)->residuals_0 = residuals_1;
    (&kernelContext_0)->obs_pixels_0 = obs_pixels_1;
    int i_0 = int(tid_0.x);
    if(i_0 >= res_dims_1[int(0)])
    {
        return;
    }
    int cam_0 = (&kernelContext_0)->obs_cam_idx_0[i_0];
    int vmBase_0 = cam_0 * int(16);
    int inBase_0 = cam_0 * int(4);
    int ptBase_0 = (&kernelContext_0)->obs_pt_idx_0[i_0] * int(3);
    float fx_0 = (&kernelContext_0)->intrinsics_buf_0[inBase_0];
    float fy_0 = (&kernelContext_0)->intrinsics_buf_0[inBase_0 + int(1)];
    float cx_0 = (&kernelContext_0)->intrinsics_buf_0[inBase_0 + int(2)];
    float cy_0 = (&kernelContext_0)->intrinsics_buf_0[inBase_0 + int(3)];
    float Px_0 = (&kernelContext_0)->points_0[ptBase_0];
    float Py_0 = (&kernelContext_0)->points_0[ptBase_0 + int(1)];
    float Pz_0 = (&kernelContext_0)->points_0[ptBase_0 + int(2)];
    float xc_0 = (&kernelContext_0)->viewmats_0[vmBase_0] * Px_0 + (&kernelContext_0)->viewmats_0[vmBase_0 + int(1)] * Py_0 + (&kernelContext_0)->viewmats_0[vmBase_0 + int(2)] * Pz_0 + (&kernelContext_0)->viewmats_0[vmBase_0 + int(3)];
    float yc_0 = (&kernelContext_0)->viewmats_0[vmBase_0 + int(4)] * Px_0 + (&kernelContext_0)->viewmats_0[vmBase_0 + int(5)] * Py_0 + (&kernelContext_0)->viewmats_0[vmBase_0 + int(6)] * Pz_0 + (&kernelContext_0)->viewmats_0[vmBase_0 + int(7)];
    float zc_0 = (&kernelContext_0)->viewmats_0[vmBase_0 + int(8)] * Px_0 + (&kernelContext_0)->viewmats_0[vmBase_0 + int(9)] * Py_0 + (&kernelContext_0)->viewmats_0[vmBase_0 + int(10)] * Pz_0 + (&kernelContext_0)->viewmats_0[vmBase_0 + int(11)];
    *((&kernelContext_0)->z_cam_0+i_0) = zc_0;
    if(zc_0 <= 0.00100000004749745)
    {
        int _S1 = i_0 * int(2);
        *((&kernelContext_0)->residuals_0+_S1) = 0.0;
        *((&kernelContext_0)->residuals_0+(_S1 + int(1))) = 0.0;
        return;
    }
    float invZ_0 = 1.0 / zc_0;
    float predV_0 = fy_0 * yc_0 * invZ_0 + cy_0;
    int _S2 = i_0 * int(2);
    *((&kernelContext_0)->residuals_0+_S2) = fx_0 * xc_0 * invZ_0 + cx_0 - (&kernelContext_0)->obs_pixels_0[_S2];
    int _S3 = _S2 + int(1);
    *((&kernelContext_0)->residuals_0+_S3) = predV_0 - (&kernelContext_0)->obs_pixels_0[_S3];
    return;
}

