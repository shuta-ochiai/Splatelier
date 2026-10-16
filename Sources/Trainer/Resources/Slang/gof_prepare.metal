#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct Mat3_0
{
    float r00_0;
    float r01_0;
    float r02_0;
    float r10_0;
    float r11_0;
    float r12_0;
    float r20_0;
    float r21_0;
    float r22_0;
};

Mat3_0 quat_to_rotmat_0(float qw_0, float qx_0, float qy_0, float qz_0)
{
    float inv_norm_0 = rsqrt(qx_0 * qx_0 + qy_0 * qy_0 + qz_0 * qz_0 + qw_0 * qw_0);
    float w_0 = qw_0 * inv_norm_0;
    float x_0 = qx_0 * inv_norm_0;
    float y_0 = qy_0 * inv_norm_0;
    float z_0 = qz_0 * inv_norm_0;
    float x2_0 = x_0 * x_0;
    float y2_0 = y_0 * y_0;
    float z2_0 = z_0 * z_0;
    float xy_0 = x_0 * y_0;
    float xz_0 = x_0 * z_0;
    float yz_0 = y_0 * z_0;
    float wx_0 = w_0 * x_0;
    float wy_0 = w_0 * y_0;
    float wz_0 = w_0 * z_0;
    thread Mat3_0 R_0;
    (&R_0)->r00_0 = 1.0 - 2.0 * (y2_0 + z2_0);
    (&R_0)->r01_0 = 2.0 * (xy_0 - wz_0);
    (&R_0)->r02_0 = 2.0 * (xz_0 + wy_0);
    (&R_0)->r10_0 = 2.0 * (xy_0 + wz_0);
    (&R_0)->r11_0 = 1.0 - 2.0 * (x2_0 + z2_0);
    (&R_0)->r12_0 = 2.0 * (yz_0 - wx_0);
    (&R_0)->r20_0 = 2.0 * (xz_0 - wy_0);
    (&R_0)->r21_0 = 2.0 * (yz_0 + wx_0);
    (&R_0)->r22_0 = 1.0 - 2.0 * (x2_0 + y2_0);
    return R_0;
}

struct KernelContext_0
{
    int device* prep_n_0;
    float device* quats_0;
    float device* scales_0;
    float device* viewmat_0;
    float device* means_0;
    float device* gauss_frame_0;
    float device* z_extent_0;
};

[[kernel]] void gof_prepare(uint3 tid_0 [[thread_position_in_grid]], int device* prep_n_1 [[buffer(4)]], float device* quats_1 [[buffer(1)]], float device* scales_1 [[buffer(2)]], float device* viewmat_1 [[buffer(3)]], float device* means_1 [[buffer(0)]], float device* gauss_frame_1 [[buffer(5)]], float device* z_extent_1 [[buffer(6)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->prep_n_0 = prep_n_1;
    (&kernelContext_0)->quats_0 = quats_1;
    (&kernelContext_0)->scales_0 = scales_1;
    (&kernelContext_0)->viewmat_0 = viewmat_1;
    (&kernelContext_0)->means_0 = means_1;
    (&kernelContext_0)->gauss_frame_0 = gauss_frame_1;
    (&kernelContext_0)->z_extent_0 = z_extent_1;
    int i_0 = int(tid_0.x);
    if(i_0 >= prep_n_1[int(0)])
    {
        return;
    }
    int _S1 = i_0 * int(4);
    Mat3_0 R_1 = quat_to_rotmat_0((&kernelContext_0)->quats_0[_S1], (&kernelContext_0)->quats_0[_S1 + int(1)], (&kernelContext_0)->quats_0[_S1 + int(2)], (&kernelContext_0)->quats_0[_S1 + int(3)]);
    int _S2 = i_0 * int(3);
    float isx_0 = 1.0 / max((&kernelContext_0)->scales_0[_S2], 9.99999971718068537e-10);
    int _S3 = _S2 + int(1);
    float isy_0 = 1.0 / max((&kernelContext_0)->scales_0[_S3], 9.99999971718068537e-10);
    int _S4 = _S2 + int(2);
    float isz_0 = 1.0 / max((&kernelContext_0)->scales_0[_S4], 9.99999971718068537e-10);
    float m00_0 = R_1.r00_0 * isx_0;
    float m01_0 = R_1.r10_0 * isx_0;
    float m02_0 = R_1.r20_0 * isx_0;
    float m10_0 = R_1.r01_0 * isy_0;
    float m11_0 = R_1.r11_0 * isy_0;
    float m12_0 = R_1.r21_0 * isy_0;
    float m20_0 = R_1.r02_0 * isz_0;
    float m21_0 = R_1.r12_0 * isz_0;
    float m22_0 = R_1.r22_0 * isz_0;
    float px_0 = - ((&kernelContext_0)->viewmat_0[int(0)] * (&kernelContext_0)->viewmat_0[int(3)] + (&kernelContext_0)->viewmat_0[int(4)] * (&kernelContext_0)->viewmat_0[int(7)] + (&kernelContext_0)->viewmat_0[int(8)] * (&kernelContext_0)->viewmat_0[int(11)]) - (&kernelContext_0)->means_0[_S2];
    float py_0 = - ((&kernelContext_0)->viewmat_0[int(1)] * (&kernelContext_0)->viewmat_0[int(3)] + (&kernelContext_0)->viewmat_0[int(5)] * (&kernelContext_0)->viewmat_0[int(7)] + (&kernelContext_0)->viewmat_0[int(9)] * (&kernelContext_0)->viewmat_0[int(11)]) - (&kernelContext_0)->means_0[_S3];
    float pz_0 = - ((&kernelContext_0)->viewmat_0[int(2)] * (&kernelContext_0)->viewmat_0[int(3)] + (&kernelContext_0)->viewmat_0[int(6)] * (&kernelContext_0)->viewmat_0[int(7)] + (&kernelContext_0)->viewmat_0[int(10)] * (&kernelContext_0)->viewmat_0[int(11)]) - (&kernelContext_0)->means_0[_S4];
    int o_0 = i_0 * int(12);
    *((&kernelContext_0)->gauss_frame_0+o_0) = m00_0;
    *((&kernelContext_0)->gauss_frame_0+(o_0 + int(1))) = m01_0;
    *((&kernelContext_0)->gauss_frame_0+(o_0 + int(2))) = m02_0;
    *((&kernelContext_0)->gauss_frame_0+(o_0 + int(3))) = m10_0;
    *((&kernelContext_0)->gauss_frame_0+(o_0 + int(4))) = m11_0;
    *((&kernelContext_0)->gauss_frame_0+(o_0 + int(5))) = m12_0;
    *((&kernelContext_0)->gauss_frame_0+(o_0 + int(6))) = m20_0;
    *((&kernelContext_0)->gauss_frame_0+(o_0 + int(7))) = m21_0;
    *((&kernelContext_0)->gauss_frame_0+(o_0 + int(8))) = m22_0;
    *((&kernelContext_0)->gauss_frame_0+(o_0 + int(9))) = m00_0 * px_0 + m01_0 * py_0 + m02_0 * pz_0;
    *((&kernelContext_0)->gauss_frame_0+(o_0 + int(10))) = m10_0 * px_0 + m11_0 * py_0 + m12_0 * pz_0;
    *((&kernelContext_0)->gauss_frame_0+(o_0 + int(11))) = m20_0 * px_0 + m21_0 * py_0 + m22_0 * pz_0;
    float sx_0 = (&kernelContext_0)->scales_0[_S2] * (R_1.r00_0 * (&kernelContext_0)->viewmat_0[int(8)] + R_1.r10_0 * (&kernelContext_0)->viewmat_0[int(9)] + R_1.r20_0 * (&kernelContext_0)->viewmat_0[int(10)]);
    float sy_0 = (&kernelContext_0)->scales_0[_S3] * (R_1.r01_0 * (&kernelContext_0)->viewmat_0[int(8)] + R_1.r11_0 * (&kernelContext_0)->viewmat_0[int(9)] + R_1.r21_0 * (&kernelContext_0)->viewmat_0[int(10)]);
    float sz_0 = (&kernelContext_0)->scales_0[_S4] * (R_1.r02_0 * (&kernelContext_0)->viewmat_0[int(8)] + R_1.r12_0 * (&kernelContext_0)->viewmat_0[int(9)] + R_1.r22_0 * (&kernelContext_0)->viewmat_0[int(10)]);
    *((&kernelContext_0)->z_extent_0+i_0) = 3.0 * sqrt(sx_0 * sx_0 + sy_0 * sy_0 + sz_0 * sz_0);
    return;
}

