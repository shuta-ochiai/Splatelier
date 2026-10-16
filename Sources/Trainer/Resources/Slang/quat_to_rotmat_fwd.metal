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
    float device* quats_0;
    float device* rotmats_0;
};

[[kernel]] void quat_to_rotmat_fwd(uint3 tid_0 [[thread_position_in_grid]], float device* quats_1 [[buffer(0)]], float device* rotmats_1 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->quats_0 = quats_1;
    (&kernelContext_0)->rotmats_0 = rotmats_1;
    uint i_0 = tid_0.x;
    uint qBase_0 = i_0 * 4U;
    uint rBase_0 = i_0 * 9U;
    Mat3_0 R_1 = quat_to_rotmat_0(quats_1[qBase_0], quats_1[qBase_0 + 1U], quats_1[qBase_0 + 2U], quats_1[qBase_0 + 3U]);
    *(rotmats_1+rBase_0) = R_1.r00_0;
    *(rotmats_1+(rBase_0 + 1U)) = R_1.r01_0;
    *(rotmats_1+(rBase_0 + 2U)) = R_1.r02_0;
    *(rotmats_1+(rBase_0 + 3U)) = R_1.r10_0;
    *(rotmats_1+(rBase_0 + 4U)) = R_1.r11_0;
    *(rotmats_1+(rBase_0 + 5U)) = R_1.r12_0;
    *(rotmats_1+(rBase_0 + 6U)) = R_1.r20_0;
    *(rotmats_1+(rBase_0 + 7U)) = R_1.r21_0;
    *(rotmats_1+(rBase_0 + 8U)) = R_1.r22_0;
    return;
}

