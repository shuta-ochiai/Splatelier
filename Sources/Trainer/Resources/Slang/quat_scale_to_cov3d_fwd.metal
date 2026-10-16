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

struct SymMat3_0
{
    float c00_0;
    float c01_0;
    float c02_0;
    float c11_0;
    float c12_0;
    float c22_0;
};

SymMat3_0 quat_scale_to_cov3d_0(float qw_1, float qx_1, float qy_1, float qz_1, float sx_0, float sy_0, float sz_0)
{
    Mat3_0 R_1 = quat_to_rotmat_0(qw_1, qx_1, qy_1, qz_1);
    float M00_0 = R_1.r00_0 * sx_0;
    float M01_0 = R_1.r01_0 * sy_0;
    float M02_0 = R_1.r02_0 * sz_0;
    float M10_0 = R_1.r10_0 * sx_0;
    float M11_0 = R_1.r11_0 * sy_0;
    float M12_0 = R_1.r12_0 * sz_0;
    float M20_0 = R_1.r20_0 * sx_0;
    float M21_0 = R_1.r21_0 * sy_0;
    float M22_0 = R_1.r22_0 * sz_0;
    thread SymMat3_0 C_0;
    (&C_0)->c00_0 = M00_0 * M00_0 + M01_0 * M01_0 + M02_0 * M02_0;
    (&C_0)->c01_0 = M00_0 * M10_0 + M01_0 * M11_0 + M02_0 * M12_0;
    (&C_0)->c02_0 = M00_0 * M20_0 + M01_0 * M21_0 + M02_0 * M22_0;
    (&C_0)->c11_0 = M10_0 * M10_0 + M11_0 * M11_0 + M12_0 * M12_0;
    (&C_0)->c12_0 = M10_0 * M20_0 + M11_0 * M21_0 + M12_0 * M22_0;
    (&C_0)->c22_0 = M20_0 * M20_0 + M21_0 * M21_0 + M22_0 * M22_0;
    return C_0;
}

struct KernelContext_0
{
    float device* quats_0;
    float device* scales_0;
    float device* cov3ds_0;
};

[[kernel]] void quat_scale_to_cov3d_fwd(uint3 tid_0 [[thread_position_in_grid]], float device* quats_1 [[buffer(0)]], float device* scales_1 [[buffer(1)]], float device* cov3ds_1 [[buffer(2)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->quats_0 = quats_1;
    (&kernelContext_0)->scales_0 = scales_1;
    (&kernelContext_0)->cov3ds_0 = cov3ds_1;
    uint i_0 = tid_0.x;
    uint qBase_0 = i_0 * 4U;
    uint sBase_0 = i_0 * 3U;
    uint cBase_0 = i_0 * 6U;
    SymMat3_0 C_1 = quat_scale_to_cov3d_0(quats_1[qBase_0], quats_1[qBase_0 + 1U], quats_1[qBase_0 + 2U], quats_1[qBase_0 + 3U], scales_1[sBase_0], scales_1[sBase_0 + 1U], scales_1[sBase_0 + 2U]);
    *(cov3ds_1+cBase_0) = C_1.c00_0;
    *(cov3ds_1+(cBase_0 + 1U)) = C_1.c01_0;
    *(cov3ds_1+(cBase_0 + 2U)) = C_1.c02_0;
    *(cov3ds_1+(cBase_0 + 3U)) = C_1.c11_0;
    *(cov3ds_1+(cBase_0 + 4U)) = C_1.c12_0;
    *(cov3ds_1+(cBase_0 + 5U)) = C_1.c22_0;
    return;
}

