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

float3 cameraCenter_0(float v0_0, float v1_0, float v2_0, float v3_0, float v4_0, float v5_0, float v6_0, float v7_0, float v8_0, float v9_0, float v10_0, float v11_0)
{
    return - float3(v0_0 * v3_0 + v4_0 * v7_0 + v8_0 * v11_0, v1_0 * v3_0 + v5_0 * v7_0 + v9_0 * v11_0, v2_0 * v3_0 + v6_0 * v7_0 + v10_0 * v11_0);
}

float3 rotateT_0(const Mat3_0 thread* R_1, float3 x_1)
{
    float _S1 = x_1.x;
    float _S2 = x_1.y;
    float _S3 = x_1.z;
    return float3(R_1->r00_0 * _S1 + R_1->r10_0 * _S2 + R_1->r20_0 * _S3, R_1->r01_0 * _S1 + R_1->r11_0 * _S2 + R_1->r21_0 * _S3, R_1->r02_0 * _S1 + R_1->r12_0 * _S2 + R_1->r22_0 * _S3);
}

float3 rotate_0(const Mat3_0 thread* R_2, float3 x_2)
{
    float _S4 = x_2.x;
    float _S5 = x_2.y;
    float _S6 = x_2.z;
    return float3(R_2->r00_0 * _S4 + R_2->r01_0 * _S5 + R_2->r02_0 * _S6, R_2->r10_0 * _S4 + R_2->r11_0 * _S5 + R_2->r12_0 * _S6, R_2->r20_0 * _S4 + R_2->r21_0 * _S5 + R_2->r22_0 * _S6);
}

struct KernelContext_0
{
    int device* count_0;
    float device* quats_0;
    float device* scales_0;
    float device* viewmat_0;
    float device* means_0;
    float device* normals_0;
};

[[kernel]] void geo_normals_fwd(uint3 tid_0 [[thread_position_in_grid]], int device* count_1 [[buffer(4)]], float device* quats_1 [[buffer(1)]], float device* scales_1 [[buffer(2)]], float device* viewmat_1 [[buffer(3)]], float device* means_1 [[buffer(0)]], float device* normals_1 [[buffer(7)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->count_0 = count_1;
    (&kernelContext_0)->quats_0 = quats_1;
    (&kernelContext_0)->scales_0 = scales_1;
    (&kernelContext_0)->viewmat_0 = viewmat_1;
    (&kernelContext_0)->means_0 = means_1;
    (&kernelContext_0)->normals_0 = normals_1;
    int i_0 = int(tid_0.x);
    if(i_0 >= count_1[int(0)])
    {
        return;
    }
    int _S7 = i_0 * int(4);
    Mat3_0 R_3 = quat_to_rotmat_0((&kernelContext_0)->quats_0[_S7], (&kernelContext_0)->quats_0[_S7 + int(1)], (&kernelContext_0)->quats_0[_S7 + int(2)], (&kernelContext_0)->quats_0[_S7 + int(3)]);
    int _S8 = i_0 * int(3);
    int _S9 = _S8 + int(1);
    int _S10 = _S8 + int(2);
    float3 s_0 = float3((&kernelContext_0)->scales_0[_S8], (&kernelContext_0)->scales_0[_S9], (&kernelContext_0)->scales_0[_S10]);
    float3 d_0 = float3((&kernelContext_0)->means_0[_S8], (&kernelContext_0)->means_0[_S9], (&kernelContext_0)->means_0[_S10]) - cameraCenter_0((&kernelContext_0)->viewmat_0[int(0)], (&kernelContext_0)->viewmat_0[int(1)], (&kernelContext_0)->viewmat_0[int(2)], (&kernelContext_0)->viewmat_0[int(3)], (&kernelContext_0)->viewmat_0[int(4)], (&kernelContext_0)->viewmat_0[int(5)], (&kernelContext_0)->viewmat_0[int(6)], (&kernelContext_0)->viewmat_0[int(7)], (&kernelContext_0)->viewmat_0[int(8)], (&kernelContext_0)->viewmat_0[int(9)], (&kernelContext_0)->viewmat_0[int(10)], (&kernelContext_0)->viewmat_0[int(11)]);
    thread Mat3_0 _S11 = R_3;
    float3 _S12 = rotateT_0(&_S11, d_0);
    float3 b_0 = _S12 / max(s_0 * s_0, float3(9.99999968265522539e-21) );
    thread Mat3_0 _S13 = R_3;
    float3 _S14 = rotate_0(&_S13, b_0);
    float len_0 = sqrt(max(dot(_S14, _S14), 1.00000000317107685e-30));
    float flip_0;
    if((dot(_S14, d_0)) > 0.0)
    {
        flip_0 = -1.0;
    }
    else
    {
        flip_0 = 1.0;
    }
    float3 nw_0 = _S14 * float3((flip_0 / len_0)) ;
    float _S15 = nw_0.x;
    float _S16 = nw_0.y;
    float _S17 = nw_0.z;
    *((&kernelContext_0)->normals_0+_S8) = (&kernelContext_0)->viewmat_0[int(0)] * _S15 + (&kernelContext_0)->viewmat_0[int(1)] * _S16 + (&kernelContext_0)->viewmat_0[int(2)] * _S17;
    *((&kernelContext_0)->normals_0+_S9) = (&kernelContext_0)->viewmat_0[int(4)] * _S15 + (&kernelContext_0)->viewmat_0[int(5)] * _S16 + (&kernelContext_0)->viewmat_0[int(6)] * _S17;
    *((&kernelContext_0)->normals_0+_S10) = (&kernelContext_0)->viewmat_0[int(8)] * _S15 + (&kernelContext_0)->viewmat_0[int(9)] * _S16 + (&kernelContext_0)->viewmat_0[int(10)] * _S17;
    return;
}

