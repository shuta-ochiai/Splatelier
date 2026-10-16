#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct SymMat3_0
{
    float c00_0;
    float c01_0;
    float c02_0;
    float c11_0;
    float c12_0;
    float c22_0;
};

struct CamPose_0
{
    float mx_0;
    float my_0;
    float mz_0;
    SymMat3_0 cov_0;
};

CamPose_0 world_to_cam_0(float wx_0, float wy_0, float wz_0, const SymMat3_0 thread* covW_0, float r00_0, float r01_0, float r02_0, float r10_0, float r11_0, float r12_0, float r20_0, float r21_0, float r22_0, float tx_0, float ty_0, float tz_0)
{
    thread CamPose_0 out_0;
    (&out_0)->mx_0 = r00_0 * wx_0 + r01_0 * wy_0 + r02_0 * wz_0 + tx_0;
    (&out_0)->my_0 = r10_0 * wx_0 + r11_0 * wy_0 + r12_0 * wz_0 + ty_0;
    (&out_0)->mz_0 = r20_0 * wx_0 + r21_0 * wy_0 + r22_0 * wz_0 + tz_0;
    float m00_0 = r00_0 * covW_0->c00_0 + r01_0 * covW_0->c01_0 + r02_0 * covW_0->c02_0;
    float m01_0 = r00_0 * covW_0->c01_0 + r01_0 * covW_0->c11_0 + r02_0 * covW_0->c12_0;
    float m02_0 = r00_0 * covW_0->c02_0 + r01_0 * covW_0->c12_0 + r02_0 * covW_0->c22_0;
    float m10_0 = r10_0 * covW_0->c00_0 + r11_0 * covW_0->c01_0 + r12_0 * covW_0->c02_0;
    float m11_0 = r10_0 * covW_0->c01_0 + r11_0 * covW_0->c11_0 + r12_0 * covW_0->c12_0;
    float m12_0 = r10_0 * covW_0->c02_0 + r11_0 * covW_0->c12_0 + r12_0 * covW_0->c22_0;
    float m20_0 = r20_0 * covW_0->c00_0 + r21_0 * covW_0->c01_0 + r22_0 * covW_0->c02_0;
    float m21_0 = r20_0 * covW_0->c01_0 + r21_0 * covW_0->c11_0 + r22_0 * covW_0->c12_0;
    float m22_0 = r20_0 * covW_0->c02_0 + r21_0 * covW_0->c12_0 + r22_0 * covW_0->c22_0;
    (&(&out_0)->cov_0)->c00_0 = m00_0 * r00_0 + m01_0 * r01_0 + m02_0 * r02_0;
    (&(&out_0)->cov_0)->c01_0 = m00_0 * r10_0 + m01_0 * r11_0 + m02_0 * r12_0;
    (&(&out_0)->cov_0)->c02_0 = m00_0 * r20_0 + m01_0 * r21_0 + m02_0 * r22_0;
    (&(&out_0)->cov_0)->c11_0 = m10_0 * r10_0 + m11_0 * r11_0 + m12_0 * r12_0;
    (&(&out_0)->cov_0)->c12_0 = m10_0 * r20_0 + m11_0 * r21_0 + m12_0 * r22_0;
    (&(&out_0)->cov_0)->c22_0 = m20_0 * r20_0 + m21_0 * r21_0 + m22_0 * r22_0;
    return out_0;
}

struct KernelContext_0
{
    float device* cov3ds_world_0;
    float device* means3d_world_0;
    float device* viewmat_0;
    float device* means3d_cam_0;
    float device* cov3ds_cam_0;
};

[[kernel]] void world_to_cam_fwd(uint3 tid_0 [[thread_position_in_grid]], float device* cov3ds_world_1 [[buffer(1)]], float device* means3d_world_1 [[buffer(0)]], float device* viewmat_1 [[buffer(2)]], float device* means3d_cam_1 [[buffer(3)]], float device* cov3ds_cam_1 [[buffer(4)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->cov3ds_world_0 = cov3ds_world_1;
    (&kernelContext_0)->means3d_world_0 = means3d_world_1;
    (&kernelContext_0)->viewmat_0 = viewmat_1;
    (&kernelContext_0)->means3d_cam_0 = means3d_cam_1;
    (&kernelContext_0)->cov3ds_cam_0 = cov3ds_cam_1;
    uint i_0 = tid_0.x;
    uint mBase_0 = i_0 * 3U;
    uint cBase_0 = i_0 * 6U;
    thread SymMat3_0 covW_1;
    (&covW_1)->c00_0 = cov3ds_world_1[cBase_0];
    uint _S1 = cBase_0 + 1U;
    (&covW_1)->c01_0 = cov3ds_world_1[_S1];
    uint _S2 = cBase_0 + 2U;
    (&covW_1)->c02_0 = cov3ds_world_1[_S2];
    uint _S3 = cBase_0 + 3U;
    (&covW_1)->c11_0 = cov3ds_world_1[_S3];
    uint _S4 = cBase_0 + 4U;
    (&covW_1)->c12_0 = cov3ds_world_1[_S4];
    uint _S5 = cBase_0 + 5U;
    (&covW_1)->c22_0 = cov3ds_world_1[_S5];
    float _S6 = means3d_world_1[mBase_0];
    uint _S7 = mBase_0 + 1U;
    float _S8 = means3d_world_1[_S7];
    uint _S9 = mBase_0 + 2U;
    float _S10 = means3d_world_1[_S9];
    float _S11 = viewmat_1[int(0)];
    float _S12 = viewmat_1[int(1)];
    float _S13 = viewmat_1[int(2)];
    float _S14 = viewmat_1[int(4)];
    float _S15 = viewmat_1[int(5)];
    float _S16 = viewmat_1[int(6)];
    float _S17 = viewmat_1[int(8)];
    float _S18 = viewmat_1[int(9)];
    float _S19 = viewmat_1[int(10)];
    float _S20 = viewmat_1[int(3)];
    float _S21 = viewmat_1[int(7)];
    float _S22 = viewmat_1[int(11)];
    thread SymMat3_0 _S23 = covW_1;
    CamPose_0 _S24 = world_to_cam_0(_S6, _S8, _S10, &_S23, _S11, _S12, _S13, _S14, _S15, _S16, _S17, _S18, _S19, _S20, _S21, _S22);
    *(means3d_cam_1+mBase_0) = _S24.mx_0;
    *(means3d_cam_1+_S7) = _S24.my_0;
    *(means3d_cam_1+_S9) = _S24.mz_0;
    *(cov3ds_cam_1+cBase_0) = _S24.cov_0.c00_0;
    *(cov3ds_cam_1+_S1) = _S24.cov_0.c01_0;
    *(cov3ds_cam_1+_S2) = _S24.cov_0.c02_0;
    *(cov3ds_cam_1+_S3) = _S24.cov_0.c11_0;
    *(cov3ds_cam_1+_S4) = _S24.cov_0.c12_0;
    *(cov3ds_cam_1+_S5) = _S24.cov_0.c22_0;
    return;
}

