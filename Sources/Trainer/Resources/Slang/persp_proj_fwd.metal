#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct SymMat2_0
{
    float c00_0;
    float c01_0;
    float c11_0;
};

struct ProjResult_0
{
    float px_0;
    float py_0;
    SymMat2_0 cov2d_0;
};

struct SymMat3_0
{
    float c00_1;
    float c01_1;
    float c02_0;
    float c11_1;
    float c12_0;
    float c22_0;
};

ProjResult_0 persp_proj_0(float mx_0, float my_0, float mz_0, const SymMat3_0 thread* cov3d_0, float fx_0, float fy_0, float cx_0, float cy_0, float imageWidth_0, float imageHeight_0)
{
    float rz_0 = 1.0 / mz_0;
    float rz2_0 = rz_0 * rz_0;
    float _S1 = 0.30000001192092896 * (0.5 * imageWidth_0 / fx_0);
    float _S2 = 0.30000001192092896 * (0.5 * imageHeight_0 / fy_0);
    float j00_0 = fx_0 * rz_0;
    float j02_0 = - fx_0 * (mz_0 * min((imageWidth_0 - cx_0) / fx_0 + _S1, max(- (cx_0 / fx_0 + _S1), mx_0 * rz_0))) * rz2_0;
    float j11_0 = fy_0 * rz_0;
    float j12_0 = - fy_0 * (mz_0 * min((imageHeight_0 - cy_0) / fy_0 + _S2, max(- (cy_0 / fy_0 + _S2), my_0 * rz_0))) * rz2_0;
    float t01_0 = j00_0 * cov3d_0->c01_1 + j02_0 * cov3d_0->c12_0;
    float t02_0 = j00_0 * cov3d_0->c02_0 + j02_0 * cov3d_0->c22_0;
    float t11_0 = j11_0 * cov3d_0->c11_1 + j12_0 * cov3d_0->c12_0;
    float t12_0 = j11_0 * cov3d_0->c12_0 + j12_0 * cov3d_0->c22_0;
    thread SymMat2_0 cov2d_1;
    (&cov2d_1)->c00_0 = (j00_0 * cov3d_0->c00_1 + j02_0 * cov3d_0->c02_0) * j00_0 + t02_0 * j02_0;
    (&cov2d_1)->c01_0 = t01_0 * j11_0 + t02_0 * j12_0;
    (&cov2d_1)->c11_0 = t11_0 * j11_0 + t12_0 * j12_0;
    thread ProjResult_0 r_0;
    (&r_0)->px_0 = fx_0 * mx_0 * rz_0 + cx_0;
    (&r_0)->py_0 = fy_0 * my_0 * rz_0 + cy_0;
    (&r_0)->cov2d_0 = cov2d_1;
    return r_0;
}

struct KernelContext_0
{
    float device* cov3ds_cam_0;
    float device* means3d_cam_0;
    float device* intrinsics_0;
    float device* image_dims_0;
    float device* means2d_0;
    float device* cov2ds_0;
};

[[kernel]] void persp_proj_fwd(uint3 tid_0 [[thread_position_in_grid]], float device* cov3ds_cam_1 [[buffer(1)]], float device* means3d_cam_1 [[buffer(0)]], float device* intrinsics_1 [[buffer(2)]], float device* image_dims_1 [[buffer(3)]], float device* means2d_1 [[buffer(4)]], float device* cov2ds_1 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->cov3ds_cam_0 = cov3ds_cam_1;
    (&kernelContext_0)->means3d_cam_0 = means3d_cam_1;
    (&kernelContext_0)->intrinsics_0 = intrinsics_1;
    (&kernelContext_0)->image_dims_0 = image_dims_1;
    (&kernelContext_0)->means2d_0 = means2d_1;
    (&kernelContext_0)->cov2ds_0 = cov2ds_1;
    uint i_0 = tid_0.x;
    uint mBase_0 = i_0 * 3U;
    uint cBase_0 = i_0 * 6U;
    uint pBase_0 = i_0 * 2U;
    thread SymMat3_0 cov3d_1;
    (&cov3d_1)->c00_1 = cov3ds_cam_1[cBase_0];
    (&cov3d_1)->c01_1 = cov3ds_cam_1[cBase_0 + 1U];
    (&cov3d_1)->c02_0 = cov3ds_cam_1[cBase_0 + 2U];
    (&cov3d_1)->c11_1 = cov3ds_cam_1[cBase_0 + 3U];
    (&cov3d_1)->c12_0 = cov3ds_cam_1[cBase_0 + 4U];
    (&cov3d_1)->c22_0 = cov3ds_cam_1[cBase_0 + 5U];
    float _S3 = means3d_cam_1[mBase_0];
    uint _S4 = mBase_0 + 1U;
    float _S5 = means3d_cam_1[_S4];
    uint _S6 = mBase_0 + 2U;
    float _S7 = means3d_cam_1[_S6];
    float _S8 = intrinsics_1[int(0)];
    float _S9 = intrinsics_1[int(1)];
    float _S10 = intrinsics_1[int(2)];
    float _S11 = intrinsics_1[int(3)];
    float _S12 = image_dims_1[int(0)];
    float _S13 = image_dims_1[int(1)];
    thread SymMat3_0 _S14 = cov3d_1;
    ProjResult_0 _S15 = persp_proj_0(_S3, _S5, _S7, &_S14, _S8, _S9, _S10, _S11, _S12, _S13);
    *(means2d_1+pBase_0) = _S15.px_0;
    *(means2d_1+(pBase_0 + 1U)) = _S15.py_0;
    *(cov2ds_1+mBase_0) = _S15.cov2d_0.c00_0;
    *(cov2ds_1+_S4) = _S15.cov2d_0.c01_0;
    *(cov2ds_1+_S6) = _S15.cov2d_0.c11_0;
    return;
}

