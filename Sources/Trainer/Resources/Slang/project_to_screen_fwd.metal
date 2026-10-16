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

struct CamPose_0
{
    float mx_0;
    float my_0;
    float mz_0;
    SymMat3_0 cov_0;
};

CamPose_0 world_to_cam_0(float wx_1, float wy_1, float wz_1, const SymMat3_0 thread* covW_0, float r00_1, float r01_1, float r02_1, float r10_1, float r11_1, float r12_1, float r20_1, float r21_1, float r22_1, float tx_0, float ty_0, float tz_0)
{
    thread CamPose_0 out_0;
    (&out_0)->mx_0 = r00_1 * wx_1 + r01_1 * wy_1 + r02_1 * wz_1 + tx_0;
    (&out_0)->my_0 = r10_1 * wx_1 + r11_1 * wy_1 + r12_1 * wz_1 + ty_0;
    (&out_0)->mz_0 = r20_1 * wx_1 + r21_1 * wy_1 + r22_1 * wz_1 + tz_0;
    float m00_0 = r00_1 * covW_0->c00_0 + r01_1 * covW_0->c01_0 + r02_1 * covW_0->c02_0;
    float m01_0 = r00_1 * covW_0->c01_0 + r01_1 * covW_0->c11_0 + r02_1 * covW_0->c12_0;
    float m02_0 = r00_1 * covW_0->c02_0 + r01_1 * covW_0->c12_0 + r02_1 * covW_0->c22_0;
    float m10_0 = r10_1 * covW_0->c00_0 + r11_1 * covW_0->c01_0 + r12_1 * covW_0->c02_0;
    float m11_0 = r10_1 * covW_0->c01_0 + r11_1 * covW_0->c11_0 + r12_1 * covW_0->c12_0;
    float m12_0 = r10_1 * covW_0->c02_0 + r11_1 * covW_0->c12_0 + r12_1 * covW_0->c22_0;
    float m20_0 = r20_1 * covW_0->c00_0 + r21_1 * covW_0->c01_0 + r22_1 * covW_0->c02_0;
    float m21_0 = r20_1 * covW_0->c01_0 + r21_1 * covW_0->c11_0 + r22_1 * covW_0->c12_0;
    float m22_0 = r20_1 * covW_0->c02_0 + r21_1 * covW_0->c12_0 + r22_1 * covW_0->c22_0;
    (&(&out_0)->cov_0)->c00_0 = m00_0 * r00_1 + m01_0 * r01_1 + m02_0 * r02_1;
    (&(&out_0)->cov_0)->c01_0 = m00_0 * r10_1 + m01_0 * r11_1 + m02_0 * r12_1;
    (&(&out_0)->cov_0)->c02_0 = m00_0 * r20_1 + m01_0 * r21_1 + m02_0 * r22_1;
    (&(&out_0)->cov_0)->c11_0 = m10_0 * r10_1 + m11_0 * r11_1 + m12_0 * r12_1;
    (&(&out_0)->cov_0)->c12_0 = m10_0 * r20_1 + m11_0 * r21_1 + m12_0 * r22_1;
    (&(&out_0)->cov_0)->c22_0 = m20_0 * r20_1 + m21_0 * r21_1 + m22_0 * r22_1;
    return out_0;
}

struct SymMat2_0
{
    float c00_1;
    float c01_1;
    float c11_1;
};

struct ProjResult_0
{
    float px_0;
    float py_0;
    SymMat2_0 cov2d_0;
};

ProjResult_0 persp_proj_0(float mx_1, float my_1, float mz_1, const SymMat3_0 thread* cov3d_0, float fx_0, float fy_0, float cx_0, float cy_0, float imageWidth_0, float imageHeight_0)
{
    float rz_0 = 1.0 / mz_1;
    float rz2_0 = rz_0 * rz_0;
    float _S1 = 0.30000001192092896 * (0.5 * imageWidth_0 / fx_0);
    float _S2 = 0.30000001192092896 * (0.5 * imageHeight_0 / fy_0);
    float j00_0 = fx_0 * rz_0;
    float j02_0 = - fx_0 * (mz_1 * min((imageWidth_0 - cx_0) / fx_0 + _S1, max(- (cx_0 / fx_0 + _S1), mx_1 * rz_0))) * rz2_0;
    float j11_0 = fy_0 * rz_0;
    float j12_0 = - fy_0 * (mz_1 * min((imageHeight_0 - cy_0) / fy_0 + _S2, max(- (cy_0 / fy_0 + _S2), my_1 * rz_0))) * rz2_0;
    float t01_0 = j00_0 * cov3d_0->c01_0 + j02_0 * cov3d_0->c12_0;
    float t02_0 = j00_0 * cov3d_0->c02_0 + j02_0 * cov3d_0->c22_0;
    float t11_0 = j11_0 * cov3d_0->c11_0 + j12_0 * cov3d_0->c12_0;
    float t12_0 = j11_0 * cov3d_0->c12_0 + j12_0 * cov3d_0->c22_0;
    thread SymMat2_0 cov2d_1;
    (&cov2d_1)->c00_1 = (j00_0 * cov3d_0->c00_0 + j02_0 * cov3d_0->c02_0) * j00_0 + t02_0 * j02_0;
    (&cov2d_1)->c01_1 = t01_0 * j11_0 + t02_0 * j12_0;
    (&cov2d_1)->c11_1 = t11_0 * j11_0 + t12_0 * j12_0;
    thread ProjResult_0 r_0;
    (&r_0)->px_0 = fx_0 * mx_1 * rz_0 + cx_0;
    (&r_0)->py_0 = fy_0 * my_1 * rz_0 + cy_0;
    (&r_0)->cov2d_0 = cov2d_1;
    return r_0;
}

struct ProjectedScreen_0
{
    float px_1;
    float py_1;
    float conic_a_0;
    float conic_b_0;
    float conic_c_0;
    float depth_0;
    float cov2d_a_0;
    float cov2d_b_0;
    float cov2d_c_0;
};

ProjectedScreen_0 project_to_screen_0(float wx_2, float wy_2, float wz_2, float qw_2, float qx_2, float qy_2, float qz_2, float sx_1, float sy_1, float sz_1, float r00_2, float r01_2, float r02_2, float r10_2, float r11_2, float r12_2, float r20_2, float r21_2, float r22_2, float tx_1, float ty_1, float tz_1, float fx_1, float fy_1, float cx_1, float cy_1, float imageWidth_1, float imageHeight_1)
{
    thread SymMat3_0 _S3 = quat_scale_to_cov3d_0(qw_2, qx_2, qy_2, qz_2, sx_1, sy_1, sz_1);
    CamPose_0 _S4 = world_to_cam_0(wx_2, wy_2, wz_2, &_S3, r00_2, r01_2, r02_2, r10_2, r11_2, r12_2, r20_2, r21_2, r22_2, tx_1, ty_1, tz_1);
    float _S5 = max(_S4.mz_0, 0.00999999977648258);
    thread SymMat3_0 _S6 = _S4.cov_0;
    ProjResult_0 _S7 = persp_proj_0(_S4.mx_0, _S4.my_0, _S5, &_S6, fx_1, fy_1, cx_1, cy_1, imageWidth_1, imageHeight_1);
    float a_0 = _S7.cov2d_0.c00_1 + 0.30000001192092896;
    float c_0 = _S7.cov2d_0.c11_1 + 0.30000001192092896;
    float det_0 = a_0 * c_0 - _S7.cov2d_0.c01_1 * _S7.cov2d_0.c01_1;
    float inv_det_0 = 1.0 / max(det_0, 9.99999971718068537e-10);
    bool culled_0;
    if((_S4.mz_0) >= 0.00999999977648258)
    {
        culled_0 = (_S4.mz_0) <= 1.0e+10;
    }
    else
    {
        culled_0 = false;
    }
    if(!culled_0)
    {
        culled_0 = true;
    }
    else
    {
        culled_0 = !(det_0 > 0.0);
    }
    thread ProjectedScreen_0 out_1;
    (&out_1)->px_1 = _S7.px_0;
    (&out_1)->py_1 = _S7.py_0;
    (&out_1)->conic_a_0 = c_0 * inv_det_0;
    (&out_1)->conic_b_0 = - _S7.cov2d_0.c01_1 * inv_det_0;
    (&out_1)->conic_c_0 = a_0 * inv_det_0;
    (&out_1)->depth_0 = _S4.mz_0;
    float _S8;
    if(culled_0)
    {
        _S8 = 0.0;
    }
    else
    {
        _S8 = a_0;
    }
    (&out_1)->cov2d_a_0 = _S8;
    if(culled_0)
    {
        _S8 = 0.0;
    }
    else
    {
        _S8 = _S7.cov2d_0.c01_1;
    }
    (&out_1)->cov2d_b_0 = _S8;
    if(culled_0)
    {
        _S8 = 0.0;
    }
    else
    {
        _S8 = c_0;
    }
    (&out_1)->cov2d_c_0 = _S8;
    return out_1;
}

struct KernelContext_0
{
    int device* n_buf_0;
    float device* means3d_world_0;
    float device* quats_0;
    float device* scales_0;
    float device* viewmat_0;
    float device* intrinsics_0;
    float device* image_dims_0;
    float device* means2d_0;
    float device* conics_0;
    float device* depths_0;
    float device* opacities_0;
    int device* radii_0;
};

[[kernel]] void project_to_screen_fwd(uint3 tid_0 [[thread_position_in_grid]], int device* n_buf_1 [[buffer(7)]], float device* means3d_world_1 [[buffer(0)]], float device* quats_1 [[buffer(1)]], float device* scales_1 [[buffer(2)]], float device* viewmat_1 [[buffer(3)]], float device* intrinsics_1 [[buffer(4)]], float device* image_dims_1 [[buffer(5)]], float device* means2d_1 [[buffer(8)]], float device* conics_1 [[buffer(9)]], float device* depths_1 [[buffer(10)]], float device* opacities_1 [[buffer(6)]], int device* radii_1 [[buffer(11)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->n_buf_0 = n_buf_1;
    (&kernelContext_0)->means3d_world_0 = means3d_world_1;
    (&kernelContext_0)->quats_0 = quats_1;
    (&kernelContext_0)->scales_0 = scales_1;
    (&kernelContext_0)->viewmat_0 = viewmat_1;
    (&kernelContext_0)->intrinsics_0 = intrinsics_1;
    (&kernelContext_0)->image_dims_0 = image_dims_1;
    (&kernelContext_0)->means2d_0 = means2d_1;
    (&kernelContext_0)->conics_0 = conics_1;
    (&kernelContext_0)->depths_0 = depths_1;
    (&kernelContext_0)->opacities_0 = opacities_1;
    (&kernelContext_0)->radii_0 = radii_1;
    uint i_0 = tid_0.x;
    if(int(i_0) >= n_buf_1[int(0)])
    {
        return;
    }
    uint wBase_0 = i_0 * 3U;
    uint qBase_0 = i_0 * 4U;
    uint mBase_0 = i_0 * 2U;
    uint _S9 = wBase_0 + 1U;
    uint _S10 = wBase_0 + 2U;
    ProjectedScreen_0 out_2 = project_to_screen_0((&kernelContext_0)->means3d_world_0[wBase_0], (&kernelContext_0)->means3d_world_0[_S9], (&kernelContext_0)->means3d_world_0[_S10], (&kernelContext_0)->quats_0[qBase_0], (&kernelContext_0)->quats_0[qBase_0 + 1U], (&kernelContext_0)->quats_0[qBase_0 + 2U], (&kernelContext_0)->quats_0[qBase_0 + 3U], (&kernelContext_0)->scales_0[wBase_0], (&kernelContext_0)->scales_0[_S9], (&kernelContext_0)->scales_0[_S10], (&kernelContext_0)->viewmat_0[int(0)], (&kernelContext_0)->viewmat_0[int(1)], (&kernelContext_0)->viewmat_0[int(2)], (&kernelContext_0)->viewmat_0[int(4)], (&kernelContext_0)->viewmat_0[int(5)], (&kernelContext_0)->viewmat_0[int(6)], (&kernelContext_0)->viewmat_0[int(8)], (&kernelContext_0)->viewmat_0[int(9)], (&kernelContext_0)->viewmat_0[int(10)], (&kernelContext_0)->viewmat_0[int(3)], (&kernelContext_0)->viewmat_0[int(7)], (&kernelContext_0)->viewmat_0[int(11)], (&kernelContext_0)->intrinsics_0[int(0)], (&kernelContext_0)->intrinsics_0[int(1)], (&kernelContext_0)->intrinsics_0[int(2)], (&kernelContext_0)->intrinsics_0[int(3)], (&kernelContext_0)->image_dims_0[int(0)], (&kernelContext_0)->image_dims_0[int(1)]);
    *((&kernelContext_0)->means2d_0+mBase_0) = out_2.px_1;
    uint _S11 = mBase_0 + 1U;
    *((&kernelContext_0)->means2d_0+_S11) = out_2.py_1;
    *((&kernelContext_0)->conics_0+wBase_0) = out_2.conic_a_0;
    *((&kernelContext_0)->conics_0+_S9) = out_2.conic_b_0;
    *((&kernelContext_0)->conics_0+_S10) = out_2.conic_c_0;
    *((&kernelContext_0)->depths_0+i_0) = out_2.depth_0;
    float opac_0 = (&kernelContext_0)->opacities_0[i_0];
    if(opac_0 < 0.00392156885936856)
    {
        *((&kernelContext_0)->radii_0+mBase_0) = int(0);
        *((&kernelContext_0)->radii_0+_S11) = int(0);
    }
    else
    {
        float _S12 = min(3.32999992370605469, sqrt(2.0 * log(opac_0 / 0.00392156885936856)));
        int rx_0 = int(ceil(_S12 * sqrt(max(0.0, out_2.cov2d_a_0))));
        int ry_0 = int(ceil(_S12 * sqrt(max(0.0, out_2.cov2d_c_0))));
        float maxR_0 = 4.0 * max((&kernelContext_0)->image_dims_0[int(0)], (&kernelContext_0)->image_dims_0[int(1)]);
        bool _S13;
        if(float(rx_0) > maxR_0)
        {
            _S13 = true;
        }
        else
        {
            _S13 = float(ry_0) > maxR_0;
        }
        int rx_1;
        int ry_1;
        if(_S13)
        {
            rx_1 = int(0);
            ry_1 = int(0);
        }
        else
        {
            rx_1 = rx_0;
            ry_1 = ry_0;
        }
        *((&kernelContext_0)->radii_0+mBase_0) = rx_1;
        *((&kernelContext_0)->radii_0+_S11) = ry_1;
    }
    return;
}

