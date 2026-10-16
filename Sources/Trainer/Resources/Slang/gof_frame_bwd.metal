#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct Tuple_0
{
    float value0_0;
};

Tuple_0 s_apply_rsqrt_0(float x_0)
{
    Tuple_0 _S1 = { rsqrt(x_0) };
    return _S1;
}

struct s_bwdCallableCtx_rsqrt_0
{
    float _S2;
};

struct DiffPair_float_0
{
    float primal_0;
    float differential_0;
};

void _d_rsqrt_0(DiffPair_float_0 thread* dpx_0, float dOut_0)
{
    float _S3 = -0.5 / ((*dpx_0).primal_0 * sqrt((*dpx_0).primal_0)) * dOut_0;
    dpx_0->primal_0 = (*dpx_0).primal_0;
    dpx_0->differential_0 = _S3;
    return;
}

void s_bwdProp_rsqrt_0(const s_bwdCallableCtx_rsqrt_0 thread* ctx_0, float thread* d_x_0, float d_out_0)
{
    thread DiffPair_float_0 _S4;
    (&_S4)->primal_0 = ctx_0->_S2;
    _d_rsqrt_0(&_S4, d_out_0);
    *d_x_0 = (&_S4)->differential_0;
    return;
}

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

struct s_paramCtx_s_bwdCallableCtx_quat_to_rotmat_0
{
    float value0_1;
    float value1_0;
    float value2_0;
    float value3_0;
};

struct s_bwdCallableCtx_quat_to_rotmat_0
{
    s_paramCtx_s_bwdCallableCtx_quat_to_rotmat_0 value0_2;
};

void s_bwdProp_quat_to_rotmat_0(const s_bwdCallableCtx_quat_to_rotmat_0 thread* _S5, float thread* _S6, float thread* _S7, float thread* _S8, float thread* _S9, const Mat3_0 thread* _S10)
{
    float _S11 = (&_S5->value0_2)->value1_0 * (&_S5->value0_2)->value1_0 + (&_S5->value0_2)->value2_0 * (&_S5->value0_2)->value2_0 + (&_S5->value0_2)->value3_0 * (&_S5->value0_2)->value3_0 + (&_S5->value0_2)->value0_1 * (&_S5->value0_2)->value0_1;
    Tuple_0 _S12 = s_apply_rsqrt_0(_S11);
    thread s_bwdCallableCtx_rsqrt_0 _S13;
    (&_S13)->_S2 = _S11;
    float w_0 = (&_S5->value0_2)->value0_1 * _S12.value0_0;
    float x_1 = (&_S5->value0_2)->value1_0 * _S12.value0_0;
    float y_0 = (&_S5->value0_2)->value2_0 * _S12.value0_0;
    float z_0 = (&_S5->value0_2)->value3_0 * _S12.value0_0;
    float _S14 = 2.0 * - _S10->r22_0;
    float _S15 = 2.0 * _S10->r21_0;
    float _S16 = 2.0 * _S10->r20_0;
    float _S17 = 2.0 * _S10->r12_0;
    float _S18 = 2.0 * - _S10->r11_0;
    float _S19 = 2.0 * _S10->r10_0;
    float _S20 = 2.0 * _S10->r02_0;
    float _S21 = 2.0 * _S10->r01_0;
    float _S22 = 2.0 * - _S10->r00_0;
    float _S23 = _S19 + - _S21;
    float _S24 = - _S16 + _S20;
    float _S25 = _S15 + - _S17;
    float _S26 = _S15 + _S17;
    float _S27 = _S16 + _S20;
    float _S28 = _S19 + _S21;
    float _S29 = z_0 * (_S18 + _S22);
    float _S30 = y_0 * (_S14 + _S22);
    float _S31 = x_1 * (_S14 + _S18);
    float _S32 = w_0 * _S23 + y_0 * _S26 + x_1 * _S27 + _S29 + _S29;
    float _S33 = _S12.value0_0 * _S32;
    float _S34 = w_0 * _S24 + z_0 * _S26 + x_1 * _S28 + _S30 + _S30;
    float _S35 = _S12.value0_0 * _S34;
    float _S36 = w_0 * _S25 + z_0 * _S27 + y_0 * _S28 + _S31 + _S31;
    float _S37 = _S12.value0_0 * _S36;
    float _S38 = z_0 * _S23 + y_0 * _S24 + x_1 * _S25;
    float _S39 = _S12.value0_0 * _S38;
    float _S40 = (&_S5->value0_2)->value3_0 * _S32 + (&_S5->value0_2)->value2_0 * _S34 + (&_S5->value0_2)->value1_0 * _S36 + (&_S5->value0_2)->value0_1 * _S38;
    thread float _S41 = 0.0;
    thread s_bwdCallableCtx_rsqrt_0 _S42 = _S13;
    s_bwdProp_rsqrt_0(&_S42, &_S41, _S40);
    float _S43 = (&_S5->value0_2)->value0_1 * _S41;
    float _S44 = (&_S5->value0_2)->value3_0 * _S41;
    float _S45 = (&_S5->value0_2)->value2_0 * _S41;
    float _S46 = (&_S5->value0_2)->value1_0 * _S41;
    float _S47 = _S35 + _S45 + _S45;
    float _S48 = _S37 + _S46 + _S46;
    float _S49 = _S33 + _S44 + _S44;
    *_S6 = _S39 + _S43 + _S43;
    *_S7 = _S48;
    *_S8 = _S47;
    *_S9 = _S49;
    return;
}

void _S50(DiffPair_float_0 thread* _S51, DiffPair_float_0 thread* _S52, DiffPair_float_0 thread* _S53, DiffPair_float_0 thread* _S54, const Mat3_0 thread* _S55)
{
    thread s_paramCtx_s_bwdCallableCtx_quat_to_rotmat_0 _S56;
    (&_S56)->value0_1 = _S51->primal_0;
    (&_S56)->value1_0 = _S52->primal_0;
    (&_S56)->value2_0 = _S53->primal_0;
    (&_S56)->value3_0 = _S54->primal_0;
    thread s_bwdCallableCtx_quat_to_rotmat_0 _S57;
    (&_S57)->value0_2 = _S56;
    s_bwdProp_quat_to_rotmat_0(&_S57, &_S51->differential_0, &_S52->differential_0, &_S53->differential_0, &_S54->differential_0, _S55);
    return;
}

Mat3_0 quat_to_rotmat_0(float qw_0, float qx_0, float qy_0, float qz_0)
{
    float inv_norm_0 = rsqrt(qx_0 * qx_0 + qy_0 * qy_0 + qz_0 * qz_0 + qw_0 * qw_0);
    float w_1 = qw_0 * inv_norm_0;
    float x_2 = qx_0 * inv_norm_0;
    float y_1 = qy_0 * inv_norm_0;
    float z_1 = qz_0 * inv_norm_0;
    float x2_0 = x_2 * x_2;
    float y2_0 = y_1 * y_1;
    float z2_0 = z_1 * z_1;
    float xy_0 = x_2 * y_1;
    float xz_0 = x_2 * z_1;
    float yz_0 = y_1 * z_1;
    float wx_0 = w_1 * x_2;
    float wy_0 = w_1 * y_1;
    float wz_0 = w_1 * z_1;
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

struct QuatGrad_0
{
    float w_2;
    float x_3;
    float y_2;
    float z_2;
};

QuatGrad_0 quat_to_rotmat_grad_0(float qw_1, float qx_1, float qy_1, float qz_1, const Mat3_0 thread* cot_0)
{
    thread DiffPair_float_0 dpW_0;
    (&dpW_0)->primal_0 = qw_1;
    (&dpW_0)->differential_0 = 0.0;
    thread DiffPair_float_0 dpX_0;
    (&dpX_0)->primal_0 = qx_1;
    (&dpX_0)->differential_0 = 0.0;
    thread DiffPair_float_0 dpY_0;
    (&dpY_0)->primal_0 = qy_1;
    (&dpY_0)->differential_0 = 0.0;
    thread DiffPair_float_0 dpZ_0;
    (&dpZ_0)->primal_0 = qz_1;
    (&dpZ_0)->differential_0 = 0.0;
    _S50(&dpW_0, &dpX_0, &dpY_0, &dpZ_0, cot_0);
    thread QuatGrad_0 g_0;
    (&g_0)->w_2 = dpW_0.differential_0;
    (&g_0)->x_3 = dpX_0.differential_0;
    (&g_0)->y_2 = dpY_0.differential_0;
    (&g_0)->z_2 = dpZ_0.differential_0;
    return g_0;
}

struct KernelContext_0
{
    int device* count_0;
    float device* quats_0;
    float device* scales_0;
    float device* viewmat_0;
    float device* means_0;
    float device* v_frames_0;
    float device* d_means_0;
    float device* d_quats_0;
    float device* d_scales_0;
};

[[kernel]] void gof_frame_bwd(uint3 tid_0 [[thread_position_in_grid]], int device* count_1 [[buffer(4)]], float device* quats_1 [[buffer(1)]], float device* scales_1 [[buffer(2)]], float device* viewmat_1 [[buffer(3)]], float device* means_1 [[buffer(0)]], float device* v_frames_1 [[buffer(6)]], float device* d_means_1 [[buffer(8)]], float device* d_quats_1 [[buffer(9)]], float device* d_scales_1 [[buffer(10)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->count_0 = count_1;
    (&kernelContext_0)->quats_0 = quats_1;
    (&kernelContext_0)->scales_0 = scales_1;
    (&kernelContext_0)->viewmat_0 = viewmat_1;
    (&kernelContext_0)->means_0 = means_1;
    (&kernelContext_0)->v_frames_0 = v_frames_1;
    (&kernelContext_0)->d_means_0 = d_means_1;
    (&kernelContext_0)->d_quats_0 = d_quats_1;
    (&kernelContext_0)->d_scales_0 = d_scales_1;
    int i_0 = int(tid_0.x);
    if(i_0 >= count_1[int(0)])
    {
        return;
    }
    int _S58 = i_0 * int(4);
    float qw_2 = (&kernelContext_0)->quats_0[_S58];
    int _S59 = _S58 + int(1);
    float qx_2 = (&kernelContext_0)->quats_0[_S59];
    int _S60 = _S58 + int(2);
    float qy_2 = (&kernelContext_0)->quats_0[_S60];
    int _S61 = _S58 + int(3);
    float qz_2 = (&kernelContext_0)->quats_0[_S61];
    Mat3_0 R_1 = quat_to_rotmat_0(qw_2, qx_2, qy_2, qz_2);
    int _S62 = i_0 * int(3);
    int _S63 = _S62 + int(1);
    int _S64 = _S62 + int(2);
    float3 inv_0 = float3(1.0)  / max(float3((&kernelContext_0)->scales_0[_S62], (&kernelContext_0)->scales_0[_S63], (&kernelContext_0)->scales_0[_S64]), float3(9.99999971718068537e-10) );
    float3 p_0 = cameraCenter_0((&kernelContext_0)->viewmat_0[int(0)], (&kernelContext_0)->viewmat_0[int(1)], (&kernelContext_0)->viewmat_0[int(2)], (&kernelContext_0)->viewmat_0[int(3)], (&kernelContext_0)->viewmat_0[int(4)], (&kernelContext_0)->viewmat_0[int(5)], (&kernelContext_0)->viewmat_0[int(6)], (&kernelContext_0)->viewmat_0[int(7)], (&kernelContext_0)->viewmat_0[int(8)], (&kernelContext_0)->viewmat_0[int(9)], (&kernelContext_0)->viewmat_0[int(10)], (&kernelContext_0)->viewmat_0[int(11)]) - float3((&kernelContext_0)->means_0[_S62], (&kernelContext_0)->means_0[_S63], (&kernelContext_0)->means_0[_S64]);
    int o_0 = i_0 * int(12);
    float _S65 = inv_0.x;
    float3 m0_0 = float3(R_1.r00_0, R_1.r10_0, R_1.r20_0) * float3(_S65) ;
    float _S66 = inv_0.y;
    float3 m1_0 = float3(R_1.r01_0, R_1.r11_0, R_1.r21_0) * float3(_S66) ;
    float _S67 = inv_0.z;
    float3 m2_0 = float3(R_1.r02_0, R_1.r12_0, R_1.r22_0) * float3(_S67) ;
    float3 _S68 = float3((&kernelContext_0)->v_frames_0[o_0 + int(9)]) ;
    float3 v0_1 = float3((&kernelContext_0)->v_frames_0[o_0], (&kernelContext_0)->v_frames_0[o_0 + int(1)], (&kernelContext_0)->v_frames_0[o_0 + int(2)]) + _S68 * p_0;
    float3 _S69 = float3((&kernelContext_0)->v_frames_0[o_0 + int(10)]) ;
    float3 v1_1 = float3((&kernelContext_0)->v_frames_0[o_0 + int(3)], (&kernelContext_0)->v_frames_0[o_0 + int(4)], (&kernelContext_0)->v_frames_0[o_0 + int(5)]) + _S69 * p_0;
    float3 _S70 = float3((&kernelContext_0)->v_frames_0[o_0 + int(11)]) ;
    float3 v2_1 = float3((&kernelContext_0)->v_frames_0[o_0 + int(6)], (&kernelContext_0)->v_frames_0[o_0 + int(7)], (&kernelContext_0)->v_frames_0[o_0 + int(8)]) + _S70 * p_0;
    float3 vp_0 = _S68 * m0_0 + _S69 * m1_0 + _S70 * m2_0;
    thread Mat3_0 dR_0;
    (&dR_0)->r00_0 = v0_1.x * _S65;
    (&dR_0)->r10_0 = v0_1.y * _S65;
    (&dR_0)->r20_0 = v0_1.z * _S65;
    (&dR_0)->r01_0 = v1_1.x * _S66;
    (&dR_0)->r11_0 = v1_1.y * _S66;
    (&dR_0)->r21_0 = v1_1.z * _S66;
    (&dR_0)->r02_0 = v2_1.x * _S67;
    (&dR_0)->r12_0 = v2_1.y * _S67;
    (&dR_0)->r22_0 = v2_1.z * _S67;
    float3 ds_0 = - float3(dot(v0_1, m0_0) * _S65, dot(v1_1, m1_0) * _S66, dot(v2_1, m2_0) * _S67);
    thread Mat3_0 _S71 = dR_0;
    QuatGrad_0 _S72 = quat_to_rotmat_grad_0(qw_2, qx_2, qy_2, qz_2, &_S71);
    *((&kernelContext_0)->d_means_0+_S62) = - vp_0.x;
    *((&kernelContext_0)->d_means_0+_S63) = - vp_0.y;
    *((&kernelContext_0)->d_means_0+_S64) = - vp_0.z;
    *((&kernelContext_0)->d_quats_0+_S58) = _S72.w_2;
    *((&kernelContext_0)->d_quats_0+_S59) = _S72.x_3;
    *((&kernelContext_0)->d_quats_0+_S60) = _S72.y_2;
    *((&kernelContext_0)->d_quats_0+_S61) = _S72.z_2;
    *((&kernelContext_0)->d_scales_0+_S62) = ds_0.x;
    *((&kernelContext_0)->d_scales_0+_S63) = ds_0.y;
    *((&kernelContext_0)->d_scales_0+_S64) = ds_0.z;
    return;
}

