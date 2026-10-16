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

float3 rotateT_0(const Mat3_0 thread* R_1, float3 x_3)
{
    float _S58 = x_3.x;
    float _S59 = x_3.y;
    float _S60 = x_3.z;
    return float3(R_1->r00_0 * _S58 + R_1->r10_0 * _S59 + R_1->r20_0 * _S60, R_1->r01_0 * _S58 + R_1->r11_0 * _S59 + R_1->r21_0 * _S60, R_1->r02_0 * _S58 + R_1->r12_0 * _S59 + R_1->r22_0 * _S60);
}

float3 rotate_0(const Mat3_0 thread* R_2, float3 x_4)
{
    float _S61 = x_4.x;
    float _S62 = x_4.y;
    float _S63 = x_4.z;
    return float3(R_2->r00_0 * _S61 + R_2->r01_0 * _S62 + R_2->r02_0 * _S63, R_2->r10_0 * _S61 + R_2->r11_0 * _S62 + R_2->r12_0 * _S63, R_2->r20_0 * _S61 + R_2->r21_0 * _S62 + R_2->r22_0 * _S63);
}

Mat3_0 zeroMat3_0()
{
    thread Mat3_0 Z_0;
    (&Z_0)->r00_0 = 0.0;
    (&Z_0)->r01_0 = 0.0;
    (&Z_0)->r02_0 = 0.0;
    (&Z_0)->r10_0 = 0.0;
    (&Z_0)->r11_0 = 0.0;
    (&Z_0)->r12_0 = 0.0;
    (&Z_0)->r20_0 = 0.0;
    (&Z_0)->r21_0 = 0.0;
    (&Z_0)->r22_0 = 0.0;
    return Z_0;
}

Mat3_0 addOuter_0(const Mat3_0 thread* R_3, float3 u_0, float3 v_0)
{
    thread Mat3_0 _S64 = *R_3;
    float _S65 = u_0.x;
    float _S66 = v_0.x;
    (&_S64)->r00_0 = (&_S64)->r00_0 + _S65 * _S66;
    float _S67 = v_0.y;
    (&_S64)->r01_0 = (&_S64)->r01_0 + _S65 * _S67;
    float _S68 = v_0.z;
    (&_S64)->r02_0 = (&_S64)->r02_0 + _S65 * _S68;
    float _S69 = u_0.y;
    (&_S64)->r10_0 = (&_S64)->r10_0 + _S69 * _S66;
    (&_S64)->r11_0 = (&_S64)->r11_0 + _S69 * _S67;
    (&_S64)->r12_0 = (&_S64)->r12_0 + _S69 * _S68;
    float _S70 = u_0.z;
    (&_S64)->r20_0 = (&_S64)->r20_0 + _S70 * _S66;
    (&_S64)->r21_0 = (&_S64)->r21_0 + _S70 * _S67;
    (&_S64)->r22_0 = (&_S64)->r22_0 + _S70 * _S68;
    return _S64;
}

struct QuatGrad_0
{
    float w_2;
    float x_5;
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
    (&g_0)->x_5 = dpX_0.differential_0;
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
    float device* v_normals_0;
    float device* d_means_0;
    float device* d_quats_0;
    float device* d_scales_0;
};

[[kernel]] void geo_normals_bwd(uint3 tid_0 [[thread_position_in_grid]], int device* count_1 [[buffer(4)]], float device* quats_1 [[buffer(1)]], float device* scales_1 [[buffer(2)]], float device* viewmat_1 [[buffer(3)]], float device* means_1 [[buffer(0)]], float device* v_normals_1 [[buffer(5)]], float device* d_means_1 [[buffer(8)]], float device* d_quats_1 [[buffer(9)]], float device* d_scales_1 [[buffer(10)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->count_0 = count_1;
    (&kernelContext_0)->quats_0 = quats_1;
    (&kernelContext_0)->scales_0 = scales_1;
    (&kernelContext_0)->viewmat_0 = viewmat_1;
    (&kernelContext_0)->means_0 = means_1;
    (&kernelContext_0)->v_normals_0 = v_normals_1;
    (&kernelContext_0)->d_means_0 = d_means_1;
    (&kernelContext_0)->d_quats_0 = d_quats_1;
    (&kernelContext_0)->d_scales_0 = d_scales_1;
    int i_0 = int(tid_0.x);
    if(i_0 >= count_1[int(0)])
    {
        return;
    }
    int _S71 = i_0 * int(4);
    float qw_2 = (&kernelContext_0)->quats_0[_S71];
    int _S72 = _S71 + int(1);
    float qx_2 = (&kernelContext_0)->quats_0[_S72];
    int _S73 = _S71 + int(2);
    float qy_2 = (&kernelContext_0)->quats_0[_S73];
    int _S74 = _S71 + int(3);
    float qz_2 = (&kernelContext_0)->quats_0[_S74];
    Mat3_0 R_4 = quat_to_rotmat_0(qw_2, qx_2, qy_2, qz_2);
    int _S75 = i_0 * int(3);
    int _S76 = _S75 + int(1);
    int _S77 = _S75 + int(2);
    float3 s_0 = float3((&kernelContext_0)->scales_0[_S75], (&kernelContext_0)->scales_0[_S76], (&kernelContext_0)->scales_0[_S77]);
    float3 d_0 = float3((&kernelContext_0)->means_0[_S75], (&kernelContext_0)->means_0[_S76], (&kernelContext_0)->means_0[_S77]) - cameraCenter_0((&kernelContext_0)->viewmat_0[int(0)], (&kernelContext_0)->viewmat_0[int(1)], (&kernelContext_0)->viewmat_0[int(2)], (&kernelContext_0)->viewmat_0[int(3)], (&kernelContext_0)->viewmat_0[int(4)], (&kernelContext_0)->viewmat_0[int(5)], (&kernelContext_0)->viewmat_0[int(6)], (&kernelContext_0)->viewmat_0[int(7)], (&kernelContext_0)->viewmat_0[int(8)], (&kernelContext_0)->viewmat_0[int(9)], (&kernelContext_0)->viewmat_0[int(10)], (&kernelContext_0)->viewmat_0[int(11)]);
    thread Mat3_0 _S78 = R_4;
    float3 _S79 = rotateT_0(&_S78, d_0);
    float3 s2_0 = max(s_0 * s_0, float3(9.99999968265522539e-21) );
    float3 b_0 = _S79 / s2_0;
    thread Mat3_0 _S80 = R_4;
    float3 _S81 = rotate_0(&_S80, b_0);
    float len_0 = sqrt(max(dot(_S81, _S81), 1.00000000317107685e-30));
    float flip_0;
    if((dot(_S81, d_0)) > 0.0)
    {
        flip_0 = -1.0;
    }
    else
    {
        flip_0 = 1.0;
    }
    float3 _S82 = float3(len_0) ;
    float3 nhat_0 = _S81 / _S82;
    float _S83 = (&kernelContext_0)->v_normals_0[_S75];
    float _S84 = (&kernelContext_0)->v_normals_0[_S76];
    float _S85 = (&kernelContext_0)->v_normals_0[_S77];
    float3 vw_0 = float3((&kernelContext_0)->viewmat_0[int(0)] * _S83 + (&kernelContext_0)->viewmat_0[int(4)] * _S84 + (&kernelContext_0)->viewmat_0[int(8)] * _S85, (&kernelContext_0)->viewmat_0[int(1)] * _S83 + (&kernelContext_0)->viewmat_0[int(5)] * _S84 + (&kernelContext_0)->viewmat_0[int(9)] * _S85, (&kernelContext_0)->viewmat_0[int(2)] * _S83 + (&kernelContext_0)->viewmat_0[int(6)] * _S84 + (&kernelContext_0)->viewmat_0[int(10)] * _S85) * float3(flip_0) ;
    float3 vn_0 = (vw_0 - nhat_0 * float3(dot(nhat_0, vw_0)) ) / _S82;
    thread Mat3_0 _S86 = zeroMat3_0();
    Mat3_0 _S87 = addOuter_0(&_S86, vn_0, b_0);
    thread Mat3_0 _S88 = R_4;
    float3 _S89 = rotateT_0(&_S88, vn_0);
    float3 va_0 = _S89 / s2_0;
    float3 ds_0 = float3(-2.0)  * _S89 * _S79 / (s2_0 * max(s_0, float3(1.00000001335143196e-10) ));
    thread Mat3_0 _S90 = _S87;
    Mat3_0 _S91 = addOuter_0(&_S90, d_0, va_0);
    thread Mat3_0 _S92 = R_4;
    float3 _S93 = rotate_0(&_S92, va_0);
    thread Mat3_0 _S94 = _S91;
    QuatGrad_0 _S95 = quat_to_rotmat_grad_0(qw_2, qx_2, qy_2, qz_2, &_S94);
    *((&kernelContext_0)->d_means_0+_S75) = _S93.x;
    *((&kernelContext_0)->d_means_0+_S76) = _S93.y;
    *((&kernelContext_0)->d_means_0+_S77) = _S93.z;
    *((&kernelContext_0)->d_quats_0+_S71) = _S95.w_2;
    *((&kernelContext_0)->d_quats_0+_S72) = _S95.x_5;
    *((&kernelContext_0)->d_quats_0+_S73) = _S95.y_2;
    *((&kernelContext_0)->d_quats_0+_S74) = _S95.z_2;
    *((&kernelContext_0)->d_scales_0+_S75) = ds_0.x;
    *((&kernelContext_0)->d_scales_0+_S76) = ds_0.y;
    *((&kernelContext_0)->d_scales_0+_S77) = ds_0.z;
    return;
}

