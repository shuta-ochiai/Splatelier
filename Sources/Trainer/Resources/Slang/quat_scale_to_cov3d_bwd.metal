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

struct SymMat3_0
{
    float c00_0;
    float c01_0;
    float c02_0;
    float c11_0;
    float c12_0;
    float c22_0;
};

struct s_paramCtx_s_bwdCallableCtx_quat_scale_to_cov3d_0
{
    float value0_1;
    float value1_0;
    float value2_0;
    float value3_0;
    float value4_0;
    float value5_0;
    float value6_0;
};

struct s_bwdCallableCtx_quat_scale_to_cov3d_0
{
    s_paramCtx_s_bwdCallableCtx_quat_scale_to_cov3d_0 value0_2;
};

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

struct Tuple_1
{
    Mat3_0 value0_3;
};

Tuple_1 s_apply_quat_to_rotmat_0(float _S5, float _S6, float _S7, float _S8)
{
    Tuple_0 _S9 = s_apply_rsqrt_0(_S6 * _S6 + _S7 * _S7 + _S8 * _S8 + _S5 * _S5);
    float w_0 = _S5 * _S9.value0_0;
    float x_1 = _S6 * _S9.value0_0;
    float y_0 = _S7 * _S9.value0_0;
    float z_0 = _S8 * _S9.value0_0;
    float x2_0 = x_1 * x_1;
    float y2_0 = y_0 * y_0;
    float z2_0 = z_0 * z_0;
    float xy_0 = x_1 * y_0;
    float xz_0 = x_1 * z_0;
    float yz_0 = y_0 * z_0;
    float wx_0 = w_0 * x_1;
    float wy_0 = w_0 * y_0;
    float wz_0 = w_0 * z_0;
    float _S10 = 2.0 * (xy_0 - wz_0);
    float _S11 = 2.0 * (xz_0 + wy_0);
    float _S12 = 2.0 * (xy_0 + wz_0);
    float _S13 = 1.0 - 2.0 * (x2_0 + z2_0);
    float _S14 = 2.0 * (yz_0 - wx_0);
    float _S15 = 2.0 * (xz_0 - wy_0);
    float _S16 = 2.0 * (yz_0 + wx_0);
    float _S17 = 1.0 - 2.0 * (x2_0 + y2_0);
    thread Tuple_1 _S18;
    (&(&_S18)->value0_3)->r00_0 = 1.0 - 2.0 * (y2_0 + z2_0);
    (&(&_S18)->value0_3)->r01_0 = _S10;
    (&(&_S18)->value0_3)->r02_0 = _S11;
    (&(&_S18)->value0_3)->r10_0 = _S12;
    (&(&_S18)->value0_3)->r11_0 = _S13;
    (&(&_S18)->value0_3)->r12_0 = _S14;
    (&(&_S18)->value0_3)->r20_0 = _S15;
    (&(&_S18)->value0_3)->r21_0 = _S16;
    (&(&_S18)->value0_3)->r22_0 = _S17;
    return _S18;
}

struct s_paramCtx_s_bwdCallableCtx_quat_to_rotmat_0
{
    float value0_4;
    float value1_1;
    float value2_1;
    float value3_1;
};

Mat3_0 Mat3_x24_syn_dzero_0()
{
    thread Mat3_0 result_0;
    (&result_0)->r00_0 = 0.0;
    (&result_0)->r01_0 = 0.0;
    (&result_0)->r02_0 = 0.0;
    (&result_0)->r10_0 = 0.0;
    (&result_0)->r11_0 = 0.0;
    (&result_0)->r12_0 = 0.0;
    (&result_0)->r20_0 = 0.0;
    (&result_0)->r21_0 = 0.0;
    (&result_0)->r22_0 = 0.0;
    return result_0;
}

struct s_bwdCallableCtx_quat_to_rotmat_0
{
    s_paramCtx_s_bwdCallableCtx_quat_to_rotmat_0 value0_5;
};

void s_bwdProp_quat_to_rotmat_0(const s_bwdCallableCtx_quat_to_rotmat_0 thread* _S19, float thread* _S20, float thread* _S21, float thread* _S22, float thread* _S23, const Mat3_0 thread* _S24)
{
    float _S25 = (&_S19->value0_5)->value1_1 * (&_S19->value0_5)->value1_1 + (&_S19->value0_5)->value2_1 * (&_S19->value0_5)->value2_1 + (&_S19->value0_5)->value3_1 * (&_S19->value0_5)->value3_1 + (&_S19->value0_5)->value0_4 * (&_S19->value0_5)->value0_4;
    Tuple_0 _S26 = s_apply_rsqrt_0(_S25);
    thread s_bwdCallableCtx_rsqrt_0 _S27;
    (&_S27)->_S2 = _S25;
    float w_1 = (&_S19->value0_5)->value0_4 * _S26.value0_0;
    float x_2 = (&_S19->value0_5)->value1_1 * _S26.value0_0;
    float y_1 = (&_S19->value0_5)->value2_1 * _S26.value0_0;
    float z_1 = (&_S19->value0_5)->value3_1 * _S26.value0_0;
    float _S28 = 2.0 * - _S24->r22_0;
    float _S29 = 2.0 * _S24->r21_0;
    float _S30 = 2.0 * _S24->r20_0;
    float _S31 = 2.0 * _S24->r12_0;
    float _S32 = 2.0 * - _S24->r11_0;
    float _S33 = 2.0 * _S24->r10_0;
    float _S34 = 2.0 * _S24->r02_0;
    float _S35 = 2.0 * _S24->r01_0;
    float _S36 = 2.0 * - _S24->r00_0;
    float _S37 = _S33 + - _S35;
    float _S38 = - _S30 + _S34;
    float _S39 = _S29 + - _S31;
    float _S40 = _S29 + _S31;
    float _S41 = _S30 + _S34;
    float _S42 = _S33 + _S35;
    float _S43 = z_1 * (_S32 + _S36);
    float _S44 = y_1 * (_S28 + _S36);
    float _S45 = x_2 * (_S28 + _S32);
    float _S46 = w_1 * _S37 + y_1 * _S40 + x_2 * _S41 + _S43 + _S43;
    float _S47 = _S26.value0_0 * _S46;
    float _S48 = w_1 * _S38 + z_1 * _S40 + x_2 * _S42 + _S44 + _S44;
    float _S49 = _S26.value0_0 * _S48;
    float _S50 = w_1 * _S39 + z_1 * _S41 + y_1 * _S42 + _S45 + _S45;
    float _S51 = _S26.value0_0 * _S50;
    float _S52 = z_1 * _S37 + y_1 * _S38 + x_2 * _S39;
    float _S53 = _S26.value0_0 * _S52;
    float _S54 = (&_S19->value0_5)->value3_1 * _S46 + (&_S19->value0_5)->value2_1 * _S48 + (&_S19->value0_5)->value1_1 * _S50 + (&_S19->value0_5)->value0_4 * _S52;
    thread float _S55 = 0.0;
    thread s_bwdCallableCtx_rsqrt_0 _S56 = _S27;
    s_bwdProp_rsqrt_0(&_S56, &_S55, _S54);
    float _S57 = (&_S19->value0_5)->value0_4 * _S55;
    float _S58 = (&_S19->value0_5)->value3_1 * _S55;
    float _S59 = (&_S19->value0_5)->value2_1 * _S55;
    float _S60 = (&_S19->value0_5)->value1_1 * _S55;
    float _S61 = _S49 + _S59 + _S59;
    float _S62 = _S51 + _S60 + _S60;
    float _S63 = _S47 + _S58 + _S58;
    *_S20 = _S53 + _S57 + _S57;
    *_S21 = _S62;
    *_S22 = _S61;
    *_S23 = _S63;
    return;
}

void s_bwdProp_quat_scale_to_cov3d_0(const s_bwdCallableCtx_quat_scale_to_cov3d_0 thread* _S64, float thread* _S65, float thread* _S66, float thread* _S67, float thread* _S68, float thread* _S69, float thread* _S70, float thread* _S71, const SymMat3_0 thread* _S72)
{
    Tuple_1 _S73 = s_apply_quat_to_rotmat_0((&_S64->value0_2)->value0_1, (&_S64->value0_2)->value1_0, (&_S64->value0_2)->value2_0, (&_S64->value0_2)->value3_0);
    thread s_paramCtx_s_bwdCallableCtx_quat_to_rotmat_0 _S74;
    (&_S74)->value0_4 = (&_S64->value0_2)->value0_1;
    (&_S74)->value1_1 = (&_S64->value0_2)->value1_0;
    (&_S74)->value2_1 = (&_S64->value0_2)->value2_0;
    (&_S74)->value3_1 = (&_S64->value0_2)->value3_0;
    float M00_0 = _S73.value0_3.r00_0 * (&_S64->value0_2)->value4_0;
    float M01_0 = _S73.value0_3.r01_0 * (&_S64->value0_2)->value5_0;
    float M02_0 = _S73.value0_3.r02_0 * (&_S64->value0_2)->value6_0;
    float M10_0 = _S73.value0_3.r10_0 * (&_S64->value0_2)->value4_0;
    float M11_0 = _S73.value0_3.r11_0 * (&_S64->value0_2)->value5_0;
    float M12_0 = _S73.value0_3.r12_0 * (&_S64->value0_2)->value6_0;
    float M20_0 = _S73.value0_3.r20_0 * (&_S64->value0_2)->value4_0;
    float M21_0 = _S73.value0_3.r21_0 * (&_S64->value0_2)->value5_0;
    float M22_0 = _S73.value0_3.r22_0 * (&_S64->value0_2)->value6_0;
    float _S75 = M22_0 * _S72->c22_0;
    float _S76 = M21_0 * _S72->c22_0;
    float _S77 = M20_0 * _S72->c22_0;
    float _S78 = M12_0 * _S72->c11_0;
    float _S79 = M11_0 * _S72->c11_0;
    float _S80 = M10_0 * _S72->c11_0;
    float _S81 = M02_0 * _S72->c00_0;
    float _S82 = M01_0 * _S72->c00_0;
    float _S83 = M00_0 * _S72->c00_0;
    float _S84 = _S75 + _S75 + M12_0 * _S72->c12_0 + M02_0 * _S72->c02_0;
    float _S85 = _S73.value0_3.r22_0 * _S84;
    float _S86 = (&_S64->value0_2)->value6_0 * _S84;
    float _S87 = _S76 + _S76 + M11_0 * _S72->c12_0 + M01_0 * _S72->c02_0;
    float _S88 = _S73.value0_3.r21_0 * _S87;
    float _S89 = (&_S64->value0_2)->value5_0 * _S87;
    float _S90 = _S77 + _S77 + M10_0 * _S72->c12_0 + M00_0 * _S72->c02_0;
    float _S91 = _S73.value0_3.r20_0 * _S90;
    float _S92 = (&_S64->value0_2)->value4_0 * _S90;
    float _S93 = M22_0 * _S72->c12_0 + _S78 + _S78 + M02_0 * _S72->c01_0;
    float _S94 = _S73.value0_3.r12_0 * _S93;
    float _S95 = (&_S64->value0_2)->value6_0 * _S93;
    float _S96 = M21_0 * _S72->c12_0 + _S79 + _S79 + M01_0 * _S72->c01_0;
    float _S97 = _S73.value0_3.r11_0 * _S96;
    float _S98 = (&_S64->value0_2)->value5_0 * _S96;
    float _S99 = M20_0 * _S72->c12_0 + _S80 + _S80 + M00_0 * _S72->c01_0;
    float _S100 = _S73.value0_3.r10_0 * _S99;
    float _S101 = (&_S64->value0_2)->value4_0 * _S99;
    float _S102 = M22_0 * _S72->c02_0 + M12_0 * _S72->c01_0 + _S81 + _S81;
    float _S103 = _S73.value0_3.r02_0 * _S102;
    float _S104 = (&_S64->value0_2)->value6_0 * _S102;
    float _S105 = M21_0 * _S72->c02_0 + M11_0 * _S72->c01_0 + _S82 + _S82;
    float _S106 = _S73.value0_3.r01_0 * _S105;
    float _S107 = (&_S64->value0_2)->value5_0 * _S105;
    float _S108 = M20_0 * _S72->c02_0 + M10_0 * _S72->c01_0 + _S83 + _S83;
    float _S109 = _S73.value0_3.r00_0 * _S108;
    float _S110 = (&_S64->value0_2)->value4_0 * _S108;
    thread Mat3_0 _S111 = Mat3_x24_syn_dzero_0();
    (&_S111)->r22_0 = _S86;
    (&_S111)->r21_0 = _S89;
    (&_S111)->r20_0 = _S92;
    (&_S111)->r12_0 = _S95;
    (&_S111)->r11_0 = _S98;
    (&_S111)->r10_0 = _S101;
    (&_S111)->r02_0 = _S104;
    (&_S111)->r01_0 = _S107;
    (&_S111)->r00_0 = _S110;
    thread float _S112 = 0.0;
    thread float _S113 = 0.0;
    thread float _S114 = 0.0;
    thread float _S115 = 0.0;
    thread s_bwdCallableCtx_quat_to_rotmat_0 _S116;
    (&_S116)->value0_5 = _S74;
    thread Mat3_0 _S117 = _S111;
    s_bwdProp_quat_to_rotmat_0(&_S116, &_S112, &_S113, &_S114, &_S115, &_S117);
    float _S118 = _S91 + _S100 + _S109;
    float _S119 = _S88 + _S97 + _S106;
    float _S120 = _S85 + _S94 + _S103;
    *_S65 = _S112;
    *_S66 = _S113;
    *_S67 = _S114;
    *_S68 = _S115;
    *_S69 = _S118;
    *_S70 = _S119;
    *_S71 = _S120;
    return;
}

void _S121(DiffPair_float_0 thread* _S122, DiffPair_float_0 thread* _S123, DiffPair_float_0 thread* _S124, DiffPair_float_0 thread* _S125, DiffPair_float_0 thread* _S126, DiffPair_float_0 thread* _S127, DiffPair_float_0 thread* _S128, const SymMat3_0 thread* _S129)
{
    thread s_paramCtx_s_bwdCallableCtx_quat_scale_to_cov3d_0 _S130;
    (&_S130)->value0_1 = _S122->primal_0;
    (&_S130)->value1_0 = _S123->primal_0;
    (&_S130)->value2_0 = _S124->primal_0;
    (&_S130)->value3_0 = _S125->primal_0;
    (&_S130)->value4_0 = _S126->primal_0;
    (&_S130)->value5_0 = _S127->primal_0;
    (&_S130)->value6_0 = _S128->primal_0;
    thread s_bwdCallableCtx_quat_scale_to_cov3d_0 _S131;
    (&_S131)->value0_2 = _S130;
    s_bwdProp_quat_scale_to_cov3d_0(&_S131, &_S122->differential_0, &_S123->differential_0, &_S124->differential_0, &_S125->differential_0, &_S126->differential_0, &_S127->differential_0, &_S128->differential_0, _S129);
    return;
}

struct QuatScaleGrad_0
{
    float qw_0;
    float qx_0;
    float qy_0;
    float qz_0;
    float sx_0;
    float sy_0;
    float sz_0;
};

QuatScaleGrad_0 quat_scale_to_cov3d_grad_0(float qw_1, float qx_1, float qy_1, float qz_1, float sx_1, float sy_1, float sz_1, const SymMat3_0 thread* cot_0)
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
    thread DiffPair_float_0 dpSx_0;
    (&dpSx_0)->primal_0 = sx_1;
    (&dpSx_0)->differential_0 = 0.0;
    thread DiffPair_float_0 dpSy_0;
    (&dpSy_0)->primal_0 = sy_1;
    (&dpSy_0)->differential_0 = 0.0;
    thread DiffPair_float_0 dpSz_0;
    (&dpSz_0)->primal_0 = sz_1;
    (&dpSz_0)->differential_0 = 0.0;
    _S121(&dpW_0, &dpX_0, &dpY_0, &dpZ_0, &dpSx_0, &dpSy_0, &dpSz_0, cot_0);
    thread QuatScaleGrad_0 g_0;
    (&g_0)->qw_0 = dpW_0.differential_0;
    (&g_0)->qx_0 = dpX_0.differential_0;
    (&g_0)->qy_0 = dpY_0.differential_0;
    (&g_0)->qz_0 = dpZ_0.differential_0;
    (&g_0)->sx_0 = dpSx_0.differential_0;
    (&g_0)->sy_0 = dpSy_0.differential_0;
    (&g_0)->sz_0 = dpSz_0.differential_0;
    return g_0;
}

struct KernelContext_0
{
    float device* cot_cov_0;
    float device* quats_bwd_0;
    float device* scales_bwd_0;
    float device* grad_quats_0;
    float device* grad_scales_0;
};

[[kernel]] void quat_scale_to_cov3d_bwd(uint3 tid_0 [[thread_position_in_grid]], float device* cot_cov_1 [[buffer(5)]], float device* quats_bwd_1 [[buffer(3)]], float device* scales_bwd_1 [[buffer(4)]], float device* grad_quats_1 [[buffer(6)]], float device* grad_scales_1 [[buffer(7)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->cot_cov_0 = cot_cov_1;
    (&kernelContext_0)->quats_bwd_0 = quats_bwd_1;
    (&kernelContext_0)->scales_bwd_0 = scales_bwd_1;
    (&kernelContext_0)->grad_quats_0 = grad_quats_1;
    (&kernelContext_0)->grad_scales_0 = grad_scales_1;
    uint i_0 = tid_0.x;
    uint qBase_0 = i_0 * 4U;
    uint sBase_0 = i_0 * 3U;
    uint cBase_0 = i_0 * 6U;
    thread SymMat3_0 cot_1;
    (&cot_1)->c00_0 = cot_cov_1[cBase_0];
    (&cot_1)->c01_0 = cot_cov_1[cBase_0 + 1U];
    (&cot_1)->c02_0 = cot_cov_1[cBase_0 + 2U];
    (&cot_1)->c11_0 = cot_cov_1[cBase_0 + 3U];
    (&cot_1)->c12_0 = cot_cov_1[cBase_0 + 4U];
    (&cot_1)->c22_0 = cot_cov_1[cBase_0 + 5U];
    float _S132 = quats_bwd_1[qBase_0];
    uint _S133 = qBase_0 + 1U;
    float _S134 = quats_bwd_1[_S133];
    uint _S135 = qBase_0 + 2U;
    float _S136 = quats_bwd_1[_S135];
    uint _S137 = qBase_0 + 3U;
    float _S138 = quats_bwd_1[_S137];
    float _S139 = scales_bwd_1[sBase_0];
    uint _S140 = sBase_0 + 1U;
    float _S141 = scales_bwd_1[_S140];
    uint _S142 = sBase_0 + 2U;
    float _S143 = scales_bwd_1[_S142];
    thread SymMat3_0 _S144 = cot_1;
    QuatScaleGrad_0 _S145 = quat_scale_to_cov3d_grad_0(_S132, _S134, _S136, _S138, _S139, _S141, _S143, &_S144);
    *(grad_quats_1+qBase_0) = _S145.qw_0;
    *(grad_quats_1+_S133) = _S145.qx_0;
    *(grad_quats_1+_S135) = _S145.qy_0;
    *(grad_quats_1+_S137) = _S145.qz_0;
    *(grad_scales_1+sBase_0) = _S145.sx_0;
    *(grad_scales_1+_S140) = _S145.sy_0;
    *(grad_scales_1+_S142) = _S145.sz_0;
    return;
}

