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

struct QuatGrad_0
{
    float w_1;
    float x_2;
    float y_1;
    float z_1;
};

QuatGrad_0 quat_to_rotmat_grad_0(float qw_0, float qx_0, float qy_0, float qz_0, const Mat3_0 thread* cot_0)
{
    thread DiffPair_float_0 dpW_0;
    (&dpW_0)->primal_0 = qw_0;
    (&dpW_0)->differential_0 = 0.0;
    thread DiffPair_float_0 dpX_0;
    (&dpX_0)->primal_0 = qx_0;
    (&dpX_0)->differential_0 = 0.0;
    thread DiffPair_float_0 dpY_0;
    (&dpY_0)->primal_0 = qy_0;
    (&dpY_0)->differential_0 = 0.0;
    thread DiffPair_float_0 dpZ_0;
    (&dpZ_0)->primal_0 = qz_0;
    (&dpZ_0)->differential_0 = 0.0;
    _S50(&dpW_0, &dpX_0, &dpY_0, &dpZ_0, cot_0);
    thread QuatGrad_0 g_0;
    (&g_0)->w_1 = dpW_0.differential_0;
    (&g_0)->x_2 = dpX_0.differential_0;
    (&g_0)->y_1 = dpY_0.differential_0;
    (&g_0)->z_1 = dpZ_0.differential_0;
    return g_0;
}

struct KernelContext_0
{
    float device* cot_R_0;
    float device* quats_bwd_0;
    float device* grad_quats_0;
};

[[kernel]] void quat_to_rotmat_bwd(uint3 tid_0 [[thread_position_in_grid]], float device* cot_R_1 [[buffer(3)]], float device* quats_bwd_1 [[buffer(2)]], float device* grad_quats_1 [[buffer(4)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->cot_R_0 = cot_R_1;
    (&kernelContext_0)->quats_bwd_0 = quats_bwd_1;
    (&kernelContext_0)->grad_quats_0 = grad_quats_1;
    uint i_0 = tid_0.x;
    uint qBase_0 = i_0 * 4U;
    uint rBase_0 = i_0 * 9U;
    thread Mat3_0 cot_1;
    (&cot_1)->r00_0 = cot_R_1[rBase_0];
    (&cot_1)->r01_0 = cot_R_1[rBase_0 + 1U];
    (&cot_1)->r02_0 = cot_R_1[rBase_0 + 2U];
    (&cot_1)->r10_0 = cot_R_1[rBase_0 + 3U];
    (&cot_1)->r11_0 = cot_R_1[rBase_0 + 4U];
    (&cot_1)->r12_0 = cot_R_1[rBase_0 + 5U];
    (&cot_1)->r20_0 = cot_R_1[rBase_0 + 6U];
    (&cot_1)->r21_0 = cot_R_1[rBase_0 + 7U];
    (&cot_1)->r22_0 = cot_R_1[rBase_0 + 8U];
    float _S58 = quats_bwd_1[qBase_0];
    uint _S59 = qBase_0 + 1U;
    float _S60 = quats_bwd_1[_S59];
    uint _S61 = qBase_0 + 2U;
    float _S62 = quats_bwd_1[_S61];
    uint _S63 = qBase_0 + 3U;
    float _S64 = quats_bwd_1[_S63];
    thread Mat3_0 _S65 = cot_1;
    QuatGrad_0 _S66 = quat_to_rotmat_grad_0(_S58, _S60, _S62, _S64, &_S65);
    *(grad_quats_1+qBase_0) = _S66.w_1;
    *(grad_quats_1+_S59) = _S66.x_2;
    *(grad_quats_1+_S61) = _S66.y_1;
    *(grad_quats_1+_S63) = _S66.z_1;
    return;
}

