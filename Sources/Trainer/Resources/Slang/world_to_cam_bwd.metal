#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct DiffPair_float_0
{
    float primal_0;
    float differential_0;
};

struct SymMat3_0
{
    float c00_0;
    float c01_0;
    float c02_0;
    float c11_0;
    float c12_0;
    float c22_0;
};

struct DiffPair_SymMat3_0
{
    SymMat3_0 primal_0;
    SymMat3_0 differential_0;
};

struct CamPose_0
{
    float mx_0;
    float my_0;
    float mz_0;
    SymMat3_0 cov_0;
};

struct s_paramCtx_s_bwdCallableCtx_world_to_cam_0
{
    float value4_0;
    float value5_0;
    float value6_0;
    float value7_0;
    float value8_0;
    float value9_0;
    float value10_0;
    float value11_0;
    float value12_0;
};

struct s_bwdCallableCtx_world_to_cam_0
{
    s_paramCtx_s_bwdCallableCtx_world_to_cam_0 value0_0;
};

SymMat3_0 SymMat3_x24_syn_dzero_0()
{
    thread SymMat3_0 result_0;
    (&result_0)->c00_0 = 0.0;
    (&result_0)->c01_0 = 0.0;
    (&result_0)->c02_0 = 0.0;
    (&result_0)->c11_0 = 0.0;
    (&result_0)->c12_0 = 0.0;
    (&result_0)->c22_0 = 0.0;
    return result_0;
}

SymMat3_0 SymMat3_x24_syn_dadd_0(const SymMat3_0 thread* SLANG_anonymous_0_0, const SymMat3_0 thread* SLANG_anonymous_1_0)
{
    thread SymMat3_0 result_1;
    (&result_1)->c00_0 = SLANG_anonymous_0_0->c00_0 + SLANG_anonymous_1_0->c00_0;
    (&result_1)->c01_0 = SLANG_anonymous_0_0->c01_0 + SLANG_anonymous_1_0->c01_0;
    (&result_1)->c02_0 = SLANG_anonymous_0_0->c02_0 + SLANG_anonymous_1_0->c02_0;
    (&result_1)->c11_0 = SLANG_anonymous_0_0->c11_0 + SLANG_anonymous_1_0->c11_0;
    (&result_1)->c12_0 = SLANG_anonymous_0_0->c12_0 + SLANG_anonymous_1_0->c12_0;
    (&result_1)->c22_0 = SLANG_anonymous_0_0->c22_0 + SLANG_anonymous_1_0->c22_0;
    return result_1;
}

void s_bwdProp_world_to_cam_0(const s_bwdCallableCtx_world_to_cam_0 thread* _S1, float thread* _S2, float thread* _S3, float thread* _S4, SymMat3_0 thread* _S5, const CamPose_0 thread* _S6)
{
    SymMat3_0 _S7 = SymMat3_x24_syn_dzero_0();
    CamPose_0 _S8 = *_S6;
    (&(&_S8)->cov_0)->c22_0 = 0.0;
    float s_diff_m22_T_0 = (&_S1->value0_0)->value12_0 * (&_S6->cov_0)->c22_0;
    float s_diff_m21_T_0 = (&_S1->value0_0)->value11_0 * (&_S6->cov_0)->c22_0;
    float s_diff_m20_T_0 = (&_S1->value0_0)->value10_0 * (&_S6->cov_0)->c22_0;
    (&(&_S8)->cov_0)->c12_0 = 0.0;
    float _S9 = (&_S1->value0_0)->value12_0 * (&_S6->cov_0)->c12_0;
    float _S10 = (&_S1->value0_0)->value11_0 * (&_S6->cov_0)->c12_0;
    float _S11 = (&_S1->value0_0)->value10_0 * (&_S6->cov_0)->c12_0;
    (&(&_S8)->cov_0)->c11_0 = 0.0;
    float _S12 = (&_S1->value0_0)->value9_0 * (&_S6->cov_0)->c11_0;
    float _S13 = (&_S1->value0_0)->value8_0 * (&_S6->cov_0)->c11_0;
    float _S14 = (&_S1->value0_0)->value7_0 * (&_S6->cov_0)->c11_0;
    (&(&_S8)->cov_0)->c02_0 = 0.0;
    float _S15 = (&_S1->value0_0)->value12_0 * (&_S6->cov_0)->c02_0;
    float _S16 = (&_S1->value0_0)->value11_0 * (&_S6->cov_0)->c02_0;
    float _S17 = (&_S1->value0_0)->value10_0 * (&_S6->cov_0)->c02_0;
    (&(&_S8)->cov_0)->c01_0 = 0.0;
    float _S18 = (&_S1->value0_0)->value9_0 * (&_S6->cov_0)->c01_0;
    float _S19 = (&_S1->value0_0)->value8_0 * (&_S6->cov_0)->c01_0;
    float _S20 = (&_S1->value0_0)->value7_0 * (&_S6->cov_0)->c01_0;
    (&(&_S8)->cov_0)->c00_0 = 0.0;
    float _S21 = _S9 + _S12;
    float _S22 = _S10 + _S13;
    float _S23 = _S11 + _S14;
    float _S24 = _S15 + _S18 + (&_S1->value0_0)->value6_0 * (&_S6->cov_0)->c00_0;
    float _S25 = _S16 + _S19 + (&_S1->value0_0)->value5_0 * (&_S6->cov_0)->c00_0;
    float _S26 = _S17 + _S20 + (&_S1->value0_0)->value4_0 * (&_S6->cov_0)->c00_0;
    float _S27 = (&_S1->value0_0)->value12_0 * s_diff_m22_T_0 + (&_S1->value0_0)->value9_0 * _S21 + (&_S1->value0_0)->value6_0 * _S24;
    float _S28 = (&_S1->value0_0)->value11_0 * s_diff_m22_T_0 + (&_S1->value0_0)->value12_0 * s_diff_m21_T_0 + (&_S1->value0_0)->value8_0 * _S21 + (&_S1->value0_0)->value9_0 * _S22 + (&_S1->value0_0)->value5_0 * _S24 + (&_S1->value0_0)->value6_0 * _S25;
    float _S29 = (&_S1->value0_0)->value11_0 * s_diff_m21_T_0 + (&_S1->value0_0)->value8_0 * _S22 + (&_S1->value0_0)->value5_0 * _S25;
    float _S30 = (&_S1->value0_0)->value10_0 * s_diff_m22_T_0 + (&_S1->value0_0)->value12_0 * s_diff_m20_T_0 + (&_S1->value0_0)->value7_0 * _S21 + (&_S1->value0_0)->value9_0 * _S23 + (&_S1->value0_0)->value4_0 * _S24 + (&_S1->value0_0)->value6_0 * _S26;
    float _S31 = (&_S1->value0_0)->value10_0 * s_diff_m21_T_0 + (&_S1->value0_0)->value11_0 * s_diff_m20_T_0 + (&_S1->value0_0)->value7_0 * _S22 + (&_S1->value0_0)->value8_0 * _S23 + (&_S1->value0_0)->value4_0 * _S25 + (&_S1->value0_0)->value5_0 * _S26;
    float _S32 = (&_S1->value0_0)->value10_0 * s_diff_m20_T_0 + (&_S1->value0_0)->value7_0 * _S23 + (&_S1->value0_0)->value4_0 * _S26;
    float _S33 = (&_S1->value0_0)->value12_0 * _S8.mz_0;
    float _S34 = (&_S1->value0_0)->value11_0 * _S8.mz_0;
    float _S35 = (&_S1->value0_0)->value10_0 * _S8.mz_0;
    float _S36 = (&_S1->value0_0)->value9_0 * _S8.my_0;
    float _S37 = (&_S1->value0_0)->value8_0 * _S8.my_0;
    float _S38 = (&_S1->value0_0)->value7_0 * _S8.my_0;
    float _S39 = (&_S1->value0_0)->value6_0 * _S8.mx_0;
    float _S40 = (&_S1->value0_0)->value5_0 * _S8.mx_0;
    float _S41 = (&_S1->value0_0)->value4_0 * _S8.mx_0;
    thread SymMat3_0 _S42 = _S7;
    (&_S42)->c22_0 = _S27;
    (&_S42)->c12_0 = _S28;
    (&_S42)->c11_0 = _S29;
    (&_S42)->c02_0 = _S30;
    (&_S42)->c01_0 = _S31;
    (&_S42)->c00_0 = _S32;
    thread SymMat3_0 _S43 = _S7;
    thread SymMat3_0 _S44 = _S42;
    SymMat3_0 _S45 = SymMat3_x24_syn_dadd_0(&_S43, &_S44);
    float _S46 = _S33 + _S36 + _S39;
    float _S47 = _S34 + _S37 + _S40;
    *_S2 = _S35 + _S38 + _S41;
    *_S3 = _S47;
    *_S4 = _S46;
    *_S5 = _S45;
    return;
}

void _S48(DiffPair_float_0 thread* _S49, DiffPair_float_0 thread* _S50, DiffPair_float_0 thread* _S51, DiffPair_SymMat3_0 thread* _S52, float _S53, float _S54, float _S55, float _S56, float _S57, float _S58, float _S59, float _S60, float _S61, float _S62, float _S63, float _S64, const CamPose_0 thread* _S65)
{
    thread s_paramCtx_s_bwdCallableCtx_world_to_cam_0 _S66;
    (&_S66)->value4_0 = _S53;
    (&_S66)->value5_0 = _S54;
    (&_S66)->value6_0 = _S55;
    (&_S66)->value7_0 = _S56;
    (&_S66)->value8_0 = _S57;
    (&_S66)->value9_0 = _S58;
    (&_S66)->value10_0 = _S59;
    (&_S66)->value11_0 = _S60;
    (&_S66)->value12_0 = _S61;
    thread s_bwdCallableCtx_world_to_cam_0 _S67;
    (&_S67)->value0_0 = _S66;
    s_bwdProp_world_to_cam_0(&_S67, &_S49->differential_0, &_S50->differential_0, &_S51->differential_0, &_S52->differential_0, _S65);
    return;
}

struct WorldToCamGrad_0
{
    float wx_0;
    float wy_0;
    float wz_0;
    SymMat3_0 covW_0;
};

WorldToCamGrad_0 world_to_cam_grad_0(float wx_1, float wy_1, float wz_1, const SymMat3_0 thread* covW_1, float r00_0, float r01_0, float r02_0, float r10_0, float r11_0, float r12_0, float r20_0, float r21_0, float r22_0, float tx_0, float ty_0, float tz_0, const CamPose_0 thread* cot_0)
{
    thread DiffPair_float_0 dpWx_0;
    (&dpWx_0)->primal_0 = wx_1;
    (&dpWx_0)->differential_0 = 0.0;
    thread DiffPair_float_0 dpWy_0;
    (&dpWy_0)->primal_0 = wy_1;
    (&dpWy_0)->differential_0 = 0.0;
    thread DiffPair_float_0 dpWz_0;
    (&dpWz_0)->primal_0 = wz_1;
    (&dpWz_0)->differential_0 = 0.0;
    thread SymMat3_0 zero_0;
    (&zero_0)->c00_0 = 0.0;
    (&zero_0)->c01_0 = 0.0;
    (&zero_0)->c02_0 = 0.0;
    (&zero_0)->c11_0 = 0.0;
    (&zero_0)->c12_0 = 0.0;
    (&zero_0)->c22_0 = 0.0;
    thread DiffPair_SymMat3_0 dpCov_0;
    (&dpCov_0)->primal_0 = *covW_1;
    (&dpCov_0)->differential_0 = zero_0;
    _S48(&dpWx_0, &dpWy_0, &dpWz_0, &dpCov_0, r00_0, r01_0, r02_0, r10_0, r11_0, r12_0, r20_0, r21_0, r22_0, tx_0, ty_0, tz_0, cot_0);
    thread WorldToCamGrad_0 g_0;
    (&g_0)->wx_0 = dpWx_0.differential_0;
    (&g_0)->wy_0 = dpWy_0.differential_0;
    (&g_0)->wz_0 = dpWz_0.differential_0;
    (&g_0)->covW_0 = dpCov_0.differential_0;
    return g_0;
}

struct KernelContext_0
{
    float device* cov3ds_world_bwd_0;
    float device* cot_means3d_cam_0;
    float device* cot_cov3ds_cam_0;
    float device* means3d_world_bwd_0;
    float device* viewmat_bwd_0;
    float device* grad_means3d_world_0;
    float device* grad_cov3ds_world_0;
};

[[kernel]] void world_to_cam_bwd(uint3 tid_0 [[thread_position_in_grid]], float device* cov3ds_world_bwd_1 [[buffer(6)]], float device* cot_means3d_cam_1 [[buffer(8)]], float device* cot_cov3ds_cam_1 [[buffer(9)]], float device* means3d_world_bwd_1 [[buffer(5)]], float device* viewmat_bwd_1 [[buffer(7)]], float device* grad_means3d_world_1 [[buffer(10)]], float device* grad_cov3ds_world_1 [[buffer(11)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->cov3ds_world_bwd_0 = cov3ds_world_bwd_1;
    (&kernelContext_0)->cot_means3d_cam_0 = cot_means3d_cam_1;
    (&kernelContext_0)->cot_cov3ds_cam_0 = cot_cov3ds_cam_1;
    (&kernelContext_0)->means3d_world_bwd_0 = means3d_world_bwd_1;
    (&kernelContext_0)->viewmat_bwd_0 = viewmat_bwd_1;
    (&kernelContext_0)->grad_means3d_world_0 = grad_means3d_world_1;
    (&kernelContext_0)->grad_cov3ds_world_0 = grad_cov3ds_world_1;
    uint i_0 = tid_0.x;
    uint mBase_0 = i_0 * 3U;
    uint cBase_0 = i_0 * 6U;
    thread SymMat3_0 covW_2;
    (&covW_2)->c00_0 = cov3ds_world_bwd_1[cBase_0];
    uint _S68 = cBase_0 + 1U;
    (&covW_2)->c01_0 = cov3ds_world_bwd_1[_S68];
    uint _S69 = cBase_0 + 2U;
    (&covW_2)->c02_0 = cov3ds_world_bwd_1[_S69];
    uint _S70 = cBase_0 + 3U;
    (&covW_2)->c11_0 = cov3ds_world_bwd_1[_S70];
    uint _S71 = cBase_0 + 4U;
    (&covW_2)->c12_0 = cov3ds_world_bwd_1[_S71];
    uint _S72 = cBase_0 + 5U;
    (&covW_2)->c22_0 = cov3ds_world_bwd_1[_S72];
    thread CamPose_0 cot_1;
    (&cot_1)->mx_0 = cot_means3d_cam_1[mBase_0];
    uint _S73 = mBase_0 + 1U;
    (&cot_1)->my_0 = cot_means3d_cam_1[_S73];
    uint _S74 = mBase_0 + 2U;
    (&cot_1)->mz_0 = cot_means3d_cam_1[_S74];
    (&(&cot_1)->cov_0)->c00_0 = cot_cov3ds_cam_1[cBase_0];
    (&(&cot_1)->cov_0)->c01_0 = cot_cov3ds_cam_1[_S68];
    (&(&cot_1)->cov_0)->c02_0 = cot_cov3ds_cam_1[_S69];
    (&(&cot_1)->cov_0)->c11_0 = cot_cov3ds_cam_1[_S70];
    (&(&cot_1)->cov_0)->c12_0 = cot_cov3ds_cam_1[_S71];
    (&(&cot_1)->cov_0)->c22_0 = cot_cov3ds_cam_1[_S72];
    float _S75 = means3d_world_bwd_1[mBase_0];
    float _S76 = means3d_world_bwd_1[_S73];
    float _S77 = means3d_world_bwd_1[_S74];
    float _S78 = viewmat_bwd_1[int(0)];
    float _S79 = viewmat_bwd_1[int(1)];
    float _S80 = viewmat_bwd_1[int(2)];
    float _S81 = viewmat_bwd_1[int(4)];
    float _S82 = viewmat_bwd_1[int(5)];
    float _S83 = viewmat_bwd_1[int(6)];
    float _S84 = viewmat_bwd_1[int(8)];
    float _S85 = viewmat_bwd_1[int(9)];
    float _S86 = viewmat_bwd_1[int(10)];
    float _S87 = viewmat_bwd_1[int(3)];
    float _S88 = viewmat_bwd_1[int(7)];
    float _S89 = viewmat_bwd_1[int(11)];
    thread SymMat3_0 _S90 = covW_2;
    thread CamPose_0 _S91 = cot_1;
    WorldToCamGrad_0 _S92 = world_to_cam_grad_0(_S75, _S76, _S77, &_S90, _S78, _S79, _S80, _S81, _S82, _S83, _S84, _S85, _S86, _S87, _S88, _S89, &_S91);
    *(grad_means3d_world_1+mBase_0) = _S92.wx_0;
    *(grad_means3d_world_1+_S73) = _S92.wy_0;
    *(grad_means3d_world_1+_S74) = _S92.wz_0;
    *(grad_cov3ds_world_1+cBase_0) = _S92.covW_0.c00_0;
    *(grad_cov3ds_world_1+_S68) = _S92.covW_0.c01_0;
    *(grad_cov3ds_world_1+_S69) = _S92.covW_0.c02_0;
    *(grad_cov3ds_world_1+_S70) = _S92.covW_0.c11_0;
    *(grad_cov3ds_world_1+_S71) = _S92.covW_0.c12_0;
    *(grad_cov3ds_world_1+_S72) = _S92.covW_0.c22_0;
    return;
}

