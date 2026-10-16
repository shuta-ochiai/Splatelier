#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct Tuple_0
{
    float value0_0;
};

Tuple_0 s_apply_min_0(float x_0, float y_0)
{
    Tuple_0 _S1 = { min(x_0, y_0) };
    return _S1;
}

struct s_bwdCallableCtx_min_0
{
    float _S2;
    float _S3;
};

struct DiffPair_float_0
{
    float primal_0;
    float differential_0;
};

void _d_min_0(DiffPair_float_0 thread* dpx_0, DiffPair_float_0 thread* dpy_0, float dOut_0)
{
    DiffPair_float_0 _S4 = *dpx_0;
    float _S5;
    if(((*dpx_0).primal_0) < ((*dpy_0).primal_0))
    {
        _S5 = dOut_0;
    }
    else
    {
        if(((*dpx_0).primal_0) > ((*dpy_0).primal_0))
        {
            _S5 = 0.0;
        }
        else
        {
            _S5 = 0.5 * dOut_0;
        }
    }
    dpx_0->primal_0 = _S4.primal_0;
    dpx_0->differential_0 = _S5;
    DiffPair_float_0 _S6 = *dpy_0;
    if(((*dpy_0).primal_0) < (_S4.primal_0))
    {
        _S5 = dOut_0;
    }
    else
    {
        if(((*dpy_0).primal_0) > ((*dpx_0).primal_0))
        {
            _S5 = 0.0;
        }
        else
        {
            _S5 = 0.5 * dOut_0;
        }
    }
    dpy_0->primal_0 = _S6.primal_0;
    dpy_0->differential_0 = _S5;
    return;
}

void s_bwdProp_min_0(const s_bwdCallableCtx_min_0 thread* ctx_0, float thread* d_x_0, float thread* d_y_0, float d_out_0)
{
    thread DiffPair_float_0 _S7;
    (&_S7)->primal_0 = ctx_0->_S2;
    thread DiffPair_float_0 _S8;
    (&_S8)->primal_0 = ctx_0->_S3;
    _d_min_0(&_S7, &_S8, d_out_0);
    *d_x_0 = (&_S7)->differential_0;
    *d_y_0 = (&_S8)->differential_0;
    return;
}

struct Tuple_1
{
    float value0_1;
};

Tuple_1 s_apply_max_0(float x_1, float y_1)
{
    Tuple_1 _S9 = { max(x_1, y_1) };
    return _S9;
}

struct s_bwdCallableCtx_max_0
{
    float _S10;
    float _S11;
};

void _d_max_0(DiffPair_float_0 thread* dpx_1, DiffPair_float_0 thread* dpy_1, float dOut_1)
{
    DiffPair_float_0 _S12 = *dpx_1;
    float _S13;
    if(((*dpx_1).primal_0) > ((*dpy_1).primal_0))
    {
        _S13 = dOut_1;
    }
    else
    {
        if(((*dpx_1).primal_0) < ((*dpy_1).primal_0))
        {
            _S13 = 0.0;
        }
        else
        {
            _S13 = 0.5 * dOut_1;
        }
    }
    dpx_1->primal_0 = _S12.primal_0;
    dpx_1->differential_0 = _S13;
    DiffPair_float_0 _S14 = *dpy_1;
    if(((*dpy_1).primal_0) > (_S12.primal_0))
    {
        _S13 = dOut_1;
    }
    else
    {
        if(((*dpy_1).primal_0) < ((*dpx_1).primal_0))
        {
            _S13 = 0.0;
        }
        else
        {
            _S13 = 0.5 * dOut_1;
        }
    }
    dpy_1->primal_0 = _S14.primal_0;
    dpy_1->differential_0 = _S13;
    return;
}

void s_bwdProp_max_0(const s_bwdCallableCtx_max_0 thread* ctx_1, float thread* d_x_1, float thread* d_y_1, float d_out_1)
{
    thread DiffPair_float_0 _S15;
    (&_S15)->primal_0 = ctx_1->_S10;
    thread DiffPair_float_0 _S16;
    (&_S16)->primal_0 = ctx_1->_S11;
    _d_max_0(&_S15, &_S16, d_out_1);
    *d_x_1 = (&_S15)->differential_0;
    *d_y_1 = (&_S16)->differential_0;
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

struct DiffPair_SymMat3_0
{
    SymMat3_0 primal_0;
    SymMat3_0 differential_0;
};

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

struct s_paramCtx_s_bwdCallableCtx_persp_proj_0
{
    float value0_2;
    float value1_0;
    float value2_0;
    SymMat3_0 value3_0;
    float value4_0;
    float value5_0;
    float value6_0;
    float value7_0;
    float value8_0;
    float value9_0;
};

struct s_bwdCallableCtx_persp_proj_0
{
    s_paramCtx_s_bwdCallableCtx_persp_proj_0 value0_3;
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

void s_bwdProp_persp_proj_0(const s_bwdCallableCtx_persp_proj_0 thread* _S17, float thread* _S18, float thread* _S19, float thread* _S20, SymMat3_0 thread* _S21, const ProjResult_0 thread* _S22)
{
    float rz_0 = 1.0 / (&_S17->value0_3)->value2_0;
    float _S23 = (&_S17->value0_3)->value2_0 * (&_S17->value0_3)->value2_0;
    float rz2_0 = rz_0 * rz_0;
    float _S24 = 0.30000001192092896 * (0.5 * (&_S17->value0_3)->value8_0 / (&_S17->value0_3)->value4_0);
    float lim_x_pos_0 = ((&_S17->value0_3)->value8_0 - (&_S17->value0_3)->value6_0) / (&_S17->value0_3)->value4_0 + _S24;
    float _S25 = 0.30000001192092896 * (0.5 * (&_S17->value0_3)->value9_0 / (&_S17->value0_3)->value5_0);
    float lim_y_pos_0 = ((&_S17->value0_3)->value9_0 - (&_S17->value0_3)->value7_0) / (&_S17->value0_3)->value5_0 + _S25;
    float lim_y_neg_0 = (&_S17->value0_3)->value7_0 / (&_S17->value0_3)->value5_0 + _S25;
    float _S26 = - ((&_S17->value0_3)->value6_0 / (&_S17->value0_3)->value4_0 + _S24);
    float _S27 = (&_S17->value0_3)->value0_2 * rz_0;
    Tuple_1 _S28 = s_apply_max_0(_S26, _S27);
    thread s_bwdCallableCtx_max_0 _S29;
    (&_S29)->_S10 = _S26;
    (&_S29)->_S11 = _S27;
    Tuple_0 _S30 = s_apply_min_0(lim_x_pos_0, _S28.value0_1);
    thread s_bwdCallableCtx_min_0 _S31;
    (&_S31)->_S2 = lim_x_pos_0;
    (&_S31)->_S3 = _S28.value0_1;
    float tx_0 = (&_S17->value0_3)->value2_0 * _S30.value0_0;
    float _S32 = - lim_y_neg_0;
    float _S33 = (&_S17->value0_3)->value1_0 * rz_0;
    Tuple_1 _S34 = s_apply_max_0(_S32, _S33);
    thread s_bwdCallableCtx_max_0 _S35;
    (&_S35)->_S10 = _S32;
    (&_S35)->_S11 = _S33;
    Tuple_0 _S36 = s_apply_min_0(lim_y_pos_0, _S34.value0_1);
    thread s_bwdCallableCtx_min_0 _S37;
    (&_S37)->_S2 = lim_y_pos_0;
    (&_S37)->_S3 = _S34.value0_1;
    float j00_0 = (&_S17->value0_3)->value4_0 * rz_0;
    float _S38 = - (&_S17->value0_3)->value4_0;
    float _S39 = _S38 * tx_0;
    float j02_0 = _S39 * rz2_0;
    float j11_0 = (&_S17->value0_3)->value5_0 * rz_0;
    float _S40 = - (&_S17->value0_3)->value5_0;
    float _S41 = _S40 * ((&_S17->value0_3)->value2_0 * _S36.value0_0);
    float j12_0 = _S41 * rz2_0;
    float t02_0 = j00_0 * (&(&_S17->value0_3)->value3_0)->c02_0 + j02_0 * (&(&_S17->value0_3)->value3_0)->c22_0;
    SymMat3_0 _S42 = SymMat3_x24_syn_dzero_0();
    float _S43 = (&_S17->value0_3)->value5_0 * (&_S17->value0_3)->value1_0 * _S22->py_0;
    float _S44 = (&_S17->value0_3)->value5_0 * (rz_0 * _S22->py_0);
    float _S45 = (&_S17->value0_3)->value4_0 * (&_S17->value0_3)->value0_2 * _S22->px_0;
    float _S46 = (&_S17->value0_3)->value4_0 * (rz_0 * _S22->px_0);
    float s_diff_t12_T_0 = j12_0 * (&_S22->cov2d_0)->c11_1;
    float s_diff_t11_T_0 = j11_0 * (&_S22->cov2d_0)->c11_1;
    float s_diff_t01_T_0 = j11_0 * (&_S22->cov2d_0)->c01_1;
    float s_diff_t00_T_0 = j00_0 * (&_S22->cov2d_0)->c00_1;
    float _S47 = j11_0 * s_diff_t11_T_0;
    float _S48 = j12_0 * (&_S22->cov2d_0)->c01_1 + j02_0 * (&_S22->cov2d_0)->c00_1;
    float _S49 = j00_0 * s_diff_t01_T_0;
    float _S50 = j00_0 * s_diff_t00_T_0;
    float _S51 = j12_0 * s_diff_t12_T_0 + j02_0 * _S48;
    float _S52 = j11_0 * s_diff_t12_T_0 + j12_0 * s_diff_t11_T_0 + j02_0 * s_diff_t01_T_0;
    float _S53 = j00_0 * _S48 + j02_0 * s_diff_t00_T_0;
    float _S54 = (j11_0 * (&(&_S17->value0_3)->value3_0)->c12_0 + j12_0 * (&(&_S17->value0_3)->value3_0)->c22_0) * (&_S22->cov2d_0)->c11_1 + t02_0 * (&_S22->cov2d_0)->c01_1 + (&(&_S17->value0_3)->value3_0)->c22_0 * s_diff_t12_T_0 + (&(&_S17->value0_3)->value3_0)->c12_0 * s_diff_t11_T_0;
    float _S55 = _S41 * _S54;
    float s_diff_ty_T_0 = _S40 * (rz2_0 * _S54);
    float _S56 = (&_S17->value0_3)->value5_0 * ((j11_0 * (&(&_S17->value0_3)->value3_0)->c11_0 + j12_0 * (&(&_S17->value0_3)->value3_0)->c12_0) * (&_S22->cov2d_0)->c11_1 + (j00_0 * (&(&_S17->value0_3)->value3_0)->c01_0 + j02_0 * (&(&_S17->value0_3)->value3_0)->c12_0) * (&_S22->cov2d_0)->c01_1 + (&(&_S17->value0_3)->value3_0)->c12_0 * s_diff_t12_T_0 + (&(&_S17->value0_3)->value3_0)->c11_0 * s_diff_t11_T_0);
    float _S57 = t02_0 * (&_S22->cov2d_0)->c00_1 + (&(&_S17->value0_3)->value3_0)->c22_0 * _S48 + (&(&_S17->value0_3)->value3_0)->c12_0 * s_diff_t01_T_0 + (&(&_S17->value0_3)->value3_0)->c02_0 * s_diff_t00_T_0;
    float _S58 = _S39 * _S57;
    float s_diff_tx_T_0 = _S38 * (rz2_0 * _S57);
    float _S59 = (&_S17->value0_3)->value4_0 * ((j00_0 * (&(&_S17->value0_3)->value3_0)->c00_0 + j02_0 * (&(&_S17->value0_3)->value3_0)->c02_0) * (&_S22->cov2d_0)->c00_1 + (&(&_S17->value0_3)->value3_0)->c02_0 * _S48 + (&(&_S17->value0_3)->value3_0)->c01_0 * s_diff_t01_T_0 + (&(&_S17->value0_3)->value3_0)->c00_0 * s_diff_t00_T_0);
    float _S60 = (&_S17->value0_3)->value2_0 * s_diff_ty_T_0;
    float _S61 = _S36.value0_0 * s_diff_ty_T_0;
    thread float _S62 = 0.0;
    thread float _S63 = 0.0;
    thread s_bwdCallableCtx_min_0 _S64 = _S37;
    s_bwdProp_min_0(&_S64, &_S62, &_S63, _S60);
    thread float _S65 = 0.0;
    thread float _S66 = 0.0;
    thread s_bwdCallableCtx_max_0 _S67 = _S35;
    s_bwdProp_max_0(&_S67, &_S65, &_S66, _S63);
    float _S68 = (&_S17->value0_3)->value1_0 * _S66;
    float _S69 = rz_0 * _S66;
    float _S70 = (&_S17->value0_3)->value2_0 * s_diff_tx_T_0;
    float _S71 = _S30.value0_0 * s_diff_tx_T_0;
    thread float _S72 = 0.0;
    thread float _S73 = 0.0;
    thread s_bwdCallableCtx_min_0 _S74 = _S31;
    s_bwdProp_min_0(&_S74, &_S72, &_S73, _S70);
    thread float _S75 = 0.0;
    thread float _S76 = 0.0;
    thread s_bwdCallableCtx_max_0 _S77 = _S29;
    s_bwdProp_max_0(&_S77, &_S75, &_S76, _S73);
    float _S78 = rz_0 * (_S55 + _S58);
    float _S79 = - ((_S43 + _S45 + _S56 + _S59 + _S68 + (&_S17->value0_3)->value0_2 * _S76 + _S78 + _S78) / _S23);
    float _S80 = _S44 + _S69;
    float _S81 = _S46 + rz_0 * _S76;
    thread SymMat3_0 _S82 = _S42;
    (&_S82)->c22_0 = _S51;
    (&_S82)->c12_0 = _S52;
    (&_S82)->c11_0 = _S47;
    (&_S82)->c02_0 = _S53;
    (&_S82)->c01_0 = _S49;
    (&_S82)->c00_0 = _S50;
    thread SymMat3_0 _S83 = _S42;
    thread SymMat3_0 _S84 = _S82;
    SymMat3_0 _S85 = SymMat3_x24_syn_dadd_0(&_S83, &_S84);
    float _S86 = _S61 + _S71 + _S79;
    *_S18 = _S81;
    *_S19 = _S80;
    *_S20 = _S86;
    *_S21 = _S85;
    return;
}

void _S87(DiffPair_float_0 thread* _S88, DiffPair_float_0 thread* _S89, DiffPair_float_0 thread* _S90, DiffPair_SymMat3_0 thread* _S91, float _S92, float _S93, float _S94, float _S95, float _S96, float _S97, const ProjResult_0 thread* _S98)
{
    thread s_paramCtx_s_bwdCallableCtx_persp_proj_0 _S99;
    (&_S99)->value0_2 = _S88->primal_0;
    (&_S99)->value1_0 = _S89->primal_0;
    (&_S99)->value2_0 = _S90->primal_0;
    (&_S99)->value3_0 = _S91->primal_0;
    (&_S99)->value4_0 = _S92;
    (&_S99)->value5_0 = _S93;
    (&_S99)->value6_0 = _S94;
    (&_S99)->value7_0 = _S95;
    (&_S99)->value8_0 = _S96;
    (&_S99)->value9_0 = _S97;
    thread s_bwdCallableCtx_persp_proj_0 _S100;
    (&_S100)->value0_3 = _S99;
    s_bwdProp_persp_proj_0(&_S100, &_S88->differential_0, &_S89->differential_0, &_S90->differential_0, &_S91->differential_0, _S98);
    return;
}

struct PerspProjGrad_0
{
    float mx_0;
    float my_0;
    float mz_0;
    SymMat3_0 cov3d_0;
};

PerspProjGrad_0 persp_proj_grad_0(float mx_1, float my_1, float mz_1, const SymMat3_0 thread* cov3d_1, float fx_0, float fy_0, float cx_0, float cy_0, float imageWidth_0, float imageHeight_0, const ProjResult_0 thread* cot_0)
{
    thread DiffPair_float_0 dpMx_0;
    (&dpMx_0)->primal_0 = mx_1;
    (&dpMx_0)->differential_0 = 0.0;
    thread DiffPair_float_0 dpMy_0;
    (&dpMy_0)->primal_0 = my_1;
    (&dpMy_0)->differential_0 = 0.0;
    thread DiffPair_float_0 dpMz_0;
    (&dpMz_0)->primal_0 = mz_1;
    (&dpMz_0)->differential_0 = 0.0;
    thread SymMat3_0 zero_0;
    (&zero_0)->c00_0 = 0.0;
    (&zero_0)->c01_0 = 0.0;
    (&zero_0)->c02_0 = 0.0;
    (&zero_0)->c11_0 = 0.0;
    (&zero_0)->c12_0 = 0.0;
    (&zero_0)->c22_0 = 0.0;
    thread DiffPair_SymMat3_0 dpCov_0;
    (&dpCov_0)->primal_0 = *cov3d_1;
    (&dpCov_0)->differential_0 = zero_0;
    _S87(&dpMx_0, &dpMy_0, &dpMz_0, &dpCov_0, fx_0, fy_0, cx_0, cy_0, imageWidth_0, imageHeight_0, cot_0);
    thread PerspProjGrad_0 g_0;
    (&g_0)->mx_0 = dpMx_0.differential_0;
    (&g_0)->my_0 = dpMy_0.differential_0;
    (&g_0)->mz_0 = dpMz_0.differential_0;
    (&g_0)->cov3d_0 = dpCov_0.differential_0;
    return g_0;
}

struct KernelContext_0
{
    float device* cov3ds_cam_bwd_0;
    float device* cot_mean2d_0;
    float device* cot_cov2d_0;
    float device* means3d_cam_bwd_0;
    float device* intrinsics_bwd_0;
    float device* image_dims_bwd_0;
    float device* grad_means3d_0;
    float device* grad_cov3d_0;
};

[[kernel]] void persp_proj_bwd(uint3 tid_0 [[thread_position_in_grid]], float device* cov3ds_cam_bwd_1 [[buffer(7)]], float device* cot_mean2d_1 [[buffer(10)]], float device* cot_cov2d_1 [[buffer(11)]], float device* means3d_cam_bwd_1 [[buffer(6)]], float device* intrinsics_bwd_1 [[buffer(8)]], float device* image_dims_bwd_1 [[buffer(9)]], float device* grad_means3d_1 [[buffer(12)]], float device* grad_cov3d_1 [[buffer(13)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->cov3ds_cam_bwd_0 = cov3ds_cam_bwd_1;
    (&kernelContext_0)->cot_mean2d_0 = cot_mean2d_1;
    (&kernelContext_0)->cot_cov2d_0 = cot_cov2d_1;
    (&kernelContext_0)->means3d_cam_bwd_0 = means3d_cam_bwd_1;
    (&kernelContext_0)->intrinsics_bwd_0 = intrinsics_bwd_1;
    (&kernelContext_0)->image_dims_bwd_0 = image_dims_bwd_1;
    (&kernelContext_0)->grad_means3d_0 = grad_means3d_1;
    (&kernelContext_0)->grad_cov3d_0 = grad_cov3d_1;
    uint i_0 = tid_0.x;
    uint mBase_0 = i_0 * 3U;
    uint cBase_0 = i_0 * 6U;
    uint pBase_0 = i_0 * 2U;
    thread SymMat3_0 cov3d_2;
    (&cov3d_2)->c00_0 = cov3ds_cam_bwd_1[cBase_0];
    uint _S101 = cBase_0 + 1U;
    (&cov3d_2)->c01_0 = cov3ds_cam_bwd_1[_S101];
    uint _S102 = cBase_0 + 2U;
    (&cov3d_2)->c02_0 = cov3ds_cam_bwd_1[_S102];
    uint _S103 = cBase_0 + 3U;
    (&cov3d_2)->c11_0 = cov3ds_cam_bwd_1[_S103];
    uint _S104 = cBase_0 + 4U;
    (&cov3d_2)->c12_0 = cov3ds_cam_bwd_1[_S104];
    uint _S105 = cBase_0 + 5U;
    (&cov3d_2)->c22_0 = cov3ds_cam_bwd_1[_S105];
    thread ProjResult_0 cot_1;
    (&cot_1)->px_0 = cot_mean2d_1[pBase_0];
    (&cot_1)->py_0 = cot_mean2d_1[pBase_0 + 1U];
    (&(&cot_1)->cov2d_0)->c00_1 = cot_cov2d_1[mBase_0];
    uint _S106 = mBase_0 + 1U;
    (&(&cot_1)->cov2d_0)->c01_1 = cot_cov2d_1[_S106];
    uint _S107 = mBase_0 + 2U;
    (&(&cot_1)->cov2d_0)->c11_1 = cot_cov2d_1[_S107];
    float _S108 = means3d_cam_bwd_1[mBase_0];
    float _S109 = means3d_cam_bwd_1[_S106];
    float _S110 = means3d_cam_bwd_1[_S107];
    float _S111 = intrinsics_bwd_1[int(0)];
    float _S112 = intrinsics_bwd_1[int(1)];
    float _S113 = intrinsics_bwd_1[int(2)];
    float _S114 = intrinsics_bwd_1[int(3)];
    float _S115 = image_dims_bwd_1[int(0)];
    float _S116 = image_dims_bwd_1[int(1)];
    thread SymMat3_0 _S117 = cov3d_2;
    thread ProjResult_0 _S118 = cot_1;
    PerspProjGrad_0 _S119 = persp_proj_grad_0(_S108, _S109, _S110, &_S117, _S111, _S112, _S113, _S114, _S115, _S116, &_S118);
    *(grad_means3d_1+mBase_0) = _S119.mx_0;
    *(grad_means3d_1+_S106) = _S119.my_0;
    *(grad_means3d_1+_S107) = _S119.mz_0;
    *(grad_cov3d_1+cBase_0) = _S119.cov3d_0.c00_0;
    *(grad_cov3d_1+_S101) = _S119.cov3d_0.c01_0;
    *(grad_cov3d_1+_S102) = _S119.cov3d_0.c02_0;
    *(grad_cov3d_1+_S103) = _S119.cov3d_0.c11_0;
    *(grad_cov3d_1+_S104) = _S119.cov3d_0.c12_0;
    *(grad_cov3d_1+_S105) = _S119.cov3d_0.c22_0;
    return;
}

