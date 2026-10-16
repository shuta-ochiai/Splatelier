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

struct Tuple_1
{
    float value0_1;
};

Tuple_1 s_apply_min_0(float x_1, float y_0)
{
    Tuple_1 _S5 = { min(x_1, y_0) };
    return _S5;
}

struct s_bwdCallableCtx_min_0
{
    float _S6;
    float _S7;
};

void _d_min_0(DiffPair_float_0 thread* dpx_1, DiffPair_float_0 thread* dpy_0, float dOut_1)
{
    DiffPair_float_0 _S8 = *dpx_1;
    float _S9;
    if(((*dpx_1).primal_0) < ((*dpy_0).primal_0))
    {
        _S9 = dOut_1;
    }
    else
    {
        if(((*dpx_1).primal_0) > ((*dpy_0).primal_0))
        {
            _S9 = 0.0;
        }
        else
        {
            _S9 = 0.5 * dOut_1;
        }
    }
    dpx_1->primal_0 = _S8.primal_0;
    dpx_1->differential_0 = _S9;
    DiffPair_float_0 _S10 = *dpy_0;
    if(((*dpy_0).primal_0) < (_S8.primal_0))
    {
        _S9 = dOut_1;
    }
    else
    {
        if(((*dpy_0).primal_0) > ((*dpx_1).primal_0))
        {
            _S9 = 0.0;
        }
        else
        {
            _S9 = 0.5 * dOut_1;
        }
    }
    dpy_0->primal_0 = _S10.primal_0;
    dpy_0->differential_0 = _S9;
    return;
}

void s_bwdProp_min_0(const s_bwdCallableCtx_min_0 thread* ctx_1, float thread* d_x_1, float thread* d_y_0, float d_out_1)
{
    thread DiffPair_float_0 _S11;
    (&_S11)->primal_0 = ctx_1->_S6;
    thread DiffPair_float_0 _S12;
    (&_S12)->primal_0 = ctx_1->_S7;
    _d_min_0(&_S11, &_S12, d_out_1);
    *d_x_1 = (&_S11)->differential_0;
    *d_y_0 = (&_S12)->differential_0;
    return;
}

struct Tuple_2
{
    float value0_2;
};

Tuple_2 s_apply_max_0(float x_2, float y_1)
{
    Tuple_2 _S13 = { max(x_2, y_1) };
    return _S13;
}

struct s_bwdCallableCtx_max_0
{
    float _S14;
    float _S15;
};

void _d_max_0(DiffPair_float_0 thread* dpx_2, DiffPair_float_0 thread* dpy_1, float dOut_2)
{
    DiffPair_float_0 _S16 = *dpx_2;
    float _S17;
    if(((*dpx_2).primal_0) > ((*dpy_1).primal_0))
    {
        _S17 = dOut_2;
    }
    else
    {
        if(((*dpx_2).primal_0) < ((*dpy_1).primal_0))
        {
            _S17 = 0.0;
        }
        else
        {
            _S17 = 0.5 * dOut_2;
        }
    }
    dpx_2->primal_0 = _S16.primal_0;
    dpx_2->differential_0 = _S17;
    DiffPair_float_0 _S18 = *dpy_1;
    if(((*dpy_1).primal_0) > (_S16.primal_0))
    {
        _S17 = dOut_2;
    }
    else
    {
        if(((*dpy_1).primal_0) < ((*dpx_2).primal_0))
        {
            _S17 = 0.0;
        }
        else
        {
            _S17 = 0.5 * dOut_2;
        }
    }
    dpy_1->primal_0 = _S18.primal_0;
    dpy_1->differential_0 = _S17;
    return;
}

void s_bwdProp_max_0(const s_bwdCallableCtx_max_0 thread* ctx_2, float thread* d_x_2, float thread* d_y_1, float d_out_2)
{
    thread DiffPair_float_0 _S19;
    (&_S19)->primal_0 = ctx_2->_S14;
    thread DiffPair_float_0 _S20;
    (&_S20)->primal_0 = ctx_2->_S15;
    _d_max_0(&_S19, &_S20, d_out_2);
    *d_x_2 = (&_S19)->differential_0;
    *d_y_1 = (&_S20)->differential_0;
    return;
}

struct ProjectedScreen_0
{
    float px_0;
    float py_0;
    float conic_a_0;
    float conic_b_0;
    float conic_c_0;
    float depth_0;
    float cov2d_a_0;
    float cov2d_b_0;
    float cov2d_c_0;
};

struct s_paramCtx_s_bwdCallableCtx_project_to_screen_0
{
    float value0_3;
    float value1_0;
    float value2_0;
    float value3_0;
    float value4_0;
    float value5_0;
    float value6_0;
    float value7_0;
    float value8_0;
    float value9_0;
    float value10_0;
    float value11_0;
    float value12_0;
    float value13_0;
    float value14_0;
    float value15_0;
    float value16_0;
    float value17_0;
    float value18_0;
    float value19_0;
    float value20_0;
    float value21_0;
    float value22_0;
    float value23_0;
    float value24_0;
    float value25_0;
    float value26_0;
    float value27_0;
};

struct s_bwdCallableCtx_project_to_screen_0
{
    s_paramCtx_s_bwdCallableCtx_project_to_screen_0 value0_4;
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

struct Tuple_3
{
    SymMat3_0 value0_5;
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

struct Tuple_4
{
    Mat3_0 value0_6;
};

Tuple_4 s_apply_quat_to_rotmat_0(float _S21, float _S22, float _S23, float _S24)
{
    Tuple_0 _S25 = s_apply_rsqrt_0(_S22 * _S22 + _S23 * _S23 + _S24 * _S24 + _S21 * _S21);
    float w_0 = _S21 * _S25.value0_0;
    float x_3 = _S22 * _S25.value0_0;
    float y_2 = _S23 * _S25.value0_0;
    float z_0 = _S24 * _S25.value0_0;
    float x2_0 = x_3 * x_3;
    float y2_0 = y_2 * y_2;
    float z2_0 = z_0 * z_0;
    float xy_0 = x_3 * y_2;
    float xz_0 = x_3 * z_0;
    float yz_0 = y_2 * z_0;
    float wx_0 = w_0 * x_3;
    float wy_0 = w_0 * y_2;
    float wz_0 = w_0 * z_0;
    float _S26 = 2.0 * (xy_0 - wz_0);
    float _S27 = 2.0 * (xz_0 + wy_0);
    float _S28 = 2.0 * (xy_0 + wz_0);
    float _S29 = 1.0 - 2.0 * (x2_0 + z2_0);
    float _S30 = 2.0 * (yz_0 - wx_0);
    float _S31 = 2.0 * (xz_0 - wy_0);
    float _S32 = 2.0 * (yz_0 + wx_0);
    float _S33 = 1.0 - 2.0 * (x2_0 + y2_0);
    thread Tuple_4 _S34;
    (&(&_S34)->value0_6)->r00_0 = 1.0 - 2.0 * (y2_0 + z2_0);
    (&(&_S34)->value0_6)->r01_0 = _S26;
    (&(&_S34)->value0_6)->r02_0 = _S27;
    (&(&_S34)->value0_6)->r10_0 = _S28;
    (&(&_S34)->value0_6)->r11_0 = _S29;
    (&(&_S34)->value0_6)->r12_0 = _S30;
    (&(&_S34)->value0_6)->r20_0 = _S31;
    (&(&_S34)->value0_6)->r21_0 = _S32;
    (&(&_S34)->value0_6)->r22_0 = _S33;
    return _S34;
}

Tuple_3 s_apply_quat_scale_to_cov3d_0(float _S35, float _S36, float _S37, float _S38, float _S39, float _S40, float _S41)
{
    Tuple_4 _S42 = s_apply_quat_to_rotmat_0(_S35, _S36, _S37, _S38);
    float M00_0 = _S42.value0_6.r00_0 * _S39;
    float M01_0 = _S42.value0_6.r01_0 * _S40;
    float M02_0 = _S42.value0_6.r02_0 * _S41;
    float M10_0 = _S42.value0_6.r10_0 * _S39;
    float M11_0 = _S42.value0_6.r11_0 * _S40;
    float M12_0 = _S42.value0_6.r12_0 * _S41;
    float M20_0 = _S42.value0_6.r20_0 * _S39;
    float M21_0 = _S42.value0_6.r21_0 * _S40;
    float M22_0 = _S42.value0_6.r22_0 * _S41;
    float _S43 = M00_0 * M10_0 + M01_0 * M11_0 + M02_0 * M12_0;
    float _S44 = M00_0 * M20_0 + M01_0 * M21_0 + M02_0 * M22_0;
    float _S45 = M10_0 * M10_0 + M11_0 * M11_0 + M12_0 * M12_0;
    float _S46 = M10_0 * M20_0 + M11_0 * M21_0 + M12_0 * M22_0;
    float _S47 = M20_0 * M20_0 + M21_0 * M21_0 + M22_0 * M22_0;
    thread Tuple_3 _S48;
    (&(&_S48)->value0_5)->c00_0 = M00_0 * M00_0 + M01_0 * M01_0 + M02_0 * M02_0;
    (&(&_S48)->value0_5)->c01_0 = _S43;
    (&(&_S48)->value0_5)->c02_0 = _S44;
    (&(&_S48)->value0_5)->c11_0 = _S45;
    (&(&_S48)->value0_5)->c12_0 = _S46;
    (&(&_S48)->value0_5)->c22_0 = _S47;
    return _S48;
}

struct s_paramCtx_s_bwdCallableCtx_quat_scale_to_cov3d_0
{
    float value0_7;
    float value1_1;
    float value2_1;
    float value3_1;
    float value4_1;
    float value5_1;
    float value6_1;
};

struct CamPose_0
{
    float mx_0;
    float my_0;
    float mz_0;
    SymMat3_0 cov_0;
};

struct Tuple_5
{
    CamPose_0 value0_8;
};

Tuple_5 s_apply_world_to_cam_0(float _S49, float _S50, float _S51, const SymMat3_0 thread* _S52, float r00_1, float r01_1, float r02_1, float r10_1, float r11_1, float r12_1, float r20_1, float r21_1, float r22_1, float tx_0, float ty_0, float tz_0)
{
    CamPose_0 _S53;
    (&_S53)->mx_0 = r00_1 * _S49 + r01_1 * _S50 + r02_1 * _S51 + tx_0;
    (&_S53)->my_0 = r10_1 * _S49 + r11_1 * _S50 + r12_1 * _S51 + ty_0;
    (&_S53)->mz_0 = r20_1 * _S49 + r21_1 * _S50 + r22_1 * _S51 + tz_0;
    float m00_0 = r00_1 * _S52->c00_0 + r01_1 * _S52->c01_0 + r02_1 * _S52->c02_0;
    float m01_0 = r00_1 * _S52->c01_0 + r01_1 * _S52->c11_0 + r02_1 * _S52->c12_0;
    float m02_0 = r00_1 * _S52->c02_0 + r01_1 * _S52->c12_0 + r02_1 * _S52->c22_0;
    float m10_0 = r10_1 * _S52->c00_0 + r11_1 * _S52->c01_0 + r12_1 * _S52->c02_0;
    float m11_0 = r10_1 * _S52->c01_0 + r11_1 * _S52->c11_0 + r12_1 * _S52->c12_0;
    float m12_0 = r10_1 * _S52->c02_0 + r11_1 * _S52->c12_0 + r12_1 * _S52->c22_0;
    float m20_0 = r20_1 * _S52->c00_0 + r21_1 * _S52->c01_0 + r22_1 * _S52->c02_0;
    float m21_0 = r20_1 * _S52->c01_0 + r21_1 * _S52->c11_0 + r22_1 * _S52->c12_0;
    float m22_0 = r20_1 * _S52->c02_0 + r21_1 * _S52->c12_0 + r22_1 * _S52->c22_0;
    (&(&_S53)->cov_0)->c00_0 = m00_0 * r00_1 + m01_0 * r01_1 + m02_0 * r02_1;
    (&(&_S53)->cov_0)->c01_0 = m00_0 * r10_1 + m01_0 * r11_1 + m02_0 * r12_1;
    (&(&_S53)->cov_0)->c02_0 = m00_0 * r20_1 + m01_0 * r21_1 + m02_0 * r22_1;
    (&(&_S53)->cov_0)->c11_0 = m10_0 * r10_1 + m11_0 * r11_1 + m12_0 * r12_1;
    (&(&_S53)->cov_0)->c12_0 = m10_0 * r20_1 + m11_0 * r21_1 + m12_0 * r22_1;
    (&(&_S53)->cov_0)->c22_0 = m20_0 * r20_1 + m21_0 * r21_1 + m22_0 * r22_1;
    thread Tuple_5 _S54;
    (&_S54)->value0_8 = _S53;
    return _S54;
}

struct s_paramCtx_s_bwdCallableCtx_world_to_cam_0
{
    float value4_2;
    float value5_2;
    float value6_2;
    float value7_1;
    float value8_1;
    float value9_1;
    float value10_1;
    float value11_1;
    float value12_1;
};

struct SymMat2_0
{
    float c00_1;
    float c01_1;
    float c11_1;
};

struct ProjResult_0
{
    float px_1;
    float py_1;
    SymMat2_0 cov2d_0;
};

struct Tuple_6
{
    ProjResult_0 value0_9;
};

Tuple_6 s_apply_persp_proj_0(float _S55, float _S56, float _S57, const SymMat3_0 thread* _S58, float fx_0, float fy_0, float cx_0, float cy_0, float imageWidth_0, float imageHeight_0)
{
    float rz_0 = 1.0 / _S57;
    float rz2_0 = rz_0 * rz_0;
    float _S59 = 0.30000001192092896 * (0.5 * imageWidth_0 / fx_0);
    float _S60 = 0.30000001192092896 * (0.5 * imageHeight_0 / fy_0);
    float j00_0 = fx_0 * rz_0;
    float j02_0 = - fx_0 * (_S57 * s_apply_min_0((imageWidth_0 - cx_0) / fx_0 + _S59, s_apply_max_0(- (cx_0 / fx_0 + _S59), _S55 * rz_0).value0_2).value0_1) * rz2_0;
    float j11_0 = fy_0 * rz_0;
    float j12_0 = - fy_0 * (_S57 * s_apply_min_0((imageHeight_0 - cy_0) / fy_0 + _S60, s_apply_max_0(- (cy_0 / fy_0 + _S60), _S56 * rz_0).value0_2).value0_1) * rz2_0;
    float t02_0 = j00_0 * _S58->c02_0 + j02_0 * _S58->c22_0;
    SymMat2_0 _S61 = { (j00_0 * _S58->c00_0 + j02_0 * _S58->c02_0) * j00_0 + t02_0 * j02_0, (j00_0 * _S58->c01_0 + j02_0 * _S58->c12_0) * j11_0 + t02_0 * j12_0, (j11_0 * _S58->c11_0 + j12_0 * _S58->c12_0) * j11_0 + (j11_0 * _S58->c12_0 + j12_0 * _S58->c22_0) * j12_0 };
    float _S62 = fy_0 * _S56 * rz_0 + cy_0;
    thread Tuple_6 _S63;
    (&(&_S63)->value0_9)->px_1 = fx_0 * _S55 * rz_0 + cx_0;
    (&(&_S63)->value0_9)->py_1 = _S62;
    (&(&_S63)->value0_9)->cov2d_0 = _S61;
    return _S63;
}

struct s_paramCtx_s_bwdCallableCtx_persp_proj_0
{
    float value0_10;
    float value1_2;
    float value2_2;
    SymMat3_0 value3_2;
    float value4_3;
    float value5_3;
    float value6_3;
    float value7_2;
    float value8_2;
    float value9_2;
};

SymMat2_0 SymMat2_x24_syn_dzero_0()
{
    thread SymMat2_0 result_0;
    (&result_0)->c00_1 = 0.0;
    (&result_0)->c01_1 = 0.0;
    (&result_0)->c11_1 = 0.0;
    return result_0;
}

ProjResult_0 ProjResult_x24_syn_dzero_0()
{
    thread ProjResult_0 result_1;
    (&result_1)->px_1 = 0.0;
    (&result_1)->py_1 = 0.0;
    (&result_1)->cov2d_0 = SymMat2_x24_syn_dzero_0();
    return result_1;
}

SymMat2_0 SymMat2_x24_syn_dadd_0(const SymMat2_0 thread* SLANG_anonymous_0_0, const SymMat2_0 thread* SLANG_anonymous_1_0)
{
    thread SymMat2_0 result_2;
    (&result_2)->c00_1 = SLANG_anonymous_0_0->c00_1 + SLANG_anonymous_1_0->c00_1;
    (&result_2)->c01_1 = SLANG_anonymous_0_0->c01_1 + SLANG_anonymous_1_0->c01_1;
    (&result_2)->c11_1 = SLANG_anonymous_0_0->c11_1 + SLANG_anonymous_1_0->c11_1;
    return result_2;
}

ProjResult_0 ProjResult_x24_syn_dadd_0(const ProjResult_0 thread* SLANG_anonymous_0_1, const ProjResult_0 thread* SLANG_anonymous_1_1)
{
    thread ProjResult_0 result_3;
    (&result_3)->px_1 = SLANG_anonymous_0_1->px_1 + SLANG_anonymous_1_1->px_1;
    (&result_3)->py_1 = SLANG_anonymous_0_1->py_1 + SLANG_anonymous_1_1->py_1;
    SymMat2_0 _S64 = SymMat2_x24_syn_dadd_0(&SLANG_anonymous_0_1->cov2d_0, &SLANG_anonymous_1_1->cov2d_0);
    (&result_3)->cov2d_0 = _S64;
    return result_3;
}

struct s_bwdCallableCtx_persp_proj_0
{
    s_paramCtx_s_bwdCallableCtx_persp_proj_0 value0_11;
};

SymMat3_0 SymMat3_x24_syn_dzero_0()
{
    thread SymMat3_0 result_4;
    (&result_4)->c00_0 = 0.0;
    (&result_4)->c01_0 = 0.0;
    (&result_4)->c02_0 = 0.0;
    (&result_4)->c11_0 = 0.0;
    (&result_4)->c12_0 = 0.0;
    (&result_4)->c22_0 = 0.0;
    return result_4;
}

SymMat3_0 SymMat3_x24_syn_dadd_0(const SymMat3_0 thread* SLANG_anonymous_0_2, const SymMat3_0 thread* SLANG_anonymous_1_2)
{
    thread SymMat3_0 result_5;
    (&result_5)->c00_0 = SLANG_anonymous_0_2->c00_0 + SLANG_anonymous_1_2->c00_0;
    (&result_5)->c01_0 = SLANG_anonymous_0_2->c01_0 + SLANG_anonymous_1_2->c01_0;
    (&result_5)->c02_0 = SLANG_anonymous_0_2->c02_0 + SLANG_anonymous_1_2->c02_0;
    (&result_5)->c11_0 = SLANG_anonymous_0_2->c11_0 + SLANG_anonymous_1_2->c11_0;
    (&result_5)->c12_0 = SLANG_anonymous_0_2->c12_0 + SLANG_anonymous_1_2->c12_0;
    (&result_5)->c22_0 = SLANG_anonymous_0_2->c22_0 + SLANG_anonymous_1_2->c22_0;
    return result_5;
}

void s_bwdProp_persp_proj_0(const s_bwdCallableCtx_persp_proj_0 thread* _S65, float thread* _S66, float thread* _S67, float thread* _S68, SymMat3_0 thread* _S69, const ProjResult_0 thread* _S70)
{
    float rz_1 = 1.0 / (&_S65->value0_11)->value2_2;
    float _S71 = (&_S65->value0_11)->value2_2 * (&_S65->value0_11)->value2_2;
    float rz2_1 = rz_1 * rz_1;
    float _S72 = 0.30000001192092896 * (0.5 * (&_S65->value0_11)->value8_2 / (&_S65->value0_11)->value4_3);
    float lim_x_pos_0 = ((&_S65->value0_11)->value8_2 - (&_S65->value0_11)->value6_3) / (&_S65->value0_11)->value4_3 + _S72;
    float _S73 = 0.30000001192092896 * (0.5 * (&_S65->value0_11)->value9_2 / (&_S65->value0_11)->value5_3);
    float lim_y_pos_0 = ((&_S65->value0_11)->value9_2 - (&_S65->value0_11)->value7_2) / (&_S65->value0_11)->value5_3 + _S73;
    float lim_y_neg_0 = (&_S65->value0_11)->value7_2 / (&_S65->value0_11)->value5_3 + _S73;
    float _S74 = - ((&_S65->value0_11)->value6_3 / (&_S65->value0_11)->value4_3 + _S72);
    float _S75 = (&_S65->value0_11)->value0_10 * rz_1;
    Tuple_2 _S76 = s_apply_max_0(_S74, _S75);
    thread s_bwdCallableCtx_max_0 _S77;
    (&_S77)->_S14 = _S74;
    (&_S77)->_S15 = _S75;
    Tuple_1 _S78 = s_apply_min_0(lim_x_pos_0, _S76.value0_2);
    thread s_bwdCallableCtx_min_0 _S79;
    (&_S79)->_S6 = lim_x_pos_0;
    (&_S79)->_S7 = _S76.value0_2;
    float tx_1 = (&_S65->value0_11)->value2_2 * _S78.value0_1;
    float _S80 = - lim_y_neg_0;
    float _S81 = (&_S65->value0_11)->value1_2 * rz_1;
    Tuple_2 _S82 = s_apply_max_0(_S80, _S81);
    thread s_bwdCallableCtx_max_0 _S83;
    (&_S83)->_S14 = _S80;
    (&_S83)->_S15 = _S81;
    Tuple_1 _S84 = s_apply_min_0(lim_y_pos_0, _S82.value0_2);
    thread s_bwdCallableCtx_min_0 _S85;
    (&_S85)->_S6 = lim_y_pos_0;
    (&_S85)->_S7 = _S82.value0_2;
    float j00_1 = (&_S65->value0_11)->value4_3 * rz_1;
    float _S86 = - (&_S65->value0_11)->value4_3;
    float _S87 = _S86 * tx_1;
    float j02_1 = _S87 * rz2_1;
    float j11_1 = (&_S65->value0_11)->value5_3 * rz_1;
    float _S88 = - (&_S65->value0_11)->value5_3;
    float _S89 = _S88 * ((&_S65->value0_11)->value2_2 * _S84.value0_1);
    float j12_1 = _S89 * rz2_1;
    float t02_1 = j00_1 * (&(&_S65->value0_11)->value3_2)->c02_0 + j02_1 * (&(&_S65->value0_11)->value3_2)->c22_0;
    SymMat3_0 _S90 = SymMat3_x24_syn_dzero_0();
    float _S91 = (&_S65->value0_11)->value5_3 * (&_S65->value0_11)->value1_2 * _S70->py_1;
    float _S92 = (&_S65->value0_11)->value5_3 * (rz_1 * _S70->py_1);
    float _S93 = (&_S65->value0_11)->value4_3 * (&_S65->value0_11)->value0_10 * _S70->px_1;
    float _S94 = (&_S65->value0_11)->value4_3 * (rz_1 * _S70->px_1);
    float s_diff_t12_T_0 = j12_1 * (&_S70->cov2d_0)->c11_1;
    float s_diff_t11_T_0 = j11_1 * (&_S70->cov2d_0)->c11_1;
    float s_diff_t01_T_0 = j11_1 * (&_S70->cov2d_0)->c01_1;
    float s_diff_t00_T_0 = j00_1 * (&_S70->cov2d_0)->c00_1;
    float _S95 = j11_1 * s_diff_t11_T_0;
    float _S96 = j12_1 * (&_S70->cov2d_0)->c01_1 + j02_1 * (&_S70->cov2d_0)->c00_1;
    float _S97 = j00_1 * s_diff_t01_T_0;
    float _S98 = j00_1 * s_diff_t00_T_0;
    float _S99 = j12_1 * s_diff_t12_T_0 + j02_1 * _S96;
    float _S100 = j11_1 * s_diff_t12_T_0 + j12_1 * s_diff_t11_T_0 + j02_1 * s_diff_t01_T_0;
    float _S101 = j00_1 * _S96 + j02_1 * s_diff_t00_T_0;
    float _S102 = (j11_1 * (&(&_S65->value0_11)->value3_2)->c12_0 + j12_1 * (&(&_S65->value0_11)->value3_2)->c22_0) * (&_S70->cov2d_0)->c11_1 + t02_1 * (&_S70->cov2d_0)->c01_1 + (&(&_S65->value0_11)->value3_2)->c22_0 * s_diff_t12_T_0 + (&(&_S65->value0_11)->value3_2)->c12_0 * s_diff_t11_T_0;
    float _S103 = _S89 * _S102;
    float s_diff_ty_T_0 = _S88 * (rz2_1 * _S102);
    float _S104 = (&_S65->value0_11)->value5_3 * ((j11_1 * (&(&_S65->value0_11)->value3_2)->c11_0 + j12_1 * (&(&_S65->value0_11)->value3_2)->c12_0) * (&_S70->cov2d_0)->c11_1 + (j00_1 * (&(&_S65->value0_11)->value3_2)->c01_0 + j02_1 * (&(&_S65->value0_11)->value3_2)->c12_0) * (&_S70->cov2d_0)->c01_1 + (&(&_S65->value0_11)->value3_2)->c12_0 * s_diff_t12_T_0 + (&(&_S65->value0_11)->value3_2)->c11_0 * s_diff_t11_T_0);
    float _S105 = t02_1 * (&_S70->cov2d_0)->c00_1 + (&(&_S65->value0_11)->value3_2)->c22_0 * _S96 + (&(&_S65->value0_11)->value3_2)->c12_0 * s_diff_t01_T_0 + (&(&_S65->value0_11)->value3_2)->c02_0 * s_diff_t00_T_0;
    float _S106 = _S87 * _S105;
    float s_diff_tx_T_0 = _S86 * (rz2_1 * _S105);
    float _S107 = (&_S65->value0_11)->value4_3 * ((j00_1 * (&(&_S65->value0_11)->value3_2)->c00_0 + j02_1 * (&(&_S65->value0_11)->value3_2)->c02_0) * (&_S70->cov2d_0)->c00_1 + (&(&_S65->value0_11)->value3_2)->c02_0 * _S96 + (&(&_S65->value0_11)->value3_2)->c01_0 * s_diff_t01_T_0 + (&(&_S65->value0_11)->value3_2)->c00_0 * s_diff_t00_T_0);
    float _S108 = (&_S65->value0_11)->value2_2 * s_diff_ty_T_0;
    float _S109 = _S84.value0_1 * s_diff_ty_T_0;
    thread float _S110 = 0.0;
    thread float _S111 = 0.0;
    thread s_bwdCallableCtx_min_0 _S112 = _S85;
    s_bwdProp_min_0(&_S112, &_S110, &_S111, _S108);
    thread float _S113 = 0.0;
    thread float _S114 = 0.0;
    thread s_bwdCallableCtx_max_0 _S115 = _S83;
    s_bwdProp_max_0(&_S115, &_S113, &_S114, _S111);
    float _S116 = (&_S65->value0_11)->value1_2 * _S114;
    float _S117 = rz_1 * _S114;
    float _S118 = (&_S65->value0_11)->value2_2 * s_diff_tx_T_0;
    float _S119 = _S78.value0_1 * s_diff_tx_T_0;
    thread float _S120 = 0.0;
    thread float _S121 = 0.0;
    thread s_bwdCallableCtx_min_0 _S122 = _S79;
    s_bwdProp_min_0(&_S122, &_S120, &_S121, _S118);
    thread float _S123 = 0.0;
    thread float _S124 = 0.0;
    thread s_bwdCallableCtx_max_0 _S125 = _S77;
    s_bwdProp_max_0(&_S125, &_S123, &_S124, _S121);
    float _S126 = rz_1 * (_S103 + _S106);
    float _S127 = - ((_S91 + _S93 + _S104 + _S107 + _S116 + (&_S65->value0_11)->value0_10 * _S124 + _S126 + _S126) / _S71);
    float _S128 = _S92 + _S117;
    float _S129 = _S94 + rz_1 * _S124;
    thread SymMat3_0 _S130 = _S90;
    (&_S130)->c22_0 = _S99;
    (&_S130)->c12_0 = _S100;
    (&_S130)->c11_0 = _S95;
    (&_S130)->c02_0 = _S101;
    (&_S130)->c01_0 = _S97;
    (&_S130)->c00_0 = _S98;
    thread SymMat3_0 _S131 = _S90;
    thread SymMat3_0 _S132 = _S130;
    SymMat3_0 _S133 = SymMat3_x24_syn_dadd_0(&_S131, &_S132);
    float _S134 = _S109 + _S119 + _S127;
    *_S66 = _S129;
    *_S67 = _S128;
    *_S68 = _S134;
    *_S69 = _S133;
    return;
}

CamPose_0 CamPose_x24_syn_dzero_0()
{
    thread CamPose_0 result_6;
    (&result_6)->mx_0 = 0.0;
    (&result_6)->my_0 = 0.0;
    (&result_6)->mz_0 = 0.0;
    (&result_6)->cov_0 = SymMat3_x24_syn_dzero_0();
    return result_6;
}

struct s_bwdCallableCtx_world_to_cam_0
{
    s_paramCtx_s_bwdCallableCtx_world_to_cam_0 value0_12;
};

void s_bwdProp_world_to_cam_0(const s_bwdCallableCtx_world_to_cam_0 thread* _S135, float thread* _S136, float thread* _S137, float thread* _S138, SymMat3_0 thread* _S139, const CamPose_0 thread* _S140)
{
    SymMat3_0 _S141 = SymMat3_x24_syn_dzero_0();
    CamPose_0 _S142 = *_S140;
    (&(&_S142)->cov_0)->c22_0 = 0.0;
    float s_diff_m22_T_0 = (&_S135->value0_12)->value12_1 * (&_S140->cov_0)->c22_0;
    float s_diff_m21_T_0 = (&_S135->value0_12)->value11_1 * (&_S140->cov_0)->c22_0;
    float s_diff_m20_T_0 = (&_S135->value0_12)->value10_1 * (&_S140->cov_0)->c22_0;
    (&(&_S142)->cov_0)->c12_0 = 0.0;
    float _S143 = (&_S135->value0_12)->value12_1 * (&_S140->cov_0)->c12_0;
    float _S144 = (&_S135->value0_12)->value11_1 * (&_S140->cov_0)->c12_0;
    float _S145 = (&_S135->value0_12)->value10_1 * (&_S140->cov_0)->c12_0;
    (&(&_S142)->cov_0)->c11_0 = 0.0;
    float _S146 = (&_S135->value0_12)->value9_1 * (&_S140->cov_0)->c11_0;
    float _S147 = (&_S135->value0_12)->value8_1 * (&_S140->cov_0)->c11_0;
    float _S148 = (&_S135->value0_12)->value7_1 * (&_S140->cov_0)->c11_0;
    (&(&_S142)->cov_0)->c02_0 = 0.0;
    float _S149 = (&_S135->value0_12)->value12_1 * (&_S140->cov_0)->c02_0;
    float _S150 = (&_S135->value0_12)->value11_1 * (&_S140->cov_0)->c02_0;
    float _S151 = (&_S135->value0_12)->value10_1 * (&_S140->cov_0)->c02_0;
    (&(&_S142)->cov_0)->c01_0 = 0.0;
    float _S152 = (&_S135->value0_12)->value9_1 * (&_S140->cov_0)->c01_0;
    float _S153 = (&_S135->value0_12)->value8_1 * (&_S140->cov_0)->c01_0;
    float _S154 = (&_S135->value0_12)->value7_1 * (&_S140->cov_0)->c01_0;
    (&(&_S142)->cov_0)->c00_0 = 0.0;
    float _S155 = _S143 + _S146;
    float _S156 = _S144 + _S147;
    float _S157 = _S145 + _S148;
    float _S158 = _S149 + _S152 + (&_S135->value0_12)->value6_2 * (&_S140->cov_0)->c00_0;
    float _S159 = _S150 + _S153 + (&_S135->value0_12)->value5_2 * (&_S140->cov_0)->c00_0;
    float _S160 = _S151 + _S154 + (&_S135->value0_12)->value4_2 * (&_S140->cov_0)->c00_0;
    float _S161 = (&_S135->value0_12)->value12_1 * s_diff_m22_T_0 + (&_S135->value0_12)->value9_1 * _S155 + (&_S135->value0_12)->value6_2 * _S158;
    float _S162 = (&_S135->value0_12)->value11_1 * s_diff_m22_T_0 + (&_S135->value0_12)->value12_1 * s_diff_m21_T_0 + (&_S135->value0_12)->value8_1 * _S155 + (&_S135->value0_12)->value9_1 * _S156 + (&_S135->value0_12)->value5_2 * _S158 + (&_S135->value0_12)->value6_2 * _S159;
    float _S163 = (&_S135->value0_12)->value11_1 * s_diff_m21_T_0 + (&_S135->value0_12)->value8_1 * _S156 + (&_S135->value0_12)->value5_2 * _S159;
    float _S164 = (&_S135->value0_12)->value10_1 * s_diff_m22_T_0 + (&_S135->value0_12)->value12_1 * s_diff_m20_T_0 + (&_S135->value0_12)->value7_1 * _S155 + (&_S135->value0_12)->value9_1 * _S157 + (&_S135->value0_12)->value4_2 * _S158 + (&_S135->value0_12)->value6_2 * _S160;
    float _S165 = (&_S135->value0_12)->value10_1 * s_diff_m21_T_0 + (&_S135->value0_12)->value11_1 * s_diff_m20_T_0 + (&_S135->value0_12)->value7_1 * _S156 + (&_S135->value0_12)->value8_1 * _S157 + (&_S135->value0_12)->value4_2 * _S159 + (&_S135->value0_12)->value5_2 * _S160;
    float _S166 = (&_S135->value0_12)->value10_1 * s_diff_m20_T_0 + (&_S135->value0_12)->value7_1 * _S157 + (&_S135->value0_12)->value4_2 * _S160;
    float _S167 = (&_S135->value0_12)->value12_1 * _S142.mz_0;
    float _S168 = (&_S135->value0_12)->value11_1 * _S142.mz_0;
    float _S169 = (&_S135->value0_12)->value10_1 * _S142.mz_0;
    float _S170 = (&_S135->value0_12)->value9_1 * _S142.my_0;
    float _S171 = (&_S135->value0_12)->value8_1 * _S142.my_0;
    float _S172 = (&_S135->value0_12)->value7_1 * _S142.my_0;
    float _S173 = (&_S135->value0_12)->value6_2 * _S142.mx_0;
    float _S174 = (&_S135->value0_12)->value5_2 * _S142.mx_0;
    float _S175 = (&_S135->value0_12)->value4_2 * _S142.mx_0;
    thread SymMat3_0 _S176 = _S141;
    (&_S176)->c22_0 = _S161;
    (&_S176)->c12_0 = _S162;
    (&_S176)->c11_0 = _S163;
    (&_S176)->c02_0 = _S164;
    (&_S176)->c01_0 = _S165;
    (&_S176)->c00_0 = _S166;
    thread SymMat3_0 _S177 = _S141;
    thread SymMat3_0 _S178 = _S176;
    SymMat3_0 _S179 = SymMat3_x24_syn_dadd_0(&_S177, &_S178);
    float _S180 = _S167 + _S170 + _S173;
    float _S181 = _S168 + _S171 + _S174;
    *_S136 = _S169 + _S172 + _S175;
    *_S137 = _S181;
    *_S138 = _S180;
    *_S139 = _S179;
    return;
}

struct s_bwdCallableCtx_quat_scale_to_cov3d_0
{
    s_paramCtx_s_bwdCallableCtx_quat_scale_to_cov3d_0 value0_13;
};

struct s_paramCtx_s_bwdCallableCtx_quat_to_rotmat_0
{
    float value0_14;
    float value1_3;
    float value2_3;
    float value3_3;
};

Mat3_0 Mat3_x24_syn_dzero_0()
{
    thread Mat3_0 result_7;
    (&result_7)->r00_0 = 0.0;
    (&result_7)->r01_0 = 0.0;
    (&result_7)->r02_0 = 0.0;
    (&result_7)->r10_0 = 0.0;
    (&result_7)->r11_0 = 0.0;
    (&result_7)->r12_0 = 0.0;
    (&result_7)->r20_0 = 0.0;
    (&result_7)->r21_0 = 0.0;
    (&result_7)->r22_0 = 0.0;
    return result_7;
}

struct s_bwdCallableCtx_quat_to_rotmat_0
{
    s_paramCtx_s_bwdCallableCtx_quat_to_rotmat_0 value0_15;
};

void s_bwdProp_quat_to_rotmat_0(const s_bwdCallableCtx_quat_to_rotmat_0 thread* _S182, float thread* _S183, float thread* _S184, float thread* _S185, float thread* _S186, const Mat3_0 thread* _S187)
{
    float _S188 = (&_S182->value0_15)->value1_3 * (&_S182->value0_15)->value1_3 + (&_S182->value0_15)->value2_3 * (&_S182->value0_15)->value2_3 + (&_S182->value0_15)->value3_3 * (&_S182->value0_15)->value3_3 + (&_S182->value0_15)->value0_14 * (&_S182->value0_15)->value0_14;
    Tuple_0 _S189 = s_apply_rsqrt_0(_S188);
    thread s_bwdCallableCtx_rsqrt_0 _S190;
    (&_S190)->_S2 = _S188;
    float w_1 = (&_S182->value0_15)->value0_14 * _S189.value0_0;
    float x_4 = (&_S182->value0_15)->value1_3 * _S189.value0_0;
    float y_3 = (&_S182->value0_15)->value2_3 * _S189.value0_0;
    float z_1 = (&_S182->value0_15)->value3_3 * _S189.value0_0;
    float _S191 = 2.0 * - _S187->r22_0;
    float _S192 = 2.0 * _S187->r21_0;
    float _S193 = 2.0 * _S187->r20_0;
    float _S194 = 2.0 * _S187->r12_0;
    float _S195 = 2.0 * - _S187->r11_0;
    float _S196 = 2.0 * _S187->r10_0;
    float _S197 = 2.0 * _S187->r02_0;
    float _S198 = 2.0 * _S187->r01_0;
    float _S199 = 2.0 * - _S187->r00_0;
    float _S200 = _S196 + - _S198;
    float _S201 = - _S193 + _S197;
    float _S202 = _S192 + - _S194;
    float _S203 = _S192 + _S194;
    float _S204 = _S193 + _S197;
    float _S205 = _S196 + _S198;
    float _S206 = z_1 * (_S195 + _S199);
    float _S207 = y_3 * (_S191 + _S199);
    float _S208 = x_4 * (_S191 + _S195);
    float _S209 = w_1 * _S200 + y_3 * _S203 + x_4 * _S204 + _S206 + _S206;
    float _S210 = _S189.value0_0 * _S209;
    float _S211 = w_1 * _S201 + z_1 * _S203 + x_4 * _S205 + _S207 + _S207;
    float _S212 = _S189.value0_0 * _S211;
    float _S213 = w_1 * _S202 + z_1 * _S204 + y_3 * _S205 + _S208 + _S208;
    float _S214 = _S189.value0_0 * _S213;
    float _S215 = z_1 * _S200 + y_3 * _S201 + x_4 * _S202;
    float _S216 = _S189.value0_0 * _S215;
    float _S217 = (&_S182->value0_15)->value3_3 * _S209 + (&_S182->value0_15)->value2_3 * _S211 + (&_S182->value0_15)->value1_3 * _S213 + (&_S182->value0_15)->value0_14 * _S215;
    thread float _S218 = 0.0;
    thread s_bwdCallableCtx_rsqrt_0 _S219 = _S190;
    s_bwdProp_rsqrt_0(&_S219, &_S218, _S217);
    float _S220 = (&_S182->value0_15)->value0_14 * _S218;
    float _S221 = (&_S182->value0_15)->value3_3 * _S218;
    float _S222 = (&_S182->value0_15)->value2_3 * _S218;
    float _S223 = (&_S182->value0_15)->value1_3 * _S218;
    float _S224 = _S212 + _S222 + _S222;
    float _S225 = _S214 + _S223 + _S223;
    float _S226 = _S210 + _S221 + _S221;
    *_S183 = _S216 + _S220 + _S220;
    *_S184 = _S225;
    *_S185 = _S224;
    *_S186 = _S226;
    return;
}

void s_bwdProp_quat_scale_to_cov3d_0(const s_bwdCallableCtx_quat_scale_to_cov3d_0 thread* _S227, float thread* _S228, float thread* _S229, float thread* _S230, float thread* _S231, float thread* _S232, float thread* _S233, float thread* _S234, const SymMat3_0 thread* _S235)
{
    Tuple_4 _S236 = s_apply_quat_to_rotmat_0((&_S227->value0_13)->value0_7, (&_S227->value0_13)->value1_1, (&_S227->value0_13)->value2_1, (&_S227->value0_13)->value3_1);
    thread s_paramCtx_s_bwdCallableCtx_quat_to_rotmat_0 _S237;
    (&_S237)->value0_14 = (&_S227->value0_13)->value0_7;
    (&_S237)->value1_3 = (&_S227->value0_13)->value1_1;
    (&_S237)->value2_3 = (&_S227->value0_13)->value2_1;
    (&_S237)->value3_3 = (&_S227->value0_13)->value3_1;
    float M00_1 = _S236.value0_6.r00_0 * (&_S227->value0_13)->value4_1;
    float M01_1 = _S236.value0_6.r01_0 * (&_S227->value0_13)->value5_1;
    float M02_1 = _S236.value0_6.r02_0 * (&_S227->value0_13)->value6_1;
    float M10_1 = _S236.value0_6.r10_0 * (&_S227->value0_13)->value4_1;
    float M11_1 = _S236.value0_6.r11_0 * (&_S227->value0_13)->value5_1;
    float M12_1 = _S236.value0_6.r12_0 * (&_S227->value0_13)->value6_1;
    float M20_1 = _S236.value0_6.r20_0 * (&_S227->value0_13)->value4_1;
    float M21_1 = _S236.value0_6.r21_0 * (&_S227->value0_13)->value5_1;
    float M22_1 = _S236.value0_6.r22_0 * (&_S227->value0_13)->value6_1;
    float _S238 = M22_1 * _S235->c22_0;
    float _S239 = M21_1 * _S235->c22_0;
    float _S240 = M20_1 * _S235->c22_0;
    float _S241 = M12_1 * _S235->c11_0;
    float _S242 = M11_1 * _S235->c11_0;
    float _S243 = M10_1 * _S235->c11_0;
    float _S244 = M02_1 * _S235->c00_0;
    float _S245 = M01_1 * _S235->c00_0;
    float _S246 = M00_1 * _S235->c00_0;
    float _S247 = _S238 + _S238 + M12_1 * _S235->c12_0 + M02_1 * _S235->c02_0;
    float _S248 = _S236.value0_6.r22_0 * _S247;
    float _S249 = (&_S227->value0_13)->value6_1 * _S247;
    float _S250 = _S239 + _S239 + M11_1 * _S235->c12_0 + M01_1 * _S235->c02_0;
    float _S251 = _S236.value0_6.r21_0 * _S250;
    float _S252 = (&_S227->value0_13)->value5_1 * _S250;
    float _S253 = _S240 + _S240 + M10_1 * _S235->c12_0 + M00_1 * _S235->c02_0;
    float _S254 = _S236.value0_6.r20_0 * _S253;
    float _S255 = (&_S227->value0_13)->value4_1 * _S253;
    float _S256 = M22_1 * _S235->c12_0 + _S241 + _S241 + M02_1 * _S235->c01_0;
    float _S257 = _S236.value0_6.r12_0 * _S256;
    float _S258 = (&_S227->value0_13)->value6_1 * _S256;
    float _S259 = M21_1 * _S235->c12_0 + _S242 + _S242 + M01_1 * _S235->c01_0;
    float _S260 = _S236.value0_6.r11_0 * _S259;
    float _S261 = (&_S227->value0_13)->value5_1 * _S259;
    float _S262 = M20_1 * _S235->c12_0 + _S243 + _S243 + M00_1 * _S235->c01_0;
    float _S263 = _S236.value0_6.r10_0 * _S262;
    float _S264 = (&_S227->value0_13)->value4_1 * _S262;
    float _S265 = M22_1 * _S235->c02_0 + M12_1 * _S235->c01_0 + _S244 + _S244;
    float _S266 = _S236.value0_6.r02_0 * _S265;
    float _S267 = (&_S227->value0_13)->value6_1 * _S265;
    float _S268 = M21_1 * _S235->c02_0 + M11_1 * _S235->c01_0 + _S245 + _S245;
    float _S269 = _S236.value0_6.r01_0 * _S268;
    float _S270 = (&_S227->value0_13)->value5_1 * _S268;
    float _S271 = M20_1 * _S235->c02_0 + M10_1 * _S235->c01_0 + _S246 + _S246;
    float _S272 = _S236.value0_6.r00_0 * _S271;
    float _S273 = (&_S227->value0_13)->value4_1 * _S271;
    thread Mat3_0 _S274 = Mat3_x24_syn_dzero_0();
    (&_S274)->r22_0 = _S249;
    (&_S274)->r21_0 = _S252;
    (&_S274)->r20_0 = _S255;
    (&_S274)->r12_0 = _S258;
    (&_S274)->r11_0 = _S261;
    (&_S274)->r10_0 = _S264;
    (&_S274)->r02_0 = _S267;
    (&_S274)->r01_0 = _S270;
    (&_S274)->r00_0 = _S273;
    thread float _S275 = 0.0;
    thread float _S276 = 0.0;
    thread float _S277 = 0.0;
    thread float _S278 = 0.0;
    thread s_bwdCallableCtx_quat_to_rotmat_0 _S279;
    (&_S279)->value0_15 = _S237;
    thread Mat3_0 _S280 = _S274;
    s_bwdProp_quat_to_rotmat_0(&_S279, &_S275, &_S276, &_S277, &_S278, &_S280);
    float _S281 = _S254 + _S263 + _S272;
    float _S282 = _S251 + _S260 + _S269;
    float _S283 = _S248 + _S257 + _S266;
    *_S228 = _S275;
    *_S229 = _S276;
    *_S230 = _S277;
    *_S231 = _S278;
    *_S232 = _S281;
    *_S233 = _S282;
    *_S234 = _S283;
    return;
}

void s_bwdProp_project_to_screen_0(const s_bwdCallableCtx_project_to_screen_0 thread* _S284, float thread* _S285, float thread* _S286, float thread* _S287, float thread* _S288, float thread* _S289, float thread* _S290, float thread* _S291, float thread* _S292, float thread* _S293, float thread* _S294, const ProjectedScreen_0 thread* _S295)
{
    Tuple_3 _S296 = s_apply_quat_scale_to_cov3d_0((&_S284->value0_4)->value3_0, (&_S284->value0_4)->value4_0, (&_S284->value0_4)->value5_0, (&_S284->value0_4)->value6_0, (&_S284->value0_4)->value7_0, (&_S284->value0_4)->value8_0, (&_S284->value0_4)->value9_0);
    thread s_paramCtx_s_bwdCallableCtx_quat_scale_to_cov3d_0 _S297;
    (&_S297)->value0_7 = (&_S284->value0_4)->value3_0;
    (&_S297)->value1_1 = (&_S284->value0_4)->value4_0;
    (&_S297)->value2_1 = (&_S284->value0_4)->value5_0;
    (&_S297)->value3_1 = (&_S284->value0_4)->value6_0;
    (&_S297)->value4_1 = (&_S284->value0_4)->value7_0;
    (&_S297)->value5_1 = (&_S284->value0_4)->value8_0;
    (&_S297)->value6_1 = (&_S284->value0_4)->value9_0;
    s_paramCtx_s_bwdCallableCtx_quat_scale_to_cov3d_0 _S298 = _S297;
    thread SymMat3_0 _S299 = _S296.value0_5;
    Tuple_5 _S300 = s_apply_world_to_cam_0((&_S284->value0_4)->value0_3, (&_S284->value0_4)->value1_0, (&_S284->value0_4)->value2_0, &_S299, (&_S284->value0_4)->value10_0, (&_S284->value0_4)->value11_0, (&_S284->value0_4)->value12_0, (&_S284->value0_4)->value13_0, (&_S284->value0_4)->value14_0, (&_S284->value0_4)->value15_0, (&_S284->value0_4)->value16_0, (&_S284->value0_4)->value17_0, (&_S284->value0_4)->value18_0, (&_S284->value0_4)->value19_0, (&_S284->value0_4)->value20_0, (&_S284->value0_4)->value21_0);
    thread s_paramCtx_s_bwdCallableCtx_world_to_cam_0 _S301;
    (&_S301)->value4_2 = (&_S284->value0_4)->value10_0;
    (&_S301)->value5_2 = (&_S284->value0_4)->value11_0;
    (&_S301)->value6_2 = (&_S284->value0_4)->value12_0;
    (&_S301)->value7_1 = (&_S284->value0_4)->value13_0;
    (&_S301)->value8_1 = (&_S284->value0_4)->value14_0;
    (&_S301)->value9_1 = (&_S284->value0_4)->value15_0;
    (&_S301)->value10_1 = (&_S284->value0_4)->value16_0;
    (&_S301)->value11_1 = (&_S284->value0_4)->value17_0;
    (&_S301)->value12_1 = (&_S284->value0_4)->value18_0;
    s_paramCtx_s_bwdCallableCtx_world_to_cam_0 _S302 = _S301;
    Tuple_2 _S303 = s_apply_max_0(_S300.value0_8.mz_0, 0.00999999977648258);
    thread s_bwdCallableCtx_max_0 _S304;
    (&_S304)->_S14 = _S300.value0_8.mz_0;
    (&_S304)->_S15 = 0.00999999977648258;
    s_bwdCallableCtx_max_0 _S305 = _S304;
    thread SymMat3_0 _S306 = _S300.value0_8.cov_0;
    Tuple_6 _S307 = s_apply_persp_proj_0(_S300.value0_8.mx_0, _S300.value0_8.my_0, _S303.value0_2, &_S306, (&_S284->value0_4)->value22_0, (&_S284->value0_4)->value23_0, (&_S284->value0_4)->value24_0, (&_S284->value0_4)->value25_0, (&_S284->value0_4)->value26_0, (&_S284->value0_4)->value27_0);
    thread s_paramCtx_s_bwdCallableCtx_persp_proj_0 _S308;
    (&_S308)->value0_10 = _S300.value0_8.mx_0;
    (&_S308)->value1_2 = _S300.value0_8.my_0;
    (&_S308)->value2_2 = _S303.value0_2;
    (&_S308)->value3_2 = _S300.value0_8.cov_0;
    (&_S308)->value4_3 = (&_S284->value0_4)->value22_0;
    (&_S308)->value5_3 = (&_S284->value0_4)->value23_0;
    (&_S308)->value6_3 = (&_S284->value0_4)->value24_0;
    (&_S308)->value7_2 = (&_S284->value0_4)->value25_0;
    (&_S308)->value8_2 = (&_S284->value0_4)->value26_0;
    (&_S308)->value9_2 = (&_S284->value0_4)->value27_0;
    s_paramCtx_s_bwdCallableCtx_persp_proj_0 _S309 = _S308;
    float a_0 = _S307.value0_9.cov2d_0.c00_1 + 0.30000001192092896;
    float c_0 = _S307.value0_9.cov2d_0.c11_1 + 0.30000001192092896;
    float det_0 = a_0 * c_0 - _S307.value0_9.cov2d_0.c01_1 * _S307.value0_9.cov2d_0.c01_1;
    Tuple_2 _S310 = s_apply_max_0(det_0, 9.99999971718068537e-10);
    thread s_bwdCallableCtx_max_0 _S311;
    (&_S311)->_S14 = det_0;
    (&_S311)->_S15 = 9.99999971718068537e-10;
    s_bwdCallableCtx_max_0 _S312 = _S311;
    float inv_det_0 = 1.0 / _S310.value0_2;
    float _S313 = _S310.value0_2 * _S310.value0_2;
    bool culled_0;
    if((_S300.value0_8.mz_0) >= 0.00999999977648258)
    {
        culled_0 = (_S300.value0_8.mz_0) <= 1.0e+10;
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
    float _S314 = - _S307.value0_9.cov2d_0.c01_1;
    ProjResult_0 _S315 = ProjResult_x24_syn_dzero_0();
    float _S316 = _S295->px_0;
    float _S317 = _S295->py_0;
    float _S318 = _S295->conic_a_0;
    float _S319 = _S295->conic_b_0;
    float _S320 = _S295->conic_c_0;
    float _S321 = _S295->depth_0;
    float _S322 = _S295->cov2d_a_0;
    float _S323 = _S295->cov2d_b_0;
    float _S324 = _S295->cov2d_c_0;
    float _S325;
    if(culled_0)
    {
        _S325 = 0.0;
    }
    else
    {
        _S325 = _S324;
    }
    float _S326;
    if(culled_0)
    {
        _S326 = 0.0;
    }
    else
    {
        _S326 = _S323;
    }
    float _S327;
    if(culled_0)
    {
        _S327 = 0.0;
    }
    else
    {
        _S327 = _S322;
    }
    float _S328 = a_0 * _S320 + _S314 * _S319 + c_0 * _S318;
    float _S329 = inv_det_0 * _S320 + _S327;
    float _S330 = - (inv_det_0 * _S319) + _S326;
    float _S331 = inv_det_0 * _S318 + _S325;
    thread ProjResult_0 _S332 = _S315;
    (&_S332)->py_1 = _S317;
    (&_S332)->px_1 = _S316;
    thread ProjResult_0 _S333 = _S315;
    thread ProjResult_0 _S334 = _S332;
    ProjResult_0 _S335 = ProjResult_x24_syn_dadd_0(&_S333, &_S334);
    float _S336 = - (_S328 / _S313);
    thread float _S337 = 0.0;
    thread float _S338 = 0.0;
    thread s_bwdCallableCtx_max_0 _S339 = _S312;
    s_bwdProp_max_0(&_S339, &_S337, &_S338, _S336);
    float _S340 = _S307.value0_9.cov2d_0.c01_1 * - _S337;
    float _S341 = a_0 * _S337 + _S331;
    float _S342 = _S340 + _S340 + _S330;
    float _S343 = c_0 * _S337 + _S329;
    thread SymMat2_0 _S344 = SymMat2_x24_syn_dzero_0();
    (&_S344)->c11_1 = _S341;
    (&_S344)->c01_1 = _S342;
    (&_S344)->c00_1 = _S343;
    thread ProjResult_0 _S345 = _S315;
    (&_S345)->cov2d_0 = _S344;
    thread ProjResult_0 _S346 = _S335;
    thread ProjResult_0 _S347 = _S345;
    ProjResult_0 _S348 = ProjResult_x24_syn_dadd_0(&_S346, &_S347);
    thread float _S349 = 0.0;
    thread float _S350 = 0.0;
    thread float _S351 = 0.0;
    thread SymMat3_0 _S352;
    (&_S352)->c00_0 = 0.0;
    (&_S352)->c01_0 = 0.0;
    (&_S352)->c02_0 = 0.0;
    (&_S352)->c11_0 = 0.0;
    (&_S352)->c12_0 = 0.0;
    (&_S352)->c22_0 = 0.0;
    thread s_bwdCallableCtx_persp_proj_0 _S353;
    (&_S353)->value0_11 = _S309;
    thread ProjResult_0 _S354 = _S348;
    s_bwdProp_persp_proj_0(&_S353, &_S349, &_S350, &_S351, &_S352, &_S354);
    thread float _S355 = 0.0;
    thread float _S356 = 0.0;
    thread s_bwdCallableCtx_max_0 _S357 = _S305;
    s_bwdProp_max_0(&_S357, &_S355, &_S356, _S351);
    float _S358 = _S355 + _S321;
    thread CamPose_0 _S359 = CamPose_x24_syn_dzero_0();
    (&_S359)->cov_0 = _S352;
    (&_S359)->my_0 = _S350;
    (&_S359)->mx_0 = _S349;
    (&_S359)->mz_0 = _S358;
    thread float _S360 = 0.0;
    thread float _S361 = 0.0;
    thread float _S362 = 0.0;
    thread SymMat3_0 _S363;
    (&_S363)->c00_0 = 0.0;
    (&_S363)->c01_0 = 0.0;
    (&_S363)->c02_0 = 0.0;
    (&_S363)->c11_0 = 0.0;
    (&_S363)->c12_0 = 0.0;
    (&_S363)->c22_0 = 0.0;
    thread s_bwdCallableCtx_world_to_cam_0 _S364;
    (&_S364)->value0_12 = _S302;
    thread CamPose_0 _S365 = _S359;
    s_bwdProp_world_to_cam_0(&_S364, &_S360, &_S361, &_S362, &_S363, &_S365);
    thread float _S366 = 0.0;
    thread float _S367 = 0.0;
    thread float _S368 = 0.0;
    thread float _S369 = 0.0;
    thread float _S370 = 0.0;
    thread float _S371 = 0.0;
    thread float _S372 = 0.0;
    thread s_bwdCallableCtx_quat_scale_to_cov3d_0 _S373;
    (&_S373)->value0_13 = _S298;
    thread SymMat3_0 _S374 = _S363;
    s_bwdProp_quat_scale_to_cov3d_0(&_S373, &_S366, &_S367, &_S368, &_S369, &_S370, &_S371, &_S372, &_S374);
    *_S285 = _S360;
    *_S286 = _S361;
    *_S287 = _S362;
    *_S288 = _S366;
    *_S289 = _S367;
    *_S290 = _S368;
    *_S291 = _S369;
    *_S292 = _S370;
    *_S293 = _S371;
    *_S294 = _S372;
    return;
}

void _S375(DiffPair_float_0 thread* _S376, DiffPair_float_0 thread* _S377, DiffPair_float_0 thread* _S378, DiffPair_float_0 thread* _S379, DiffPair_float_0 thread* _S380, DiffPair_float_0 thread* _S381, DiffPair_float_0 thread* _S382, DiffPair_float_0 thread* _S383, DiffPair_float_0 thread* _S384, DiffPair_float_0 thread* _S385, float _S386, float _S387, float _S388, float _S389, float _S390, float _S391, float _S392, float _S393, float _S394, float _S395, float _S396, float _S397, float _S398, float _S399, float _S400, float _S401, float _S402, float _S403, const ProjectedScreen_0 thread* _S404)
{
    thread s_paramCtx_s_bwdCallableCtx_project_to_screen_0 _S405;
    (&_S405)->value0_3 = _S376->primal_0;
    (&_S405)->value1_0 = _S377->primal_0;
    (&_S405)->value2_0 = _S378->primal_0;
    (&_S405)->value3_0 = _S379->primal_0;
    (&_S405)->value4_0 = _S380->primal_0;
    (&_S405)->value5_0 = _S381->primal_0;
    (&_S405)->value6_0 = _S382->primal_0;
    (&_S405)->value7_0 = _S383->primal_0;
    (&_S405)->value8_0 = _S384->primal_0;
    (&_S405)->value9_0 = _S385->primal_0;
    (&_S405)->value10_0 = _S386;
    (&_S405)->value11_0 = _S387;
    (&_S405)->value12_0 = _S388;
    (&_S405)->value13_0 = _S389;
    (&_S405)->value14_0 = _S390;
    (&_S405)->value15_0 = _S391;
    (&_S405)->value16_0 = _S392;
    (&_S405)->value17_0 = _S393;
    (&_S405)->value18_0 = _S394;
    (&_S405)->value19_0 = _S395;
    (&_S405)->value20_0 = _S396;
    (&_S405)->value21_0 = _S397;
    (&_S405)->value22_0 = _S398;
    (&_S405)->value23_0 = _S399;
    (&_S405)->value24_0 = _S400;
    (&_S405)->value25_0 = _S401;
    (&_S405)->value26_0 = _S402;
    (&_S405)->value27_0 = _S403;
    thread s_bwdCallableCtx_project_to_screen_0 _S406;
    (&_S406)->value0_4 = _S405;
    s_bwdProp_project_to_screen_0(&_S406, &_S376->differential_0, &_S377->differential_0, &_S378->differential_0, &_S379->differential_0, &_S380->differential_0, &_S381->differential_0, &_S382->differential_0, &_S383->differential_0, &_S384->differential_0, &_S385->differential_0, _S404);
    return;
}

struct ProjectScreenGrad_0
{
    float wx_1;
    float wy_1;
    float wz_1;
    float qw_0;
    float qx_0;
    float qy_0;
    float qz_0;
    float sx_0;
    float sy_0;
    float sz_0;
};

ProjectScreenGrad_0 project_to_screen_grad_0(float wx_2, float wy_2, float wz_2, float qw_1, float qx_1, float qy_1, float qz_1, float sx_1, float sy_1, float sz_1, float r00_2, float r01_2, float r02_2, float r10_2, float r11_2, float r12_2, float r20_2, float r21_2, float r22_2, float tx_2, float ty_1, float tz_1, float fx_1, float fy_1, float cx_1, float cy_1, float imageWidth_1, float imageHeight_1, const ProjectedScreen_0 thread* cot_0)
{
    thread DiffPair_float_0 dpWx_0;
    (&dpWx_0)->primal_0 = wx_2;
    (&dpWx_0)->differential_0 = 0.0;
    thread DiffPair_float_0 dpWy_0;
    (&dpWy_0)->primal_0 = wy_2;
    (&dpWy_0)->differential_0 = 0.0;
    thread DiffPair_float_0 dpWz_0;
    (&dpWz_0)->primal_0 = wz_2;
    (&dpWz_0)->differential_0 = 0.0;
    thread DiffPair_float_0 dpQw_0;
    (&dpQw_0)->primal_0 = qw_1;
    (&dpQw_0)->differential_0 = 0.0;
    thread DiffPair_float_0 dpQx_0;
    (&dpQx_0)->primal_0 = qx_1;
    (&dpQx_0)->differential_0 = 0.0;
    thread DiffPair_float_0 dpQy_0;
    (&dpQy_0)->primal_0 = qy_1;
    (&dpQy_0)->differential_0 = 0.0;
    thread DiffPair_float_0 dpQz_0;
    (&dpQz_0)->primal_0 = qz_1;
    (&dpQz_0)->differential_0 = 0.0;
    thread DiffPair_float_0 dpSx_0;
    (&dpSx_0)->primal_0 = sx_1;
    (&dpSx_0)->differential_0 = 0.0;
    thread DiffPair_float_0 dpSy_0;
    (&dpSy_0)->primal_0 = sy_1;
    (&dpSy_0)->differential_0 = 0.0;
    thread DiffPair_float_0 dpSz_0;
    (&dpSz_0)->primal_0 = sz_1;
    (&dpSz_0)->differential_0 = 0.0;
    _S375(&dpWx_0, &dpWy_0, &dpWz_0, &dpQw_0, &dpQx_0, &dpQy_0, &dpQz_0, &dpSx_0, &dpSy_0, &dpSz_0, r00_2, r01_2, r02_2, r10_2, r11_2, r12_2, r20_2, r21_2, r22_2, tx_2, ty_1, tz_1, fx_1, fy_1, cx_1, cy_1, imageWidth_1, imageHeight_1, cot_0);
    thread ProjectScreenGrad_0 g_0;
    (&g_0)->wx_1 = dpWx_0.differential_0;
    (&g_0)->wy_1 = dpWy_0.differential_0;
    (&g_0)->wz_1 = dpWz_0.differential_0;
    (&g_0)->qw_0 = dpQw_0.differential_0;
    (&g_0)->qx_0 = dpQx_0.differential_0;
    (&g_0)->qy_0 = dpQy_0.differential_0;
    (&g_0)->qz_0 = dpQz_0.differential_0;
    (&g_0)->sx_0 = dpSx_0.differential_0;
    (&g_0)->sy_0 = dpSy_0.differential_0;
    (&g_0)->sz_0 = dpSz_0.differential_0;
    return g_0;
}

struct KernelContext_0
{
    int device* n_bwd_0;
    float device* cot_means2d_0;
    float device* cot_conics_0;
    float device* means3d_world_bwd_0;
    float device* quats_bwd_0;
    float device* scales_bwd_0;
    float device* viewmat_bwd_0;
    float device* intrinsics_bwd_0;
    float device* image_dims_bwd_0;
    float device* grad_means3d_world_0;
    float device* grad_quats_0;
    float device* grad_scales_0;
};

[[kernel]] void project_to_screen_bwd(uint3 tid_0 [[thread_position_in_grid]], int device* n_bwd_1 [[buffer(18)]], float device* cot_means2d_1 [[buffer(19)]], float device* cot_conics_1 [[buffer(20)]], float device* means3d_world_bwd_1 [[buffer(12)]], float device* quats_bwd_1 [[buffer(13)]], float device* scales_bwd_1 [[buffer(14)]], float device* viewmat_bwd_1 [[buffer(15)]], float device* intrinsics_bwd_1 [[buffer(16)]], float device* image_dims_bwd_1 [[buffer(17)]], float device* grad_means3d_world_1 [[buffer(21)]], float device* grad_quats_1 [[buffer(22)]], float device* grad_scales_1 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->n_bwd_0 = n_bwd_1;
    (&kernelContext_0)->cot_means2d_0 = cot_means2d_1;
    (&kernelContext_0)->cot_conics_0 = cot_conics_1;
    (&kernelContext_0)->means3d_world_bwd_0 = means3d_world_bwd_1;
    (&kernelContext_0)->quats_bwd_0 = quats_bwd_1;
    (&kernelContext_0)->scales_bwd_0 = scales_bwd_1;
    (&kernelContext_0)->viewmat_bwd_0 = viewmat_bwd_1;
    (&kernelContext_0)->intrinsics_bwd_0 = intrinsics_bwd_1;
    (&kernelContext_0)->image_dims_bwd_0 = image_dims_bwd_1;
    (&kernelContext_0)->grad_means3d_world_0 = grad_means3d_world_1;
    (&kernelContext_0)->grad_quats_0 = grad_quats_1;
    (&kernelContext_0)->grad_scales_0 = grad_scales_1;
    uint i_0 = tid_0.x;
    if(int(i_0) >= n_bwd_1[int(0)])
    {
        return;
    }
    uint wBase_0 = i_0 * 3U;
    uint qBase_0 = i_0 * 4U;
    uint mBase_0 = i_0 * 2U;
    thread ProjectedScreen_0 cot_1;
    (&cot_1)->px_0 = (&kernelContext_0)->cot_means2d_0[mBase_0];
    (&cot_1)->py_0 = (&kernelContext_0)->cot_means2d_0[mBase_0 + 1U];
    (&cot_1)->conic_a_0 = (&kernelContext_0)->cot_conics_0[wBase_0];
    uint _S407 = wBase_0 + 1U;
    (&cot_1)->conic_b_0 = (&kernelContext_0)->cot_conics_0[_S407];
    uint _S408 = wBase_0 + 2U;
    (&cot_1)->conic_c_0 = (&kernelContext_0)->cot_conics_0[_S408];
    (&cot_1)->depth_0 = 0.0;
    (&cot_1)->cov2d_a_0 = 0.0;
    (&cot_1)->cov2d_b_0 = 0.0;
    (&cot_1)->cov2d_c_0 = 0.0;
    float _S409 = (&kernelContext_0)->means3d_world_bwd_0[wBase_0];
    float _S410 = (&kernelContext_0)->means3d_world_bwd_0[_S407];
    float _S411 = (&kernelContext_0)->means3d_world_bwd_0[_S408];
    float _S412 = (&kernelContext_0)->quats_bwd_0[qBase_0];
    uint _S413 = qBase_0 + 1U;
    float _S414 = (&kernelContext_0)->quats_bwd_0[_S413];
    uint _S415 = qBase_0 + 2U;
    float _S416 = (&kernelContext_0)->quats_bwd_0[_S415];
    uint _S417 = qBase_0 + 3U;
    float _S418 = (&kernelContext_0)->quats_bwd_0[_S417];
    float _S419 = (&kernelContext_0)->scales_bwd_0[wBase_0];
    float _S420 = (&kernelContext_0)->scales_bwd_0[_S407];
    float _S421 = (&kernelContext_0)->scales_bwd_0[_S408];
    float _S422 = (&kernelContext_0)->viewmat_bwd_0[int(0)];
    float _S423 = (&kernelContext_0)->viewmat_bwd_0[int(1)];
    float _S424 = (&kernelContext_0)->viewmat_bwd_0[int(2)];
    float _S425 = (&kernelContext_0)->viewmat_bwd_0[int(4)];
    float _S426 = (&kernelContext_0)->viewmat_bwd_0[int(5)];
    float _S427 = (&kernelContext_0)->viewmat_bwd_0[int(6)];
    float _S428 = (&kernelContext_0)->viewmat_bwd_0[int(8)];
    float _S429 = (&kernelContext_0)->viewmat_bwd_0[int(9)];
    float _S430 = (&kernelContext_0)->viewmat_bwd_0[int(10)];
    float _S431 = (&kernelContext_0)->viewmat_bwd_0[int(3)];
    float _S432 = (&kernelContext_0)->viewmat_bwd_0[int(7)];
    float _S433 = (&kernelContext_0)->viewmat_bwd_0[int(11)];
    float _S434 = (&kernelContext_0)->intrinsics_bwd_0[int(0)];
    float _S435 = (&kernelContext_0)->intrinsics_bwd_0[int(1)];
    float _S436 = (&kernelContext_0)->intrinsics_bwd_0[int(2)];
    float _S437 = (&kernelContext_0)->intrinsics_bwd_0[int(3)];
    float _S438 = (&kernelContext_0)->image_dims_bwd_0[int(0)];
    float _S439 = (&kernelContext_0)->image_dims_bwd_0[int(1)];
    thread ProjectedScreen_0 _S440 = cot_1;
    ProjectScreenGrad_0 _S441 = project_to_screen_grad_0(_S409, _S410, _S411, _S412, _S414, _S416, _S418, _S419, _S420, _S421, _S422, _S423, _S424, _S425, _S426, _S427, _S428, _S429, _S430, _S431, _S432, _S433, _S434, _S435, _S436, _S437, _S438, _S439, &_S440);
    *((&kernelContext_0)->grad_means3d_world_0+wBase_0) = _S441.wx_1;
    *((&kernelContext_0)->grad_means3d_world_0+_S407) = _S441.wy_1;
    *((&kernelContext_0)->grad_means3d_world_0+_S408) = _S441.wz_1;
    *((&kernelContext_0)->grad_quats_0+qBase_0) = _S441.qw_0;
    *((&kernelContext_0)->grad_quats_0+_S413) = _S441.qx_0;
    *((&kernelContext_0)->grad_quats_0+_S415) = _S441.qy_0;
    *((&kernelContext_0)->grad_quats_0+_S417) = _S441.qz_0;
    *((&kernelContext_0)->grad_scales_0+wBase_0) = _S441.sx_0;
    *((&kernelContext_0)->grad_scales_0+_S407) = _S441.sy_0;
    *((&kernelContext_0)->grad_scales_0+_S408) = _S441.sz_0;
    return;
}

