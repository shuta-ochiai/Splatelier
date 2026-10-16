#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct s_bwdCallableCtx_max_0
{
    float3 _S1;
    float3 _S2;
};

struct DiffPair_vectorx3Cfloatx2C3x3E_0
{
    float3 primal_0;
    float3 differential_0;
};

struct DiffPair_float_0
{
    float primal_0;
    float differential_0;
};

void _d_max_0(DiffPair_float_0 thread* dpx_0, DiffPair_float_0 thread* dpy_0, float dOut_0)
{
    DiffPair_float_0 _S3 = *dpx_0;
    float _S4;
    if(((*dpx_0).primal_0) > ((*dpy_0).primal_0))
    {
        _S4 = dOut_0;
    }
    else
    {
        if(((*dpx_0).primal_0) < ((*dpy_0).primal_0))
        {
            _S4 = 0.0;
        }
        else
        {
            _S4 = 0.5 * dOut_0;
        }
    }
    dpx_0->primal_0 = _S3.primal_0;
    dpx_0->differential_0 = _S4;
    DiffPair_float_0 _S5 = *dpy_0;
    if(((*dpy_0).primal_0) > (_S3.primal_0))
    {
        _S4 = dOut_0;
    }
    else
    {
        if(((*dpy_0).primal_0) < ((*dpx_0).primal_0))
        {
            _S4 = 0.0;
        }
        else
        {
            _S4 = 0.5 * dOut_0;
        }
    }
    dpy_0->primal_0 = _S5.primal_0;
    dpy_0->differential_0 = _S4;
    return;
}

void _d_max_vector_0(DiffPair_vectorx3Cfloatx2C3x3E_0 thread* dpx_1, DiffPair_vectorx3Cfloatx2C3x3E_0 thread* dpy_1, float3 dOut_1)
{
    thread DiffPair_float_0 left_dp_0;
    (&left_dp_0)->primal_0 = (*dpx_1).primal_0[int(0)];
    (&left_dp_0)->differential_0 = 0.0;
    thread DiffPair_float_0 right_dp_0;
    (&right_dp_0)->primal_0 = (*dpy_1).primal_0[int(0)];
    (&right_dp_0)->differential_0 = 0.0;
    _d_max_0(&left_dp_0, &right_dp_0, dOut_1[int(0)]);
    thread float3 left_d_result_0;
    left_d_result_0.x = left_dp_0.differential_0;
    thread float3 right_d_result_0;
    right_d_result_0.x = right_dp_0.differential_0;
    thread DiffPair_float_0 left_dp_1;
    (&left_dp_1)->primal_0 = (*dpx_1).primal_0[int(1)];
    (&left_dp_1)->differential_0 = 0.0;
    thread DiffPair_float_0 right_dp_1;
    (&right_dp_1)->primal_0 = (*dpy_1).primal_0[int(1)];
    (&right_dp_1)->differential_0 = 0.0;
    _d_max_0(&left_dp_1, &right_dp_1, dOut_1[int(1)]);
    left_d_result_0.y = left_dp_1.differential_0;
    right_d_result_0.y = right_dp_1.differential_0;
    thread DiffPair_float_0 left_dp_2;
    (&left_dp_2)->primal_0 = (*dpx_1).primal_0[int(2)];
    (&left_dp_2)->differential_0 = 0.0;
    thread DiffPair_float_0 right_dp_2;
    (&right_dp_2)->primal_0 = (*dpy_1).primal_0[int(2)];
    (&right_dp_2)->differential_0 = 0.0;
    _d_max_0(&left_dp_2, &right_dp_2, dOut_1[int(2)]);
    left_d_result_0.z = left_dp_2.differential_0;
    right_d_result_0.z = right_dp_2.differential_0;
    dpx_1->primal_0 = (*dpx_1).primal_0;
    dpx_1->differential_0 = left_d_result_0;
    dpy_1->primal_0 = (*dpy_1).primal_0;
    dpy_1->differential_0 = right_d_result_0;
    return;
}

void s_bwdProp_max_0(const s_bwdCallableCtx_max_0 thread* ctx_0, float3 thread* d_x_0, float3 thread* d_y_0, float3 d_out_0)
{
    thread DiffPair_vectorx3Cfloatx2C3x3E_0 _S6;
    (&_S6)->primal_0 = ctx_0->_S1;
    thread DiffPair_vectorx3Cfloatx2C3x3E_0 _S7;
    (&_S7)->primal_0 = ctx_0->_S2;
    _d_max_vector_0(&_S6, &_S7, d_out_0);
    *d_x_0 = (&_S6)->differential_0;
    *d_y_0 = (&_S7)->differential_0;
    return;
}

struct Tuple_0
{
    float value0_0;
};

Tuple_0 s_apply_rsqrt_0(float x_0)
{
    Tuple_0 _S8 = { rsqrt(x_0) };
    return _S8;
}

struct s_bwdCallableCtx_rsqrt_0
{
    float _S9;
};

void _d_rsqrt_0(DiffPair_float_0 thread* dpx_2, float dOut_2)
{
    float _S10 = -0.5 / ((*dpx_2).primal_0 * sqrt((*dpx_2).primal_0)) * dOut_2;
    dpx_2->primal_0 = (*dpx_2).primal_0;
    dpx_2->differential_0 = _S10;
    return;
}

void s_bwdProp_rsqrt_0(const s_bwdCallableCtx_rsqrt_0 thread* ctx_1, float thread* d_x_1, float d_out_1)
{
    thread DiffPair_float_0 _S11;
    (&_S11)->primal_0 = ctx_1->_S9;
    _d_rsqrt_0(&_S11, d_out_1);
    *d_x_1 = (&_S11)->differential_0;
    return;
}

struct Tuple_1
{
    float value0_1;
};

Tuple_1 s_apply_dot_0(float3 x_1, float3 y_0)
{
    Tuple_1 _S12 = { dot(x_1, y_0) };
    return _S12;
}

struct s_bwdCallableCtx_dot_0
{
    float3 _S13;
    float3 _S14;
};

void _d_dot_0(DiffPair_vectorx3Cfloatx2C3x3E_0 thread* dpx_3, DiffPair_vectorx3Cfloatx2C3x3E_0 thread* dpy_2, float dOut_3)
{
    thread float3 x_d_result_0;
    x_d_result_0.x = (*dpy_2).primal_0[int(0)] * dOut_3;
    thread float3 y_d_result_0;
    y_d_result_0.x = (*dpx_3).primal_0[int(0)] * dOut_3;
    x_d_result_0.y = (*dpy_2).primal_0[int(1)] * dOut_3;
    y_d_result_0.y = (*dpx_3).primal_0[int(1)] * dOut_3;
    x_d_result_0.z = (*dpy_2).primal_0[int(2)] * dOut_3;
    y_d_result_0.z = (*dpx_3).primal_0[int(2)] * dOut_3;
    dpx_3->primal_0 = (*dpx_3).primal_0;
    dpx_3->differential_0 = x_d_result_0;
    dpy_2->primal_0 = (*dpy_2).primal_0;
    dpy_2->differential_0 = y_d_result_0;
    return;
}

void s_bwdProp_dot_0(const s_bwdCallableCtx_dot_0 thread* ctx_2, float3 thread* d_x_2, float3 thread* d_y_1, float d_out_2)
{
    thread DiffPair_vectorx3Cfloatx2C3x3E_0 _S15;
    (&_S15)->primal_0 = ctx_2->_S13;
    thread DiffPair_vectorx3Cfloatx2C3x3E_0 _S16;
    (&_S16)->primal_0 = ctx_2->_S14;
    _d_dot_0(&_S15, &_S16, d_out_2);
    *d_x_2 = (&_S15)->differential_0;
    *d_y_1 = (&_S16)->differential_0;
    return;
}

struct SHBands16_0
{
    float3 b0_0;
    float3 b1_0;
    float3 b2_0;
    float3 b3_0;
    float3 b4_0;
    float3 b5_0;
    float3 b6_0;
    float3 b7_0;
    float3 b8_0;
    float3 b9_0;
    float3 b10_0;
    float3 b11_0;
    float3 b12_0;
    float3 b13_0;
    float3 b14_0;
    float3 b15_0;
};

struct DiffPair_SHBands16_0
{
    SHBands16_0 primal_0;
    SHBands16_0 differential_0;
};

struct s_paramCtx_s_bwdCallableCtx_sh_eval_deg3_0
{
    int value0_2;
    float3 value1_0;
    SHBands16_0 value2_0;
};

struct s_bwdCallableCtx_sh_eval_deg3_0
{
    s_paramCtx_s_bwdCallableCtx_sh_eval_deg3_0 value0_3;
};

SHBands16_0 SHBands16_x24_syn_dzero_0()
{
    thread SHBands16_0 result_0;
    float3 _S17 = float3(0.0) ;
    (&result_0)->b0_0 = _S17;
    (&result_0)->b1_0 = _S17;
    (&result_0)->b2_0 = _S17;
    (&result_0)->b3_0 = _S17;
    (&result_0)->b4_0 = _S17;
    (&result_0)->b5_0 = _S17;
    (&result_0)->b6_0 = _S17;
    (&result_0)->b7_0 = _S17;
    (&result_0)->b8_0 = _S17;
    (&result_0)->b9_0 = _S17;
    (&result_0)->b10_0 = _S17;
    (&result_0)->b11_0 = _S17;
    (&result_0)->b12_0 = _S17;
    (&result_0)->b13_0 = _S17;
    (&result_0)->b14_0 = _S17;
    (&result_0)->b15_0 = _S17;
    return result_0;
}

SHBands16_0 SHBands16_x24_syn_dadd_0(const SHBands16_0 thread* SLANG_anonymous_0_0, const SHBands16_0 thread* SLANG_anonymous_1_0)
{
    thread SHBands16_0 result_1;
    (&result_1)->b0_0 = SLANG_anonymous_0_0->b0_0 + SLANG_anonymous_1_0->b0_0;
    (&result_1)->b1_0 = SLANG_anonymous_0_0->b1_0 + SLANG_anonymous_1_0->b1_0;
    (&result_1)->b2_0 = SLANG_anonymous_0_0->b2_0 + SLANG_anonymous_1_0->b2_0;
    (&result_1)->b3_0 = SLANG_anonymous_0_0->b3_0 + SLANG_anonymous_1_0->b3_0;
    (&result_1)->b4_0 = SLANG_anonymous_0_0->b4_0 + SLANG_anonymous_1_0->b4_0;
    (&result_1)->b5_0 = SLANG_anonymous_0_0->b5_0 + SLANG_anonymous_1_0->b5_0;
    (&result_1)->b6_0 = SLANG_anonymous_0_0->b6_0 + SLANG_anonymous_1_0->b6_0;
    (&result_1)->b7_0 = SLANG_anonymous_0_0->b7_0 + SLANG_anonymous_1_0->b7_0;
    (&result_1)->b8_0 = SLANG_anonymous_0_0->b8_0 + SLANG_anonymous_1_0->b8_0;
    (&result_1)->b9_0 = SLANG_anonymous_0_0->b9_0 + SLANG_anonymous_1_0->b9_0;
    (&result_1)->b10_0 = SLANG_anonymous_0_0->b10_0 + SLANG_anonymous_1_0->b10_0;
    (&result_1)->b11_0 = SLANG_anonymous_0_0->b11_0 + SLANG_anonymous_1_0->b11_0;
    (&result_1)->b12_0 = SLANG_anonymous_0_0->b12_0 + SLANG_anonymous_1_0->b12_0;
    (&result_1)->b13_0 = SLANG_anonymous_0_0->b13_0 + SLANG_anonymous_1_0->b13_0;
    (&result_1)->b14_0 = SLANG_anonymous_0_0->b14_0 + SLANG_anonymous_1_0->b14_0;
    (&result_1)->b15_0 = SLANG_anonymous_0_0->b15_0 + SLANG_anonymous_1_0->b15_0;
    return result_1;
}

void s_bwdProp_sh_eval_deg3_0(const s_bwdCallableCtx_sh_eval_deg3_0 thread* _S18, float3 thread* _S19, SHBands16_0 thread* _S20, float3 _S21)
{
    float3 _S22 = float3(0.0) ;
    Tuple_1 _S23 = s_apply_dot_0((&_S18->value0_3)->value1_0, (&_S18->value0_3)->value1_0);
    thread s_bwdCallableCtx_dot_0 _S24;
    (&_S24)->_S13 = (&_S18->value0_3)->value1_0;
    (&_S24)->_S14 = (&_S18->value0_3)->value1_0;
    s_bwdCallableCtx_dot_0 _S25 = _S24;
    Tuple_0 _S26 = s_apply_rsqrt_0(_S23.value0_1);
    thread s_bwdCallableCtx_rsqrt_0 _S27;
    (&_S27)->_S9 = _S23.value0_1;
    s_bwdCallableCtx_rsqrt_0 _S28 = _S27;
    float _S29 = (&_S18->value0_3)->value1_0.x;
    float x_2 = _S29 * _S26.value0_0;
    float _S30 = (&_S18->value0_3)->value1_0.y;
    float y_1 = _S30 * _S26.value0_0;
    float _S31 = (&_S18->value0_3)->value1_0.z;
    float z_0 = _S31 * _S26.value0_0;
    float3 result_2 = float3(0.282094806432724)  * (&(&_S18->value0_3)->value2_0)->b0_0;
    int _S32 = (&_S18->value0_3)->value0_2;
    bool _S33 = ((&_S18->value0_3)->value0_2) >= int(1);
    float3 result_3;
    float3 _S34;
    float3 _S35;
    float3 _S36;
    float3 _S37;
    float3 _S38;
    float3 _S39;
    if(_S33)
    {
        float3 _S40 = float3((-0.48860251903533936 * y_1)) ;
        float3 _S41 = float3((0.48860251903533936 * z_0)) ;
        float3 _S42 = float3((-0.48860251903533936 * x_2)) ;
        result_3 = result_2 + _S40 * (&(&_S18->value0_3)->value2_0)->b1_0 + _S41 * (&(&_S18->value0_3)->value2_0)->b2_0 + _S42 * (&(&_S18->value0_3)->value2_0)->b3_0;
        _S34 = _S42;
        _S35 = (&(&_S18->value0_3)->value2_0)->b3_0;
        _S36 = _S41;
        _S37 = (&(&_S18->value0_3)->value2_0)->b2_0;
        _S38 = _S40;
        _S39 = (&(&_S18->value0_3)->value2_0)->b1_0;
    }
    else
    {
        result_3 = result_2;
        _S34 = _S22;
        _S35 = _S22;
        _S36 = _S22;
        _S37 = _S22;
        _S38 = _S22;
        _S39 = _S22;
    }
    bool _S43 = _S32 >= int(2);
    float3 _S44;
    float3 _S45;
    float3 _S46;
    float3 _S47;
    float3 _S48;
    float3 _S49;
    float3 _S50;
    float3 _S51;
    float3 _S52;
    float3 _S53;
    if(_S43)
    {
        float xx_0 = x_2 * x_2;
        float yy_0 = y_1 * y_1;
        float3 _S54 = float3((1.09254848957061768 * (x_2 * y_1))) ;
        float3 _S55 = float3((-1.09254848957061768 * (y_1 * z_0))) ;
        float3 _S56 = float3((0.31539157032966614 * (2.0 * (z_0 * z_0) - xx_0 - yy_0))) ;
        float3 _S57 = float3((-1.09254848957061768 * (x_2 * z_0))) ;
        float3 _S58 = float3((0.54627424478530884 * (xx_0 - yy_0))) ;
        result_3 = result_3 + _S54 * (&(&_S18->value0_3)->value2_0)->b4_0 + _S55 * (&(&_S18->value0_3)->value2_0)->b5_0 + _S56 * (&(&_S18->value0_3)->value2_0)->b6_0 + _S57 * (&(&_S18->value0_3)->value2_0)->b7_0 + _S58 * (&(&_S18->value0_3)->value2_0)->b8_0;
        _S44 = _S58;
        _S45 = (&(&_S18->value0_3)->value2_0)->b8_0;
        _S46 = _S57;
        _S47 = (&(&_S18->value0_3)->value2_0)->b7_0;
        _S48 = _S56;
        _S49 = (&(&_S18->value0_3)->value2_0)->b6_0;
        _S50 = _S55;
        _S51 = (&(&_S18->value0_3)->value2_0)->b5_0;
        _S52 = _S54;
        _S53 = (&(&_S18->value0_3)->value2_0)->b4_0;
    }
    else
    {
        _S44 = _S22;
        _S45 = _S22;
        _S46 = _S22;
        _S47 = _S22;
        _S48 = _S22;
        _S49 = _S22;
        _S50 = _S22;
        _S51 = _S22;
        _S52 = _S22;
        _S53 = _S22;
    }
    bool _S59 = _S32 >= int(3);
    float3 _S60;
    float3 _S61;
    float3 _S62;
    float3 _S63;
    float3 _S64;
    float3 _S65;
    float3 _S66;
    float3 _S67;
    float3 _S68;
    float3 _S69;
    float3 _S70;
    float3 _S71;
    float3 _S72;
    float3 _S73;
    float _S74;
    float _S75;
    float _S76;
    float _S77;
    float _S78;
    float _S79;
    float _S80;
    float _S81;
    float _S82;
    float _S83;
    float _S84;
    float _S85;
    if(_S59)
    {
        float xx_1 = x_2 * x_2;
        float yy_1 = y_1 * y_1;
        float zz_0 = z_0 * z_0;
        float _S86 = -0.59004360437393188 * y_1;
        float _S87 = 3.0 * xx_1;
        float _S88 = _S87 - yy_1;
        float3 _S89 = float3((_S86 * _S88)) ;
        float _S90 = 2.89061141014099121 * (x_2 * y_1);
        float3 _S91 = float3((_S90 * z_0)) ;
        float _S92 = -0.4570457935333252 * y_1;
        float _S93 = 4.0 * zz_0 - xx_1 - yy_1;
        float3 _S94 = float3((_S92 * _S93)) ;
        float _S95 = 0.37317633628845215 * z_0;
        float _S96 = 3.0 * yy_1;
        float _S97 = 2.0 * zz_0 - _S87 - _S96;
        float3 _S98 = float3((_S95 * _S97)) ;
        float _S99 = -0.4570457935333252 * x_2;
        float3 _S100 = float3((_S99 * _S93)) ;
        float _S101 = 1.44530570507049561 * z_0;
        float _S102 = xx_1 - yy_1;
        float3 _S103 = float3((_S101 * _S102)) ;
        float _S104 = -0.59004360437393188 * x_2;
        float _S105 = xx_1 - _S96;
        float3 _S106 = float3((_S104 * _S105)) ;
        result_3 = result_3 + _S89 * (&(&_S18->value0_3)->value2_0)->b9_0 + _S91 * (&(&_S18->value0_3)->value2_0)->b10_0 + _S94 * (&(&_S18->value0_3)->value2_0)->b11_0 + _S98 * (&(&_S18->value0_3)->value2_0)->b12_0 + _S100 * (&(&_S18->value0_3)->value2_0)->b13_0 + _S103 * (&(&_S18->value0_3)->value2_0)->b14_0 + _S106 * (&(&_S18->value0_3)->value2_0)->b15_0;
        _S60 = _S106;
        _S61 = (&(&_S18->value0_3)->value2_0)->b15_0;
        _S74 = _S104;
        _S75 = _S105;
        _S62 = _S103;
        _S63 = (&(&_S18->value0_3)->value2_0)->b14_0;
        _S76 = _S101;
        _S77 = _S102;
        _S64 = _S100;
        _S65 = (&(&_S18->value0_3)->value2_0)->b13_0;
        _S78 = _S99;
        _S79 = _S93;
        _S66 = _S98;
        _S67 = (&(&_S18->value0_3)->value2_0)->b12_0;
        _S80 = _S95;
        _S81 = _S97;
        _S68 = _S94;
        _S69 = (&(&_S18->value0_3)->value2_0)->b11_0;
        _S82 = _S92;
        _S70 = _S91;
        _S71 = (&(&_S18->value0_3)->value2_0)->b10_0;
        _S83 = _S90;
        _S72 = _S89;
        _S73 = (&(&_S18->value0_3)->value2_0)->b9_0;
        _S84 = _S86;
        _S85 = _S88;
    }
    else
    {
        _S60 = _S22;
        _S61 = _S22;
        _S74 = 0.0;
        _S75 = 0.0;
        _S62 = _S22;
        _S63 = _S22;
        _S76 = 0.0;
        _S77 = 0.0;
        _S64 = _S22;
        _S65 = _S22;
        _S78 = 0.0;
        _S79 = 0.0;
        _S66 = _S22;
        _S67 = _S22;
        _S80 = 0.0;
        _S81 = 0.0;
        _S68 = _S22;
        _S69 = _S22;
        _S82 = 0.0;
        _S70 = _S22;
        _S71 = _S22;
        _S83 = 0.0;
        _S72 = _S22;
        _S73 = _S22;
        _S84 = 0.0;
        _S85 = 0.0;
    }
    float3 _S107 = float3(0.0, 0.0, 0.0);
    thread s_bwdCallableCtx_max_0 _S108;
    (&_S108)->_S1 = result_3 + float3(0.5, 0.5, 0.5);
    (&_S108)->_S2 = _S107;
    s_bwdCallableCtx_max_0 _S109 = _S108;
    SHBands16_0 _S110 = SHBands16_x24_syn_dzero_0();
    thread float3 _S111 = _S22;
    thread float3 _S112 = _S22;
    thread s_bwdCallableCtx_max_0 _S113 = _S109;
    s_bwdProp_max_0(&_S113, &_S111, &_S112, _S21);
    float3 s_diff_result_T_0 = _S111;
    SHBands16_0 _S114;
    if(_S59)
    {
        float3 _S115 = _S60 * s_diff_result_T_0;
        float3 _S116 = _S61 * s_diff_result_T_0;
        float _S117 = _S116[int(0)] + _S116[int(1)] + _S116[int(2)];
        float _S118 = _S74 * _S117;
        float3 _S119 = _S62 * s_diff_result_T_0;
        float3 _S120 = _S63 * s_diff_result_T_0;
        float _S121 = _S120[int(0)] + _S120[int(1)] + _S120[int(2)];
        float _S122 = _S76 * _S121;
        float _S123 = 1.44530570507049561 * (_S77 * _S121);
        float3 _S124 = _S64 * s_diff_result_T_0;
        float3 _S125 = _S65 * s_diff_result_T_0;
        float _S126 = _S125[int(0)] + _S125[int(1)] + _S125[int(2)];
        float3 _S127 = _S66 * s_diff_result_T_0;
        float3 _S128 = _S67 * s_diff_result_T_0;
        float _S129 = _S128[int(0)] + _S128[int(1)] + _S128[int(2)];
        float _S130 = _S80 * _S129;
        float _S131 = - _S130;
        float _S132 = 0.37317633628845215 * (_S81 * _S129);
        float3 _S133 = _S68 * s_diff_result_T_0;
        float3 _S134 = _S69 * s_diff_result_T_0;
        float _S135 = _S134[int(0)] + _S134[int(1)] + _S134[int(2)];
        float _S136 = _S78 * _S126 + _S82 * _S135;
        float _S137 = - _S136;
        float _S138 = -0.4570457935333252 * (_S79 * _S135);
        float3 _S139 = _S70 * s_diff_result_T_0;
        float3 _S140 = _S71 * s_diff_result_T_0;
        float _S141 = _S140[int(0)] + _S140[int(1)] + _S140[int(2)];
        float _S142 = _S83 * _S141;
        float s_diff_xy_T_0 = 2.89061141014099121 * (z_0 * _S141);
        float3 _S143 = _S72 * s_diff_result_T_0;
        float3 _S144 = _S73 * s_diff_result_T_0;
        float _S145 = _S144[int(0)] + _S144[int(1)] + _S144[int(2)];
        float _S146 = _S84 * _S145;
        float _S147 = -0.59004360437393188 * (_S85 * _S145);
        float _S148 = x_2 * s_diff_xy_T_0;
        float _S149 = z_0 * (2.0 * _S130 + 4.0 * _S136);
        float _S150 = y_1 * (- _S122 + 3.0 * (- _S118 + _S131) + _S137 + - _S146);
        float _S151 = x_2 * (_S118 + _S122 + _S137 + 3.0 * (_S131 + _S146));
        float _S152 = -0.59004360437393188 * (_S75 * _S117) + -0.4570457935333252 * (_S79 * _S126) + y_1 * s_diff_xy_T_0 + _S151 + _S151;
        thread SHBands16_0 _S153 = _S110;
        (&_S153)->b15_0 = _S115;
        (&_S153)->b14_0 = _S119;
        (&_S153)->b13_0 = _S124;
        (&_S153)->b12_0 = _S127;
        (&_S153)->b11_0 = _S133;
        (&_S153)->b10_0 = _S139;
        (&_S153)->b9_0 = _S143;
        thread SHBands16_0 _S154 = _S110;
        thread SHBands16_0 _S155 = _S153;
        SHBands16_0 _S156 = SHBands16_x24_syn_dadd_0(&_S154, &_S155);
        float _S157 = _S138 + _S147 + _S148 + _S150 + _S150;
        float _S158 = _S123 + _S132 + _S142 + _S149 + _S149;
        _S114 = _S156;
        _S74 = _S152;
        _S75 = _S157;
        _S76 = _S158;
    }
    else
    {
        _S114 = _S110;
        _S74 = 0.0;
        _S75 = 0.0;
        _S76 = 0.0;
    }
    if(_S43)
    {
        float3 _S159 = _S44 * s_diff_result_T_0;
        float3 _S160 = _S45 * s_diff_result_T_0;
        float _S161 = 0.54627424478530884 * (_S160[int(0)] + _S160[int(1)] + _S160[int(2)]);
        float3 _S162 = _S46 * s_diff_result_T_0;
        float3 _S163 = _S47 * s_diff_result_T_0;
        float s_diff_xz_T_0 = -1.09254848957061768 * (_S163[int(0)] + _S163[int(1)] + _S163[int(2)]);
        float3 _S164 = _S48 * s_diff_result_T_0;
        float3 _S165 = _S49 * s_diff_result_T_0;
        float _S166 = 0.31539157032966614 * (_S165[int(0)] + _S165[int(1)] + _S165[int(2)]);
        float _S167 = - _S166;
        float3 _S168 = _S50 * s_diff_result_T_0;
        float3 _S169 = _S51 * s_diff_result_T_0;
        float s_diff_yz_T_0 = -1.09254848957061768 * (_S169[int(0)] + _S169[int(1)] + _S169[int(2)]);
        float3 _S170 = _S52 * s_diff_result_T_0;
        float3 _S171 = _S53 * s_diff_result_T_0;
        float s_diff_xy_T_1 = 1.09254848957061768 * (_S171[int(0)] + _S171[int(1)] + _S171[int(2)]);
        float _S172 = x_2 * s_diff_xz_T_0;
        float _S173 = z_0 * s_diff_xz_T_0;
        float _S174 = y_1 * s_diff_yz_T_0;
        float _S175 = z_0 * s_diff_yz_T_0;
        float _S176 = x_2 * s_diff_xy_T_1;
        float _S177 = y_1 * s_diff_xy_T_1;
        float _S178 = z_0 * (2.0 * _S166);
        float _S179 = y_1 * (- _S161 + _S167);
        float _S180 = x_2 * (_S161 + _S167);
        thread SHBands16_0 _S181 = _S110;
        (&_S181)->b8_0 = _S159;
        (&_S181)->b7_0 = _S162;
        (&_S181)->b6_0 = _S164;
        (&_S181)->b5_0 = _S168;
        (&_S181)->b4_0 = _S170;
        thread SHBands16_0 _S182 = _S114;
        thread SHBands16_0 _S183 = _S181;
        SHBands16_0 _S184 = SHBands16_x24_syn_dadd_0(&_S182, &_S183);
        float _S185 = _S175 + _S176 + _S179 + _S179 + _S75;
        float _S186 = _S172 + _S174 + _S178 + _S178 + _S76;
        _S74 = _S173 + _S177 + _S180 + _S180 + _S74;
        _S114 = _S184;
        _S75 = _S186;
        _S76 = _S185;
    }
    else
    {
        float _S187 = _S75;
        _S75 = _S76;
        _S76 = _S187;
    }
    if(_S33)
    {
        float3 _S188 = _S34 * s_diff_result_T_0;
        float3 _S189 = _S35 * s_diff_result_T_0;
        float3 _S190 = _S36 * s_diff_result_T_0;
        float3 _S191 = _S37 * s_diff_result_T_0;
        float _S192 = 0.48860251903533936 * (_S191[int(0)] + _S191[int(1)] + _S191[int(2)]);
        float3 _S193 = _S38 * s_diff_result_T_0;
        float3 _S194 = _S39 * s_diff_result_T_0;
        float _S195 = -0.48860251903533936 * (_S194[int(0)] + _S194[int(1)] + _S194[int(2)]);
        float _S196 = -0.48860251903533936 * (_S189[int(0)] + _S189[int(1)] + _S189[int(2)]) + _S74;
        thread SHBands16_0 _S197 = _S110;
        (&_S197)->b3_0 = _S188;
        (&_S197)->b2_0 = _S190;
        (&_S197)->b1_0 = _S193;
        thread SHBands16_0 _S198 = _S114;
        thread SHBands16_0 _S199 = _S197;
        SHBands16_0 _S200 = SHBands16_x24_syn_dadd_0(&_S198, &_S199);
        float _S201 = _S195 + _S76;
        _S74 = _S192 + _S75;
        _S75 = _S201;
        _S76 = _S196;
        _S114 = _S200;
    }
    else
    {
        float _S202 = _S74;
        _S74 = _S75;
        _S75 = _S76;
        _S76 = _S202;
    }
    float3 _S203 = float3(0.282094806432724)  * s_diff_result_T_0;
    float _S204 = _S26.value0_0 * _S74;
    float _S205 = _S26.value0_0 * _S75;
    float _S206 = _S26.value0_0 * _S76;
    float _S207 = _S31 * _S74 + _S30 * _S75 + _S29 * _S76;
    thread float _S208 = 0.0;
    thread s_bwdCallableCtx_rsqrt_0 _S209 = _S28;
    s_bwdProp_rsqrt_0(&_S209, &_S208, _S207);
    thread float3 _S210 = _S22;
    thread float3 _S211 = _S22;
    thread s_bwdCallableCtx_dot_0 _S212 = _S25;
    s_bwdProp_dot_0(&_S212, &_S210, &_S211, _S208);
    thread SHBands16_0 _S213 = _S110;
    (&_S213)->b0_0 = _S203;
    thread SHBands16_0 _S214 = _S114;
    thread SHBands16_0 _S215 = _S213;
    SHBands16_0 _S216 = SHBands16_x24_syn_dadd_0(&_S214, &_S215);
    *_S19 = _S210 + _S211 + float3(_S206, _S205, _S204);
    *_S20 = _S216;
    return;
}

void _S217(int _S218, DiffPair_vectorx3Cfloatx2C3x3E_0 thread* _S219, DiffPair_SHBands16_0 thread* _S220, float3 _S221)
{
    thread s_paramCtx_s_bwdCallableCtx_sh_eval_deg3_0 _S222;
    (&_S222)->value0_2 = _S218;
    (&_S222)->value1_0 = _S219->primal_0;
    (&_S222)->value2_0 = _S220->primal_0;
    thread s_bwdCallableCtx_sh_eval_deg3_0 _S223;
    (&_S223)->value0_3 = _S222;
    s_bwdProp_sh_eval_deg3_0(&_S223, &_S219->differential_0, &_S220->differential_0, _S221);
    return;
}

struct ShEvalGrad_0
{
    float3 dir_0;
    SHBands16_0 sh_0;
};

ShEvalGrad_0 sh_eval_deg3_grad_0(int degree_0, float3 dir_1, const SHBands16_0 thread* sh_1, float3 cot_0)
{
    float3 _S224 = float3(0.0, 0.0, 0.0);
    thread DiffPair_vectorx3Cfloatx2C3x3E_0 dpDir_0;
    (&dpDir_0)->primal_0 = dir_1;
    (&dpDir_0)->differential_0 = _S224;
    thread SHBands16_0 zero_0;
    float3 _S225 = float3(0.0) ;
    (&zero_0)->b0_0 = _S225;
    (&zero_0)->b1_0 = _S225;
    (&zero_0)->b2_0 = _S225;
    (&zero_0)->b3_0 = _S225;
    (&zero_0)->b4_0 = _S225;
    (&zero_0)->b5_0 = _S225;
    (&zero_0)->b6_0 = _S225;
    (&zero_0)->b7_0 = _S225;
    (&zero_0)->b8_0 = _S225;
    (&zero_0)->b9_0 = _S225;
    (&zero_0)->b10_0 = _S225;
    (&zero_0)->b11_0 = _S225;
    (&zero_0)->b12_0 = _S225;
    (&zero_0)->b13_0 = _S225;
    (&zero_0)->b14_0 = _S225;
    (&zero_0)->b15_0 = _S225;
    thread DiffPair_SHBands16_0 dpSh_0;
    (&dpSh_0)->primal_0 = *sh_1;
    (&dpSh_0)->differential_0 = zero_0;
    _S217(degree_0, &dpDir_0, &dpSh_0, cot_0);
    thread ShEvalGrad_0 g_0;
    (&g_0)->dir_0 = dpDir_0.differential_0;
    (&g_0)->sh_0 = (&dpSh_0)->differential_0;
    return g_0;
}

struct KernelContext_0
{
    int device* n_bwd_0;
    int device* radii_bwd_0;
    float device* grad_dirs_0;
    float device* grad_sh0_0;
    float device* grad_shN_0;
    int device* degree_bwd_0;
    float device* dirs_bwd_0;
    float device* sh0_bwd_0;
    float device* shN_bwd_0;
    float device* cot_color_0;
};

[[kernel]] void sh_eval_deg3_bwd(uint3 tid_0 [[thread_position_in_grid]], int device* n_bwd_1 [[buffer(11)]], int device* radii_bwd_1 [[buffer(13)]], float device* grad_dirs_1 [[buffer(14)]], float device* grad_sh0_1 [[buffer(15)]], float device* grad_shN_1 [[buffer(16)]], int device* degree_bwd_1 [[buffer(10)]], float device* dirs_bwd_1 [[buffer(7)]], float device* sh0_bwd_1 [[buffer(8)]], float device* shN_bwd_1 [[buffer(9)]], float device* cot_color_1 [[buffer(12)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->n_bwd_0 = n_bwd_1;
    (&kernelContext_0)->radii_bwd_0 = radii_bwd_1;
    (&kernelContext_0)->grad_dirs_0 = grad_dirs_1;
    (&kernelContext_0)->grad_sh0_0 = grad_sh0_1;
    (&kernelContext_0)->grad_shN_0 = grad_shN_1;
    (&kernelContext_0)->degree_bwd_0 = degree_bwd_1;
    (&kernelContext_0)->dirs_bwd_0 = dirs_bwd_1;
    (&kernelContext_0)->sh0_bwd_0 = sh0_bwd_1;
    (&kernelContext_0)->shN_bwd_0 = shN_bwd_1;
    (&kernelContext_0)->cot_color_0 = cot_color_1;
    uint i_0 = tid_0.x;
    if(int(i_0) >= n_bwd_1[int(0)])
    {
        return;
    }
    uint dBase_0 = i_0 * 3U;
    uint sN_0 = i_0 * 9U;
    uint _S226 = i_0 * 2U;
    bool _S227;
    if((&kernelContext_0)->radii_bwd_0[_S226] <= int(0))
    {
        _S227 = true;
    }
    else
    {
        _S227 = (&kernelContext_0)->radii_bwd_0[_S226 + 1U] <= int(0);
    }
    if(_S227)
    {
        *((&kernelContext_0)->grad_dirs_0+dBase_0) = 0.0;
        uint _S228 = dBase_0 + 1U;
        *((&kernelContext_0)->grad_dirs_0+_S228) = 0.0;
        uint _S229 = dBase_0 + 2U;
        *((&kernelContext_0)->grad_dirs_0+_S229) = 0.0;
        *((&kernelContext_0)->grad_sh0_0+dBase_0) = 0.0;
        *((&kernelContext_0)->grad_sh0_0+_S228) = 0.0;
        *((&kernelContext_0)->grad_sh0_0+_S229) = 0.0;
        uint k_0 = 0U;
        for(;;)
        {
            if(k_0 < 9U)
            {
            }
            else
            {
                break;
            }
            *((&kernelContext_0)->grad_shN_0+(sN_0 + k_0)) = 0.0;
            k_0 = k_0 + 1U;
        }
        return;
    }
    int deg_0 = (&kernelContext_0)->degree_bwd_0[int(0)];
    uint _S230 = dBase_0 + 1U;
    uint _S231 = dBase_0 + 2U;
    float3 dir_2 = float3((&kernelContext_0)->dirs_bwd_0[dBase_0], (&kernelContext_0)->dirs_bwd_0[_S230], (&kernelContext_0)->dirs_bwd_0[_S231]);
    thread SHBands16_0 sh_2;
    (&sh_2)->b0_0 = float3((&kernelContext_0)->sh0_bwd_0[dBase_0], (&kernelContext_0)->sh0_bwd_0[_S230], (&kernelContext_0)->sh0_bwd_0[_S231]);
    float3 _S232 = float3(0.0) ;
    (&sh_2)->b1_0 = _S232;
    (&sh_2)->b2_0 = _S232;
    (&sh_2)->b3_0 = _S232;
    (&sh_2)->b4_0 = _S232;
    (&sh_2)->b5_0 = _S232;
    (&sh_2)->b6_0 = _S232;
    (&sh_2)->b7_0 = _S232;
    (&sh_2)->b8_0 = _S232;
    (&sh_2)->b9_0 = _S232;
    (&sh_2)->b10_0 = _S232;
    (&sh_2)->b11_0 = _S232;
    (&sh_2)->b12_0 = _S232;
    (&sh_2)->b13_0 = _S232;
    (&sh_2)->b14_0 = _S232;
    (&sh_2)->b15_0 = _S232;
    if(deg_0 >= int(1))
    {
        (&sh_2)->b1_0 = float3((&kernelContext_0)->shN_bwd_0[sN_0], (&kernelContext_0)->shN_bwd_0[sN_0 + 1U], (&kernelContext_0)->shN_bwd_0[sN_0 + 2U]);
        (&sh_2)->b2_0 = float3((&kernelContext_0)->shN_bwd_0[sN_0 + 3U], (&kernelContext_0)->shN_bwd_0[sN_0 + 4U], (&kernelContext_0)->shN_bwd_0[sN_0 + 5U]);
        (&sh_2)->b3_0 = float3((&kernelContext_0)->shN_bwd_0[sN_0 + 6U], (&kernelContext_0)->shN_bwd_0[sN_0 + 7U], (&kernelContext_0)->shN_bwd_0[sN_0 + 8U]);
    }
    float3 cot_1 = float3((&kernelContext_0)->cot_color_0[dBase_0], (&kernelContext_0)->cot_color_0[_S230], (&kernelContext_0)->cot_color_0[_S231]);
    thread SHBands16_0 _S233 = sh_2;
    ShEvalGrad_0 _S234 = sh_eval_deg3_grad_0(deg_0, dir_2, &_S233, cot_1);
    *((&kernelContext_0)->grad_dirs_0+dBase_0) = _S234.dir_0.x;
    *((&kernelContext_0)->grad_dirs_0+_S230) = _S234.dir_0.y;
    *((&kernelContext_0)->grad_dirs_0+_S231) = _S234.dir_0.z;
    *((&kernelContext_0)->grad_sh0_0+dBase_0) = _S234.sh_0.b0_0.x;
    *((&kernelContext_0)->grad_sh0_0+_S230) = _S234.sh_0.b0_0.y;
    *((&kernelContext_0)->grad_sh0_0+_S231) = _S234.sh_0.b0_0.z;
    *((&kernelContext_0)->grad_shN_0+sN_0) = _S234.sh_0.b1_0.x;
    *((&kernelContext_0)->grad_shN_0+(sN_0 + 1U)) = _S234.sh_0.b1_0.y;
    *((&kernelContext_0)->grad_shN_0+(sN_0 + 2U)) = _S234.sh_0.b1_0.z;
    *((&kernelContext_0)->grad_shN_0+(sN_0 + 3U)) = _S234.sh_0.b2_0.x;
    *((&kernelContext_0)->grad_shN_0+(sN_0 + 4U)) = _S234.sh_0.b2_0.y;
    *((&kernelContext_0)->grad_shN_0+(sN_0 + 5U)) = _S234.sh_0.b2_0.z;
    *((&kernelContext_0)->grad_shN_0+(sN_0 + 6U)) = _S234.sh_0.b3_0.x;
    *((&kernelContext_0)->grad_shN_0+(sN_0 + 7U)) = _S234.sh_0.b3_0.y;
    *((&kernelContext_0)->grad_shN_0+(sN_0 + 8U)) = _S234.sh_0.b3_0.z;
    return;
}

