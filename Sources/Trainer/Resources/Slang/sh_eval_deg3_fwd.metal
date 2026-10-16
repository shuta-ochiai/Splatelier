#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
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

float3 sh_eval_deg3_0(int degree_0, float3 dir_0, const SHBands16_0 thread* sh_0)
{
    float invLen_0 = rsqrt(dot(dir_0, dir_0));
    float x_0 = dir_0.x * invLen_0;
    float y_0 = dir_0.y * invLen_0;
    float z_0 = dir_0.z * invLen_0;
    float3 result_0 = float3(0.282094806432724)  * sh_0->b0_0;
    float3 result_1;
    if(degree_0 >= int(1))
    {
        result_1 = result_0 + float3((-0.48860251903533936 * y_0))  * sh_0->b1_0 + float3((0.48860251903533936 * z_0))  * sh_0->b2_0 + float3((-0.48860251903533936 * x_0))  * sh_0->b3_0;
    }
    else
    {
        result_1 = result_0;
    }
    if(degree_0 >= int(2))
    {
        float xx_0 = x_0 * x_0;
        float yy_0 = y_0 * y_0;
        result_1 = result_1 + float3((1.09254848957061768 * (x_0 * y_0)))  * sh_0->b4_0 + float3((-1.09254848957061768 * (y_0 * z_0)))  * sh_0->b5_0 + float3((0.31539157032966614 * (2.0 * (z_0 * z_0) - xx_0 - yy_0)))  * sh_0->b6_0 + float3((-1.09254848957061768 * (x_0 * z_0)))  * sh_0->b7_0 + float3((0.54627424478530884 * (xx_0 - yy_0)))  * sh_0->b8_0;
    }
    if(degree_0 >= int(3))
    {
        float xx_1 = x_0 * x_0;
        float yy_1 = y_0 * y_0;
        float zz_0 = z_0 * z_0;
        float _S1 = 3.0 * xx_1;
        float _S2 = 4.0 * zz_0 - xx_1 - yy_1;
        float _S3 = 3.0 * yy_1;
        result_1 = result_1 + float3((-0.59004360437393188 * y_0 * (_S1 - yy_1)))  * sh_0->b9_0 + float3((2.89061141014099121 * (x_0 * y_0) * z_0))  * sh_0->b10_0 + float3((-0.4570457935333252 * y_0 * _S2))  * sh_0->b11_0 + float3((0.37317633628845215 * z_0 * (2.0 * zz_0 - _S1 - _S3)))  * sh_0->b12_0 + float3((-0.4570457935333252 * x_0 * _S2))  * sh_0->b13_0 + float3((1.44530570507049561 * z_0 * (xx_1 - yy_1)))  * sh_0->b14_0 + float3((-0.59004360437393188 * x_0 * (xx_1 - _S3)))  * sh_0->b15_0;
    }
    return max(result_1 + float3(0.5, 0.5, 0.5), float3(0.0, 0.0, 0.0));
}

struct KernelContext_0
{
    int device* n_buf_0;
    int device* radii_0;
    half device* colors_0;
    int device* degree_buf_0;
    float device* dirs_0;
    float device* sh0_0;
    float device* shN_0;
};

[[kernel]] void sh_eval_deg3_fwd(uint3 tid_0 [[thread_position_in_grid]], int device* n_buf_1 [[buffer(4)]], int device* radii_1 [[buffer(5)]], half device* colors_1 [[buffer(6)]], int device* degree_buf_1 [[buffer(3)]], float device* dirs_1 [[buffer(0)]], float device* sh0_1 [[buffer(1)]], float device* shN_1 [[buffer(2)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->n_buf_0 = n_buf_1;
    (&kernelContext_0)->radii_0 = radii_1;
    (&kernelContext_0)->colors_0 = colors_1;
    (&kernelContext_0)->degree_buf_0 = degree_buf_1;
    (&kernelContext_0)->dirs_0 = dirs_1;
    (&kernelContext_0)->sh0_0 = sh0_1;
    (&kernelContext_0)->shN_0 = shN_1;
    uint i_0 = tid_0.x;
    if(int(i_0) >= n_buf_1[int(0)])
    {
        return;
    }
    uint oBase_0 = i_0 * 3U;
    uint _S4 = i_0 * 2U;
    bool _S5;
    if((&kernelContext_0)->radii_0[_S4] <= int(0))
    {
        _S5 = true;
    }
    else
    {
        _S5 = (&kernelContext_0)->radii_0[_S4 + 1U] <= int(0);
    }
    if(_S5)
    {
        *((&kernelContext_0)->colors_0+oBase_0) = 0.0;
        *((&kernelContext_0)->colors_0+(oBase_0 + 1U)) = 0.0;
        *((&kernelContext_0)->colors_0+(oBase_0 + 2U)) = 0.0;
        return;
    }
    uint sN_0 = i_0 * 9U;
    int deg_0 = (&kernelContext_0)->degree_buf_0[int(0)];
    uint _S6 = oBase_0 + 1U;
    uint _S7 = oBase_0 + 2U;
    float3 dir_1 = float3((&kernelContext_0)->dirs_0[oBase_0], (&kernelContext_0)->dirs_0[_S6], (&kernelContext_0)->dirs_0[_S7]);
    thread SHBands16_0 sh_1;
    (&sh_1)->b0_0 = float3((&kernelContext_0)->sh0_0[oBase_0], (&kernelContext_0)->sh0_0[_S6], (&kernelContext_0)->sh0_0[_S7]);
    float3 _S8 = float3(0.0) ;
    (&sh_1)->b1_0 = _S8;
    (&sh_1)->b2_0 = _S8;
    (&sh_1)->b3_0 = _S8;
    (&sh_1)->b4_0 = _S8;
    (&sh_1)->b5_0 = _S8;
    (&sh_1)->b6_0 = _S8;
    (&sh_1)->b7_0 = _S8;
    (&sh_1)->b8_0 = _S8;
    (&sh_1)->b9_0 = _S8;
    (&sh_1)->b10_0 = _S8;
    (&sh_1)->b11_0 = _S8;
    (&sh_1)->b12_0 = _S8;
    (&sh_1)->b13_0 = _S8;
    (&sh_1)->b14_0 = _S8;
    (&sh_1)->b15_0 = _S8;
    if(deg_0 >= int(1))
    {
        (&sh_1)->b1_0 = float3((&kernelContext_0)->shN_0[sN_0], (&kernelContext_0)->shN_0[sN_0 + 1U], (&kernelContext_0)->shN_0[sN_0 + 2U]);
        (&sh_1)->b2_0 = float3((&kernelContext_0)->shN_0[sN_0 + 3U], (&kernelContext_0)->shN_0[sN_0 + 4U], (&kernelContext_0)->shN_0[sN_0 + 5U]);
        (&sh_1)->b3_0 = float3((&kernelContext_0)->shN_0[sN_0 + 6U], (&kernelContext_0)->shN_0[sN_0 + 7U], (&kernelContext_0)->shN_0[sN_0 + 8U]);
    }
    thread SHBands16_0 _S9 = sh_1;
    float3 _S10 = sh_eval_deg3_0(deg_0, dir_1, &_S9);
    *((&kernelContext_0)->colors_0+oBase_0) = half(_S10.x);
    *((&kernelContext_0)->colors_0+_S6) = half(_S10.y);
    *((&kernelContext_0)->colors_0+_S7) = half(_S10.z);
    return;
}

