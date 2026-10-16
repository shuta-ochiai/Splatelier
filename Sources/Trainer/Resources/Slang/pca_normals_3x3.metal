#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* pca_dims_0;
    float device* normals_0;
    int device* knIndices_0;
    float device* positions_0;
};

[[kernel]] void pca_normals_3x3(uint3 tid_0 [[thread_position_in_grid]], int device* pca_dims_1 [[buffer(2)]], float device* normals_1 [[buffer(3)]], int device* knIndices_1 [[buffer(1)]], float device* positions_1 [[buffer(0)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->pca_dims_0 = pca_dims_1;
    (&kernelContext_0)->normals_0 = normals_1;
    (&kernelContext_0)->knIndices_0 = knIndices_1;
    (&kernelContext_0)->positions_0 = positions_1;
    int i_0 = int(tid_0.x);
    int K_0 = pca_dims_1[int(1)];
    if(i_0 >= pca_dims_1[int(0)])
    {
        return;
    }
    int _S1 = i_0 * int(3);
    *((&kernelContext_0)->normals_0+_S1) = 0.0;
    int _S2 = _S1 + int(1);
    *((&kernelContext_0)->normals_0+_S2) = 0.0;
    int _S3 = _S1 + int(2);
    *((&kernelContext_0)->normals_0+_S3) = 1.0;
    if(K_0 < int(2))
    {
        return;
    }
    int k_0 = int(0);
    float cx_0 = 0.0;
    float cy_0 = 0.0;
    float cz_0 = 0.0;
    for(;;)
    {
        if(k_0 < K_0)
        {
        }
        else
        {
            break;
        }
        int _S4 = (&kernelContext_0)->knIndices_0[i_0 * K_0 + k_0] * int(3);
        float cx_1 = cx_0 + (&kernelContext_0)->positions_0[_S4];
        float cy_1 = cy_0 + (&kernelContext_0)->positions_0[_S4 + int(1)];
        float cz_1 = cz_0 + (&kernelContext_0)->positions_0[_S4 + int(2)];
        k_0 = k_0 + int(1);
        cx_0 = cx_1;
        cy_0 = cy_1;
        cz_0 = cz_1;
    }
    float invK_0 = 1.0 / float(K_0);
    float cx_2 = cx_0 * invK_0;
    float cy_2 = cy_0 * invK_0;
    float cz_2 = cz_0 * invK_0;
    k_0 = int(0);
    float a00_0 = 0.0;
    float a01_0 = 0.0;
    float a02_0 = 0.0;
    float a11_0 = 0.0;
    float a12_0 = 0.0;
    float a22_0 = 0.0;
    for(;;)
    {
        if(k_0 < K_0)
        {
        }
        else
        {
            break;
        }
        int _S5 = (&kernelContext_0)->knIndices_0[i_0 * K_0 + k_0] * int(3);
        float dx_0 = (&kernelContext_0)->positions_0[_S5] - cx_2;
        float dy_0 = (&kernelContext_0)->positions_0[_S5 + int(1)] - cy_2;
        float dz_0 = (&kernelContext_0)->positions_0[_S5 + int(2)] - cz_2;
        float a00_1 = a00_0 + dx_0 * dx_0;
        float a01_1 = a01_0 + dx_0 * dy_0;
        float a02_1 = a02_0 + dx_0 * dz_0;
        float a11_1 = a11_0 + dy_0 * dy_0;
        float a12_1 = a12_0 + dy_0 * dz_0;
        float a22_1 = a22_0 + dz_0 * dz_0;
        k_0 = k_0 + int(1);
        a00_0 = a00_1;
        a01_0 = a01_1;
        a02_0 = a02_1;
        a11_0 = a11_1;
        a12_0 = a12_1;
        a22_0 = a22_1;
    }
    float a00_2 = a00_0 * invK_0;
    float a01_2 = a01_0 * invK_0;
    float a02_2 = a02_0 * invK_0;
    float a11_2 = a11_0 * invK_0;
    float a12_2 = a12_0 * invK_0;
    float a22_2 = a22_0 * invK_0;
    float p1_0 = a01_2 * a01_2 + a02_2 * a02_2 + a12_2 * a12_2;
    float lambdaMin_0;
    if(p1_0 < 9.99999968265522539e-21)
    {
        lambdaMin_0 = min(a00_2, min(a11_2, a22_2));
    }
    else
    {
        float q_0 = (a00_2 + a11_2 + a22_2) / 3.0;
        float dq0_0 = a00_2 - q_0;
        float dq1_0 = a11_2 - q_0;
        float dq2_0 = a22_2 - q_0;
        float p_0 = sqrt((dq0_0 * dq0_0 + dq1_0 * dq1_0 + dq2_0 * dq2_0 + 2.0 * p1_0) / 6.0);
        float b11_0 = dq1_0 / p_0;
        float b22_0 = dq2_0 / p_0;
        float b01_0 = a01_2 / p_0;
        float b02_0 = a02_2 / p_0;
        float b12_0 = a12_2 / p_0;
        lambdaMin_0 = q_0 + 2.0 * p_0 * cos(acos(clamp((dq0_0 / p_0 * (b11_0 * b22_0 - b12_0 * b12_0) - b01_0 * (b01_0 * b22_0 - b12_0 * b02_0) + b02_0 * (b01_0 * b12_0 - b11_0 * b02_0)) * 0.5, -1.0, 1.0)) / 3.0 + 2.09439516067504883);
    }
    float3 r0_0 = float3(a00_2 - lambdaMin_0, a01_2, a02_2);
    float3 r1_0 = float3(a01_2, a11_2 - lambdaMin_0, a12_2);
    float3 r2_0 = float3(a02_2, a12_2, a22_2 - lambdaMin_0);
    float3 c01_0 = cross(r0_0, r1_0);
    float3 c02_0 = cross(r0_0, r2_0);
    float3 c12_0 = cross(r1_0, r2_0);
    float l01_0 = dot(c01_0, c01_0);
    float l02_0 = dot(c02_0, c02_0);
    float l12_0 = dot(c12_0, c12_0);
    bool _S6;
    if(l01_0 >= l02_0)
    {
        _S6 = l01_0 >= l12_0;
    }
    else
    {
        _S6 = false;
    }
    float3 n_0;
    if(_S6)
    {
        n_0 = c01_0;
    }
    else
    {
        if(l02_0 >= l12_0)
        {
            n_0 = c02_0;
        }
        else
        {
            n_0 = c12_0;
        }
    }
    float nLen2_0 = dot(n_0, n_0);
    if(nLen2_0 > 1.00000004581370497e-18)
    {
        float invN_0 = 1.0 / sqrt(nLen2_0);
        *((&kernelContext_0)->normals_0+_S1) = n_0.x * invN_0;
        *((&kernelContext_0)->normals_0+_S2) = n_0.y * invN_0;
        *((&kernelContext_0)->normals_0+_S3) = n_0.z * invN_0;
    }
    return;
}

