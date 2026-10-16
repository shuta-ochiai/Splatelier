#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* hpp_dims_0;
    float device* Hpp_in_0;
    float device* Hpp_inv_0;
};

[[kernel]] void ba_hpp_invert(uint3 tid_0 [[thread_position_in_grid]], int device* hpp_dims_1 [[buffer(1)]], float device* Hpp_in_1 [[buffer(0)]], float device* Hpp_inv_1 [[buffer(2)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->hpp_dims_0 = hpp_dims_1;
    (&kernelContext_0)->Hpp_in_0 = Hpp_in_1;
    (&kernelContext_0)->Hpp_inv_0 = Hpp_inv_1;
    int j_0 = int(tid_0.x);
    if(j_0 >= hpp_dims_1[int(0)])
    {
        return;
    }
    int base_0 = j_0 * int(9);
    float a00_0 = (&kernelContext_0)->Hpp_in_0[base_0];
    int _S1 = base_0 + int(1);
    float a01_0 = (&kernelContext_0)->Hpp_in_0[_S1];
    int _S2 = base_0 + int(2);
    float a02_0 = (&kernelContext_0)->Hpp_in_0[_S2];
    int _S3 = base_0 + int(3);
    float a10_0 = (&kernelContext_0)->Hpp_in_0[_S3];
    int _S4 = base_0 + int(4);
    float a11_0 = (&kernelContext_0)->Hpp_in_0[_S4];
    int _S5 = base_0 + int(5);
    float a12_0 = (&kernelContext_0)->Hpp_in_0[_S5];
    int _S6 = base_0 + int(6);
    float a20_0 = (&kernelContext_0)->Hpp_in_0[_S6];
    int _S7 = base_0 + int(7);
    float a21_0 = (&kernelContext_0)->Hpp_in_0[_S7];
    int _S8 = base_0 + int(8);
    float a22_0 = (&kernelContext_0)->Hpp_in_0[_S8];
    float c00_0 = a11_0 * a22_0 - a12_0 * a21_0;
    float c01_0 = - (a10_0 * a22_0 - a12_0 * a20_0);
    float c02_0 = a10_0 * a21_0 - a11_0 * a20_0;
    float det_0 = a00_0 * c00_0 + a01_0 * c01_0 + a02_0 * c02_0;
    if((abs(det_0)) < 9.999999960041972e-13)
    {
        *((&kernelContext_0)->Hpp_inv_0+base_0) = 1.0;
        *((&kernelContext_0)->Hpp_inv_0+_S1) = 0.0;
        *((&kernelContext_0)->Hpp_inv_0+_S2) = 0.0;
        *((&kernelContext_0)->Hpp_inv_0+_S3) = 0.0;
        *((&kernelContext_0)->Hpp_inv_0+_S4) = 1.0;
        *((&kernelContext_0)->Hpp_inv_0+_S5) = 0.0;
        *((&kernelContext_0)->Hpp_inv_0+_S6) = 0.0;
        *((&kernelContext_0)->Hpp_inv_0+_S7) = 0.0;
        *((&kernelContext_0)->Hpp_inv_0+_S8) = 1.0;
        return;
    }
    float invDet_0 = 1.0 / det_0;
    *((&kernelContext_0)->Hpp_inv_0+base_0) = c00_0 * invDet_0;
    *((&kernelContext_0)->Hpp_inv_0+_S1) = - (a01_0 * a22_0 - a02_0 * a21_0) * invDet_0;
    *((&kernelContext_0)->Hpp_inv_0+_S2) = (a01_0 * a12_0 - a02_0 * a11_0) * invDet_0;
    *((&kernelContext_0)->Hpp_inv_0+_S3) = c01_0 * invDet_0;
    *((&kernelContext_0)->Hpp_inv_0+_S4) = (a00_0 * a22_0 - a02_0 * a20_0) * invDet_0;
    *((&kernelContext_0)->Hpp_inv_0+_S5) = - (a00_0 * a12_0 - a02_0 * a10_0) * invDet_0;
    *((&kernelContext_0)->Hpp_inv_0+_S6) = c02_0 * invDet_0;
    *((&kernelContext_0)->Hpp_inv_0+_S7) = - (a00_0 * a21_0 - a01_0 * a20_0) * invDet_0;
    *((&kernelContext_0)->Hpp_inv_0+_S8) = (a00_0 * a11_0 - a01_0 * a10_0) * invDet_0;
    return;
}

