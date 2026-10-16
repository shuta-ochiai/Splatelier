#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* dims_0;
    int device* target_census_0;
    float device* intrinsics_target_0;
    float device* intrinsics_ref_0;
    float device* rt_0;
    float device* depths_0;
    int device* ref_census_0;
    int device* cost_volume_0;
};

[[kernel]] void plane_sweep_cost(uint3 tid_0 [[thread_position_in_grid]], int device* dims_1 [[buffer(6)]], int device* target_census_1 [[buffer(0)]], float device* intrinsics_target_1 [[buffer(2)]], float device* intrinsics_ref_1 [[buffer(3)]], float device* rt_1 [[buffer(4)]], float device* depths_1 [[buffer(5)]], int device* ref_census_1 [[buffer(1)]], int device* cost_volume_1 [[buffer(7)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->dims_0 = dims_1;
    (&kernelContext_0)->target_census_0 = target_census_1;
    (&kernelContext_0)->intrinsics_target_0 = intrinsics_target_1;
    (&kernelContext_0)->intrinsics_ref_0 = intrinsics_ref_1;
    (&kernelContext_0)->rt_0 = rt_1;
    (&kernelContext_0)->depths_0 = depths_1;
    (&kernelContext_0)->ref_census_0 = ref_census_1;
    (&kernelContext_0)->cost_volume_0 = cost_volume_1;
    int W_0 = dims_1[int(0)];
    int H_0 = dims_1[int(1)];
    int _S1 = dims_1[int(2)];
    int _S2 = dims_1[int(3)];
    int _S3 = dims_1[int(4)];
    int xt_0 = int(tid_0.x);
    int yt_0 = int(tid_0.y);
    bool _S4;
    if(xt_0 >= W_0)
    {
        _S4 = true;
    }
    else
    {
        _S4 = yt_0 >= H_0;
    }
    if(_S4)
    {
        return;
    }
    int pix_t_0 = yt_0 * W_0 + xt_0;
    int _S5 = (&kernelContext_0)->target_census_0[pix_t_0];
    float _S6 = (&kernelContext_0)->intrinsics_ref_0[int(0)];
    float _S7 = (&kernelContext_0)->intrinsics_ref_0[int(1)];
    float _S8 = (&kernelContext_0)->intrinsics_ref_0[int(2)];
    float _S9 = (&kernelContext_0)->intrinsics_ref_0[int(3)];
    float _S10 = (&kernelContext_0)->rt_0[int(0)];
    float _S11 = (&kernelContext_0)->rt_0[int(1)];
    float _S12 = (&kernelContext_0)->rt_0[int(2)];
    float _S13 = (&kernelContext_0)->rt_0[int(3)];
    float _S14 = (&kernelContext_0)->rt_0[int(4)];
    float _S15 = (&kernelContext_0)->rt_0[int(5)];
    float _S16 = (&kernelContext_0)->rt_0[int(6)];
    float _S17 = (&kernelContext_0)->rt_0[int(7)];
    float _S18 = (&kernelContext_0)->rt_0[int(8)];
    float _S19 = (&kernelContext_0)->rt_0[int(9)];
    float _S20 = (&kernelContext_0)->rt_0[int(10)];
    float _S21 = (&kernelContext_0)->rt_0[int(11)];
    float _S22 = (float(xt_0) + 0.5 - (&kernelContext_0)->intrinsics_target_0[int(2)]) / (&kernelContext_0)->intrinsics_target_0[int(0)];
    float _S23 = (float(yt_0) + 0.5 - (&kernelContext_0)->intrinsics_target_0[int(3)]) / (&kernelContext_0)->intrinsics_target_0[int(1)];
    int k_0 = int(0);
    for(;;)
    {
        if(k_0 < _S3)
        {
        }
        else
        {
            break;
        }
        float d_0 = (&kernelContext_0)->depths_0[k_0];
        float Xt_0 = _S22 * d_0;
        float Yt_0 = _S23 * d_0;
        float Xr_0 = _S10 * Xt_0 + _S11 * Yt_0 + _S12 * d_0 + _S19;
        float Yr_0 = _S13 * Xt_0 + _S14 * Yt_0 + _S15 * d_0 + _S20;
        float Zr_0 = _S16 * Xt_0 + _S17 * Yt_0 + _S18 * d_0 + _S21;
        int cost_0;
        if(Zr_0 > 9.99999997475242708e-07)
        {
            int ix_0 = int(floor(_S6 * Xr_0 / Zr_0 + _S8));
            int iy_0 = int(floor(_S7 * Yr_0 / Zr_0 + _S9));
            if(ix_0 >= int(0))
            {
                _S4 = ix_0 < _S1;
            }
            else
            {
                _S4 = false;
            }
            bool _S24;
            if(_S4)
            {
                _S24 = iy_0 >= int(0);
            }
            else
            {
                _S24 = false;
            }
            bool _S25;
            if(_S24)
            {
                _S25 = iy_0 < _S2;
            }
            else
            {
                _S25 = false;
            }
            if(_S25)
            {
                cost_0 = int(((uint)popcount(((_S5 ^ ((&kernelContext_0)->ref_census_0)[iy_0 * _S1 + ix_0]) & int(33554431)))));
            }
            else
            {
                cost_0 = int(25);
            }
        }
        else
        {
            cost_0 = int(25);
        }
        *((&kernelContext_0)->cost_volume_0+(pix_t_0 * _S3 + k_0)) = cost_0;
        k_0 = k_0 + int(1);
    }
    return;
}

