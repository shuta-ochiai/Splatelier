#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* dims_0;
    int device* target_census_0;
    float device* intrinsics_target_0;
    float device* depths_0;
    float device* intrinsics_ref_0;
    float device* rt_0;
    int device* ref_census_0;
    int device* cost_volume_0;
};

[[kernel]] void plane_sweep_cost_median(uint3 tid_0 [[thread_position_in_grid]], int device* dims_1 [[buffer(6)]], int device* target_census_1 [[buffer(0)]], float device* intrinsics_target_1 [[buffer(2)]], float device* depths_1 [[buffer(5)]], float device* intrinsics_ref_1 [[buffer(3)]], float device* rt_1 [[buffer(4)]], int device* ref_census_1 [[buffer(1)]], int device* cost_volume_1 [[buffer(7)]])
{
    int cost_0;
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->dims_0 = dims_1;
    (&kernelContext_0)->target_census_0 = target_census_1;
    (&kernelContext_0)->intrinsics_target_0 = intrinsics_target_1;
    (&kernelContext_0)->depths_0 = depths_1;
    (&kernelContext_0)->intrinsics_ref_0 = intrinsics_ref_1;
    (&kernelContext_0)->rt_0 = rt_1;
    (&kernelContext_0)->ref_census_0 = ref_census_1;
    (&kernelContext_0)->cost_volume_0 = cost_volume_1;
    int W_0 = dims_1[int(0)];
    int H_0 = dims_1[int(1)];
    int _S1 = dims_1[int(2)];
    int _S2 = dims_1[int(3)];
    int _S3 = dims_1[int(4)];
    int _S4 = min(dims_1[int(5)], int(8));
    int xt_0 = int(tid_0.x);
    int yt_0 = int(tid_0.y);
    bool _S5;
    if(xt_0 >= W_0)
    {
        _S5 = true;
    }
    else
    {
        _S5 = yt_0 >= H_0;
    }
    if(_S5)
    {
        return;
    }
    int pix_t_0 = yt_0 * W_0 + xt_0;
    int _S6 = (&kernelContext_0)->target_census_0[pix_t_0];
    float _S7 = (float(xt_0) + 0.5 - (&kernelContext_0)->intrinsics_target_0[int(2)]) / (&kernelContext_0)->intrinsics_target_0[int(0)];
    float _S8 = (float(yt_0) + 0.5 - (&kernelContext_0)->intrinsics_target_0[int(3)]) / (&kernelContext_0)->intrinsics_target_0[int(1)];
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
        float _S9 = _S7 * d_0;
        float _S10 = _S8 * d_0;
        thread array<int, int(8)> c_0;
        int m_0 = int(0);
        for(;;)
        {
            if(m_0 < _S4)
            {
            }
            else
            {
                break;
            }
            int _S11 = m_0 * int(4);
            float fx_r_0 = (&kernelContext_0)->intrinsics_ref_0[_S11];
            float fy_r_0 = (&kernelContext_0)->intrinsics_ref_0[_S11 + int(1)];
            float cx_r_0 = (&kernelContext_0)->intrinsics_ref_0[_S11 + int(2)];
            float cy_r_0 = (&kernelContext_0)->intrinsics_ref_0[_S11 + int(3)];
            int o_0 = m_0 * int(12);
            float Xr_0 = (&kernelContext_0)->rt_0[o_0] * _S9 + (&kernelContext_0)->rt_0[o_0 + int(1)] * _S10 + (&kernelContext_0)->rt_0[o_0 + int(2)] * d_0 + (&kernelContext_0)->rt_0[o_0 + int(9)];
            float Yr_0 = (&kernelContext_0)->rt_0[o_0 + int(3)] * _S9 + (&kernelContext_0)->rt_0[o_0 + int(4)] * _S10 + (&kernelContext_0)->rt_0[o_0 + int(5)] * d_0 + (&kernelContext_0)->rt_0[o_0 + int(10)];
            float Zr_0 = (&kernelContext_0)->rt_0[o_0 + int(6)] * _S9 + (&kernelContext_0)->rt_0[o_0 + int(7)] * _S10 + (&kernelContext_0)->rt_0[o_0 + int(8)] * d_0 + (&kernelContext_0)->rt_0[o_0 + int(11)];
            if(Zr_0 > 9.99999997475242708e-07)
            {
                int ix_0 = int(floor(fx_r_0 * Xr_0 / Zr_0 + cx_r_0));
                int iy_0 = int(floor(fy_r_0 * Yr_0 / Zr_0 + cy_r_0));
                if(ix_0 >= int(0))
                {
                    _S5 = ix_0 < _S1;
                }
                else
                {
                    _S5 = false;
                }
                bool _S12;
                if(_S5)
                {
                    _S12 = iy_0 >= int(0);
                }
                else
                {
                    _S12 = false;
                }
                bool _S13;
                if(_S12)
                {
                    _S13 = iy_0 < _S2;
                }
                else
                {
                    _S13 = false;
                }
                if(_S13)
                {
                    cost_0 = int(((uint)popcount(((_S6 ^ ((&kernelContext_0)->ref_census_0)[m_0 * _S2 * _S1 + iy_0 * _S1 + ix_0]) & int(33554431)))));
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
            c_0[m_0] = cost_0;
            m_0 = m_0 + int(1);
        }
        cost_0 = int(0);
        for(;;)
        {
            if(cost_0 < _S4)
            {
            }
            else
            {
                break;
            }
            int i_0 = cost_0 % int(2);
            for(;;)
            {
                if(i_0 < (_S4 - int(1)))
                {
                }
                else
                {
                    break;
                }
                int _S14 = max(c_0[i_0], c_0[i_0 + int(1)]);
                c_0[i_0] = min(c_0[i_0], c_0[i_0 + int(1)]);
                c_0[i_0 + int(1)] = _S14;
                i_0 = i_0 + int(2);
            }
            cost_0 = cost_0 + int(1);
        }
        *((&kernelContext_0)->cost_volume_0+(pix_t_0 * _S3 + k_0)) = c_0[_S4 / int(2)];
        k_0 = k_0 + int(1);
    }
    return;
}

