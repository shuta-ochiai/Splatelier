#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* schur_dims_0;
    int device* point_obs_counts_0;
    int device* point_obs_starts_0;
    float device* bp_in_0;
    float device* Hpp_inv_in_0;
    int device* sorted_obs_idx_0;
    int device* htp_cam_idx_0;
    float device* htp_blocks_0;
    float device* redRHS_0;
    float device* redLHS_0;
};

[[kernel]] void ba_schur_reduce(uint3 tid_0 [[thread_position_in_grid]], int device* schur_dims_1 [[buffer(10)]], int device* point_obs_counts_1 [[buffer(9)]], int device* point_obs_starts_1 [[buffer(8)]], float device* bp_in_1 [[buffer(4)]], float device* Hpp_inv_in_1 [[buffer(3)]], int device* sorted_obs_idx_1 [[buffer(7)]], int device* htp_cam_idx_1 [[buffer(6)]], float device* htp_blocks_1 [[buffer(5)]], float device* redRHS_1 [[buffer(12)]], float device* redLHS_1 [[buffer(11)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->schur_dims_0 = schur_dims_1;
    (&kernelContext_0)->point_obs_counts_0 = point_obs_counts_1;
    (&kernelContext_0)->point_obs_starts_0 = point_obs_starts_1;
    (&kernelContext_0)->bp_in_0 = bp_in_1;
    (&kernelContext_0)->Hpp_inv_in_0 = Hpp_inv_in_1;
    (&kernelContext_0)->sorted_obs_idx_0 = sorted_obs_idx_1;
    (&kernelContext_0)->htp_cam_idx_0 = htp_cam_idx_1;
    (&kernelContext_0)->htp_blocks_0 = htp_blocks_1;
    (&kernelContext_0)->redRHS_0 = redRHS_1;
    (&kernelContext_0)->redLHS_0 = redLHS_1;
    int j_0 = int(tid_0.x);
    if(j_0 >= schur_dims_1[int(0)])
    {
        return;
    }
    int _S1 = (&kernelContext_0)->schur_dims_0[int(1)];
    int _S2 = (&kernelContext_0)->schur_dims_0[int(2)];
    int count_0 = (&kernelContext_0)->point_obs_counts_0[j_0];
    if(count_0 == int(0))
    {
        return;
    }
    int _S3 = (&kernelContext_0)->point_obs_starts_0[j_0];
    int hppBase_0 = j_0 * int(9);
    int _S4 = j_0 * int(3);
    float bp0_0 = (&kernelContext_0)->bp_in_0[_S4];
    float bp1_0 = (&kernelContext_0)->bp_in_0[_S4 + int(1)];
    float bp2_0 = (&kernelContext_0)->bp_in_0[_S4 + int(2)];
    float hi00_0 = (&kernelContext_0)->Hpp_inv_in_0[hppBase_0];
    float hi01_0 = (&kernelContext_0)->Hpp_inv_in_0[hppBase_0 + int(1)];
    float hi02_0 = (&kernelContext_0)->Hpp_inv_in_0[hppBase_0 + int(2)];
    float hi10_0 = (&kernelContext_0)->Hpp_inv_in_0[hppBase_0 + int(3)];
    float hi11_0 = (&kernelContext_0)->Hpp_inv_in_0[hppBase_0 + int(4)];
    float hi12_0 = (&kernelContext_0)->Hpp_inv_in_0[hppBase_0 + int(5)];
    float hi20_0 = (&kernelContext_0)->Hpp_inv_in_0[hppBase_0 + int(6)];
    float hi21_0 = (&kernelContext_0)->Hpp_inv_in_0[hppBase_0 + int(7)];
    float hi22_0 = (&kernelContext_0)->Hpp_inv_in_0[hppBase_0 + int(8)];
    float _S5 = hi00_0 * bp0_0 + hi01_0 * bp1_0 + hi02_0 * bp2_0;
    float _S6 = hi10_0 * bp0_0 + hi11_0 * bp1_0 + hi12_0 * bp2_0;
    float _S7 = hi20_0 * bp0_0 + hi21_0 * bp1_0 + hi22_0 * bp2_0;
    int k_0 = int(0);
    for(;;)
    {
        if(k_0 < count_0)
        {
        }
        else
        {
            break;
        }
        int obsI_0 = (&kernelContext_0)->sorted_obs_idx_0[_S3 + k_0];
        int camI_0 = (&kernelContext_0)->htp_cam_idx_0[obsI_0];
        if(camI_0 < _S2)
        {
            k_0 = k_0 + int(1);
            continue;
        }
        int _S8 = camI_0 - _S2;
        int _S9 = obsI_0 * int(18);
        int r_0 = int(0);
        for(;;)
        {
            if(r_0 < int(6))
            {
            }
            else
            {
                break;
            }
            int _S10 = _S9 + r_0 * int(3);
            float _S11 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->redRHS_0+(_S8 * int(6) + r_0))), - ((&kernelContext_0)->htp_blocks_0[_S10] * _S5 + (&kernelContext_0)->htp_blocks_0[_S10 + int(1)] * _S6 + (&kernelContext_0)->htp_blocks_0[_S10 + int(2)] * _S7), memory_order_relaxed);
            r_0 = r_0 + int(1);
        }
        int k2_0 = int(0);
        for(;;)
        {
            if(k2_0 < count_0)
            {
            }
            else
            {
                break;
            }
            int obsK_0 = (&kernelContext_0)->sorted_obs_idx_0[_S3 + k2_0];
            int camK_0 = (&kernelContext_0)->htp_cam_idx_0[obsK_0];
            if(camK_0 < _S2)
            {
                k2_0 = k2_0 + int(1);
                continue;
            }
            int _S12 = camK_0 - _S2;
            int bK_0 = obsK_0 * int(18);
            int _S13 = bK_0 + int(1);
            int _S14 = bK_0 + int(2);
            float _S15 = hi00_0 * (&kernelContext_0)->htp_blocks_0[bK_0] + hi01_0 * (&kernelContext_0)->htp_blocks_0[_S13] + hi02_0 * (&kernelContext_0)->htp_blocks_0[_S14];
            int _S16 = bK_0 + int(3);
            int _S17 = _S16 + int(1);
            int _S18 = _S16 + int(2);
            float _S19 = hi00_0 * (&kernelContext_0)->htp_blocks_0[_S16] + hi01_0 * (&kernelContext_0)->htp_blocks_0[_S17] + hi02_0 * (&kernelContext_0)->htp_blocks_0[_S18];
            int _S20 = bK_0 + int(6);
            int _S21 = _S20 + int(1);
            int _S22 = _S20 + int(2);
            float _S23 = hi00_0 * (&kernelContext_0)->htp_blocks_0[_S20] + hi01_0 * (&kernelContext_0)->htp_blocks_0[_S21] + hi02_0 * (&kernelContext_0)->htp_blocks_0[_S22];
            int _S24 = bK_0 + int(9);
            int _S25 = _S24 + int(1);
            int _S26 = _S24 + int(2);
            float _S27 = hi00_0 * (&kernelContext_0)->htp_blocks_0[_S24] + hi01_0 * (&kernelContext_0)->htp_blocks_0[_S25] + hi02_0 * (&kernelContext_0)->htp_blocks_0[_S26];
            int _S28 = bK_0 + int(12);
            int _S29 = _S28 + int(1);
            int _S30 = _S28 + int(2);
            float _S31 = hi00_0 * (&kernelContext_0)->htp_blocks_0[_S28] + hi01_0 * (&kernelContext_0)->htp_blocks_0[_S29] + hi02_0 * (&kernelContext_0)->htp_blocks_0[_S30];
            int _S32 = bK_0 + int(15);
            int _S33 = _S32 + int(1);
            int _S34 = _S32 + int(2);
            float _S35 = hi00_0 * (&kernelContext_0)->htp_blocks_0[_S32] + hi01_0 * (&kernelContext_0)->htp_blocks_0[_S33] + hi02_0 * (&kernelContext_0)->htp_blocks_0[_S34];
            float _S36 = hi10_0 * (&kernelContext_0)->htp_blocks_0[bK_0] + hi11_0 * (&kernelContext_0)->htp_blocks_0[_S13] + hi12_0 * (&kernelContext_0)->htp_blocks_0[_S14];
            float _S37 = hi10_0 * (&kernelContext_0)->htp_blocks_0[_S16] + hi11_0 * (&kernelContext_0)->htp_blocks_0[_S17] + hi12_0 * (&kernelContext_0)->htp_blocks_0[_S18];
            float _S38 = hi10_0 * (&kernelContext_0)->htp_blocks_0[_S20] + hi11_0 * (&kernelContext_0)->htp_blocks_0[_S21] + hi12_0 * (&kernelContext_0)->htp_blocks_0[_S22];
            float _S39 = hi10_0 * (&kernelContext_0)->htp_blocks_0[_S24] + hi11_0 * (&kernelContext_0)->htp_blocks_0[_S25] + hi12_0 * (&kernelContext_0)->htp_blocks_0[_S26];
            float _S40 = hi10_0 * (&kernelContext_0)->htp_blocks_0[_S28] + hi11_0 * (&kernelContext_0)->htp_blocks_0[_S29] + hi12_0 * (&kernelContext_0)->htp_blocks_0[_S30];
            float _S41 = hi10_0 * (&kernelContext_0)->htp_blocks_0[_S32] + hi11_0 * (&kernelContext_0)->htp_blocks_0[_S33] + hi12_0 * (&kernelContext_0)->htp_blocks_0[_S34];
            float _S42 = hi20_0 * (&kernelContext_0)->htp_blocks_0[bK_0] + hi21_0 * (&kernelContext_0)->htp_blocks_0[_S13] + hi22_0 * (&kernelContext_0)->htp_blocks_0[_S14];
            float _S43 = hi20_0 * (&kernelContext_0)->htp_blocks_0[_S16] + hi21_0 * (&kernelContext_0)->htp_blocks_0[_S17] + hi22_0 * (&kernelContext_0)->htp_blocks_0[_S18];
            float _S44 = hi20_0 * (&kernelContext_0)->htp_blocks_0[_S20] + hi21_0 * (&kernelContext_0)->htp_blocks_0[_S21] + hi22_0 * (&kernelContext_0)->htp_blocks_0[_S22];
            float _S45 = hi20_0 * (&kernelContext_0)->htp_blocks_0[_S24] + hi21_0 * (&kernelContext_0)->htp_blocks_0[_S25] + hi22_0 * (&kernelContext_0)->htp_blocks_0[_S26];
            float _S46 = hi20_0 * (&kernelContext_0)->htp_blocks_0[_S28] + hi21_0 * (&kernelContext_0)->htp_blocks_0[_S29] + hi22_0 * (&kernelContext_0)->htp_blocks_0[_S30];
            float _S47 = hi20_0 * (&kernelContext_0)->htp_blocks_0[_S32] + hi21_0 * (&kernelContext_0)->htp_blocks_0[_S33] + hi22_0 * (&kernelContext_0)->htp_blocks_0[_S34];
            int r_1 = int(0);
            for(;;)
            {
                if(r_1 < int(6))
                {
                }
                else
                {
                    break;
                }
                int _S48 = _S9 + r_1 * int(3);
                float hi0_0 = (&kernelContext_0)->htp_blocks_0[_S48];
                float hi1_0 = (&kernelContext_0)->htp_blocks_0[_S48 + int(1)];
                float hi2_0 = (&kernelContext_0)->htp_blocks_0[_S48 + int(2)];
                float m1_0 = hi0_0 * _S19 + hi1_0 * _S37 + hi2_0 * _S43;
                float m2_0 = hi0_0 * _S23 + hi1_0 * _S38 + hi2_0 * _S44;
                float m3_0 = hi0_0 * _S27 + hi1_0 * _S39 + hi2_0 * _S45;
                float m4_0 = hi0_0 * _S31 + hi1_0 * _S40 + hi2_0 * _S46;
                float m5_0 = hi0_0 * _S35 + hi1_0 * _S41 + hi2_0 * _S47;
                int rowBase_0 = (_S8 * int(6) + r_1) * _S1 + _S12 * int(6);
                float _S49 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->redLHS_0+rowBase_0)), - (hi0_0 * _S15 + hi1_0 * _S36 + hi2_0 * _S42), memory_order_relaxed);
                float _S50 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->redLHS_0+(rowBase_0 + int(1)))), - m1_0, memory_order_relaxed);
                float _S51 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->redLHS_0+(rowBase_0 + int(2)))), - m2_0, memory_order_relaxed);
                float _S52 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->redLHS_0+(rowBase_0 + int(3)))), - m3_0, memory_order_relaxed);
                float _S53 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->redLHS_0+(rowBase_0 + int(4)))), - m4_0, memory_order_relaxed);
                float _S54 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->redLHS_0+(rowBase_0 + int(5)))), - m5_0, memory_order_relaxed);
                r_1 = r_1 + int(1);
            }
            k2_0 = k2_0 + int(1);
        }
        k_0 = k_0 + int(1);
    }
    return;
}

