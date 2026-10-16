#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* bs_dims_0;
    float device* deltaPt_out_0;
    int device* point_obs_counts_bs_0;
    int device* point_obs_starts_bs_0;
    float device* bp_bs_0;
    int device* sorted_obs_idx_bs_0;
    int device* htp_cam_idx_bs_0;
    float device* deltaCam_bs_0;
    float device* htp_blocks_bs_0;
    float device* Hpp_inv_bs_0;
};

[[kernel]] void ba_back_substitute(uint3 tid_0 [[thread_position_in_grid]], int device* bs_dims_1 [[buffer(21)]], float device* deltaPt_out_1 [[buffer(22)]], int device* point_obs_counts_bs_1 [[buffer(19)]], int device* point_obs_starts_bs_1 [[buffer(18)]], float device* bp_bs_1 [[buffer(14)]], int device* sorted_obs_idx_bs_1 [[buffer(17)]], int device* htp_cam_idx_bs_1 [[buffer(16)]], float device* deltaCam_bs_1 [[buffer(20)]], float device* htp_blocks_bs_1 [[buffer(15)]], float device* Hpp_inv_bs_1 [[buffer(13)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->bs_dims_0 = bs_dims_1;
    (&kernelContext_0)->deltaPt_out_0 = deltaPt_out_1;
    (&kernelContext_0)->point_obs_counts_bs_0 = point_obs_counts_bs_1;
    (&kernelContext_0)->point_obs_starts_bs_0 = point_obs_starts_bs_1;
    (&kernelContext_0)->bp_bs_0 = bp_bs_1;
    (&kernelContext_0)->sorted_obs_idx_bs_0 = sorted_obs_idx_bs_1;
    (&kernelContext_0)->htp_cam_idx_bs_0 = htp_cam_idx_bs_1;
    (&kernelContext_0)->deltaCam_bs_0 = deltaCam_bs_1;
    (&kernelContext_0)->htp_blocks_bs_0 = htp_blocks_bs_1;
    (&kernelContext_0)->Hpp_inv_bs_0 = Hpp_inv_bs_1;
    int j_0 = int(tid_0.x);
    if(j_0 >= bs_dims_1[int(0)])
    {
        return;
    }
    int _S1 = j_0 * int(3);
    *((&kernelContext_0)->deltaPt_out_0+_S1) = 0.0;
    int _S2 = _S1 + int(1);
    *((&kernelContext_0)->deltaPt_out_0+_S2) = 0.0;
    int _S3 = _S1 + int(2);
    *((&kernelContext_0)->deltaPt_out_0+_S3) = 0.0;
    int count_0 = (&kernelContext_0)->point_obs_counts_bs_0[j_0];
    if(count_0 == int(0))
    {
        return;
    }
    int _S4 = (&kernelContext_0)->point_obs_starts_bs_0[j_0];
    float _S5 = (&kernelContext_0)->bp_bs_0[_S1];
    float _S6 = (&kernelContext_0)->bp_bs_0[_S2];
    float _S7 = (&kernelContext_0)->bp_bs_0[_S3];
    int k_0 = int(0);
    float rhs0_0 = _S5;
    float rhs1_0 = _S6;
    float rhs2_0 = _S7;
    for(;;)
    {
        if(k_0 < count_0)
        {
        }
        else
        {
            break;
        }
        int obsI_0 = (&kernelContext_0)->sorted_obs_idx_bs_0[_S4 + k_0];
        int dcBase_0 = (&kernelContext_0)->htp_cam_idx_bs_0[obsI_0] * int(6);
        float dc0_0 = (&kernelContext_0)->deltaCam_bs_0[dcBase_0];
        float dc1_0 = (&kernelContext_0)->deltaCam_bs_0[dcBase_0 + int(1)];
        float dc2_0 = (&kernelContext_0)->deltaCam_bs_0[dcBase_0 + int(2)];
        float dc3_0 = (&kernelContext_0)->deltaCam_bs_0[dcBase_0 + int(3)];
        float dc4_0 = (&kernelContext_0)->deltaCam_bs_0[dcBase_0 + int(4)];
        float dc5_0 = (&kernelContext_0)->deltaCam_bs_0[dcBase_0 + int(5)];
        int bI_0 = obsI_0 * int(18);
        int _S8 = bI_0 + int(3);
        int _S9 = bI_0 + int(6);
        int _S10 = bI_0 + int(9);
        int _S11 = bI_0 + int(12);
        int _S12 = bI_0 + int(15);
        float rhs0_1 = rhs0_0 - ((&kernelContext_0)->htp_blocks_bs_0[bI_0] * dc0_0 + (&kernelContext_0)->htp_blocks_bs_0[_S8] * dc1_0 + (&kernelContext_0)->htp_blocks_bs_0[_S9] * dc2_0 + (&kernelContext_0)->htp_blocks_bs_0[_S10] * dc3_0 + (&kernelContext_0)->htp_blocks_bs_0[_S11] * dc4_0 + (&kernelContext_0)->htp_blocks_bs_0[_S12] * dc5_0);
        float rhs1_1 = rhs1_0 - ((&kernelContext_0)->htp_blocks_bs_0[bI_0 + int(1)] * dc0_0 + (&kernelContext_0)->htp_blocks_bs_0[_S8 + int(1)] * dc1_0 + (&kernelContext_0)->htp_blocks_bs_0[_S9 + int(1)] * dc2_0 + (&kernelContext_0)->htp_blocks_bs_0[_S10 + int(1)] * dc3_0 + (&kernelContext_0)->htp_blocks_bs_0[_S11 + int(1)] * dc4_0 + (&kernelContext_0)->htp_blocks_bs_0[_S12 + int(1)] * dc5_0);
        float rhs2_1 = rhs2_0 - ((&kernelContext_0)->htp_blocks_bs_0[bI_0 + int(2)] * dc0_0 + (&kernelContext_0)->htp_blocks_bs_0[_S8 + int(2)] * dc1_0 + (&kernelContext_0)->htp_blocks_bs_0[_S9 + int(2)] * dc2_0 + (&kernelContext_0)->htp_blocks_bs_0[_S10 + int(2)] * dc3_0 + (&kernelContext_0)->htp_blocks_bs_0[_S11 + int(2)] * dc4_0 + (&kernelContext_0)->htp_blocks_bs_0[_S12 + int(2)] * dc5_0);
        k_0 = k_0 + int(1);
        rhs0_0 = rhs0_1;
        rhs1_0 = rhs1_1;
        rhs2_0 = rhs2_1;
    }
    int hi_0 = j_0 * int(9);
    float hi10_0 = (&kernelContext_0)->Hpp_inv_bs_0[hi_0 + int(3)];
    float hi11_0 = (&kernelContext_0)->Hpp_inv_bs_0[hi_0 + int(4)];
    float hi12_0 = (&kernelContext_0)->Hpp_inv_bs_0[hi_0 + int(5)];
    float hi20_0 = (&kernelContext_0)->Hpp_inv_bs_0[hi_0 + int(6)];
    float hi21_0 = (&kernelContext_0)->Hpp_inv_bs_0[hi_0 + int(7)];
    float hi22_0 = (&kernelContext_0)->Hpp_inv_bs_0[hi_0 + int(8)];
    *((&kernelContext_0)->deltaPt_out_0+_S1) = (&kernelContext_0)->Hpp_inv_bs_0[hi_0] * rhs0_0 + (&kernelContext_0)->Hpp_inv_bs_0[hi_0 + int(1)] * rhs1_0 + (&kernelContext_0)->Hpp_inv_bs_0[hi_0 + int(2)] * rhs2_0;
    *((&kernelContext_0)->deltaPt_out_0+_S2) = hi10_0 * rhs0_0 + hi11_0 * rhs1_0 + hi12_0 * rhs2_0;
    *((&kernelContext_0)->deltaPt_out_0+_S3) = hi20_0 * rhs0_0 + hi21_0 * rhs1_0 + hi22_0 * rhs2_0;
    return;
}

