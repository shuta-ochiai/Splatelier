#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* ss_dims_0;
    int device* ss_tile_0;
    int device* ss_tile_start_0;
    int device* ss_flatten_0;
    int device* ss_mask_0;
    int device* ss_offsets8_0;
    int device* ss_rank_0;
    int device* ss_flatten8_0;
};

[[kernel]] void tile_split_scatter(uint3 tid_0 [[thread_position_in_grid]], int device* ss_dims_1 [[buffer(38)]], int device* ss_tile_1 [[buffer(33)]], int device* ss_tile_start_1 [[buffer(36)]], int device* ss_flatten_1 [[buffer(32)]], int device* ss_mask_1 [[buffer(34)]], int device* ss_offsets8_1 [[buffer(37)]], int device* ss_rank_1 [[buffer(35)]], int device* ss_flatten8_1 [[buffer(39)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->ss_dims_0 = ss_dims_1;
    (&kernelContext_0)->ss_tile_0 = ss_tile_1;
    (&kernelContext_0)->ss_tile_start_0 = ss_tile_start_1;
    (&kernelContext_0)->ss_flatten_0 = ss_flatten_1;
    (&kernelContext_0)->ss_mask_0 = ss_mask_1;
    (&kernelContext_0)->ss_offsets8_0 = ss_offsets8_1;
    (&kernelContext_0)->ss_rank_0 = ss_rank_1;
    (&kernelContext_0)->ss_flatten8_0 = ss_flatten8_1;
    int k_0 = int(tid_0.x);
    if(k_0 >= ss_dims_1[int(1)])
    {
        return;
    }
    int numTilesX16_0 = (&kernelContext_0)->ss_dims_0[int(0)];
    int _S1 = (&kernelContext_0)->ss_dims_0[int(2)];
    int t_0 = (&kernelContext_0)->ss_tile_0[k_0];
    int _S2 = t_0 % numTilesX16_0;
    int _S3 = t_0 / numTilesX16_0;
    int _S4 = (&kernelContext_0)->ss_tile_start_0[t_0];
    int _S5 = (&kernelContext_0)->ss_flatten_0[k_0];
    int q_0 = int(0);
    for(;;)
    {
        if(q_0 < int(4))
        {
        }
        else
        {
            break;
        }
        int _S6 = q_0 * (&kernelContext_0)->ss_dims_0[int(1)];
        int _S7 = _S6 + k_0;
        if((&kernelContext_0)->ss_mask_0[_S7] != int(0))
        {
            *((&kernelContext_0)->ss_flatten8_0+((&kernelContext_0)->ss_offsets8_0[(_S3 * int(2) + q_0 / int(2)) * _S1 + _S2 * int(2) + q_0 % int(2)] + (&kernelContext_0)->ss_rank_0[_S7] - (&kernelContext_0)->ss_rank_0[_S6 + _S4])) = _S5;
        }
        q_0 = q_0 + int(1);
    }
    return;
}

