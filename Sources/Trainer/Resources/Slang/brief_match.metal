#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* dims_0;
    int device* query_desc_0;
    float device* query_lines_0;
    float device* train_xy_0;
    int device* train_desc_0;
    int device* best_idx_0;
    int device* best_dist_0;
};

[[kernel]] void brief_match(uint3 tid_0 [[thread_position_in_grid]], int device* dims_1 [[buffer(4)]], int device* query_desc_1 [[buffer(0)]], float device* query_lines_1 [[buffer(3)]], float device* train_xy_1 [[buffer(2)]], int device* train_desc_1 [[buffer(1)]], int device* best_idx_1 [[buffer(5)]], int device* best_dist_1 [[buffer(6)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->dims_0 = dims_1;
    (&kernelContext_0)->query_desc_0 = query_desc_1;
    (&kernelContext_0)->query_lines_0 = query_lines_1;
    (&kernelContext_0)->train_xy_0 = train_xy_1;
    (&kernelContext_0)->train_desc_0 = train_desc_1;
    (&kernelContext_0)->best_idx_0 = best_idx_1;
    (&kernelContext_0)->best_dist_0 = best_dist_1;
    int q_0 = int(tid_0.x);
    int _S1 = dims_1[int(1)];
    int ratioQ8_0 = dims_1[int(2)];
    int bandPx_0 = dims_1[int(3)];
    if(q_0 >= dims_1[int(0)])
    {
        return;
    }
    int _S2 = q_0 * int(8);
    int _S3 = (&kernelContext_0)->query_desc_0[_S2];
    int _S4 = (&kernelContext_0)->query_desc_0[_S2 + int(1)];
    int _S5 = (&kernelContext_0)->query_desc_0[_S2 + int(2)];
    int _S6 = (&kernelContext_0)->query_desc_0[_S2 + int(3)];
    int _S7 = (&kernelContext_0)->query_desc_0[_S2 + int(4)];
    int _S8 = (&kernelContext_0)->query_desc_0[_S2 + int(5)];
    int _S9 = (&kernelContext_0)->query_desc_0[_S2 + int(6)];
    int _S10 = (&kernelContext_0)->query_desc_0[_S2 + int(7)];
    int _S11 = q_0 * int(3);
    float _S12 = (&kernelContext_0)->query_lines_0[_S11];
    float _S13 = (&kernelContext_0)->query_lines_0[_S11 + int(1)];
    float _S14 = (&kernelContext_0)->query_lines_0[_S11 + int(2)];
    float _S15 = float(bandPx_0);
    int d1_0 = int(1000);
    int d2_0 = int(1000);
    int i1_0 = int(-1);
    int t_0 = int(0);
    for(;;)
    {
        if(t_0 < _S1)
        {
        }
        else
        {
            break;
        }
        if(bandPx_0 > int(0))
        {
            int _S16 = t_0 * int(2);
            if((abs(_S12 * (&kernelContext_0)->train_xy_0[_S16] + _S13 * (&kernelContext_0)->train_xy_0[_S16 + int(1)] + _S14)) > _S15)
            {
                t_0 = t_0 + int(1);
                continue;
            }
        }
        int _S17 = t_0 * int(8);
        int d_0 = int(((uint)popcount((_S3 ^ ((&kernelContext_0)->train_desc_0)[_S17]))) + ((uint)popcount((_S4 ^ ((&kernelContext_0)->train_desc_0)[_S17 + int(1)]))) + ((uint)popcount((_S5 ^ ((&kernelContext_0)->train_desc_0)[_S17 + int(2)]))) + ((uint)popcount((_S6 ^ ((&kernelContext_0)->train_desc_0)[_S17 + int(3)]))) + ((uint)popcount((_S7 ^ ((&kernelContext_0)->train_desc_0)[_S17 + int(4)]))) + ((uint)popcount((_S8 ^ ((&kernelContext_0)->train_desc_0)[_S17 + int(5)]))) + ((uint)popcount((_S9 ^ ((&kernelContext_0)->train_desc_0)[_S17 + int(6)]))) + ((uint)popcount((_S10 ^ ((&kernelContext_0)->train_desc_0)[_S17 + int(7)]))));
        int d1_1;
        int d2_1;
        int i1_1;
        if(d_0 < d1_0)
        {
            d1_1 = d_0;
            d2_1 = d1_0;
            i1_1 = t_0;
        }
        else
        {
            if(d_0 < d2_0)
            {
                d1_1 = d_0;
            }
            else
            {
                d1_1 = d2_0;
            }
            int _S18 = d1_1;
            d1_1 = d1_0;
            d2_1 = _S18;
            i1_1 = i1_0;
        }
        d1_0 = d1_1;
        d2_0 = d2_1;
        i1_0 = i1_1;
        t_0 = t_0 + int(1);
    }
    if((d1_0 * int(256)) < (d2_0 * ratioQ8_0))
    {
        *((&kernelContext_0)->best_idx_0+q_0) = i1_0;
        *((&kernelContext_0)->best_dist_0+q_0) = d1_0;
    }
    else
    {
        *((&kernelContext_0)->best_idx_0+q_0) = int(-1);
        *((&kernelContext_0)->best_dist_0+q_0) = d1_0;
    }
    return;
}

