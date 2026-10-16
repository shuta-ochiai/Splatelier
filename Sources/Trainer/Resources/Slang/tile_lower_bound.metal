#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* lb_dims_0;
    int device* tile_offsets_0;
    int device* sorted_tile_ids_0;
};

[[kernel]] void tile_lower_bound(uint3 tid_0 [[thread_position_in_grid]], int device* lb_dims_1 [[buffer(22)]], int device* tile_offsets_1 [[buffer(23)]], int device* sorted_tile_ids_1 [[buffer(21)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->lb_dims_0 = lb_dims_1;
    (&kernelContext_0)->tile_offsets_0 = tile_offsets_1;
    (&kernelContext_0)->sorted_tile_ids_0 = sorted_tile_ids_1;
    int t_0 = int(tid_0.x);
    int numTiles_0 = lb_dims_1[int(1)];
    if(t_0 > numTiles_0)
    {
        return;
    }
    int nIsects_0 = (&kernelContext_0)->lb_dims_0[int(0)];
    if(t_0 == numTiles_0)
    {
        *((&kernelContext_0)->tile_offsets_0+t_0) = nIsects_0;
        return;
    }
    int lo_0 = int(0);
    int hi_0 = nIsects_0;
    for(;;)
    {
        if(lo_0 < hi_0)
        {
        }
        else
        {
            break;
        }
        int mid_0 = (lo_0 + hi_0) >> 1U;
        if((&kernelContext_0)->sorted_tile_ids_0[mid_0] < t_0)
        {
            lo_0 = mid_0 + int(1);
        }
        else
        {
            hi_0 = mid_0;
        }
    }
    *((&kernelContext_0)->tile_offsets_0+t_0) = lo_0;
    return;
}

