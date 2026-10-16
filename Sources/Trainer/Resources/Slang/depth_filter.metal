#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* dims_0;
    float device* depth_map_0;
    float device* confidence_map_0;
    int device* agg_cost_0;
    int device* exclude_radius_buf_0;
    float device* threshold_buf_0;
    float device* depths_0;
};

[[kernel]] void depth_filter(uint3 tid_0 [[thread_position_in_grid]], int device* dims_1 [[buffer(2)]], float device* depth_map_1 [[buffer(5)]], float device* confidence_map_1 [[buffer(6)]], int device* agg_cost_1 [[buffer(0)]], int device* exclude_radius_buf_1 [[buffer(4)]], float device* threshold_buf_1 [[buffer(3)]], float device* depths_1 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->dims_0 = dims_1;
    (&kernelContext_0)->depth_map_0 = depth_map_1;
    (&kernelContext_0)->confidence_map_0 = confidence_map_1;
    (&kernelContext_0)->agg_cost_0 = agg_cost_1;
    (&kernelContext_0)->exclude_radius_buf_0 = exclude_radius_buf_1;
    (&kernelContext_0)->threshold_buf_0 = threshold_buf_1;
    (&kernelContext_0)->depths_0 = depths_1;
    int W_0 = dims_1[int(0)];
    int H_0 = dims_1[int(1)];
    int K_0 = dims_1[int(2)];
    int x_0 = int(tid_0.x);
    int y_0 = int(tid_0.y);
    bool _S1;
    if(x_0 >= W_0)
    {
        _S1 = true;
    }
    else
    {
        _S1 = y_0 >= H_0;
    }
    if(_S1)
    {
        return;
    }
    int pix_0 = y_0 * W_0 + x_0;
    if(K_0 <= int(0))
    {
        *((&kernelContext_0)->depth_map_0+pix_0) = -1.0;
        *((&kernelContext_0)->confidence_map_0+pix_0) = 0.0;
        return;
    }
    int _S2 = pix_0 * K_0;
    int bestCost_0 = (&kernelContext_0)->agg_cost_0[_S2];
    int bestIdx_0 = int(0);
    int k_0 = int(1);
    for(;;)
    {
        if(k_0 < K_0)
        {
        }
        else
        {
            break;
        }
        int c_0 = (&kernelContext_0)->agg_cost_0[_S2 + k_0];
        if(c_0 < bestCost_0)
        {
            bestCost_0 = c_0;
            bestIdx_0 = k_0;
        }
        k_0 = k_0 + int(1);
    }
    int _S3 = (&kernelContext_0)->exclude_radius_buf_0[int(0)];
    int secondCost_0 = int(2147483647);
    k_0 = int(0);
    for(;;)
    {
        if(k_0 < K_0)
        {
        }
        else
        {
            break;
        }
        int d_0 = k_0 - bestIdx_0;
        int d_1;
        if(d_0 < int(0))
        {
            d_1 = - d_0;
        }
        else
        {
            d_1 = d_0;
        }
        if(d_1 <= _S3)
        {
            k_0 = k_0 + int(1);
            continue;
        }
        int c_1 = (&kernelContext_0)->agg_cost_0[_S2 + k_0];
        int secondCost_1;
        if(c_1 < secondCost_0)
        {
            secondCost_1 = c_1;
        }
        else
        {
            secondCost_1 = secondCost_0;
        }
        secondCost_0 = secondCost_1;
        k_0 = k_0 + int(1);
    }
    if(secondCost_0 < int(2147483647))
    {
        _S1 = secondCost_0 > int(0);
    }
    else
    {
        _S1 = false;
    }
    float conf_0;
    if(_S1)
    {
        conf_0 = 1.0 - float(bestCost_0) / float(secondCost_0);
    }
    else
    {
        conf_0 = 0.0;
    }
    *((&kernelContext_0)->confidence_map_0+pix_0) = conf_0;
    if(conf_0 >= ((&kernelContext_0)->threshold_buf_0)[int(0)])
    {
        float _S4 = (&kernelContext_0)->depths_0[bestIdx_0];
        if(K_0 >= int(3))
        {
            _S1 = bestIdx_0 > int(0);
        }
        else
        {
            _S1 = false;
        }
        if(_S1)
        {
            _S1 = bestIdx_0 < (K_0 - int(1));
        }
        else
        {
            _S1 = false;
        }
        float chosenDepth_0;
        if(_S1)
        {
            int _S5 = _S2 + bestIdx_0;
            float cm_0 = float((&kernelContext_0)->agg_cost_0[_S5 - int(1)]);
            float cp_0 = float((&kernelContext_0)->agg_cost_0[_S5 + int(1)]);
            float denom_0 = cm_0 - 2.0 * float(bestCost_0) + cp_0;
            if(denom_0 > 0.0)
            {
                float wBest_0 = 1.0 / (&kernelContext_0)->depths_0[bestIdx_0];
                chosenDepth_0 = 1.0 / max(wBest_0 + clamp((cm_0 - cp_0) / (2.0 * denom_0), -0.5, 0.5) * (1.0 / (&kernelContext_0)->depths_0[bestIdx_0 + int(1)] - wBest_0), 9.99999997475242708e-07);
            }
            else
            {
                chosenDepth_0 = _S4;
            }
        }
        else
        {
            chosenDepth_0 = _S4;
        }
        *((&kernelContext_0)->depth_map_0+pix_0) = chosenDepth_0;
    }
    else
    {
        *((&kernelContext_0)->depth_map_0+pix_0) = -1.0;
    }
    return;
}

