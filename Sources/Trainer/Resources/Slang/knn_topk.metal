#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* knn_dims_0;
    float device* positions_0;
    int device* outIndices_0;
    float device* outDistances_0;
};

[[kernel]] void knn_topk(uint3 tid_0 [[thread_position_in_grid]], int device* knn_dims_1 [[buffer(1)]], float device* positions_1 [[buffer(0)]], int device* outIndices_1 [[buffer(2)]], float device* outDistances_1 [[buffer(3)]])
{
    int hop_0;
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->knn_dims_0 = knn_dims_1;
    (&kernelContext_0)->positions_0 = positions_1;
    (&kernelContext_0)->outIndices_0 = outIndices_1;
    (&kernelContext_0)->outDistances_0 = outDistances_1;
    int i_0 = int(tid_0.x);
    int N_0 = knn_dims_1[int(0)];
    int K_0 = knn_dims_1[int(1)];
    if(i_0 >= N_0)
    {
        return;
    }
    int K_1;
    if(K_0 > int(16))
    {
        K_1 = int(16);
    }
    else
    {
        K_1 = K_0;
    }
    int _S1 = i_0 * int(3);
    float _S2 = (&kernelContext_0)->positions_0[_S1];
    float _S3 = (&kernelContext_0)->positions_0[_S1 + int(1)];
    float _S4 = (&kernelContext_0)->positions_0[_S1 + int(2)];
    thread array<float, int(17)> bestD2_0;
    thread array<int, int(17)> bestIdx_0;
    int s_0 = int(0);
    for(;;)
    {
        if(s_0 < int(17))
        {
        }
        else
        {
            break;
        }
        bestD2_0[s_0] = 3.40282306073709653e+38;
        bestIdx_0[s_0] = int(-1);
        s_0 = s_0 + int(1);
    }
    int j_0 = int(0);
    for(;;)
    {
        if(j_0 < N_0)
        {
        }
        else
        {
            break;
        }
        int _S5 = j_0 * int(3);
        float dx_0 = (&kernelContext_0)->positions_0[_S5] - _S2;
        float dy_0 = (&kernelContext_0)->positions_0[_S5 + int(1)] - _S3;
        float dz_0 = (&kernelContext_0)->positions_0[_S5 + int(2)] - _S4;
        float d2_0 = dx_0 * dx_0 + dy_0 * dy_0 + dz_0 * dz_0;
        if(d2_0 >= bestD2_0[K_1])
        {
            j_0 = j_0 + int(1);
            continue;
        }
        hop_0 = int(0);
        int slot_0 = K_1;
        for(;;)
        {
            if(hop_0 < int(16))
            {
            }
            else
            {
                break;
            }
            if(slot_0 <= int(0))
            {
                break;
            }
            int _S6 = slot_0 - int(1);
            if((bestD2_0[_S6]) <= d2_0)
            {
                break;
            }
            bestD2_0[slot_0] = bestD2_0[_S6];
            bestIdx_0[slot_0] = bestIdx_0[_S6];
            hop_0 = hop_0 + int(1);
            slot_0 = _S6;
        }
        bestD2_0[slot_0] = d2_0;
        bestIdx_0[slot_0] = j_0;
        j_0 = j_0 + int(1);
    }
    hop_0 = int(0);
    for(;;)
    {
        if(hop_0 < K_1)
        {
        }
        else
        {
            break;
        }
        int _S7 = i_0 * K_1 + hop_0;
        int _S8 = hop_0 + int(1);
        *((&kernelContext_0)->outIndices_0+_S7) = bestIdx_0[_S8];
        *((&kernelContext_0)->outDistances_0+_S7) = sqrt(max(0.0, bestD2_0[_S8]));
        hop_0 = _S8;
    }
    return;
}

