#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* grid_dims_0;
    float device* grid_h_0;
    float device* positions_0;
    int device* query_cell_0;
    int device* cell_coords_0;
    int device* cell_start_0;
    float device* sorted_pos_0;
    int device* sorted_orig_0;
    int device* outIndices_0;
    float device* outDistances_0;
};

[[kernel]] void knn_grid(uint3 tid_0 [[thread_position_in_grid]], int device* grid_dims_1 [[buffer(6)]], float device* grid_h_1 [[buffer(7)]], float device* positions_1 [[buffer(0)]], int device* query_cell_1 [[buffer(1)]], int device* cell_coords_1 [[buffer(4)]], int device* cell_start_1 [[buffer(5)]], float device* sorted_pos_1 [[buffer(2)]], int device* sorted_orig_1 [[buffer(3)]], int device* outIndices_1 [[buffer(8)]], float device* outDistances_1 [[buffer(9)]])
{
    int dx_0;
    int dy_0;
    int dz_0;
    bool _S1;
    int c_0;
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->grid_dims_0 = grid_dims_1;
    (&kernelContext_0)->grid_h_0 = grid_h_1;
    (&kernelContext_0)->positions_0 = positions_1;
    (&kernelContext_0)->query_cell_0 = query_cell_1;
    (&kernelContext_0)->cell_coords_0 = cell_coords_1;
    (&kernelContext_0)->cell_start_0 = cell_start_1;
    (&kernelContext_0)->sorted_pos_0 = sorted_pos_1;
    (&kernelContext_0)->sorted_orig_0 = sorted_orig_1;
    (&kernelContext_0)->outIndices_0 = outIndices_1;
    (&kernelContext_0)->outDistances_0 = outDistances_1;
    int i_0 = int(tid_0.x);
    int N_0 = grid_dims_1[int(0)];
    int K_0 = grid_dims_1[int(1)];
    int _S2 = grid_dims_1[int(2)];
    int _S3 = grid_dims_1[int(3)];
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
    float _S4 = (&kernelContext_0)->grid_h_0[int(0)];
    int _S5 = i_0 * int(3);
    float _S6 = (&kernelContext_0)->positions_0[_S5];
    int _S7 = _S5 + int(1);
    float _S8 = (&kernelContext_0)->positions_0[_S7];
    int _S9 = _S5 + int(2);
    float _S10 = (&kernelContext_0)->positions_0[_S9];
    int _S11 = (&kernelContext_0)->query_cell_0[_S5];
    int _S12 = (&kernelContext_0)->query_cell_0[_S7];
    int _S13 = (&kernelContext_0)->query_cell_0[_S9];
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
    bool done_0 = false;
    int r_0 = int(0);
    for(;;)
    {
        bool _S14;
        if(r_0 <= _S3)
        {
            _S14 = !done_0;
        }
        else
        {
            _S14 = false;
        }
        if(_S14)
        {
        }
        else
        {
            break;
        }
        int _S15 = - r_0;
        dx_0 = _S15;
        for(;;)
        {
            if(dx_0 <= r_0)
            {
            }
            else
            {
                break;
            }
            dy_0 = _S15;
            for(;;)
            {
                if(dy_0 <= r_0)
                {
                }
                else
                {
                    break;
                }
                dz_0 = _S15;
                for(;;)
                {
                    if(dz_0 <= r_0)
                    {
                    }
                    else
                    {
                        break;
                    }
                    if((max(abs(dx_0), max(abs(dy_0), abs(dz_0)))) != r_0)
                    {
                        dz_0 = dz_0 + int(1);
                        continue;
                    }
                    int _S16 = _S11 + dx_0;
                    int _S17 = _S12 + dy_0;
                    int _S18 = _S13 + dz_0;
                    int _S19 = _S2 - int(1);
                    int lo_0 = int(0);
                    int hi_0 = _S19;
                    for(;;)
                    {
                        if(lo_0 <= hi_0)
                        {
                        }
                        else
                        {
                            c_0 = int(-1);
                            break;
                        }
                        int mid_0 = (lo_0 + hi_0) >> 1U;
                        int _S20 = mid_0 * int(3);
                        int mx_0 = (&kernelContext_0)->cell_coords_0[_S20];
                        int my_0 = (&kernelContext_0)->cell_coords_0[_S20 + int(1)];
                        int mz_0 = (&kernelContext_0)->cell_coords_0[_S20 + int(2)];
                        if(mx_0 != _S16)
                        {
                            if(mx_0 < _S16)
                            {
                                s_0 = int(-1);
                            }
                            else
                            {
                                s_0 = int(1);
                            }
                            c_0 = s_0;
                        }
                        else
                        {
                            if(my_0 != _S17)
                            {
                                if(my_0 < _S17)
                                {
                                    s_0 = int(-1);
                                }
                                else
                                {
                                    s_0 = int(1);
                                }
                            }
                            else
                            {
                                if(mz_0 != _S18)
                                {
                                    if(mz_0 < _S18)
                                    {
                                        s_0 = int(-1);
                                    }
                                    else
                                    {
                                        s_0 = int(1);
                                    }
                                }
                                else
                                {
                                    s_0 = int(0);
                                }
                            }
                            c_0 = s_0;
                        }
                        if(c_0 == int(0))
                        {
                            c_0 = mid_0;
                            break;
                        }
                        if(c_0 < int(0))
                        {
                            lo_0 = mid_0 + int(1);
                        }
                        else
                        {
                            hi_0 = mid_0 - int(1);
                        }
                    }
                    if(c_0 < int(0))
                    {
                        dz_0 = dz_0 + int(1);
                        continue;
                    }
                    s_0 = (&kernelContext_0)->cell_start_0[c_0];
                    for(;;)
                    {
                        if(s_0 < ((&kernelContext_0)->cell_start_0)[c_0 + int(1)])
                        {
                        }
                        else
                        {
                            break;
                        }
                        int _S21 = s_0 * int(3);
                        float ex_0 = (&kernelContext_0)->sorted_pos_0[_S21] - _S6;
                        float ey_0 = (&kernelContext_0)->sorted_pos_0[_S21 + int(1)] - _S8;
                        float ez_0 = (&kernelContext_0)->sorted_pos_0[_S21 + int(2)] - _S10;
                        float d2_0 = ex_0 * ex_0 + ey_0 * ey_0 + ez_0 * ez_0;
                        int j_0 = (&kernelContext_0)->sorted_orig_0[s_0];
                        if(d2_0 > bestD2_0[K_1])
                        {
                            _S1 = true;
                        }
                        else
                        {
                            if(d2_0 == bestD2_0[K_1])
                            {
                                _S1 = j_0 > bestIdx_0[K_1];
                            }
                            else
                            {
                                _S1 = false;
                            }
                        }
                        if(_S1)
                        {
                            s_0 = s_0 + int(1);
                            continue;
                        }
                        int hop_0 = int(0);
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
                            int _S22 = slot_0 - int(1);
                            bool _S23;
                            if((bestD2_0[_S22]) < d2_0)
                            {
                                _S23 = true;
                            }
                            else
                            {
                                if((bestD2_0[_S22]) == d2_0)
                                {
                                    _S23 = (bestIdx_0[_S22]) < j_0;
                                }
                                else
                                {
                                    _S23 = false;
                                }
                            }
                            if(_S23)
                            {
                                break;
                            }
                            bestD2_0[slot_0] = bestD2_0[_S22];
                            bestIdx_0[slot_0] = bestIdx_0[_S22];
                            hop_0 = hop_0 + int(1);
                            slot_0 = _S22;
                        }
                        bestD2_0[slot_0] = d2_0;
                        bestIdx_0[slot_0] = j_0;
                        s_0 = s_0 + int(1);
                    }
                    dz_0 = dz_0 + int(1);
                }
                dy_0 = dy_0 + int(1);
            }
            dx_0 = dx_0 + int(1);
        }
        float bound_0 = float(r_0) * _S4 * 0.99998998641967773;
        if((bestIdx_0[K_1]) >= int(0))
        {
            _S1 = (bestD2_0[K_1]) < (bound_0 * bound_0);
        }
        else
        {
            _S1 = false;
        }
        if(_S1)
        {
            done_0 = true;
        }
        r_0 = r_0 + int(1);
    }
    if(!done_0)
    {
        s_0 = int(0);
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
        dx_0 = int(0);
        for(;;)
        {
            if(dx_0 < N_0)
            {
            }
            else
            {
                break;
            }
            int _S24 = dx_0 * int(3);
            float ex_1 = (&kernelContext_0)->positions_0[_S24] - _S6;
            float ey_1 = (&kernelContext_0)->positions_0[_S24 + int(1)] - _S8;
            float ez_1 = (&kernelContext_0)->positions_0[_S24 + int(2)] - _S10;
            float d2_1 = ex_1 * ex_1 + ey_1 * ey_1 + ez_1 * ez_1;
            if(d2_1 >= bestD2_0[K_1])
            {
                dx_0 = dx_0 + int(1);
                continue;
            }
            dy_0 = int(0);
            dz_0 = K_1;
            for(;;)
            {
                if(dy_0 < int(16))
                {
                }
                else
                {
                    break;
                }
                if(dz_0 <= int(0))
                {
                    break;
                }
                int _S25 = dz_0 - int(1);
                if((bestD2_0[_S25]) <= d2_1)
                {
                    break;
                }
                bestD2_0[dz_0] = bestD2_0[_S25];
                bestIdx_0[dz_0] = bestIdx_0[_S25];
                dy_0 = dy_0 + int(1);
                dz_0 = _S25;
            }
            bestD2_0[dz_0] = d2_1;
            bestIdx_0[dz_0] = dx_0;
            dx_0 = dx_0 + int(1);
        }
    }
    dx_0 = int(0);
    for(;;)
    {
        if(dx_0 < K_1)
        {
        }
        else
        {
            break;
        }
        int _S26 = i_0 * K_1 + dx_0;
        int _S27 = dx_0 + int(1);
        *((&kernelContext_0)->outIndices_0+_S26) = bestIdx_0[_S27];
        *((&kernelContext_0)->outDistances_0+_S26) = sqrt(max(0.0, bestD2_0[_S27]));
        dx_0 = _S27;
    }
    return;
}

