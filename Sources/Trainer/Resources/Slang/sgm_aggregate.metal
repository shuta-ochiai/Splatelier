#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* dims_0;
    int device* cost_volume_0;
    int device* L_out_0;
};

[[kernel]] void sgm_aggregate(uint3 tid_0 [[thread_position_in_grid]], int device* dims_1 [[buffer(1)]], int device* cost_volume_1 [[buffer(0)]], int device* L_out_1 [[buffer(2)]])
{
    int term2_0;
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->dims_0 = dims_1;
    (&kernelContext_0)->cost_volume_0 = cost_volume_1;
    (&kernelContext_0)->L_out_0 = L_out_1;
    int W_0 = dims_1[int(0)];
    int H_0 = dims_1[int(1)];
    int K_0 = dims_1[int(2)];
    int _S1 = dims_1[int(3)];
    int _S2 = dims_1[int(4)];
    int dir_0 = dims_1[int(5)];
    int t_0 = int(tid_0.x);
    int length_0;
    int k0_0;
    int y0_0;
    int x0_0;
    int dx_0;
    int dy_0;
    if(dir_0 == int(0))
    {
        if(t_0 >= H_0)
        {
            return;
        }
        y0_0 = t_0;
        x0_0 = int(0);
        length_0 = W_0;
        dx_0 = int(1);
        dy_0 = int(0);
    }
    else
    {
        if(dir_0 == int(1))
        {
            if(t_0 >= H_0)
            {
                return;
            }
            int _S3 = W_0 - int(1);
            y0_0 = t_0;
            x0_0 = _S3;
            length_0 = W_0;
            dx_0 = int(-1);
            dy_0 = int(0);
        }
        else
        {
            if(dir_0 == int(2))
            {
                if(t_0 >= W_0)
                {
                    return;
                }
                y0_0 = int(0);
                x0_0 = t_0;
                length_0 = H_0;
                dx_0 = int(0);
                dy_0 = int(1);
            }
            else
            {
                if(dir_0 == int(3))
                {
                    if(t_0 >= W_0)
                    {
                        return;
                    }
                    y0_0 = H_0 - int(1);
                    x0_0 = t_0;
                    length_0 = H_0;
                    dx_0 = int(0);
                    dy_0 = int(-1);
                }
                else
                {
                    bool _S4 = dir_0 == int(4);
                    bool _S5;
                    if(_S4)
                    {
                        _S5 = true;
                    }
                    else
                    {
                        _S5 = dir_0 == int(5);
                    }
                    if(_S5)
                    {
                        if(t_0 >= (W_0 + H_0 - int(1)))
                        {
                            return;
                        }
                        int diag_0 = t_0 - (H_0 - int(1));
                        if(diag_0 >= int(0))
                        {
                            length_0 = min(W_0 - diag_0, H_0);
                            k0_0 = diag_0;
                            y0_0 = int(0);
                        }
                        else
                        {
                            int _S6 = - diag_0;
                            length_0 = min(W_0, H_0 + diag_0);
                            k0_0 = int(0);
                            y0_0 = _S6;
                        }
                        if(_S4)
                        {
                            x0_0 = y0_0;
                            dx_0 = k0_0;
                            dy_0 = int(1);
                        }
                        else
                        {
                            int _S7 = length_0 - int(1);
                            int _S8 = k0_0 + _S7;
                            x0_0 = y0_0 + _S7;
                            dx_0 = _S8;
                            dy_0 = int(-1);
                        }
                        y0_0 = x0_0;
                        x0_0 = dx_0;
                        dx_0 = dy_0;
                    }
                    else
                    {
                        int _S9 = W_0 + H_0 - int(1);
                        if(t_0 >= _S9)
                        {
                            return;
                        }
                        if(t_0 < W_0)
                        {
                            length_0 = t_0;
                            k0_0 = int(0);
                        }
                        else
                        {
                            int _S10 = W_0 - int(1);
                            int _S11 = t_0 - _S10;
                            length_0 = _S10;
                            k0_0 = _S11;
                        }
                        int _S12 = min(min(t_0 + int(1), _S9 - t_0), min(W_0, H_0));
                        if(dir_0 == int(6))
                        {
                            y0_0 = k0_0;
                            x0_0 = length_0;
                            dx_0 = int(-1);
                            dy_0 = int(1);
                        }
                        else
                        {
                            int _S13 = _S12 - int(1);
                            int _S14 = length_0 - _S13;
                            y0_0 = k0_0 + _S13;
                            x0_0 = _S14;
                            dx_0 = int(1);
                            dy_0 = int(-1);
                        }
                        length_0 = _S12;
                    }
                }
            }
        }
    }
    int _S15 = min(K_0, int(64));
    thread array<int, int(64)> L_prev_0;
    thread array<int, int(64)> L_curr_0;
    int _S16 = y0_0 * W_0 + x0_0;
    k0_0 = int(0);
    for(;;)
    {
        if(k0_0 < int(64))
        {
        }
        else
        {
            break;
        }
        if(k0_0 < _S15)
        {
            int _S17 = _S16 * K_0 + k0_0;
            int c0_0 = (&kernelContext_0)->cost_volume_0[_S17];
            L_prev_0[k0_0] = c0_0;
            *((&kernelContext_0)->L_out_0+_S17) = c0_0;
        }
        k0_0 = k0_0 + int(1);
    }
    int x_0 = x0_0;
    int y_0 = y0_0;
    int step_0 = int(1);
    for(;;)
    {
        if(step_0 < length_0)
        {
        }
        else
        {
            break;
        }
        int x_1 = x_0 + dx_0;
        int y_1 = y_0 + dy_0;
        int _S18 = y_1 * W_0 + x_1;
        int minLprev_0 = L_prev_0[int(0)];
        int k_0 = int(1);
        for(;;)
        {
            if(k_0 < _S15)
            {
            }
            else
            {
                break;
            }
            if((L_prev_0[k_0]) < minLprev_0)
            {
                minLprev_0 = L_prev_0[k_0];
            }
            k_0 = k_0 + int(1);
        }
        int k_1 = int(0);
        for(;;)
        {
            if(k_1 < _S15)
            {
            }
            else
            {
                break;
            }
            int _S19 = _S18 * K_0 + k_1;
            int c_0 = (&kernelContext_0)->cost_volume_0[_S19];
            int term1_0 = L_prev_0[k_1];
            if(k_1 > int(0))
            {
                term2_0 = L_prev_0[k_1 - int(1)] + _S1;
            }
            else
            {
                term2_0 = int(1000000);
            }
            int term3_0;
            if(k_1 < (_S15 - int(1)))
            {
                term3_0 = L_prev_0[k_1 + int(1)] + _S1;
            }
            else
            {
                term3_0 = int(1000000);
            }
            int lv_0 = c_0 + min(min(term1_0, term2_0), min(term3_0, minLprev_0 + _S2)) - minLprev_0;
            L_curr_0[k_1] = lv_0;
            *((&kernelContext_0)->L_out_0+_S19) = lv_0;
            k_1 = k_1 + int(1);
        }
        term2_0 = int(0);
        for(;;)
        {
            if(term2_0 < _S15)
            {
            }
            else
            {
                break;
            }
            L_prev_0[term2_0] = L_curr_0[term2_0];
            term2_0 = term2_0 + int(1);
        }
        int _S20 = step_0 + int(1);
        x_0 = x_1;
        y_0 = y_1;
        step_0 = _S20;
    }
    return;
}

