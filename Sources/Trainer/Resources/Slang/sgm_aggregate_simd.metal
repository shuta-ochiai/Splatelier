#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* dims_0;
    int device* cost_volume_0;
    int device* L_out_0;
    uint _S1;
};

[[kernel]] void sgm_aggregate_simd(uint3 tid_0 [[thread_position_in_grid]], uint _builtinWaveLaneIndex_0 [[thread_index_in_simdgroup]], int device* dims_1 [[buffer(1)]], int device* cost_volume_1 [[buffer(0)]], int device* L_out_1 [[buffer(2)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->dims_0 = dims_1;
    (&kernelContext_0)->cost_volume_0 = cost_volume_1;
    (&kernelContext_0)->L_out_0 = L_out_1;
    (&kernelContext_0)->_S1 = _builtinWaveLaneIndex_0;
    int W_0 = dims_1[int(0)];
    int H_0 = dims_1[int(1)];
    int K_0 = dims_1[int(2)];
    int _S2 = dims_1[int(3)];
    int _S3 = dims_1[int(4)];
    int dir_0 = dims_1[int(5)];
    int t_0 = int(tid_0.x) / int(32);
    int lane_0 = int(_builtinWaveLaneIndex_0);
    int length_0;
    int Lp0_0;
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
            int _S4 = W_0 - int(1);
            y0_0 = t_0;
            x0_0 = _S4;
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
                    bool _S5 = dir_0 == int(4);
                    bool _S6;
                    if(_S5)
                    {
                        _S6 = true;
                    }
                    else
                    {
                        _S6 = dir_0 == int(5);
                    }
                    if(_S6)
                    {
                        if(t_0 >= (W_0 + H_0 - int(1)))
                        {
                            return;
                        }
                        int diag_0 = t_0 - (H_0 - int(1));
                        if(diag_0 >= int(0))
                        {
                            length_0 = min(W_0 - diag_0, H_0);
                            Lp0_0 = diag_0;
                            y0_0 = int(0);
                        }
                        else
                        {
                            int _S7 = - diag_0;
                            length_0 = min(W_0, H_0 + diag_0);
                            Lp0_0 = int(0);
                            y0_0 = _S7;
                        }
                        if(_S5)
                        {
                            x0_0 = y0_0;
                            dx_0 = Lp0_0;
                            dy_0 = int(1);
                        }
                        else
                        {
                            int _S8 = length_0 - int(1);
                            int _S9 = Lp0_0 + _S8;
                            x0_0 = y0_0 + _S8;
                            dx_0 = _S9;
                            dy_0 = int(-1);
                        }
                        y0_0 = x0_0;
                        x0_0 = dx_0;
                        dx_0 = dy_0;
                    }
                    else
                    {
                        int _S10 = W_0 + H_0 - int(1);
                        if(t_0 >= _S10)
                        {
                            return;
                        }
                        if(t_0 < W_0)
                        {
                            length_0 = t_0;
                            Lp0_0 = int(0);
                        }
                        else
                        {
                            int _S11 = W_0 - int(1);
                            int _S12 = t_0 - _S11;
                            length_0 = _S11;
                            Lp0_0 = _S12;
                        }
                        int _S13 = min(min(t_0 + int(1), _S10 - t_0), min(W_0, H_0));
                        if(dir_0 == int(6))
                        {
                            y0_0 = Lp0_0;
                            x0_0 = length_0;
                            dx_0 = int(-1);
                            dy_0 = int(1);
                        }
                        else
                        {
                            int _S14 = _S13 - int(1);
                            int _S15 = length_0 - _S14;
                            y0_0 = Lp0_0 + _S14;
                            x0_0 = _S15;
                            dx_0 = int(1);
                            dy_0 = int(-1);
                        }
                        length_0 = _S13;
                    }
                }
            }
        }
    }
    int d1_0 = lane_0 + int(32);
    bool v0_0 = lane_0 < K_0;
    bool v1_0 = d1_0 < K_0;
    int pix0_0 = y0_0 * W_0 + x0_0;
    if(v0_0)
    {
        int _S16 = pix0_0 * K_0 + lane_0;
        int Lp0_1 = (&kernelContext_0)->cost_volume_0[_S16];
        *((&kernelContext_0)->L_out_0+_S16) = Lp0_1;
        Lp0_0 = Lp0_1;
    }
    else
    {
        Lp0_0 = int(1000000);
    }
    int Lp1_0;
    if(v1_0)
    {
        int _S17 = pix0_0 * K_0 + d1_0;
        int Lp1_1 = (&kernelContext_0)->cost_volume_0[_S17];
        *((&kernelContext_0)->L_out_0+_S17) = Lp1_1;
        Lp1_0 = Lp1_1;
    }
    else
    {
        Lp1_0 = int(1000000);
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
        int pix_0 = y_1 * W_0 + x_1;
        int minPrev_0 = simd_min(min(Lp0_0, Lp1_0));
        int _S18 = max(lane_0 - int(1), int(0));
        int _S19 = min(lane_0 + int(1), int(31));
        int sh0m_0 = (simd_shuffle((Lp0_0), ushort((_S18))));
        int sh1m_0 = (simd_shuffle((Lp1_0), ushort((_S18))));
        int sh0p_0 = (simd_shuffle((Lp0_0), ushort((_S19))));
        int sh1p_0 = (simd_shuffle((Lp1_0), ushort((_S19))));
        int last0_0 = (simd_shuffle((Lp0_0), ushort((int(31)))));
        int first1_0 = (simd_shuffle((Lp1_0), ushort((int(0)))));
        bool _S20 = lane_0 > int(0);
        int left0_0;
        if(_S20)
        {
            left0_0 = sh0m_0;
        }
        else
        {
            left0_0 = int(1000000);
        }
        int left1_0;
        if(_S20)
        {
            left1_0 = sh1m_0;
        }
        else
        {
            left1_0 = last0_0;
        }
        bool _S21 = lane_0 < int(31);
        int right0_0;
        if(_S21)
        {
            right0_0 = sh0p_0;
        }
        else
        {
            right0_0 = first1_0;
        }
        int right1_0;
        if(_S21)
        {
            right1_0 = sh1p_0;
        }
        else
        {
            right1_0 = int(1000000);
        }
        int t4_0 = minPrev_0 + _S3;
        int Lc0_0;
        if(v0_0)
        {
            int _S22 = pix_0 * K_0 + lane_0;
            int Lc0_1 = (&kernelContext_0)->cost_volume_0[_S22] + min(min(Lp0_0, left0_0 + _S2), min(right0_0 + _S2, t4_0)) - minPrev_0;
            *((&kernelContext_0)->L_out_0+_S22) = Lc0_1;
            Lc0_0 = Lc0_1;
        }
        else
        {
            Lc0_0 = int(1000000);
        }
        int Lc1_0;
        if(v1_0)
        {
            int _S23 = pix_0 * K_0 + d1_0;
            int Lc1_1 = (&kernelContext_0)->cost_volume_0[_S23] + min(min(Lp1_0, left1_0 + _S2), min(right1_0 + _S2, t4_0)) - minPrev_0;
            *((&kernelContext_0)->L_out_0+_S23) = Lc1_1;
            Lc1_0 = Lc1_1;
        }
        else
        {
            Lc1_0 = int(1000000);
        }
        int _S24 = step_0 + int(1);
        x_0 = x_1;
        y_0 = y_1;
        Lp0_0 = Lc0_0;
        Lp1_0 = Lc1_0;
        step_0 = _S24;
    }
    return;
}

