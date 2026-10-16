#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* targets_0;
    int device* frame_dims_0;
    float device* depths_0;
    float device* out_depth_0;
    int device* params_0;
    float device* rel_tol_buf_0;
    float device* frame_info_0;
    int device* neighbors_0;
};

[[kernel]] void mvs_fusion_consistency(uint3 tid_0 [[thread_position_in_grid]], int device* targets_1 [[buffer(3)]], int device* frame_dims_1 [[buffer(2)]], float device* depths_1 [[buffer(0)]], float device* out_depth_1 [[buffer(7)]], int device* params_1 [[buffer(5)]], float device* rel_tol_buf_1 [[buffer(6)]], float device* frame_info_1 [[buffer(1)]], int device* neighbors_1 [[buffer(4)]])
{
    float depthSum_0;
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->targets_0 = targets_1;
    (&kernelContext_0)->frame_dims_0 = frame_dims_1;
    (&kernelContext_0)->depths_0 = depths_1;
    (&kernelContext_0)->out_depth_0 = out_depth_1;
    (&kernelContext_0)->params_0 = params_1;
    (&kernelContext_0)->rel_tol_buf_0 = rel_tol_buf_1;
    (&kernelContext_0)->frame_info_0 = frame_info_1;
    (&kernelContext_0)->neighbors_0 = neighbors_1;
    int t_0 = int(tid_0.z);
    int _S1 = t_0 * int(2);
    int ts_0 = targets_1[_S1];
    int outBase_0 = targets_1[_S1 + int(1)];
    int _S2 = ts_0 * int(3);
    int W_0 = frame_dims_1[_S2];
    int H_0 = frame_dims_1[_S2 + int(1)];
    int x_0 = int(tid_0.x);
    int y_0 = int(tid_0.y);
    bool _S3;
    if(x_0 >= W_0)
    {
        _S3 = true;
    }
    else
    {
        _S3 = y_0 >= H_0;
    }
    if(_S3)
    {
        return;
    }
    int pix_0 = y_0 * W_0 + x_0;
    float d_0 = (&kernelContext_0)->depths_0[(&kernelContext_0)->frame_dims_0[_S2 + int(2)] + pix_0];
    if(d_0 <= 0.0)
    {
        *((&kernelContext_0)->out_depth_0+(outBase_0 + pix_0)) = -1.0;
        return;
    }
    int _S4 = (&kernelContext_0)->params_0[int(0)];
    int vMin_0 = (&kernelContext_0)->params_0[int(1)];
    bool averageDepth_0 = (&kernelContext_0)->params_0[int(2)] != int(0);
    float _S5 = (&kernelContext_0)->rel_tol_buf_0[int(0)];
    int ti_0 = ts_0 * int(16);
    float dx_0 = (float(x_0) + 0.5 - (&kernelContext_0)->frame_info_0[ti_0 + int(14)]) / (&kernelContext_0)->frame_info_0[ti_0 + int(12)] * d_0 - (&kernelContext_0)->frame_info_0[ti_0 + int(3)];
    float dy_0 = (float(y_0) + 0.5 - (&kernelContext_0)->frame_info_0[ti_0 + int(15)]) / (&kernelContext_0)->frame_info_0[ti_0 + int(13)] * d_0 - (&kernelContext_0)->frame_info_0[ti_0 + int(7)];
    int _S6 = ti_0 + int(11);
    float dz_0 = d_0 - (&kernelContext_0)->frame_info_0[_S6];
    int _S7 = ti_0 + int(8);
    float _S8 = (&kernelContext_0)->frame_info_0[ti_0] * dx_0 + (&kernelContext_0)->frame_info_0[ti_0 + int(4)] * dy_0 + (&kernelContext_0)->frame_info_0[_S7] * dz_0;
    int _S9 = ti_0 + int(9);
    float _S10 = (&kernelContext_0)->frame_info_0[ti_0 + int(1)] * dx_0 + (&kernelContext_0)->frame_info_0[ti_0 + int(5)] * dy_0 + (&kernelContext_0)->frame_info_0[_S9] * dz_0;
    int _S11 = ti_0 + int(10);
    float _S12 = (&kernelContext_0)->frame_info_0[ti_0 + int(2)] * dx_0 + (&kernelContext_0)->frame_info_0[ti_0 + int(6)] * dy_0 + (&kernelContext_0)->frame_info_0[_S11] * dz_0;
    int k_0 = int(0);
    int conf_0 = int(0);
    float depthSum_1 = d_0;
    for(;;)
    {
        if(k_0 < _S4)
        {
        }
        else
        {
            break;
        }
        int ns_0 = (&kernelContext_0)->neighbors_0[t_0 * _S4 + k_0];
        if(ns_0 < int(0))
        {
            depthSum_0 = depthSum_1;
            k_0 = k_0 + int(1);
            depthSum_1 = depthSum_0;
            continue;
        }
        int ni_0 = ns_0 * int(16);
        int _S13 = ni_0 + int(8);
        int _S14 = ni_0 + int(9);
        int _S15 = ni_0 + int(10);
        int _S16 = ni_0 + int(11);
        float cz_0 = (&kernelContext_0)->frame_info_0[_S13] * _S8 + (&kernelContext_0)->frame_info_0[_S14] * _S10 + (&kernelContext_0)->frame_info_0[_S15] * _S12 + (&kernelContext_0)->frame_info_0[_S16];
        if(cz_0 <= 9.99999997475242708e-07)
        {
            depthSum_0 = depthSum_1;
            k_0 = k_0 + int(1);
            depthSum_1 = depthSum_0;
            continue;
        }
        int _S17 = ni_0 + int(1);
        int _S18 = ni_0 + int(2);
        int _S19 = ni_0 + int(3);
        int _S20 = ni_0 + int(4);
        int _S21 = ni_0 + int(5);
        int _S22 = ni_0 + int(6);
        int _S23 = ni_0 + int(7);
        float nfx_0 = (&kernelContext_0)->frame_info_0[ni_0 + int(12)];
        float nfy_0 = (&kernelContext_0)->frame_info_0[ni_0 + int(13)];
        float ncx_0 = (&kernelContext_0)->frame_info_0[ni_0 + int(14)];
        float ncy_0 = (&kernelContext_0)->frame_info_0[ni_0 + int(15)];
        float fu_0 = floor(nfx_0 * ((&kernelContext_0)->frame_info_0[ni_0] * _S8 + (&kernelContext_0)->frame_info_0[_S17] * _S10 + (&kernelContext_0)->frame_info_0[_S18] * _S12 + (&kernelContext_0)->frame_info_0[_S19]) / cz_0 + ncx_0);
        float fv_0 = floor(nfy_0 * ((&kernelContext_0)->frame_info_0[_S20] * _S8 + (&kernelContext_0)->frame_info_0[_S21] * _S10 + (&kernelContext_0)->frame_info_0[_S22] * _S12 + (&kernelContext_0)->frame_info_0[_S23]) / cz_0 + ncy_0);
        int _S24 = ns_0 * int(3);
        int nw_0 = (&kernelContext_0)->frame_dims_0[_S24];
        int nh_0 = (&kernelContext_0)->frame_dims_0[_S24 + int(1)];
        if(fu_0 >= 0.0)
        {
            _S3 = fu_0 < float(nw_0);
        }
        else
        {
            _S3 = false;
        }
        bool _S25;
        if(_S3)
        {
            _S25 = fv_0 >= 0.0;
        }
        else
        {
            _S25 = false;
        }
        bool _S26;
        if(_S25)
        {
            _S26 = fv_0 < float(nh_0);
        }
        else
        {
            _S26 = false;
        }
        if(!_S26)
        {
            depthSum_0 = depthSum_1;
            k_0 = k_0 + int(1);
            depthSum_1 = depthSum_0;
            continue;
        }
        int ux_0 = int(fu_0);
        int vy_0 = int(fv_0);
        float dj_0 = (&kernelContext_0)->depths_0[(&kernelContext_0)->frame_dims_0[_S24 + int(2)] + vy_0 * nw_0 + ux_0];
        if(dj_0 <= 0.0)
        {
            depthSum_0 = depthSum_1;
            k_0 = k_0 + int(1);
            depthSum_1 = depthSum_0;
            continue;
        }
        int conf_1;
        if((abs(cz_0 - dj_0) / cz_0) < _S5)
        {
            int conf_2 = conf_0 + int(1);
            if(averageDepth_0)
            {
                float ex_0 = (float(ux_0) + 0.5 - ncx_0) / nfx_0 * dj_0 - (&kernelContext_0)->frame_info_0[_S19];
                float ey_0 = (float(vy_0) + 0.5 - ncy_0) / nfy_0 * dj_0 - (&kernelContext_0)->frame_info_0[_S23];
                float ez_0 = dj_0 - (&kernelContext_0)->frame_info_0[_S16];
                depthSum_0 = depthSum_1 + ((&kernelContext_0)->frame_info_0[_S7] * ((&kernelContext_0)->frame_info_0[ni_0] * ex_0 + (&kernelContext_0)->frame_info_0[_S20] * ey_0 + (&kernelContext_0)->frame_info_0[_S13] * ez_0) + (&kernelContext_0)->frame_info_0[_S9] * ((&kernelContext_0)->frame_info_0[_S17] * ex_0 + (&kernelContext_0)->frame_info_0[_S21] * ey_0 + (&kernelContext_0)->frame_info_0[_S14] * ez_0) + (&kernelContext_0)->frame_info_0[_S11] * ((&kernelContext_0)->frame_info_0[_S18] * ex_0 + (&kernelContext_0)->frame_info_0[_S22] * ey_0 + (&kernelContext_0)->frame_info_0[_S15] * ez_0) + (&kernelContext_0)->frame_info_0[_S6]);
            }
            else
            {
                depthSum_0 = depthSum_1;
            }
            conf_1 = conf_2;
        }
        else
        {
            conf_1 = conf_0;
            depthSum_0 = depthSum_1;
        }
        conf_0 = conf_1;
        k_0 = k_0 + int(1);
        depthSum_1 = depthSum_0;
    }
    if(conf_0 >= vMin_0)
    {
        if(averageDepth_0)
        {
            depthSum_1 = depthSum_1 / float(conf_0 + int(1));
        }
        else
        {
            depthSum_1 = d_0;
        }
        depthSum_0 = depthSum_1;
    }
    else
    {
        depthSum_0 = -1.0;
    }
    *((&kernelContext_0)->out_depth_0+(outBase_0 + pix_0)) = depthSum_0;
    return;
}

