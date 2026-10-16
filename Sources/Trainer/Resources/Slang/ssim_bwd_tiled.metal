#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* dims_0;
    float device* dL_dmap_0;
    half device* dm_dmu1_0;
    half device* dm_dsigma1_sq_0;
    half device* dm_dsigma12_0;
    float device* gauss_0;
    float device* img1_0;
    float device* img2_0;
    float device* dL_dimg1_0;
    float device* lossParams_0;
    array<float, int(676)> threadgroup* sb_P0_0;
    array<float, int(676)> threadgroup* sb_P1_0;
    array<float, int(676)> threadgroup* sb_P2_0;
    array<float, int(416)> threadgroup* sb_h0_0;
    array<float, int(416)> threadgroup* sb_h1_0;
    array<float, int(416)> threadgroup* sb_h2_0;
};

[[kernel]] void ssim_bwd_tiled(uint3 gid_0 [[thread_position_in_grid]], uint3 ltid_0 [[thread_position_in_threadgroup]], uint3 grp_0 [[threadgroup_position_in_grid]], int device* dims_1 [[buffer(7)]], float device* dL_dmap_1 [[buffer(2)]], half device* dm_dmu1_1 [[buffer(3)]], half device* dm_dsigma1_sq_1 [[buffer(4)]], half device* dm_dsigma12_1 [[buffer(5)]], float device* gauss_1 [[buffer(6)]], float device* img1_1 [[buffer(0)]], float device* img2_1 [[buffer(1)]], float device* dL_dimg1_1 [[buffer(9)]], float device* lossParams_1 [[buffer(8)]])
{
    bool _S1;
    float dL_0;
    float a1_0;
    float a2_0;
    int k_0;
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->dims_0 = dims_1;
    (&kernelContext_0)->dL_dmap_0 = dL_dmap_1;
    (&kernelContext_0)->dm_dmu1_0 = dm_dmu1_1;
    (&kernelContext_0)->dm_dsigma1_sq_0 = dm_dsigma1_sq_1;
    (&kernelContext_0)->dm_dsigma12_0 = dm_dsigma12_1;
    (&kernelContext_0)->gauss_0 = gauss_1;
    (&kernelContext_0)->img1_0 = img1_1;
    (&kernelContext_0)->img2_0 = img2_1;
    (&kernelContext_0)->dL_dimg1_0 = dL_dimg1_1;
    (&kernelContext_0)->lossParams_0 = lossParams_1;
    threadgroup array<float, int(676)> sb_P0_1;
    (&kernelContext_0)->sb_P0_0 = &sb_P0_1;
    threadgroup array<float, int(676)> sb_P1_1;
    (&kernelContext_0)->sb_P1_0 = &sb_P1_1;
    threadgroup array<float, int(676)> sb_P2_1;
    (&kernelContext_0)->sb_P2_0 = &sb_P2_1;
    threadgroup array<float, int(416)> sb_h0_1;
    (&kernelContext_0)->sb_h0_0 = &sb_h0_1;
    threadgroup array<float, int(416)> sb_h1_1;
    (&kernelContext_0)->sb_h1_0 = &sb_h1_1;
    threadgroup array<float, int(416)> sb_h2_1;
    (&kernelContext_0)->sb_h2_0 = &sb_h2_1;
    int _S2 = dims_1[int(0)];
    int _S3 = dims_1[int(1)];
    int _S4 = dims_1[int(2)];
    int rad_0 = dims_1[int(3)];
    int lx_0 = int(ltid_0.x);
    int ly_0 = int(ltid_0.y);
    int _S5 = ly_0 * int(16) + lx_0;
    int _S6 = int(grp_0.x) * int(16);
    int _S7 = int(grp_0.y) * int(16);
    int _S8 = int(gid_0.x);
    int _S9 = int(gid_0.y);
    int _S10 = int(2) * rad_0 + int(1);
    int c_0 = int(0);
    for(;;)
    {
        if(c_0 < _S4)
        {
        }
        else
        {
            break;
        }
        int i_0 = _S5;
        for(;;)
        {
            if(i_0 < int(676))
            {
            }
            else
            {
                break;
            }
            int hx_0 = _S6 - rad_0 + i_0 % int(26);
            int hy_0 = _S7 - rad_0 + i_0 / int(26);
            if(hx_0 >= int(0))
            {
                _S1 = hx_0 < _S3;
            }
            else
            {
                _S1 = false;
            }
            bool _S11;
            if(_S1)
            {
                _S11 = hy_0 >= int(0);
            }
            else
            {
                _S11 = false;
            }
            bool inb_0;
            if(_S11)
            {
                inb_0 = hy_0 < _S2;
            }
            else
            {
                inb_0 = false;
            }
            int gidx_0 = (hy_0 * _S3 + hx_0) * _S4 + c_0;
            if(inb_0)
            {
                dL_0 = (&kernelContext_0)->dL_dmap_0[gidx_0];
            }
            else
            {
                dL_0 = 0.0;
            }
            if(inb_0)
            {
                a1_0 = float((&kernelContext_0)->dm_dmu1_0[gidx_0]) * dL_0;
            }
            else
            {
                a1_0 = 0.0;
            }
            (*(&kernelContext_0)->sb_P0_0)[i_0] = a1_0;
            if(inb_0)
            {
                a2_0 = float((&kernelContext_0)->dm_dsigma1_sq_0[gidx_0]) * dL_0;
            }
            else
            {
                a2_0 = 0.0;
            }
            (*(&kernelContext_0)->sb_P1_0)[i_0] = a2_0;
            float _S12;
            if(inb_0)
            {
                _S12 = float((&kernelContext_0)->dm_dsigma12_0[gidx_0]) * dL_0;
            }
            else
            {
                _S12 = 0.0;
            }
            (*(&kernelContext_0)->sb_P2_0)[i_0] = _S12;
            i_0 = i_0 + int(256);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int j_0 = _S5;
        for(;;)
        {
            if(j_0 < int(416))
            {
            }
            else
            {
                break;
            }
            int _S13 = j_0 / int(16) * int(26) + j_0 % int(16);
            k_0 = int(0);
            dL_0 = 0.0;
            a1_0 = 0.0;
            a2_0 = 0.0;
            for(;;)
            {
                if(k_0 < _S10)
                {
                }
                else
                {
                    break;
                }
                float w_0 = (&kernelContext_0)->gauss_0[k_0];
                int _S14 = _S13 + k_0;
                float a0_0 = dL_0 + w_0 * (*(&kernelContext_0)->sb_P0_0)[_S14];
                float a1_1 = a1_0 + w_0 * (*(&kernelContext_0)->sb_P1_0)[_S14];
                float a2_1 = a2_0 + w_0 * (*(&kernelContext_0)->sb_P2_0)[_S14];
                k_0 = k_0 + int(1);
                dL_0 = a0_0;
                a1_0 = a1_1;
                a2_0 = a2_1;
            }
            (*(&kernelContext_0)->sb_h0_0)[j_0] = dL_0;
            (*(&kernelContext_0)->sb_h1_0)[j_0] = a1_0;
            (*(&kernelContext_0)->sb_h2_0)[j_0] = a2_0;
            j_0 = j_0 + int(256);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S8 < _S3)
        {
            _S1 = _S9 < _S2;
        }
        else
        {
            _S1 = false;
        }
        if(_S1)
        {
            k_0 = int(0);
            dL_0 = 0.0;
            a1_0 = 0.0;
            a2_0 = 0.0;
            for(;;)
            {
                if(k_0 < _S10)
                {
                }
                else
                {
                    break;
                }
                float w_1 = (&kernelContext_0)->gauss_0[k_0];
                int pidx_0 = (ly_0 + k_0) * int(16) + lx_0;
                float s0_0 = dL_0 + w_1 * (*(&kernelContext_0)->sb_h0_0)[pidx_0];
                float s1_0 = a1_0 + w_1 * (*(&kernelContext_0)->sb_h1_0)[pidx_0];
                float s2_0 = a2_0 + w_1 * (*(&kernelContext_0)->sb_h2_0)[pidx_0];
                k_0 = k_0 + int(1);
                dL_0 = s0_0;
                a1_0 = s1_0;
                a2_0 = s2_0;
            }
            int o_0 = (_S9 * _S3 + _S8) * _S4 + c_0;
            float p1_0 = (&kernelContext_0)->img1_0[o_0];
            float p2_0 = (&kernelContext_0)->img2_0[o_0];
            *((&kernelContext_0)->dL_dimg1_0+o_0) = dL_0 + 2.0 * p1_0 * a1_0 + p2_0 * a2_0 + (&kernelContext_0)->lossParams_0[int(0)] * float((int(sign((p1_0 - p2_0))))) * (&kernelContext_0)->lossParams_0[int(1)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        c_0 = c_0 + int(1);
    }
    return;
}

