#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
float ellipse_rr_0(float opac_0)
{
    float _S1 = min(3.32999992370605469, sqrt(2.0 * log(opac_0 * 63.75)));
    return _S1 * _S1;
}

float2 ellipse_x_range_0(float a_0, float b_0, float c_0, float rr_0, float y0_0, float y1_0)
{
    float det_0 = a_0 * c_0 - b_0 * b_0;
    float _S2 = - b_0;
    float slope_0 = _S2 / a_0;
    float hwMax_0 = sqrt(rr_0 / a_0);
    if(det_0 > (0.00100000004749745 * a_0 * c_0))
    {
        float ymax_0 = sqrt(rr_0 * a_0 / det_0);
        bool _S3;
        if(y0_0 > ymax_0)
        {
            _S3 = true;
        }
        else
        {
            _S3 = y1_0 < (- ymax_0);
        }
        if(_S3)
        {
            return float2(1.0, -1.0);
        }
        float ym_0 = _S2 / c_0 * sqrt(rr_0 * c_0 / det_0);
        float v0_0 = clamp(- ym_0, y0_0, y1_0);
        float v1_0 = clamp(ym_0, y0_0, y1_0);
        float _S4 = a_0 * rr_0;
        float pad_0 = 0.00100000004749745 * hwMax_0;
        return float2(slope_0 * v0_0 - sqrt(max(0.0, _S4 - det_0 * v0_0 * v0_0)) / a_0 - pad_0, slope_0 * v1_0 + sqrt(max(0.0, _S4 - det_0 * v1_0 * v1_0)) / a_0 + pad_0);
    }
    float m0_0 = slope_0 * y0_0;
    float m1_0 = slope_0 * y1_0;
    return float2(min(m0_0, m1_0) - hwMax_0, max(m0_0, m1_0) + hwMax_0);
}

int2 scanline_tile_range_0(int line_0, bool rowMajor_0, float px_0, float py_0, float ca_0, float cb_0, float cc_0, float rr_1, int rx_0, int ry_0, int tileSize_0, int xMin_0, int yMin_0, int xMax_0, int yMax_0)
{
    float u_0;
    if(rowMajor_0)
    {
        u_0 = px_0;
    }
    else
    {
        u_0 = py_0;
    }
    float w_0;
    if(rowMajor_0)
    {
        w_0 = py_0;
    }
    else
    {
        w_0 = px_0;
    }
    float qa_0;
    if(rowMajor_0)
    {
        qa_0 = ca_0;
    }
    else
    {
        qa_0 = cc_0;
    }
    float qc_0;
    if(rowMajor_0)
    {
        qc_0 = cc_0;
    }
    else
    {
        qc_0 = ca_0;
    }
    int lo_0;
    if(rowMajor_0)
    {
        lo_0 = ry_0;
    }
    else
    {
        lo_0 = rx_0;
    }
    float wLim_0 = float(lo_0);
    if(rowMajor_0)
    {
        lo_0 = xMin_0;
    }
    else
    {
        lo_0 = yMin_0;
    }
    int hi_0;
    if(rowMajor_0)
    {
        hi_0 = xMax_0;
    }
    else
    {
        hi_0 = yMax_0;
    }
    float t_0 = float(tileSize_0);
    float _S5 = float(line_0) * t_0;
    float2 r_0 = ellipse_x_range_0(qa_0, cb_0, qc_0, rr_1, max(_S5 - w_0, - wLim_0), min(_S5 + t_0 - w_0, wLim_0));
    float _S6 = r_0.x;
    float _S7 = r_0.y;
    if(!(_S6 <= _S7))
    {
        return int2(int(1), int(0));
    }
    return int2(max(int(floor((_S6 + u_0) / t_0)), lo_0), min(int(floor((_S7 + u_0) / t_0)), hi_0));
}

int quadrant_mask_0(float px_1, float py_1, int rx_1, int ry_1, float ca_1, float cb_1, float cc_1, float opac_1, int tx_0, int ty_0, int numTilesX8_0, int numTilesY8_0)
{
    float _S8 = ellipse_rr_0(opac_1);
    float _S9 = float(rx_1);
    int _S10 = max(int(0), int(floor((px_1 - _S9) / 8.0)));
    int _S11 = min(numTilesX8_0 - int(1), int(floor((px_1 + _S9) / 8.0)));
    float _S12 = float(ry_1);
    int _S13 = max(int(0), int(floor((py_1 - _S12) / 8.0)));
    int _S14 = min(numTilesY8_0 - int(1), int(floor((py_1 + _S12) / 8.0)));
    int qy_0 = int(0);
    int mask_0 = int(0);
    for(;;)
    {
        if(qy_0 < int(2))
        {
        }
        else
        {
            break;
        }
        int row_0 = ty_0 * int(2) + qy_0;
        bool _S15;
        if(row_0 < _S13)
        {
            _S15 = true;
        }
        else
        {
            _S15 = row_0 > _S14;
        }
        int mask_1;
        if(_S15)
        {
            mask_1 = mask_0;
            qy_0 = qy_0 + int(1);
            mask_0 = mask_1;
            continue;
        }
        int2 _S16 = scanline_tile_range_0(row_0, true, px_1, py_1, ca_1, cb_1, cc_1, _S8, rx_1, ry_1, int(8), _S10, _S13, _S11, _S14);
        int qx_0 = int(0);
        mask_1 = mask_0;
        for(;;)
        {
            if(qx_0 < int(2))
            {
            }
            else
            {
                break;
            }
            int col_0 = tx_0 * int(2) + qx_0;
            bool _S17;
            if(col_0 >= (_S16.x))
            {
                _S17 = col_0 <= (_S16.y);
            }
            else
            {
                _S17 = false;
            }
            if(_S17)
            {
                mask_1 = mask_1 | (int(1) << uint(qy_0 * int(2) + qx_0));
            }
            qx_0 = qx_0 + int(1);
        }
        qy_0 = qy_0 + int(1);
        mask_0 = mask_1;
    }
    return mask_0;
}

struct KernelContext_0
{
    int device* sc_dims_0;
    int device* sc_offsets_0;
    int device* sc_flatten_0;
    float device* sc_means2d_0;
    int device* sc_radii_0;
    float device* sc_conics_0;
    float device* sc_opacities_0;
    int device* sc_counts_0;
    uint _S18;
    array<int, int(32)> threadgroup* sc_partial_0;
};

[[kernel]] void tile_split_count(uint3 sv_groupthreadid_0 [[thread_position_in_threadgroup]], uint3 gid_0 [[threadgroup_position_in_grid]], uint _builtinWaveLaneIndex_0 [[thread_index_in_simdgroup]], int device* sc_dims_1 [[buffer(30)]], int device* sc_offsets_1 [[buffer(25)]], int device* sc_flatten_1 [[buffer(24)]], float device* sc_means2d_1 [[buffer(26)]], int device* sc_radii_1 [[buffer(27)]], float device* sc_conics_1 [[buffer(28)]], float device* sc_opacities_1 [[buffer(29)]], int device* sc_counts_1 [[buffer(31)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->sc_dims_0 = sc_dims_1;
    (&kernelContext_0)->sc_offsets_0 = sc_offsets_1;
    (&kernelContext_0)->sc_flatten_0 = sc_flatten_1;
    (&kernelContext_0)->sc_means2d_0 = sc_means2d_1;
    (&kernelContext_0)->sc_radii_0 = sc_radii_1;
    (&kernelContext_0)->sc_conics_0 = sc_conics_1;
    (&kernelContext_0)->sc_opacities_0 = sc_opacities_1;
    (&kernelContext_0)->sc_counts_0 = sc_counts_1;
    threadgroup array<int, int(32)> sc_partial_1;
    (&kernelContext_0)->sc_partial_0 = &sc_partial_1;
    uint sv_groupindex_0 = (sv_groupthreadid_0[int(2)] + sv_groupthreadid_0[int(1)]) * 256U + sv_groupthreadid_0[int(0)];
    (&kernelContext_0)->_S18 = _builtinWaveLaneIndex_0;
    int t_1 = int(gid_0.x);
    int numTilesX16_0 = sc_dims_1[int(0)];
    int _S19 = t_1 % numTilesX16_0;
    int _S20 = t_1 / numTilesX16_0;
    int _S21 = (&kernelContext_0)->sc_offsets_0[t_1 + int(1)];
    int _S22 = int(sv_groupindex_0);
    int k_0 = (&kernelContext_0)->sc_offsets_0[t_1] + _S22;
    int c0_0 = int(0);
    int c1_0 = int(0);
    int c2_0 = int(0);
    int c3_0 = int(0);
    for(;;)
    {
        if(k_0 < _S21)
        {
        }
        else
        {
            break;
        }
        int gi_0 = (&kernelContext_0)->sc_flatten_0[k_0];
        int _S23 = gi_0 * int(2);
        int _S24 = _S23 + int(1);
        int _S25 = gi_0 * int(3);
        int m_0 = quadrant_mask_0((&kernelContext_0)->sc_means2d_0[_S23], (&kernelContext_0)->sc_means2d_0[_S24], (&kernelContext_0)->sc_radii_0[_S23], (&kernelContext_0)->sc_radii_0[_S24], (&kernelContext_0)->sc_conics_0[_S25], (&kernelContext_0)->sc_conics_0[_S25 + int(1)], (&kernelContext_0)->sc_conics_0[_S25 + int(2)], (&kernelContext_0)->sc_opacities_0[gi_0], _S19, _S20, (&kernelContext_0)->sc_dims_0[int(1)], (&kernelContext_0)->sc_dims_0[int(2)]);
        int c0_1 = c0_0 + (m_0 & int(1));
        int c1_1 = c1_0 + ((m_0 >> 1U) & int(1));
        int c2_1 = c2_0 + ((m_0 >> 2U) & int(1));
        int c3_1 = c3_0 + ((m_0 >> 3U) & int(1));
        k_0 = k_0 + int(256);
        c0_0 = c0_1;
        c1_0 = c1_1;
        c2_0 = c2_1;
        c3_0 = c3_1;
    }
    int c0_2 = simd_sum(c0_0);
    int c1_2 = simd_sum(c1_0);
    int c2_2 = simd_sum(c2_0);
    int c3_2 = simd_sum(c3_0);
    uint simd_0 = sv_groupindex_0 / 32U;
    if(((&kernelContext_0)->_S18) == 0U)
    {
        uint _S26 = simd_0 * 4U;
        (*(&kernelContext_0)->sc_partial_0)[_S26] = c0_2;
        (*(&kernelContext_0)->sc_partial_0)[_S26 + 1U] = c1_2;
        (*(&kernelContext_0)->sc_partial_0)[_S26 + 2U] = c2_2;
        (*(&kernelContext_0)->sc_partial_0)[_S26 + 3U] = c3_2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sv_groupindex_0 < 4U)
    {
        int s_0 = int(0);
        int sum_0 = int(0);
        for(;;)
        {
            if(s_0 < int(8))
            {
            }
            else
            {
                break;
            }
            int sum_1 = sum_0 + (*(&kernelContext_0)->sc_partial_0)[s_0 * int(4) + _S22];
            s_0 = s_0 + int(1);
            sum_0 = sum_1;
        }
        *((&kernelContext_0)->sc_counts_0+(t_1 * int(4) + _S22)) = sum_0;
    }
    return;
}

