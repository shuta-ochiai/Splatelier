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

struct KernelContext_0
{
    int device* sp_dims_0;
    int device* sp_flatten_0;
    int device* sp_tile_0;
    float device* sp_means2d_0;
    int device* sp_radii_0;
    float device* sp_conics_0;
    float device* sp_opacities_0;
    int device* sp_mask_0;
};

[[kernel]] void tile_split_mask(uint3 tid_0 [[thread_position_in_grid]], int device* sp_dims_1 [[buffer(30)]], int device* sp_flatten_1 [[buffer(24)]], int device* sp_tile_1 [[buffer(25)]], float device* sp_means2d_1 [[buffer(26)]], int device* sp_radii_1 [[buffer(27)]], float device* sp_conics_1 [[buffer(28)]], float device* sp_opacities_1 [[buffer(29)]], int device* sp_mask_1 [[buffer(31)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->sp_dims_0 = sp_dims_1;
    (&kernelContext_0)->sp_flatten_0 = sp_flatten_1;
    (&kernelContext_0)->sp_tile_0 = sp_tile_1;
    (&kernelContext_0)->sp_means2d_0 = sp_means2d_1;
    (&kernelContext_0)->sp_radii_0 = sp_radii_1;
    (&kernelContext_0)->sp_conics_0 = sp_conics_1;
    (&kernelContext_0)->sp_opacities_0 = sp_opacities_1;
    (&kernelContext_0)->sp_mask_0 = sp_mask_1;
    int k_0 = int(tid_0.x);
    if(k_0 >= sp_dims_1[int(1)])
    {
        return;
    }
    int numTilesX16_0 = (&kernelContext_0)->sp_dims_0[int(0)];
    int numTilesX8_0 = (&kernelContext_0)->sp_dims_0[int(2)];
    int numTilesY8_0 = (&kernelContext_0)->sp_dims_0[int(3)];
    int gi_0 = (&kernelContext_0)->sp_flatten_0[k_0];
    int t_1 = (&kernelContext_0)->sp_tile_0[k_0];
    int _S8 = t_1 % numTilesX16_0;
    int _S9 = t_1 / numTilesX16_0;
    int _S10 = gi_0 * int(2);
    float px_1 = (&kernelContext_0)->sp_means2d_0[_S10];
    int _S11 = _S10 + int(1);
    float py_1 = (&kernelContext_0)->sp_means2d_0[_S11];
    int rx_1 = (&kernelContext_0)->sp_radii_0[_S10];
    int ry_1 = (&kernelContext_0)->sp_radii_0[_S11];
    int _S12 = gi_0 * int(3);
    float _S13 = (&kernelContext_0)->sp_conics_0[_S12];
    float _S14 = (&kernelContext_0)->sp_conics_0[_S12 + int(1)];
    float _S15 = (&kernelContext_0)->sp_conics_0[_S12 + int(2)];
    float _S16 = ellipse_rr_0((&kernelContext_0)->sp_opacities_0[gi_0]);
    float _S17 = float(rx_1);
    int _S18 = max(int(0), int(floor((px_1 - _S17) / 8.0)));
    int _S19 = min(numTilesX8_0 - int(1), int(floor((px_1 + _S17) / 8.0)));
    float _S20 = float(ry_1);
    int _S21 = max(int(0), int(floor((py_1 - _S20) / 8.0)));
    int _S22 = min(numTilesY8_0 - int(1), int(floor((py_1 + _S20) / 8.0)));
    int qy_0 = int(0);
    for(;;)
    {
        if(qy_0 < int(2))
        {
        }
        else
        {
            break;
        }
        int row_0 = _S9 * int(2) + qy_0;
        int2 _S23 = int2(int(1), int(0));
        bool _S24;
        if(row_0 >= _S21)
        {
            _S24 = row_0 <= _S22;
        }
        else
        {
            _S24 = false;
        }
        int2 r_1;
        if(_S24)
        {
            r_1 = scanline_tile_range_0(row_0, true, px_1, py_1, _S13, _S14, _S15, _S16, rx_1, ry_1, int(8), _S18, _S21, _S19, _S22);
        }
        else
        {
            r_1 = _S23;
        }
        int qx_0 = int(0);
        for(;;)
        {
            if(qx_0 < int(2))
            {
            }
            else
            {
                break;
            }
            int col_0 = _S8 * int(2) + qx_0;
            int device* _S25 = (&kernelContext_0)->sp_mask_0+((qy_0 * int(2) + qx_0) * (&kernelContext_0)->sp_dims_0[int(1)] + k_0);
            bool _S26;
            if(col_0 >= (r_1.x))
            {
                _S26 = col_0 <= (r_1.y);
            }
            else
            {
                _S26 = false;
            }
            int _S27;
            if(_S26)
            {
                _S27 = int(1);
            }
            else
            {
                _S27 = int(0);
            }
            *_S25 = _S27;
            qx_0 = qx_0 + int(1);
        }
        qy_0 = qy_0 + int(1);
    }
    return;
}

