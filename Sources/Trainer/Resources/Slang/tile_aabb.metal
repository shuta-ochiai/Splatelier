#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
float ellipse_rr_0(float opac_0)
{
    float _S1 = min(3.32999992370605469, sqrt(2.0 * log(opac_0 * 255.0)));
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
    int device* aabb_dims_0;
    int device* radii_0;
    float device* depths_0;
    int device* aabb_0;
    int device* aabb_count_0;
    float device* means2d_0;
    float device* conics_0;
    float device* opacities_0;
};

[[kernel]] void tile_aabb(uint3 tid_0 [[thread_position_in_grid]], int device* aabb_dims_1 [[buffer(5)]], int device* radii_1 [[buffer(1)]], float device* depths_1 [[buffer(2)]], int device* aabb_1 [[buffer(6)]], int device* aabb_count_1 [[buffer(7)]], float device* means2d_1 [[buffer(0)]], float device* conics_1 [[buffer(3)]], float device* opacities_1 [[buffer(4)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->aabb_dims_0 = aabb_dims_1;
    (&kernelContext_0)->radii_0 = radii_1;
    (&kernelContext_0)->depths_0 = depths_1;
    (&kernelContext_0)->aabb_0 = aabb_1;
    (&kernelContext_0)->aabb_count_0 = aabb_count_1;
    (&kernelContext_0)->means2d_0 = means2d_1;
    (&kernelContext_0)->conics_0 = conics_1;
    (&kernelContext_0)->opacities_0 = opacities_1;
    int i_0 = int(tid_0.x);
    if(i_0 >= aabb_dims_1[int(3)])
    {
        return;
    }
    int numTilesX_0 = (&kernelContext_0)->aabb_dims_0[int(0)];
    int numTilesY_0 = (&kernelContext_0)->aabb_dims_0[int(1)];
    int tileSize_1 = (&kernelContext_0)->aabb_dims_0[int(2)];
    int _S8 = i_0 * int(2);
    int rx_1 = (&kernelContext_0)->radii_0[_S8];
    int _S9 = _S8 + int(1);
    int ry_1 = (&kernelContext_0)->radii_0[_S9];
    float d_0 = (&kernelContext_0)->depths_0[i_0];
    bool _S10;
    if(rx_1 <= int(0))
    {
        _S10 = true;
    }
    else
    {
        _S10 = ry_1 <= int(0);
    }
    if(_S10)
    {
        _S10 = true;
    }
    else
    {
        _S10 = d_0 <= 0.0;
    }
    if(_S10)
    {
        int _S11 = i_0 * int(4);
        *((&kernelContext_0)->aabb_0+_S11) = int(1);
        *((&kernelContext_0)->aabb_0+(_S11 + int(1))) = int(1);
        *((&kernelContext_0)->aabb_0+(_S11 + int(2))) = int(-1);
        *((&kernelContext_0)->aabb_0+(_S11 + int(3))) = int(-1);
        *((&kernelContext_0)->aabb_count_0+i_0) = int(0);
        return;
    }
    float px_1 = (&kernelContext_0)->means2d_0[_S8];
    float py_1 = (&kernelContext_0)->means2d_0[_S9];
    float _S12 = float(rx_1);
    float _S13 = float(ry_1);
    float _S14 = float(tileSize_1);
    int _S15 = max(int(0), int(floor((px_1 - _S12) / _S14)));
    int _S16 = min(numTilesX_0 - int(1), int(floor((px_1 + _S12) / _S14)));
    int _S17 = max(int(0), int(floor((py_1 - _S13) / _S14)));
    int _S18 = min(numTilesY_0 - int(1), int(floor((py_1 + _S13) / _S14)));
    if(_S16 < _S15)
    {
        _S10 = true;
    }
    else
    {
        _S10 = _S18 < _S17;
    }
    if(_S10)
    {
        int _S19 = i_0 * int(4);
        *((&kernelContext_0)->aabb_0+_S19) = int(1);
        *((&kernelContext_0)->aabb_0+(_S19 + int(1))) = int(1);
        *((&kernelContext_0)->aabb_0+(_S19 + int(2))) = int(-1);
        *((&kernelContext_0)->aabb_0+(_S19 + int(3))) = int(-1);
        *((&kernelContext_0)->aabb_count_0+i_0) = int(0);
        return;
    }
    int _S20 = i_0 * int(4);
    *((&kernelContext_0)->aabb_0+_S20) = _S15;
    *((&kernelContext_0)->aabb_0+(_S20 + int(1))) = _S17;
    *((&kernelContext_0)->aabb_0+(_S20 + int(2))) = _S16;
    *((&kernelContext_0)->aabb_0+(_S20 + int(3))) = _S18;
    if((&kernelContext_0)->aabb_dims_0[int(4)] == int(0))
    {
        *((&kernelContext_0)->aabb_count_0+i_0) = (_S16 - _S15 + int(1)) * (_S18 - _S17 + int(1));
        return;
    }
    int _S21 = i_0 * int(3);
    float _S22 = (&kernelContext_0)->conics_0[_S21];
    float _S23 = (&kernelContext_0)->conics_0[_S21 + int(1)];
    float _S24 = (&kernelContext_0)->conics_0[_S21 + int(2)];
    float _S25 = ellipse_rr_0((&kernelContext_0)->opacities_0[i_0]);
    bool rowMajor_1 = (_S18 - _S17) <= (_S16 - _S15);
    int first_0;
    if(rowMajor_1)
    {
        first_0 = _S17;
    }
    else
    {
        first_0 = _S15;
    }
    int _S26;
    if(rowMajor_1)
    {
        _S26 = _S18;
    }
    else
    {
        _S26 = _S16;
    }
    int line_1 = first_0;
    int count_0 = int(0);
    for(;;)
    {
        if(line_1 <= _S26)
        {
        }
        else
        {
            break;
        }
        int2 r_1 = scanline_tile_range_0(line_1, rowMajor_1, px_1, py_1, _S22, _S23, _S24, _S25, rx_1, ry_1, tileSize_1, _S15, _S17, _S16, _S18);
        int count_1 = count_0 + max(int(0), r_1.y - r_1.x + int(1));
        line_1 = line_1 + int(1);
        count_0 = count_1;
    }
    *((&kernelContext_0)->aabb_count_0+i_0) = count_0;
    return;
}

