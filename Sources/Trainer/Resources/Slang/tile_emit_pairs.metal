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
    int device* emit_dims_0;
    int device* aabb_count_e_0;
    int device* aabb_e_0;
    int device* starts_0;
    float device* depths_e_0;
    int device* aabb_dims_e_0;
    int device* tile_ids_0;
    int device* sort_keys_0;
    int device* gauss_indices_0;
    int device* radii_e_0;
    float device* means2d_e_0;
    float device* conics_e_0;
    float device* opacities_e_0;
};

[[kernel]] void tile_emit_pairs(uint3 tid_0 [[thread_position_in_grid]], int device* emit_dims_1 [[buffer(12)]], int device* aabb_count_e_1 [[buffer(9)]], int device* aabb_e_1 [[buffer(8)]], int device* starts_1 [[buffer(10)]], float device* depths_e_1 [[buffer(11)]], int device* aabb_dims_e_1 [[buffer(17)]], int device* tile_ids_1 [[buffer(18)]], int device* sort_keys_1 [[buffer(19)]], int device* gauss_indices_1 [[buffer(20)]], int device* radii_e_1 [[buffer(14)]], float device* means2d_e_1 [[buffer(13)]], float device* conics_e_1 [[buffer(15)]], float device* opacities_e_1 [[buffer(16)]])
{
    int j_0;
    int tileId_0;
    bool _S8;
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->emit_dims_0 = emit_dims_1;
    (&kernelContext_0)->aabb_count_e_0 = aabb_count_e_1;
    (&kernelContext_0)->aabb_e_0 = aabb_e_1;
    (&kernelContext_0)->starts_0 = starts_1;
    (&kernelContext_0)->depths_e_0 = depths_e_1;
    (&kernelContext_0)->aabb_dims_e_0 = aabb_dims_e_1;
    (&kernelContext_0)->tile_ids_0 = tile_ids_1;
    (&kernelContext_0)->sort_keys_0 = sort_keys_1;
    (&kernelContext_0)->gauss_indices_0 = gauss_indices_1;
    (&kernelContext_0)->radii_e_0 = radii_e_1;
    (&kernelContext_0)->means2d_e_0 = means2d_e_1;
    (&kernelContext_0)->conics_e_0 = conics_e_1;
    (&kernelContext_0)->opacities_e_0 = opacities_e_1;
    int i_0 = int(tid_0.x);
    if(i_0 >= emit_dims_1[int(1)])
    {
        return;
    }
    int count_0 = (&kernelContext_0)->aabb_count_e_0[i_0];
    if(count_0 == int(0))
    {
        return;
    }
    int _S9 = (&kernelContext_0)->emit_dims_0[int(0)];
    int _S10 = i_0 * int(4);
    int xMin_1 = (&kernelContext_0)->aabb_e_0[_S10];
    int yMin_1 = (&kernelContext_0)->aabb_e_0[_S10 + int(1)];
    int xMax_1 = (&kernelContext_0)->aabb_e_0[_S10 + int(2)];
    int yMax_1 = (&kernelContext_0)->aabb_e_0[_S10 + int(3)];
    int _S11 = (&kernelContext_0)->starts_0[i_0];
    int _S12 = (&kernelContext_0)->emit_dims_0[int(2)];
    int nearBits_0 = (&kernelContext_0)->emit_dims_0[int(4)];
    int _S13 = (clamp((as_type<int>(((&kernelContext_0)->depths_e_0[i_0]))), nearBits_0, (&kernelContext_0)->emit_dims_0[int(5)]) - nearBits_0) >> uint((&kernelContext_0)->emit_dims_0[int(3)]);
    int ty_0;
    int tx_0;
    int k_0;
    if((&kernelContext_0)->aabb_dims_e_0[int(4)] == int(0))
    {
        ty_0 = yMin_1;
        k_0 = int(0);
        for(;;)
        {
            if(ty_0 <= yMax_1)
            {
            }
            else
            {
                break;
            }
            tx_0 = xMin_1;
            for(;;)
            {
                if(tx_0 <= xMax_1)
                {
                }
                else
                {
                    break;
                }
                int slot_0 = _S11 + k_0;
                int tileId_1 = ty_0 * _S9 + tx_0;
                *((&kernelContext_0)->tile_ids_0+slot_0) = tileId_1;
                *((&kernelContext_0)->sort_keys_0+slot_0) = (tileId_1 << uint(_S12)) | _S13;
                *((&kernelContext_0)->gauss_indices_0+slot_0) = i_0;
                int _S14 = k_0 + int(1);
                tx_0 = tx_0 + int(1);
                k_0 = _S14;
            }
            ty_0 = ty_0 + int(1);
        }
        return;
    }
    int _S15 = (&kernelContext_0)->aabb_dims_e_0[int(2)];
    int _S16 = i_0 * int(2);
    int _S17 = (&kernelContext_0)->radii_e_0[_S16];
    int _S18 = _S16 + int(1);
    int _S19 = (&kernelContext_0)->radii_e_0[_S18];
    float _S20 = (&kernelContext_0)->means2d_e_0[_S16];
    float _S21 = (&kernelContext_0)->means2d_e_0[_S18];
    int _S22 = i_0 * int(3);
    float _S23 = (&kernelContext_0)->conics_e_0[_S22];
    float _S24 = (&kernelContext_0)->conics_e_0[_S22 + int(1)];
    float _S25 = (&kernelContext_0)->conics_e_0[_S22 + int(2)];
    float _S26 = ellipse_rr_0((&kernelContext_0)->opacities_e_0[i_0]);
    bool rowMajor_1 = (yMax_1 - yMin_1) <= (xMax_1 - xMin_1);
    if(rowMajor_1)
    {
        ty_0 = yMin_1;
    }
    else
    {
        ty_0 = xMin_1;
    }
    if(rowMajor_1)
    {
        tx_0 = yMax_1;
    }
    else
    {
        tx_0 = xMax_1;
    }
    int line_1 = ty_0;
    k_0 = int(0);
    for(;;)
    {
        if(line_1 <= tx_0)
        {
        }
        else
        {
            break;
        }
        int2 r_1 = scanline_tile_range_0(line_1, rowMajor_1, _S20, _S21, _S23, _S24, _S25, _S26, _S17, _S19, _S15, xMin_1, yMin_1, xMax_1, yMax_1);
        j_0 = r_1.x;
        for(;;)
        {
            if(j_0 <= (r_1.y))
            {
                _S8 = k_0 < count_0;
            }
            else
            {
                _S8 = false;
            }
            if(_S8)
            {
            }
            else
            {
                break;
            }
            int slot_1 = _S11 + k_0;
            if(rowMajor_1)
            {
                tileId_0 = line_1 * _S9 + j_0;
            }
            else
            {
                tileId_0 = j_0 * _S9 + line_1;
            }
            *((&kernelContext_0)->tile_ids_0+slot_1) = tileId_0;
            *((&kernelContext_0)->sort_keys_0+slot_1) = (tileId_0 << uint(_S12)) | _S13;
            *((&kernelContext_0)->gauss_indices_0+slot_1) = i_0;
            int _S27 = k_0 + int(1);
            j_0 = j_0 + int(1);
            k_0 = _S27;
        }
        line_1 = line_1 + int(1);
    }
    if(rowMajor_1)
    {
        line_1 = xMin_1;
    }
    else
    {
        line_1 = yMin_1;
    }
    int _S28;
    if(rowMajor_1)
    {
        _S28 = xMax_1;
    }
    else
    {
        _S28 = yMax_1;
    }
    int line_2 = ty_0;
    for(;;)
    {
        if(line_2 <= tx_0)
        {
            _S8 = k_0 < count_0;
        }
        else
        {
            _S8 = false;
        }
        if(_S8)
        {
        }
        else
        {
            break;
        }
        int2 _S29 = scanline_tile_range_0(line_2, rowMajor_1, _S20, _S21, _S23, _S24, _S25, _S26, _S17, _S19, _S15, xMin_1, yMin_1, xMax_1, yMax_1);
        j_0 = line_1;
        int k_1 = k_0;
        for(;;)
        {
            bool _S30;
            if(j_0 <= _S28)
            {
                _S30 = k_1 < count_0;
            }
            else
            {
                _S30 = false;
            }
            if(_S30)
            {
            }
            else
            {
                break;
            }
            bool _S31;
            if(j_0 >= (_S29.x))
            {
                _S31 = j_0 <= (_S29.y);
            }
            else
            {
                _S31 = false;
            }
            if(_S31)
            {
                j_0 = j_0 + int(1);
                continue;
            }
            int slot_2 = _S11 + k_1;
            if(rowMajor_1)
            {
                tileId_0 = line_2 * _S9 + j_0;
            }
            else
            {
                tileId_0 = j_0 * _S9 + line_2;
            }
            *((&kernelContext_0)->tile_ids_0+slot_2) = tileId_0;
            *((&kernelContext_0)->sort_keys_0+slot_2) = (tileId_0 << uint(_S12)) | _S13;
            *((&kernelContext_0)->gauss_indices_0+slot_2) = i_0;
            k_1 = k_1 + int(1);
            j_0 = j_0 + int(1);
        }
        line_2 = line_2 + int(1);
        k_0 = k_1;
    }
    return;
}

