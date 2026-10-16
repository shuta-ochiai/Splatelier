#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* dlt_dims_0;
    float device* points_out_0;
    float device* points_in_0;
    int device* obs_counts_0;
    int device* obs_starts_0;
    int device* obs_cam_idx_0;
    float device* obs_pixels_0;
    float device* intrinsics_buf_0;
    float device* viewmats_0;
};

[[kernel]] void dlt_triangulate(uint3 tid_0 [[thread_position_in_grid]], int device* dlt_dims_1 [[buffer(7)]], float device* points_out_1 [[buffer(8)]], float device* points_in_1 [[buffer(6)]], int device* obs_counts_1 [[buffer(5)]], int device* obs_starts_1 [[buffer(4)]], int device* obs_cam_idx_1 [[buffer(2)]], float device* obs_pixels_1 [[buffer(3)]], float device* intrinsics_buf_1 [[buffer(1)]], float device* viewmats_1 [[buffer(0)]])
{
    int minIdx_0;
    float minVal_0;
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->dlt_dims_0 = dlt_dims_1;
    (&kernelContext_0)->points_out_0 = points_out_1;
    (&kernelContext_0)->points_in_0 = points_in_1;
    (&kernelContext_0)->obs_counts_0 = obs_counts_1;
    (&kernelContext_0)->obs_starts_0 = obs_starts_1;
    (&kernelContext_0)->obs_cam_idx_0 = obs_cam_idx_1;
    (&kernelContext_0)->obs_pixels_0 = obs_pixels_1;
    (&kernelContext_0)->intrinsics_buf_0 = intrinsics_buf_1;
    (&kernelContext_0)->viewmats_0 = viewmats_1;
    int j_0 = int(tid_0.x);
    if(j_0 >= dlt_dims_1[int(0)])
    {
        return;
    }
    int _S1 = j_0 * int(3);
    *((&kernelContext_0)->points_out_0+_S1) = (&kernelContext_0)->points_in_0[_S1];
    int _S2 = _S1 + int(1);
    *((&kernelContext_0)->points_out_0+_S2) = (&kernelContext_0)->points_in_0[_S2];
    int _S3 = _S1 + int(2);
    *((&kernelContext_0)->points_out_0+_S3) = (&kernelContext_0)->points_in_0[_S3];
    int count_0 = (&kernelContext_0)->obs_counts_0[j_0];
    if(count_0 < int(2))
    {
        return;
    }
    int _S4 = (&kernelContext_0)->obs_starts_0[j_0];
    thread array<float, int(16)> ATA_0;
    int i_0 = int(0);
    for(;;)
    {
        if(i_0 < int(16))
        {
        }
        else
        {
            break;
        }
        ATA_0[i_0] = 0.0;
        i_0 = i_0 + int(1);
    }
    int k_0 = int(0);
    for(;;)
    {
        if(k_0 < count_0)
        {
        }
        else
        {
            break;
        }
        int obs_idx_0 = _S4 + k_0;
        int cam_0 = (&kernelContext_0)->obs_cam_idx_0[obs_idx_0];
        int _S5 = obs_idx_0 * int(2);
        float u_0 = (&kernelContext_0)->obs_pixels_0[_S5];
        float v_0 = (&kernelContext_0)->obs_pixels_0[_S5 + int(1)];
        int vmBase_0 = cam_0 * int(16);
        int inBase_0 = cam_0 * int(4);
        float fx_0 = (&kernelContext_0)->intrinsics_buf_0[inBase_0];
        float fy_0 = (&kernelContext_0)->intrinsics_buf_0[inBase_0 + int(1)];
        float cx_0 = (&kernelContext_0)->intrinsics_buf_0[inBase_0 + int(2)];
        float cy_0 = (&kernelContext_0)->intrinsics_buf_0[inBase_0 + int(3)];
        int _S6 = vmBase_0 + int(8);
        int _S7 = vmBase_0 + int(9);
        int _S8 = vmBase_0 + int(10);
        int _S9 = vmBase_0 + int(11);
        float p3_0_0 = (&kernelContext_0)->viewmats_0[_S6];
        float p3_1_0 = (&kernelContext_0)->viewmats_0[_S7];
        float p3_2_0 = (&kernelContext_0)->viewmats_0[_S8];
        float p3_3_0 = (&kernelContext_0)->viewmats_0[_S9];
        float row0_0_0 = u_0 * p3_0_0 - (fx_0 * (&kernelContext_0)->viewmats_0[vmBase_0] + cx_0 * (&kernelContext_0)->viewmats_0[_S6]);
        float row0_1_0 = u_0 * p3_1_0 - (fx_0 * (&kernelContext_0)->viewmats_0[vmBase_0 + int(1)] + cx_0 * (&kernelContext_0)->viewmats_0[_S7]);
        float row0_2_0 = u_0 * p3_2_0 - (fx_0 * (&kernelContext_0)->viewmats_0[vmBase_0 + int(2)] + cx_0 * (&kernelContext_0)->viewmats_0[_S8]);
        float row0_3_0 = u_0 * p3_3_0 - (fx_0 * (&kernelContext_0)->viewmats_0[vmBase_0 + int(3)] + cx_0 * (&kernelContext_0)->viewmats_0[_S9]);
        float row1_0_0 = v_0 * p3_0_0 - (fy_0 * (&kernelContext_0)->viewmats_0[vmBase_0 + int(4)] + cy_0 * (&kernelContext_0)->viewmats_0[_S6]);
        float row1_1_0 = v_0 * p3_1_0 - (fy_0 * (&kernelContext_0)->viewmats_0[vmBase_0 + int(5)] + cy_0 * (&kernelContext_0)->viewmats_0[_S7]);
        float row1_2_0 = v_0 * p3_2_0 - (fy_0 * (&kernelContext_0)->viewmats_0[vmBase_0 + int(6)] + cy_0 * (&kernelContext_0)->viewmats_0[_S8]);
        float row1_3_0 = v_0 * p3_3_0 - (fy_0 * (&kernelContext_0)->viewmats_0[vmBase_0 + int(7)] + cy_0 * (&kernelContext_0)->viewmats_0[_S9]);
        ATA_0[int(0)] = ATA_0[int(0)] + (row0_0_0 * row0_0_0 + row1_0_0 * row1_0_0);
        ATA_0[int(1)] = ATA_0[int(1)] + (row0_0_0 * row0_1_0 + row1_0_0 * row1_1_0);
        ATA_0[int(2)] = ATA_0[int(2)] + (row0_0_0 * row0_2_0 + row1_0_0 * row1_2_0);
        ATA_0[int(3)] = ATA_0[int(3)] + (row0_0_0 * row0_3_0 + row1_0_0 * row1_3_0);
        ATA_0[int(4)] = ATA_0[int(4)] + (row0_1_0 * row0_0_0 + row1_1_0 * row1_0_0);
        ATA_0[int(5)] = ATA_0[int(5)] + (row0_1_0 * row0_1_0 + row1_1_0 * row1_1_0);
        ATA_0[int(6)] = ATA_0[int(6)] + (row0_1_0 * row0_2_0 + row1_1_0 * row1_2_0);
        ATA_0[int(7)] = ATA_0[int(7)] + (row0_1_0 * row0_3_0 + row1_1_0 * row1_3_0);
        ATA_0[int(8)] = ATA_0[int(8)] + (row0_2_0 * row0_0_0 + row1_2_0 * row1_0_0);
        ATA_0[int(9)] = ATA_0[int(9)] + (row0_2_0 * row0_1_0 + row1_2_0 * row1_1_0);
        ATA_0[int(10)] = ATA_0[int(10)] + (row0_2_0 * row0_2_0 + row1_2_0 * row1_2_0);
        ATA_0[int(11)] = ATA_0[int(11)] + (row0_2_0 * row0_3_0 + row1_2_0 * row1_3_0);
        ATA_0[int(12)] = ATA_0[int(12)] + (row0_3_0 * row0_0_0 + row1_3_0 * row1_0_0);
        ATA_0[int(13)] = ATA_0[int(13)] + (row0_3_0 * row0_1_0 + row1_3_0 * row1_1_0);
        ATA_0[int(14)] = ATA_0[int(14)] + (row0_3_0 * row0_2_0 + row1_3_0 * row1_2_0);
        ATA_0[int(15)] = ATA_0[int(15)] + (row0_3_0 * row0_3_0 + row1_3_0 * row1_3_0);
        k_0 = k_0 + int(1);
    }
    thread array<float, int(16)> V_0;
    i_0 = int(0);
    for(;;)
    {
        if(i_0 < int(16))
        {
        }
        else
        {
            break;
        }
        V_0[i_0] = 0.0;
        i_0 = i_0 + int(1);
    }
    V_0[int(0)] = 1.0;
    V_0[int(5)] = 1.0;
    V_0[int(10)] = 1.0;
    V_0[int(15)] = 1.0;
    int sweep_0 = int(0);
    for(;;)
    {
        if(sweep_0 < int(60))
        {
        }
        else
        {
            break;
        }
        float a01_0 = abs(ATA_0[int(1)]);
        if(a01_0 > 0.0)
        {
            minVal_0 = a01_0;
        }
        else
        {
            minVal_0 = 0.0;
        }
        float a02_0 = abs(ATA_0[int(2)]);
        float maxAbs_0;
        if(a02_0 > minVal_0)
        {
            maxAbs_0 = a02_0;
            minIdx_0 = int(2);
        }
        else
        {
            maxAbs_0 = minVal_0;
            minIdx_0 = int(1);
        }
        float a03_0 = abs(ATA_0[int(3)]);
        int q_0;
        float maxAbs_1;
        if(a03_0 > maxAbs_0)
        {
            maxAbs_1 = a03_0;
            q_0 = int(3);
        }
        else
        {
            maxAbs_1 = maxAbs_0;
            q_0 = minIdx_0;
        }
        float a12_0 = abs(ATA_0[int(6)]);
        int p_0;
        int q_1;
        float maxAbs_2;
        if(a12_0 > maxAbs_1)
        {
            maxAbs_2 = a12_0;
            p_0 = int(1);
            q_1 = int(2);
        }
        else
        {
            maxAbs_2 = maxAbs_1;
            p_0 = int(0);
            q_1 = q_0;
        }
        float a13_0 = abs(ATA_0[int(7)]);
        int p_1;
        int q_2;
        float maxAbs_3;
        if(a13_0 > maxAbs_2)
        {
            maxAbs_3 = a13_0;
            p_1 = int(1);
            q_2 = int(3);
        }
        else
        {
            maxAbs_3 = maxAbs_2;
            p_1 = p_0;
            q_2 = q_1;
        }
        float a23_0 = abs(ATA_0[int(11)]);
        int p_2;
        int q_3;
        float maxAbs_4;
        if(a23_0 > maxAbs_3)
        {
            maxAbs_4 = a23_0;
            p_2 = int(2);
            q_3 = int(3);
        }
        else
        {
            maxAbs_4 = maxAbs_3;
            p_2 = p_1;
            q_3 = q_2;
        }
        if(maxAbs_4 < 9.99999982451670044e-15)
        {
            break;
        }
        int _S10 = p_2 * int(4);
        float app_0 = ATA_0[_S10 + p_2];
        int _S11 = q_3 * int(4);
        float aqq_0 = ATA_0[_S11 + q_3];
        float apq_0 = ATA_0[_S10 + q_3];
        float theta_0 = (ATA_0[_S11 + q_3] - ATA_0[_S10 + p_2]) / (2.0 * ATA_0[_S10 + q_3]);
        float _S12 = abs(theta_0);
        float t_0;
        if(_S12 < 1.0e+10)
        {
            if(theta_0 >= 0.0)
            {
                t_0 = 1.0;
            }
            else
            {
                t_0 = -1.0;
            }
            t_0 = t_0 / (_S12 + sqrt(theta_0 * theta_0 + 1.0));
        }
        else
        {
            t_0 = 1.0 / (2.0 * theta_0);
        }
        float c_0 = 1.0 / sqrt(t_0 * t_0 + 1.0);
        float _S13 = t_0 * c_0;
        i_0 = int(0);
        for(;;)
        {
            if(i_0 < int(4))
            {
            }
            else
            {
                break;
            }
            bool _S14;
            if(i_0 == p_2)
            {
                _S14 = true;
            }
            else
            {
                _S14 = i_0 == q_3;
            }
            if(_S14)
            {
                i_0 = i_0 + int(1);
                continue;
            }
            int _S15 = i_0 * int(4);
            float newAip_0 = c_0 * ATA_0[_S15 + p_2] - _S13 * ATA_0[_S15 + q_3];
            float newAiq_0 = _S13 * ATA_0[_S15 + p_2] + c_0 * ATA_0[_S15 + q_3];
            ATA_0[_S15 + p_2] = newAip_0;
            ATA_0[_S10 + i_0] = newAip_0;
            ATA_0[_S15 + q_3] = newAiq_0;
            ATA_0[_S11 + i_0] = newAiq_0;
            i_0 = i_0 + int(1);
        }
        float _S16 = t_0 * apq_0;
        ATA_0[_S10 + p_2] = app_0 - _S16;
        ATA_0[_S11 + q_3] = aqq_0 + _S16;
        ATA_0[_S10 + q_3] = 0.0;
        ATA_0[_S11 + p_2] = 0.0;
        int i_1 = int(0);
        for(;;)
        {
            if(i_1 < int(4))
            {
            }
            else
            {
                break;
            }
            int _S17 = i_1 * int(4);
            float vip_0 = V_0[_S17 + p_2];
            float viq_0 = V_0[_S17 + q_3];
            V_0[_S17 + p_2] = c_0 * V_0[_S17 + p_2] - _S13 * V_0[_S17 + q_3];
            V_0[_S17 + q_3] = _S13 * vip_0 + c_0 * viq_0;
            i_1 = i_1 + int(1);
        }
        sweep_0 = sweep_0 + int(1);
    }
    float minVal_1 = ATA_0[int(0)];
    if((ATA_0[int(5)]) < ATA_0[int(0)])
    {
        minVal_0 = ATA_0[int(5)];
        minIdx_0 = int(1);
    }
    else
    {
        minVal_0 = minVal_1;
        minIdx_0 = int(0);
    }
    if((ATA_0[int(10)]) < minVal_0)
    {
        minVal_0 = ATA_0[int(10)];
        minIdx_0 = int(2);
    }
    if((ATA_0[int(15)]) < minVal_0)
    {
        minIdx_0 = int(3);
    }
    float w_0 = V_0[int(12) + minIdx_0];
    if((abs(V_0[int(12) + minIdx_0])) < 9.99999971718068537e-10)
    {
        return;
    }
    *((&kernelContext_0)->points_out_0+_S1) = V_0[minIdx_0] / w_0;
    *((&kernelContext_0)->points_out_0+_S2) = V_0[int(4) + minIdx_0] / w_0;
    *((&kernelContext_0)->points_out_0+_S3) = V_0[int(8) + minIdx_0] / w_0;
    return;
}

