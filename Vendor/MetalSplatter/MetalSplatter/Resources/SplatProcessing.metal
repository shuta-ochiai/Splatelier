#import "SplatProcessing.h"

// MARK: - Spherical Harmonics Evaluation

/// Evaluates spherical harmonics to compute view-dependent color.
/// - Parameters:
///   - dir: Normalized view direction vector (from camera to splat)
///   - sh0: DC term (raw SH band 0 coefficients, RGB)
///   - shCoeffs: Pointer to higher-order SH coefficients (can be null for degree 0)
///   - shDegree: The SH degree (0-3)
///   - splatIndex: Index of the splat (for computing offset into shCoeffs)
/// - Returns: Final RGB color after SH evaluation
inline half3 evaluateSH(float3 dir,
                        half3 sh0,
                        device const half* shCoeffs,
                        SHDegree shDegree,
                        uint splatIndex) {
    // Degree 0 (constant/ambient)
    float3 result = SH_C0 * float3(sh0);

    if (shDegree >= SHDegree1 && shCoeffs != nullptr) {
        float x = dir.x, y = dir.y, z = dir.z;

        // Calculate offset: SH1=3, SH2=8, SH3=15 coefficients (each is RGB triplet)
        uint coeffsPerSplat = (shDegree == SHDegree1) ? 3 :
                              (shDegree == SHDegree2) ? 8 : 15;
        device const packed_half3* sh = (device const packed_half3*)(shCoeffs + splatIndex * coeffsPerSplat * 3);

        // Degree 1: 3 basis functions
        result -= SH_C1 * y * float3(sh[0]);
        result += SH_C1 * z * float3(sh[1]);
        result -= SH_C1 * x * float3(sh[2]);

        if (shDegree >= SHDegree2) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            float xxMinusYy = xx - yy;
            float xxPlusYy = xx + yy;

            // Degree 2: 5 basis functions
            result += SH_C2_0 * xy * float3(sh[3]);
            result += SH_C2_1 * yz * float3(sh[4]);
            result += SH_C2_2 * (2.0f * zz - xxPlusYy) * float3(sh[5]);
            result += SH_C2_3 * xz * float3(sh[6]);
            result += SH_C2_4 * xxMinusYy * float3(sh[7]);

            if (shDegree >= SHDegree3) {
                float zz4MinusXxYy = 4.0f * zz - xxPlusYy;

                // Degree 3: 7 basis functions
                result += SH_C3_0 * y * (3.0f * xx - yy) * float3(sh[8]);
                result += SH_C3_1 * xy * z * float3(sh[9]);
                result += SH_C3_2 * y * zz4MinusXxYy * float3(sh[10]);
                result += SH_C3_3 * z * (2.0f * zz - 3.0f * xxPlusYy) * float3(sh[11]);
                result += SH_C3_4 * x * zz4MinusXxYy * float3(sh[12]);
                result += SH_C3_5 * z * xxMinusYy * float3(sh[13]);
                result += SH_C3_6 * x * (xx - 3.0f * yy) * float3(sh[14]);
            }
        }
    }

    // Add 0.5 bias and clamp to non-negative (critical for correct colors)
    return half3(max(result + 0.5f, 0.0f));
}

// MARK: - Covariance Projection

// LOCAL PATCH (see VENDORED.md): calcCovariance2D and decomposeCovariance are rewritten from the
// EWA splatting formula (Zwicker et al. 2001) and the closed-form eigen decomposition of a
// symmetric 2x2 matrix. The constants (1.3, 0.3, 0.1) are upstream's, so the footprints are unchanged.

float3 calcCovariance2D(float3 viewPos,
                        packed_half3 cov3Da,
                        packed_half3 cov3Db,
                        float4x4 viewMatrix,
                        float focalX,
                        float focalY,
                        float tanHalfFovX,
                        float tanHalfFovY) {
    // Covariance in view space: R Σ Rᵀ, where R is the rotation part of the view matrix.
    float3x3 rotation = float3x3(viewMatrix[0].xyz, viewMatrix[1].xyz, viewMatrix[2].xyz);
    float3x3 worldCovariance = float3x3(float3(cov3Da.x, cov3Da.y, cov3Da.z),
                                        float3(cov3Da.y, cov3Db.x, cov3Db.y),
                                        float3(cov3Da.z, cov3Db.y, cov3Db.z));
    float3x3 viewCovariance = rotation * worldCovariance * transpose(rotation);

    // Rows of the Jacobian of (u, v) = (fx x / z, fy y / z) at the splat center. x/z and y/z are
    // clamped to 30% beyond the field of view (gsplat does the same), so a splat far off screen
    // does not get a stretched footprint. The projected center itself is not clamped.
    float tx = clamp(viewPos.x / viewPos.z, -1.3f * tanHalfFovX, 1.3f * tanHalfFovX);
    float ty = clamp(viewPos.y / viewPos.z, -1.3f * tanHalfFovY, 1.3f * tanHalfFovY);
    float3 du = (focalX / viewPos.z) * float3(1, 0, -tx);
    float3 dv = (focalY / viewPos.z) * float3(0, 1, -ty);

    // Screen-space covariance J Σ Jᵀ, dilated by 0.3 px² on the diagonal (gsplat's eps2d) so every
    // splat covers about a pixel. The training renderer applies the same dilation.
    float3 covarianceDu = viewCovariance * du;
    float3 covarianceDv = viewCovariance * dv;
    return float3(dot(du, covarianceDu) + 0.3f,
                  dot(du, covarianceDv),
                  dot(dv, covarianceDv) + 0.3f);
}

// cov2D is a symmetric 2D covariance matrix | a b | flattened to (a, b, d).
//                                           | b d |
// v1 and v2 are its principal axes scaled by their standard deviations, v1 along the major axis.
void decomposeCovariance(float3 cov2D, thread float2 &v1, thread float2 &v2) {
    float a = cov2D.x;
    float b = cov2D.y;
    float d = cov2D.z;

    // Eigenvalues (a + d) / 2 ± sqrt(((a - d) / 2)² + b²). The square root is floored at 0.1 px²
    // as upstream does.
    float mid = 0.5f * (a + d);
    float halfDiff = 0.5f * (a - d);
    float halfGap = max(0.1f, sqrt(halfDiff * halfDiff + b * b));
    float majorVariance = mid + halfGap;
    float minorVariance = mid - halfGap;

    // (A - λI) v = 0 gives the major axis v ∝ (b, λ_major - a) = (b, halfGap - halfDiff).
    float2 majorAxis = (b == 0) ? ((a > d) ? float2(1, 0) : float2(0, 1))
                                : normalize(float2(b, halfGap - halfDiff));
    float2 minorAxis = float2(majorAxis.y, -majorAxis.x);

    v1 = majorAxis * sqrt(majorVariance);
    v2 = minorAxis * sqrt(minorVariance);
}

FragmentIn splatVertex(Splat splat,
                       Uniforms uniforms,
                       uint relativeVertexIndex,
                       device const half* shCoefficients,
                       SHDegree shDegree,
                       uint splatIndex) {
    FragmentIn out;

    float4 viewPosition4 = uniforms.viewMatrix * float4(splat.position, 1);
    float3 viewPosition3 = viewPosition4.xyz;

    // Early cull: skip splats behind or at the camera plane
    // (In view space, camera looks down -z, so z >= 0 means behind camera)
    if (viewPosition3.z >= 0) {
        out.position = float4(1, 1, 0, 1);
        return out;
    }

    half3 srgbColor;
    if (shDegree == SHDegree0) {
        // Fast path for SH0
        srgbColor = half3(max(SH_C0 * float3(splat.color.rgb) + 0.5f, 0.0f));
    } else {
        float3 worldPosition = float3(splat.position);
        float3 viewDir = normalize(worldPosition - float3(uniforms.cameraPosition));
        srgbColor = evaluateSH(viewDir, splat.color.rgb, shCoefficients, shDegree, splatIndex);
    }

    float3 cov2D = calcCovariance2D(viewPosition3, splat.covA, splat.covB,
                                    uniforms.viewMatrix,
                                    uniforms.focalX, uniforms.focalY,
                                    uniforms.tanHalfFovX, uniforms.tanHalfFovY);

    float2 axis1;
    float2 axis2;
    decomposeCovariance(cov2D, axis1, axis2);

    float4 projectedCenter = uniforms.projectionMatrix * viewPosition4;

    float bounds = 1.2 * projectedCenter.w;
    if (projectedCenter.z < 0.0 ||
        projectedCenter.z > projectedCenter.w ||
        projectedCenter.x < -bounds ||
        projectedCenter.x > bounds ||
        projectedCenter.y < -bounds ||
        projectedCenter.y > bounds) {
        out.position = float4(1, 1, 0, 1);
        return out;
    }

    const half2 relativeCoordinatesArray[] = { { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } };
    half2 relativeCoordinates = relativeCoordinatesArray[relativeVertexIndex];
    half2 screenSizeFloat = half2(uniforms.screenSize.x, uniforms.screenSize.y);
    half2 projectedScreenDelta =
        (relativeCoordinates.x * half2(axis1) + relativeCoordinates.y * half2(axis2))
        * 2
        * kBoundsRadius
        / screenSizeFloat;

    out.position = float4(projectedCenter.x + projectedScreenDelta.x * projectedCenter.w,
                          projectedCenter.y + projectedScreenDelta.y * projectedCenter.w,
                          projectedCenter.z,
                          projectedCenter.w);
    out.relativePosition = kBoundsRadius * relativeCoordinates;

    // LOCAL PATCH (see VENDORED.md): output the sRGB-encoded color as-is. The render-path
    // vertex shaders convert it to the pipeline's blend space via colorForBlending().
    out.color = half4(srgbColor, splat.color.a);
    return out;
}

half splatFragmentAlpha(half2 relativePosition, half splatAlpha) {
    half negativeMagnitudeSquared = -dot(relativePosition, relativePosition);
    return (negativeMagnitudeSquared < -kBoundsRadiusSquared) ? 0 : exp(0.5 * negativeMagnitudeSquared) * splatAlpha;
}
