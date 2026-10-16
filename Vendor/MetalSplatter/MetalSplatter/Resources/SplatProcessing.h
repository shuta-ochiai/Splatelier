#import "ShaderCommon.h"

// MARK: - Color Blend Space (LOCAL PATCH, see VENDORED.md)

/// Selects the color space splats are alpha-blended in. Set per pipeline by SplatRenderer from
/// its `colorFormat`:
/// - true  (sRGB render target): colors are linearized and blended in linear space; the
///   hardware re-encodes to sRGB on store. This is upstream MetalSplatter's behaviour.
/// - false (non-sRGB target, e.g. .bgra8Unorm): sRGB-encoded colors are blended directly,
///   which is how 3DGS trainers (gsplat, splatelier) alpha-composite during
///   training. Linear blending of those colors brightens every pixel where colors mix.
constant bool kLinearColorBlending [[function_constant(0)]];

/// Converts an sRGB-encoded splat color to the pipeline's blend space.
inline half3 colorForBlending(half3 srgb) {
    // Using gamma 2.2 approximation (standard for 3DGS training)
    return kLinearColorBlending ? pow(srgb, half3(2.2h)) : srgb;
}

// MARK: - Spherical Harmonics

/// Evaluates spherical harmonics to compute view-dependent color.
half3 evaluateSH(float3 dir,
                 half3 sh0,
                 device const half* shCoeffs,
                 SHDegree shDegree,
                 uint splatIndex);

// MARK: - Covariance

float3 calcCovariance2D(float3 viewPos,
                        packed_half3 cov3Da,
                        packed_half3 cov3Db,
                        float4x4 viewMatrix,
                        float focalX,
                        float focalY,
                        float tanHalfFovX,
                        float tanHalfFovY);

void decomposeCovariance(float3 cov2D, thread float2 &v1, thread float2 &v2);

// MARK: - Vertex Processing

/// Vertex processing with spherical harmonics evaluation.
/// All splats use this path - splat.color contains raw SH0 coefficients.
/// For SH degree 0, shCoefficients can be nullptr.
FragmentIn splatVertex(Splat splat,
                       Uniforms uniforms,
                       uint relativeVertexIndex,
                       device const half* shCoefficients,
                       SHDegree shDegree,
                       uint splatIndex);

// MARK: - Fragment Processing

half splatFragmentAlpha(half2 relativePosition, half splatAlpha);
