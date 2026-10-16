// LOCAL PATCH (see VENDORED.md): stochastic transparency render path.
// The coverage thresholds and the quad resolve are adapted from SuperSplat
// (https://github.com/playcanvas/supersplat), Copyright (c) 2011-2026 PlayCanvas Ltd., MIT License.
//
// Instead of blending back-to-front sorted splats, each fragment is kept with probability equal
// to its alpha and written opaque with a depth test, so draw order does not matter and no sort
// is needed. In expectation this is the sorted "over" composite. Following SuperSplat
// (PlayCanvas, MIT; "StochasticSplats" 1 spp stochastic transparency):
// - Coverage thresholds are stratified over each screen-space 2x2 pixel quad: the quad's four
//   pixels take the four strata of [0, 1) in a per-(quad, splat) scrambled order, so a splat with
//   alpha a covers 4a +/- 1 of the quad's pixels.
// - An optional resolve pass averages each quad and bilinearly interpolates between quad
//   centres, replacing most of the sampling noise with quantization error.
// The pattern depends only on the pixel and the splat, so a still frame does not flicker.

#include "SplatProcessing.h"

// Unlike FragmentIn, no half-typed varyings: mixing them with a 32-bit [[flat]] varying made every
// varying but the position arrive as zero on an M-series Mac (fine under Metal shader validation,
// which changes the varying layout). The color is constant over the splat, so it travels flat,
// packed as half4 bits.
typedef struct
{
    float4 position [[position]];
    float2 relativePosition;   // Ranges from -kBoundsRadius to +kBoundsRadius
    uint2 packedColor [[flat]]; // half4 color, as_type-packed
    uint splatID [[flat]];      // Stable per-splat hash input for the coverage thresholds
} StochasticFragmentIn;

vertex StochasticFragmentIn stochasticSplatVertexShader(uint vertexID [[vertex_id]],
                                                        uint instanceID [[instance_id]],
                                                        ushort amplificationID [[amplification_id]],
                                                        device const ChunkInfo* chunks [[ buffer(BufferIndexChunks) ]],
                                                        constant ChunkedSplatIndex* splatIndexArray [[ buffer(BufferIndexSplatIndex) ]],
                                                        constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]]) {
    Uniforms uniforms = uniformsArray.uniforms[min(int(amplificationID), kMaxViewCount)];

    StochasticFragmentIn out;

    uint drawID = instanceID * uniforms.indexedSplatCount + (vertexID / 4);
    if (drawID >= uniforms.splatCount) {
        out.position = float4(1, 1, 0, 1);
        return out;
    }

    // The index buffer is sorted back to front for some recent camera pose. Reading it in reverse
    // draws roughly front to back, so the depth test rejects most fragments behind the first
    // opaque coverage before they are shaded.
    ChunkedSplatIndex idx = splatIndexArray[uniforms.splatCount - 1 - drawID];

    if (idx.chunkIndex >= uniforms.chunkCount) {
        out.position = float4(1, 1, 0, 1);
        return out;
    }

    ChunkInfo chunk = chunks[idx.chunkIndex];

    if (idx.splatIndex >= chunk.splatCount || !chunk.enabled) {
        out.position = float4(1, 1, 0, 1);
        return out;
    }

    Splat splat = chunk.splats[idx.splatIndex];

    FragmentIn splatOut = splatVertex(splat, uniforms, vertexID % 4,
                                      chunk.shCoefficients, chunk.shDegree,
                                      idx.splatIndex);
    out.position = splatOut.position;
    out.relativePosition = float2(splatOut.relativePosition);
    out.packedColor = as_type<uint2>(half4(colorForBlending(splatOut.color.rgb), splatOut.color.a));
    out.splatID = idx.splatIndex ^ (uint(idx.chunkIndex) * 0x9E3779B9u);
    return out;
}

/// Integer hash (Chris Wellons' "prospector" mix) -> uniform uint.
static inline uint hashU32(uint x) {
    uint v = x;
    v ^= v >> 16;
    v *= 0x7feb352du;
    v ^= v >> 15;
    v *= 0x846ca68bu;
    v ^= v >> 16;
    return v;
}

fragment half4 stochasticSplatFragmentShader(StochasticFragmentIn in [[stage_in]]) {
    half4 color = as_type<half4>(in.packedColor);
    half alpha = splatFragmentAlpha(half2(in.relativePosition), color.a);

    uint2 pixel = uint2(in.position.xy);
    uint2 quad = pixel >> 1;
    uint h = hashU32((quad.x * 1973u) ^ (quad.y * 9277u) ^ ((in.splatID + 1u) * 26699u));
    uint stratum = ((pixel.y & 1u) * 2u + (pixel.x & 1u)) ^ (h & 3u);
    float threshold = (float(stratum) + float(h >> 8) * (1.0f / 16777216.0f)) * 0.25f;
    if (threshold >= float(alpha)) {
        discard_fragment();
    }
    // Premultiplied like the sorted path: opaque coverage, empty pixels keep the clear color.
    return half4(color.rgb, 1);
}

// MARK: - 2x2 quad resolve

typedef struct
{
    float4 position [[position]];
} StochasticResolveIn;

/// A single triangle covering the entire screen.
vertex StochasticResolveIn stochasticResolveVertexShader(uint vertexID [[vertex_id]]) {
    StochasticResolveIn out;
    out.position = float4((vertexID == 2) ? 3.0 : -1.0,
                          (vertexID == 0) ? -3.0 : 1.0,
                          0.0,
                          1.0);
    return out;
}

fragment half4 stochasticResolveFragmentShader(StochasticResolveIn in [[stage_in]],
                                               texture2d<half> source [[texture(0)]]) {
    constexpr sampler bilinear(coord::pixel, address::clamp_to_edge, filter::linear);
    // Quad space: integer coordinates sit on the centres of the 2x2 quads the coverage thresholds
    // are stratified over (pixel centres are at +0.5, so quad q's centre is at 2q + 1).
    float2 u = (in.position.xy - 1.0f) * 0.5f;
    float2 q0 = floor(u);
    float2 f = u - q0;
    // A bilinear tap at a quad's centre is the mean of its four pixels.
    float2 c = q0 * 2.0f + 1.0f;
    half4 a = source.sample(bilinear, c);
    half4 b = source.sample(bilinear, c + float2(2, 0));
    half4 d = source.sample(bilinear, c + float2(0, 2));
    half4 e = source.sample(bilinear, c + float2(2, 2));
    return mix(mix(a, b, half(f.x)), mix(d, e, half(f.x)), half(f.y));
}
