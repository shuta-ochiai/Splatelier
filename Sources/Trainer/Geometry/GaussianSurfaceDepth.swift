// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Differentiable surface depth of the Gaussians: per pixel, the depth at which the order-free ray
// transmittance crosses 0.5 (`gof_depth.slang`), and its gradient (`gof_depth_bwd`, G³S eq. 13).
// Implemented from the papers only (Gaussian Opacity Fields, Yu et al. 2024; Geometry-Grounded
// Gaussian Splatting, Zhang et al., arXiv 2601.17835); no code from their repositories.
//
// The kernels see each Gaussian through its ray-space frame: M = S⁻¹Rᵀ and o_g = M (campos - μ)
// (`gof_prepare`). The backward kernel returns the gradient w.r.t. that frame and the opacity, and
// `gof_frame_bwd` (geo_reg.slang) chains it to the parameters.

import Foundation
import MLX
import MLXFast

public enum GaussianSurfaceDepth {

    private static let prepareKernel = SlangCompiledKernel.required("gof_prepare")
    private static let depthKernel = SlangCompiledKernel.required("gof_depth")
    // Many pixels add into the same Gaussian's cells.
    private static let backwardKernel = SlangCompiledKernel.required("gof_depth_bwd", atomicOutputs: true)
    // One SIMD group per pixel, for sparse samples.
    private static let depthCoopKernel = SlangCompiledKernel.required("gof_depth_coop")
    private static let backwardCoopKernel = SlangCompiledKernel.required("gof_depth_bwd_coop", atomicOutputs: true)
    private static let frameBackwardKernel = SlangCompiledKernel.required("gof_frame_bwd")

    public struct Parameters: Sendable {
        public var profile: GaussianOpacityField.TransmittanceProfile
        /// Bisection steps of the 0.5 crossing (⌈n / 3⌉ passes, each splitting the interval into 8).
        public var bisectionIterations: Int
        /// Camera-space depth below which Gaussians are ignored.
        public var nearPlane: Float
        /// Evaluate only every `stride`-th pixel in each direction, at (i s + s/2, j s + s/2): the
        /// depth map is then [H / s, W / s].
        public var stride: Int
        /// One SIMD group per pixel instead of one thread (same result; keeps the GPU busy when
        /// there are few pixels). nil: when `stride` > 1.
        public var cooperative: Bool?

        public init(profile: GaussianOpacityField.TransmittanceProfile = .stochasticSolid,
                    bisectionIterations: Int = 15, nearPlane: Float = 0.01, stride: Int = 1,
                    cooperative: Bool? = nil) {
            precondition(bisectionIterations >= 0 && nearPlane > 0)
            self.cooperative = cooperative
            precondition(stride >= 1)
            self.profile = profile
            self.bisectionIterations = bisectionIterations
            self.nearPlane = nearPlane
            self.stride = stride
        }
    }

    /// A rendered depth map with what `backward` needs.
    public struct Result {
        let cooperative: Bool
        /// [H / stride, W / stride] camera-space depth; <= 0 where the ray never reaches opacity 0.5.
        public let depth: MLXArray
        /// d log T / dt at the depth, from the search's final interval (0 where not measured).
        let slope: MLXArray
        public let stride: Int
        let frames: MLXArray       // [N * 12]
        let extents: MLXArray      // [N]
        let depthParams: MLXArray  // [3] (nearPlane, maxExtent, profile)
    }

    /// The surface depth of `model` seen from `camera`, using the tile lists of a forward pass
    /// (`cache`). nil if no Gaussian is in view.
    public static func render(
        model: GaussianModel, camera: Camera, cache: GaussianRenderer.ForwardCache,
        parameters: Parameters = Parameters()
    ) -> Result? {
        let n = model.count
        guard n > 0, cache.bin.flattenIds.shape[0] > 0 else { return nil }
        let prep = prepareKernel(
            [model.means, model.quats, cache.scales, camera.viewmat, MLXArray([Int32(n)], [1])],
            grid: (((n + 63) / 64) * 64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[n * 12], [n]],
            outputDTypes: [.float32, .float32]
        )
        let frames = prep[0], extents = prep[1]
        // Largest extent among Gaussians in the tile lists (lets the search stop scanning a
        // depth-sorted list early). Kept on the GPU (no sync).
        let visible = (cache.radii[0..., 0] .> 0) .|| (cache.radii[0..., 1] .> 0)
        let maxExtent = MLX.where(visible, extents, MLXArray(Float(0))).max().reshaped([1])
        let depthParams = MLX.concatenated([
            MLXArray([parameters.nearPlane]), maxExtent, MLXArray([Float(parameters.profile.rawValue)])])
        let s = parameters.stride
        let (ws, hs) = (camera.width / s, camera.height / s)
        let coop = parameters.cooperative ?? (s > 1)
        let out = (coop ? depthCoopKernel : depthKernel)(
            [frames, extents, cache.depths, cache.opacities, cache.means2d, cache.conics,
             cache.bin.flattenIds, cache.bin.tileOffsets,
             dims(camera, cache, iterations: parameters.bisectionIterations, stride: s), cameraBuffer(camera),
             depthParams],
            grid: coop ? (ws * 32, hs, 1) : (ws, hs, 1),
            threadGroup: coop ? (32, 4, 1) : (16, 16, 1),
            outputShapes: [[ws * hs], [ws * hs]],
            outputDTypes: [.float32, .float32]
        )
        return Result(
            cooperative: coop, depth: out[0].reshaped([hs, ws]), slope: out[1], stride: s, frames: frames,
            extents: extents, depthParams: depthParams)
    }

    /// Gradients of a loss w.r.t. the raw parameters, given `vDepth` = dL/d(depth), shaped as
    /// `result.depth` (0 where the depth is invalid).
    public static func backward(
        _ result: Result, model: GaussianModel, camera: Camera, cache: GaussianRenderer.ForwardCache,
        vDepth: MLXArray
    ) -> (dMeans: MLXArray, dQuats: MLXArray, dLogScales: MLXArray, dLogitOpacities: MLXArray) {
        let n = model.count
        let s = result.stride
        let (ws, hs) = (camera.width / s, camera.height / s)
        let out = (result.cooperative ? backwardCoopKernel : backwardKernel)(
            [result.frames, result.extents, cache.depths, cache.opacities, cache.means2d, cache.conics,
             cache.bin.flattenIds, cache.bin.tileOffsets, dims(camera, cache, iterations: 0, stride: s),
             cameraBuffer(camera), result.depthParams,
             result.depth.reshaped([ws * hs]), result.slope, vDepth.reshaped([ws * hs])],
            grid: result.cooperative ? (ws * 32, hs, 1) : (ws, hs, 1),
            threadGroup: result.cooperative ? (32, 4, 1) : (16, 16, 1),
            outputShapes: [[n * 12], [n]],
            outputDTypes: [.float32, .float32],
            initValue: 0.0
        )
        let chain = frameBackward(
            means: model.means, quats: model.quats, scales: cache.scales, camera: camera,
            vFrames: out[0].reshaped([n, 12]))
        return (chain.dMeans, chain.dQuats, chain.dScales * cache.scales,
                out[1] * cache.opacities * (1 - cache.opacities))
    }

    /// The gradient w.r.t. the `gof_prepare` frame [N, 12] chained to means, quats and (activated)
    /// scales (gof_frame_bwd).
    static func frameBackward(
        means: MLXArray, quats: MLXArray, scales: MLXArray, camera: Camera, vFrames: MLXArray
    ) -> (dMeans: MLXArray, dQuats: MLXArray, dScales: MLXArray) {
        let n = means.shape[0]
        let out = frameBackwardKernel(
            [means, quats, scales, camera.viewmat, MLXArray([Int32(n)], [1]), vFrames],
            grid: (((n + 63) / 64) * 64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[n * 3], [n * 4], [n * 3]],
            outputDTypes: [.float32, .float32, .float32]
        )
        return (out[0].reshaped([n, 3]), out[1].reshaped([n, 4]), out[2].reshaped([n, 3]))
    }

    private static func dims(
        _ camera: Camera, _ cache: GaussianRenderer.ForwardCache, iterations: Int, stride: Int
    ) -> MLXArray {
        MLXArray([Int32(camera.width), Int32(camera.height), Int32(camera.tileSize),
                  Int32(cache.bin.numTilesX), Int32(iterations), Int32(stride)], [6])
    }

    private static func cameraBuffer(_ camera: Camera) -> MLXArray {
        MLX.concatenated([camera.viewmat, camera.intrinsics])
    }
}
