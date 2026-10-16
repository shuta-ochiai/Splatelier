// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXFast

/// Differentiable Gaussian Splatting renderer. Forward does projection → tile binning → rasterize;
/// backward returns gradients of the raw (pre-activation) parameters.
public final class GaussianRenderer {

    public struct ForwardCache {
        public let means2d: MLXArray         // [N, 2]
        public let conics: MLXArray          // [N, 3]
        public let depths: MLXArray          // [N]
        public let radii: MLXArray           // [N, 2] int32 (radius_x, radius_y)
        public let colors: MLXArray          // [N, 3] fp16, RGB from SH evaluation
        public let shDirs: MLXArray          // [N, 3] unit view directions
        public let shDirsLen: MLXArray       // [N, 1] ‖means - campos‖ (for the normalize Jacobian in backward)
        public let sh0: MLXArray             // [N, 3] band 0 (DC)
        public let shNFlat: MLXArray         // [N, 9] higher-order coefficients (shN flattened)
        public let shDegree: Int             // SH degree used in this forward
        public let scales: MLXArray          // [N, 3] exp(logScales)
        public let opacities: MLXArray       // [N] sigmoid(logitOpacities)
        public let bin: TileBinningResult
        public let renderAlpha: MLXArray     // [H, W]
        public let lastIds: MLXArray         // [H, W] int32

        // Replay cache for PocketGS Operator T (Sec. III-C). nil / replayKMax=0 when disabled.
        // Holds only cache_id [K_max, H, W] and cache_count [H, W] (bwd reconstructs C_in / α).
        public let replayCacheId: MLXArray?
        public let replayCacheCount: MLXArray?
        public let replayRenderColor: MLXArray?   // for computing R_{k+1} in backward
        public let replayKMax: Int                // 0 means replay disabled
    }

    public struct Gradients {
        public let dMeans: MLXArray            // [N, 3]
        public let dQuats: MLXArray            // [N, 4]
        public let dLogScales: MLXArray        // [N, 3]
        public let dLogitOpacities: MLXArray   // [N]
        public let dSh0: MLXArray              // [N, 3]
        public let dShN: MLXArray              // [N, 3, 3]
        public init(dMeans: MLXArray, dQuats: MLXArray, dLogScales: MLXArray,
                    dLogitOpacities: MLXArray, dSh0: MLXArray, dShN: MLXArray) {
            self.dMeans = dMeans; self.dQuats = dQuats; self.dLogScales = dLogScales
            self.dLogitOpacities = dLogitOpacities; self.dSh0 = dSh0; self.dShN = dShN
        }
    }

    private let projectFwd: MLXFast.MLXFastKernel
    private let projectBwd: MLXFast.MLXFastKernel
    private let rasterizeFwd: MLXFast.MLXFastKernel
    /// SIMD-group cooperative version (`rasterize_bwd_coop`). Same gradients as `rasterize_bwd`,
    /// with atomics reduced to per-warp.
    private let rasterizeBwd: MLXFast.MLXFastKernel
    private let shEvalFwd: MLXFast.MLXFastKernel
    private let shEvalBwd: MLXFast.MLXFastKernel
    /// Kernels for the replay cache. Loaded on first use.
    private var replayKernels: (fwd: MLXFast.MLXFastKernel, bwd: MLXFast.MLXFastKernel)?

    /// Use the replay cache from PocketGS Operator T (Sec. III-C). If true, forward writes cache_id
    /// / cache_count and backward uses the eq. 8 replay kernel (false uses the mathematically
    /// equivalent T-reconstruction kernel). Backward picks its path from the cache's `replayKMax`,
    /// so it follows the setting at forward time.
    public var useReplayCache: Bool = false

    /// In tile binning, drop tiles inside the AABB that the ellipse does not cover (TileBinning's
    /// scanline culling). Reduces intersections (sort / rasterize work) without changing the
    /// render. false uses every tile in the AABB (the tests compare the two).
    public var scanlineTileCulling: Bool = true

    /// Per-pixel fragment limit of the replay cache. 64 is meant to cover most object-centric
    /// scenes (check shortfalls with `lastReplayCacheOverflowPixels`). Memory is H*W*(4*K_max + 4)
    /// bytes.
    public var replayKMax: Int = 64

    /// Hook for per-stage timing (ms). Measured at the existing eval barriers. nil means no
    /// overhead.
    public nonisolated(unsafe) static var profileHook: ((_ stage: String, _ ms: Double) -> Void)?

    private func mark(_ stage: String, since start: Date) {
        guard Self.profileHook != nil || TrainingSignpost.log.signpostsEnabled else { return }
        let ms = Date().timeIntervalSince(start) * 1000
        Self.profileHook?(stage, ms)
        TrainingSignpost.event(stage, ms: ms)
    }

    /// Step statistics for debugging training behavior (means gradients via the projection path /
    /// SH direction path, radii distribution, model state).
    public struct Diagnostic: Sendable {
        public let n: Int
        public let radiiMaxX: Int32
        public let radiiMaxY: Int32
        public let radiiCulled: Int        // count with rx<=0 and ry<=0
        public let dMeansAbsMax: Float
        public let dMeansAbsMean: Float
        public let gradDirsAbsMax: Float
        public let gradDirsAbsMean: Float
        public let vColorsAbsMax: Float
        public let vColorsAbsMean: Float
        public let scaleMax: Float
        public let scaleMedian: Float
        public let scaleAbove1Count: Int    // any axis > 1 m
        public let scaleAbove005Count: Int  // any axis > 0.05 m
        public let scaleAbove010Count: Int  // any axis > 0.10 m
        public let scaleAbove020Count: Int  // any axis > 0.20 m
        public let opacityMax: Float
        public let opacityMedian: Float
        public let opacityBelowThreshCount: Int   // opa < 1/255
        // The Gaussian with the largest radii (to see whether it is the same one across iters)
        public let maxRadiiGaussIndex: Int
        public let maxRadiiGaussScale: Float       // scale of the largest axis
        public let maxRadiiGaussOpac: Float
        public let maxRadiiGaussDepth: Float       // camera-space z
    }

    /// If true, fill `lastDiagnostic` on every backward. Adds eval barriers, so normally off.
    public nonisolated(unsafe) static var diagnosticEnabled = false
    public nonisolated(unsafe) static var lastDiagnostic: Diagnostic?

    /// Number of pixels where the replay cache was truncated at K_max. Updated only on
    /// diagnosticEnabled iters (otherwise the previous value). If > 0 happens often, raise
    /// `replayKMax`.
    public nonisolated(unsafe) static var lastReplayCacheOverflowPixels: Int = 0

    /// cache_count bit layout: bits 0-7 = fragment count K, bit 30 = overflow flag, bit 31 = 0.
    /// Must match the constants in rasterize_fwd_replay.slang / rasterize_bwd_replay.slang exactly
    /// (a mismatch causes a GPU OOB crash).
    public static let replayCacheCountMask: Int32   = 0x3FFFFFFF
    public static let replayCacheOverflowBit: Int32 = 0x40000000

    public init() throws {
        self.projectFwd = try SlangCompiledKernel.load(name: "project_to_screen_fwd")
        self.projectBwd = try SlangCompiledKernel.load(name: "project_to_screen_bwd")
        self.rasterizeFwd = try SlangCompiledKernel.load(name: "rasterize_fwd")
        self.rasterizeBwd = try SlangCompiledKernel.load(name: "rasterize_bwd_coop")
        self.shEvalFwd = try SlangCompiledKernel.load(name: "sh_eval_deg3_fwd")
        self.shEvalBwd = try SlangCompiledKernel.load(name: "sh_eval_deg3_bwd")
    }

    private func ensureReplayKernels() throws -> (
        fwd: MLXFast.MLXFastKernel, bwd: MLXFast.MLXFastKernel
    ) {
        if let k = replayKernels { return k }
        let k = (fwd: try SlangCompiledKernel.load(name: "rasterize_fwd_replay"),
                 bwd: try SlangCompiledKernel.load(name: "rasterize_bwd_replay"))
        replayKernels = k
        return k
    }

    /// Renders `model` from `camera` and returns color, alpha, and the cache for backward.
    public func forward(
        model: GaussianModel, camera: Camera, shDegree: Int = 3
    ) -> (renderColor: MLXArray, renderAlpha: MLXArray, cache: ForwardCache) {
        let n = model.count
        let tParam = Date()
        let scales = model.scales()
        let opacities = model.opacities()
        eval(scales, opacities)
        mark("param_activations", since: tParam)

        let imageDims = MLXArray([Float(camera.width), Float(camera.height)], [2])
        let nBufProj = MLXArray([Int32(n)], [1])
        let tProj = Date()
        let projOut = projectFwd(
            [model.means, model.quats, scales, camera.viewmat, camera.intrinsics,
             imageDims, opacities, nBufProj],
            grid: (((n + 63) / 64) * 64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[n * 2], [n * 3], [n], [n * 2]],
            outputDTypes: [.float32, .float32, .float32, .int32]
        )
        let means2d = projOut[0].reshaped([n, 2])
        let conics  = projOut[1].reshaped([n, 3])
        let depths  = projOut[2]
        let radii   = projOut[3].reshaped([n, 2])
        eval(means2d, conics, depths, radii)
        mark("project_fwd", since: tProj)

        let tBin = Date()
        let bin = TileBinning.bin(
            means2d: means2d, radii: radii, depths: depths,
            conics: scanlineTileCulling ? conics : nil,
            opacities: scanlineTileCulling ? opacities : nil,
            imageWidth: camera.width, imageHeight: camera.height,
            tileSize: camera.tileSize
        )
        mark("tile_binning", since: tBin)

        // SH color = clamp(SH_eval(coeffs, dir) + 0.5, 0), dir = normalize(means - campos), campos
        // = -Rᵀt. Pass the unit direction (as gsplat does) and apply the normalize Jacobian on the
        // MLX side in backward. Passing an unnormalized dir to the kernel made training unstable.
        let tSH = Date()
        let vm = camera.viewmat.reshaped([4, 4])
        let camR = vm[0..<3, 0..<3]
        let camT = vm[0..<3, 3..<4]
        // A matrix-vector product stays full float32 (MLX uses TF32 only for wider matmuls).
        let campos = (-camR.transposed().matmul(camT)).reshaped([3])
        let shDirsUnnorm = model.means - campos                                   // [N, 3]
        let shDirsLen = MLX.sqrt(MLX.maximum(
            (shDirsUnnorm * shDirsUnnorm).sum(axis: 1, keepDims: true),
            MLXArray(1e-12 as Float)
        ))                                                                        // [N, 1]
        let shDirs = shDirsUnnorm / shDirsLen                                     // [N, 3] unit vectors
        // Pass sh0 and shN as separate buffers (the reshape does not copy). The kernel skips
        // higher-order loads based on degree.
        let sh0 = model.sh0                                                       // [N, 3]
        let shNFlat = model.shN.reshaped([n, GaussianModel.shRestBands * 3])      // [N, 9]
        let degBuf = MLXArray([Int32(shDegree)], [1])
        let nBuf = MLXArray([Int32(n)], [1])
        // Pass radii so SH evaluation is skipped for culled Gaussians (same visibility test as
        // tile_binning). Colors are stored as fp16 to halve bandwidth. The rasterizer reads them
        // promoted to fp32; accumulation and gradients are fp32.
        let shOut = shEvalFwd(
            [shDirs, sh0, shNFlat, degBuf, nBuf, radii],
            grid: (((n + 63) / 64) * 64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[n * 3]],
            outputDTypes: [.float16]
        )
        let rgb = shOut[0].reshaped([n, 3])   // fp16
        eval(rgb)
        mark("sh_eval_fwd", since: tSH)

        let tRaster = Date()
        let totalPixels = camera.width * camera.height

        let renderColor: MLXArray
        let renderAlpha: MLXArray
        let lastIds: MLXArray
        let cacheId: MLXArray?
        let cacheCount: MLXArray?
        let cachedRenderColor: MLXArray?
        let kMax: Int

        // If every Gaussian is culled, flattenIds is empty. Binding a zero-size buffer SIGABRTs,
        // so short-circuit and return the background color with alpha=0.
        if bin.flattenIds.shape[0] == 0 {
            kMax = 0
            cacheId = nil; cacheCount = nil
            cachedRenderColor = nil
            let bg = MLX.broadcast(
                camera.background.reshaped([1, 1, 3]),
                to: [camera.height, camera.width, 3]
            )
            renderColor = bg
            renderAlpha = MLXArray.zeros([camera.height, camera.width])
            lastIds = MLXArray.zeros([totalPixels], dtype: .int32)
            eval(renderColor, renderAlpha, lastIds)
            mark("rasterize_fwd", since: tRaster)
            let cache = ForwardCache(
                means2d: means2d, conics: conics, depths: depths, radii: radii,
                colors: rgb, shDirs: shDirs, shDirsLen: shDirsLen,
                sh0: sh0, shNFlat: shNFlat, shDegree: shDegree,
                scales: scales, opacities: opacities,
                bin: bin, renderAlpha: renderAlpha, lastIds: lastIds,
                replayCacheId: nil, replayCacheCount: nil,
                replayRenderColor: nil, replayKMax: 0
            )
            return (renderColor: renderColor, renderAlpha: renderAlpha, cache: cache)
        }

        if useReplayCache {
            // PocketGS Operator T: output cache_id / cache_count (bwd reconstructs C_in / α).
            // Overflow is packed into bit 30 of cache_count rather than an atomic, so no initValue
            // (huge zero-fill) is needed.
            kMax = replayKMax
            let dims5 = MLXArray(
                [Int32(camera.width), Int32(camera.height),
                 Int32(camera.tileSize), Int32(bin.numTilesX), Int32(kMax)],
                [5]
            )
            let (fwdKernel, _) = try! ensureReplayKernels()
            let cacheSlots = kMax * totalPixels
            let rasterOut = fwdKernel(
                [
                    means2d, conics, rgb, opacities,
                    bin.flattenIds, bin.tileOffsets,
                    dims5, camera.background,
                ],
                grid: (camera.width, camera.height, 1),
                threadGroup: (camera.tileSize, camera.tileSize, 1),
                outputShapes: [
                    [totalPixels * 3],   // render_color
                    [totalPixels],        // render_alpha
                    [cacheSlots],         // cache_id (only `count` entries are written, and bwd reads only `count`)
                    [totalPixels],        // cache_count (bit 30 = overflow flag)
                ],
                outputDTypes: [
                    .float32, .float32, .int32, .int32,
                ]
            )
            renderColor = rasterOut[0].reshaped([camera.height, camera.width, 3])
            renderAlpha = rasterOut[1].reshaped([camera.height, camera.width])
            cacheId = rasterOut[2]
            let cacheCountRaw = rasterOut[3]
            cacheCount = cacheCountRaw
            eval(renderColor, renderAlpha)
            // The overflow count is computed only on diag iters (avoids a GPU→CPU readback
            // normally). Summing overflowBit (2^30) as int32 overflows at 2+ pixels, so convert to
            // 0/1 before counting.
            if Self.diagnosticEnabled {
                let overflowMask = MLXArray(Self.replayCacheOverflowBit)
                let overflowed = ((cacheCountRaw & overflowMask) .!= 0).asType(.int32)
                Self.lastReplayCacheOverflowPixels = overflowed.sum().item(Int.self)
            }
            // Unused by replay, but lastIds is non-optional, so pass a dummy.
            lastIds = MLXArray.zeros([totalPixels], dtype: .int32)
            cachedRenderColor = renderColor
        } else {
            kMax = 0
            cacheId = nil
            cacheCount = nil
            cachedRenderColor = nil
            let dims4 = MLXArray(
                [Int32(camera.width), Int32(camera.height),
                 Int32(camera.tileSize), Int32(bin.numTilesX)],
                [4]
            )
            let rasterInputs = [
                means2d, conics, rgb, opacities,
                bin.flattenIds, bin.tileOffsets,
                dims4, camera.background,
            ]
            let rasterOut = rasterizeFwd(
                rasterInputs,
                grid: (camera.width, camera.height, 1),
                threadGroup: (camera.tileSize, camera.tileSize, 1),
                outputShapes: [[totalPixels * 3], [totalPixels], [totalPixels]],
                outputDTypes: [.float32, .float32, .int32]
            )
            renderColor = rasterOut[0].reshaped([camera.height, camera.width, 3])
            renderAlpha = rasterOut[1].reshaped([camera.height, camera.width])
            lastIds = rasterOut[2]
            eval(renderColor, renderAlpha, lastIds)
        }
        mark("rasterize_fwd", since: tRaster)

        let cache = ForwardCache(
            means2d: means2d, conics: conics, depths: depths, radii: radii,
            colors: rgb, shDirs: shDirs, shDirsLen: shDirsLen,
            sh0: sh0, shNFlat: shNFlat, shDegree: shDegree,
            scales: scales, opacities: opacities,
            bin: bin, renderAlpha: renderAlpha, lastIds: lastIds,
            replayCacheId: cacheId,
            replayCacheCount: cacheCount,
            replayRenderColor: cachedRenderColor,
            replayKMax: kMax
        )
        return (renderColor, renderAlpha, cache)
    }

    /// Backpropagation. vRenderColor / vRenderAlpha are dL/d(color) / dL/d(alpha), either [H, W(,
    /// 3)] or flat.
    public func backward(
        model: GaussianModel, camera: Camera, cache: ForwardCache,
        vRenderColor: MLXArray, vRenderAlpha: MLXArray
    ) -> Gradients {
        let n = model.count
        let totalPixels = camera.width * camera.height
        // Slang reads flat buffers.
        let vColorFlat = vRenderColor.reshaped([totalPixels * 3])
        let vAlphaFlat = vRenderAlpha.reshaped([totalPixels])

        let tRasterBwd = Date()

        // If forward short-circuited on an empty flattenIds, all gradients are 0 too (avoids
        // binding a zero-size buffer).
        if cache.bin.flattenIds.shape[0] == 0 {
            let dMeansZ  = MLXArray.zeros([n, 3])
            let dQuatsZ  = MLXArray.zeros([n, 4])
            let dLogSZ   = MLXArray.zeros([n, 3])
            let dLogOpZ  = MLXArray.zeros([n])
            let dSh0Z    = MLXArray.zeros([n, 3])
            let dShNZ    = MLXArray.zeros([n, GaussianModel.shRestBands, 3])
            eval(dMeansZ, dQuatsZ, dLogSZ, dLogOpZ, dSh0Z, dShNZ)
            mark("rasterize_bwd", since: tRasterBwd)
            return Gradients(
                dMeans: dMeansZ, dQuats: dQuatsZ, dLogScales: dLogSZ,
                dLogitOpacities: dLogOpZ, dSh0: dSh0Z, dShN: dShNZ
            )
        }

        let vMeans2d: MLXArray
        let vConics: MLXArray
        let vColors: MLXArray
        let vOpacities: MLXArray

        if cache.replayKMax > 0,
           let cacheId = cache.replayCacheId,
           let cacheCount = cache.replayCacheCount,
           let cachedRenderColor = cache.replayRenderColor
        {
            // PocketGS Operator T: eq. 8 replay backward. The cache is pixel-keyed, so tile info is
            // not needed.
            let dims5 = MLXArray(
                [Int32(camera.width), Int32(camera.height),
                 Int32(camera.tileSize), Int32(cache.bin.numTilesX), Int32(cache.replayKMax)],
                [5]
            )
            let (_, bwdKernel) = try! ensureReplayKernels()
            // The background is absorbed into R_{k+1}, so it is not passed.
            let rasterBwd = bwdKernel(
                [
                    cache.means2d, cache.conics, cache.colors, cache.opacities,
                    dims5,
                    cachedRenderColor.reshaped([totalPixels * 3]),
                    cache.renderAlpha.reshaped([totalPixels]),
                    cacheId, cacheCount,
                    vColorFlat, vAlphaFlat,
                ],
                grid: (camera.width, camera.height, 1),
                threadGroup: (camera.tileSize, camera.tileSize, 1),
                outputShapes: [[n * 2], [n * 3], [n * 3], [n]],
                outputDTypes: [.float32, .float32, .float32, .float32],
                initValue: 0.0
            )
            vMeans2d   = rasterBwd[0].reshaped([n, 2])
            vConics    = rasterBwd[1].reshaped([n, 3])
            vColors    = rasterBwd[2].reshaped([n, 3])
            vOpacities = rasterBwd[3]
        } else {
            // gsplat-style T-reconstruction backward.
            let dims4 = MLXArray(
                [Int32(camera.width), Int32(camera.height),
                 Int32(camera.tileSize), Int32(cache.bin.numTilesX)],
                [4]
            )
            let rasterBwd = rasterizeBwd(
                [
                    cache.means2d, cache.conics, cache.colors, cache.opacities,
                    cache.bin.flattenIds, cache.bin.tileOffsets, dims4, camera.background,
                    cache.renderAlpha.reshaped([totalPixels]),
                    cache.lastIds,
                    vColorFlat, vAlphaFlat,
                ],
                grid: (camera.width, camera.height, 1),
                threadGroup: (camera.tileSize, camera.tileSize, 1),
                outputShapes: [[n * 2], [n * 3], [n * 3], [n]],
                outputDTypes: [.float32, .float32, .float32, .float32],
                initValue: 0.0
            )
            vMeans2d   = rasterBwd[0].reshaped([n, 2])
            vConics    = rasterBwd[1].reshaped([n, 3])
            vColors    = rasterBwd[2].reshaped([n, 3])
            vOpacities = rasterBwd[3]
        }
        eval(vMeans2d, vConics, vColors, vOpacities)
        mark("rasterize_bwd", since: tRasterBwd)

        let tProjBwd = Date()
        let (dMeans, dQuats, dLogScales, dLogitOpacities) = projectionBackward(
            model: model, camera: camera, cache: cache,
            vMeans2d: vMeans2d, vConics: vConics, vOpacities: vOpacities)
        eval(dMeans, dQuats, dLogScales, dLogitOpacities)
        mark("project_bwd", since: tProjBwd)

        // SH bwd (as in gsplat) returns gradDirs, the gradient w.r.t. the unit dir. For means,
        // multiply by the Jacobian (I - n nᵀ)/‖u‖ of dir = u/‖u‖, with u = means - campos, in MLX.
        let tSHBwd = Date()
        let degBuf = MLXArray([Int32(cache.shDegree)], [1])
        let nBuf = MLXArray([Int32(n)], [1])
        // Skip SH backward for Gaussians culled by radii (the kernel writes a 0 gradient).
        let shBwd = shEvalBwd(
            [cache.shDirs, cache.sh0, cache.shNFlat, degBuf, nBuf, vColors, cache.radii],
            grid: (((n + 63) / 64) * 64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[n * 3], [n * 3], [n * GaussianModel.shRestBands * 3]],
            outputDTypes: [.float32, .float32, .float32]
        )
        let gradDirs = shBwd[0].reshaped([n, 3])
        let dSh0 = shBwd[1].reshaped([n, 3])
        let dShN = shBwd[2].reshaped([n, GaussianModel.shRestBands, 3])
        let nDotG = (cache.shDirs * gradDirs).sum(axis: 1, keepDims: true)        // [N, 1]
        let dMeansFromSH = (gradDirs - cache.shDirs * nDotG) / cache.shDirsLen    // [N, 3]
        let dMeansTotal = dMeans + dMeansFromSH
        eval(dSh0, dShN, dMeansTotal)
        mark("sh_eval_bwd", since: tSHBwd)

        if Self.diagnosticEnabled {
            Self.lastDiagnostic = makeDiagnostic(
                model: model, cache: cache, dMeans: dMeans, dMeansFromSH: dMeansFromSH, vColors: vColors)
        }

        return Gradients(
            dMeans: dMeansTotal,
            dQuats: dQuats,
            dLogScales: dLogScales,
            dLogitOpacities: dLogitOpacities,
            dSh0: dSh0,
            dShN: dShN
        )
    }

    /// Gradients of the raw means / quats / log-scales / logit-opacities from the screen-space
    /// cotangents of a rasterize backward (activation derivatives included). Project bwd takes only
    /// the means2d / conics cotangents (depth / cov2d are 0 inside the kernel).
    private func projectionBackward(
        model: GaussianModel, camera: Camera, cache: ForwardCache,
        vMeans2d: MLXArray, vConics: MLXArray, vOpacities: MLXArray
    ) -> (dMeans: MLXArray, dQuats: MLXArray, dLogScales: MLXArray, dLogitOpacities: MLXArray) {
        let n = model.count
        let imageDims = MLXArray([Float(camera.width), Float(camera.height)], [2])
        let nBufProj = MLXArray([Int32(n)], [1])
        let projBwd = projectBwd(
            [
                model.means, model.quats, cache.scales,
                camera.viewmat, camera.intrinsics, imageDims, nBufProj,
                vMeans2d, vConics,
            ],
            grid: (((n + 63) / 64) * 64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[n * 3], [n * 4], [n * 3]],
            outputDTypes: [.float32, .float32, .float32]
        )
        // Activation derivatives: exp → scale, sigmoid → opacity * (1 - opacity).
        return (projBwd[0].reshaped([n, 3]), projBwd[1].reshaped([n, 4]),
                projBwd[2].reshaped([n, 3]) * cache.scales,
                vOpacities * (cache.opacities * (1.0 - cache.opacities)))
    }

    /// Composites a per-Gaussian 3-vector (e.g. a normal) [N, 3] with the alpha weights of the
    /// forward that made `cache`, over a zero background. [H, W, 3]. When the forward kept a replay
    /// cache, only every `stride`-th pixel (i s + s/2, j s + s/2) is composited, from that cache;
    /// the other pixels are 0.
    public func renderFeatures(_ features: MLXArray, camera: Camera, cache: ForwardCache, stride: Int = 1) -> MLXArray {
        guard cache.bin.flattenIds.shape[0] > 0 else { return MLXArray.zeros([camera.height, camera.width, 3]) }
        if cache.replayKMax > 0, let cacheId = cache.replayCacheId, let cacheCount = cache.replayCacheCount {
            let out = Self.featuresReplayKernel(
                [cache.means2d, cache.conics, features.asType(.float16), cache.opacities,
                 MLXArray([Int32(camera.width), Int32(camera.height), Int32(stride)], [3]), cacheId, cacheCount],
                grid: (camera.width / stride, camera.height / stride, 1),
                threadGroup: (16, 16, 1),
                outputShapes: [[camera.width * camera.height * 3]],
                outputDTypes: [.float32],
                initValue: 0.0
            )
            return out[0].reshaped([camera.height, camera.width, 3])
        }
        let dims4 = MLXArray(
            [Int32(camera.width), Int32(camera.height), Int32(camera.tileSize), Int32(cache.bin.numTilesX)], [4])
        let out = rasterizeFwd(
            [cache.means2d, cache.conics, features.asType(.float16), cache.opacities,
             cache.bin.flattenIds, cache.bin.tileOffsets, dims4, MLXArray.zeros([3])],
            grid: (camera.width, camera.height, 1),
            threadGroup: (camera.tileSize, camera.tileSize, 1),
            outputShapes: [[camera.width * camera.height * 3], [camera.width * camera.height],
                           [camera.width * camera.height]],
            outputDTypes: [.float32, .float32, .int32]
        )
        return out[0].reshaped([camera.height, camera.width, 3])
    }

    private static let featuresReplayKernel = SlangCompiledKernel.required("features_replay")

    /// Backward of `renderFeatures` (`rendered` is its output): the parameter gradients through the
    /// alpha weights and dL/d(features) [N, 3]. `vRenderAlpha` adds a cotangent of the forward's
    /// alpha. Uses the replay cache when the forward kept one.
    public func backwardFeatures(
        model: GaussianModel, camera: Camera, cache: ForwardCache, features: MLXArray, rendered: MLXArray,
        vRendered: MLXArray, vRenderAlpha: MLXArray
    ) -> (dMeans: MLXArray, dQuats: MLXArray, dLogScales: MLXArray, dLogitOpacities: MLXArray, vFeatures: MLXArray) {
        let n = model.count
        let totalPixels = camera.width * camera.height
        guard cache.bin.flattenIds.shape[0] > 0 else {
            return (MLXArray.zeros([n, 3]), MLXArray.zeros([n, 4]), MLXArray.zeros([n, 3]), MLXArray.zeros([n]),
                    MLXArray.zeros([n, 3]))
        }
        let featuresHalf = features.asType(.float16)
        let out: [MLXArray]
        if cache.replayKMax > 0, let cacheId = cache.replayCacheId, let cacheCount = cache.replayCacheCount {
            let dims5 = MLXArray(
                [Int32(camera.width), Int32(camera.height), Int32(camera.tileSize),
                 Int32(cache.bin.numTilesX), Int32(cache.replayKMax)], [5])
            let (_, bwdKernel) = try! ensureReplayKernels()
            out = bwdKernel(
                [cache.means2d, cache.conics, featuresHalf, cache.opacities, dims5,
                 rendered.reshaped([totalPixels * 3]), cache.renderAlpha.reshaped([totalPixels]),
                 cacheId, cacheCount,
                 vRendered.reshaped([totalPixels * 3]), vRenderAlpha.reshaped([totalPixels])],
                grid: (camera.width, camera.height, 1),
                threadGroup: (camera.tileSize, camera.tileSize, 1),
                outputShapes: [[n * 2], [n * 3], [n * 3], [n]],
                outputDTypes: [.float32, .float32, .float32, .float32],
                initValue: 0.0
            )
        } else {
            let dims4 = MLXArray(
                [Int32(camera.width), Int32(camera.height), Int32(camera.tileSize), Int32(cache.bin.numTilesX)], [4])
            out = rasterizeBwd(
                [cache.means2d, cache.conics, featuresHalf, cache.opacities,
                 cache.bin.flattenIds, cache.bin.tileOffsets, dims4, MLXArray.zeros([3]),
                 cache.renderAlpha.reshaped([totalPixels]), cache.lastIds,
                 vRendered.reshaped([totalPixels * 3]), vRenderAlpha.reshaped([totalPixels])],
                grid: (camera.width, camera.height, 1),
                threadGroup: (camera.tileSize, camera.tileSize, 1),
                outputShapes: [[n * 2], [n * 3], [n * 3], [n]],
                outputDTypes: [.float32, .float32, .float32, .float32],
                initValue: 0.0
            )
        }
        let g = projectionBackward(
            model: model, camera: camera, cache: cache,
            vMeans2d: out[0].reshaped([n, 2]), vConics: out[1].reshaped([n, 3]), vOpacities: out[3])
        return (g.dMeans, g.dQuats, g.dLogScales, g.dLogitOpacities, out[2].reshaped([n, 3]))
    }

    /// Statistics for diagnostic iters. Reads back to the host, so call it only on
    /// `diagnosticEnabled` steps. `dMeans` is the means gradient via the projection path;
    /// `dMeansFromSH` via the SH direction path (after the normalize chain).
    private func makeDiagnostic(
        model: GaussianModel, cache: ForwardCache,
        dMeans: MLXArray, dMeansFromSH: MLXArray, vColors: MLXArray
    ) -> Diagnostic {
        let n = model.count
        let dMa = MLX.abs(dMeans)
        let gdAbs = MLX.abs(dMeansFromSH)
        let vcAbs = MLX.abs(vColors)
        let scalesAbs = cache.scales
        let opacities = cache.opacities    // [N], in (0,1)
        eval(dMa, gdAbs, vcAbs)
        let radiiHost = cache.radii.asArray(Int32.self)
        let scalesHost = scalesAbs.asArray(Float.self)        // [N*3]
        let opaHost = opacities.asArray(Float.self)           // [N]
        let depthsHost = cache.depths.asArray(Float.self)      // [N]
        var maxX: Int32 = 0
        var maxY: Int32 = 0
        var culled = 0
        var maxRadii: Int = 0
        var maxRadiiIdx: Int = 0
        for i in 0..<n {
            let rx = radiiHost[i * 2 + 0]
            let ry = radiiHost[i * 2 + 1]
            if rx > maxX { maxX = rx }
            if ry > maxY { maxY = ry }
            if rx <= 0 && ry <= 0 { culled += 1 }
            let r = Int(max(rx, ry))
            if r > maxRadii { maxRadii = r; maxRadiiIdx = i }
        }
        var scaleMax: Float = 0
        var scaleAbove1 = 0
        var scaleAbove005 = 0
        var scaleAbove010 = 0
        var scaleAbove020 = 0
        for i in 0..<n {
            let s0 = scalesHost[i*3+0], s1 = scalesHost[i*3+1], s2 = scalesHost[i*3+2]
            let s = max(s0, max(s1, s2))
            if s > scaleMax { scaleMax = s }
            if s > 1.0 { scaleAbove1 += 1 }
            if s > 0.05 { scaleAbove005 += 1 }
            if s > 0.10 { scaleAbove010 += 1 }
            if s > 0.20 { scaleAbove020 += 1 }
        }
        let perGaussMaxScale: [Float] = (0..<n).map {
            max(scalesHost[$0*3], max(scalesHost[$0*3+1], scalesHost[$0*3+2]))
        }
        let scaleMedian = perGaussMaxScale.sorted()[n / 2]
        let opaMax = opaHost.max() ?? 0
        let opaMedian = opaHost.sorted()[n / 2]
        let opaThresh: Float = 1.0 / 255.0
        var opaBelow = 0
        for o in opaHost { if o < opaThresh { opaBelow += 1 } }
        let mri = maxRadiiIdx
        let mriScale = max(scalesHost[mri*3], max(scalesHost[mri*3+1], scalesHost[mri*3+2]))
        return Diagnostic(
            n: n,
            radiiMaxX: maxX, radiiMaxY: maxY, radiiCulled: culled,
            dMeansAbsMax: dMa.max().item(Float.self),
            dMeansAbsMean: dMa.mean().item(Float.self),
            gradDirsAbsMax: gdAbs.max().item(Float.self),
            gradDirsAbsMean: gdAbs.mean().item(Float.self),
            vColorsAbsMax: vcAbs.max().item(Float.self),
            vColorsAbsMean: vcAbs.mean().item(Float.self),
            scaleMax: scaleMax,
            scaleMedian: scaleMedian,
            scaleAbove1Count: scaleAbove1,
            scaleAbove005Count: scaleAbove005,
            scaleAbove010Count: scaleAbove010,
            scaleAbove020Count: scaleAbove020,
            opacityMax: opaMax,
            opacityMedian: opaMedian,
            opacityBelowThreshCount: opaBelow,
            maxRadiiGaussIndex: mri,
            maxRadiiGaussScale: mriScale,
            maxRadiiGaussOpac: opaHost[mri],
            maxRadiiGaussDepth: depthsHost[mri]
        )
    }
}
