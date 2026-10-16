// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Geometric regularization of training, after Geometry-Grounded Gaussian Splatting (Zhang et al.,
// arXiv 2601.17835, Sec. 4.2-4.3 and App. C; implemented from the paper only): the surface depth is
// the 0.5 crossing of the stochastic-solid transmittance (`GaussianSurfaceDepth`), differentiated to
// every Gaussian along the ray, and two losses act on it:
//   - normal consistency (2DGS, Huang et al. 2024, as G³S App. C eq. 25): the alpha-composited
//     Gaussian normals N should agree with the normal ñ of the depth map, L_n = mean(α - N·ñ). A
//     Gaussian's normal is that of the plane its ray-wise peaks lie on under the local affine
//     approximation, Σ⁻¹d (RaDe-GS), turned toward the camera (geo_reg.slang).
//   - MVS depth: in place of G³S's multi-view photometric/geometric terms, the depth should match
//     the training's own MVS depth maps where those exist, L_mvs = mean(|t - d_mvs| / d_mvs).

import Foundation
import MLX

/// An MVS depth map of a training view (same pose, its own resolution and intrinsics).
public struct DepthTarget: @unchecked Sendable {
    public let width: Int
    public let height: Int
    /// [height * width] camera-space depth, <= 0 where MVS gave none.
    public let depth: MLXArray
    public let intrinsics: Intrinsics

    public init(width: Int, height: Int, depth: [Float], intrinsics: Intrinsics) {
        precondition(depth.count == width * height)
        self.width = width
        self.height = height
        self.depth = MLXArray(depth, [width * height])
        self.intrinsics = intrinsics
    }
}

public struct GeometryRegularization: Sendable {
    /// Weight of the normal-consistency loss per step (G³S uses 0.05). With `every` > 1 the steps
    /// that regularize use `every` times the weights, keeping the average.
    public var normalWeight: Float
    /// Weight of the relative MVS depth loss per step.
    public var mvsDepthWeight: Float
    /// Share of the iterations before the regularization starts (G³S: 7K of 30K).
    public var startFraction: Float
    /// Regularize every n-th step from the start on.
    public var every: Int
    /// The losses are taken at every s-th pixel of the step's view in each direction, with s
    /// chosen so that about this many samples span its width (the depth search runs only there;
    /// the normal map is composited at full resolution and sampled).
    public var sampleWidth: Int
    public var depth: GaussianSurfaceDepth.Parameters

    public init(normalWeight: Float = 0.05, mvsDepthWeight: Float = 0.05, startFraction: Float = 0.3,
                every: Int = 8, sampleWidth: Int = 180,
                depth: GaussianSurfaceDepth.Parameters = .init(profile: .stochasticSolid, bisectionIterations: 9)) {
        precondition(normalWeight >= 0 && mvsDepthWeight >= 0 && startFraction >= 0 && every >= 1)
        precondition(sampleWidth >= 8)
        self.normalWeight = normalWeight
        self.mvsDepthWeight = mvsDepthWeight
        self.startFraction = startFraction
        self.every = every
        self.sampleWidth = sampleWidth
        self.depth = depth
    }

    private static let normalsKernel = SlangCompiledKernel.required("geo_normals_fwd")
    private static let normalsBackwardKernel = SlangCompiledKernel.required("geo_normals_bwd")

    /// Evaluate and time each stage for the training profiler (adds GPU syncs; off by default).
    public nonisolated(unsafe) static var profileStages = false

    public struct Losses: Sendable {
        public var normal: Float = 0
        public var mvsDepth: Float = 0
        /// Share of the pixels with a surface depth.
        public var depthCoverage: Float = 0
    }

    /// Gradients of the regularization w.r.t. the raw parameters for one view (the step's camera
    /// and forward cache), with the loss values. nil if nothing is in view.
    func gradients(
        model: GaussianModel, renderer: GaussianRenderer, camera: Camera, cache: GaussianRenderer.ForwardCache,
        mvsDepth: DepthTarget?
    ) -> (dMeans: MLXArray, dQuats: MLXArray, dLogScales: MLXArray, dLogitOpacities: MLXArray,
          losses: () -> Losses)? {
        // Per-stage timing for the training profiler (each stage is evaluated at its end).
        var t0 = Date()
        func stage(_ name: String, _ arrays: MLXArray...) {
            guard Self.profileStages, let hook = GaussianTrainer.profileHook else { return }
            eval(arrays)
            hook(name, Date().timeIntervalSince(t0) * 1000)
            t0 = Date()
        }
        let h = camera.height, w = camera.width
        let stride = Swift.max(1, Int((Float(w) / Float(sampleWidth)).rounded()))
        var depthParameters = depth
        depthParameters.stride = stride
        guard let surface = GaussianSurfaceDepth.render(
            model: model, camera: camera, cache: cache, parameters: depthParameters) else { return nil }
        stage("geo_depth_fwd", surface.depth)
        let normals = Self.cameraNormals(means: model.means, quats: model.quats, scales: cache.scales, camera: camera)
        let normalMap = renderer.renderFeatures(normals, camera: camera, cache: cache, stride: stride)
        stage("geo_normals", normals, normalMap)

        // The samples: pixel (i s + s/2, j s + s/2) of the view for entry (i, j) of the depth map.
        let intr = camera.intrinsics.asArray(Float.self)
        let (hs, ws) = (h / stride, w / stride)
        let offset = stride / 2
        let columns = (0 ..< ws).map { $0 * stride + offset }
        let rows = (0 ..< hs).map { $0 * stride + offset }
        let xs = columns.map { (Float($0) + 0.5 - intr[2]) / intr[0] }
        let ys = rows.map { (Float($0) + 0.5 - intr[3]) / intr[1] }
        let samplePixel = MLXArray(rows.flatMap { r in columns.map { Int32(r * w + $0) } })

        // Constant masks: valid depth with valid neighbors and no depth discontinuity (normals),
        // and the samples with an MVS depth.
        let t = surface.depth
        let valid = (t .> 0).asType(.float32)
        let mvs = mvsDepth.map { Self.mvsSamples($0, xs: xs, ys: ys) }

        let wn = normalWeight * Float(every), wm = mvsDepthWeight * Float(every)
        let lossFn: ([MLXArray]) -> [MLXArray] = { x in
            let depthMap = x[0]
            let nMap = x[1].reshaped([h * w, 3]).take(samplePixel, axis: 0).reshaped([hs, ws, 3])
            let alpha = x[2].reshaped([h * w]).take(samplePixel).reshaped([hs, ws])
            let (n, nMask) = Self.depthNormals(depthMap, valid: valid, xs: xs, ys: ys)
            let inner = { (a: MLXArray) in a[1 ..< (hs - 1), 1 ..< (ws - 1)] }
            let nCount = MLX.maximum(nMask.sum(), MLXArray(Float(1)))
            let normalLoss = (nMask * (inner(alpha) - (inner(nMap) * n).sum(axis: 2))).sum() / nCount
            var total = wn * normalLoss
            var mvsLoss = MLXArray(Float(0))
            if let mvs {
                let m = mvs.mask * valid
                let mCount = MLX.maximum(m.sum(), MLXArray(Float(1)))
                mvsLoss = (m * MLX.abs(depthMap - mvs.target) / mvs.target).sum() / mCount
                total = total + wm * mvsLoss
            }
            return [total, normalLoss, mvsLoss]
        }
        let (values, cotangents) = vjp(
            { lossFn($0) }, primals: [t, normalMap, cache.renderAlpha],
            cotangents: [MLXArray(Float(1)), MLXArray(Float(0)), MLXArray(Float(0))])
        let vDepth = cotangents[0] * valid, vNormalMap = cotangents[1], vAlpha = cotangents[2]
        stage("geo_loss", vDepth, vNormalMap, vAlpha)

        let gd = GaussianSurfaceDepth.backward(surface, model: model, camera: camera, cache: cache, vDepth: vDepth)
        stage("geo_depth_bwd", gd.dMeans, gd.dQuats, gd.dLogScales, gd.dLogitOpacities)
        let gn = renderer.backwardFeatures(
            model: model, camera: camera, cache: cache, features: normals, rendered: normalMap,
            vRendered: vNormalMap, vRenderAlpha: vAlpha)
        stage("geo_normal_bwd", gn.dMeans, gn.vFeatures)
        let gf = Self.cameraNormalsBackward(
            means: model.means, quats: model.quats, scales: cache.scales, camera: camera, vNormals: gn.vFeatures)
        stage("geo_chain", gf.dMeans, gf.dQuats, gf.dScales)

        // Computed along with the gradients, read back only when asked.
        let coverage = valid.mean()
        asyncEval(values[1], values[2], coverage)
        let losses = {
            Losses(normal: values[1].item(Float.self), mvsDepth: values[2].item(Float.self),
                   depthCoverage: coverage.item(Float.self))
        }
        return (gd.dMeans + gn.dMeans + gf.dMeans,
                gd.dQuats + gn.dQuats + gf.dQuats,
                gd.dLogScales + gn.dLogScales + gf.dScales * cache.scales,
                gd.dLogitOpacities + gn.dLogitOpacities,
                losses)
    }

    /// Camera-space normals [N, 3] of the Gaussians (geo_normals_fwd): Σ⁻¹d = R S⁻² Rᵀ d with
    /// d = μ - campos, normalized and turned toward the camera.
    static func cameraNormals(means: MLXArray, quats: MLXArray, scales: MLXArray, camera: Camera) -> MLXArray {
        let n = means.shape[0]
        return normalsKernel(
            [means, quats, scales, camera.viewmat, MLXArray([Int32(n)], [1])],
            grid: (((n + 63) / 64) * 64, 1, 1), threadGroup: (64, 1, 1),
            outputShapes: [[n * 3]], outputDTypes: [.float32]
        )[0].reshaped([n, 3])
    }

    /// Backward of `cameraNormals` w.r.t. means, quats and (activated) scales (geo_normals_bwd).
    static func cameraNormalsBackward(
        means: MLXArray, quats: MLXArray, scales: MLXArray, camera: Camera, vNormals: MLXArray
    ) -> (dMeans: MLXArray, dQuats: MLXArray, dScales: MLXArray) {
        let n = means.shape[0]
        let out = normalsBackwardKernel(
            [means, quats, scales, camera.viewmat, MLXArray([Int32(n)], [1]), vNormals],
            grid: (((n + 63) / 64) * 64, 1, 1), threadGroup: (64, 1, 1),
            outputShapes: [[n * 3], [n * 4], [n * 3]], outputDTypes: [.float32, .float32, .float32]
        )
        return (out[0].reshaped([n, 3]), out[1].reshaped([n, 4]), out[2].reshaped([n, 3]))
    }

    /// Normals ñ [H-2, W-2, 3] of a depth map from central differences of its back-projected
    /// points, facing the camera, with the mask of the entries where they are meaningful. `xs` /
    /// `ys` are the normalized image coordinates of its columns / rows.
    static func depthNormals(_ t: MLXArray, valid: MLXArray, xs: [Float], ys: [Float]) -> (MLXArray, MLXArray) {
        let h = t.shape[0], w = t.shape[1]
        let px = t * MLXArray(xs).reshaped([1, w]), py = t * MLXArray(ys).reshaped([h, 1])
        func dx(_ a: MLXArray) -> MLXArray { a[1 ..< (h - 1), 2 ..< w] - a[1 ..< (h - 1), 0 ..< (w - 2)] }
        func dy(_ a: MLXArray) -> MLXArray { a[2 ..< h, 1 ..< (w - 1)] - a[0 ..< (h - 2), 1 ..< (w - 1)] }
        let ax = dx(px), ay = dx(py), az = dx(t)
        let bx = dy(px), by = dy(py), bz = dy(t)
        // -(a × b): x right, y down, z forward, so a fronto-parallel plane gets (0, 0, -1).
        let cx = -(ay * bz - az * by), cy = -(az * bx - ax * bz), cz = -(ax * by - ay * bx)
        let len = MLX.sqrt(MLX.maximum(cx * cx + cy * cy + cz * cz, MLXArray(Float(1e-20))))
        let n = MLX.stacked([cx / len, cy / len, cz / len], axis: 2)
        // Both neighbors on each axis valid, and no jump over 5% of the depth between them.
        let center = t[1 ..< (h - 1), 1 ..< (w - 1)]
        let neighbors = valid[1 ..< (h - 1), 1 ..< (w - 1)]
            * valid[1 ..< (h - 1), 2 ..< w] * valid[1 ..< (h - 1), 0 ..< (w - 2)]
            * valid[2 ..< h, 1 ..< (w - 1)] * valid[0 ..< (h - 2), 1 ..< (w - 1)]
        let smooth = ((MLX.abs(az) .< 0.05 * center) .&& (MLX.abs(bz) .< 0.05 * center)).asType(.float32)
        return (n, stopGradient(neighbors * smooth))
    }

    /// The MVS depth at each sample (same pose; the MVS pixel containing the sample's ray), [rows,
    /// columns], with the mask of the samples that land on an MVS depth.
    static func mvsSamples(_ target: DepthTarget, xs: [Float], ys: [Float]) -> (target: MLXArray, mask: MLXArray) {
        let ti = target.intrinsics
        var index = [Int32](repeating: 0, count: xs.count * ys.count)
        var inside = [Float](repeating: 0, count: index.count)
        for (i, y) in ys.enumerated() {
            let v = (y * ti.fy + ti.cy).rounded(.down)
            for (j, x) in xs.enumerated() {
                let u = (x * ti.fx + ti.cx).rounded(.down)
                guard u >= 0, v >= 0, u < Float(target.width), v < Float(target.height) else { continue }
                index[i * xs.count + j] = Int32(v) * Int32(target.width) + Int32(u)
                inside[i * xs.count + j] = 1
            }
        }
        let depth = target.depth.take(MLXArray(index)).reshaped([ys.count, xs.count])
        let mask = MLXArray(inside).reshaped([ys.count, xs.count]) * (depth .> 0).asType(.float32)
        return (MLX.maximum(depth, MLXArray(Float(1e-3))), mask)
    }
}
