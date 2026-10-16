// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX

extension TrainingDriver {

    /// Per-view PSNR / SSIM evaluation results. `*CC` values have the render matched to the target
    /// with a per-channel affine (gsplat `color_correct_affine`); their difference from raw is the
    /// contribution of global color/exposure shifts (AE/AWB drift, etc.).
    public struct EvalResult: Sendable {
        public let perViewPSNR: [Float]
        public let perViewSSIM: [Float]
        public let perViewPSNRCC: [Float]
        public let perViewSSIMCC: [Float]
        public let viewNames: [String]
        /// Per-view maximum per-pixel channel error |render−target|. Local floaters barely move
        /// mean PSNR, so this is a worst-case metric for detecting them.
        public let perViewMaxErr: [Float]
        /// Names of views whose render contained NaN/Inf (canary for fp16 overflow/underflow).
        public let nonFiniteViews: [String]
        public var meanPSNR: Float {
            perViewPSNR.isEmpty ? 0 : perViewPSNR.reduce(0, +) / Float(perViewPSNR.count)
        }
        /// PSNR of the worst view (so floater/broken views are not hidden by the average).
        public var minPSNR: Float { perViewPSNR.min() ?? 0 }
        /// Maximum per-pixel error across all views (strength of the worst floater).
        public var worstMaxErr: Float { perViewMaxErr.max() ?? 0 }
        /// 95th percentile of per-view max per-pixel error (the worst-case band excluding one-off
        /// outliers).
        public var p95MaxErr: Float {
            guard !perViewMaxErr.isEmpty else { return 0 }
            let s = perViewMaxErr.sorted()
            return s[Swift.min(s.count - 1, Int(0.95 * Float(s.count)))]
        }
        public var meanSSIM: Float {
            perViewSSIM.isEmpty ? 0 : perViewSSIM.reduce(0, +) / Float(perViewSSIM.count)
        }
        public var meanPSNRCC: Float {
            perViewPSNRCC.isEmpty ? 0 : perViewPSNRCC.reduce(0, +) / Float(perViewPSNRCC.count)
        }
        public var meanSSIMCC: Float {
            perViewSSIMCC.isEmpty ? 0 : perViewSSIMCC.reduce(0, +) / Float(perViewSSIMCC.count)
        }
    }

    /// Matches the render to the target with a per-channel affine (MLX port of gsplat
    /// `color_correct_affine`, Apache-2.0). Solves `a*ref + b = img` by least squares and maps back
    /// to ref space with `(img - b)/a`. Inputs are [0,1], shape [..., 3].
    static func colorCorrectAffine(render: MLXArray, target: MLXArray) -> MLXArray {
        let c = render.dim(-1)
        let img = render.reshaped([-1, c])      // [N, C]
        let ref = target.reshaped([-1, c])      // [N, C]
        let refMean = ref.mean(axis: 0)                 // [C]
        let imgMean = img.mean(axis: 0)                 // [C]
        let refImgMean = (ref * img).mean(axis: 0)      // [C]
        let refRefMean = (ref * ref).mean(axis: 0)      // [C]
        let varRef = MLX.maximum(refRefMean - refMean * refMean, MLXArray(Float(1e-8)))
        var a = (refImgMean - refMean * imgMean) / varRef    // [C]
        let b = imgMean - a * refMean                        // [C]
        // The inverse diverges when a is near 0, so replace it with 1
        a = MLX.which(MLX.abs(a) .< MLXArray(Float(1e-8)), MLXArray(Float(1)), a)
        let corrected = (img - b) / a                        // broadcast [C] over [N, C]
        let clamped = MLX.minimum(MLX.maximum(corrected, MLXArray(Float(0))), MLXArray(Float(1)))
        return clamped.reshaped(render.shape)
    }

    /// Renders a trained PLY from every view of a COLMAP scene and measures per-view PSNR / SSIM
    /// (no held-out split).
    public static func evaluate(
        plyURL: URL,
        sceneDirectory: URL,
        maxImageDimension: Int = 256,
        tileSize: Int = 16,
        log: (String) -> Void = { print($0) }
    ) throws -> EvalResult {
        let model = try PLYReader.readGaussianPLY(url: plyURL)
        log("loaded \(model.count) gaussians from \(plyURL.lastPathComponent)")

        let scene = try ColmapLoader.load(directory: sceneDirectory)
        log("loaded \(scene.cameras.count) cameras")

        let background = MLXArray([Float(0), 0, 0], [3])
        let imagesDir = sceneDirectory.appendingPathComponent("images")
        let views = ColmapTraining.makeViews(
            scene: scene, imagesDirectory: imagesDir,
            maxDimension: maxImageDimension, tileSize: tileSize, background: background
        )
        log("built \(views.count) eval views (max dim \(maxImageDimension))")

        let renderer = try GaussianRenderer()
        return try evaluate(views: views, log: log) { v in
            renderer.forward(model: model, camera: v.camera, shDegree: 3).renderColor
        }
    }

    /// Measures PSNR / SSIM (raw + color-corrected) over all `views` with an arbitrary render
    /// function (`render(v)` → [H,W,3] float).
    public static func evaluate(
        views: [TrainView], log: (String) -> Void = { print($0) },
        render: (TrainView) -> MLXArray
    ) throws -> EvalResult {
        let ssimKernel = try FusedSSIM()
        var psnrList: [Float] = []
        var ssimList: [Float] = []
        var psnrCCList: [Float] = []
        var ssimCCList: [Float] = []
        var maxErrList: [Float] = []
        var nonFinite: [String] = []
        var names: [String] = []
        func psnr(_ a: MLXArray, _ b: MLXArray) -> (psnr: Float, mse: MLXArray) {
            let diff = a - b
            let mse = (diff * diff).mean()
            let mseHost = mse.item(Float.self)
            return (mseHost > 1e-12 ? -10 * Foundation.log10(mseHost) : 200, mse)
        }
        for v in views {
            let renderColor = render(v)
            let target = v.imageFloat   // uint8→float only once per view
            // NaN/Inf canary. MLXArray has no isFinite, so check the sum on the host (NaN/Inf
            // propagate into the sum).
            if !renderColor.sum().item(Float.self).isFinite { nonFinite.append(v.name) }
            let maxErr = MLX.abs(renderColor - target).max().item(Float.self)
            maxErrList.append(maxErr)
            let (rawPSNR, _) = psnr(renderColor, target)
            let (rawSSIMMap, _, _) = ssimKernel.forward(renderColor, target)
            let rawSSIM = rawSSIMMap.asType(.float32).mean()   // taking the mean of the fp16 map directly overflows the sum to inf
            // color-corrected: the value with global color/exposure shifts removed.
            let corrected = colorCorrectAffine(render: renderColor, target: target)
            let (ccPSNR, _) = psnr(corrected, target)
            let (ccSSIMMap, _, _) = ssimKernel.forward(corrected, target)
            let ccSSIM = ccSSIMMap.asType(.float32).mean()
            eval(rawSSIM, ccSSIM)
            psnrList.append(rawPSNR)
            ssimList.append(rawSSIM.item(Float.self))
            psnrCCList.append(ccPSNR)
            ssimCCList.append(ccSSIM.item(Float.self))
            names.append(v.name)
        }
        let r = EvalResult(
            perViewPSNR: psnrList, perViewSSIM: ssimList,
            perViewPSNRCC: psnrCCList, perViewSSIMCC: ssimCCList,
            viewNames: names, perViewMaxErr: maxErrList, nonFiniteViews: nonFinite)
        log(String(format: "eval mean PSNR=%.3f dB, SSIM=%.4f | color-corrected PSNR=%.3f dB, SSIM=%.4f | ΔPSNR=%.3f dB over %d views",
                   r.meanPSNR, r.meanSSIM, r.meanPSNRCC, r.meanSSIMCC,
                   r.meanPSNRCC - r.meanPSNR, psnrList.count))
        // Floater-sensitive worst-case gate line. For fp16 A/B, look at this rather than the mean.
        log(String(format: "eval worst-case: minPSNR=%.3f dB, maxPixelErr=%.4f, p95PixelErr=%.4f%@",
                   r.minPSNR, r.worstMaxErr, r.p95MaxErr,
                   nonFinite.isEmpty ? "" : " | ⚠️ NaN/Inf in \(nonFinite.count) view(s): \(nonFinite.prefix(3).joined(separator: ","))"))
        return r
    }

    /// Divergence between renders of two PLYs (e.g. fp32 baseline and fp16 candidate) from the same
    /// views. No GT in between, so local floaters show up directly as the max per-pixel divergence.
    public struct ModelComparison: Sendable {
        public let perViewPSNR: [Float]      // render agreement of A vs B (higher = more alike)
        public let perViewMaxErr: [Float]    // max per-pixel channel error of |A−B|
        public let viewNames: [String]
        public let nonFiniteViews: [String]  // either render had NaN/Inf
        public var minPSNR: Float { perViewPSNR.min() ?? 0 }
        public var worstMaxErr: Float { perViewMaxErr.max() ?? 0 }
        public var worstViewName: String {
            guard let i = perViewMaxErr.indices.max(by: { perViewMaxErr[$0] < perViewMaxErr[$1] })
            else { return "" }
            return viewNames[i]
        }
    }

    /// Renders `plyA` (baseline) and `plyB` (candidate) from every view and measures their
    /// divergence.
    public static func compareModels(
        plyA: URL, plyB: URL,
        sceneDirectory: URL,
        maxImageDimension: Int = 256,
        tileSize: Int = 16,
        log: (String) -> Void = { print($0) }
    ) throws -> ModelComparison {
        let modelA = try PLYReader.readGaussianPLY(url: plyA)
        let modelB = try PLYReader.readGaussianPLY(url: plyB)
        log("compare: A=\(modelA.count)g (\(plyA.lastPathComponent)) vs B=\(modelB.count)g (\(plyB.lastPathComponent))")
        let scene = try ColmapLoader.load(directory: sceneDirectory)
        let background = MLXArray([Float(0), 0, 0], [3])
        let views = ColmapTraining.makeViews(
            scene: scene, imagesDirectory: sceneDirectory.appendingPathComponent("images"),
            maxDimension: maxImageDimension, tileSize: tileSize, background: background)
        let renderer = try GaussianRenderer()
        var psnrList: [Float] = [], maxErrList: [Float] = [], names: [String] = []
        var nonFinite: [String] = []
        for v in views {
            let a = renderer.forward(model: modelA, camera: v.camera, shDegree: 3).renderColor
            let b = renderer.forward(model: modelB, camera: v.camera, shDegree: 3).renderColor
            if !(a.sum().item(Float.self).isFinite && b.sum().item(Float.self).isFinite) {
                nonFinite.append(v.name)
            }
            let diff = a - b
            let mse = (diff * diff).mean().item(Float.self)
            psnrList.append(mse > 1e-12 ? -10 * Foundation.log10(mse) : 200)
            maxErrList.append(MLX.abs(diff).max().item(Float.self))
            names.append(v.name)
        }
        let c = ModelComparison(perViewPSNR: psnrList, perViewMaxErr: maxErrList,
                                viewNames: names, nonFiniteViews: nonFinite)
        log(String(format: "compare worst-case: minPSNR(A vs B)=%.2f dB, maxPixelErr=%.4f @ %@%@",
                   c.minPSNR, c.worstMaxErr, c.worstViewName,
                   nonFinite.isEmpty ? "" : " | ⚠️ NaN/Inf in \(nonFinite.count) view(s)"))
        return c
    }
}
