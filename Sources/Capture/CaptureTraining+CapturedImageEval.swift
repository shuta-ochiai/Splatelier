// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Held-out evaluation against the images as captured (distorted, not resampled) as ground truth.
// Used to compare a model trained on undistorted images with one trained without, against the same
// ground truth (undistorted ground-truth images are blurred by resampling, so PSNR / SSIM against
// them rises just because they are easier to match). Both are rendered at `scale` times the
// resolution and resampled back to the capture pixel grid with the same Catmull-Rom bicubic (the
// undistorted model is sampled at the redistorted positions). The `margin` px border includes the
// area cropped by undistortion, so it is excluded for both.

import Foundation
import MLX
import Trainer

extension CaptureTraining {

    static func evaluateAgainstCapturedImages(
        frames: [CapturedFrame], trainer: GaussianTrainer, tileSize: Int, background: MLXArray,
        scale: Int = 2, margin: Int = 8, log: (String) -> Void
    ) throws {
        var views: [TrainView] = []
        var framesByName: [String: CapturedFrame] = [:]
        for (i, f) in frames.enumerated() {
            let name = String(format: "captured_%06d", i)
            let target = f.originalImage[margin..<(f.height - margin), margin..<(f.width - margin)]
            eval(target)
            // Rendering happens in the closure below, so this camera only exists to satisfy
            // TrainView's shape.
            let k = f.originalIntrinsics
            let camera = Camera(
                viewmat: MLXArray(f.viewmat, [16]),
                intrinsics: MLXArray([k.fx, k.fy, k.cx - Float(margin), k.cy - Float(margin)], [4]),
                width: f.width - 2 * margin, height: f.height - 2 * margin,
                tileSize: tileSize, background: background)
            views.append(TrainView(camera: camera, image: target, name: name))
            framesByName[name] = f
        }
        let s = Float(scale)
        _ = try TrainingDriver.evaluate(views: views, log: log) { v in
            let f = framesByName[v.name]!
            let k = f.intrinsics
            let camera = Camera(
                viewmat: MLXArray(f.viewmat, [16]),
                intrinsics: MLXArray([k.fx * s, k.fy * s, k.cx * s, k.cy * s], [4]),
                width: f.width * scale, height: f.height * scale,
                tileSize: tileSize, background: background)
            let render = trainer.renderForEval(camera: camera)
            let (xs, ys) = LensUndistortion.captureToModelCoordinates(
                width: f.width, height: f.height, capture: f.originalIntrinsics, model: k,
                distortion: f.lensUndistortion?.distortion ?? .none, scale: s)
            let sampled = LensUndistortion.sampleBicubic(
                render, x: MLXArray(xs, [xs.count]), y: MLXArray(ys, [ys.count])
            ).reshaped([f.height, f.width, 3])
            return MLX.clip(sampled[margin..<(f.height - margin), margin..<(f.width - margin)], min: 0, max: 1)
        }
    }
}
