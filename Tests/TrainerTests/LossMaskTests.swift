// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import Testing
@testable import Trainer

/// A 64×32 camera at the origin looking down +z.
private func maskTestCamera() -> Camera {
    Camera(
        viewmat: MLXArray([Float(1), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [16]),
        intrinsics: MLXArray([Float(80), 80, 32, 16], [4]),
        width: 64, height: 32, tileSize: 16,
        background: MLXArray([Float(0), 0, 0], [3]))
}

/// Two small dark Gaussians 2 m ahead: one projects to x≈12 px (left half), one to x≈52 px
/// (right half), each more than an SSIM window away from the middle of the image.
private func leftAndRightGaussians() -> GaussianModel {
    GaussianModel(
        means: MLXArray([Float(-0.5), 0, 2, 0.5, 0, 2], [2, 3]),
        quats: MLXArray([Float(1), 0, 0, 0, 1, 0, 0, 0], [2, 4]),
        logScales: MLXArray([Float](repeating: Foundation.log(0.02), count: 6), [2, 3]),
        logitOpacities: MLXArray([Float(0), 0], [2]),
        sh0: MLXArray.zeros([2, 3]),
        shN: MLXArray.zeros([2, GaussianModel.shRestBands, 3]))
}

@Test
func maskedTargetTakesThePredictionWhereIgnored() {
    let target = MLXArray([Float(1), 1, 1, 1, 1, 1], [1, 2, 3])
    let predicted = MLXArray([Float(0.2), 0.2, 0.2, 0.4, 0.4, 0.4], [1, 2, 3])
    let mask = MLXArray([Float(0), 1], [1, 2, 1])
    let out = GaussianTrainer.maskedTarget(target, predicted: predicted, mask: mask).asArray(Float.self)
    #expect(out == [0.2, 0.2, 0.2, 1, 1, 1])
    #expect(GaussianTrainer.maskedTarget(target, predicted: predicted, mask: nil).asArray(Float.self)
            == target.asArray(Float.self))
}

/// Trains a few steps towards a white image and returns how far each Gaussian's color moved.
private func colorChange(lossMask: MLXArray?, lambdaSSIM: Float, backgroundSH: Bool) throws -> [Float] {
    let model = leftAndRightGaussians()
    let trainer = try GaussianTrainer(model: model, renderer: try GaussianRenderer(), lambdaSSIM: lambdaSSIM,
                                      enableBackgroundSH: backgroundSH)
    let camera = maskTestCamera()
    let target = MLXArray.ones([camera.height, camera.width, 3])
    let before = model.sh0.asArray(Float.self)
    for _ in 0..<3 {
        trainer.step(camera: camera, targetImage: target, shDegree: 0, lossMask: lossMask)
    }
    let after = model.sh0.asArray(Float.self)
    return (0..<2).map { g in (0..<3).map { abs(after[g * 3 + $0] - before[g * 3 + $0]) }.max()! }
}

// L1 alone, L1 + D-SSIM, and with the background SH (whose alpha loss must skip ignored pixels too).
@Test(arguments: [(Float(0), false), (0.2, false), (0.2, true)])
func ignoredPixelsDoNotMoveTheGaussiansOnThem(lambdaSSIM: Float, backgroundSH: Bool) throws {
    // Without a mask both Gaussians brighten towards the white image.
    let free = try colorChange(lossMask: nil, lambdaSSIM: lambdaSSIM, backgroundSH: backgroundSH)
    #expect(free[0] > 1e-4 && free[1] > 1e-4)
    // The left half ignored: the left Gaussian gets no gradient, the right one still learns.
    let masked = try colorChange(lossMask: MLXArray([Float(0), 1], [1, 2]), lambdaSSIM: lambdaSSIM,
                                 backgroundSH: backgroundSH)
    #expect(masked[0] == 0)
    #expect(masked[1] > 1e-4)
}

/// The view's stored log gain (red) after one fitting step towards an image that is white on the
/// left and nearly black on the right.
private func fittedLogGain(lossMask: MLXArray?) throws -> Float {
    let model = leftAndRightGaussians()
    let trainer = try GaussianTrainer(model: model, renderer: try GaussianRenderer())
    trainer.photometricGains = PhotometricGains(viewCount: 1)
    let camera = maskTestCamera()
    let left = MLXArray.ones([camera.height, camera.width / 2, 3])
    let right = MLXArray.full([camera.height, camera.width / 2, 3], values: MLXArray(Float(0.02)))
    let target = MLX.concatenated([left, right], axis: 1)
    trainer.step(camera: camera, targetImage: target, shDegree: 0, lossMask: lossMask,
                 viewIndex: 0, fitGain: true)
    return trainer.photometricGains!.logGains[0]
}

@Test
func ignoredPixelsStayOutOfTheGainFit() throws {
    // Both halves: the white left half dominates and the gain brightens the render.
    #expect(try fittedLogGain(lossMask: nil) > 0)
    // The left half ignored: only the dark right half is fitted, so the gain darkens it.
    #expect(try fittedLogGain(lossMask: MLXArray([Float(0), 1], [1, 2])) < 0)
}
