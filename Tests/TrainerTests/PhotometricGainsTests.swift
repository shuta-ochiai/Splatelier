// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import Testing
@testable import Trainer

@Test func photometricGainsStartAtOne() {
    let gains = PhotometricGains(viewCount: 4)
    for i in 0..<4 {
        let g = gains.gain(view: i)
        #expect(g == SIMD3<Float>(1, 1, 1))
    }
}

/// A fit moves the gain part of the way to `Σxt / Σx²`, the mean log gain over the views stays 0,
/// and the gain stays inside the bound.
@Test func photometricGainsFitMovesTowardsLeastSquaresGain() {
    let config = PhotometricGainConfig(startFraction: 0, rate: 1, maxLogGain: 0.7)
    let gains = PhotometricGains(viewCount: 2, config: config)
    // Exact fit for view 1: the image is the render × (2, 1, 0.5).
    gains.fit(view: 1, crossSum: SIMD3(2, 1, 0.5), squareSum: SIMD3(1, 1, 1))
    let g0 = gains.gain(view: 0), g1 = gains.gain(view: 1)
    // The mean removal splits the difference between the two views, so only the ratio is known.
    // log 2 = 0.693 is inside the bound.
    #expect(abs(g1.x / g0.x - 2) < 1e-3)
    #expect(abs(g1.y / g0.y - 1) < 1e-3)
    #expect(abs(g1.z / g0.z - 0.5) < 1e-3)
    for c in 0..<3 { #expect(abs(gains.logGain(view: 0)[c] + gains.logGain(view: 1)[c]) < 1e-5) }

    // A gain far from 1 is cut at the bound; a channel without any render says nothing.
    let bounded = PhotometricGains(viewCount: 1, config: config)
    bounded.fit(view: 0, crossSum: SIMD3(100, 0, 1), squareSum: SIMD3(1, 0, 1))
    #expect(bounded.logGains[0] <= 0.7 + 1e-6)
    #expect(bounded.logGains[1] == 0)
}

private func makeCamera() -> Camera {
    Camera(
        viewmat: MLXArray([Float(1), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [16]),
        intrinsics: MLXArray([Float(80), Float(80), Float(16), Float(16)], [4]),
        width: 32, height: 32, tileSize: 16,
        background: MLXArray([Float(0), Float(0), Float(0)], [3]))
}

private func makeModel(colors: [Float], opacity: Float) -> GaussianModel {
    let n = 2
    return GaussianModel(
        means: MLXArray([Float(-0.4), 0.3, 4.0, 0.5, -0.2, 4.0], [n, 3]),
        quats: MLXArray([Float(1), 0, 0, 0, 1, 0, 0, 0], [n, 4]),
        logScales: MLXArray([Float](repeating: log(0.4), count: n * 3), [n, 3]),
        logitOpacities: MLXArray([logit(opacity), logit(opacity)], [n]),
        sh0: (MLXArray(colors, [n, 3]) - 0.5) / GaussianModel.shC0,
        shN: MLXArray.zeros([n, GaussianModel.shRestBands, 3]))
}

/// Two images of one scene, the second shot at another exposure and white balance. Without gains
/// the scene cannot match both; with them the loss gets much lower and the fitted gains of the two
/// views differ by the exposure change.
@Test func trainerFitsAnExposureChangeBetweenViews() throws {
    let renderer = try GaussianRenderer()
    let cam = makeCamera()
    let (image, _, _) = renderer.forward(
        model: makeModel(colors: [0.6, 0.2, 0.1, 0.1, 0.5, 0.3], opacity: 0.8), camera: cam)
    let factors: [Float] = [1.5, 1.0, 0.7]
    let changed = image * MLXArray(factors, [3])
    eval(image, changed)
    let targets = [image, changed]

    func run(gains: PhotometricGains?, background: Bool = false) throws -> (loss: Float, trainer: GaussianTrainer) {
        let trainer = try GaussianTrainer(
            model: makeModel(colors: [0.4, 0.4, 0.4, 0.4, 0.4, 0.4], opacity: 0.8), renderer: renderer,
            meansLR: 0, quatsLR: 0, logScalesLR: 0, logitOpacitiesLR: 0, sh0LR: 2e-2,
            enableBackgroundSH: background, backgroundSHDegree: 0, bgCoeffsLR: 2e-2, alphaLossWeight: 0.15)
        trainer.photometricGains = gains
        var last: Float = 0
        for i in 0..<400 {
            last = trainer.step(camera: cam, targetImage: targets[i % 2], shDegree: 0,
                                viewIndex: gains == nil ? nil : i % 2, fitGain: gains != nil)
        }
        return (last, trainer)
    }
    let plain = try run(gains: nil)
    let gains = PhotometricGains(viewCount: 2, config: PhotometricGainConfig(startFraction: 0, rate: 0.5, maxLogGain: 0.7))
    let fitted = try run(gains: gains)
    let g0 = gains.gain(view: 0), g1 = gains.gain(view: 1)
    print("photometric gains: loss without \(plain.loss), with \(fitted.loss); gain ratio \(g1 / g0) vs \(factors)")
    #expect(fitted.loss < plain.loss * 0.5, "the gains did not lower the loss: \(plain.loss) vs \(fitted.loss)")
    for c in 0..<3 { #expect(abs(g1[c] / g0[c] / factors[c] - 1) < 0.1, "channel \(c): \(g1[c] / g0[c]) vs \(factors[c])") }
}

/// The same with the background environment map on, which the app always uses: the gain scales the
/// whole composite, and the alpha loss's background test sees the image before the gain.
@Test func trainerFitsAnExposureChangeWithBackgroundSH() throws {
    let renderer = try GaussianRenderer()
    let cam = makeCamera()
    // The scene on a grey backdrop, so the background has something to explain.
    let (render, alpha, _) = renderer.forward(
        model: makeModel(colors: [0.6, 0.2, 0.1, 0.1, 0.5, 0.3], opacity: 0.8), camera: cam)
    let image = render + (1 - alpha.reshaped([32, 32, 1])) * 0.3
    let factors: [Float] = [1.4, 1.0, 0.75]
    let targets = [image, image * MLXArray(factors, [3])]
    eval(targets[0], targets[1])

    let trainer = try GaussianTrainer(
        model: makeModel(colors: [0.4, 0.4, 0.4, 0.4, 0.4, 0.4], opacity: 0.8), renderer: renderer,
        meansLR: 0, quatsLR: 0, logScalesLR: 0, logitOpacitiesLR: 0, sh0LR: 2e-2,
        enableBackgroundSH: true, backgroundSHDegree: 0, bgCoeffsLR: 2e-2, alphaLossWeight: 0.15)
    let gains = PhotometricGains(viewCount: 2, config: PhotometricGainConfig(startFraction: 0, rate: 0.5, maxLogGain: 0.7))
    trainer.photometricGains = gains
    var first: Float = 0, last: Float = 0
    for i in 0..<500 {
        last = trainer.step(camera: cam, targetImage: targets[i % 2], shDegree: 0, viewIndex: i % 2, fitGain: true)
        if i == 0 { first = last }
    }
    let g0 = gains.gain(view: 0), g1 = gains.gain(view: 1)
    print("photometric gains with bg-SH: loss \(first) -> \(last); gain ratio \(g1 / g0) vs \(factors)")
    #expect(last < first * 0.2)
    for c in 0..<3 { #expect(abs(g1[c] / g0[c] / factors[c] - 1) < 0.15, "channel \(c): \(g1[c] / g0[c]) vs \(factors[c])") }
}
