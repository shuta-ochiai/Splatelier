// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXRandom
import Testing
@testable import Trainer

private func makeCamera() -> Camera {
    let viewmat = MLXArray(
        [Float(1), Float(0), Float(0), Float(0),
         Float(0), Float(1), Float(0), Float(0),
         Float(0), Float(0), Float(1), Float(0),
         Float(0), Float(0), Float(0), Float(1)],
        [16]
    )
    return Camera(
        viewmat: viewmat,
        intrinsics: MLXArray([Float(80), Float(80), Float(16), Float(16)], [4]),
        width: 32, height: 32, tileSize: 16,
        background: MLXArray([Float(0), Float(0), Float(0)], [3])
    )
}

/// Target model: 2 fairly large, separated Gaussians with known positions and colors. Forward
/// renders the target image.
private func makeTargetModel() -> GaussianModel {
    let n = 2
    return GaussianModel(
        means: MLXArray([Float(-0.4), Float(0.3), Float(4.0),
                         Float(0.5), Float(-0.2), Float(4.0)], [n, 3]),
        quats: MLXArray([Float(1), Float(0), Float(0), Float(0),
                          Float(1), Float(0), Float(0), Float(0)], [n, 4]),
        logScales: MLXArray([Float](repeating: log(0.4), count: n * 3), [n, 3]),
        logitOpacities: MLXArray([logit(0.8), logit(0.8)], [n]),
        sh0: (MLXArray([Float(0.9), Float(0.2), Float(0.1),
                        Float(0.1), Float(0.8), Float(0.3)], [n, 3]) - 0.5) / GaussianModel.shC0,
        shN: MLXArray.zeros([n, GaussianModel.shRestBands, 3])
    )
}

/// Initial model with the same count and parameters offset from target.
private func makeSourceModel() -> GaussianModel {
    let n = 2
    return GaussianModel(
        means: MLXArray([Float(0.0), Float(0.0), Float(4.0),
                         Float(0.0), Float(0.0), Float(4.0)], [n, 3]),
        quats: MLXArray([Float(1), Float(0), Float(0), Float(0),
                          Float(1), Float(0), Float(0), Float(0)], [n, 4]),
        logScales: MLXArray([Float](repeating: log(0.4), count: n * 3), [n, 3]),
        logitOpacities: MLXArray([logit(0.5), logit(0.5)], [n]),
        sh0: MLXArray.zeros([n, 3]),   // rgb 0.5 ⇒ sh0 = 0
        shN: MLXArray.zeros([n, GaussianModel.shRestBands, 3])
    )
}

@Test
func trainerOverfitsSingleImage() throws {
    let renderer = try GaussianRenderer()
    let cam = makeCamera()

    let target = makeTargetModel()
    let (targetImage, _, _) = renderer.forward(model: target, camera: cam)
    eval(targetImage)

    let source = makeSourceModel()
    // Single-view overfit, so use a higher LR.
    let trainer = try GaussianTrainer(
        model: source, renderer: renderer,
        meansLR: 5e-3,
        quatsLR: 1e-3,
        logScalesLR: 5e-3,
        logitOpacitiesLR: 5e-2,
        // rgb = C0·sh0 + 0.5 scales the effective color step by C0 (~0.28), so set it higher.
        sh0LR: 4e-2
    )

    let initialLoss = trainer.step(camera: cam, targetImage: targetImage)
    var lastLoss: Float = initialLoss
    let steps = 200
    for _ in 0..<steps {
        lastLoss = trainer.step(camera: cam, targetImage: targetImage)
    }

    print("trainer loss: initial=\(initialLoss), after \(steps) steps=\(lastLoss)")
    #expect(lastLoss < initialLoss,
            "loss didn't decrease: initial=\(initialLoss), final=\(lastLoss)")
    // A 2-Gaussian toy scene drops below 1/10 in 200 steps.
    #expect(lastLoss < initialLoss * 0.1,
            "loss only dropped to 10%% at best after \(steps) steps: initial=\(initialLoss), final=\(lastLoss)")
}

/// Overfit with bg-SH (background SH environment map) enabled. Over the full path of compositing +
/// alpha loss + bg coefficient gradient + Adam, loss decreases and bgCoeffs are actually updated.
@Test func trainerWithBackgroundSHTrainsAndUpdatesCoeffs() throws {
    let renderer = try GaussianRenderer()
    let cam = makeCamera()

    let target = makeTargetModel()
    let (targetImage, _, _) = renderer.forward(model: target, camera: cam)
    eval(targetImage)

    let source = makeSourceModel()
    let trainer = try GaussianTrainer(
        model: source, renderer: renderer,
        meansLR: 5e-3, quatsLR: 1e-3, logScalesLR: 5e-3,
        logitOpacitiesLR: 5e-2, sh0LR: 4e-2,
        enableBackgroundSH: true, backgroundSHDegree: 3, bgCoeffsLR: 1e-2
    )

    let coeffs0 = trainer.bgCoeffs.asArray(Float.self)
    let initialLoss = trainer.step(camera: cam, targetImage: targetImage)
    var lastLoss = initialLoss
    for _ in 0..<200 { lastLoss = trainer.step(camera: cam, targetImage: targetImage) }
    eval(trainer.bgCoeffs)
    let coeffs1 = trainer.bgCoeffs.asArray(Float.self)

    print("bg-SH overfit loss: initial=\(initialLoss), final=\(lastLoss)")
    #expect(lastLoss < initialLoss * 0.3,
            "bg-SH overfit did not converge: \(initialLoss) -> \(lastLoss)")
    var maxDelta: Float = 0
    for i in 0..<coeffs0.count { maxDelta = max(maxDelta, abs(coeffs1[i] - coeffs0[i])) }
    #expect(maxDelta > 1e-4,
            "bgCoeffs did not update — bg gradient not flowing (maxDelta=\(maxDelta))")
}
