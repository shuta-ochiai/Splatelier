// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import Testing
@testable import Trainer

private func makeStoreTestCamera() -> Camera {
    let viewmat = MLXArray(
        [Float(1), 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1, 0,
         0, 0, 0, 1],
        [16]
    )
    return Camera(
        viewmat: viewmat,
        intrinsics: MLXArray([Float(80), 80, 16, 16], [4]),
        width: 32, height: 32, tileSize: 16,
        background: MLXArray([Float(0), 0, 0], [3])
    )
}

@Test
func updateParamsWithOptimizerAppendKeepsModelAndStateConsistent() throws {
    let renderer = try GaussianRenderer()
    let model = GaussianModel.random(count: 5, seed: 1)
    let trainer = try GaussianTrainer(model: model, renderer: renderer)
    #expect(trainer.gaussianCount == 5)

    // Duplicate the first 2 and append. Moments for the added rows start at 0 (same as gsplat's
    // sample_add).
    let addCount = 2
    trainer.updateParamsWithOptimizer(
        paramFn: { _, p in MLX.concatenated([p, p[0..<addCount]], axis: 0) },
        momentFn: { _, m in
            let tail = [addCount] + Array(m.shape.dropFirst())
            return MLX.concatenated([m, MLXArray.zeros(tail, dtype: m.dtype)], axis: 0)
        }
    )

    #expect(trainer.gaussianCount == 7)
    #expect(model.means.shape == [7, 3])
    #expect(model.quats.shape == [7, 4])
    #expect(model.logScales.shape == [7, 3])
    #expect(model.logitOpacities.shape == [7])
    #expect(model.sh0.shape == [7, 3])
    #expect(model.shN.shape == [7, GaussianModel.shRestBands, 3])

    // A step running fine after growth = the Adam moments were resized along with it.
    let cam = makeStoreTestCamera()
    let target = MLXArray.zeros([cam.height, cam.width, 3])
    let loss = trainer.step(camera: cam, targetImage: target)
    #expect(loss.isFinite, "step after growth produced non-finite loss")
    #expect(trainer.gaussianCount == 7)
}

@Test
func updateParamsWithOptimizerIdentityIsNoOp() throws {
    let renderer = try GaussianRenderer()
    let model = GaussianModel.random(count: 4, seed: 2)
    let trainer = try GaussianTrainer(model: model, renderer: renderer)
    let before = model.means.asArray(Float.self)

    trainer.updateParamsWithOptimizer(paramFn: { _, p in p }, momentFn: { _, m in m })

    #expect(trainer.gaussianCount == 4)
    #expect(model.means.asArray(Float.self) == before)
}
