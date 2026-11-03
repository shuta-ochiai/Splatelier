// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// BENCH_STAGES=1 (train-capture): after training, runs the parts of a training step on the trained
// model, each repeated over a few views, and logs the GPU energy and time per repetition. Each
// repetition waits for the GPU, so the numbers are the parts' own cost, without the overlap of a
// real step. The full step is measured the same way for reference (it trains the model further, so
// it runs last).

#if os(macOS)

import Foundation
import MLX
import Trainer

func benchmarkStepStages(session: TrainingDriver.ResumableTrainingSession, log: (String) -> Void) {
    let trainer = session.trainer
    let renderer = trainer.renderer
    let views = Array(session.views.prefix(8))
    guard !views.isEmpty, let ssim = try? FusedSSIM() else { return }
    let reps = 40
    let shDegree = session.maxShDegree

    func measure(_ label: String, _ body: (TrainView, Camera, MLXArray) -> Void,
                 resample: (num: Int, den: Int)) {
        // Warm up (pipelines, compile traces).
        for v in views.prefix(2) {
            let (cam, target) = scaled(v, resample)
            body(v, cam, target)
        }
        let e0 = TrainingDriver.gpuEnergyReader?() ?? 0
        let t0 = Date()
        for i in 0..<reps {
            let v = views[i % views.count]
            let (cam, target) = scaled(v, resample)
            autoreleasepool { body(v, cam, target) }
        }
        let ms = Date().timeIntervalSince(t0) * 1000 / Double(reps)
        let mj = ((TrainingDriver.gpuEnergyReader?() ?? 0) - e0) * 1000 / Double(reps)
        log(String(format: "[bench %d/%d] %-14@ %6.2f ms  %6.1f mJ", resample.num, resample.den,
                   label as NSString, ms, mj))
    }
    func scaled(_ v: TrainView, _ r: (num: Int, den: Int)) -> (Camera, MLXArray) {
        if r.num >= r.den { return (v.camera, v.imageFloat) }
        let target = TrainingDriver.resampleRational(v.image, num: r.num, den: r.den)
        eval(target)
        return (TrainingDriver.coarseCameraScaled(v.camera, num: r.num, den: r.den), target)
    }

    for r in [(num: 1, den: 3), (num: 2, den: 3), (num: 1, den: 1)] {
        var pairs = 0
        measure("target", { _, _, t in eval(t) }, resample: r)
        measure("project", { _, cam, _ in
            let m = trainer.model
            let p = renderer.projectOnly(model: m, camera: cam)
            eval(p)
        }, resample: r)
        measure("project+bin", { _, cam, _ in
            let p = renderer.projectAndBin(model: trainer.model, camera: cam)
            eval(p.flattenIds, p.tileOffsets)
        }, resample: r)
        measure("forward", { _, cam, _ in
            let out = renderer.forward(model: trainer.model, camera: cam, shDegree: shDegree)
            eval(out.renderColor, out.renderAlpha)
            pairs += out.cache.bin.tileOffsets[out.cache.bin.numTiles].item(Int.self)
        }, resample: r)
        log("[bench \(r.num)/\(r.den)] tile pairs per view: \(pairs / (reps + 2))")
        measure("fwd+ssim", { _, cam, t in
            let out = renderer.forward(model: trainer.model, camera: cam, shDegree: shDegree)
            let (map, cache, _) = ssim.forward(out.renderColor, t)
            let v = ssim.backward(out.renderColor, t, dLdMap: MLXArray.zeros(like: map) - 1e-6,
                                  cache: cache, oneMinusLambda: 0.5, invN: 1e-6)
            eval(v)
        }, resample: r)
        if let bgK = try? BackgroundSHKernels() {
            for perThread in [1, BackgroundSHKernels.pixelsPerThread] {
                measure("bg bwd /\(perThread)", { _, cam, t in
                    let vm = cam.viewmat.asArray(Float.self), k = cam.intrinsics.asArray(Float.self)
                    let g = bgK.backwardCoeffs(
                        vRender: t, renderAlpha: t.mean(axis: 2), width: cam.width, height: cam.height,
                        degree: 4, fx: k[0], fy: k[1], cx: k[2], cy: k[3], viewmatRowMajor: vm,
                        pixelsPerThread: perThread)
                    eval(g)
                }, resample: r)
            }
        }
        measure("fwd+bwd", { _, cam, t in
            let out = renderer.forward(model: trainer.model, camera: cam, shDegree: shDegree)
            let vColor = (out.renderColor - t) * 1e-6
            let g = renderer.backward(model: trainer.model, camera: cam, cache: out.cache,
                                      vRenderColor: vColor, vRenderAlpha: MLXArray.zeros(like: out.renderAlpha))
            eval(g.dMeans, g.dQuats, g.dLogScales, g.dLogitOpacities, g.dSh0, g.dShN)
        }, resample: r)
        measure("step", { v, cam, t in
            let i = session.views.firstIndex { $0.name == v.name }
            _ = trainer.step(camera: cam, targetImage: t, shDegree: shDegree,
                             surfaceMask: v.surfaceMask, opaqueMask: v.opaqueMask,
                             viewIndex: i, fitGain: trainer.photometricGains != nil)
            eval(trainer.model.means, trainer.model.sh0, trainer.model.shN)
        }, resample: r)
    }
}

#endif
