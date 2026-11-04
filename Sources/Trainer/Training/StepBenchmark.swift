// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Runs the parts of a training step on a trained model, each repeated over a few views, and logs
// the time per repetition and, where `gpuEnergyReader` exists (the Mac CLI), the GPU energy. Each
// repetition waits for the GPU, so the numbers are the parts' own cost, without the overlap of a
// real step; the process CPU time per repetition is logged next to it. The rasterizer and the
// whole training step are measured with 16-px and 8-px tiles on the same model and views, in
// alternating order so that a device heating up during the benchmark does not favour one of them.

import Foundation
import MLX

extension TrainingDriver {

    public static func benchmarkStepStages(session: ResumableTrainingSession, log: (String) -> Void) {
        let trainer = session.trainer
        let renderer = trainer.renderer
        let views = Array(session.views.prefix(8))
        guard !views.isEmpty, let ssim = try? FusedSSIM() else { return }
        let reps = 20
        var lastCPU: Double?
        let shDegree = session.maxShDegree

        func scaled(_ v: TrainView, _ r: (num: Int, den: Int), tile: Int) -> (Camera, MLXArray) {
            let base = r.num >= r.den ? v.camera : coarseCameraScaled(v.camera, num: r.num, den: r.den)
            let cam = Camera(viewmat: base.viewmat, intrinsics: base.intrinsics, width: base.width,
                             height: base.height, tileSize: tile, background: base.background)
            let target = r.num >= r.den ? v.imageFloat : resampleRational(v.image, num: r.num, den: r.den)
            eval(target)
            return (cam, target)
        }
        // CPU time of the whole process (all threads), ms.
        func cpuMs() -> Double {
            var u = rusage()
            getrusage(RUSAGE_SELF, &u)
            func ms(_ t: timeval) -> Double { Double(t.tv_sec) * 1000 + Double(t.tv_usec) / 1000 }
            return ms(u.ru_utime) + ms(u.ru_stime)
        }
        // Mean wall ms, process CPU ms and mJ (nil without an energy reader) per repetition.
        func measure(_ r: (num: Int, den: Int), tile: Int,
                     _ body: (TrainView, Camera, MLXArray) -> Void) -> (ms: Double, mj: Double?) {
            for v in views.prefix(2) {
                let (cam, target) = scaled(v, r, tile: tile)
                body(v, cam, target)
            }
            let e0 = gpuEnergyReader?()
            let c0 = cpuMs()
            let t0 = Date()
            for i in 0..<reps {
                let v = views[i % views.count]
                let (cam, target) = scaled(v, r, tile: tile)
                autoreleasepool { body(v, cam, target) }
            }
            let ms = Date().timeIntervalSince(t0) * 1000 / Double(reps)
            lastCPU = (cpuMs() - c0) / Double(reps)
            let mj = e0.flatMap { e0 in gpuEnergyReader?().map { ($0 - e0) * 1000 / Double(reps) } }
            return (ms, mj)
        }
        func line(_ r: (num: Int, den: Int), _ label: String, _ m: (ms: Double, mj: Double?), cpu: Double? = nil) {
            log(String(format: "[bench %d/%d] %-26@ %7.2f ms%@%@", r.num, r.den, label as NSString, m.ms,
                       cpu.map { String(format: "  cpu %6.2f ms", $0) } ?? "",
                       m.mj.map { String(format: "  %6.1f mJ", $0) } ?? ""))
        }

        let variants: [(label: String, tile: Int)] = [("16px", 16), ("8px", 8)]
        for r in [(num: 2, den: 3), (num: 1, den: 1)] {
            line(r, "target", measure(r, tile: 16) { _, _, t in eval(t) })
            line(r, "d-ssim fwd+bwd", measure(r, tile: 16) { _, _, t in
                let (map, cache, _) = ssim.forward(t, t)
                eval(ssim.backward(t, t, dLdMap: MLXArray.zeros(like: map) - 1e-6, cache: cache,
                                   oneMinusLambda: 0.5, invN: 1e-6))
            })
            // Each variant twice, in forward then reverse order; the lines show the mean.
            var sums: [String: [(ms: Double, mj: Double?)]] = [:]
            var pairs: [String: Int] = [:]
            var cpus: [String: [Double]] = [:]
            for pass in 0..<2 {
                for v in (pass == 0 ? variants : variants.reversed()) {
                    let bin = measure(r, tile: v.tile) { _, cam, _ in
                        let p = renderer.projectAndBin(model: trainer.model, camera: cam)
                        eval(p.flattenIds, p.tileOffsets)
                    }
                    let fwd = measure(r, tile: v.tile) { _, cam, _ in
                        let out = renderer.forward(model: trainer.model, camera: cam, shDegree: shDegree)
                        eval(out.renderColor, out.renderAlpha)
                        if pass == 0 { pairs[v.label, default: 0] += out.cache.bin.tileOffsets[out.cache.bin.numTiles].item(Int.self) }
                    }
                    let both = measure(r, tile: v.tile) { _, cam, t in
                        let out = renderer.forward(model: trainer.model, camera: cam, shDegree: shDegree)
                        let g = renderer.backward(model: trainer.model, camera: cam, cache: out.cache,
                                                  vRenderColor: (out.renderColor - t) * 1e-6,
                                                  vRenderAlpha: MLXArray.zeros(like: out.renderAlpha))
                        eval(g.dMeans, g.dQuats, g.dLogScales, g.dLogitOpacities, g.dSh0, g.dShN)
                    }
                    let bothCPU = lastCPU
                    // The whole training step (trains the model further, after its export).
                    let step = measure(r, tile: v.tile) { view, cam, t in
                        let i = session.views.firstIndex { $0.name == view.name }
                        _ = trainer.step(camera: cam, targetImage: t, shDegree: shDegree,
                                         surfaceMask: view.surfaceMask, opaqueMask: view.opaqueMask,
                                         viewIndex: i, fitGain: trainer.photometricGains != nil)
                        eval(trainer.model.means, trainer.model.sh0, trainer.model.shN)
                    }
                    sums[v.label, default: []] += [bin, fwd, both, step]
                    cpus[v.label, default: []] += [bothCPU ?? 0, lastCPU ?? 0]
                }
            }
            for v in variants {
                let s = sums[v.label] ?? []
                guard s.count == 8 else { continue }
                func mean(_ i: Int) -> (ms: Double, mj: Double?) {
                    let a = s[i], b = s[i + 4]
                    return ((a.ms + b.ms) / 2, a.mj.flatMap { x in b.mj.map { (x + $0) / 2 } })
                }
                let c = cpus[v.label] ?? [0, 0, 0, 0]
                line(r, "\(v.label) project+bin", mean(0))
                line(r, "\(v.label) forward", mean(1))
                line(r, "\(v.label) fwd+bwd", mean(2), cpu: (c[0] + c[2]) / 2)
                line(r, "\(v.label) training step", mean(3), cpu: (c[1] + c[3]) / 2)
                log("[bench \(r.num)/\(r.den)] \(v.label) tile pairs per view: \((pairs[v.label] ?? 0) / (reps + 2))")
            }
        }
    }
}
