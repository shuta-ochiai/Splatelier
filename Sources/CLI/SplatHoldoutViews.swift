// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// `holdout-views <dataset.valset> <outDir> <a.ply> <b.ply>`: renders two trained PLYs at the views
// train-capture held out (EVAL_SPLIT=<a>.split.json, else EVAL_HOLDOUT as in training), at half the
// frame size, and reports each view's PSNR against the photo for both. For the views where b loses
// the most it writes the photo, both renders and a map of where b's error is larger (red) or
// smaller (blue) than a's. Background splats count as foreground here (no background mask).

import Foundation
import MLX
import Trainer
import Capture

func runHoldoutViews(valsetURL: URL, outDir: URL, plyURLs: [URL], log: (String) -> Void) throws {
    precondition(plyURLs.count == 2)
    let env = ProcessInfo.processInfo.environment
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
    let scene = try CaptureDatasetStore.load(from: valsetURL)
    let geometry = CaptureTraining.prepareGeometry(scene: scene, log: { _ in })
    let frames = try EvaluationSplit.frames(
        geometry: geometry, defaultEvery: env["EVAL_HOLDOUT"].flatMap { Int($0) } ?? 8)?.heldOut ?? []
    let models = try plyURLs.map { try PLYReader.readGaussianPLY(url: $0) }
    let renderer = try GaussianRenderer()
    let factor = 2
    struct Row { let index: Int; let psnr: [Double]; let photo: MLXArray; let renders: [MLXArray] }
    var rows: [Row] = []
    for (vi, f) in frames.enumerated() {
        let img = f.imageFloat
        let h = img.shape[0] / factor, w = img.shape[1] / factor
        let photo = img[0..<(h * factor), 0..<(w * factor)].reshaped([h, factor, w, factor, 3]).mean(axes: [1, 3])
        let k = f.intrinsics
        let s = 1 / Float(factor)
        let cam = Camera(viewmat: MLXArray(f.viewmat, [16]),
                         intrinsics: MLXArray([k.fx * s, k.fy * s, k.cx * s, k.cy * s], [4]),
                         width: w, height: h, background: MLXArray([Float(0), 0, 0], [3]))
        var psnr: [Double] = []
        var renders: [MLXArray] = []
        for m in models {
            let r = MLX.clip(renderer.forward(model: m, camera: cam, shDegree: GaussianModel.maxShDegree).renderColor, min: 0, max: 1)
            let mse = ((r - photo) * (r - photo)).mean().item(Float.self)
            psnr.append(10 * log10(1 / Double(max(mse, 1e-10))))
            renders.append(r)
        }
        rows.append(Row(index: vi, psnr: psnr, photo: photo, renders: renders))
    }
    let mean = { (i: Int) in rows.map { $0.psnr[i] }.reduce(0, +) / Double(max(1, rows.count)) }
    print(String(format: "%d held-out views: mean PSNR %@ %.3f dB, %@ %.3f dB", rows.count,
                 plyURLs[0].lastPathComponent, mean(0), plyURLs[1].lastPathComponent, mean(1)))
    print("view   A PSNR   B PSNR   B−A")
    for r in rows.sorted(by: { ($0.psnr[1] - $0.psnr[0]) < ($1.psnr[1] - $1.psnr[0]) }) {
        print(String(format: "%4d  %7.2f  %7.2f  %+6.2f", r.index, r.psnr[0], r.psnr[1], r.psnr[1] - r.psnr[0]))
    }
    for r in rows.sorted(by: { ($0.psnr[1] - $0.psnr[0]) < ($1.psnr[1] - $1.psnr[0]) }).prefix(4) {
        let base = outDir.appendingPathComponent(String(format: "view%02d", r.index)).path
        saveImagePNG(r.photo, to: URL(fileURLWithPath: base + "_photo.png"))
        saveImagePNG(r.renders[0], to: URL(fileURLWithPath: base + "_A.png"))
        saveImagePNG(r.renders[1], to: URL(fileURLWithPath: base + "_B.png"))
        // Per-pixel error difference: red where B's error is larger, blue where smaller (±0.2).
        let eA = MLX.abs(r.renders[0] - r.photo).mean(axis: 2)
        let eB = MLX.abs(r.renders[1] - r.photo).mean(axis: 2)
        let d = MLX.clip((eB - eA) / 0.2, min: -1, max: 1)
        let pos = MLX.maximum(d, 0), neg = MLX.maximum(-d, 0)
        let map = MLX.stacked([1 - neg, 1 - pos - neg, 1 - pos], axis: 2)
        saveImagePNG(map, to: URL(fileURLWithPath: base + "_diff.png"))
    }
    log("wrote the 4 views where B loses most to \(outDir.path)")
}
