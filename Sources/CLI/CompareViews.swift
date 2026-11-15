// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// `compare-views <dataset.valset> <outDir> <before.ply> <after.ply> <before.obj> <after.obj>`:
// at 3 held-out views (EVAL_SPLIT, else EVAL_HOLDOUT as in training), evenly spread, writes the
// photo, both splat renders and both meshes (shaded) at half the frame size, plus turns.json with
// the clockwise quarter turns that stand each view upright (by its pose), for a side-by-side sheet.

import Foundation
import MLX
import Trainer
import Capture
import SplatViewerKit

func runCompareViews(valsetURL: URL, outDir: URL, plyURLs: [URL], objURLs: [URL], log: (String) -> Void) throws {
    let env = ProcessInfo.processInfo.environment
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
    let scene = try CaptureDatasetStore.load(from: valsetURL)
    let geometry = CaptureTraining.prepareGeometry(scene: scene, log: { _ in })
    let heldOut = try EvaluationSplit.frames(
        geometry: geometry, defaultEvery: env["EVAL_HOLDOUT"].flatMap { Int($0) } ?? 8)?.heldOut ?? []
    guard !heldOut.isEmpty else { log("no held-out views"); return }
    let picks = (0..<3).map { heldOut[min(heldOut.count - 1, ($0 * 2 + 1) * heldOut.count / 6)] }
    let models = try plyURLs.map { url -> GaussianModel in
        var model = try PLYReader.readGaussianPLY(url: url)
        let sidecar = url.deletingPathExtension().appendingPathExtension("background.json")
        if let data = try? Data(contentsOf: sidecar),
           let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
           let count = json["backgroundSplatCount"] as? Int {
            model.environmentCount = Swift.min(count, model.count)
        }
        return model
    }
    let meshes = try objURLs.map { try readOBJ($0) }
    let renderer = try GaussianRenderer()
    let factor = 2
    var turns: [Int] = []
    for (vi, f) in picks.enumerated() {
        let img = f.imageFloat
        let h = img.shape[0] / factor, w = img.shape[1] / factor
        let photo = img[0..<(h * factor), 0..<(w * factor)].reshaped([h, factor, w, factor, 3]).mean(axes: [1, 3])
        let k = f.intrinsics, s = 1 / Float(factor)
        let intr: Intrinsics = (fx: k.fx * s, fy: k.fy * s, cx: k.cx * s, cy: k.cy * s)
        let base = outDir.appendingPathComponent("view\(vi)").path
        saveImagePNG(photo, to: URL(fileURLWithPath: base + "_photo.png"))
        let cam = Camera(viewmat: MLXArray(f.viewmat, [16]),
                         intrinsics: MLXArray([intr.fx, intr.fy, intr.cx, intr.cy], [4]),
                         width: w, height: h, background: MLXArray([Float(0), 0, 0], [3]))
        for (mi, m) in models.enumerated() {
            let r = MLX.clip(renderer.forward(model: m, camera: cam, shDegree: GaussianModel.maxShDegree).renderColor, min: 0, max: 1)
            saveImagePNG(r, to: URL(fileURLWithPath: base + (mi == 0 ? "_gs_before.png" : "_gs_after.png")))
        }
        for (mi, mesh) in meshes.enumerated() {
            let r = rasterize(mesh, viewmat: f.viewmat, intrinsics: intr, width: w, height: h)
            try writePNG(shading(depth: r.depth, width: w, height: h, intrinsics: intr), width: w, height: h,
                         to: URL(fileURLWithPath: base + (mi == 0 ? "_mesh_before.png" : "_mesh_after.png")))
        }
        turns.append(SplatViewerThumbnailOrientation.clockwiseQuarterTurns(viewmat: f.viewmat) ?? 1)
    }
    try JSONSerialization.data(withJSONObject: turns).write(to: outDir.appendingPathComponent("turns.json"))
    log("wrote 3 views to \(outDir.path)")
}
