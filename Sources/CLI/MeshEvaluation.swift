// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// `mesh-eval <dataset.valset> <mesh.obj>...`: compares meshes of a capture on the views
// train-capture held out (EVAL_SPLIT=<output>.split.json from that run; else recomputed with
// EVAL_HOLDOUT, default 8, VIEW_CAP / EVAL_HOLDOUT_ALL as in training), with no ground-truth geometry. Each mesh is rasterized (z-buffer, vertex colors) at every held-out view
// and measured against what the capture has:
//   coverage     share of the view's pixels the mesh covers
//   PSNR         vertex colors vs the photo, over the covered pixels
//   MVS agree    relative depth difference to the view's own MVS depth map where both exist
//                (median, and the share within 1%); MVS is noisy itself, so this is a proxy
//   MVS covered  share of the view's MVS pixels the mesh covers
//   roughness    median angle (deg) between the normal of the rendered depth and the mean normal
//                of its 7×7 neighborhood: lower is smoother
// env: EVAL_SCALE=<s> render scale relative to the frames (default 0.5), EVAL_IMAGES=<dir> writes
// the photo and each mesh's color / shaded renders of the first 3 held-out views.

import Foundation
import MLX
import simd
import Trainer
import Capture

func runMeshEvaluation(valsetURL: URL, meshURLs: [URL], log: (String) -> Void) throws {
    let env = ProcessInfo.processInfo.environment
    let holdoutEvery = env["EVAL_HOLDOUT"].flatMap { Int($0) } ?? 8
    let scale = env["EVAL_SCALE"].flatMap { Float($0) } ?? 0.5
    let factor = Swift.max(1, Int((1 / scale).rounded()))
    let imagesDir = env["EVAL_IMAGES"].map { URL(fileURLWithPath: $0) }
    if let imagesDir { try FileManager.default.createDirectory(at: imagesDir, withIntermediateDirectories: true) }

    let scene = try CaptureDatasetStore.load(from: valsetURL)
    let geometry = CaptureTraining.prepareGeometry(scene: scene, log: { _ in })
    let frames = try EvaluationSplit.frames(geometry: geometry, defaultEvery: holdoutEvery)?.heldOut ?? []
    var mvsByPose: [[Float]: CostVolumeMVS.DepthView] = [:]
    for map in try geometry.depthViews() { mvsByPose[map.viewmat] = map }
    log("mesh-eval: \(frames.count) held-out views (every \(holdoutEvery)), render 1/\(factor)")

    // Photos at the render resolution (box average).
    struct View {
        let frame: CapturedFrame
        let width: Int, height: Int
        let intrinsics: Intrinsics
        let photo: [SIMD3<Float>]
    }
    let views: [View] = frames.map { f in
        let w = f.width / factor, h = f.height / factor
        let src = f.image.asArray(UInt8.self)
        var photo = [SIMD3<Float>](repeating: .zero, count: w * h)
        for y in 0 ..< h {
            for x in 0 ..< w {
                var acc = SIMD3<Float>.zero
                for dy in 0 ..< factor {
                    for dx in 0 ..< factor {
                        let i = ((y * factor + dy) * f.width + x * factor + dx) * 3
                        acc += SIMD3(Float(src[i]), Float(src[i + 1]), Float(src[i + 2]))
                    }
                }
                photo[y * w + x] = acc / Float(factor * factor * 255)
            }
        }
        let s = 1 / Float(factor)
        let k = f.intrinsics
        return View(frame: f, width: w, height: h,
                    intrinsics: (fx: k.fx * s, fy: k.fy * s, cx: k.cx * s, cy: k.cy * s), photo: photo)
    }

    func pad(_ s: String, _ n: Int) -> String { s.count >= n ? s : s + String(repeating: " ", count: n - s.count) }
    print(pad("mesh", 28) + " coverage    PSNR  MVS rel.err  MVS <1%  MVS covered  rough°")
    for url in meshURLs {
        let mesh = try readOBJ(url)
        var coverage: [Double] = [], psnr: [Double] = [], relErr: [Double] = [], within: [Double] = []
        var mvsCovered: [Double] = [], rough: [Double] = []
        for (vi, view) in views.enumerated() {
            let r = rasterize(mesh, viewmat: view.frame.viewmat, intrinsics: view.intrinsics,
                              width: view.width, height: view.height)
            let n = view.width * view.height
            var covered = 0
            var se = 0.0
            for i in 0 ..< n where r.depth[i] < .infinity {
                covered += 1
                let d = r.color[i] - view.photo[i]
                se += Double((d * d).sum())
            }
            coverage.append(Double(covered) / Double(n))
            if covered > 0 { psnr.append(10 * log10(1 / (se / Double(covered * 3)))) }

            if let mvs = mvsByPose[view.frame.viewmat] {
                var errs: [Float] = []
                var mvsPixels = 0
                for y in 0 ..< mvs.height {
                    for x in 0 ..< mvs.width {
                        let z = mvs.depth[y * mvs.width + x]
                        guard z > 0 else { continue }
                        mvsPixels += 1
                        let u = Int(((Float(x) + 0.5 - mvs.intrinsics.cx) / mvs.intrinsics.fx * view.intrinsics.fx
                                     + view.intrinsics.cx).rounded(.down))
                        let v = Int(((Float(y) + 0.5 - mvs.intrinsics.cy) / mvs.intrinsics.fy * view.intrinsics.fy
                                     + view.intrinsics.cy).rounded(.down))
                        guard u >= 0, v >= 0, u < view.width, v < view.height else { continue }
                        let zm = r.depth[v * view.width + u]
                        if zm < .infinity { errs.append(abs(zm - z) / z) }
                    }
                }
                if !errs.isEmpty {
                    errs.sort()
                    relErr.append(Double(errs[errs.count / 2]))
                    within.append(Double(errs.filter { $0 < 0.01 }.count) / Double(errs.count))
                }
                if mvsPixels > 0 { mvsCovered.append(Double(errs.count) / Double(mvsPixels)) }
            }
            if let rr = roughness(depth: r.depth, width: view.width, height: view.height, intrinsics: view.intrinsics) {
                rough.append(Double(rr))
            }
            if let imagesDir, vi < 3 {
                let base = url.deletingPathExtension().lastPathComponent
                if url == meshURLs[0] {
                    try writePNG(view.photo, width: view.width, height: view.height,
                                 to: imagesDir.appendingPathComponent("view\(vi)_photo.png"))
                }
                try writePNG(r.color.enumerated().map { r.depth[$0.offset] < .infinity ? $0.element : SIMD3(1, 1, 1) },
                             width: view.width, height: view.height,
                             to: imagesDir.appendingPathComponent("view\(vi)_\(base)_color.png"))
                try writePNG(shading(depth: r.depth, width: view.width, height: view.height, intrinsics: view.intrinsics),
                             width: view.width, height: view.height,
                             to: imagesDir.appendingPathComponent("view\(vi)_\(base)_shaded.png"))
            }
        }
        func mean(_ a: [Double]) -> Double { a.isEmpty ? .nan : a.reduce(0, +) / Double(a.count) }
        print(pad(url.deletingPathExtension().lastPathComponent, 28) + String(
            format: " %7.1f%% %7.2f %11.2f%% %7.1f%% %11.1f%% %7.2f",
            100 * mean(coverage), mean(psnr), 100 * mean(relErr), 100 * mean(within),
            100 * mean(mvsCovered), mean(rough)))
    }
}

/// The training / held-out split of a train-capture run, by frame timestamp (stable across runs,
/// unlike a split recomputed from re-run BA poses). Written as `<output>.split.json`.
struct EvaluationSplit: Codable {
    var training: [TimeInterval]
    var heldOut: [TimeInterval]

    static func write(training: [CapturedFrame], heldOut: [CapturedFrame], to url: URL) throws {
        let split = EvaluationSplit(training: training.map(\.timestamp), heldOut: heldOut.map(\.timestamp))
        try JSONEncoder().encode(split).write(to: url)
    }

    /// The split from EVAL_SPLIT=<path> if set, else recomputed with EVAL_HOLDOUT (default
    /// `defaultEvery`; VIEW_CAP / EVAL_HOLDOUT_ALL as in training). nil without either.
    static func frames(geometry: CaptureGeometry, defaultEvery: Int) throws
        -> (training: [CapturedFrame], heldOut: [CapturedFrame])? {
        let env = ProcessInfo.processInfo.environment
        if let path = env["EVAL_SPLIT"] {
            let split = try JSONDecoder().decode(EvaluationSplit.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
            let training = Set(split.training), heldOut = Set(split.heldOut)
            let frames = geometry.scene.frames
            return (frames.filter { training.contains($0.timestamp) }, frames.filter { heldOut.contains($0.timestamp) })
        }
        let every = env["EVAL_HOLDOUT"].flatMap { Int($0) } ?? defaultEvery
        guard every > 1 else { return nil }
        return CaptureTraining.evaluationSplit(
            geometry: geometry, holdoutEvery: every, trainingViewCap: env["VIEW_CAP"].flatMap { Int($0) },
            holdoutBeforeSelection: ["1", "on", "yes", "true"].contains(env["EVAL_HOLDOUT_ALL"]?.lowercased() ?? ""))
    }
}

/// A triangle mesh as `MeshOBJWriter` writes it (`v x y z r g b`, `f a//a b//b c//c`).
private func readOBJ(_ url: URL) throws -> (positions: [SIMD3<Float>], colors: [SIMD3<Float>], indices: [UInt32]) {
    let data = try Data(contentsOf: url)
    var positions: [SIMD3<Float>] = [], colors: [SIMD3<Float>] = [], indices: [UInt32] = []
    data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
        let bytes = raw.bindMemory(to: UInt8.self)
        var i = 0
        let n = bytes.count
        func skipSpaces() { while i < n, bytes[i] == 32 { i += 1 } }
        func number() -> Float {
            skipSpaces()
            let start = i
            while i < n, bytes[i] != 32, bytes[i] != 10, bytes[i] != 13 { i += 1 }
            return Float(String(decoding: UnsafeBufferPointer(rebasing: bytes[start ..< i]), as: UTF8.self)) ?? 0
        }
        func index() -> UInt32 {
            skipSpaces()
            var v: UInt32 = 0
            while i < n, bytes[i] >= 48, bytes[i] <= 57 { v = v * 10 + UInt32(bytes[i] - 48); i += 1 }
            while i < n, bytes[i] != 32, bytes[i] != 10 { i += 1 }   // skip "//n"
            return v - 1
        }
        while i < n {
            if bytes[i] == 118, i + 1 < n, bytes[i + 1] == 32 {          // "v "
                i += 2
                positions.append(SIMD3(number(), number(), number()))
                colors.append(SIMD3(number(), number(), number()))
            } else if bytes[i] == 102, i + 1 < n, bytes[i + 1] == 32 {   // "f "
                i += 2
                indices.append(index()); indices.append(index()); indices.append(index())
            }
            while i < n, bytes[i] != 10 { i += 1 }
            i += 1
        }
    }
    return (positions, colors, indices)
}

/// Z-buffer rasterization with perspective-correct depth and vertex colors. Pixel centers at +0.5.
private func rasterize(
    _ mesh: (positions: [SIMD3<Float>], colors: [SIMD3<Float>], indices: [UInt32]),
    viewmat v: [Float], intrinsics k: Intrinsics, width w: Int, height h: Int
) -> (depth: [Float], color: [SIMD3<Float>]) {
    var depth = [Float](repeating: .infinity, count: w * h)
    var color = [SIMD3<Float>](repeating: .zero, count: w * h)
    let cam = mesh.positions.map { p in
        SIMD3<Float>(v[0] * p.x + v[1] * p.y + v[2] * p.z + v[3],
                     v[4] * p.x + v[5] * p.y + v[6] * p.z + v[7],
                     v[8] * p.x + v[9] * p.y + v[10] * p.z + v[11])
    }
    let screen = cam.map { c in SIMD2<Float>(k.fx * c.x / c.z + k.cx, k.fy * c.y / c.z + k.cy) }
    depth.withUnsafeMutableBufferPointer { db in
        color.withUnsafeMutableBufferPointer { cb in
            for t in stride(from: 0, to: mesh.indices.count, by: 3) {
                let i0 = Int(mesh.indices[t]), i1 = Int(mesh.indices[t + 1]), i2 = Int(mesh.indices[t + 2])
                let z0 = cam[i0].z, z1 = cam[i1].z, z2 = cam[i2].z
                guard z0 > 0.01, z1 > 0.01, z2 > 0.01 else { continue }
                let a = screen[i0], b = screen[i1], c = screen[i2]
                let area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
                guard abs(area) > 1e-12 else { continue }
                let x0 = Swift.max(0, Int((Swift.min(a.x, b.x, c.x) - 0.5).rounded(.up)))
                let x1 = Swift.min(w - 1, Int((Swift.max(a.x, b.x, c.x) - 0.5).rounded(.down)))
                let y0 = Swift.max(0, Int((Swift.min(a.y, b.y, c.y) - 0.5).rounded(.up)))
                let y1 = Swift.min(h - 1, Int((Swift.max(a.y, b.y, c.y) - 0.5).rounded(.down)))
                guard x0 <= x1, y0 <= y1 else { continue }
                let inv = 1 / area
                for y in y0 ... y1 {
                    let py = Float(y) + 0.5
                    for x in x0 ... x1 {
                        let px = Float(x) + 0.5
                        let w0 = ((b.x - px) * (c.y - py) - (b.y - py) * (c.x - px)) * inv
                        let w1 = ((c.x - px) * (a.y - py) - (c.y - py) * (a.x - px)) * inv
                        let w2 = 1 - w0 - w1
                        guard w0 >= 0, w1 >= 0, w2 >= 0 else { continue }
                        let iz = w0 / z0 + w1 / z1 + w2 / z2
                        let z = 1 / iz
                        let p = y * w + x
                        guard z < db[p] else { continue }
                        db[p] = z
                        cb[p] = (w0 / z0 * mesh.colors[i0] + w1 / z1 * mesh.colors[i1] + w2 / z2 * mesh.colors[i2]) * z
                    }
                }
            }
        }
    }
    return (depth, color)
}

/// Camera-space normals of a depth map from central differences (nil where a neighbor is missing
/// or the depth jumps by more than 5%).
private func depthNormals(depth: [Float], width w: Int, height h: Int, intrinsics k: Intrinsics) -> [SIMD3<Float>?] {
    func point(_ x: Int, _ y: Int) -> SIMD3<Float> {
        let z = depth[y * w + x]
        return SIMD3((Float(x) + 0.5 - k.cx) / k.fx * z, (Float(y) + 0.5 - k.cy) / k.fy * z, z)
    }
    var normals = [SIMD3<Float>?](repeating: nil, count: w * h)
    for y in 1 ..< h - 1 {
        for x in 1 ..< w - 1 {
            let z = depth[y * w + x]
            guard z < .infinity else { continue }
            let nb = [depth[y * w + x - 1], depth[y * w + x + 1], depth[(y - 1) * w + x], depth[(y + 1) * w + x]]
            guard nb.allSatisfy({ $0 < .infinity && abs($0 - z) < 0.05 * z }) else { continue }
            let a = point(x + 1, y) - point(x - 1, y)
            let b = point(x, y + 1) - point(x, y - 1)
            let n = -simd_cross(a, b)
            let len = simd_length(n)
            if len > 0 { normals[y * w + x] = n / len }
        }
    }
    return normals
}

/// Median angle (deg) between each normal and the normalized mean normal of its 7×7 neighborhood.
private func roughness(depth: [Float], width w: Int, height h: Int, intrinsics k: Intrinsics) -> Float? {
    let normals = depthNormals(depth: depth, width: w, height: h, intrinsics: k)
    var angles: [Float] = []
    let r = 3
    for y in r ..< h - r {
        for x in r ..< w - r {
            guard let n = normals[y * w + x] else { continue }
            var sum = SIMD3<Float>.zero
            var count = 0
            for dy in -r ... r {
                for dx in -r ... r {
                    if let m = normals[(y + dy) * w + x + dx] { sum += m; count += 1 }
                }
            }
            guard count >= 25 else { continue }
            let mean = simd_normalize(sum)
            angles.append(acos(Swift.max(-1, Swift.min(1, simd_dot(n, mean)))) * 180 / .pi)
        }
    }
    guard !angles.isEmpty else { return nil }
    angles.sort()
    return angles[angles.count / 2]
}

/// Gray shading of a depth map by its normals (light from the camera); white where empty.
private func shading(depth: [Float], width w: Int, height h: Int, intrinsics k: Intrinsics) -> [SIMD3<Float>] {
    depthNormals(depth: depth, width: w, height: h, intrinsics: k).enumerated().map { i, n in
        guard depth[i] < .infinity else { return SIMD3(1, 1, 1) }
        guard let n else { return SIMD3(0.3, 0.3, 0.3) }
        let s = 0.15 + 0.85 * abs(n.z)
        return SIMD3(s, s, s)
    }
}

private func writePNG(_ pixels: [SIMD3<Float>], width: Int, height: Int, to url: URL) throws {
    #if os(macOS)
    let flat = pixels.flatMap { [$0.x, $0.y, $0.z] }
    saveImagePNG(MLXArray(flat, [height, width, 3]), to: url)
    #endif
}
