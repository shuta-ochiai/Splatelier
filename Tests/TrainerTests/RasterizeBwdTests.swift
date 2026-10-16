// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXRandom
import Testing
@testable import Trainer

private struct BwdScene {
    let imageWidth: Int
    let imageHeight: Int
    let tileSize: Int
    var numTilesX: Int { (imageWidth + tileSize - 1) / tileSize }
    let means2d: MLXArray         // [N, 2]
    let conics: MLXArray          // [N, 3]
    let colors: MLXArray          // [N, 3]
    let opacities: MLXArray       // [N]
    let radii: MLXArray           // [N, 2] int32 (rx, ry)
    let depths: MLXArray          // [N]
    let background: MLXArray      // [3]
}

private func slangFwd(scene: BwdScene, bin: TileBinningResult)
    throws -> (renderColor: MLXArray, renderAlpha: MLXArray, lastIds: MLXArray)
{
    let kernel = try SlangCompiledKernel.load(name: "rasterize_fwd")
    let dims = MLXArray(
        [Int32(scene.imageWidth), Int32(scene.imageHeight),
         Int32(scene.tileSize), Int32(scene.numTilesX)],
        [4]
    )
    let totalPixels = scene.imageWidth * scene.imageHeight
    let outputs = kernel(
        [
            scene.means2d, scene.conics, scene.colors, scene.opacities,
            bin.flattenIds, bin.tileOffsets,
            dims, scene.background,
        ],
        grid: (scene.imageWidth, scene.imageHeight, 1),
        threadGroup: (scene.tileSize, scene.tileSize, 1),
        outputShapes: [[totalPixels * 3], [totalPixels], [totalPixels]],
        outputDTypes: [.float32, .float32, .int32]
    )
    eval(outputs[0], outputs[1], outputs[2])
    return (outputs[0], outputs[1], outputs[2])
}

private func slangBwd(
    scene: BwdScene,
    bin: TileBinningResult,
    renderAlpha: MLXArray, lastIds: MLXArray,
    vRenderColor: MLXArray, vRenderAlpha: MLXArray
) throws -> (vMeans2d: MLXArray, vConics: MLXArray, vColors: MLXArray, vOpacities: MLXArray) {
    let kernel = try SlangCompiledKernel.load(name: "rasterize_bwd")
    let n = scene.means2d.shape[0]
    let dims = MLXArray(
        [Int32(scene.imageWidth), Int32(scene.imageHeight),
         Int32(scene.tileSize), Int32(scene.numTilesX)],
        [4]
    )
    let outputs = kernel(
        [
            scene.means2d, scene.conics, scene.colors, scene.opacities,
            bin.flattenIds, bin.tileOffsets, dims, scene.background,
            renderAlpha, lastIds, vRenderColor, vRenderAlpha,
        ],
        grid: (scene.imageWidth, scene.imageHeight, 1),
        threadGroup: (scene.tileSize, scene.tileSize, 1),
        outputShapes: [[n * 2], [n * 3], [n * 3], [n]],
        outputDTypes: [.float32, .float32, .float32, .float32],
        initValue: 0.0
    )
    eval(outputs[0], outputs[1], outputs[2], outputs[3])
    return (
        outputs[0].reshaped([n, 2]),
        outputs[1].reshaped([n, 3]),
        outputs[2].reshaped([n, 3]),
        outputs[3]
    )
}

/// loss = Σ cotColor·fwd_color + Σ cotAlpha·fwd_alpha with the CPU reference rasterizer.
private func sceneLoss(
    means2d: [Float], conics: [Float], colors: [Float], opacities: [Float],
    background: [Float],
    flattenIds: [Int32], tileOffsets: [Int32],
    imageWidth: Int, imageHeight: Int, tileSize: Int,
    cotColor: [Float], cotAlpha: [Float]
) -> Float {
    let r = referenceRasterize(
        means2d: means2d, conics: conics, colors: colors, opacities: opacities,
        flattenIds: flattenIds, tileOffsets: tileOffsets,
        imageWidth: imageWidth, imageHeight: imageHeight, tileSize: tileSize,
        background: background
    )
    let totalPixels = imageWidth * imageHeight
    var loss: Float = 0
    for i in 0..<(totalPixels * 3) { loss += cotColor[i] * r.renderColor[i] }
    for i in 0..<totalPixels      { loss += cotAlpha[i] * r.renderAlpha[i] }
    return loss
}

@Test
func rasterizeBwdMatchesFiniteDifferences() throws {
    // Overlap 2 Gaussians at different depths in one 16x16 tile so front-to-back compositing and
    // the (1 - alpha) chain matter.
    let scene = BwdScene(
        imageWidth: 16, imageHeight: 16, tileSize: 16,
        means2d: MLXArray(
            [Float(7.0), Float(8.0),
             Float(9.0), Float(8.0)],
            [2, 2]
        ),
        conics: MLXArray(
            [Float(0.04), Float(0.0), Float(0.04),
             Float(0.03), Float(0.005), Float(0.03)],
            [2, 3]
        ),
        colors: MLXArray(
            [Float(1.0), Float(0.2), Float(0.1),
             Float(0.1), Float(0.8), Float(0.6)],
            [2, 3]
        ),
        opacities: MLXArray([Float(0.7), Float(0.6)], [2]),
        radii: MLXArray([Int32(8), Int32(8), Int32(8), Int32(8)], [2, 2]),
        depths: MLXArray([Float(1.5), Float(3.0)], [2]),
        background: MLXArray([Float(0.0), Float(0.0), Float(0.0)], [3])
    )
    let n = 2
    let W = scene.imageWidth, H = scene.imageHeight
    let totalPixels = W * H

    let bin = TileBinning.bin(
        means2d: scene.means2d, radii: scene.radii, depths: scene.depths,
        imageWidth: W, imageHeight: H, tileSize: scene.tileSize
    )

    let (_, renderAlpha, lastIds) = try slangFwd(scene: scene, bin: bin)

    MLXRandom.seed(42)
    let cotColor = MLXRandom.normal([totalPixels * 3]) * 0.05
    let cotAlpha = MLXRandom.normal([totalPixels]) * 0.05
    eval(cotColor, cotAlpha)

    let bwd = try slangBwd(
        scene: scene, bin: bin,
        renderAlpha: renderAlpha, lastIds: lastIds,
        vRenderColor: cotColor, vRenderAlpha: cotAlpha
    )
    let analyticalMeans = bwd.vMeans2d.asArray(Float.self)
    let analyticalConics = bwd.vConics.asArray(Float.self)
    let analyticalColors = bwd.vColors.asArray(Float.self)
    let analyticalOpacities = bwd.vOpacities.asArray(Float.self)

    let meansHost = scene.means2d.asArray(Float.self)
    let conicsHost = scene.conics.asArray(Float.self)
    let colorsHost = scene.colors.asArray(Float.self)
    let opacitiesHost = scene.opacities.asArray(Float.self)
    let bgHost = scene.background.asArray(Float.self)
    let flattenHost = bin.flattenIds.asArray(Int32.self)
    let offsetsHost = bin.tileOffsets.asArray(Int32.self)
    let cotColorHost = cotColor.asArray(Float.self)
    let cotAlphaHost = cotAlpha.asArray(Float.self)

    func loss(
        means: [Float] = [], conics: [Float] = [],
        colors: [Float] = [], opacities: [Float] = []
    ) -> Float {
        sceneLoss(
            means2d: means.isEmpty ? meansHost : means,
            conics: conics.isEmpty ? conicsHost : conics,
            colors: colors.isEmpty ? colorsHost : colors,
            opacities: opacities.isEmpty ? opacitiesHost : opacities,
            background: bgHost,
            flattenIds: flattenHost, tileOffsets: offsetsHost,
            imageWidth: W, imageHeight: H, tileSize: scene.tileSize,
            cotColor: cotColorHost, cotAlpha: cotAlphaHost
        )
    }

    func check(
        label: String, primal: [Float], grad: [Float], eps: Float,
        perturb: (_ idx: Int, _ delta: Float) -> Float,
        absTol: Float = 5e-3, relTol: Float = 5e-2
    ) {
        var maxAbs: Float = 0, maxRel: Float = 0
        var worstIdx = -1
        for idx in 0..<primal.count {
            let lp = perturb(idx, +eps)
            let lm = perturb(idx, -eps)
            let numerical = (lp - lm) / (2 * eps)
            let analytical = grad[idx]
            let absDiff = abs(numerical - analytical)
            let scale = max(abs(numerical), abs(analytical), 1e-4)
            if absDiff > maxAbs { maxAbs = absDiff; worstIdx = idx }
            maxRel = max(maxRel, absDiff / scale)
        }
        #expect(
            maxAbs < absTol || maxRel < relTol,
            "\(label) grad off: maxAbs=\(maxAbs) maxRel=\(maxRel) at idx=\(worstIdx)"
        )
    }

    // ---- v_colors (linear in colors, easiest to check) ----
    var pc = colorsHost
    check(label: "v_colors", primal: colorsHost, grad: analyticalColors, eps: 1e-3,
          perturb: { idx, delta in
              pc[idx] = colorsHost[idx] + delta
              defer { pc[idx] = colorsHost[idx] }
              return loss(colors: pc)
          })

    // ---- v_opacities ----
    var po = opacitiesHost
    check(label: "v_opacities", primal: opacitiesHost, grad: analyticalOpacities, eps: 1e-3,
          perturb: { idx, delta in
              po[idx] = opacitiesHost[idx] + delta
              defer { po[idx] = opacitiesHost[idx] }
              return loss(opacities: po)
          })

    // ---- v_conics (nonlinear via exp(-sigma)) ----
    var pn = conicsHost
    check(label: "v_conics", primal: conicsHost, grad: analyticalConics, eps: 1e-4,
          perturb: { idx, delta in
              pn[idx] = conicsHost[idx] + delta
              defer { pn[idx] = conicsHost[idx] }
              return loss(conics: pn)
          })

    // ---- v_means2d ----
    var pm = meansHost
    check(label: "v_means2d", primal: meansHost, grad: analyticalMeans, eps: 1e-3,
          perturb: { idx, delta in
              pm[idx] = meansHost[idx] + delta
              defer { pm[idx] = meansHost[idx] }
              return loss(means: pm)
          })

    _ = n  // silence the unused warning
}

// MARK: - coop backward returns the same gradients as per-pixel backward

/// Dispatches `rasterize_bwd_coop` (SIMD-group cooperative reduction). Same arguments as
/// `rasterize_bwd`.
private func slangBwdCoop(
    scene: BwdScene,
    bin: TileBinningResult,
    renderAlpha: MLXArray, lastIds: MLXArray,
    vRenderColor: MLXArray, vRenderAlpha: MLXArray
) throws -> (vMeans2d: MLXArray, vConics: MLXArray, vColors: MLXArray, vOpacities: MLXArray) {
    let kernel = try SlangCompiledKernel.load(name: "rasterize_bwd_coop")
    let n = scene.means2d.shape[0]
    let dims = MLXArray(
        [Int32(scene.imageWidth), Int32(scene.imageHeight),
         Int32(scene.tileSize), Int32(scene.numTilesX)],
        [4]
    )
    let outputs = kernel(
        [
            scene.means2d, scene.conics, scene.colors, scene.opacities,
            bin.flattenIds, bin.tileOffsets, dims, scene.background,
            renderAlpha, lastIds, vRenderColor, vRenderAlpha,
        ],
        grid: (scene.imageWidth, scene.imageHeight, 1),
        threadGroup: (scene.tileSize, scene.tileSize, 1),
        outputShapes: [[n * 2], [n * 3], [n * 3], [n]],
        outputDTypes: [.float32, .float32, .float32, .float32],
        initValue: 0.0
    )
    eval(outputs[0], outputs[1], outputs[2], outputs[3])
    return (
        outputs[0].reshaped([n, 2]),
        outputs[1].reshaped([n, 3]),
        outputs[2].reshaped([n, 3]),
        outputs[3]
    )
}

private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    let d = MLX.abs(a - b).max().asArray(Float.self)[0]
    return d
}

private func maxRelDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    let ah = a.asArray(Float.self)
    let bh = b.asArray(Float.self)
    var worst: Float = 0
    for i in 0..<ah.count {
        let scale = max(abs(ah[i]), abs(bh[i]), 1e-4)
        worst = max(worst, abs(ah[i] - bh[i]) / scale)
    }
    return worst
}

/// coop == per-pixel backward on a scene with 2 overlapping Gaussians in 1 tile (16×16 = 8
/// SIMD-groups).
@Test
func rasterizeBwdCoopMatchesPerPixelSingleTile() throws {
    let scene = BwdScene(
        imageWidth: 16, imageHeight: 16, tileSize: 16,
        means2d: MLXArray([Float(7.0), 8.0, 9.0, 8.0], [2, 2]),
        conics: MLXArray([Float(0.04), 0.0, 0.04, 0.03, 0.005, 0.03], [2, 3]),
        colors: MLXArray([Float(1.0), 0.2, 0.1, 0.1, 0.8, 0.6], [2, 3]),
        opacities: MLXArray([Float(0.7), 0.6], [2]),
        radii: MLXArray([Int32(8), 8, 8, 8], [2, 2]),
        depths: MLXArray([Float(1.5), 3.0], [2]),
        background: MLXArray([Float(0.1), 0.2, 0.3], [3])
    )
    let W = scene.imageWidth, H = scene.imageHeight
    let totalPixels = W * H
    let bin = TileBinning.bin(
        means2d: scene.means2d, radii: scene.radii, depths: scene.depths,
        imageWidth: W, imageHeight: H, tileSize: scene.tileSize
    )
    let (_, renderAlpha, lastIds) = try slangFwd(scene: scene, bin: bin)
    MLXRandom.seed(3)
    let cotColor = MLXRandom.normal([totalPixels * 3]) * 0.1
    let cotAlpha = MLXRandom.normal([totalPixels]) * 0.1
    eval(cotColor, cotAlpha)

    let ref = try slangBwd(scene: scene, bin: bin, renderAlpha: renderAlpha,
                           lastIds: lastIds, vRenderColor: cotColor, vRenderAlpha: cotAlpha)
    let coop = try slangBwdCoop(scene: scene, bin: bin, renderAlpha: renderAlpha,
                                lastIds: lastIds, vRenderColor: cotColor, vRenderAlpha: cotAlpha)

    #expect(maxAbsDiff(ref.vColors, coop.vColors) < 1e-4, "v_colors mismatch")
    #expect(maxAbsDiff(ref.vConics, coop.vConics) < 1e-4, "v_conics mismatch")
    #expect(maxAbsDiff(ref.vMeans2d, coop.vMeans2d) < 1e-4, "v_means2d mismatch")
    #expect(maxAbsDiff(ref.vOpacities, coop.vOpacities) < 1e-4, "v_opacities mismatch")
}

/// coop == per-pixel with multiple tiles, many Gaussians and deep fragment lists (including partial
/// warps / tile borders / cull).
@Test
func rasterizeBwdCoopMatchesPerPixelMultiTile() throws {
    let dim = 64, ts = 16, n = 3000, radius = 3
    MLXRandom.seed(99)
    let means2d = MLXRandom.uniform(low: Float(0), high: Float(dim), [n, 2])
    let invv = Float(1) / Float(radius * radius)
    let conics = MLX.broadcast(MLXArray([invv, Float(0), invv], [1, 3]), to: [n, 3])
    let colors = MLXRandom.uniform(low: Float(0), high: Float(1), [n, 3])
    let opacities = MLXRandom.uniform(low: Float(0.3), high: Float(0.95), [n])
    let depths = MLXRandom.uniform(low: Float(0.5), high: Float(6), [n])
    let rad = Int32(3 * radius)
    let radii = MLXArray(Array(repeating: rad, count: n * 2), [n, 2])
    let background = MLXArray([Float(0.05), 0.1, 0.15], [3])
    eval(means2d, conics, colors, opacities, depths, radii)

    let scene = BwdScene(
        imageWidth: dim, imageHeight: dim, tileSize: ts,
        means2d: means2d, conics: conics, colors: colors,
        opacities: opacities, radii: radii, depths: depths, background: background
    )
    let totalPixels = dim * dim
    let bin = TileBinning.bin(
        means2d: means2d, radii: radii, depths: depths,
        imageWidth: dim, imageHeight: dim, tileSize: ts
    )
    let (_, renderAlpha, lastIds) = try slangFwd(scene: scene, bin: bin)
    MLXRandom.seed(5)
    let cotColor = MLXRandom.normal([totalPixels * 3]) * 0.05
    let cotAlpha = MLXRandom.normal([totalPixels]) * 0.05
    eval(cotColor, cotAlpha)

    let ref = try slangBwd(scene: scene, bin: bin, renderAlpha: renderAlpha,
                           lastIds: lastIds, vRenderColor: cotColor, vRenderAlpha: cotAlpha)
    let coop = try slangBwdCoop(scene: scene, bin: bin, renderAlpha: renderAlpha,
                                lastIds: lastIds, vRenderColor: cotColor, vRenderAlpha: cotAlpha)

    // atomic add order differs, so compare with a tolerance.
    #expect(maxRelDiff(ref.vColors, coop.vColors) < 2e-3, "v_colors rel mismatch \(maxRelDiff(ref.vColors, coop.vColors))")
    #expect(maxRelDiff(ref.vConics, coop.vConics) < 2e-3, "v_conics rel mismatch \(maxRelDiff(ref.vConics, coop.vConics))")
    #expect(maxRelDiff(ref.vMeans2d, coop.vMeans2d) < 2e-3, "v_means2d rel mismatch \(maxRelDiff(ref.vMeans2d, coop.vMeans2d))")
    #expect(maxRelDiff(ref.vOpacities, coop.vOpacities) < 2e-3, "v_opacities rel mismatch \(maxRelDiff(ref.vOpacities, coop.vOpacities))")
}

/// Bench comparing median ms of per-pixel `rasterize_bwd` and coop `rasterize_bwd_coop` (relative
/// comparison, so debug is fine). Run: `BENCH_COOP=1 swift test --filter rasterizeBwdCoopBench`
/// (tune with BENCH_N/DIM/RADIUS/ITERS/WARMUP)
@Test
func rasterizeBwdCoopBench() throws {
    let env = ProcessInfo.processInfo.environment
    guard env["BENCH_COOP"] != nil else { return }
    func envInt(_ k: String, _ d: Int) -> Int { env[k].flatMap { Int($0) } ?? d }
    let N = envInt("BENCH_N", 120_000), dim = envInt("BENCH_DIM", 512)
    let radius = envInt("BENCH_RADIUS", 4)
    let iters = envInt("BENCH_ITERS", 120), warmup = envInt("BENCH_WARMUP", 20)
    let ts = 16, totalPixels = dim * dim

    MLXRandom.seed(1234)
    let means2d = MLXRandom.uniform(low: Float(0), high: Float(dim), [N, 2])
    let invv = Float(1) / Float(radius * radius)
    let conics = MLX.broadcast(MLXArray([invv, Float(0), invv], [1, 3]), to: [N, 3])
    let colors = MLXRandom.uniform(low: Float(0), high: Float(1), [N, 3])
    let opacities = MLXRandom.uniform(low: Float(0.3), high: Float(0.95), [N])
    let depths = MLXRandom.uniform(low: Float(0.5), high: Float(6), [N])
    let radii = MLXArray(Array(repeating: Int32(3 * radius), count: N * 2), [N, 2])
    let background = MLXArray([Float(0), 0, 0], [3])
    eval(means2d, conics, colors, opacities, depths, radii)

    let scene = BwdScene(
        imageWidth: dim, imageHeight: dim, tileSize: ts,
        means2d: means2d, conics: conics, colors: colors,
        opacities: opacities, radii: radii, depths: depths, background: background
    )
    let bin = TileBinning.bin(means2d: means2d, radii: radii, depths: depths,
                              imageWidth: dim, imageHeight: dim, tileSize: ts)
    let (_, renderAlpha, lastIds) = try slangFwd(scene: scene, bin: bin)
    let cotColor = MLXRandom.normal([totalPixels * 3]) * 0.01
    let cotAlpha = MLXRandom.normal([totalPixels]) * 0.01
    eval(cotColor, cotAlpha)

    func bench(_ name: String) -> Double {
        let kernel = try! SlangCompiledKernel.load(name: name)
        let n = N
        let dims = MLXArray([Int32(dim), Int32(dim), Int32(ts), Int32(scene.numTilesX)], [4])
        func once() {
            let o = kernel(
                [means2d, conics, colors, opacities, bin.flattenIds, bin.tileOffsets,
                 dims, background, renderAlpha, lastIds, cotColor, cotAlpha],
                grid: (dim, dim, 1), threadGroup: (ts, ts, 1),
                outputShapes: [[n * 2], [n * 3], [n * 3], [n]],
                outputDTypes: [.float32, .float32, .float32, .float32], initValue: 0.0)
            eval(o[0], o[1], o[2], o[3])
        }
        for _ in 0..<warmup { once() }
        var t: [Double] = []
        for _ in 0..<iters { let s = Date(); once(); t.append(Date().timeIntervalSince(s) * 1000) }
        t.sort(); return t[t.count / 2]
    }
    let perPixel = bench("rasterize_bwd")
    let coop = bench("rasterize_bwd_coop")
    print("=== coop bwd bench: N=\(N) dim=\(dim) radius=\(radius) ===")
    print(String(format: "per-pixel (atomic/thread) = %.3f ms", perPixel))
    print(String(format: "coop (simd_sum/warp)      = %.3f ms  (%.2fx speedup)", coop, perPixel / coop))
}
