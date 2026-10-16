// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXRandom
import Testing
@testable import Trainer

/// Small scene + camera for running the renderer end to end.
private func makeSmokeScene(seed: UInt64 = 0) -> (model: GaussianModel, camera: Camera) {
    MLXRandom.seed(seed)
    let n = 4
    let means = MLXArray(
        [Float(0.0), Float(0.0), Float(4.0),
         Float(0.3), Float(0.1), Float(4.2),
         Float(-0.2), Float(0.2), Float(4.5),
         Float(0.1), Float(-0.3), Float(3.8)],
        [n, 3]
    )
    let quats = MLXArray(
        [Float(1), Float(0), Float(0), Float(0),
         Float(1), Float(0), Float(0), Float(0),
         Float(1), Float(0), Float(0), Float(0),
         Float(1), Float(0), Float(0), Float(0)],
        [n, 4]
    )
    // Keep them wide so the 1/255 alpha cull does not kick in inside the image (sharp ones make FD
    // pick up discontinuities absent from the analytic gradient).
    let logScales = MLXArray([Float](repeating: log(0.4), count: n * 3), [n, 3])
    let logitOpacities = MLXArray([Float](repeating: logit(0.7), count: n), [n])
    // Keep colors away from 0 / 1 so FD does not cross the kink of the SH relu(rgb≥0).
    let rgb = MLXArray(
        [Float(0.8), Float(0.2), Float(0.2),
         Float(0.2), Float(0.8), Float(0.2),
         Float(0.2), Float(0.2), Float(0.8),
         Float(0.6), Float(0.6), Float(0.6)],
        [n, 3]
    )
    let sh0 = (rgb - 0.5) / GaussianModel.shC0
    let shN = MLXArray.zeros([n, GaussianModel.shRestBands, 3])
    let model = GaussianModel(
        means: means, quats: quats, logScales: logScales,
        logitOpacities: logitOpacities, sh0: sh0, shN: shN
    )
    eval(model.means, model.quats, model.logScales, model.logitOpacities, model.sh0, model.shN)

    // Identity viewmat (looking along +z from the origin).
    let viewmat = MLXArray(
        [Float(1), Float(0), Float(0), Float(0),
         Float(0), Float(1), Float(0), Float(0),
         Float(0), Float(0), Float(1), Float(0),
         Float(0), Float(0), Float(0), Float(1)],
        [16]
    )
    let intrinsics = MLXArray([Float(80.0), Float(80.0), Float(16.0), Float(16.0)], [4])
    let background = MLXArray([Float(0), Float(0), Float(0)], [3])
    // Make the whole image one tile. With several tiles, FD position perturbations change tile
    // membership and cause discontinuities absent from the analytic gradient.
    let camera = Camera(
        viewmat: viewmat, intrinsics: intrinsics,
        width: 32, height: 32, tileSize: 32, background: background
    )
    return (model, camera)
}

@Test
func gaussianRendererForwardProducesPlausibleImage() throws {
    let (model, cam) = makeSmokeScene()
    let renderer = try GaussianRenderer()
    let (renderColor, renderAlpha, cache) = renderer.forward(model: model, camera: cam)
    eval(renderColor, renderAlpha)

    #expect(renderColor.shape == [cam.height, cam.width, 3])
    #expect(renderAlpha.shape == [cam.height, cam.width])

    // Some pixel receives a contribution (alpha > 0).
    let alpha = renderAlpha.asArray(Float.self)
    let maxAlpha = alpha.max() ?? 0
    #expect(maxAlpha > 0.01, "render produced no visible contribution (max alpha \(maxAlpha))")

    // All 4 are in front of the camera and inside the image, so there is at least one intersection.
    #expect(cache.bin.nIsects > 0, "tile binning produced no intersections")

    // All depths are positive.
    let depths = cache.depths.asArray(Float.self)
    for d in depths { #expect(d > 0, "depth \(d) is not positive") }
}

private struct BwdFixture {
    let model: GaussianModel
    let cam: Camera
    let renderer: GaussianRenderer
    let cache: GaussianRenderer.ForwardCache
    let grads: GaussianRenderer.Gradients
    let cotColorH: [Float]
    let cotAlphaH: [Float]
    let meansH: [Float]
    let quatsH: [Float]
    let logScalesH: [Float]
    let logitOpacH: [Float]
    let sh0H: [Float]
    let shNH: [Float]
}

private func makeBwdFixture() throws -> BwdFixture {
    let (model, cam) = makeSmokeScene(seed: 1)
    let renderer = try GaussianRenderer()
    let (_, _, cache) = renderer.forward(model: model, camera: cam)

    MLXRandom.seed(7)
    let totalPixels = cam.width * cam.height
    let vRenderColor = MLXRandom.normal([totalPixels * 3]) * 0.01
    let vRenderAlpha = MLXRandom.normal([totalPixels]) * 0.01
    eval(vRenderColor, vRenderAlpha)

    let grads = renderer.backward(
        model: model, camera: cam, cache: cache,
        vRenderColor: vRenderColor, vRenderAlpha: vRenderAlpha
    )

    return BwdFixture(
        model: model, cam: cam, renderer: renderer, cache: cache, grads: grads,
        cotColorH: vRenderColor.asArray(Float.self),
        cotAlphaH: vRenderAlpha.asArray(Float.self),
        meansH: model.means.asArray(Float.self),
        quatsH: model.quats.asArray(Float.self),
        logScalesH: model.logScales.asArray(Float.self),
        logitOpacH: model.logitOpacities.asArray(Float.self),
        sh0H: model.sh0.asArray(Float.self),
        shNH: model.shN.asArray(Float.self)
    )
}

private func snapshotModel(_ f: BwdFixture) -> GaussianModel {
    GaussianModel(
        means: MLXArray(f.meansH, [f.model.count, 3]),
        quats: MLXArray(f.quatsH, [f.model.count, 4]),
        logScales: MLXArray(f.logScalesH, [f.model.count, 3]),
        logitOpacities: MLXArray(f.logitOpacH, [f.model.count]),
        sh0: MLXArray(f.sh0H, [f.model.count, 3]),
        shN: MLXArray(f.shNH, [f.model.count, GaussianModel.shRestBands, 3])
    )
}

private func lossWith(_ f: BwdFixture, model: GaussianModel) -> Float {
    let (r, a, _) = f.renderer.forward(model: model, camera: f.cam)
    let rh = r.asArray(Float.self)
    let ah = a.asArray(Float.self)
    var sum: Float = 0
    for i in 0..<rh.count { sum += f.cotColorH[i] * rh[i] }
    for i in 0..<ah.count { sum += f.cotAlphaH[i] * ah[i] }
    return sum
}

private func checkFiniteDiff(
    label: String, host: [Float], analytical: [Float],
    f: BwdFixture, eps: Float,
    perturbAndRebuild: (inout GaussianModel, Int, Float) -> Void,
    absTol: Float = 1e-2, relTol: Float = 1e-1
) {
    var maxAbs: Float = 0, maxRel: Float = 0
    var worstIdx = -1
    var worstNum: Float = 0
    var worstAna: Float = 0
    for idx in 0..<host.count {
        var mPlus = snapshotModel(f)
        perturbAndRebuild(&mPlus, idx, +eps)
        let lp = lossWith(f, model: mPlus)
        var mMinus = snapshotModel(f)
        perturbAndRebuild(&mMinus, idx, -eps)
        let lm = lossWith(f, model: mMinus)
        let numerical = (lp - lm) / (2 * eps)
        let a = analytical[idx]
        let absD = abs(numerical - a)
        let scale = max(abs(numerical), abs(a), 1e-4)
        if absD > maxAbs {
            maxAbs = absD
            worstIdx = idx
            worstNum = numerical
            worstAna = a
        }
        maxRel = max(maxRel, absD / scale)
    }
    let ok = maxAbs < absTol || maxRel < relTol
    if !ok {
        Issue.record(
            "\(label): maxAbs=\(maxAbs) maxRel=\(maxRel) worst idx=\(worstIdx) numerical=\(worstNum) analytical=\(worstAna)"
        )
    }
    #expect(ok)
}

@Test
func rendererBackwardSh0MatchFD() throws {
    let f = try makeBwdFixture()
    let analytical = f.grads.dSh0.asArray(Float.self)
    let n = f.model.count
    checkFiniteDiff(
        label: "dSh0", host: f.sh0H, analytical: analytical, f: f, eps: 1e-3,
        perturbAndRebuild: { snap, idx, d in
            var arr = f.sh0H; arr[idx] += d
            snap.sh0 = MLXArray(arr, [n, 3])
        }
    )
}

@Test
func rendererBackwardShNMatchFD() throws {
    let f = try makeBwdFixture()
    let analytical = f.grads.dShN.asArray(Float.self)
    let n = f.model.count
    checkFiniteDiff(
        label: "dShN", host: f.shNH, analytical: analytical, f: f, eps: 1e-3,
        perturbAndRebuild: { snap, idx, d in
            var arr = f.shNH; arr[idx] += d
            snap.shN = MLXArray(arr, [n, GaussianModel.shRestBands, 3])
        }
    )
}

@Test
func rendererBackwardLogitOpacitiesMatchFD() throws {
    let f = try makeBwdFixture()
    let analytical = f.grads.dLogitOpacities.asArray(Float.self)
    let n = f.model.count
    checkFiniteDiff(
        label: "dLogitOpacities", host: f.logitOpacH, analytical: analytical, f: f, eps: 1e-3,
        perturbAndRebuild: { snap, idx, d in
            var arr = f.logitOpacH; arr[idx] += d
            snap.logitOpacities = MLXArray(arr, [n])
        }
    )
}

@Test
func rendererBackwardLogScalesMatchFD() throws {
    let f = try makeBwdFixture()
    let analytical = f.grads.dLogScales.asArray(Float.self)
    let n = f.model.count
    // means / scale move the on-screen shape, so FD crosses the rasterizer's discrete thresholds
    // (1/255 cull, sigma sign) and picks up discontinuities (slightly worse with eps2d dilation).
    // Loosen the abs tolerance. The backward itself is verified by the standalone project_to_screen
    // / rasterize FD tests and the overfit test.
    checkFiniteDiff(
        label: "dLogScales", host: f.logScalesH, analytical: analytical, f: f, eps: 1e-3,
        perturbAndRebuild: { snap, idx, d in
            var arr = f.logScalesH; arr[idx] += d
            snap.logScales = MLXArray(arr, [n, 3])
        },
        absTol: 0.08
    )
}

@Test
func rendererBackwardMeansMatchFD() throws {
    let f = try makeBwdFixture()
    let analytical = f.grads.dMeans.asArray(Float.self)
    let n = f.model.count
    checkFiniteDiff(
        label: "dMeans", host: f.meansH, analytical: analytical, f: f, eps: 1e-3,
        perturbAndRebuild: { snap, idx, d in
            var arr = f.meansH; arr[idx] += d
            snap.means = MLXArray(arr, [n, 3])
        },
        absTol: 0.08
    )
}

/// Diagnostic helper for when the FD test becomes flaky (normally disabled; add @Test to use it).
/// Does not assert; reports intermediate values via Issue.record.
private func bisectRasterBwdMeans2dWithRendererInputs() throws {
    let f = try makeBwdFixture()

    // Re-run the bwd ourselves to get the rasterize_bwd output (v_means2d) in f.grads on its own.
    let n = f.model.count
    let kernel = try SlangCompiledKernel.load(name: "rasterize_bwd")
    let totalPixels = f.cam.width * f.cam.height
    let dims = MLXArray(
        [Int32(f.cam.width), Int32(f.cam.height),
         Int32(f.cam.tileSize), Int32(f.cache.bin.numTilesX)],
        [4]
    )
    let cotColor = MLXArray(f.cotColorH, [totalPixels * 3])
    let cotAlpha = MLXArray(f.cotAlphaH, [totalPixels])
    let outputs = kernel(
        [
            f.cache.means2d, f.cache.conics, f.cache.colors, f.cache.opacities,
            f.cache.bin.flattenIds, f.cache.bin.tileOffsets, dims, f.cam.background,
            f.cache.renderAlpha.reshaped([totalPixels]),
            f.cache.lastIds,
            cotColor, cotAlpha,
        ],
        grid: (f.cam.width, f.cam.height, 1),
        threadGroup: (f.cam.tileSize, f.cam.tileSize, 1),
        outputShapes: [[n * 2], [n * 3], [n * 3], [n]],
        outputDTypes: [.float32, .float32, .float32, .float32],
        initValue: 0.0
    )
    eval(outputs[0])
    let vMeans2dAnalytical = outputs[0].asArray(Float.self)
    Issue.record("vMeans2d (from renderer's rasterize bwd, n=\(n))=\(vMeans2dAnalytical)")

    // FD: perturb means2d directly and take the rasterize loss.
    let means2dHost = f.cache.means2d.asArray(Float.self)
    let conicsHost = f.cache.conics.asArray(Float.self)
    let colorsHost = f.cache.colors.asArray(Float.self)
    let opacitiesHost = f.cache.opacities.asArray(Float.self)
    let flattenHost = f.cache.bin.flattenIds.asArray(Int32.self)
    let offsetsHost = f.cache.bin.tileOffsets.asArray(Int32.self)
    let bgHost = f.cam.background.asArray(Float.self)

    func lossMeans2d(_ m: [Float]) -> Float {
        let r = referenceRasterize(
            means2d: m, conics: conicsHost, colors: colorsHost, opacities: opacitiesHost,
            flattenIds: flattenHost, tileOffsets: offsetsHost,
            imageWidth: f.cam.width, imageHeight: f.cam.height, tileSize: f.cam.tileSize,
            background: bgHost
        )
        var sum: Float = 0
        for i in 0..<r.renderColor.count { sum += f.cotColorH[i] * r.renderColor[i] }
        for i in 0..<r.renderAlpha.count { sum += f.cotAlphaH[i] * r.renderAlpha[i] }
        return sum
    }

    var maxAbs: Float = 0
    var worstIdx = -1
    var worstNum: Float = 0
    var worstAna: Float = 0
    let eps: Float = 1e-3
    var pm = means2dHost
    for idx in 0..<(n * 2) {
        pm[idx] = means2dHost[idx] + eps
        let lp = lossMeans2d(pm)
        pm[idx] = means2dHost[idx] - eps
        let lm = lossMeans2d(pm)
        pm[idx] = means2dHost[idx]
        let numerical = (lp - lm) / (2 * eps)
        let analyticalVal = vMeans2dAnalytical[idx]
        let absD = abs(numerical - analyticalVal)
        if absD > maxAbs {
            maxAbs = absD; worstIdx = idx
            worstNum = numerical; worstAna = analyticalVal
        }
    }
    Issue.record("rasterize_bwd vMeans2d (renderer inputs): maxAbs=\(maxAbs) worst idx=\(worstIdx) num=\(worstNum) ana=\(worstAna)")

    // Also output vConics, run project_to_screen_bwd on its own, and compare against FD of
    // world_means perturbation with bins fixed.
    eval(outputs[1])
    let vConicsAnalytical = outputs[1].asArray(Float.self)
    Issue.record("vConics (from rasterize bwd)=\(vConicsAnalytical)")

    let pBwdKernel = try SlangCompiledKernel.load(name: "project_to_screen_bwd")
    let vMeans2dArr = outputs[0]
    let vConicsArr = outputs[1]
    let imageDims = MLXArray([Float(f.cam.width), Float(f.cam.height)], [2])
    let nBufP = MLXArray([Int32(n)], [1])
    let pBwdOut = pBwdKernel(
        [
            f.model.means, f.model.quats, f.cache.scales,
            f.cam.viewmat, f.cam.intrinsics, imageDims, nBufP,
            vMeans2dArr.reshaped([n, 2]), vConicsArr.reshaped([n, 3]),
        ],
        grid: (((n + 63) / 64) * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[n * 3], [n * 4], [n * 3]],
        outputDTypes: [.float32, .float32, .float32]
    )
    eval(pBwdOut[0])
    let pBwdDMeans = pBwdOut[0].asArray(Float.self)
    Issue.record("project_bwd standalone dMeans=\(pBwdDMeans)")

    // Loss of world_means perturbation through project_to_screen_fwd → rasterize with bins fixed.
    func lossProjThenRaster(meansHost: [Float]) -> Float {
        let projKernel: MLXFast.MLXFastKernel
        do { projKernel = try SlangCompiledKernel.load(name: "project_to_screen_fwd") } catch {
            Issue.record("failed to load project_to_screen_fwd: \(error)")
            return 0
        }
        let m = MLXArray(meansHost, [n, 3])
        let projOut = projKernel(
            [m, f.model.quats, f.cache.scales, f.cam.viewmat, f.cam.intrinsics,
             MLXArray([Float(f.cam.width), Float(f.cam.height)], [2]),
             f.cache.opacities,
             MLXArray([Int32(n)], [1])],
            grid: (((n + 63) / 64) * 64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[n * 2], [n * 3], [n], [n * 2]],
            outputDTypes: [.float32, .float32, .float32, .int32]
        )
        eval(projOut[0], projOut[1])
        let newMeans2d = projOut[0].asArray(Float.self)
        let newConics  = projOut[1].asArray(Float.self)
        return referenceRasterize(
            means2d: newMeans2d, conics: newConics,
            colors: colorsHost, opacities: opacitiesHost,
            flattenIds: flattenHost, tileOffsets: offsetsHost,
            imageWidth: f.cam.width, imageHeight: f.cam.height, tileSize: f.cam.tileSize,
            background: bgHost
        ).renderColor.enumerated().reduce(0.0) { $0 + f.cotColorH[$1.offset] * $1.element }
        + referenceRasterize(
            means2d: newMeans2d, conics: newConics,
            colors: colorsHost, opacities: opacitiesHost,
            flattenIds: flattenHost, tileOffsets: offsetsHost,
            imageWidth: f.cam.width, imageHeight: f.cam.height, tileSize: f.cam.tileSize,
            background: bgHost
        ).renderAlpha.enumerated().reduce(0.0) { $0 + f.cotAlphaH[$1.offset] * $1.element }
    }

    let meansFixed = f.meansH

    // Look at project_to_screen_fwd means2d for idx 1 (g0_y) at original / +eps / -eps.
    func projectMeans2d(_ meansHost: [Float]) -> [Float] {
        let projKernel = try! SlangCompiledKernel.load(name: "project_to_screen_fwd")
        let m = MLXArray(meansHost, [n, 3])
        let projOut = projKernel(
            [m, f.model.quats, f.cache.scales, f.cam.viewmat, f.cam.intrinsics,
             MLXArray([Float(f.cam.width), Float(f.cam.height)], [2]),
             f.cache.opacities,
             MLXArray([Int32(n)], [1])],
            grid: (((n + 63) / 64) * 64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[n * 2], [n * 3], [n], [n * 2]],
            outputDTypes: [.float32, .float32, .float32, .int32]
        )
        eval(projOut[0])
        return projOut[0].asArray(Float.self)
    }
    let m2dOrig = projectMeans2d(meansFixed)
    var mPlus = meansFixed; mPlus[1] += eps
    var mMinus = meansFixed; mMinus[1] -= eps
    let m2dPlus = projectMeans2d(mPlus)
    let m2dMinus = projectMeans2d(mMinus)
    Issue.record("means2d original g0=(\(m2dOrig[0]), \(m2dOrig[1])); +eps_y g0=(\(m2dPlus[0]), \(m2dPlus[1])); -eps_y g0=(\(m2dMinus[0]), \(m2dMinus[1]))")
    Issue.record("cache means2d g0=(\(means2dHost[0]), \(means2dHost[1])); freshly projected at meansFixed g0=(\(m2dOrig[0]), \(m2dOrig[1]))")
    let lossOrig = lossProjThenRaster(meansHost: meansFixed)
    let lossPlus = lossProjThenRaster(meansHost: mPlus)
    let lossMinus = lossProjThenRaster(meansHost: mMinus)
    let lossViaCacheM2d = lossMeans2d(means2dHost)
    Issue.record("loss orig (via project_fwd)=\(lossOrig); via cached means2d=\(lossViaCacheM2d); +eps_y(world)=\(lossPlus); -eps_y(world)=\(lossMinus); FD=\(((lossPlus - lossMinus) / (2 * eps)))")

    // Feed m2dPlus / m2dMinus straight into a rasterize-only loss to isolate whether the difference
    // is on the rasterize side.
    let lossViaDirectM2dPlus = lossMeans2d(m2dPlus)
    let lossViaDirectM2dMinus = lossMeans2d(m2dMinus)
    Issue.record("via direct means2d perturb: +eps loss=\(lossViaDirectM2dPlus); -eps loss=\(lossViaDirectM2dMinus); FD=\(((lossViaDirectM2dPlus - lossViaDirectM2dMinus) / (2 * eps)))")

    var maxAbsW: Float = 0
    var worstWIdx = -1
    var worstWNum: Float = 0
    var worstWAna: Float = 0
    var pw = meansFixed
    for idx in 0..<(n * 3) {
        pw[idx] = meansFixed[idx] + eps
        let lp = lossProjThenRaster(meansHost: pw)
        pw[idx] = meansFixed[idx] - eps
        let lm = lossProjThenRaster(meansHost: pw)
        pw[idx] = meansFixed[idx]
        let numerical = (lp - lm) / (2 * eps)
        let analyticalVal = pBwdDMeans[idx]
        let absD = abs(numerical - analyticalVal)
        if absD > maxAbsW {
            maxAbsW = absD; worstWIdx = idx
            worstWNum = numerical; worstWAna = analyticalVal
        }
    }
    Issue.record("project_bwd via FD (frozen bin): maxAbs=\(maxAbsW) worst idx=\(worstWIdx) num=\(worstWNum) ana=\(worstWAna)")
}

/// Minimal scene: single Gaussian, identity view, single tile (to isolate and check the means
/// gradient).
private func makeSingleGaussianFixture() throws -> BwdFixture {
    let n = 1
    let rgb = MLXArray([Float(0.8), Float(0.3), Float(0.6)], [n, 3])
    let model = GaussianModel(
        means: MLXArray([Float(0.2), Float(0.1), Float(4.0)], [n, 3]),
        quats: MLXArray([Float(1), Float(0), Float(0), Float(0)], [n, 4]),
        logScales: MLXArray([Float(log(0.15)), Float(log(0.15)), Float(log(0.15))], [n, 3]),
        logitOpacities: MLXArray([logit(0.7)], [n]),
        sh0: (rgb - 0.5) / GaussianModel.shC0,
        shN: MLXArray.zeros([n, GaussianModel.shRestBands, 3])
    )
    eval(model.means, model.quats, model.logScales, model.logitOpacities, model.sh0, model.shN)

    let viewmat = MLXArray(
        [Float(1), Float(0), Float(0), Float(0),
         Float(0), Float(1), Float(0), Float(0),
         Float(0), Float(0), Float(1), Float(0),
         Float(0), Float(0), Float(0), Float(1)],
        [16]
    )
    let cam = Camera(
        viewmat: viewmat,
        intrinsics: MLXArray([Float(80), Float(80), Float(8), Float(8)], [4]),
        width: 16, height: 16, tileSize: 16,
        background: MLXArray([Float(0), Float(0), Float(0)], [3])
    )

    let renderer = try GaussianRenderer()
    let (_, _, cache) = renderer.forward(model: model, camera: cam)

    MLXRandom.seed(99)
    let totalPixels = cam.width * cam.height
    let vRenderColor = MLXRandom.normal([totalPixels * 3]) * 0.01
    let vRenderAlpha = MLXRandom.normal([totalPixels]) * 0.01
    eval(vRenderColor, vRenderAlpha)

    let grads = renderer.backward(
        model: model, camera: cam, cache: cache,
        vRenderColor: vRenderColor, vRenderAlpha: vRenderAlpha
    )

    return BwdFixture(
        model: model, cam: cam, renderer: renderer, cache: cache, grads: grads,
        cotColorH: vRenderColor.asArray(Float.self),
        cotAlphaH: vRenderAlpha.asArray(Float.self),
        meansH: model.means.asArray(Float.self),
        quatsH: model.quats.asArray(Float.self),
        logScalesH: model.logScales.asArray(Float.self),
        logitOpacH: model.logitOpacities.asArray(Float.self),
        sh0H: model.sh0.asArray(Float.self),
        shNH: model.shN.asArray(Float.self)
    )
}

@Test
func rendererBackwardMeansSingleGaussian() throws {
    let f = try makeSingleGaussianFixture()
    let analytical = f.grads.dMeans.asArray(Float.self)
    checkFiniteDiff(
        label: "dMeans (single)", host: f.meansH, analytical: analytical, f: f, eps: 1e-3,
        perturbAndRebuild: { snap, idx, d in
            var arr = f.meansH; arr[idx] += d
            snap.means = MLXArray(arr, [1, 3])
        },
        absTol: 0.08
    )
}

@Test
func rendererBackwardQuatsMatchFD() throws {
    let f = try makeBwdFixture()
    let analytical = f.grads.dQuats.asArray(Float.self)
    let n = f.model.count
    checkFiniteDiff(
        label: "dQuats", host: f.quatsH, analytical: analytical, f: f, eps: 1e-3,
        perturbAndRebuild: { snap, idx, d in
            var arr = f.quatsH; arr[idx] += d
            snap.quats = MLXArray(arr, [n, 4])
        }
    )
}
