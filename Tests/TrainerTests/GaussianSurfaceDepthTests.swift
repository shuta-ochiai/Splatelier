// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Gradient of the Gaussian surface depth (`gof_depth_bwd` + the frame chain in MLX) against
// central differences of the forward depth.

import Foundation
import MLX
import Testing
@testable import Trainer

/// One, then three partly overlapping, rotated, anisotropic Gaussians in front of the camera. The loss
/// is a fixed positive weighting of the depth over interior pixels; every parameter of every
/// Gaussian is checked. Opacity-field profile only: with the stochastic solid, an isolated
/// Gaussian's depth has a vertical tangent along its peak-alpha 0.75 contour (the crossing moves
/// from in front of the peak to behind it as √ln(p/0.75)), which central differences cannot
/// resolve; that profile is checked pixel by pixel against a float64 reference below.
@Test(arguments: [GaussianOpacityField.TransmittanceProfile.opacityField], [1, 3])
func surfaceDepthGradientMatchesFiniteDifferences(
    profile: GaussianOpacityField.TransmittanceProfile, count: Int
) throws {
    // means, quats, logScales, logit opacities.
    var params: [[Float]] = [
        Array([0.02, -0.01, 1.6,  -0.1, 0.03, 1.75,  0.12, -0.02, 1.9].prefix(count * 3)),
        Array([0.9, 0.2, 0.1, -0.3,  0.8, -0.1, 0.4, 0.2,  1, 0.05, -0.2, 0.1].prefix(count * 4)),
        Array([log(0.12), log(0.08), log(0.03), log(0.09), log(0.1), log(0.045), log(0.15), log(0.06),
               log(0.075)].prefix(count * 3)),
        Array([logit(0.9), logit(0.85), logit(0.9)].prefix(count)),
    ]
    let w = 48, h = 48
    let camera = Camera(
        viewmat: MLXArray([Float(1), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [16]),
        intrinsics: MLXArray([Float(150), 150, 24, 24], [4]),
        width: w, height: h, background: MLXArray.zeros([3]))
    let renderer = try GaussianRenderer()
    let parameters = GaussianSurfaceDepth.Parameters(profile: profile, bisectionIterations: 36)

    func model() -> GaussianModel {
        GaussianModel(
            means: MLXArray(params[0], [count, 3]), quats: MLXArray(params[1], [count, 4]),
            logScales: MLXArray(params[2], [count, 3]), logitOpacities: MLXArray(params[3], [count]),
            sh0: MLXArray.zeros([count, 3]), shN: MLXArray.zeros([count, GaussianModel.shRestBands, 3]))
    }
    func depth() -> (MLXArray, GaussianModel, GaussianRenderer.ForwardCache, GaussianSurfaceDepth.Result) {
        let m = model()
        let (_, _, cache) = renderer.forward(model: m, camera: camera, shDegree: 0)
        let r = GaussianSurfaceDepth.render(model: m, camera: camera, cache: cache, parameters: parameters)!
        return (r.depth, m, cache, r)
    }

    let (d0, m0, cache0, r0) = depth()
    let base = d0.asArray(Float.self)
    var seed: UInt64 = 12345
    func nextWeight() -> Float {
        seed = seed &* 6364136223846793005 &+ 1442695040888963407
        return 0.5 + Float((seed >> 33) % 1000) / 2000
    }
    // Only interior pixels: valid, with valid neighbors (5×5) at a nearby depth. At the edge of the 0.5
    // contour a small change flips a pixel to "never reached" (and the depth slope blows up), and
    // where the crossing jumps from one Gaussian to the one behind the depth is discontinuous.
    func interior(_ i: Int) -> Bool {
        let x = i % w, y = i / w
        guard x > 1, y > 1, x < w - 2, y < h - 2, base[i] > 0 else { return false }
        for dy in -2...2 {
            for dx in -2...2 {
                let j = (y + dy) * w + x + dx
                let ring = Float(max(abs(dx), abs(dy)))
                if base[j] <= 0 || abs(base[j] - base[i]) > 0.015 * ring { return false }
            }
        }
        return true
    }
    let weights = base.indices.map { interior($0) ? nextWeight() : 0 }
    let weightedPixels = weights.filter { $0 != 0 }.count
    #expect(weightedPixels > 30 * count, "the Gaussians should cover a good part of the image")

    let g = GaussianSurfaceDepth.backward(
        r0, model: m0, camera: camera, cache: cache0, vDepth: MLXArray(weights, [h, w]))
    let analytic: [[Float]] = [g.dMeans.asArray(Float.self), g.dQuats.asArray(Float.self),
                               g.dLogScales.asArray(Float.self), g.dLogitOpacities.asArray(Float.self)]

    func loss() -> Double {
        let d = depth().0.asArray(Float.self)
        var s = 0.0
        for i in 0..<d.count where weights[i] != 0 {
            #expect(d[i] > 0, "an interior pixel lost its depth")
            s += Double(weights[i]) * Double(d[i])
        }
        return s
    }
    // The float32 depth is good to ~1e-7, so a larger step keeps the difference above that noise.
    let eps: Float = 4e-3
    var checked = 0
    let names = ["means", "quats", "logScales", "logits"]
    func check(_ kind: Int, _ index: Int) {
        let v = params[kind][index]
        params[kind][index] = v + eps
        let lp = loss()
        params[kind][index] = v - eps
        let lm = loss()
        params[kind][index] = v
        let numeric = Float((lp - lm) / Double(2 * eps))
        let a = analytic[kind][index]
        let tol = 1e-3 * Float(weightedPixels) + 0.03 * max(abs(numeric), abs(a))
        #expect(abs(a - numeric) < tol, "\(profile) n\(count) px \(weightedPixels) \(names[kind])[\(index)]: analytic \(a) vs numeric \(numeric)")
        checked += 1
    }
    for kind in 0..<4 {
        for i in 0..<params[kind].count { check(kind, i) }
    }
    #expect(checked == count * 11)
    // Something must actually flow to every Gaussian.
    for i in 0..<count {
        #expect(abs(analytic[0][i * 3]) + abs(analytic[0][i * 3 + 1]) + abs(analytic[0][i * 3 + 2]) > 0)
    }
}

/// Per-pixel gradient of the depth of one Gaussian against a float64 reference (the 0.5 crossing
/// solved by bisection and differentiated by central differences, outside this code base).
@Test
func surfaceDepthGradientOfOnePixelMatchesReference() throws {
    let w = 48, h = 48
    let camera = Camera(
        viewmat: MLXArray([Float(1), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [16]),
        intrinsics: MLXArray([Float(150), 150, 24, 24], [4]),
        width: w, height: h, background: MLXArray.zeros([3]))
    let model = GaussianModel(
        means: MLXArray([Float(0.02), -0.01, 1.6], [1, 3]),
        quats: MLXArray([Float(0.9), 0.2, 0.1, -0.3], [1, 4]),
        logScales: MLXArray([log(Float(0.08)), log(Float(0.05)), log(Float(0.02))], [1, 3]),
        logitOpacities: MLXArray([logit(Float(0.7))], [1]),
        sh0: MLXArray.zeros([1, 3]), shN: MLXArray.zeros([1, GaussianModel.shRestBands, 3]))
    // (profile, pixel, depth, d/d means, d/d quats, d/d logScales, d/d logit)
    let reference: [(GaussianOpacityField.TransmittanceProfile, (Int, Int), Float, [Float], [Float], [Float], Float)] = [
        (.opacityField, (24, 24), 1.590874, [0.13703, -0.53880, 1.00134], [-0.00354, 0.00999, 0.03454, 0.00755],
         [-0.00127, -0.00001, -0.01812], -0.00865),
        (.opacityField, (21, 25), 1.605719, [0.51757, -0.65031, 1.01513], [-0.01263, 0.02175, 0.12173, 0.01718],
         [-0.02256, -0.00128, -0.01149], -0.01576),
        (.stochasticSolid, (24, 24), 1.620662, [0.22496, -0.74573, 1.00174], [-0.00922, 0.03475, 0.05035, 0.01230],
         [-0.00498, 0.00014, 0.01077], -0.02413),
        (.stochasticSolid, (21, 25), 1.648319, [1.31965, -1.06918, 1.03269], [-0.02760, 0.05900, 0.17008, 0.01322],
         [-0.06933, -0.00140, 0.02966], -0.04506),
    ]
    let renderer = try GaussianRenderer()
    let (_, _, cache) = renderer.forward(model: model, camera: camera, shDegree: 0)
    for (profile, (px, py), depth, dMeans, dQuats, dLogScales, dLogit) in reference {
        let r = GaussianSurfaceDepth.render(
            model: model, camera: camera, cache: cache,
            parameters: .init(profile: profile, bisectionIterations: 36))!
        #expect(abs(r.depth.asArray(Float.self)[py * w + px] - depth) < 1e-5)
        var v = [Float](repeating: 0, count: w * h)
        v[py * w + px] = 1
        let g = GaussianSurfaceDepth.backward(r, model: model, camera: camera, cache: cache, vDepth: MLXArray(v, [h, w]))
        let got = [g.dMeans.asArray(Float.self), g.dQuats.asArray(Float.self), g.dLogScales.asArray(Float.self),
                   g.dLogitOpacities.asArray(Float.self)]
        for (k, expected) in [dMeans, dQuats, dLogScales, [dLogit]].enumerated() {
            for i in expected.indices {
                // The reference's central differences are good to ~1e-3 of the largest entry.
                #expect(abs(got[k][i] - expected[i]) < 2e-3 * max(1, abs(expected[i])) + 1e-4,
                        "\(profile) (\(px), \(py)) kind \(k)[\(i)]: \(got[k][i]) vs \(expected[i])")
            }
        }
    }
}

/// The normal map of the geometric regularization: dL/d(features) of `backwardFeatures` against
/// central differences, and the parameter gradients through the alpha weights equal with and
/// without the replay cache. (Those alpha-weight gradients
/// are the rasterizer's own, whose finite differences are dominated by the 1/255 alpha cutoff and
/// the tile footprint.)
@Test
func normalMapGradients() throws {
    let params: [[Float]] = [
        [0.02, -0.01, 1.6,  -0.1, 0.03, 1.75,  0.12, -0.02, 1.9],
        [0.9, 0.2, 0.1, -0.3,  0.8, -0.1, 0.4, 0.2,  1, 0.05, -0.2, 0.1],
        [log(0.12), log(0.08), log(0.03), log(0.09), log(0.1), log(0.045), log(0.15), log(0.06), log(0.075)],
        [logit(0.9), logit(0.85), logit(0.9)],
    ]
    let w = 48, h = 48
    let camera = Camera(
        viewmat: MLXArray([Float(0.98), 0, -0.199, 0.05, 0, 1, 0, 0.02, 0.199, 0, 0.98, 0.1, 0, 0, 0, 1], [16]),
        intrinsics: MLXArray([Float(150), 150, 24, 24], [4]),
        width: w, height: h, background: MLXArray.zeros([3]))
    var seed: UInt64 = 7
    func next() -> Float {
        seed = seed &* 6364136223846793005 &+ 1442695040888963407
        return Float((seed >> 33) % 1000) / 1000 - 0.5
    }
    let weightN = MLXArray((0 ..< h * w * 3).map { _ in next() }, [h, w, 3])
    let weightA = MLXArray((0 ..< h * w).map { _ in next() }, [h, w])
    let model = GaussianModel(
        means: MLXArray(params[0], [3, 3]), quats: MLXArray(params[1], [3, 4]),
        logScales: MLXArray(params[2], [3, 3]), logitOpacities: MLXArray(params[3], [3]),
        sh0: MLXArray.zeros([3, 3]), shN: MLXArray.zeros([3, GaussianModel.shRestBands, 3]))

    var byReplay: [[[Float]]] = []
    var maps: [[Float]] = []
    for replay in [false, true] {
        let renderer = try GaussianRenderer()
        renderer.useReplayCache = replay
        let (_, _, cache) = renderer.forward(model: model, camera: camera, shDegree: 0)
        let normals = GeometryRegularization.cameraNormals(
            means: model.means, quats: model.quats, scales: cache.scales, camera: camera)
        let map = renderer.renderFeatures(normals, camera: camera, cache: cache)
        maps.append(map.asArray(Float.self))
        if replay {
            // Strided compositing from the replay cache: the sampled pixels as at full resolution,
            // zero elsewhere.
            let s = 3
            let strided = renderer.renderFeatures(normals, camera: camera, cache: cache, stride: s).asArray(Float.self)
            for y in 0 ..< h {
                for x in 0 ..< w {
                    let sampled = y % s == s / 2 && x % s == s / 2 && y / s < h / s && x / s < w / s
                    for c in 0 ..< 3 {
                        let i = (y * w + x) * 3 + c
                        #expect(abs(strided[i] - (sampled ? maps[1][i] : 0)) < 1e-6)
                    }
                }
            }
        }
        let g = renderer.backwardFeatures(
            model: model, camera: camera, cache: cache, features: normals, rendered: map,
            vRendered: weightN, vRenderAlpha: weightA)
        byReplay.append([g.dMeans.asArray(Float.self), g.dQuats.asArray(Float.self),
                         g.dLogScales.asArray(Float.self), g.dLogitOpacities.asArray(Float.self),
                         g.vFeatures.asArray(Float.self)])
        guard !replay else { continue }

        // dL/d(features), L = Σ weightN · renderFeatures(features).
        let f0 = normals.asArray(Float.self)
        let vFeatures = g.vFeatures.asArray(Float.self)
        func lossF(_ f: [Float]) -> Float {
            (renderer.renderFeatures(MLXArray(f, [3, 3]), camera: camera, cache: cache) * weightN).sum().item(Float.self)
        }
        for i in f0.indices {
            var fp = f0, fm = f0
            fp[i] += 0.05
            fm[i] -= 0.05
            let numeric = (lossF(fp) - lossF(fm)) / 0.1
            #expect(abs(vFeatures[i] - numeric) < 0.01 + 0.01 * abs(numeric), "features[\(i)]: \(vFeatures[i]) vs \(numeric)")
        }
    }
    // The replay compositing equals the rasterizer's (fp16 features in both).
    #expect(maps[0].indices.allSatisfy { abs(maps[0][$0] - maps[1][$0]) < 1e-5 })
    for (k, (a, b)) in zip(byReplay[0], byReplay[1]).enumerated() {
        for i in a.indices {
            #expect(abs(a[i] - b[i]) < 1e-3 + 1e-3 * abs(a[i]), "replay mismatch kind \(k)[\(i)]: \(a[i]) vs \(b[i])")
        }
    }
}

// MARK: - Per-Gaussian kernels (geo_reg.slang) against MLX references

private func cameraCenter(_ camera: Camera) -> MLXArray {
    let v = camera.viewmat.asArray(Float.self)
    return MLXArray([-(v[0] * v[3] + v[4] * v[7] + v[8] * v[11]),
                     -(v[1] * v[3] + v[5] * v[7] + v[9] * v[11]),
                     -(v[2] * v[3] + v[6] * v[7] + v[10] * v[11])], [3])
}

/// gs_math.quat_to_rotmat in MLX: the 9 entries (row-major) as [N] arrays.
private func rotation(_ quats: MLXArray) -> [MLXArray] {
    let q = quats / MLX.sqrt((quats * quats).sum(axis: 1, keepDims: true))
    let w = q[0..., 0], x = q[0..., 1], y = q[0..., 2], z = q[0..., 3]
    return [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
}

/// The `gof_prepare` frame [N, 12] in MLX (rows of M = S⁻¹Rᵀ, then o_g = M (campos - μ)).
private func referenceFrames(campos: MLXArray) -> ([MLXArray]) -> [MLXArray] {
    { x in
        let r = rotation(x[1])
        let inv = 1 / MLX.maximum(x[2], MLXArray(Float(1e-9)))
        let m = [r[0] * inv[0..., 0], r[3] * inv[0..., 0], r[6] * inv[0..., 0],
                 r[1] * inv[0..., 1], r[4] * inv[0..., 1], r[7] * inv[0..., 1],
                 r[2] * inv[0..., 2], r[5] * inv[0..., 2], r[8] * inv[0..., 2]]
        let p = campos.reshaped([1, 3]) - x[0]
        let px = p[0..., 0], py = p[0..., 1], pz = p[0..., 2]
        let og = [m[0] * px + m[1] * py + m[2] * pz, m[3] * px + m[4] * py + m[5] * pz,
                  m[6] * px + m[7] * py + m[8] * pz]
        return [MLX.stacked(m + og, axis: 1)]
    }
}

/// The camera-space normal of geo_normals_fwd in MLX (the facing sign held constant).
private func referenceNormals(campos: MLXArray, viewmat v: [Float]) -> ([MLXArray]) -> [MLXArray] {
    { x in
        let r = rotation(x[1])
        let d = x[0] - campos.reshaped([1, 3])
        let d0 = d[0..., 0], d1 = d[0..., 1], d2 = d[0..., 2]
        let inv2 = 1 / MLX.maximum(x[2] * x[2], MLXArray(Float(1e-20)))
        let b0 = (r[0] * d0 + r[3] * d1 + r[6] * d2) * inv2[0..., 0]
        let b1 = (r[1] * d0 + r[4] * d1 + r[7] * d2) * inv2[0..., 1]
        let b2 = (r[2] * d0 + r[5] * d1 + r[8] * d2) * inv2[0..., 2]
        let n0 = r[0] * b0 + r[1] * b1 + r[2] * b2
        let n1 = r[3] * b0 + r[4] * b1 + r[5] * b2
        let n2 = r[6] * b0 + r[7] * b1 + r[8] * b2
        let len = MLX.sqrt(n0 * n0 + n1 * n1 + n2 * n2)
        let flip = stopGradient(MLX.where((n0 * d0 + n1 * d1 + n2 * d2) .> 0, MLXArray(Float(-1)), MLXArray(Float(1))))
        let w0 = n0 * flip / len, w1 = n1 * flip / len, w2 = n2 * flip / len
        return [MLX.stacked([v[0] * w0 + v[1] * w1 + v[2] * w2, v[4] * w0 + v[5] * w1 + v[6] * w2,
                             v[8] * w0 + v[9] * w1 + v[10] * w2], axis: 1)]
    }
}

/// geo_normals_fwd / geo_normals_bwd / gof_frame_bwd against their MLX formulas (forward values and
/// vjp) on random Gaussians seen from a rotated camera.
@Test
func geometryKernelsMatchMLXReference() throws {
    let n = 64
    var seed: UInt64 = 11
    func next() -> Float {
        seed = seed &* 6364136223846793005 &+ 1442695040888963407
        return Float((seed >> 33) % 10000) / 10000 - 0.5
    }
    let means = MLXArray((0 ..< n * 3).map { i in i % 3 == 2 ? 2 + next() : next() }, [n, 3])
    let quats = MLXArray((0 ..< n * 4).map { _ in next() + 0.6 }, [n, 4])
    let scales = MLXArray((0 ..< n * 3).map { _ in 0.02 + 0.1 * (next() + 0.5) }, [n, 3])
    let camera = Camera(
        viewmat: MLXArray([Float(0.98), 0, -0.199, 0.05, 0, 1, 0, 0.02, 0.199, 0, 0.98, 0.1, 0, 0, 0, 1], [16]),
        intrinsics: MLXArray([Float(150), 150, 24, 24], [4]),
        width: 48, height: 48, background: MLXArray.zeros([3]))
    let campos = cameraCenter(camera)
    let normalRef = referenceNormals(campos: campos, viewmat: camera.viewmat.asArray(Float.self))
    // Relative error per entry, against at least 1% of the largest entry.
    func close(_ a: MLXArray, _ b: MLXArray, _ what: String, tol: Float) {
        let x = a.asArray(Float.self), y = b.asArray(Float.self)
        let floor = 0.01 * (y.map { abs($0) }.max() ?? 0) + 1e-12
        var worst: Float = 0
        for i in x.indices { worst = max(worst, abs(x[i] - y[i]) / max(floor, abs(y[i]))) }
        #expect(worst < tol, "\(what): worst relative error \(worst)")
    }

    let normals = GeometryRegularization.cameraNormals(means: means, quats: quats, scales: scales, camera: camera)
    close(normals, normalRef([means, quats, scales])[0], "normals", tol: 1e-4)

    let vNormals = MLXArray((0 ..< n * 3).map { _ in next() }, [n, 3])
    let gk = GeometryRegularization.cameraNormalsBackward(
        means: means, quats: quats, scales: scales, camera: camera, vNormals: vNormals)
    let (_, gn) = vjp(normalRef, primals: [means, quats, scales], cotangents: [vNormals])
    close(gk.dMeans, gn[0], "normals d/dmeans", tol: 1e-3)
    close(gk.dQuats, gn[1], "normals d/dquats", tol: 1e-3)
    close(gk.dScales, gn[2], "normals d/dscales", tol: 1e-3)

    let vFrames = MLXArray((0 ..< n * 12).map { _ in next() }, [n, 12])
    let fk = GaussianSurfaceDepth.frameBackward(
        means: means, quats: quats, scales: scales, camera: camera, vFrames: vFrames)
    let (_, gfr) = vjp(referenceFrames(campos: campos), primals: [means, quats, scales], cotangents: [vFrames])
    close(fk.dMeans, gfr[0], "frame d/dmeans", tol: 1e-3)
    close(fk.dQuats, gfr[1], "frame d/dquats", tol: 1e-3)
    close(fk.dScales, gfr[2], "frame d/dscales", tol: 1e-3)
}

/// With a stride, the depth and its gradient are those of the full-resolution pass at the sampled
/// pixels (i s + s/2, j s + s/2).
@Test
func surfaceDepthStrideSamplesTheFullResolutionPass() throws {
    let model = GaussianModel(
        means: MLXArray([Float(0.02), -0.01, 1.6, -0.1, 0.03, 1.75, 0.12, -0.02, 1.9], [3, 3]),
        quats: MLXArray([Float(0.9), 0.2, 0.1, -0.3, 0.8, -0.1, 0.4, 0.2, 1, 0.05, -0.2, 0.1], [3, 4]),
        logScales: MLXArray([log(Float(0.12)), log(0.08), log(0.03), log(0.09), log(0.1), log(0.045),
                             log(0.15), log(0.06), log(0.075)], [3, 3]),
        logitOpacities: MLXArray([logit(Float(0.9)), logit(0.85), logit(0.9)], [3]),
        sh0: MLXArray.zeros([3, 3]), shN: MLXArray.zeros([3, GaussianModel.shRestBands, 3]))
    let w = 48, h = 48, s = 3
    let camera = Camera(
        viewmat: MLXArray([Float(1), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [16]),
        intrinsics: MLXArray([Float(150), 150, 24, 24], [4]),
        width: w, height: h, background: MLXArray.zeros([3]))
    let renderer = try GaussianRenderer()
    let (_, _, cache) = renderer.forward(model: model, camera: camera, shDegree: 0)
    let full = GaussianSurfaceDepth.render(model: model, camera: camera, cache: cache)!
    let strided = GaussianSurfaceDepth.render(
        model: model, camera: camera, cache: cache, parameters: .init(stride: s, cooperative: false))!
    #expect(strided.depth.shape == [h / s, w / s])
    let fd = full.depth.asArray(Float.self), sd = strided.depth.asArray(Float.self)
    var weightsStrided = [Float](repeating: 0, count: (h / s) * (w / s))
    var weightsFull = [Float](repeating: 0, count: h * w)
    for i in 0 ..< h / s {
        for j in 0 ..< w / s {
            let p = (i * s + s / 2) * w + j * s + s / 2
            #expect(abs(sd[i * (w / s) + j] - fd[p]) < 1e-6)
            if sd[i * (w / s) + j] > 0 {
                weightsStrided[i * (w / s) + j] = Float(i + 2 * j % 5) / 7
                weightsFull[p] = weightsStrided[i * (w / s) + j]
            }
        }
    }
    let gs = GaussianSurfaceDepth.backward(
        strided, model: model, camera: camera, cache: cache, vDepth: MLXArray(weightsStrided, [h / s, w / s]))
    let gf = GaussianSurfaceDepth.backward(
        full, model: model, camera: camera, cache: cache, vDepth: MLXArray(weightsFull, [h, w]))
    for (a, b) in [(gs.dMeans, gf.dMeans), (gs.dQuats, gf.dQuats), (gs.dLogScales, gf.dLogScales),
                   (gs.dLogitOpacities, gf.dLogitOpacities)] {
        let x = a.asArray(Float.self), y = b.asArray(Float.self)
        #expect(x.indices.allSatisfy { abs(x[$0] - y[$0]) < 1e-4 * max(1, abs(y[$0])) })
    }
    #expect(sd.filter { $0 > 0 }.count > 20)
}

/// Three search passes (as in training): the depth interpolated in log T over the final interval and
/// the gradient from the slope measured there stay close to a fully converged search. (Two passes
/// are off by millimeters where the first pass falls back to a wide interval.)
@Test(arguments: [9])
func surfaceDepthFewPassesMatchesConvergedSearch(iterations: Int) throws {
    let model = GaussianModel(
        means: MLXArray([Float(0.02), -0.01, 1.6, -0.1, 0.03, 1.75, 0.12, -0.02, 1.9], [3, 3]),
        quats: MLXArray([Float(0.9), 0.2, 0.1, -0.3, 0.8, -0.1, 0.4, 0.2, 1, 0.05, -0.2, 0.1], [3, 4]),
        logScales: MLXArray([log(Float(0.12)), log(0.08), log(0.03), log(0.09), log(0.1), log(0.045),
                             log(0.15), log(0.06), log(0.075)], [3, 3]),
        logitOpacities: MLXArray([logit(Float(0.9)), logit(0.85), logit(0.9)], [3]),
        sh0: MLXArray.zeros([3, 3]), shN: MLXArray.zeros([3, GaussianModel.shRestBands, 3]))
    let w = 48, h = 48
    let camera = Camera(
        viewmat: MLXArray([Float(1), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [16]),
        intrinsics: MLXArray([Float(150), 150, 24, 24], [4]),
        width: w, height: h, background: MLXArray.zeros([3]))
    let renderer = try GaussianRenderer()
    let (_, _, cache) = renderer.forward(model: model, camera: camera, shDegree: 0)
    let exact = GaussianSurfaceDepth.render(
        model: model, camera: camera, cache: cache, parameters: .init(bisectionIterations: 36))!
    let fast = GaussianSurfaceDepth.render(
        model: model, camera: camera, cache: cache, parameters: .init(bisectionIterations: iterations))!
    let de = exact.depth.asArray(Float.self), df = fast.depth.asArray(Float.self)
    var worst: Float = 0
    for i in de.indices where de[i] > 0 {
        #expect(df[i] > 0)
        worst = max(worst, abs(df[i] - de[i]))
    }
    #expect(worst < 1e-3, "iterations \(iterations): depth off by up to \(worst) m")
    // Gradients of a positive weighting of the depth.
    let weights = MLXArray(de.indices.map { de[$0] > 0 ? Float(1 + $0 % 3) : 0 }, [h, w])
    let ge = GaussianSurfaceDepth.backward(exact, model: model, camera: camera, cache: cache, vDepth: weights)
    let gf = GaussianSurfaceDepth.backward(fast, model: model, camera: camera, cache: cache, vDepth: weights)
    for (a, b) in [(gf.dMeans, ge.dMeans), (gf.dQuats, ge.dQuats), (gf.dLogScales, ge.dLogScales),
                   (gf.dLogitOpacities, ge.dLogitOpacities)] {
        let x = a.asArray(Float.self), y = b.asArray(Float.self)
        let scale = y.map { abs($0) }.max() ?? 1
        let err = x.indices.map { abs(x[$0] - y[$0]) }.max() ?? 0
        #expect(err < 0.05 * scale, "iterations \(iterations): gradient off by \(err) of \(scale)")
    }
}

/// The cooperative kernels (one SIMD group per pixel) give the per-thread kernels' depth and
/// gradient: exactly with a converged search; with a training-like one up to the per-thread
/// kernel's early stops (where it takes the interval midpoint instead of the log T interpolation).
@Test(arguments: [9, 36])
func surfaceDepthCooperativeMatchesPerThread(iterations: Int) throws {
    let model = GaussianModel(
        means: MLXArray([Float(0.02), -0.01, 1.6, -0.1, 0.03, 1.75, 0.12, -0.02, 1.9], [3, 3]),
        quats: MLXArray([Float(0.9), 0.2, 0.1, -0.3, 0.8, -0.1, 0.4, 0.2, 1, 0.05, -0.2, 0.1], [3, 4]),
        logScales: MLXArray([log(Float(0.12)), log(0.08), log(0.03), log(0.09), log(0.1), log(0.045),
                             log(0.15), log(0.06), log(0.075)], [3, 3]),
        logitOpacities: MLXArray([logit(Float(0.9)), logit(0.85), logit(0.9)], [3]),
        sh0: MLXArray.zeros([3, 3]), shN: MLXArray.zeros([3, GaussianModel.shRestBands, 3]))
    let w = 48, h = 48, s = 2
    let camera = Camera(
        viewmat: MLXArray([Float(1), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [16]),
        intrinsics: MLXArray([Float(150), 150, 24, 24], [4]),
        width: w, height: h, background: MLXArray.zeros([3]))
    let renderer = try GaussianRenderer()
    let (_, _, cache) = renderer.forward(model: model, camera: camera, shDegree: 0)
    func render(_ coop: Bool) -> GaussianSurfaceDepth.Result {
        GaussianSurfaceDepth.render(model: model, camera: camera, cache: cache, parameters: .init(
            bisectionIterations: iterations, stride: s, cooperative: coop))!
    }
    let a = render(false), b = render(true)
    let da = a.depth.asArray(Float.self), db = b.depth.asArray(Float.self)
    #expect(da.filter { $0 > 0 }.count > 50)
    let depthTolerance: Float = iterations >= 36 ? 1e-5 : 1e-3
    let gradientTolerance: Float = iterations >= 36 ? 1e-3 : 0.05
    #expect(da.indices.allSatisfy { (da[$0] > 0) == (db[$0] > 0) && abs(da[$0] - db[$0]) < depthTolerance })
    let weights = MLXArray(da.indices.map { da[$0] > 0 ? Float(1 + $0 % 3) : 0 }, [h / s, w / s])
    let ga = GaussianSurfaceDepth.backward(a, model: model, camera: camera, cache: cache, vDepth: weights)
    let gb = GaussianSurfaceDepth.backward(b, model: model, camera: camera, cache: cache, vDepth: weights)
    for (x, y) in [(gb.dMeans, ga.dMeans), (gb.dQuats, ga.dQuats), (gb.dLogScales, ga.dLogScales),
                   (gb.dLogitOpacities, ga.dLogitOpacities)] {
        let p = x.asArray(Float.self), q = y.asArray(Float.self)
        let scale = q.map { abs($0) }.max() ?? 1
        #expect(p.indices.allSatisfy { abs(p[$0] - q[$0]) < gradientTolerance * scale }, "iterations \(iterations)")
    }
}
