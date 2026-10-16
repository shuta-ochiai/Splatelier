// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXRandom
import Testing
@testable import Trainer

/// Deterministic random numbers (for reproducible tests).
private struct SplitMix64 {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func uniform(_ lo: Float, _ hi: Float) -> Float {
        lo + (hi - lo) * Float(next() >> 40) / Float(1 << 24)
    }
}

/// Builds projected Gaussians directly (like project_to_screen: cov2d + 0.3 dilation, float
/// inverse, radii = opacity-dependent extend × sqrt(cov diagonal)). A needleFraction share is made
/// extremely elongated.
private struct ScreenSplats {
    var means2d: [Float] = []
    var conics: [Float] = []
    var opacities: [Float] = []
    var radii: [Int32] = []
    var depths: [Float] = []
    var count: Int { opacities.count }

    init(count: Int, width: Int, height: Int, maxSigma: Float, needleFraction: Float, seed: UInt64) {
        var rng = SplitMix64(state: seed)
        for g in 0..<count {
            let needle = Float(g) < needleFraction * Float(count)
            let s1 = needle ? rng.uniform(50, maxSigma * 2) : rng.uniform(0.3, maxSigma)
            let s2 = needle ? rng.uniform(0.0, 0.2) : rng.uniform(0.3, s1)
            let th = rng.uniform(0, .pi)
            let (cs, sn) = (cos(th), sin(th))
            let a = cs * cs * s1 * s1 + sn * sn * s2 * s2 + 0.3
            let b = cs * sn * (s1 * s1 - s2 * s2)
            let c = sn * sn * s1 * s1 + cs * cs * s2 * s2 + 0.3
            let det = a * c - b * b
            guard det > 0 else { continue }
            let invDet = 1 / det
            let opac = rng.uniform(0.004, 0.99)
            let extend = min(Float(3.33), (2 * log(opac / (1.0 / 255.0))).squareRoot())
            means2d += [rng.uniform(-30, Float(width) + 30), rng.uniform(-30, Float(height) + 30)]
            conics += [c * invDet, -b * invDet, a * invDet]
            opacities.append(opac)
            radii += [Int32((extend * a.squareRoot()).rounded(.up)), Int32((extend * c.squareRoot()).rounded(.up))]
            depths.append(rng.uniform(0.5, 20))
        }
    }

    func bin(width: Int, height: Int, tileSize: Int, scanline: Bool) -> TileBinningResult {
        let n = count
        return TileBinning.bin(
            means2d: MLXArray(means2d, [n, 2]),
            radii: MLXArray(radii, [n, 2]),
            depths: MLXArray(depths, [n]),
            conics: scanline ? MLXArray(conics, [n, 3]) : nil,
            opacities: scanline ? MLXArray(opacities, [n]) : nil,
            imageWidth: width, imageHeight: height, tileSize: tileSize
        )
    }
}

/// Counts (tile, gaussian) pairs. Duplicates get a value of 2 or more.
private func pairCounts(_ r: TileBinningResult) -> [Int64: Int] {
    let ids = r.flattenIds.asArray(Int32.self)
    let offs = r.tileOffsets.asArray(Int32.self)
    var out: [Int64: Int] = [:]
    for t in 0..<r.numTiles {
        for k in Int(offs[t])..<Int(offs[t + 1]) {
            out[Int64(t) << 32 | Int64(ids[k]), default: 0] += 1
        }
    }
    return out
}

@Test
func scanlineBinningCoversEveryContributingPixel() {
    // Every tile containing a pixel with alpha >= 1/255 (same formula as rasterize) must be in the
    // scanline result. The scanline result is a subset of the AABB, with no duplicates.
    let (w, h, tile) = (200, 152, 16)
    let splats = ScreenSplats(count: 600, width: w, height: h, maxSigma: 60,
                              needleFraction: 0.25, seed: 7)
    let aabb = splats.bin(width: w, height: h, tileSize: tile, scanline: false)
    let scan = splats.bin(width: w, height: h, tileSize: tile, scanline: true)
    let aabbPairs = pairCounts(aabb)
    let scanPairs = pairCounts(scan)

    #expect(scanPairs.values.allSatisfy { $0 == 1 }, "duplicate (tile, gaussian) pairs")
    #expect(scanPairs.keys.allSatisfy { aabbPairs[$0] != nil }, "scanline emitted a tile outside the AABB")
    #expect(scan.nIsects < aabb.nIsects)

    let numTilesX = (w + tile - 1) / tile
    var missing = 0
    for g in 0..<splats.count {
        let (rx, ry) = (Int(splats.radii[2 * g]), Int(splats.radii[2 * g + 1]))
        let (mx, my) = (splats.means2d[2 * g], splats.means2d[2 * g + 1])
        let (ca, cb, cc) = (splats.conics[3 * g], splats.conics[3 * g + 1], splats.conics[3 * g + 2])
        let opac = splats.opacities[g]
        let x0 = max(0, Int((mx - Float(rx)).rounded(.down)) - 1)
        let x1 = min(w - 1, Int((mx + Float(rx)).rounded(.up)) + 1)
        let y0 = max(0, Int((my - Float(ry)).rounded(.down)) - 1)
        let y1 = min(h - 1, Int((my + Float(ry)).rounded(.up)) + 1)
        guard x0 <= x1, y0 <= y1 else { continue }
        for py in y0...y1 {
            for px in x0...x1 {
                let dx = Float(px) + 0.5 - mx
                let dy = Float(py) + 0.5 - my
                let sigma = 0.5 * (ca * dx * dx + cc * dy * dy) + cb * dx * dy
                if sigma < 0 { continue }
                let alpha = min(0.99, opac * exp(-sigma))
                if alpha < 0.00392157 { continue }
                let key = Int64((py / tile) * numTilesX + px / tile) << 32 | Int64(g)
                // Pixels the AABB also misses (outside radii) are existing behavior, so out of
                // scope.
                if aabbPairs[key] != nil && scanPairs[key] == nil { missing += 1 }
            }
        }
    }
    #expect(missing == 0, "\(missing) contributing pixels lost their tile")
}

@Test
func scanlineBinningCutsTilesOfRotatedEllipse() {
    // A thin ellipse tilted 45° does not cover the tiles in the AABB corners.
    let (w, h, tile) = (256, 256, 16)
    var splats = ScreenSplats(count: 0, width: w, height: h, maxSigma: 1, needleFraction: 0, seed: 0)
    let (s1, s2): (Float, Float) = (30, 2)
    let a = 0.5 * (s1 * s1 + s2 * s2) + 0.3
    let b = 0.5 * (s1 * s1 - s2 * s2)
    let c = a
    let det = a * c - b * b
    let extend = Float(3.33)
    splats.means2d = [128, 128]
    splats.conics = [c / det, -b / det, a / det]
    splats.opacities = [0.99]
    splats.radii = [Int32((extend * a.squareRoot()).rounded(.up)), Int32((extend * c.squareRoot()).rounded(.up))]
    splats.depths = [3]
    let aabb = splats.bin(width: w, height: h, tileSize: tile, scanline: false)
    let scan = splats.bin(width: w, height: h, tileSize: tile, scanline: true)
    // Radius ceil(3.33·sqrt(452.3)) = 71 → x, y ∈ [57, 199] → tiles 3...12, 10×10 = 100.
    // The diagonal band is 2-3 tiles per row.
    #expect(aabb.nIsects == 100)
    #expect(scan.nIsects < 40, "scanline isects \(scan.nIsects)")
}

/// A scene of scattered Gaussians with rotation and anisotropy.
private func makeAnisotropicScene(count n: Int, seed: UInt64) -> (GaussianModel, Camera) {
    MLXRandom.seed(seed)
    let xy = MLXRandom.uniform(low: -1.2, high: 1.2, [n, 2])
    let z = MLXRandom.uniform(low: 2.5, high: 6.0, [n, 1])
    let means = MLX.concatenated([xy, z], axis: 1)
    let quats = MLXRandom.normal([n, 4])
    // Mix in shapes long along only one axis (needles / slabs).
    let base = MLXRandom.uniform(low: -4.5, high: -2.5, [n, 3])
    let stretch = MLXRandom.uniform(low: 0, high: 2.5, [n, 1]) * MLXArray([Float(1), 0, 0], [1, 3])
    let logScales = base + stretch
    let logitOpacities = MLXRandom.uniform(low: -4, high: 4, [n])
    let sh0 = MLXRandom.uniform(low: -1, high: 1, [n, 3])
    let shN = MLXRandom.normal([n, GaussianModel.shRestBands, 3]) * 0.1
    let model = GaussianModel(means: means, quats: quats, logScales: logScales,
                              logitOpacities: logitOpacities, sh0: sh0, shN: shN)
    eval(model.means, model.quats, model.logScales, model.logitOpacities, model.sh0, model.shN)
    let viewmat = MLXArray(
        [Float(1), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [16])
    let camera = Camera(
        viewmat: viewmat, intrinsics: MLXArray([Float(150), 150, 96, 72], [4]),
        width: 192, height: 144, tileSize: 16, background: MLXArray([Float(0.1), 0.2, 0.3], [3]))
    return (model, camera)
}

@Test(arguments: [false, true])
func scanlineBinningRendersIdentically(replayCache: Bool) throws {
    // Only non-contributing (tile, Gaussian) pairs are removed, so forward is bit-identical,
    // and backward matches too except for differences in atomic add order.
    let (model, camera) = makeAnisotropicScene(count: 3000, seed: 11)
    let aabbRenderer = try GaussianRenderer()
    aabbRenderer.scanlineTileCulling = false
    aabbRenderer.useReplayCache = replayCache
    let scanRenderer = try GaussianRenderer()
    scanRenderer.scanlineTileCulling = true
    scanRenderer.useReplayCache = replayCache

    let ref = aabbRenderer.forward(model: model, camera: camera, shDegree: 1)
    let out = scanRenderer.forward(model: model, camera: camera, shDegree: 1)
    #expect(out.cache.bin.nIsects < ref.cache.bin.nIsects)
    #expect(out.renderColor.asArray(Float.self) == ref.renderColor.asArray(Float.self))
    #expect(out.renderAlpha.asArray(Float.self) == ref.renderAlpha.asArray(Float.self))

    let pixels = camera.width * camera.height
    MLXRandom.seed(3)
    let vColor = MLXRandom.normal([pixels * 3]) * 0.01
    let vAlpha = MLXRandom.normal([pixels]) * 0.01
    eval(vColor, vAlpha)
    let gRef = aabbRenderer.backward(model: model, camera: camera, cache: ref.cache,
                                     vRenderColor: vColor, vRenderAlpha: vAlpha)
    let gOut = scanRenderer.backward(model: model, camera: camera, cache: out.cache,
                                     vRenderColor: vColor, vRenderAlpha: vAlpha)
    let pairs: [(String, MLXArray, MLXArray)] = [
        ("means", gOut.dMeans, gRef.dMeans), ("quats", gOut.dQuats, gRef.dQuats),
        ("logScales", gOut.dLogScales, gRef.dLogScales),
        ("opacities", gOut.dLogitOpacities, gRef.dLogitOpacities),
        ("sh0", gOut.dSh0, gRef.dSh0), ("shN", gOut.dShN, gRef.dShN),
    ]
    for (name, a, b) in pairs {
        let diff = MLX.abs(a - b).max().item(Float.self)
        let scale = MLX.abs(b).max().item(Float.self)
        #expect(diff <= 1e-4 * max(scale, 1e-6), "\(name): max |diff| \(diff) vs max |grad| \(scale)")
    }
}
