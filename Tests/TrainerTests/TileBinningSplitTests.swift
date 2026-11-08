// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXRandom
import Testing
@testable import Trainer

/// 8-px tiles are binned by splitting the sorted lists of 16-px tiles (`TileBinning.binSplit`).

/// Random projected Gaussians, some elongated, some reaching past the image.
private func splitTestScene(count: Int, width: Int, height: Int, seed: UInt64)
    -> (means2d: MLXArray, radii: MLXArray, depths: MLXArray, conics: MLXArray, opacities: MLXArray)
{
    MLXRandom.seed(seed)
    var means: [Float] = [], conics: [Float] = [], opac: [Float] = [], radii: [Int32] = [], depths: [Float] = []
    let u = MLXRandom.uniform(low: 0, high: 1, [count, 6]).asArray(Float.self)
    for g in 0..<count {
        let r = Array(u[g * 6 ..< g * 6 + 6])
        let s1 = 0.3 + r[0] * 40, s2 = 0.3 + r[1] * s1 * (g % 4 == 0 ? 0.05 : 1)
        let th = r[2] * .pi
        let (cs, sn) = (cos(th), sin(th))
        let a = cs * cs * s1 * s1 + sn * sn * s2 * s2 + 0.3
        let b = cs * sn * (s1 * s1 - s2 * s2)
        let c = sn * sn * s1 * s1 + cs * cs * s2 * s2 + 0.3
        let det = a * c - b * b
        let o = 0.016 + r[3] * 0.97
        let extend = min(Float(3.33), (2 * log(o / (4.0 / 255.0))).squareRoot())
        means += [-20 + r[4] * Float(width + 40), -20 + r[5] * Float(height + 40)]
        conics += [c / det, -b / det, a / det]
        opac.append(o)
        radii += [Int32((extend * a.squareRoot()).rounded(.up)), Int32((extend * c.squareRoot()).rounded(.up))]
        depths.append(0.5 + Float(g % 97) * 0.2)
    }
    return (MLXArray(means, [count, 2]), MLXArray(radii, [count, 2]), MLXArray(depths, [count]),
            MLXArray(conics, [count, 3]), MLXArray(opac, [count]))
}

@Test
func splitBinningKeepsEveryContributingPixelInDepthOrder() {
    let (w, h) = (300, 210)
    let s = splitTestScene(count: 800, width: w, height: h, seed: 5)
    let split = TileBinning.bin(means2d: s.means2d, radii: s.radii, depths: s.depths, conics: s.conics,
                                opacities: s.opacities, imageWidth: w, imageHeight: h, tileSize: 8)
    #expect(split.numTilesX == 2 * ((w + 15) / 16) && split.numTilesY == 2 * ((h + 15) / 16))
    let ids = split.flattenIds.asArray(Int32.self)
    let offs = split.tileOffsets.asArray(Int32.self)
    let depths = s.depths.asArray(Float.self)
    var pairs = Set<Int64>()
    var outOfOrder = 0
    for t in 0..<split.numTiles {
        for k in Int(offs[t]) ..< Int(offs[t + 1]) {
            #expect(pairs.insert(Int64(t) << 32 | Int64(ids[k])).inserted, "duplicate pair")
            if k > Int(offs[t]) && depths[Int(ids[k])] < depths[Int(ids[k - 1])] { outOfOrder += 1 }
        }
    }
    #expect(outOfOrder == 0)

    // Every pixel a Gaussian contributes to (alpha >= 4/255, as rasterize) has its tile listed.
    let means = s.means2d.asArray(Float.self), conics = s.conics.asArray(Float.self)
    let opac = s.opacities.asArray(Float.self), radii = s.radii.asArray(Int32.self)
    var missing = 0
    for g in 0..<opac.count {
        let (mx, my) = (means[2 * g], means[2 * g + 1])
        let (rx, ry) = (Float(radii[2 * g]), Float(radii[2 * g + 1]))
        let x0 = max(0, Int(mx - rx) - 1), x1 = min(w - 1, Int(mx + rx) + 1)
        let y0 = max(0, Int(my - ry) - 1), y1 = min(h - 1, Int(my + ry) + 1)
        guard x0 <= x1, y0 <= y1 else { continue }
        for py in y0...y1 {
            for px in x0...x1 {
                let dx = Float(px) + 0.5 - mx, dy = Float(py) + 0.5 - my
                let sigma = 0.5 * (conics[3 * g] * dx * dx + conics[3 * g + 2] * dy * dy) + conics[3 * g + 1] * dx * dy
                guard sigma >= 0, min(0.99, opac[g] * exp(-sigma)) >= 0.01568627 else { continue }
                let t = (py / 8) * split.numTilesX + px / 8
                if !pairs.contains(Int64(t) << 32 | Int64(g)) { missing += 1 }
            }
        }
    }
    #expect(missing == 0, "\(missing) contributing pixels lost their tile")
}

@Test
func eightPixelTilesRenderAsSixteenPixelTiles() throws {
    // The 8-px lists are the 16-px lists in the same order, less pairs that do not contribute to
    // the quadrant, so the forward is identical and the gradients match up to the order of atomic
    // adds.
    MLXRandom.seed(9)
    let n = 3000
    let xy = MLXRandom.uniform(low: -1.2, high: 1.2, [n, 2])
    let z = MLXRandom.uniform(low: 2.5, high: 6.0, [n, 1])
    let model = GaussianModel(
        means: MLX.concatenated([xy, z], axis: 1), quats: MLXRandom.normal([n, 4]),
        logScales: MLXRandom.uniform(low: -4.5, high: -2, [n, 3]),
        logitOpacities: MLXRandom.uniform(low: -4, high: 4, [n]),
        sh0: MLXRandom.uniform(low: -1, high: 1, [n, 3]),
        shN: MLXRandom.normal([n, GaussianModel.shRestBands, 3]) * 0.1)
    func camera(_ tile: Int) -> Camera {
        Camera(viewmat: MLXArray([Float(1), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [16]),
               intrinsics: MLXArray([Float(150), 150, 100, 75], [4]),
               width: 200, height: 150, tileSize: tile, background: MLXArray([Float(0.1), 0.2, 0.3], [3]))
    }
    let renderer = try GaussianRenderer()
    let ref = renderer.forward(model: model, camera: camera(16), shDegree: 1)
    let out = renderer.forward(model: model, camera: camera(8), shDegree: 1)
    #expect(out.renderColor.asArray(Float.self) == ref.renderColor.asArray(Float.self))
    #expect(out.renderAlpha.asArray(Float.self) == ref.renderAlpha.asArray(Float.self))

    let vColor = MLXRandom.normal([150, 200, 3]) * 1e-3
    let vAlpha = MLXArray.zeros([150, 200])
    let gRef = renderer.backward(model: model, camera: camera(16), cache: ref.cache, vRenderColor: vColor, vRenderAlpha: vAlpha)
    let gOut = renderer.backward(model: model, camera: camera(8), cache: out.cache, vRenderColor: vColor, vRenderAlpha: vAlpha)
    for (a, b) in [(gRef.dMeans, gOut.dMeans), (gRef.dLogScales, gOut.dLogScales),
                   (gRef.dLogitOpacities, gOut.dLogitOpacities), (gRef.dSh0, gOut.dSh0)] {
        let scale = MLX.abs(a).max().item(Float.self)
        #expect(MLX.abs(a - b).max().item(Float.self) <= 1e-4 * scale + 1e-7)
    }
}
