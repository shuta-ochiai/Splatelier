// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// PocketGS Operator T (Sec. III-C) forward replay cache + backward.
//   - FD: analytic gradient vs central differences (catches shared algebraic errors
//     cross-implementation comparison misses)
//   - Cross-implementation match: T-reconstruction backward and replay backward give the same
//     gradients (within 1e-4)
//   - Index order: forward writes cache_id in FTF order (front → back)
//   - overflow: stacking 6 into K_max=4 makes a forced early-break observable

import Foundation
import MLX
import MLXRandom
import Testing
@testable import Trainer

private struct ReplayScene {
    let imageWidth: Int
    let imageHeight: Int
    let tileSize: Int
    var numTilesX: Int { (imageWidth + tileSize - 1) / tileSize }
    let means2d: MLXArray         // [N, 2]
    let conics: MLXArray          // [N, 3]
    let colors: MLXArray          // [N, 3]
    let opacities: MLXArray       // [N]
    let radii: MLXArray           // [N, 2] int32
    let depths: MLXArray          // [N]
    let background: MLXArray      // [3]
}

private struct ReplayFwdOutput {
    let renderColor: MLXArray   // [H*W*3]
    let renderAlpha: MLXArray   // [H*W]
    let cacheId: MLXArray       // [K_max*H*W] (the only cache bwd needs)
    let cacheCount: MLXArray    // [H*W]
    let overflowCount: Int32
}

private func slangFwdReplay(
    scene: ReplayScene, bin: TileBinningResult, kMax: Int
) throws -> ReplayFwdOutput {
    let kernel = try SlangCompiledKernel.load(name: "rasterize_fwd_replay")
    let dims = MLXArray(
        [Int32(scene.imageWidth), Int32(scene.imageHeight),
         Int32(scene.tileSize), Int32(scene.numTilesX), Int32(kMax)],
        [5]
    )
    let totalPixels = scene.imageWidth * scene.imageHeight
    let cacheSlots = kMax * totalPixels
    // The overflow flag is packed into bit 30 of cache_count (the kernel writes every buffer, so no
    // initValue needed).
    let outputs = kernel(
        [
            scene.means2d, scene.conics, scene.colors, scene.opacities,
            bin.flattenIds, bin.tileOffsets, dims, scene.background,
        ],
        grid: (scene.imageWidth, scene.imageHeight, 1),
        threadGroup: (scene.tileSize, scene.tileSize, 1),
        outputShapes: [
            [totalPixels * 3], [totalPixels],
            [cacheSlots], [totalPixels],
        ],
        outputDTypes: [
            .float32, .float32, .int32, .int32,
        ]
    )
    eval(outputs[0], outputs[1], outputs[2], outputs[3])
    // overflowCount is the number of pixels with bit 30 set; count is masked to bits 0-29 (= K) on
    // return.
    let rawCounts = outputs[3].asArray(Int32.self)
    let overflowBit: Int32 = 0x40000000
    let countMask: Int32 = 0x3FFFFFFF
    let overflow = Int32(rawCounts.reduce(0) { $0 + (($1 & overflowBit) != 0 ? 1 : 0) })
    let maskedCounts = rawCounts.map { $0 & countMask }
    let maskedCacheCount = MLXArray(maskedCounts, [totalPixels])
    return ReplayFwdOutput(
        renderColor: outputs[0], renderAlpha: outputs[1],
        cacheId: outputs[2], cacheCount: maskedCacheCount,
        overflowCount: overflow
    )
}

private func slangBwdReplay(
    scene: ReplayScene, kMax: Int,
    fwd: ReplayFwdOutput,
    vRenderColor: MLXArray, vRenderAlpha: MLXArray
) throws -> (vMeans2d: MLXArray, vConics: MLXArray, vColors: MLXArray, vOpacities: MLXArray) {
    let kernel = try SlangCompiledKernel.load(name: "rasterize_bwd_replay")
    let n = scene.means2d.shape[0]
    let dims = MLXArray(
        [Int32(scene.imageWidth), Int32(scene.imageHeight),
         Int32(scene.tileSize), Int32(scene.numTilesX), Int32(kMax)],
        [5]
    )
    // replay backward does not reference the background (absorbed into R_{k+1})
    let outputs = kernel(
        [
            scene.means2d, scene.conics, scene.colors, scene.opacities,
            dims,
            fwd.renderColor, fwd.renderAlpha,
            fwd.cacheId, fwd.cacheCount,
            vRenderColor, vRenderAlpha,
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

/// Scalar loss via `referenceRasterize` (for FD).
private func replaySceneLoss(
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

// MARK: - Test 1: FD vs replay backward

/// The replay backward's analytic gradient matches central differences (also catches bugs shared by
/// both implementations).
@Test
func rasterizeBwdReplayMatchesFiniteDifferences() throws {
    // Same two-overlap scene and tolerance as RasterizeBwdTests.
    let scene = ReplayScene(
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
    let W = scene.imageWidth, H = scene.imageHeight
    let totalPixels = W * H
    let kMax = 8

    let bin = TileBinning.bin(
        means2d: scene.means2d, radii: scene.radii, depths: scene.depths,
        imageWidth: W, imageHeight: H, tileSize: scene.tileSize
    )

    let fwd = try slangFwdReplay(scene: scene, bin: bin, kMax: kMax)
    #expect(fwd.overflowCount == 0, "K_max=8 should be ample for 2 gaussians")

    MLXRandom.seed(42)
    let cotColor = MLXRandom.normal([totalPixels * 3]) * 0.05
    let cotAlpha = MLXRandom.normal([totalPixels]) * 0.05
    eval(cotColor, cotAlpha)

    let bwd = try slangBwdReplay(
        scene: scene, kMax: kMax, fwd: fwd,
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
        replaySceneLoss(
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
            "\(label) FD off: maxAbs=\(maxAbs) maxRel=\(maxRel) at idx=\(worstIdx)"
        )
    }

    var pc = colorsHost
    check(label: "v_colors (replay)", primal: colorsHost, grad: analyticalColors, eps: 1e-3,
          perturb: { idx, delta in
              pc[idx] = colorsHost[idx] + delta
              defer { pc[idx] = colorsHost[idx] }
              return loss(colors: pc)
          })

    var po = opacitiesHost
    check(label: "v_opacities (replay)", primal: opacitiesHost, grad: analyticalOpacities, eps: 1e-3,
          perturb: { idx, delta in
              po[idx] = opacitiesHost[idx] + delta
              defer { po[idx] = opacitiesHost[idx] }
              return loss(opacities: po)
          })

    var pn = conicsHost
    check(label: "v_conics (replay)", primal: conicsHost, grad: analyticalConics, eps: 1e-4,
          perturb: { idx, delta in
              pn[idx] = conicsHost[idx] + delta
              defer { pn[idx] = conicsHost[idx] }
              return loss(conics: pn)
          })

    var pm = meansHost
    check(label: "v_means2d (replay)", primal: meansHost, grad: analyticalMeans, eps: 1e-3,
          perturb: { idx, delta in
              pm[idx] = meansHost[idx] + delta
              defer { pm[idx] = meansHost[idx] }
              return loss(means: pm)
          })
}

// MARK: - Test 2: cross-implementation match (replay vs existing)

/// On a multi-fragment scene, the existing backward and replay backward return the same gradients
/// (algebraically equivalent, so differences are rounding only; regression guard).
@Test
func rasterizeBwdReplayMatchesExistingBwd() throws {
    // To exercise the multi-fragment path, overlap 4 slightly so alphas near the center are
    // similar.
    let scene = ReplayScene(
        imageWidth: 16, imageHeight: 16, tileSize: 16,
        means2d: MLXArray(
            [Float(7.0), Float(7.0),
             Float(8.0), Float(8.0),
             Float(9.0), Float(8.0),
             Float(8.0), Float(9.0)],
            [4, 2]
        ),
        conics: MLXArray(
            [Float(0.05), Float(0.0), Float(0.05),
             Float(0.04), Float(0.0), Float(0.04),
             Float(0.05), Float(0.01), Float(0.04),
             Float(0.04), Float(0.0), Float(0.05)],
            [4, 3]
        ),
        colors: MLXArray(
            [Float(0.9), Float(0.1), Float(0.1),
             Float(0.1), Float(0.9), Float(0.1),
             Float(0.1), Float(0.1), Float(0.9),
             Float(0.5), Float(0.5), Float(0.2)],
            [4, 3]
        ),
        opacities: MLXArray([Float(0.6), Float(0.5), Float(0.5), Float(0.4)], [4]),
        radii: MLXArray(
            [Int32(8), Int32(8), Int32(8), Int32(8),
             Int32(8), Int32(8), Int32(8), Int32(8)], [4, 2]
        ),
        depths: MLXArray([Float(1.5), Float(2.0), Float(3.0), Float(4.0)], [4]),
        background: MLXArray([Float(0.1), Float(0.1), Float(0.1)], [3])
    )
    let W = scene.imageWidth, H = scene.imageHeight
    let totalPixels = W * H
    let n = 4
    let kMax = 16

    let bin = TileBinning.bin(
        means2d: scene.means2d, radii: scene.radii, depths: scene.depths,
        imageWidth: W, imageHeight: H, tileSize: scene.tileSize
    )

    // Existing path: forward + backward
    let existingFwdKernel = try SlangCompiledKernel.load(name: "rasterize_fwd")
    let dimsExisting = MLXArray(
        [Int32(W), Int32(H), Int32(scene.tileSize), Int32(scene.numTilesX)], [4]
    )
    let existingFwdOut = existingFwdKernel(
        [
            scene.means2d, scene.conics, scene.colors, scene.opacities,
            bin.flattenIds, bin.tileOffsets,
            dimsExisting, scene.background,
        ],
        grid: (W, H, 1),
        threadGroup: (scene.tileSize, scene.tileSize, 1),
        outputShapes: [[totalPixels * 3], [totalPixels], [totalPixels]],
        outputDTypes: [.float32, .float32, .int32]
    )
    let existingRenderColor = existingFwdOut[0]
    let existingRenderAlpha = existingFwdOut[1]
    let existingLastIds = existingFwdOut[2]
    eval(existingRenderColor, existingRenderAlpha, existingLastIds)

    // Replay path: forward
    let replayFwd = try slangFwdReplay(scene: scene, bin: bin, kMax: kMax)
    #expect(replayFwd.overflowCount == 0)

    // Both forwards' renders match (cache writes do not change accum, so they should match)
    let rcExisting = existingRenderColor.asArray(Float.self)
    let rcReplay = replayFwd.renderColor.asArray(Float.self)
    var maxRcDiff: Float = 0
    for i in 0..<rcExisting.count { maxRcDiff = max(maxRcDiff, abs(rcExisting[i] - rcReplay[i])) }
    #expect(maxRcDiff < 1e-5, "forward render_color diverged: \(maxRcDiff)")

    // Shared upstream cotangent
    MLXRandom.seed(123)
    let cotColor = MLXRandom.normal([totalPixels * 3]) * 0.05
    let cotAlpha = MLXRandom.normal([totalPixels]) * 0.05
    eval(cotColor, cotAlpha)

    // Existing backward
    let existingBwdKernel = try SlangCompiledKernel.load(name: "rasterize_bwd")
    let existingBwdOut = existingBwdKernel(
        [
            scene.means2d, scene.conics, scene.colors, scene.opacities,
            bin.flattenIds, bin.tileOffsets, dimsExisting, scene.background,
            existingRenderAlpha, existingLastIds, cotColor, cotAlpha,
        ],
        grid: (W, H, 1),
        threadGroup: (scene.tileSize, scene.tileSize, 1),
        outputShapes: [[n * 2], [n * 3], [n * 3], [n]],
        outputDTypes: [.float32, .float32, .float32, .float32],
        initValue: 0.0
    )
    eval(existingBwdOut[0], existingBwdOut[1], existingBwdOut[2], existingBwdOut[3])

    let replayBwd = try slangBwdReplay(
        scene: scene, kMax: kMax, fwd: replayFwd,
        vRenderColor: cotColor, vRenderAlpha: cotAlpha
    )

    let pairs: [(String, MLXArray, MLXArray)] = [
        ("v_means2d", existingBwdOut[0], replayBwd.vMeans2d.reshaped([n * 2])),
        ("v_conics",  existingBwdOut[1], replayBwd.vConics.reshaped([n * 3])),
        ("v_colors",  existingBwdOut[2], replayBwd.vColors.reshaped([n * 3])),
        ("v_opacities", existingBwdOut[3], replayBwd.vOpacities),
    ]
    for (name, expected, actual) in pairs {
        let e = expected.asArray(Float.self)
        let a = actual.asArray(Float.self)
        #expect(e.count == a.count, "\(name) shape mismatch")
        var maxAbs: Float = 0, maxRel: Float = 0
        for i in 0..<e.count {
            let d = abs(e[i] - a[i])
            maxAbs = max(maxAbs, d)
            let scale = max(abs(e[i]), abs(a[i]), 1e-4)
            maxRel = max(maxRel, d / scale)
        }
        #expect(maxAbs < 1e-4 || maxRel < 1e-3,
            "\(name) replay vs existing diverged: maxAbs=\(maxAbs) maxRel=\(maxRel)")
    }
}

// MARK: - Test 3: cache_id index-order trace

/// The order forward writes to cache_id is FTF (nearest first) at every pixel, consistent with
/// `flatten_ids` order (an invariant that catches reverse-walk off-by-one errors early).
@Test
func rasterizeFwdReplayPreservesFTFOrder() throws {
    // Overlap 3 at the center, clearly separated by depth.
    let scene = ReplayScene(
        imageWidth: 16, imageHeight: 16, tileSize: 16,
        means2d: MLXArray(
            [Float(7.5), Float(7.5),
             Float(8.5), Float(7.5),
             Float(8.0), Float(8.5)],
            [3, 2]
        ),
        conics: MLXArray(
            [Float(0.08), Float(0.0), Float(0.08),
             Float(0.08), Float(0.0), Float(0.08),
             Float(0.08), Float(0.0), Float(0.08)],
            [3, 3]
        ),
        colors: MLXArray(
            [Float(1.0), Float(0.0), Float(0.0),
             Float(0.0), Float(1.0), Float(0.0),
             Float(0.0), Float(0.0), Float(1.0)],
            [3, 3]
        ),
        opacities: MLXArray([Float(0.5), Float(0.5), Float(0.5)], [3]),
        radii: MLXArray([Int32(6), Int32(6), Int32(6), Int32(6), Int32(6), Int32(6)], [3, 2]),
        depths: MLXArray([Float(1.5), Float(2.5), Float(3.5)], [3]),
        background: MLXArray([Float(0.0), Float(0.0), Float(0.0)], [3])
    )
    let bin = TileBinning.bin(
        means2d: scene.means2d, radii: scene.radii, depths: scene.depths,
        imageWidth: scene.imageWidth, imageHeight: scene.imageHeight,
        tileSize: scene.tileSize
    )
    let kMax = 8
    let fwd = try slangFwdReplay(scene: scene, bin: bin, kMax: kMax)

    let cacheCount = fwd.cacheCount.asArray(Int32.self)   // [H*W]
    let cacheId = fwd.cacheId.asArray(Int32.self)         // [K_max*H*W]
    let H = scene.imageHeight
    let W = scene.imageWidth
    let totalPixels = H * W

    // Pick the center pixel (8, 8), which all Gaussians cover
    let pixIdx = 8 * W + 8
    let centerCount = Int(cacheCount[pixIdx])
    #expect(centerCount >= 1, "center pixel should have at least one cached fragment")

    // cache_id is slot-major (c * H*W + pix)
    var centerIds: [Int32] = []
    for c in 0..<centerCount {
        let slot = c * totalPixels + pixIdx
        centerIds.append(cacheId[slot])
    }

    // Expected order is ascending depth: depth = [1.5, 2.5, 3.5] → [0, 1, 2].
    let depths = scene.depths.asArray(Float.self)
    var expectedOrder = [(Int32(0), depths[0]), (Int32(1), depths[1]), (Int32(2), depths[2])]
    expectedOrder.sort { $0.1 < $1.1 }
    let expectedIds = expectedOrder.prefix(centerCount).map { $0.0 }

    #expect(centerIds == expectedIds,
        "cache_id at center should follow FTF depth order: got \(centerIds), expected \(expectedIds)")
}

// MARK: - Test 4: K_max overflow forced early-break

/// Stacking 6 into a K_max=4 cache triggers a forced early-break (overflow > 0), and cache_count
/// saturates at K_max.
@Test
func rasterizeFwdReplayForcedBreakOnOverflow() throws {
    // Stack 6 at the same position to reliably exceed K_max=4
    var means2d: [Float] = []
    var conics: [Float] = []
    var colors: [Float] = []
    var opacities: [Float] = []
    var radii: [Int32] = []
    var depths: [Float] = []
    for i in 0..<6 {
        means2d += [8.0, 8.0]
        conics += [0.05, 0.0, 0.05]
        colors += [Float(i) / 6.0, 0.5, 0.5]
        opacities += [0.5]   // slow the transmittance decay
        radii += [8, 8]
        depths += [Float(i) + 1.0]
    }
    let scene = ReplayScene(
        imageWidth: 16, imageHeight: 16, tileSize: 16,
        means2d: MLXArray(means2d, [6, 2]),
        conics: MLXArray(conics, [6, 3]),
        colors: MLXArray(colors, [6, 3]),
        opacities: MLXArray(opacities, [6]),
        radii: MLXArray(radii, [6, 2]),
        depths: MLXArray(depths, [6]),
        background: MLXArray([Float(0.0), Float(0.0), Float(0.0)], [3])
    )
    let bin = TileBinning.bin(
        means2d: scene.means2d, radii: scene.radii, depths: scene.depths,
        imageWidth: scene.imageWidth, imageHeight: scene.imageHeight,
        tileSize: scene.tileSize
    )

    let kMax = 4
    let fwd = try slangFwdReplay(scene: scene, bin: bin, kMax: kMax)

    #expect(fwd.overflowCount > 0, "expected overflow_counter > 0; got \(fwd.overflowCount)")

    let cacheCount = fwd.cacheCount.asArray(Int32.self)
    let maxC = cacheCount.max() ?? 0
    #expect(Int(maxC) <= kMax, "cache_count exceeded K_max: \(maxC)")

    let centerC = cacheCount[8 * scene.imageWidth + 8]
    #expect(Int(centerC) == kMax, "center pixel should have cache filled to K_max")
}
