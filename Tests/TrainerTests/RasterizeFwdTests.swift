// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXRandom
import Testing
@testable import Trainer

private struct RasterScene {
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

private func slangRasterizeFwd(
    scene: RasterScene, bin: TileBinningResult
) throws -> (renderColor: MLXArray, renderAlpha: MLXArray, lastIds: MLXArray) {
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

@Test
func rasterizeFwdMatchesReference() throws {
    // 32x32 (2x2 tiles). Overlap 3 Gaussians at different depths so front-to-back compositing
    // matters.
    let scene = RasterScene(
        imageWidth: 32, imageHeight: 32, tileSize: 16,
        means2d: MLXArray(
            [Float(8.0),  Float(8.0),
             Float(24.0), Float(24.0),
             Float(16.0), Float(16.0)],
            [3, 2]
        ),
        conics: MLXArray(
            [Float(0.01),  Float(0), Float(0.01),
             Float(0.01),  Float(0), Float(0.01),
             Float(0.005), Float(0), Float(0.005)],
            [3, 3]
        ),
        colors: MLXArray(
            [Float(1.0), Float(0.0), Float(0.0),
             Float(0.0), Float(1.0), Float(0.0),
             Float(0.0), Float(0.0), Float(1.0)],
            [3, 3]
        ),
        opacities: MLXArray([Float(0.8), Float(0.8), Float(0.5)], [3]),
        radii: MLXArray(
            [Int32(15), Int32(15), Int32(15), Int32(15), Int32(20), Int32(20)], [3, 2]
        ),
        depths: MLXArray([Float(2.0), Float(3.0), Float(1.0)], [3]),
        background: MLXArray([Float(0.0), Float(0.0), Float(0.0)], [3])
    )

    let bin = TileBinning.bin(
        means2d: scene.means2d, radii: scene.radii, depths: scene.depths,
        imageWidth: scene.imageWidth, imageHeight: scene.imageHeight, tileSize: scene.tileSize
    )

    let (slangColor, slangAlpha, slangLastIds) = try slangRasterizeFwd(
        scene: scene, bin: bin
    )

    let ref = referenceRasterize(
        means2d: scene.means2d.asArray(Float.self),
        conics: scene.conics.asArray(Float.self),
        colors: scene.colors.asArray(Float.self),
        opacities: scene.opacities.asArray(Float.self),
        flattenIds: bin.flattenIds.asArray(Int32.self),
        tileOffsets: bin.tileOffsets.asArray(Int32.self),
        imageWidth: scene.imageWidth, imageHeight: scene.imageHeight,
        tileSize: scene.tileSize,
        background: scene.background.asArray(Float.self)
    )

    let slangColorHost = slangColor.asArray(Float.self)
    let slangAlphaHost = slangAlpha.asArray(Float.self)
    let slangLastIdsHost = slangLastIds.asArray(Int32.self)

    var maxColorDiff: Float = 0
    for i in 0..<slangColorHost.count {
        maxColorDiff = max(maxColorDiff, abs(slangColorHost[i] - ref.renderColor[i]))
    }
    var maxAlphaDiff: Float = 0
    for i in 0..<slangAlphaHost.count {
        maxAlphaDiff = max(maxAlphaDiff, abs(slangAlphaHost[i] - ref.renderAlpha[i]))
    }
    var lastIdsMismatch = 0
    for i in 0..<slangLastIdsHost.count {
        if slangLastIdsHost[i] != ref.lastIds[i] { lastIdsMismatch += 1 }
    }

    #expect(maxColorDiff < 1e-5, "render_color max diff \(maxColorDiff)")
    #expect(maxAlphaDiff < 1e-5, "render_alpha max diff \(maxAlphaDiff)")
    #expect(lastIdsMismatch == 0, "\(lastIdsMismatch) last_ids mismatches")

    // Non-background pixels are actually painted (sanity check of the input).
    let coloured = slangColorHost.contains { $0 > 0.01 }
    #expect(coloured, "image is entirely background; test inputs may be wrong")
}

@Test
func rasterizeFwdHandlesNoIntersectionsTile() throws {
    // The only Gaussian is behind the camera and dropped by binning, so the output is background
    // color only.
    let scene = RasterScene(
        imageWidth: 16, imageHeight: 16, tileSize: 16,
        means2d: MLXArray([Float(8.0), Float(8.0)], [1, 2]),
        conics: MLXArray([Float(0.01), Float(0), Float(0.01)], [1, 3]),
        colors: MLXArray([Float(1.0), Float(0.0), Float(0.0)], [1, 3]),
        opacities: MLXArray([Float(0.8)], [1]),
        radii: MLXArray([Int32(5), Int32(5)], [1, 2]),
        depths: MLXArray([Float(-1.0)], [1]),       // behind the camera
        background: MLXArray([Float(0.5), Float(0.5), Float(0.5)], [3])
    )

    let bin = TileBinning.bin(
        means2d: scene.means2d, radii: scene.radii, depths: scene.depths,
        imageWidth: scene.imageWidth, imageHeight: scene.imageHeight, tileSize: scene.tileSize
    )
    #expect(bin.nIsects == 0)

    let (slangColor, slangAlpha, _) = try slangRasterizeFwd(scene: scene, bin: bin)
    let slangColorHost = slangColor.asArray(Float.self)
    let slangAlphaHost = slangAlpha.asArray(Float.self)

    let totalPixels = scene.imageWidth * scene.imageHeight
    for i in 0..<totalPixels {
        #expect(abs(slangColorHost[i * 3 + 0] - 0.5) < 1e-6)
        #expect(abs(slangColorHost[i * 3 + 1] - 0.5) < 1e-6)
        #expect(abs(slangColorHost[i * 3 + 2] - 0.5) < 1e-6)
        #expect(slangAlphaHost[i] == 0, "alpha should be 0 (T=1, no contribution)")
    }
}
