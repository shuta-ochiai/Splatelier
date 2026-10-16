// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation

/// Per-pixel CPU reference for the forward rasterizer. A literal transcription of
/// slang/rasterize.slang for numeric match checks.
struct CPURasterizeResult {
    let renderColor: [Float]   // [H * W * 3]
    let renderAlpha: [Float]   // [H * W]
    let lastIds: [Int32]       // [H * W]
}

func referenceRasterize(
    means2d: [Float],
    conics: [Float],
    colors: [Float],
    opacities: [Float],
    flattenIds: [Int32],
    tileOffsets: [Int32],
    imageWidth: Int,
    imageHeight: Int,
    tileSize: Int,
    background: [Float]
) -> CPURasterizeResult {
    let numTilesX = (imageWidth + tileSize - 1) / tileSize
    let totalPixels = imageWidth * imageHeight

    var renderColor = [Float](repeating: 0, count: totalPixels * 3)
    var renderAlpha = [Float](repeating: 0, count: totalPixels)
    var lastIds = [Int32](repeating: 0, count: totalPixels)

    for pyInt in 0..<imageHeight {
        for pxInt in 0..<imageWidth {
            let py = Float(pyInt) + 0.5
            let px = Float(pxInt) + 0.5
            let tileX = pxInt / tileSize
            let tileY = pyInt / tileSize
            let tileId = tileY * numTilesX + tileX
            let startIdx = Int(tileOffsets[tileId])
            let endIdx = Int(tileOffsets[tileId + 1])

            var T: Float = 1.0
            var r: Float = 0
            var g: Float = 0
            var b: Float = 0
            var lastLocalIdx: Int32 = -1
            var done = false

            for k in startIdx..<endIdx {
                if done { break }
                let gi = Int(flattenIds[k])
                let mx = means2d[gi * 2 + 0]
                let my = means2d[gi * 2 + 1]
                let ca = conics[gi * 3 + 0]
                let cb = conics[gi * 3 + 1]
                let cc = conics[gi * 3 + 2]

                let dx = px - mx
                let dy = py - my
                let sigma = 0.5 * (ca * dx * dx + cc * dy * dy) + cb * dx * dy
                if sigma < 0 { continue }

                let opac = opacities[gi]
                let alpha = min(Float(0.99), opac * Foundation.exp(-sigma))
                if alpha < Float(1.0 / 255.0) { continue }

                let nextT = T * (1 - alpha)
                if nextT < Float(1e-4) {
                    done = true
                    break
                }

                let weight = T * alpha
                r += weight * colors[gi * 3 + 0]
                g += weight * colors[gi * 3 + 1]
                b += weight * colors[gi * 3 + 2]
                lastLocalIdx = Int32(k - startIdx)
                T = nextT
            }

            let pidx = pyInt * imageWidth + pxInt
            renderColor[pidx * 3 + 0] = r + T * background[0]
            renderColor[pidx * 3 + 1] = g + T * background[1]
            renderColor[pidx * 3 + 2] = b + T * background[2]
            renderAlpha[pidx] = 1 - T
            lastIds[pidx] = lastLocalIdx
        }
    }

    return CPURasterizeResult(
        renderColor: renderColor, renderAlpha: renderAlpha, lastIds: lastIds
    )
}
