// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// SURFEL_FINAL=1 of `live-mvs-replay` (REPLAY_MODES=surfel): puts the live in-capture MVS next to the
// MVS training uses, as points drawn the same way from the same keyframe. The training MVS is the
// app's (`CaptureTraining.prepareGeometry`: BA, lens undistortion, consistency over all frames);
// its depth maps are turned back into points and added in capture order, so both panels grow
// keyframe by keyframe. It is drawn from that keyframe's BA camera over its undistorted image.

import Foundation
import MLX
import Trainer
import Capture

/// Points of one keyframe: xyz, rgb (0..1) and the size of the pixel that saw each point (m).
struct PointBatch {
    var positions: [Float] = []
    var colors: [Float] = []
    var pixelSizes: [Float] = []
}

/// The training MVS of a capture, by capture index.
struct FinalMVSPoints {
    /// Points from the depth map of each keyframe (keyframes MVS skipped have none).
    let pointsByFrame: [Int: PointBatch]
    /// BA-refined (and undistorted) frame of each capture index (nil if intake dropped it).
    let framesByIndex: [Int: CapturedFrame]
}

func prepareFinalMVS(scene: CapturedScene, log: (String) -> Void) throws -> FinalMVSPoints {
    let t0 = Date()
    let geometry = CaptureTraining.prepareGeometry(scene: scene, log: { _ in })
    let indexByTimestamp = Dictionary(
        scene.frames.enumerated().map { ($1.timestamp, $0) }, uniquingKeysWith: { a, _ in a })
    var framesByIndex: [Int: CapturedFrame] = [:]
    var indexByViewmat: [[Float]: Int] = [:]
    for f in geometry.scene.frames {
        guard let k = indexByTimestamp[f.timestamp] else { continue }
        framesByIndex[k] = f
        indexByViewmat[f.viewmat] = k
    }
    var pointsByFrame: [Int: PointBatch] = [:]
    var total = 0
    for v in try geometry.depthViews() {
        // Depth maps carry the frame's pose, not its index.
        guard let k = indexByViewmat[v.viewmat] else { continue }
        let vm = v.viewmat
        let it = v.intrinsics
        var batch = PointBatch()
        for y in 0..<v.height {
            for x in 0..<v.width {
                let z = v.depth[y * v.width + x]
                guard z > 0 else { continue }
                // Camera → world: Rᵀ(c − t).
                let cx = (Float(x) + 0.5 - it.cx) / it.fx * z - vm[3]
                let cy = (Float(y) + 0.5 - it.cy) / it.fy * z - vm[7]
                let cz = z - vm[11]
                batch.positions += [vm[0] * cx + vm[4] * cy + vm[8] * cz,
                                    vm[1] * cx + vm[5] * cy + vm[9] * cz,
                                    vm[2] * cx + vm[6] * cy + vm[10] * cz]
                let p = (y * v.width + x) * 3
                batch.colors += [Float(v.rgb[p]) / 255, Float(v.rgb[p + 1]) / 255, Float(v.rgb[p + 2]) / 255]
                batch.pixelSizes.append(z / it.fx)
            }
        }
        total += batch.pixelSizes.count
        pointsByFrame[k] = batch
    }
    log(String(format: "[replay final] training MVS: %d depth maps, %d points, BA frames %d/%d (%.1fs)",
               pointsByFrame.count, total, framesByIndex.count, scene.frames.count,
               Date().timeIntervalSince(t0)))
    return FinalMVSPoints(pointsByFrame: pointsByFrame, framesByIndex: framesByIndex)
}

/// Points added keyframe by keyframe, drawn as round Gaussians the size of the pixel that saw them
/// (or all `fixedSigma` (m), which shows where the points are without far views' large splats
/// covering thin structure).
final class PointCloudAccumulator {
    private var positions: MLXArray?
    private var colors: MLXArray?
    private var sigmas: MLXArray?

    var count: Int { positions?.shape[0] ?? 0 }

    func append(_ batch: PointBatch) {
        let n = batch.pixelSizes.count
        guard n > 0 else { return }
        let p = MLXArray(batch.positions, [n, 3])
        let c = MLXArray(batch.colors, [n, 3])
        let s = MLXArray(batch.pixelSizes, [n])
        positions = positions.map { MLX.concatenated([$0, p], axis: 0) } ?? p
        colors = colors.map { MLX.concatenated([$0, c], axis: 0) } ?? c
        sigmas = sigmas.map { MLX.concatenated([$0, s], axis: 0) } ?? s
        eval(positions!, colors!, sigmas!)
    }

    func model(fixedSigma: Float? = nil, opacity: Float = 0.9) -> GaussianModel? {
        guard let positions, let colors, let sigmas else { return nil }
        let n = positions.shape[0]
        let quats = MLX.broadcast(MLXArray([Float(1), 0, 0, 0], [1, 4]), to: [n, 4])
        let logSigma = fixedSigma.map { MLX.full([n], values: MLXArray(Foundation.log($0))) } ?? MLX.log(sigmas)
        let logScales = MLX.broadcast(logSigma.reshaped([n, 1]), to: [n, 3])
        let logits = MLX.full([n], values: MLXArray(Foundation.log(opacity / (1 - opacity))))
        return GaussianModel(
            means: positions, quats: quats, logScales: logScales, logitOpacities: logits,
            sh0: (colors - 0.5) / GaussianModel.shC0,
            shN: MLXArray.zeros([n, GaussianModel.shRestBands, 3]))
    }
}
