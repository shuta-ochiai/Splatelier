// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import Trainer
import simd

// Pre-training preprocessing: capture diagnostics logging, frame exclusion (pose outliers,
// quality), optical flow weights, BA, training view selection.
extension CaptureTraining {

    static func logIntake(scene: CapturedScene, log: (String) -> Void) {
        let frameRes = scene.frames.first.map { "\($0.width)x\($0.height)" } ?? "?"
        log("CapturedScene: \(scene.frames.count) frames @ \(frameRes), \(scene.sparsePoints.count) sparse points")

        // Capture diagnostics log. nil for scenes not from capture (ColmapScene / CLI replay).
        guard let d = scene.diagnostics else { return }
        log(String(format: "[ARKit diag] keptFrames=%d anchorUpdates=%d unsettledAnchors=%d trackingState=%@",
                   d.keptFrameCount, d.anchorUpdateCount, d.unsettledAnchorCount, d.trackingStateDescription))
        if d.lidarSupported {
            log(String(format: "[LiDAR diag] lidarSupported=true lidarFrames=%d lidarPoints=%d",
                       d.lidarFrameCount, d.lidarPointCount))
        } else {
            log("[LiDAR diag] lidarSupported=false")
        }
        if let tags = d.pixelBufferColorTags {
            // Color tags of the capture buffer (for the record). Conversion declares the input as
            // sRGB, so no tone conversion from the transfer tag (ITU_R_709_2 on device) is applied.
            log("[capture color] \(tags)")
        }
        if let mvs = d.liveMVSSummary {
            // Processing stats of the .mvs live point cloud during capture (time per keyframe,
            // dropped frames).
            log("[capture live-mvs] \(mvs)")
        }
        let baStr = scene.baRefinedPoints.map { String($0.count) } ?? "nil"
        log("[capture intake] sparsePoints=\(scene.sparsePoints.count) baRefinedPoints=\(baStr) lidarPoints=\(scene.lidarPoints.count)")
        if d.relocalizationOccurred {
            log("[ARKit warn] reason=relocalization")
        }
        if d.keptFrameCount > 0 {
            let ratio = Float(d.unsettledAnchorCount) / Float(d.keptFrameCount)
            if ratio > 0.2 {
                log(String(format: "[ARKit warn] reason=unsettled_ratio=%.2f", ratio))
            }
        }
    }

    struct IntakeResult {
        let scene: CapturedScene
        /// Per-frame sampling weights from flow. Keyed by timestamp to survive BA reordering and
        /// exclusion.
        let weightByTimestamp: [TimeInterval: Float]?
        /// Reference for detecting scale blow-up after BA.
        let preBaScale: Float
    }

    /// Excludes frames by pose outliers, then the quality filter, and assigns sampling weights from
    /// optical flow.
    static func filterFrames(
        scene: CapturedScene,
        frameQualityFilter: CaptureFrameQualityFilter.Parameters,
        opticalFlowDiagnostics: CaptureOpticalFlow.Parameters,
        flowSoftWeight: CaptureOpticalFlow.WeightParameters,
        log: (String) -> Void
    ) -> IntakeResult {
        // If frames with abnormal translation (1e6+ m) from e.g. ARKit loop closure misfires slip
        // in, the scene scale blows up and breaks everything downstream, so exclude them before BA.
        let cleanScene = filterPoseOutliers(scene: scene, log: log)
        if cleanScene.frames.count < scene.frames.count {
            log("CapturedScene after outlier filter: \(cleanScene.frames.count) frames")
        }
        precondition(!cleanScene.frames.isEmpty, "all frames filtered out as outliers (pose corruption)")
        let qualityScene = CaptureFrameQualityFilter.filter(
            scene: cleanScene,
            parameters: frameQualityFilter,
            log: log
        )
        if qualityScene.frames.count < cleanScene.frames.count {
            log("CapturedScene after quality filter: \(qualityScene.frames.count) frames")
        }
        precondition(!qualityScene.frames.isEmpty, "all frames filtered out by quality filter")
        let flowSummary = CaptureOpticalFlow.analyze(
            frames: qualityScene.frames,
            parameters: opticalFlowDiagnostics,
            log: log
        )
        // Frames that agree poorly with their neighbors are trained less often (TrainView.weight).
        let flowWeights = CaptureOpticalFlow.frameWeights(
            frameCount: qualityScene.frames.count,
            summary: flowSummary,
            parameters: flowSoftWeight,
            log: log
        )
        let weightByTimestamp: [TimeInterval: Float]? = flowSoftWeight.enabled
            ? Dictionary(
                zip(qualityScene.frames.map(\.timestamp), flowWeights),
                uniquingKeysWith: { a, b in Swift.min(a, b) }
            )
            : nil
        let preBaScale = sceneScale(qualityScene.frames)
        log(String(format: "pre-BA scene scale: %.3f m (%d frames)",
                   preBaScale, qualityScene.frames.count))
        return IntakeResult(scene: qualityScene, weightByTimestamp: weightByTimestamp, preBaScale: preBaScale)
    }

    /// PocketGS Sec. III-A 2: refines ARKit pose noise with ORB correspondences + IGR global BA.
    /// If BA shifts poses (e.g. gauge ambiguity) so the scene scale exceeds 10× pre-BA (or 100 m),
    /// the BA result is discarded and training continues with the pre-BA poses.
    static func refinePoses(
        _ scene: CapturedScene, preBaScale: Float,
        orbFastThreshold: Float, orbMaxKeypointsPerFrame: Int, useARKitTracks: Bool,
        lensDistortion: LensDistortionEstimation.Parameters = .disabled,
        log: (String) -> Void
    ) -> CapturedScene {
        let refined = CaptureBA.refine(
            scene: scene,
            fastThreshold: orbFastThreshold,
            maxKeypointsPerFrame: orbMaxKeypointsPerFrame,
            useARKitTracks: useARKitTracks,
            lensDistortion: lensDistortion,
            log: log
        )
        let postBaScale = sceneScale(refined.frames)
        log(String(format: "post-BA scene scale: %.3f m", postBaScale))
        // Reapply the outlier filter in case BA blows away a single frame.
        let cleanedPostBA = filterPoseOutliers(scene: refined, log: log)
        let postFilteredScale = sceneScale(cleanedPostBA.frames)
        if postFilteredScale > preBaScale * 10 || postFilteredScale > 100 {
            log(String(format:
                "warning: BA exploded scene scale (pre=%.3f m → post=%.3f m) — discarding BA result, using pre-BA poses",
                preBaScale, postFilteredScale))
            return scene
        }
        if cleanedPostBA.frames.count < refined.frames.count {
            log(String(format: "using post-BA outlier-filtered scene: %.3f m, %d frames",
                       postFilteredScale, cleanedPostBA.frames.count))
            return cleanedPostBA
        }
        return refined
    }

    /// Writes post-BA (and pre-BA) viewmats / intrinsics / timestamps as JSON to evaluate pose
    /// accuracy.
    static func writePoseDump(
        before: CapturedScene, after: CapturedScene, to url: URL, log: (String) -> Void
    ) {
        var preByTimestamp: [TimeInterval: [Float]] = [:]
        for f in before.frames { preByTimestamp[f.timestamp] = f.viewmat }
        let frames: [[String: Any]] = after.frames.map { f in
            [
                "timestamp": f.timestamp, "width": f.width, "height": f.height,
                "intrinsics": [f.intrinsics.fx, f.intrinsics.fy, f.intrinsics.cx, f.intrinsics.cy],
                "viewmat": f.viewmat, "viewmatPreBA": preByTimestamp[f.timestamp] ?? f.viewmat,
            ]
        }
        do {
            let data = try JSONSerialization.data(withJSONObject: ["frames": frames])
            try data.write(to: url)
            log("wrote pose dump (\(frames.count) frames) to \(url.lastPathComponent)")
        } catch {
            log("[warn] pose dump failed: \(error)")
        }
    }

    /// Thins training views to `cap` with viewpoint-coverage priority (FPS); if `holdoutEvery > 1`,
    /// every Nth is held out. The seed uses all frames, so initial point density is preserved.
    /// `holdoutBeforeSelection` takes every Nth of all frames out first and thins the rest, so the
    /// held-out set does not depend on `cap`.
    /// The training and held-out frames `train` uses with these settings (the same selection), for
    /// evaluating other reconstructions on views training did not see.
    public static func evaluationSplit(
        geometry: CaptureGeometry, holdoutEvery: Int, trainingViewCap: Int? = nil,
        holdoutBeforeSelection: Bool = false
    ) -> (training: [CapturedFrame], heldOut: [CapturedFrame]) {
        selectTrainingFrames(
            geometry.scene.frames,
            cap: trainingViewCap ?? recommendedTrainingViewCap(frameCount: geometry.capturedFrameCount),
            sceneScale: geometry.sceneScale, holdoutEvery: holdoutEvery,
            holdoutBeforeSelection: holdoutBeforeSelection, log: { _ in })
    }

    static func selectTrainingFrames(
        _ frames: [CapturedFrame], cap: Int, sceneScale: Float, holdoutEvery: Int,
        holdoutBeforeSelection: Bool = false, log: (String) -> Void
    ) -> (training: [CapturedFrame], heldOut: [CapturedFrame]) {
        if holdoutBeforeSelection, holdoutEvery > 1 {
            let heldOut = frames.enumerated().filter { $0.offset % holdoutEvery == 0 }.map(\.element)
            let candidates = frames.enumerated().filter { $0.offset % holdoutEvery != 0 }.map(\.element)
            let selectedIdx = selectCoverageViewIndices(candidates, cap: cap, sceneScale: sceneScale)
            let training = selectedIdx.map { candidates[$0] }
            log("[eval-holdout] every \(holdoutEvery) of all frames: \(heldOut.count) held-out / "
                + "\(training.count) of \(candidates.count) remaining frames for training")
            return (training, heldOut)
        }
        var training = frames
        let selectedIdx = selectCoverageViewIndices(frames, cap: cap, sceneScale: sceneScale)
        if selectedIdx.count < frames.count {
            training = selectedIdx.map { frames[$0] }
            log("[view-select] coverage FPS: \(training.count)/\(frames.count) frames for training (seed uses all)")
        }
        guard holdoutEvery > 1 else { return (training, []) }
        let heldOut = training.enumerated().filter { $0.offset % holdoutEvery == 0 }.map(\.element)
        training = training.enumerated().filter { $0.offset % holdoutEvery != 0 }.map(\.element)
        log("[eval-holdout] every \(holdoutEvery): \(heldOut.count) held-out / \(training.count) training views")
        return (training, heldOut)
    }

    /// Computes each frame's camera center (world) as `-Rᵀt`.
    static func cameraCenters(_ frames: [CapturedFrame]) -> [SIMD3<Float>] {
        frames.map { FrameSubsampler.cameraCenter(viewmat: $0.viewmat) }
    }

    /// Excludes frames with abnormal positions (1e6+ m) from tracking failure. Judged by distance
    /// from the per-component median position:
    ///   threshold = max(absoluteThresholdMeters, multiplierAboveMedian × median distance)
    /// Indoors the median distance is 0.5-2 m, so a 50 m absolute threshold cuts only obvious
    /// failures.
    static func filterPoseOutliers(
        scene: CapturedScene,
        absoluteThresholdMeters: Float = 50.0,
        multiplierAboveMedian: Float = 20.0,
        log: (String) -> Void
    ) -> CapturedScene {
        let centers = cameraCenters(scene.frames)
        guard centers.count >= 3 else { return scene }

        func median(_ xs: [Float]) -> Float {
            let s = xs.sorted()
            return s[s.count / 2]
        }
        let medianPos = SIMD3<Float>(
            median(centers.map { $0.x }),
            median(centers.map { $0.y }),
            median(centers.map { $0.z })
        )
        let distances = centers.map { c -> Float in
            let d = c - medianPos
            return (d.x * d.x + d.y * d.y + d.z * d.z).squareRoot()
        }
        let medianDist = median(distances)
        let threshold = Swift.max(absoluteThresholdMeters, multiplierAboveMedian * medianDist)

        var keepIndices: [Int] = []
        var droppedIndices: [Int] = []
        for (i, d) in distances.enumerated() {
            if d <= threshold { keepIndices.append(i) } else { droppedIndices.append(i) }
        }
        if droppedIndices.isEmpty { return scene }
        log(String(format: "pose outlier filter: dropped %d / %d frames (medianDist=%.3fm threshold=%.1fm)",
                   droppedIndices.count, scene.frames.count, medianDist, threshold))
        for i in droppedIndices {
            let c = centers[i]
            log(String(format: "  drop frame[%d] pos=(%.1e, %.1e, %.1e) dist=%.1e m",
                       i, c.x, c.y, c.z, distances[i]))
        }
        return scene.withFrames(keepIndices.map { scene.frames[$0] })
    }

    /// Thins training views to `cap` by farthest-point sampling prioritizing viewpoint coverage
    /// (`cap <= 0` keeps all frames). Distance is camera-center offset + forward-direction angle
    /// difference (scaled by sceneScale).
    static func selectCoverageViewIndices(
        _ frames: [CapturedFrame], cap: Int, sceneScale: Float
    ) -> [Int] {
        let n = frames.count
        guard cap > 0, n > cap else { return Array(0..<n) }
        let centers = cameraCenters(frames)
        // Forward direction in world = row 3 of R (viewmat is row-major world→camera).
        let forwards: [SIMD3<Float>] = frames.map { f in
            let m = f.viewmat
            let v = SIMD3<Float>(m[8], m[9], m[10])
            let l = simd_length(v)
            return l > 1e-6 ? v / l : SIMD3<Float>(0, 0, 1)
        }
        let angleWeight: Float = 0.5 * Swift.max(sceneScale, 1e-4)
        func dist(_ i: Int, _ j: Int) -> Float {
            simd_length(centers[i] - centers[j])
                + angleWeight * (1 - simd_dot(forwards[i], forwards[j]))
        }
        // FPS: seed with index 0 and add the farthest one at a time. minDist is the nearest
        // distance to the selected set.
        var selected = [0]
        var minDist = (0..<n).map { dist($0, 0) }
        minDist[0] = -1
        while selected.count < cap {
            var best = -1
            var bestD: Float = -1
            for i in 0..<n where minDist[i] > bestD {
                bestD = minDist[i]; best = i
            }
            if best < 0 { break }
            selected.append(best)
            minDist[best] = -1
            let b = best
            for i in 0..<n where minDist[i] >= 0 {
                let d = dist(i, b)
                if d < minDist[i] { minDist[i] = d }
            }
        }
        return selected.sorted()   // keep chronological order
    }

    /// scene scale = max distance of camera centers from their centroid × 1.1.
    public static func sceneScale(_ frames: [CapturedFrame]) -> Float {
        TrainingDriver.sceneScale(viewmats: frames.map(\.viewmat))
    }
}
