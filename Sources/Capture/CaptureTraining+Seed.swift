// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import Trainer
import simd

// Seed point cloud construction (fusion of LiDAR + MVS dense) and the initial model.
extension CaptureTraining {

    static func memorySnapshot(_ label: String) -> String {
        func mb(_ b: Int) -> String { String(format: "%.0f", Double(b) / (1024 * 1024)) }
        let s = MLX.Memory.snapshot()
        return "mem[\(label)] active=\(mb(s.activeMemory))M peak=\(mb(s.peakMemory))M cache=\(mb(s.cacheMemory))M"
    }

    /// Far clip for the dense seed: drop points beyond factor × sceneScale from the camera centroid
    /// (guards against far floaters from depth outliers). factor 0 disables; no-op if it would
    /// remove every point.
    struct SeedFarClip {
        let frames: [CapturedFrame]
        let sceneScale: Float
        let factor: Float

        func callAsFunction(_ cloud: SeedCloud, log: (String) -> Void) -> SeedCloud {
            guard factor > 0 else { return cloud }
            let centers = CaptureTraining.cameraCenters(frames)
            guard !centers.isEmpty else { return cloud }
            var c = SIMD3<Float>.zero
            for cc in centers { c += cc }
            c /= Float(centers.count)
            let maxR = factor * sceneScale
            let r2 = maxR * maxR
            let pos = cloud.positions.asArray(Float.self)
            let col = cloud.colors.asArray(Float.self)
            let conf = cloud.confidences.asArray(Float.self)
            let n = cloud.positions.shape[0]
            var keptP = [Float](); keptP.reserveCapacity(pos.count)
            var keptC = [Float](); keptC.reserveCapacity(col.count)
            var keptConf = [Float](); keptConf.reserveCapacity(conf.count)
            var dropped = 0
            for i in 0..<n {
                let dx = pos[i * 3] - c.x, dy = pos[i * 3 + 1] - c.y, dz = pos[i * 3 + 2] - c.z
                if dx * dx + dy * dy + dz * dz <= r2 {
                    keptP.append(pos[i * 3]); keptP.append(pos[i * 3 + 1]); keptP.append(pos[i * 3 + 2])
                    keptC.append(col[i * 3]); keptC.append(col[i * 3 + 1]); keptC.append(col[i * 3 + 2])
                    keptConf.append(conf[i])
                } else { dropped += 1 }
            }
            if dropped > 0 {
                log(String(format:
                    "[seed] far-clip: dropped %d/%d seed points beyond %.2fm (%.1f×scene_scale)",
                    dropped, n, maxR, factor))
            }
            let kept = keptP.count / 3
            guard kept > 0 else { return cloud }
            let p = MLXArray(keptP, [kept, 3])
            let cc = MLXArray(keptC, [kept, 3])
            let cf = MLXArray(keptConf, [kept])
            eval(p, cc, cf)
            return SeedCloud(label: cloud.label, positions: p, colors: cc, confidences: cf)
        }
    }

    /// MVS dense seed (far-clipped). The depth-range constraint prefers LiDAR, falling back to
    /// sparse points. `depthMaps` receives the depth maps behind the point cloud (for meshing).
    static func mvsSeedCloud(
        scene: CapturedScene, parameters: CostVolumeMVS.Parameters, downscale: Int?,
        farClip: SeedFarClip, depthMaps: ((CostVolumeMVS.DepthView) -> Void)? = nil,
        log: (String) -> Void
    ) -> SeedCloud? {
        log(memorySnapshot("pre-mvs"))
        MLX.Memory.clearCache()
        let t0 = Date()
        defer { log(String(format: "[timing] MVS(total) = %.2fs", Date().timeIntervalSince(t0))) }
        let constraint: [SIMD3<Float>]
        if scene.lidarPoints.count >= 200 {
            constraint = scene.lidarPoints
            log("MVS depth-range source: LiDAR \(constraint.count)")
        } else {
            let (pts, label) = sparseDepthRangePoints(scene: scene)
            constraint = pts
            log("MVS depth-range source: \(label)")
        }
        let viewmats = scene.frames.map { $0.viewmat }
        var intr = scene.frames.map { $0.intrinsics }
        let longSide = scene.frames.first.map { max($0.width, $0.height) } ?? 0
        let s = downscale ?? max(1, (longSide + mvsTargetLongSide - 1) / mvsTargetLongSide)
        let images = scene.frames.map { frame -> MLXArray in
            let img = frame.imageFloat
            guard s > 1 else { return img }
            // Box-average downscale by 1/s. Pixel center (x+0.5) maps to (x+0.5)·s at full size, so
            // intrinsics simply scale by 1/s. Evaluated per frame, so a frame reloaded from disk
            // (`CapturedScene.releaseSavedImages`) is resident only at the reduced size.
            let h = img.shape[0] / s, w = img.shape[1] / s
            let reduced = img[0..<(h * s), 0..<(w * s)]
                .reshaped([h, s, w, s, 3]).mean(axes: [1, 3])
            eval(reduced)
            return reduced
        }
        if s > 1 {
            let sf = Float(s)
            intr = intr.map { (fx: $0.fx / sf, fy: $0.fy / sf, cx: $0.cx / sf, cy: $0.cy / sf) }
            log("MVS downscale 1/\(s): \(images.first.map { "\($0.shape[1])x\($0.shape[0])" } ?? "-")")
        }
        let mvs = TrainingDriver.runGeometryPrior(
            viewmats: viewmats, intrinsics: intr, images: images,
            sparsePoints: constraint, parameters: parameters,
            depthMaps: depthMaps,
            log: log
        )
        MLX.Memory.clearCache()
        if CostVolumeMVS.profiling {
            let total = CostVolumeMVS.profile.values.reduce(0, +)
            log("[MVS profile] " + CostVolumeMVS.profile.sorted { $0.value > $1.value }
                .map { String(format: "%@=%.2fs", $0.key, $0.value) }.joined(separator: " ")
                + String(format: " (sum=%.2fs)", total))
        }
        var result: SeedCloud?
        if let mvs, mvs.positions.shape[0] > 0 {
            result = farClip(SeedCloud(
                label: "MVS", positions: mvs.positions, colors: mvs.colors, confidences: mvs.confidences), log: log)
        }
        log(memorySnapshot("post-mvs"))
        return result
    }

    /// PocketGS Operator I disc-like init (K_n=16, K_s=3, r_normal=0.3, opacity 0.1). The MVS seed
    /// is dense, so there is no eq. 6 scale limit (initScale 1.0, scaleCap unlimited).
    static func initialModel(
        positions: MLXArray, colors: MLXArray, maxInitGaussians: Int?, log: (String) -> Void
    ) -> GaussianModel {
        let initScale: Float = 1.0
        let scaleCap: Float = .infinity
        // Too many initial Gaussians make training OOM on iOS, so stride-subsample to
        // maxInitGaussians.
        let preCap = positions.shape[0]
        let model = GaussianModel.fromPointCloudDisc(
            positions: positions, colors: colors,
            kN: 16, kS: 3, rNormal: 0.3,
            initOpacity: 0.1,
            initScale: initScale, scaleCap: scaleCap,
            maxPoints: maxInitGaussians
        )
        if let cap = maxInitGaussians, preCap > cap {
            log("init gaussian cap: \(preCap) → \(model.count) (max=\(cap))")
        }
        log("initialised \(model.count) gaussians (PocketGS disc-like; "
            + "initScale=\(initScale), scaleCap=\(scaleCap))")
        return model
    }

    /// Returns the seed fusing the capture's LiDAR points (mid-grey) with extraSeedClouds. If there
    /// are none, synthesizes a spherical random seed. Sparse (BA / ARKit triangulated) points stay
    /// out of the seed as outlier / floater sources; they still serve for poses and the MVS depth
    /// range.
    static func captureLiDAROrSyntheticSeed(
        scene: CapturedScene, sceneScale: Float,
        syntheticSparseCount: Int, seed: UInt64,
        extraSeedClouds: [SeedCloud] = [],
        fusionVoxelSize: Float = seedFusionVoxelSize,
        log: (String) -> Void
    ) -> (positions: MLXArray, colors: MLXArray) {
        log("seed sparse source: excluded (the seed is MVS + LiDAR only)")
        // LiDAR fusion confidence is a fixed heuristic: 1.0 (real sensor).
        var seedClouds: [SeedCloud] = []
        if let lidar = pointSeedCloud(label: "lidar", points: scene.lidarPoints, confidence: 1.0) {
            seedClouds.append(lidar)
        }
        seedClouds += extraSeedClouds
        if !seedClouds.isEmpty {
            return concatenateSeedClouds(seedClouds, voxelSize: fusionVoxelSize, log: log)
        }
        log("no sparse/LiDAR points → synthesising \(syntheticSparseCount)-pt spherical seed (radius=\(sceneScale))")
        return randomSphericalSeed(
            frames: scene.frames, count: syntheticSparseCount,
            radius: sceneScale, seed: seed
        )
    }

    /// Sparse points bounding the MVS depth range when there is no LiDAR. ARKit raw points include
    /// VIO drift, so BA-refined ones are used when BA produced them.
    static func sparseDepthRangePoints(
        scene: CapturedScene
    ) -> (points: [SIMD3<Float>], label: String) {
        if let ba = scene.baRefinedPoints {
            return (ba, "BA-only \(ba.count) (ARKit raw excluded)")
        }
        return (scene.sparsePoints, "ARKit raw \(scene.sparsePoints.count)")
    }

    struct SeedCloud {
        let label: String
        let positions: MLXArray
        let colors: MLXArray
        /// Weight for Operator G's confidence-weighted fusion (`SeedFusion.fuse`).
        let confidences: MLXArray

        var count: Int { positions.shape[0] }
    }

    static func pointSeedCloud(
        label: String,
        points: [SIMD3<Float>],
        confidence: Float
    ) -> SeedCloud? {
        guard !points.isEmpty else { return nil }
        var flat: [Float] = []
        flat.reserveCapacity(points.count * 3)
        for p in points {
            flat += [p.x, p.y, p.z]
        }
        let positions = MLXArray(flat, [points.count, 3])
        let colors = MLXArray(
            [Float](repeating: 0.5, count: points.count * 3),
            [points.count, 3]
        )
        let confidences = MLXArray(
            [Float](repeating: confidence, count: points.count),
            [points.count]
        )
        return SeedCloud(label: label, positions: positions, colors: colors, confidences: confidences)
    }

    /// Voxel size for PocketGS Operator G's confidence-weighted fusion. Large enough that Operator
    /// I's 3-NN/K_s distance does not collapse. MVS is already deduplicated with its own
    /// voxelDedupSize, so when MVS is used train() lowers this to that width.
    static let seedFusionVoxelSize: Float = 0.01
    static func concatenateSeedClouds(
        _ clouds: [SeedCloud],
        voxelSize: Float = seedFusionVoxelSize,
        log: (String) -> Void
    ) -> (positions: MLXArray, colors: MLXArray) {
        precondition(!clouds.isEmpty)
        let positions: MLXArray
        let colors: MLXArray
        if clouds.count == 1 {
            positions = clouds[0].positions
            colors = clouds[0].colors
        } else {
            (positions, colors) = SeedFusion.fuse(
                clouds: clouds.map { ($0.positions, $0.colors, $0.confidences) },
                voxelSize: voxelSize
            )
            eval(positions, colors)
        }
        let summary = clouds
            .map { "\($0.label) \($0.count)" }
            .joined(separator: " + ")
        log("init seed: \(summary) = \(clouds.reduce(0) { $0 + $1.count }) raw → "
            + "\(positions.shape[0]) points (after confidence-weighted fusion, voxel=\(voxelSize)m)")
        return (positions, colors)
    }

    /// Uniformly samples `count` points inside a sphere of radius `radius` centered on the
    /// camera-center centroid (the classic 3DGS bootstrap when there are no seed points).
    static func randomSphericalSeed(
        frames: [CapturedFrame], count: Int, radius: Float, seed: UInt64
    ) -> (positions: MLXArray, colors: MLXArray) {
        var cx: Float = 0, cy: Float = 0, cz: Float = 0
        for c in cameraCenters(frames) {
            cx += c.x
            cy += c.y
            cz += c.z
        }
        let n = Float(frames.count)
        let centroid = SIMD3<Float>(cx / n, cy / n, cz / n)

        var state = seed == 0 ? 0xC0FFEEDEADBEEF : seed
        func nextFloat() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(state >> 32) / Float(UInt32.max) * 2 - 1
        }
        var posFlat: [Float] = []
        var colFlat: [Float] = []
        posFlat.reserveCapacity(count * 3)
        colFlat.reserveCapacity(count * 3)
        for _ in 0..<count {
            var x: Float, y: Float, z: Float
            repeat {
                x = nextFloat(); y = nextFloat(); z = nextFloat()
            } while x*x + y*y + z*z > 1
            posFlat += [centroid.x + radius * x,
                        centroid.y + radius * y,
                        centroid.z + radius * z]
            colFlat += [0.5, 0.5, 0.5]
        }
        return (
            positions: MLXArray(posFlat, [count, 3]),
            colors: MLXArray(colFlat, [count, 3])
        )
    }
}
