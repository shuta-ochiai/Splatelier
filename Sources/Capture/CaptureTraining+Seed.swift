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
            let depth = cloud.observedDepths.asArray(Float.self)
            let n = cloud.positions.shape[0]
            var keptP = [Float](); keptP.reserveCapacity(pos.count)
            var keptC = [Float](); keptC.reserveCapacity(col.count)
            var keptConf = [Float](); keptConf.reserveCapacity(conf.count)
            var keptDepth = [Float](); keptDepth.reserveCapacity(depth.count)
            var dropped = 0
            for i in 0..<n {
                let dx = pos[i * 3] - c.x, dy = pos[i * 3 + 1] - c.y, dz = pos[i * 3 + 2] - c.z
                if dx * dx + dy * dy + dz * dz <= r2 {
                    keptP.append(pos[i * 3]); keptP.append(pos[i * 3 + 1]); keptP.append(pos[i * 3 + 2])
                    keptC.append(col[i * 3]); keptC.append(col[i * 3 + 1]); keptC.append(col[i * 3 + 2])
                    keptConf.append(conf[i])
                    keptDepth.append(depth[i])
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
            let cd = MLXArray(keptDepth, [kept])
            eval(p, cc, cf, cd)
            return SeedCloud(label: cloud.label, positions: p, colors: cc, confidences: cf, observedDepths: cd)
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
        let lidar = scene.lidarWorldPoints
        if lidar.count >= 200 {
            constraint = lidar
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
                label: "MVS", positions: mvs.positions, colors: mvs.colors, confidences: mvs.confidences,
                observedDepths: mvs.observedDepths), log: log)
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

    /// Returns the seed fusing the capture's LiDAR points (mid-grey unless `lidarColors`) with extraSeedClouds. If there
    /// are none, synthesizes a spherical random seed. Sparse (BA / ARKit triangulated) points stay
    /// out of the seed as outlier / floater sources, except with `sparseGapFill`: the BA points
    /// where the other clouds left a gap (`sparseGapPoints`) join it, outside `maxPoints`'s
    /// thinning so the gaps keep them.
    static func captureLiDAROrSyntheticSeed(
        scene: CapturedScene, sceneScale: Float,
        syntheticSparseCount: Int, seed: UInt64,
        extraSeedClouds: [SeedCloud] = [],
        fusionVoxelSize: Float = seedFusionVoxelSize,
        allocation: SeedAllocation = .even,
        maxPoints: Int? = nil,
        // Color the LiDAR points from the frame images (mid-grey otherwise).
        lidarColors: Bool = true,
        sparseGapFill: Bool = false,
        log: (String) -> Void
    ) -> (positions: MLXArray, colors: MLXArray) {
        if !sparseGapFill { log("seed sparse source: excluded (the seed is MVS + LiDAR only)") }
        // Each LiDAR point's fusion confidence is its samples' (`LiDARSample.confidence`).
        var seedClouds: [SeedCloud] = []
        let lidarSamples = scene.lidarWorldSamples(colors: lidarColors)
        let lidarPoints = voxelAverage(lidarSamples, voxelSize: lidarVoxelSize)
        if !lidarSamples.isEmpty {
            log(String(format: "[seed] LiDAR samples %d → %d (averaged per %.0f mm voxel), mean confidence %.2f",
                       lidarSamples.count, lidarPoints.count, lidarVoxelSize * 1000,
                       lidarPoints.reduce(0) { $0 + $1.confidence } / Float(Swift.max(1, lidarPoints.count))))
        }
        var lidarCloud = HostCloud(
            positions: lidarPoints.map(\.position),
            colors: lidarPoints.map(\.color),
            confidences: lidarPoints.map(\.confidence),
            depths: lidarPoints.map(\.depth))
        var others = extraSeedClouds
        if case .confidence(let cellPixels, let lidarWeight) = allocation {
            let cells = AdaptiveCells(frames: scene.frames, cellPixels: cellPixels)
            let mvsIndex = others.firstIndex(where: { $0.label == "MVS" })
            var mvs = mvsIndex.map { hostCloud(others[$0]) } ?? HostCloud()
            if lidarCloud.count > 0, mvs.count > 0 {
                let keep = selectSources(mvs: mvs, lidar: lidarCloud, cells: cells,
                                         lidarWeight: lidarWeight, log: log)
                mvs = mvs.subset(keep.mvsKeep)
                lidarCloud = lidarCloud.subset(keep.lidarKeep)
            }
            // Cap before fusion: fusion only merges points, so the count stays within the cap.
            if let maxPoints, mvs.count + lidarCloud.count > maxPoints {
                let all = lidarCloud.appending(mvs)
                let keep = capPerCell(all, cells: cells, budget: maxPoints)
                let lidarCount = lidarCloud.count
                lidarCloud = lidarCloud.subset(Array(keep[..<lidarCount]))
                mvs = mvs.subset(Array(keep[lidarCount...]))
                log("[seed alloc] cap per cell: \(all.count) → \(lidarCloud.count + mvs.count) points "
                    + "(LiDAR \(lidarCloud.count), MVS \(mvs.count))")
            }
            if let i = mvsIndex { others[i] = seedCloud(label: "MVS", mvs) }
        }
        if lidarCloud.count > 0 { seedClouds.append(seedCloud(label: "lidar", lidarCloud)) }
        seedClouds += others.filter { $0.count > 0 }
        if !seedClouds.isEmpty {
            let fused = concatenateSeedClouds(seedClouds, voxelSize: fusionVoxelSize, log: log)
            guard sparseGapFill, let ba = scene.baRefinedPoints else { return fused }
            let candidates = farClipped(ba, frames: scene.frames, sceneScale: sceneScale)
            let gaps = sparseGapPoints(candidates, covered: seedClouds, voxelSize: sparseGapVoxelSize)
            log(String(format: "[seed] sparse gap fill: %d BA points → %d with no MVS / LiDAR point in their %.0f cm voxel",
                       candidates.count, gaps.count, sparseGapVoxelSize * 100))
            guard !gaps.isEmpty else { return fused }
            // Thin the fused clouds to make room, as `GaussianModel.fromPointCloudDisc` would
            // (evenly), so its own cap leaves the gap points alone.
            var (positions, colors) = fused
            let total = positions.shape[0]
            if let maxPoints, total + gaps.count > maxPoints, maxPoints > gaps.count {
                let count = maxPoints - gaps.count
                let idx = MLXArray((0..<count).map { Int32(Int64($0) * Int64(total) / Int64(count)) }, [count])
                positions = positions[idx]
                colors = colors[idx]
            }
            // Mid-grey: the colors converge in training.
            return (MLX.concatenated([positions, MLXArray(gaps.flatMap { [$0.x, $0.y, $0.z] }, [gaps.count, 3])]),
                    MLX.concatenated([colors, MLXArray.full([gaps.count, 3], values: MLXArray(Float(0.5)))]))
        }
        log("no sparse/LiDAR points → synthesising \(syntheticSparseCount)-pt spherical seed (radius=\(sceneScale))")
        return randomSphericalSeed(
            frames: scene.frames, count: syntheticSparseCount,
            radius: sceneScale, seed: seed
        )
    }

    static func hostCloud(_ c: SeedCloud) -> HostCloud {
        let p = c.positions.asArray(Float.self), col = c.colors.asArray(Float.self)
        return HostCloud(
            positions: Swift.stride(from: 0, to: p.count, by: 3).map { SIMD3(p[$0], p[$0 + 1], p[$0 + 2]) },
            colors: Swift.stride(from: 0, to: col.count, by: 3).map { SIMD3(col[$0], col[$0 + 1], col[$0 + 2]) },
            confidences: c.confidences.asArray(Float.self),
            depths: c.observedDepths.asArray(Float.self))
    }

    static func seedCloud(label: String, _ c: HostCloud) -> SeedCloud {
        SeedCloud(
            label: label,
            positions: MLXArray(c.positions.flatMap { [$0.x, $0.y, $0.z] }, [c.count, 3]),
            colors: MLXArray(c.colors.flatMap { [$0.x, $0.y, $0.z] }, [c.count, 3]),
            confidences: MLXArray(c.confidences, [c.count]),
            observedDepths: MLXArray(c.depths, [c.count]))
    }

    /// Voxel for `sparseGapPoints`: a BA point counts as covered when an MVS / LiDAR point shares
    /// its voxel. Well above the MVS point spacing (3 mm), so only real gaps stay uncovered.
    static let sparseGapVoxelSize: Float = 0.02

    /// The `points` whose `voxelSize` voxel holds no point of `covered`: surfaces MVS did not
    /// reconstruct (repeating or plain-ish texture where the views did not agree) but feature
    /// tracking did.
    static func sparseGapPoints(_ points: [SIMD3<Float>], covered clouds: [SeedCloud], voxelSize: Float) -> [SIMD3<Float>] {
        let scale = 1 / voxelSize
        func key(_ x: Float, _ y: Float, _ z: Float) -> Int64 {
            let ix = Int64((x * scale).rounded(.down)) & 0x1F_FFFF
            let iy = Int64((y * scale).rounded(.down)) & 0x1F_FFFF
            let iz = Int64((z * scale).rounded(.down)) & 0x1F_FFFF
            return ix | iy << 21 | iz << 42
        }
        // Index the few sparse points and look the many cloud points up, rather than hashing
        // millions of MVS voxels.
        var byVoxel: [Int64: [Int]] = [:]
        for (i, p) in points.enumerated() { byVoxel[key(p.x, p.y, p.z), default: []].append(i) }
        var covered = [Bool](repeating: false, count: points.count)
        for cloud in clouds {
            let p = cloud.positions.asArray(Float.self)
            // Every 4th point: the clouds are far denser than the voxel.
            for j in Swift.stride(from: 0, to: cloud.count, by: 4) {
                guard let ids = byVoxel[key(p[j * 3], p[j * 3 + 1], p[j * 3 + 2])] else { continue }
                for i in ids { covered[i] = true }
            }
        }
        return points.indices.filter { !covered[$0] }.map { points[$0] }
    }

    /// `points` within `factor` × sceneScale of the camera-center centroid, the dense seed's far
    /// clip (`SeedFarClip`).
    static func farClipped(
        _ points: [SIMD3<Float>], frames: [CapturedFrame], sceneScale: Float, factor: Float = 6
    ) -> [SIMD3<Float>] {
        let centers = cameraCenters(frames)
        guard !centers.isEmpty else { return points }
        let c = centers.reduce(SIMD3<Float>.zero, +) / Float(centers.count)
        let r = factor * sceneScale
        return points.filter { simd_length_squared($0 - c) <= r * r }
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

    /// Voxel the LiDAR samples are averaged over before fusion. Samples of one surface from many
    /// frames scatter by the depth error (L1 about 2.5 cm on ARKitScenes, PromptDA Table 1), far
    /// more than the fusion voxel matched to MVS (3 mm), so unaveraged they enter the seed as a
    /// thick shell with one point per sample. The voxel also sets the spacing left on well-covered
    /// surfaces, so it trades density for averaging: 5 mm keeps four times the points of 1 cm.
    static let lidarVoxelSize: Float = 0.005

    struct SeedCloud {
        let label: String
        let positions: MLXArray
        let colors: MLXArray
        /// Weight for Operator G's confidence-weighted fusion (`SeedFusion.fuse`).
        let confidences: MLXArray
        /// Depth (m) at which the frames saw each point (sets its pixel footprint).
        let observedDepths: MLXArray

        var count: Int { positions.shape[0] }
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
