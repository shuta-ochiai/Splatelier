// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// `live-mvs-replay`: replay benchmark for deciding whether MVS can run during capture (capture
// guide). Feeds the keyframes of a validation dataset (.valset) one at a time in capture order
// through
//   batch  — the current MVS mode that runs everything after capture (CostVolumeMVS.build)
//   stream — the incremental engine StreamingMVS (one depth map + consistency check per keyframe,
//            lookahead / fusionDelay 2; the live in-capture point cloud `.mvs` uses 0 / 0)
// and writes each one's point-cloud PLY and a per-step timing CSV to outDir.
//
// env:
//   REPLAY_POSES=arkit|ba     poses. arkit (default) = ARKit poses at capture (the only ones usable
//                             during capture), ba = post-BA poses from training
//   REPLAY_MODES=batch,stream  (default both). live = hands keyframes to LiveMVSReconstructor, the
//                              live in-capture point cloud, at capture-time intervals (end-to-end
//                              check incl. background queue and drops). surfel = how the in-capture
//                              display looks as surfels (splats that sharpen as you get closer)
//                              (LiveSurfelPreview.swift)
//   REPLAY_SPARSE=causal|all  sparse points for the depth range. causal (default) = add per-frame
//   ARKit
//                             feature-point observations in arrival order (old valsets without them
//                             fall back to all)
//   MVS_COVIS=<0..1>|off      lower bound on reference covisibility (default same as the app)

import Foundation
import MLX
import Trainer
import Capture
import simd

func runLiveMVSReplay(valsetURL: URL, outDir: URL) throws {
    let env = ProcessInfo.processInfo.environment
    let logErr: (String) -> Void = { FileHandle.standardError.write(Data(($0 + "\n").utf8)) }
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

    let loaded = try CaptureDatasetStore.load(from: valsetURL)
    let useBA = env["REPLAY_POSES"]?.lowercased() == "ba"
    let scene = useBA ? CaptureBA.refine(scene: loaded, log: { _ in }) : loaded
    let frames = scene.frames
    let allSparse: [SIMD3<Float>] = (useBA && scene.baRefinedPoints?.isEmpty == false)
        ? scene.baRefinedPoints! : scene.sparsePoints
    let hasObservations = frames.contains { !$0.featureObservations.isEmpty }
    let causalSparse = !useBA && hasObservations && env["REPLAY_SPARSE"]?.lowercased() != "all"
    logErr("[replay] \(frames.count) frames, poses=\(useBA ? "BA" : "ARKit"), "
        + "sparse=\(causalSparse ? "causal (per-frame ARKit features)" : "all \(allSparse.count)")")

    // MVS input (same box downscale as CaptureTraining).
    let longSide = max(frames[0].width, frames[0].height)
    let s = env["MVS_DOWNSCALE"].flatMap { Int($0) } ?? max(1, (longSide + 319) / 320)
    let images: [MLXArray] = frames.map {
        let img = StreamingMVS.boxDownscale($0.image, factor: s)
        eval(img)
        return img
    }
    let sf = Float(s)
    let intr = frames.map { (fx: $0.intrinsics.fx / sf, fy: $0.intrinsics.fy / sf,
                             cx: $0.intrinsics.cx / sf, cy: $0.intrinsics.cy / sf) }
    let viewmats = frames.map(\.viewmat)
    logErr("[replay] MVS input 1/\(s): \(images[0].shape[1])x\(images[0].shape[0])")

    // World coordinates of ARKit feature points first observed at frame k (causal sparse points).
    func newFeaturePoints(_ k: Int, seen: inout Set<UInt64>) -> [SIMD3<Float>] {
        let vm = viewmats[k]
        var out: [SIMD3<Float>] = []
        for o in frames[k].featureObservations where seen.insert(o.trackID).inserted {
            let p = o.cameraPoint
            let dx = p.x - vm[3], dy = p.y - vm[7], dz = p.z - vm[11]
            out.append(SIMD3<Float>(vm[0] * dx + vm[4] * dy + vm[8] * dz,
                                    vm[1] * dx + vm[5] * dy + vm[9] * dz,
                                    vm[2] * dx + vm[6] * dy + vm[10] * dz))
        }
        return out
    }
    var params = CaptureTraining.MVSMode.mvsParameters
    if let c = env["MVS_COVIS"] { params.refSelection.minCovisibility = mvsCovisibility(c) }
    let modes = Set((env["REPLAY_MODES"] ?? "batch,stream").split(separator: ",").map(String.init))

    if modes.contains("batch") {
        let t0 = Date()
        let out = CostVolumeMVS.build(
            inputs: .init(viewmats: viewmats, intrinsics: intr, rgbImages: images,
                          sparsePoints: allSparse, sceneCenter: StreamingMVS.cameraCentroid(viewmats)),
            parameters: params, log: { _ in })
        let dt = Date().timeIntervalSince(t0)
        try writePointPLY(out.positions, out.colors, to: outDir.appendingPathComponent("batch.ply"))
        print(String(format: "[replay batch] %d points, %.2fs (%.1f ms/frame)",
                     out.positions.shape[0], dt, 1000 * dt / Double(frames.count)))
        if CostVolumeMVS.profiling {   // MVS_PROFILE=1: total seconds per stage
            print("[replay batch profile] " + CostVolumeMVS.profile.sorted { $0.value > $1.value }
                .map { String(format: "%@ %.2fs", $0.key, $0.value) }.joined(separator: ", "))
        }
    }

    if modes.contains("live") {
        var seen = Set<UInt64>()
        try runLiveReconstructorReplay(
            frames: frames, sparse: allSparse, causal: causalSparse, parameters: params,
            newFeaturePoints: { newFeaturePoints($0, seen: &seen) },
            plyURL: outDir.appendingPathComponent("live.ply"))
    }

    if modes.contains("surfel") {
        var seen = Set<UInt64>()
        let finalMVS = env["SURFEL_FINAL"] == "1" ? try prepareFinalMVS(scene: loaded, log: logErr) : nil
        try runSurfelPreviewReplay(
            frames: frames, images: images, intrinsics: intr, viewmats: viewmats, parameters: params,
            sparse: allSparse, causal: causalSparse,
            newFeaturePoints: { newFeaturePoints($0, seen: &seen) }, finalMVS: finalMVS, outDir: outDir)
    }

    if modes.contains("stream") {
        let engine = StreamingMVS(parameters: .init(mvs: params))
        if !causalSparse { engine.addSparsePoints(allSparse) }
        var seen = Set<UInt64>()
        var csv = "frame,timestamp,census_ms,depth_ms,fusion_ms,depth_targets,fused_frames,new_points\n"
        func row(_ k: Int, _ ts: Double, _ st: StreamingMVS.Step) {
            csv += String(format: "%d,%.6f,%.3f,%.3f,%.3f,%@,%@,%d\n", k, ts,
                          1000 * st.censusSeconds, 1000 * st.depthSeconds, 1000 * st.fusionSeconds,
                          st.depthTargets.map(String.init).joined(separator: " ") as NSString,
                          st.fusedFrames.map(String.init).joined(separator: " ") as NSString,
                          st.newPoints)
        }
        // During capture the valset's full-size and downscaled images live outside MLX (pending
        // [UInt8]), so subtract them and look only at what the engine added.
        MLX.Memory.clearCache()
        let baseline = MLX.Memory.activeMemory
        MLX.Memory.peakMemory = 0
        let t0 = Date()
        var stepTimes: [Double] = []
        for k in frames.indices {
            if causalSparse { engine.addSparsePoints(newFeaturePoints(k, seen: &seen)) }
            let st = engine.addFrame(image: images[k], viewmat: viewmats[k], intrinsics: intr[k])
            stepTimes.append(st.censusSeconds + st.depthSeconds + st.fusionSeconds)
            row(k, frames[k].timestamp, st)
        }
        let fin = engine.finish()
        row(-1, frames.last!.timestamp, fin)
        let dt = Date().timeIntervalSince(t0)
        let peak = Double(MLX.Memory.peakMemory - baseline) / 1_048_576
        try csv.write(to: outDir.appendingPathComponent("stream_steps.csv"), atomically: true, encoding: .utf8)
        let (pos, col) = engine.cloud()
        try writePointPLY(pos, col, to: outDir.appendingPathComponent("stream.ply"))
        let sorted = stepTimes.sorted()
        print(String(format: "[replay stream] lookahead=%d fusionDelay=%d: %d points (pre-dedup %d), "
            + "total %.2fs, per-step median %.1f ms p90 %.1f ms max %.1f ms, finish %.2fs, engine GPU peak +%.0f MB",
            engine.parameters.lookahead, engine.parameters.fusionDelay,
            pos.shape[0], engine.pointCount, dt,
            1000 * sorted[sorted.count / 2], 1000 * sorted[sorted.count * 9 / 10], 1000 * sorted.last!,
            fin.censusSeconds + fin.depthSeconds + fin.fusionSeconds, peak))
    }
}

/// Feeds valset keyframes at capture-time intervals to LiveMVSReconstructor, the implementation of
/// the live in-capture point cloud `.mvs`. Like ARKitCaptureSession, it submits full-size RGB bytes
/// + ARKit camera-to-world + newly seen feature points, and finally prints the stopAndWait stats.
/// Writes the displayed splats at end of capture (voxel centers and mean colors) to `plyURL`.
func runLiveReconstructorReplay(
    frames: [CapturedFrame], sparse: [SIMD3<Float>], causal: Bool,
    parameters: CostVolumeMVS.Parameters,
    newFeaturePoints: (Int) -> [SIMD3<Float>], plyURL: URL
) throws {
    var last: [CaptureCoverageSplat] = []
    let lock = NSLock()
    // REPLAY_VOXEL=<m>: display voxel edge length (default is LiveMVSReconstructor's default).
    let env = ProcessInfo.processInfo.environment
    let recon: LiveMVSReconstructor
    if let v = env["REPLAY_VOXEL"].flatMap({ Float($0) }) {
        recon = LiveMVSReconstructor(
            parameters: parameters,
            accumulator: CaptureCoverageAccumulator(
                voxelSizeMeters: v, maxSplats: 1_000_000,
                viewVoxelSizeMeters: env["REPLAY_VIEW_VOXEL"].flatMap { Float($0) }),
            onSplats: { s in lock.lock(); last = s; lock.unlock() })
    } else {
        recon = LiveMVSReconstructor(
            parameters: parameters, onSplats: { s in lock.lock(); last = s; lock.unlock() })
    }
    let start = Date()
    let ts0 = frames[0].timestamp
    for (k, f) in frames.enumerated() {
        let wait = (f.timestamp - ts0) - Date().timeIntervalSince(start)
        if wait > 0 { Thread.sleep(forTimeInterval: wait) }
        let pts = causal ? newFeaturePoints(k) : (k == 0 ? sparse : [])
        recon.submit(
            rgb: f.image.asType(.uint8).asArray(UInt8.self), width: f.width, height: f.height,
            intrinsics: f.intrinsics,
            cameraToWorld: PoseConversion.cameraToWorld(fromViewmat: f.viewmat),
            newSparsePoints: pts)
    }
    let summary = recon.stopAndWait()
    lock.lock(); let splats = last; lock.unlock()
    try writePointPLY(
        MLXArray(splats.flatMap { [$0.position.x, $0.position.y, $0.position.z] }, [splats.count, 3]),
        MLXArray(splats.flatMap { c -> [Float] in let v = c.color ?? .zero; return [v.x, v.y, v.z] },
                 [splats.count, 3]),
        to: plyURL)
    print(String(format: "[replay live] %@, coverage splats %d, wall %.1fs (capture %.1fs)",
                 summary as NSString, splats.count, Date().timeIntervalSince(start),
                 frames.last!.timestamp - ts0))
    // Distribution of "number of keyframes that produced points" per voxel (input for picking the
    // threshold that highlights under-captured areas).
    let bins = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, Int.max)]
    let hist = bins.map { b in splats.filter { $0.viewCount >= b.0 && $0.viewCount <= b.1 }.count }
    let label = bins.map { $0.1 == Int.max ? "\($0.0)+" : ($0.0 == $0.1 ? "\($0.0)" : "\($0.0)-\($0.1)") }
    let total = max(1, splats.count)
    print("[replay live] viewCount: " + zip(label, hist).map {
        String(format: "%@=%.0f%%", $0.0 as NSString, 100 * Double($0.1) / Double(total)) }.joined(separator: " "))
    // REPLAY_SPLAT_CSV=<path>: dumps the final display voxels (x,y,z,r,g,b,viewCount,point count)
    // (for visual inspection).
    if let path = env["REPLAY_SPLAT_CSV"] {
        var csv = "x,y,z,r,g,b,views,points\n"
        for sp in splats {
            let c = sp.color ?? SIMD3<Float>(repeating: 0.5)
            csv += String(format: "%.5f,%.5f,%.5f,%.3f,%.3f,%.3f,%d,%d\n", sp.position.x, sp.position.y,
                          sp.position.z, c.x, c.y, c.z, sp.viewCount, sp.observationCount)
        }
        try? csv.write(toFile: path, atomically: true, encoding: .utf8)
    }
}

/// Binary PLY with xyz float + rgb uchar (for inspecting point clouds). colors are 0..1.
private func writePointPLY(_ positions: MLXArray, _ colors: MLXArray, to url: URL) throws {
    let n = positions.shape[0]
    let pos = n > 0 ? positions.asArray(Float.self) : []
    let col = n > 0 ? colors.asArray(Float.self) : []
    let header = "ply\nformat binary_little_endian 1.0\nelement vertex \(n)\n"
        + "property float x\nproperty float y\nproperty float z\n"
        + "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    var data = Data(header.utf8)
    data.reserveCapacity(data.count + n * 15)
    for i in 0..<n {
        for c in 0..<3 { withUnsafeBytes(of: pos[i * 3 + c].bitPattern.littleEndian) { data.append(contentsOf: $0) } }
        for c in 0..<3 { data.append(UInt8(max(0, min(255, (col[i * 3 + c] * 255).rounded())))) }
    }
    try data.write(to: url)
}
