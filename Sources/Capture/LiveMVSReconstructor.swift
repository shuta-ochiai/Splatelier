// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// MVS source for the live point cloud during capture (LiveReconstructionMode.mvs). For each
// incoming keyframe, computes one depth map with StreamingMVS (same cost mode as the app's MVS
// mode, configured not to wait for future frames), checks cross-view agreement, and returns the
// confirmed points as coverage splats.
//
// 2026-10-08 Mac replay (`live-mvs-replay`, Apple M3): 35-40ms per keyframe (GPU depth 25ms + CPU
// agreement check 10ms); capture runs at ~6 keyframes/s. 2cm agreement with the post-capture final
// MVS (BA poses) is 96-97% on poster/desk, 87% on rebar.
//
// GPU work runs on a background serial queue. Keyframes arriving while `maxBacklog` are pending are
// not passed to MVS (only reduces reference candidates; nothing breaks). Feature points are passed
// in full without thinning. Poses use ARKit's live values as is (coarser than post-capture BA, but
// enough for live display).

import Foundation
import simd
import MLX
import Trainer

/// Mutable state is touched only under `lock` (backlog / stopped) or on the serial `queue` (engine
/// / accumulator / stats), hence @unchecked Sendable.
public final class LiveMVSReconstructor: @unchecked Sendable {
    private let queue = DispatchQueue(label: "live-mvs", qos: .userInitiated)
    private let lock = NSLock()
    private var backlog = 0
    private var stopped = false
    private var engine: StreamingMVS
    private var accumulator: CaptureCoverageAccumulator
    private let targetLongSide: Int
    private let maxBacklog: Int
    private let onSplats: (([CaptureCoverageSplat]) -> Void)?
    private let onSurfels: (([CaptureSurfelMap.Surfel]) -> Void)?
    private var surfelMap: CaptureSurfelMap
    /// MVS-resolution intrinsics and viewmats of keyframes passed to the engine (in the engine's
    /// frame index order).
    private var engineFrames: [(intrinsics: Intrinsics, viewmat: [Float])] = []
    // Stats touched only on queue (for the diagnostics log at stop).
    private var processed = 0
    private var dropped = 0   // frames not passed because of backlog + discarded at stop
    private var busySeconds: Double = 0
    private var surfelSeconds: Double = 0   // surfel map updates (carve, insert, snapshot)
    private var coverageSeconds: Double = 0 // coverage accumulator insert + snapshot (with onSplats)

    /// - Parameters:
    ///   - parameters: MVS parameters (default matches the app's MVS mode).
    ///   - targetLongSide: target long side (px) of the MVS input. Applies the same box downscale
    ///     as post-capture MVS.
    ///   - maxBacklog: once this many frames are pending, new keyframes are not passed to MVS.
    ///   - accumulator: display voxels. Default is a 2cm display (mean of actual colors) + "how
    ///     many keyframes it was reconstructed from" (viewCount) counted on a 6cm grid. Counting at
    ///     6cm prevents single-point 2cm voxels floating slightly off the surface from being
    ///     counted as 1 view and scattering across the surface (2026-10-08 replay: counting at 2cm,
    ///     1-2 view voxels were 33-54% and scattered over the whole surface → at 6cm 6-9%, gathered
    ///     at the edges of the captured area). Beyond 80k the display grid is coarsened to show the
    ///     whole captured area, and the coarse display hides cells with fewer than 3 points
    ///     (isolated points floating off the surface).
    ///   - onSplats: if passed, returns the accumulated coverage splats whenever points are added
    ///     (called from the background queue). Without it the accumulator is not filled (the app's
    ///     capture view draws surfels only).
    ///   - surfelMap: surfel map that receives points when `onSurfels` is passed.
    ///   - onSurfels: if passed, also returns surfels (for the splat display that sharpens as you
    ///     get closer) whenever points are added (called from the background queue). Coarse surfels
    ///     floating in front of surfaces captured up close are removed before returning.
    public init(
        parameters: CostVolumeMVS.Parameters = CaptureTraining.MVSMode.mvsParameters,
        targetLongSide: Int = CaptureTraining.mvsTargetLongSide,
        maxBacklog: Int = 2,
        accumulator: CaptureCoverageAccumulator
            = CaptureCoverageAccumulator(
                voxelSizeMeters: 0.02, maxSplats: 80_000, viewVoxelSizeMeters: 0.06,
                minPointsWhenCoarse: 3),
        onSplats: (([CaptureCoverageSplat]) -> Void)? = nil,
        surfelMap: CaptureSurfelMap = CaptureSurfelMap(),
        onSurfels: (([CaptureSurfelMap.Surfel]) -> Void)? = nil
    ) {
        self.engine = StreamingMVS(
            parameters: .init(mvs: parameters, lookahead: 0, fusionDelay: 0),
            retainsCloud: false)
        engine.emitsSurfels = onSurfels != nil
        self.surfelMap = surfelMap
        self.onSurfels = onSurfels
        self.targetLongSide = targetLongSide
        self.maxBacklog = maxBacklog
        self.accumulator = accumulator
        self.onSplats = onSplats
    }

    /// Passes one keyframe (does not block the calling thread).
    /// - Parameters:
    ///   - rgb: RGB row-major `[width*height*3]`.
    ///   - intrinsics: (fx, fy, cx, cy) relative to the `rgb` resolution.
    ///   - cameraToWorld: ARKit camera-to-world.
    ///   - newSparsePoints: ARKit feature points (world) first seen in this frame. Used for depth
    ///     range and reference selection.
    public func submit(
        rgb: [UInt8], width: Int, height: Int,
        intrinsics: Intrinsics,
        cameraToWorld: simd_float4x4, newSparsePoints: [SIMD3<Float>]
    ) {
        lock.lock()
        if stopped { lock.unlock(); return }
        let accept = backlog < maxBacklog
        if accept { backlog += 1 }
        lock.unlock()
        let viewmat = PoseConversion.viewmat(fromCameraToWorld: cameraToWorld)
        queue.async { [self] in
            if !newSparsePoints.isEmpty { engine.addSparsePoints(newSparsePoints) }
            guard accept else { dropped += 1; return }
            defer {
                lock.lock(); backlog -= 1; lock.unlock()
            }
            if isStopped { dropped += 1; return }   // discard work still pending at stop
            let t0 = Date()
            let s = max(1, (max(width, height) + targetLongSide - 1) / targetLongSide)
            let image = StreamingMVS.boxDownscale(MLXArray(rgb, [height, width, 3]), factor: s)
            let sf = Float(s)
            let mvsIntrinsics: Intrinsics = (fx: intrinsics.fx / sf, fy: intrinsics.fy / sf,
                                             cx: intrinsics.cx / sf, cy: intrinsics.cy / sf)
            engineFrames.append((intrinsics: mvsIntrinsics, viewmat: viewmat))
            let step = engine.addFrame(image: image, viewmat: viewmat, intrinsics: mvsIntrinsics)
            processed += 1
            busySeconds += Date().timeIntervalSince(t0)
            guard !step.positions.isEmpty else { return }
            if let onSurfels {
                let t1 = Date()
                let surfels = updateSurfels(step)
                surfelSeconds += Date().timeIntervalSince(t1)
                onSurfels(surfels)
            }
            guard let onSplats else { return }
            let t2 = Date()
            // For each keyframe that produced points, insert position, color and frame index into
            // the voxels (the frame index lets each voxel count "how many keyframes it was
            // reconstructed from").
            let p = step.positions, c = step.colors
            var offset = 0
            for (frame, n) in zip(step.fusedFrames, step.fusedPointCounts) where n > 0 {
                var world: [SIMD3<Float>] = []
                var rgb: [SIMD3<Float>] = []
                world.reserveCapacity(n)
                rgb.reserveCapacity(n)
                for j in offset..<(offset + n) {
                    world.append(SIMD3<Float>(p[j * 3], p[j * 3 + 1], p[j * 3 + 2]))
                    rgb.append(SIMD3<Float>(c[j * 3], c[j * 3 + 1], c[j * 3 + 2]))
                }
                accumulator.insert(world, timestamp: 0, colors: rgb, frameIndex: frame)
                offset += n
            }
            let splats = accumulator.snapshot()
            coverageSeconds += Date().timeIntervalSince(t2)
            onSplats(splats)
        }
    }

    /// Updates the surfel map with the agreement-checked points and returns the surfels to display
    /// (on queue).
    private func updateSurfels(_ step: StreamingMVS.Step) -> [CaptureSurfelMap.Surfel] {
        for d in step.fusedDepths {
            let f = engineFrames[d.frame]
            surfelMap.carve(depth: d.depth, width: d.width, height: d.height,
                            intrinsics: f.intrinsics, viewmat: f.viewmat)
        }
        let p = step.positions, nrm = step.normals, c = step.colors
        var offset = 0
        for (frame, n) in zip(step.fusedFrames, step.fusedPointCounts) where n > 0 {
            let center = FrameSubsampler.cameraCenter(viewmat: engineFrames[frame].viewmat)
            for j in offset..<(offset + n) {
                surfelMap.insert(
                    position: SIMD3(p[j * 3], p[j * 3 + 1], p[j * 3 + 2]),
                    normal: SIMD3(nrm[j * 3], nrm[j * 3 + 1], nrm[j * 3 + 2]),
                    color: SIMD3(c[j * 3], c[j * 3 + 1], c[j * 3 + 2]),
                    pixelSize: step.pixelSizes[j], cameraCenter: center, frame: frame)
            }
            offset += n
        }
        return surfelMap.snapshot()
    }

    /// Ignores further submits and waits until in-flight and pending work finishes (so MLX is not
    /// used concurrently with post-capture preprocessing/training). Returns the final stats.
    @discardableResult
    public func stopAndWait() -> String {
        lock.lock(); stopped = true; lock.unlock()
        return queue.sync {
            var line = String(format: "live MVS: %d keyframes processed (%.0f ms avg), %d skipped, %d points",
                              processed, processed > 0 ? 1000 * busySeconds / Double(processed) : 0,
                              dropped, engine.pointCount)
            if onSurfels != nil {
                line += String(format: ", surfels %.1f ms avg (%d cells)",
                               processed > 0 ? 1000 * surfelSeconds / Double(processed) : 0,
                               surfelMap.cellCounts.reduce(0, +))
            }
            if onSplats != nil {
                line += String(format: ", coverage %.1f ms avg",
                               processed > 0 ? 1000 * coverageSeconds / Double(processed) : 0)
            }
            return line
        }
    }

    private var isStopped: Bool {
        lock.lock(); defer { lock.unlock() }
        return stopped
    }
}
