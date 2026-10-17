// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Refines ARKit poses with ORB correspondences + ARKit feature-point tracks + iterative BA (IGR)
// (the global BA stage of PocketGS Sec. III-A). Intrinsics and images are left unchanged.

import Foundation
import simd
import Accelerate
import MLX
import Trainer

public enum CaptureBA {

    /// Builds the fundamental matrix F (row-major 9) from two frames' w2c viewmats and intrinsics.
    /// `x_bᵀ F x_a = 0`; the epipolar line in b is `F · [x_a, y_a, 1]`. F = K_b⁻ᵀ [t]_× R K_a⁻¹.
    static func fundamentalMatrix(
        viewmatA: [Float], viewmatB: [Float],
        intrA: Intrinsics,
        intrB: Intrinsics
    ) -> [Float] {
        func rot(_ vm: [Float]) -> simd_float3x3 {
            simd_float3x3(rows: [
                SIMD3<Float>(vm[0], vm[1], vm[2]),
                SIMD3<Float>(vm[4], vm[5], vm[6]),
                SIMD3<Float>(vm[8], vm[9], vm[10]),
            ])
        }
        func kInv(_ k: Intrinsics) -> simd_float3x3 {
            simd_float3x3(rows: [
                SIMD3<Float>(1 / k.fx, 0, -k.cx / k.fx),
                SIMD3<Float>(0, 1 / k.fy, -k.cy / k.fy),
                SIMD3<Float>(0, 0, 1),
            ])
        }
        let rA = rot(viewmatA), rB = rot(viewmatB)
        let tA = SIMD3<Float>(viewmatA[3], viewmatA[7], viewmatA[11])
        let tB = SIMD3<Float>(viewmatB[3], viewmatB[7], viewmatB[11])
        let rRel = rB * rA.transpose
        let tRel = tB - rRel * tA
        let tx = simd_float3x3(rows: [
            SIMD3<Float>(0, -tRel.z, tRel.y),
            SIMD3<Float>(tRel.z, 0, -tRel.x),
            SIMD3<Float>(-tRel.y, tRel.x, 0),
        ])
        let e = tx * rRel
        let f = kInv(intrB).transpose * e * kInv(intrA)
        // simd is column-major (f[col][row]), so transpose to row-major
        return [
            f[0][0], f[1][0], f[2][0],
            f[0][1], f[1][1], f[2][1],
            f[0][2], f[1][2], f[2][2],
        ]
    }

    /// Picks non-adjacent frame pairs that are spatially close with similar view directions (loop
    /// closure candidates) from ARKit poses. The distance threshold is relative to the orbit
    /// diameter (max pairwise camera-center distance) to avoid scene dependence. |i-j| < minGap is
    /// excluded since consecutive pairs cover it. Returns a < b.
    static func loopClosureCandidates(
        viewmats: [[Float]],
        minGap: Int = 10,
        maxPerFrame: Int = 3,
        maxDistFraction: Float = 0.25,
        maxAngleCos: Float = 0.707
    ) -> [(a: Int, b: Int)] {
        let n = viewmats.count
        guard n >= minGap + 2 else { return [] }
        var centers: [SIMD3<Float>] = []
        var forwards: [SIMD3<Float>] = []
        centers.reserveCapacity(n); forwards.reserveCapacity(n)
        for vm in viewmats {
            centers.append(FrameSubsampler.cameraCenter(viewmat: vm))
            var fwd = SIMD3<Float>(vm[8], vm[9], vm[10])   // Row 3 of w2c R = world view direction
            let len = simd_length(fwd)
            if len > 1e-6 { fwd /= len }
            forwards.append(fwd)
        }
        var diameter: Float = 0
        for a in 0..<n {
            for b in (a + 1)..<n { diameter = max(diameter, simd_distance(centers[a], centers[b])) }
        }
        guard diameter > 1e-6 else { return [] }
        let maxDist = maxDistFraction * diameter
        var out: [(a: Int, b: Int)] = []
        for i in 0..<n {
            let jStart = i + minGap
            guard jStart < n else { continue }
            var cands: [(j: Int, d: Float)] = []
            for j in jStart..<n {
                let d = simd_distance(centers[i], centers[j])
                if d > maxDist { continue }
                if simd_dot(forwards[i], forwards[j]) < maxAngleCos { continue }
                cands.append((j, d))
            }
            cands.sort { $0.d < $1.d }
            for c in cands.prefix(maxPerFrame) { out.append((a: i, b: c.j)) }
        }
        return out
    }

    /// Inverse depth ρ = 1/z of a keypoint (in that frame's camera coordinates) and its standard
    /// deviation
    struct KeypointDepth { let rho: Float; let sigma: Float }

    /// w2c row-major viewmat → (R, camera center C)
    private static func poseRC(_ vm: [Float]) -> (R: simd_float3x3, C: SIMD3<Float>) {
        let R = simd_float3x3(rows: [
            SIMD3<Float>(vm[0], vm[1], vm[2]),
            SIMD3<Float>(vm[4], vm[5], vm[6]),
            SIMD3<Float>(vm[8], vm[9], vm[10]),
        ])
        return (R, -(R.transpose * SIMD3<Float>(vm[3], vm[7], vm[11])))
    }

    /// pixel (center = integer + 0.5) → view ray in camera coordinates (z = 1)
    private static func ray(_ uv: SIMD2<Float>, _ k: Intrinsics) -> SIMD3<Float> {
        SIMD3<Float>((uv.x - k.cx) / k.fx, (uv.y - k.cy) / k.fy, 1)
    }

    /// Triangulates consecutive-frame correspondences with ARKit poses by the midpoint method and
    /// returns per-keypoint inverse depth (`[frame][keypoint index]`, smallest σ if several). σ_ρ =
    /// √2·σpx / f / b⊥ (b⊥ = baseline perpendicular to the view ray). Those with σ_ρ ≥ ρ/2 due to
    /// insufficient parallax are not returned.
    static func keypointDepths(
        matches: [(a: Int, b: Int, ia: Int, ib: Int)],
        features: [ORBPipeline.FrameFeatures],
        frames: [CapturedFrame],
        sigmaPx: Float = 1.5
    ) -> [[Int: KeypointDepth]] {
        var out = [[Int: KeypointDepth]](repeating: [:], count: frames.count)
        let poses = frames.map { poseRC($0.viewmat) }
        func put(_ f: Int, _ idx: Int, _ d: KeypointDepth) {
            if let old = out[f][idx], old.sigma <= d.sigma { return }
            out[f][idx] = d
        }
        for m in matches {
            let (Ra, Ca) = poses[m.a], (Rb, Cb) = poses[m.b]
            let ka = frames[m.a].intrinsics, kb = frames[m.b].intrinsics
            let pa = features[m.a].coords[m.ia], pb = features[m.b].coords[m.ib]
            let da = Ra.transpose * ray(SIMD2<Float>(Float(pa.x) + 0.5, Float(pa.y) + 0.5), ka)
            let db = Rb.transpose * ray(SIMD2<Float>(Float(pb.x) + 0.5, Float(pb.y) + 0.5), kb)
            // min |Ca + s·da − Cb − u·db|². da, db have z = 1 in camera coordinates, so s, u are
            // depths
            let w = Ca - Cb
            let aa = simd_dot(da, da), ab = simd_dot(da, db), bb = simd_dot(db, db)
            let dw = simd_dot(da, w), ew = simd_dot(db, w)
            let den = aa * bb - ab * ab
            guard den > 1e-9 else { continue }
            let s = (ab * ew - bb * dw) / den
            let u = (aa * ew - ab * dw) / den
            guard s > 1e-3, u > 1e-3 else { continue }
            let base = Cb - Ca
            let bPerpA = simd_length(simd_cross(base, simd_normalize(da)))
            let bPerpB = simd_length(simd_cross(base, simd_normalize(db)))
            let sigAng = 1.414 * sigmaPx / ka.fx
            let rA = 1 / s, sA = sigAng / max(bPerpA, 1e-6)
            let rB = 1 / u, sB = sigAng / max(bPerpB, 1e-6)
            if sA < 0.5 * rA { put(m.a, m.ia, KeypointDepth(rho: rA, sigma: sA)) }
            if sB < 0.5 * rB { put(m.b, m.ib, KeypointDepth(rho: rB, sigma: sB)) }
        }
        return out
    }

    /// Returns the frame's depth range (5th-95th percentile of inverse depth widened by 0.8-1.25x)
    /// as a `KeypointDepth` whose ρ±3σ covers it (nil if fewer than minCount keypoints have depth).
    /// Used to check loop correspondences whose keypoint has no depth. In narrow-depth scenes like
    /// walls it is tight enough to reject the neighboring copy of the same pattern.
    static func frameDepthRanges(_ depths: [[Int: KeypointDepth]], minCount: Int = 20) -> [KeypointDepth?] {
        depths.map { d in
            guard d.count >= minCount else { return nil }
            let r = d.values.map(\.rho).sorted()
            let lo = 0.8 * r[Int(0.05 * Float(r.count - 1))]
            let hi = 1.25 * r[Int(0.95 * Float(r.count - 1))]
            return KeypointDepth(rho: 0.5 * (lo + hi), sigma: (hi - lo) / 6)
        }
    }

    struct LoopGateStats {
        var kept = 0, rejected = 0, noDepth = 0, rangeOnly = 0, droppedPairs = 0
        var distances: [Float] = []
        func percentile(_ q: Float) -> Float {
            guard !distances.isEmpty else { return 0 }
            let s = distances.sorted()
            return s[min(s.count - 1, Int(q * Float(s.count - 1) + 0.5))]
        }
    }

    /// Distance [px] between the predicted segment (inverse depth ρ±3σ) where a src keypoint lands
    /// in dst and the corresponding position uvDst. nil if the whole segment is behind the camera.
    /// dst camera coordinates are linear in ρ (A + ρ·B), so ρ is clipped to the z > 0 range and
    /// both ends projected.
    static func predictedDistance(
        uvSrc: SIMD2<Float>, depth: KeypointDepth, uvDst: SIMD2<Float>,
        src: CapturedFrame, dst: CapturedFrame
    ) -> Float? {
        let (Rs, Cs) = poseRC(src.viewmat)
        let (Rd, Cd) = poseRC(dst.viewmat)
        let dir = Rs.transpose * ray(uvSrc, src.intrinsics)
        let A = Rd * dir
        let B = Rd * (Cs - Cd)
        var lo = max(0, depth.rho - 3 * depth.sigma), hi = depth.rho + 3 * depth.sigma
        let zEps: Float = 1e-3
        // Clip to the range where z(ρ) = A.z + ρ·B.z > zEps
        if abs(B.z) < 1e-9 {
            guard A.z > zEps else { return nil }
        } else {
            let r0 = (zEps - A.z) / B.z
            if B.z > 0 { lo = max(lo, r0) } else { hi = min(hi, r0) }
            guard lo <= hi else { return nil }
        }
        let k = dst.intrinsics
        func proj(_ r: Float) -> SIMD2<Float> {
            let p = A + r * B
            return SIMD2<Float>(k.fx * p.x / p.z + k.cx, k.fy * p.y / p.z + k.cy)
        }
        let p0 = proj(lo), p1 = proj(hi)
        let seg = p1 - p0
        let len2 = simd_length_squared(seg)
        let t = len2 > 1e-9 ? max(0, min(1, simd_dot(uvDst - p0, seg) / len2)) : 0
        return simd_distance(uvDst, p0 + t * seg)
    }

    /// Checks loop pair (a, b) correspondences against positions predicted from keypoint depth (a
    /// side first, else b side, else the frame's depth range) and returns only those within radius.
    /// Correspondences that cannot be predicted are dropped.
    /// - Returns: (kept correspondences, count dropped for deviating from the predicted position)
    static func loopDepthGate(
        matches: [ORBPipeline.Match], a: Int, b: Int,
        features: [ORBPipeline.FrameFeatures], depths: [[Int: KeypointDepth]],
        frameRanges: [KeypointDepth?]? = nil,
        frames: [CapturedFrame], radius: Float, stats: inout LoopGateStats
    ) -> (kept: [ORBPipeline.Match], rejected: Int) {
        var kept: [ORBPipeline.Match] = []
        var rejected = 0
        for m in matches {
            let ca = features[a].coords[m.queryIdx], cb = features[b].coords[m.trainIdx]
            let uvA = SIMD2<Float>(Float(ca.x) + 0.5, Float(ca.y) + 0.5)
            let uvB = SIMD2<Float>(Float(cb.x) + 0.5, Float(cb.y) + 0.5)
            var dist: Float?
            if let d = depths[a][m.queryIdx] {
                dist = predictedDistance(uvSrc: uvA, depth: d, uvDst: uvB, src: frames[a], dst: frames[b])
            }
            if dist == nil, let d = depths[b][m.trainIdx] {
                dist = predictedDistance(uvSrc: uvB, depth: d, uvDst: uvA, src: frames[b], dst: frames[a])
            }
            if dist == nil, let fr = frameRanges {
                if let d = fr[a] {
                    dist = predictedDistance(uvSrc: uvA, depth: d, uvDst: uvB, src: frames[a], dst: frames[b])
                }
                if dist == nil, let d = fr[b] {
                    dist = predictedDistance(uvSrc: uvB, depth: d, uvDst: uvA, src: frames[b], dst: frames[a])
                }
                if dist != nil { stats.rangeOnly += 1 }
            }
            guard let dd = dist else { stats.noDepth += 1; continue }
            stats.distances.append(dd)
            if dd <= radius {
                kept.append(m)
                stats.kept += 1
            } else {
                rejected += 1
                stats.rejected += 1
            }
        }
        return (kept, rejected)
    }

    /// One observation of an ARKit feature-point track. Pixel convention as ORB: pixel center =
    /// integer + 0.5.
    struct TrackObservation: Equatable {
        let frame: Int
        let pixel: SIMD2<Float>
    }

    struct ARKitTrackStats: Equatable {
        var candidates = 0     // IDs projected into at least minTrackLength frames
        var lowTexture = 0     // IDs dropped because the reference patch was textureless
        var aligned = 0        // Observations (other than the reference) successfully aligned by NCC
        var rejected = 0       // Observations dropped for NCC below threshold
        var tracks = 0         // Tracks remaining with at least minTrackLength observations
        var observations = 0   // Total observations of the remaining tracks (including the reference)
    }

    /// Builds multi-view tracks using ARKit feature-point IDs only for association; 2D positions
    /// are measured by patch matching (NCC), because ARKit's projections jitter by a few px, and
    /// using them directly as observations would make BA circle back to the ARKit poses. The middle
    /// observation is the reference patch (textureless ones are dropped); other observations are
    /// searched around their projections and refined to sub-pixel.
    /// - Parameter luma: row-major [H·W] luma (0..255) of a frame. Called at most once per frame in
    ///   each of the reference and matching passes, so not all frames are held at once.
    static func measureARKitTracks(
        featureObservations: [[ARKitFeatureObservation]],
        intrinsics: [Intrinsics],
        sizes: [(width: Int, height: Int)],
        luma: (Int) -> [Float],
        patchRadius: Int = 5,
        searchRadius: Int = 8,
        minNCC: Float = 0.8,
        minTexture: Float = 4,
        minTrackLength: Int = 3
    ) -> (tracks: [[TrackObservation]], stats: ARKitTrackStats) {
        let n = featureObservations.count
        precondition(intrinsics.count == n && sizes.count == n)
        var stats = ARKitTrackStats()
        let r = patchRadius
        let side = 2 * r + 1
        let area = side * side
        // Margin so the patch stays inside the image even with search + 1px refinement + 1px for
        // the parabola
        let margin = r + searchRadius + 2

        var byTrack: [UInt64: [(frame: Int, u: Float, v: Float)]] = [:]
        for f in 0..<n {
            let k = intrinsics[f]
            let (w, h) = sizes[f]
            for o in featureObservations[f] {
                let p = o.cameraPoint
                guard p.z > 1e-3 else { continue }
                let u = k.fx * p.x / p.z + k.cx
                let v = k.fy * p.y / p.z + k.cy
                guard u >= Float(margin), v >= Float(margin),
                      u < Float(w - margin), v < Float(h - margin) else { continue }
                byTrack[o.trackID, default: []].append((f, u, v))
            }
        }
        let ids = byTrack.keys.sorted().filter { byTrack[$0]!.count >= minTrackLength }
        stats.candidates = ids.count

        struct Track {
            var obs: [(frame: Int, u: Float, v: Float)]
            var refIndex: Int
            var patch: [Float] = []   // Zero-mean, L2-normalized reference patch
            var refPixel = SIMD2<Float>(0, 0)
            var measured: [TrackObservation] = []
        }
        var tracks: [Track] = ids.map { id in
            let obs = byTrack[id]!.sorted { $0.frame < $1.frame }
            return Track(obs: obs, refIndex: obs.count / 2)
        }

        // Extract reference patches
        var refsByFrame: [Int: [Int]] = [:]
        for (t, tr) in tracks.enumerated() { refsByFrame[tr.obs[tr.refIndex].frame, default: []].append(t) }
        for f in refsByFrame.keys.sorted() {
            let img = luma(f)
            let w = sizes[f].width
            for t in refsByFrame[f]! {
                let o = tracks[t].obs[tracks[t].refIndex]
                let x0 = Int(o.u), y0 = Int(o.v)
                var p = [Float](repeating: 0, count: area)
                var mean: Float = 0
                for dy in -r...r {
                    for dx in -r...r {
                        let val = img[(y0 + dy) * w + (x0 + dx)]
                        p[(dy + r) * side + (dx + r)] = val
                        mean += val
                    }
                }
                mean /= Float(area)
                var ss: Float = 0
                for i in 0..<area { p[i] -= mean; ss += p[i] * p[i] }
                guard (ss / Float(area)).squareRoot() >= minTexture else { continue }
                let inv = 1 / ss.squareRoot()
                for i in 0..<area { p[i] *= inv }
                tracks[t].patch = p
                tracks[t].refPixel = SIMD2<Float>(Float(x0) + 0.5, Float(y0) + 0.5)
            }
        }
        stats.lowTexture = tracks.filter { $0.patch.isEmpty }.count

        // NCC-match the non-reference observations
        var jobsByFrame: [Int: [(track: Int, obs: Int)]] = [:]
        for (t, tr) in tracks.enumerated() where !tr.patch.isEmpty {
            for (i, o) in tr.obs.enumerated() where i != tr.refIndex {
                jobsByFrame[o.frame, default: []].append((t, i))
            }
        }
        for f in jobsByFrame.keys.sorted() {
            let img = luma(f)
            let w = sizes[f].width
            // The reference patch is zero-mean with unit norm, so ncc = Σ p·t / sqrt(Σt² − (Σt)²/N)
            func ncc(_ p: [Float], _ cx: Int, _ cy: Int) -> Float {
                var st: Float = 0, stt: Float = 0, spt: Float = 0
                for dy in -r...r {
                    let row = (cy + dy) * w + cx
                    let prow = (dy + r) * side + r
                    for dx in -r...r {
                        let val = img[row + dx]
                        st += val; stt += val * val; spt += p[prow + dx] * val
                    }
                }
                let varT = stt - st * st / Float(area)
                return varT > 1e-6 ? spt / varT.squareRoot() : -1
            }
            for (t, i) in jobsByFrame[f]! {
                let p = tracks[t].patch
                let o = tracks[t].obs[i]
                let x0 = Int(o.u), y0 = Int(o.v)
                var best: Float = -2
                var bx = x0, by = y0
                for dy in stride(from: -searchRadius, through: searchRadius, by: 2) {
                    for dx in stride(from: -searchRadius, through: searchRadius, by: 2) {
                        let c = ncc(p, x0 + dx, y0 + dy)
                        if c > best { best = c; bx = x0 + dx; by = y0 + dy }
                    }
                }
                let cx0 = bx, cy0 = by
                for dy in -1...1 {
                    for dx in -1...1 where dx != 0 || dy != 0 {
                        let c = ncc(p, cx0 + dx, cy0 + dy)
                        if c > best { best = c; bx = cx0 + dx; by = cy0 + dy }
                    }
                }
                guard best >= minNCC else { stats.rejected += 1; continue }
                // Sub-pixel refinement by independent parabola fits in x and y
                func vertex(_ m: Float, _ c: Float, _ pl: Float) -> Float {
                    let den = m - 2 * c + pl
                    guard den < -1e-6 else { return 0 }
                    return max(-0.5, min(0.5, 0.5 * (m - pl) / den))
                }
                let sx = vertex(ncc(p, bx - 1, by), best, ncc(p, bx + 1, by))
                let sy = vertex(ncc(p, bx, by - 1), best, ncc(p, bx, by + 1))
                stats.aligned += 1
                tracks[t].measured.append(TrackObservation(
                    frame: o.frame,
                    pixel: SIMD2<Float>(Float(bx) + 0.5 + sx, Float(by) + 0.5 + sy)))
            }
        }

        var out: [[TrackObservation]] = []
        for tr in tracks where !tr.patch.isEmpty {
            var obs = tr.measured
            obs.append(TrackObservation(frame: tr.obs[tr.refIndex].frame, pixel: tr.refPixel))
            guard obs.count >= minTrackLength else { continue }
            obs.sort { $0.frame < $1.frame }
            out.append(obs)
            stats.observations += obs.count
        }
        stats.tracks = out.count
        return (out, stats)
    }

    /// Returns the scene with capture poses refined by `IterativeGeometricRefinement`. Returns the
    /// original scene if there are no correspondences.
    /// - Parameter useARKitTracks: if ARKit feature observations exist, add multi-view tracks as BA
    ///   observations.
    /// - Parameter lensDistortion: if enabled, estimate radial lens distortion; if accepted, redo
    ///   BA with undistorted observations and replace the returned frames' images and intrinsics
    ///   with undistorted ones.
    public static func refine(
        scene: CapturedScene,
        parameters: IterativeGeometricRefinement.Parameters = .init(),
        fastThreshold: Float = 0.05,
        loweRatio: Float = 0.8,
        maxKeypointsPerFrame: Int = 1500,
        useARKitTracks: Bool = true,
        lensDistortion: LensDistortionEstimation.Parameters = .disabled,
        log: (String) -> Void = { print($0) }
    ) -> CapturedScene {
        guard scene.frames.count >= 2 else { return scene }
        let n = scene.frames.count
        var features: [ORBPipeline.FrameFeatures] = []
        features.reserveCapacity(n)
        let extractStart = Date()
        for frame in scene.frames {
            let luma = CensusTransform.rgbToLuma(frame.imageFloat)
            features.append(ORBPipeline.extract(
                luma: luma, fastThreshold: fastThreshold, maxKeypoints: maxKeypointsPerFrame
            ))
        }
        let keypointCounts = features.map(\.coords.count)
        log(String(format: "ORB extract: %d frames in %.2fs, keypoints/frame min=%d mean=%.0f max=%d",
                   n, Date().timeIntervalSince(extractStart), keypointCounts.min() ?? 0,
                   Double(keypointCounts.reduce(0, +)) / Double(n), keypointCounts.max() ?? 0))
        // Matching of consecutive frame pairs
        struct PairMatch {
            let camA: Int
            let camB: Int
            let uvA: SIMD2<Float>
            let uvB: SIMD2<Float>
        }
        var pairs: [PairMatch] = []
        pairs.reserveCapacity(n * 200)
        // Keypoint indices of consecutive-frame correspondences (for the depth gate on loop
        // correspondences)
        var consecutiveKeypointMatches: [(a: Int, b: Int, ia: Int, ib: Int)] = []
        // Remove out-of-band candidates before Hamming with the epipolar band from the ARKit-pose F
        // (speedup + outlier removal). The band is wide enough to absorb ARKit rotation noise (a
        // few degrees). Pairs with too few correspondences fall back to unguided.
        let longSide = Float(max(scene.frames[0].width, scene.frames[0].height))
        let epiBandPx = max(24.0, 0.08 * longSide)
        let minGuidedMatches = 8
        var guidedFallbacks = 0
        // Log guided / unguided correspondence counts for some pairs to check the band keeps true
        // matches
        let diagPairs: Set<Int> = Set([0, (n - 1) / 2, n - 2].filter { $0 >= 0 && $0 < n - 1 })
        var epipolarRetain: [String] = []
        let matchStart = Date()
        // Match all pairs together (no GPU wait per pair). Unbanded matching runs only for the
        // diagnostic pairs and for pairs whose banded correspondences were too few and fall back to
        // unbanded.
        func guide(_ i: Int, _ j: Int) -> ORBPipeline.EpipolarGuide {
            let fa = scene.frames[i], fb = scene.frames[j]
            let fMat = fundamentalMatrix(
                viewmatA: fa.viewmat, viewmatB: fb.viewmat,
                intrA: fa.intrinsics, intrB: fb.intrinsics
            )
            return ORBPipeline.EpipolarGuide(fundamentalAB: fMat, bandPx: epiBandPx)
        }
        let guided = ORBPipeline.matchBatch(
            (0..<(n - 1)).map { (a: features[$0], b: features[$0 + 1], epipolar: guide($0, $0 + 1)) },
            loweRatio: loweRatio)
        let unguidedPairs = (0..<(n - 1)).filter { diagPairs.contains($0) || guided[$0].count < minGuidedMatches }
        let unguided = Dictionary(uniqueKeysWithValues: zip(unguidedPairs, ORBPipeline.matchBatch(
            unguidedPairs.map { (a: features[$0], b: features[$0 + 1], epipolar: nil) }, loweRatio: loweRatio)))
        for i in 0..<(n - 1) {
            var ms = guided[i]
            let guidedCount = ms.count
            // If retain% is extremely low, suspect the sign of F or the band width
            if diagPairs.contains(i), let u = unguided[i] {
                let retain = u.count > 0 ? 100.0 * Double(guidedCount) / Double(u.count) : 0
                epipolarRetain.append(String(format: "pair[%d] %d/%d=%.0f%%", i, guidedCount, u.count, retain))
            }
            if ms.count < minGuidedMatches {
                ms = unguided[i]!
                guidedFallbacks += 1
            }
            for m in ms {
                let pa = features[i].coords[m.queryIdx]
                let pb = features[i + 1].coords[m.trainIdx]
                pairs.append(PairMatch(
                    camA: i, camB: i + 1,
                    uvA: SIMD2<Float>(Float(pa.x) + 0.5, Float(pa.y) + 0.5),
                    uvB: SIMD2<Float>(Float(pb.x) + 0.5, Float(pb.y) + 0.5)
                ))
                consecutiveKeypointMatches.append((i, i + 1, m.queryIdx, m.trainIdx))
            }
        }

        // Loop closure pairs: the consecutive chain alone does not constrain the orbit's start <->
        // end, so drift does not close; verify nearby non-adjacent frame pairs with the same
        // matching and add them.
        let consecutiveMatchCount = pairs.count
        let loopCandidates = loopClosureCandidates(viewmats: scene.frames.map { $0.viewmat })
        // With repeated patterns (e.g. identical posters in a row) matches jump to the neighboring
        // copy along the epipolar line, and these wrong matches are mutually consistent, so IGR
        // outlier removal cannot remove them; BA then twists poses by up to 17° and the pattern
        // doubles (2026-10-08 Conan poster). Predict the position in the loop partner from
        // consecutive-frame depth (or the frame's depth range) and drop distant matches. Loop pairs
        // never fall back to unguided.
        let depths = loopCandidates.isEmpty ? [] : keypointDepths(
            matches: consecutiveKeypointMatches, features: features, frames: scene.frames)
        let frameRanges = frameDepthRanges(depths)
        let gateRadius = max(16, 0.04 * longSide)
        var loopFramePairs = 0
        var gateStats = LoopGateStats()
        let loopRaw = ORBPipeline.matchBatch(
            loopCandidates.map { (a: features[$0.0], b: features[$0.1], epipolar: guide($0.0, $0.1)) },
            loweRatio: loweRatio)
        for ((i, j), raw) in zip(loopCandidates, loopRaw) {
            let (ms, rejected) = loopDepthGate(
                matches: raw, a: i, b: j, features: features, depths: depths,
                frameRanges: frameRanges,
                frames: scene.frames, radius: gateRadius, stats: &gateStats)
            // Wrong loop matches easily break BA, so accept only pairs with a minimum number of
            // consistent matches where a majority of the checkable matches are consistent (pairs
            // mixing up the pattern are dropped entirely)
            guard ms.count >= minGuidedMatches, ms.count >= rejected else {
                gateStats.droppedPairs += 1
                continue
            }
            loopFramePairs += 1
            for m in ms {
                let pa = features[i].coords[m.queryIdx]
                let pb = features[j].coords[m.trainIdx]
                pairs.append(PairMatch(
                    camA: i, camB: j,
                    uvA: SIMD2<Float>(Float(pa.x) + 0.5, Float(pa.y) + 0.5),
                    uvB: SIMD2<Float>(Float(pb.x) + 0.5, Float(pb.y) + 0.5)
                ))
            }
        }
        let loopCorrespondences = pairs.count - consecutiveMatchCount
        log(String(format: "loop closure: %d candidate(s) → %d verified frame-pair(s), +%d correspondences",
                   loopCandidates.count, loopFramePairs, loopCorrespondences))
        log(String(format: "loop depth gate (r=%.0fpx): matches kept %d / rejected %d / no depth %d "
                  + "(frame depth range only %d); dropped pairs %d; distance p50 %.1f p90 %.1f max %.1f px",
                   gateRadius, gateStats.kept, gateStats.rejected, gateStats.noDepth, gateStats.rangeOnly,
                   gateStats.droppedPairs, gateStats.percentile(0.5), gateStats.percentile(0.9),
                   gateStats.percentile(1.0)))

        let matchSec = Date().timeIntervalSince(matchStart)
        log(String(format: "ORB match: %d consecutive inlier pairs across %d pair(s) in %.2fs "
                  + "(epipolar band=%dpx, unguided fallback=%d/%d pairs; total incl. loop=%d); "
                  + "epipolar retain guided/unguided %@",
                   consecutiveMatchCount, n - 1, matchSec, Int(epiBandPx), guidedFallbacks, n - 1,
                   pairs.count, epipolarRetain.joined(separator: " ") as NSString))
        // ORB correspondences constrain only 2 views, so add multi-view ARKit feature tracks to
        // strengthen BA
        var arkitTracks: [[TrackObservation]] = []
        if useARKitTracks, scene.frames.contains(where: { !$0.featureObservations.isEmpty }) {
            let trackStart = Date()
            let built = measureARKitTracks(
                featureObservations: scene.frames.map(\.featureObservations),
                intrinsics: scene.frames.map(\.intrinsics),
                sizes: scene.frames.map { (width: $0.width, height: $0.height) },
                luma: { f in
                    // BT.601 luma (0..255). Generated per frame and discarded
                    let rgb = (scene.frames[f].imageFloat * 255).asArray(Float.self)
                    var out = [Float](repeating: 0, count: rgb.count / 3)
                    for i in out.indices {
                        out[i] = 0.299 * rgb[i * 3] + 0.587 * rgb[i * 3 + 1] + 0.114 * rgb[i * 3 + 2]
                    }
                    return out
                }
            )
            arkitTracks = built.tracks
            let st = built.stats
            let maxLen = arkitTracks.map(\.count).max() ?? 0
            log(String(format: "ARKit tracks: %d track(s), %d obs (mean len %.1f, max %d) in %.2fs; "
                      + "candidates=%d lowTexture=%d aligned=%d rejected=%d",
                       st.tracks, st.observations,
                       st.tracks > 0 ? Double(st.observations) / Double(st.tracks) : 0, maxLen,
                       Date().timeIntervalSince(trackStart),
                       st.candidates, st.lowTexture, st.aligned, st.rejected))
        }
        if pairs.isEmpty && arkitTracks.isEmpty {
            log("warning: 0 ORB match — keeping ARKit poses untouched")
            return scene
        }
        // Point layout: [ORB 2-view pairs | ARKit multi-view tracks]. IGR runs BA first, so points
        // are triangulated with DLT beforehand (initializing at the same point makes BA round 0
        // ill-conditioned and the cameras diverge).
        let nPoints = pairs.count + arkitTracks.count
        let intr: [Intrinsics] = scene.frames.map { $0.intrinsics }
        let viewmats: [[Float]] = scene.frames.map { $0.viewmat }

        var observations: [BundleAdjustment.Observation] = []
        observations.reserveCapacity(pairs.count * 2 + arkitTracks.reduce(0) { $0 + $1.count })
        var firstCam: [Int] = []   // First observing camera of each point (for diagnostics)
        firstCam.reserveCapacity(nPoints)
        for (j, p) in pairs.enumerated() {
            observations.append(BundleAdjustment.Observation(
                cameraIndex: p.camA, pointIndex: j, pixel: p.uvA
            ))
            observations.append(BundleAdjustment.Observation(
                cameraIndex: p.camB, pointIndex: j, pixel: p.uvB
            ))
            firstCam.append(p.camA)
        }
        for (t, track) in arkitTracks.enumerated() {
            let j = pairs.count + t
            for o in track {
                observations.append(BundleAdjustment.Observation(
                    cameraIndex: o.frame, pointIndex: j, pixel: o.pixel
                ))
            }
            firstCam.append(track[0].frame)
        }

        // Points with 2+ observations are solved by DLT. State points are zero-filled for API
        // reasons.
        let zeroPoints = [SIMD3<Float>](repeating: SIMD3<Float>(0, 0, 0), count: nPoints)
        let state0Centroid = BundleAdjustment.State(viewmats: viewmats, points: zeroPoints)
        let triangulatedPoints = IterativeGeometricRefinement.retriangulate(
            state: state0Centroid, intrinsics: intr, observations: observations
        )
        // Diagnostics: number of points in front of their first observing camera
        var validTriangulated = 0
        var behindCount = 0
        for (j, p) in triangulatedPoints.enumerated() {
            let vm = viewmats[firstCam[j]]
            let z = vm[8] * p.x + vm[9] * p.y + vm[10] * p.z + vm[11]
            if z > 1e-3 { validTriangulated += 1 } else { behindCount += 1 }
        }
        log("CaptureBA pre-triangulate: \(validTriangulated)/\(nPoints) points in front of 1st cam "
          + "(behind=\(behindCount))")

        let state0 = BundleAdjustment.State(viewmats: viewmats, points: triangulatedPoints)
        let initialResult = IterativeGeometricRefinement.refine(
            state: state0, intrinsics: intr, observations: observations,
            parameters: parameters, log: log
        )
        // Lens distortion: if the estimate is accepted, redo BA from the ARKit poses with
        // undistorted observations. Not applied on top of frames that are already undistorted.
        var lensDistortion = lensDistortion
        if lensDistortion.enabled, scene.frames.contains(where: { $0.lensUndistortion != nil }) {
            log("lens distortion: frames are already undistorted — skipping estimation")
            lensDistortion = .disabled
        }
        let (result, distortion) = LensDistortionEstimation.refineAlternating(
            initial: initialResult, intrinsics: intr,
            sizes: scene.frames.map { (width: $0.width, height: $0.height) },
            observations: observations, parameters: lensDistortion,
            solve: { obs in
                let points = IterativeGeometricRefinement.retriangulate(
                    state: state0Centroid, intrinsics: intr, observations: obs)
                return IterativeGeometricRefinement.refine(
                    state: BundleAdjustment.State(viewmats: viewmats, points: points),
                    intrinsics: intr, observations: obs, parameters: parameters, log: log)
            },
            log: log)

        // Cameras left with no observations are unconstrained in BA and undefined, so revert them
        // to ARKit poses
        let observedCams: Set<Int> = Set(result.observations.map { $0.cameraIndex })
        // 5a. BA fixes only the first camera, so global scale, orientation and position can drift
        //     from ARKit (gauge freedom). Align the post-BA camera centers to the ARKit camera
        //     centers with a similarity transform and apply it to poses and points. Relative
        //     accuracy is unchanged, and ARKit's metric scale and frame are kept (same frame as raw
        //     ARKit points and LiDAR points).
        var refinedState = result.state
        let camIdx = observedCams.sorted()
        if camIdx.count >= 3,
           let sim = umeyamaSimilarity(
               src: camIdx.map { SIMD3<Double>(FrameSubsampler.cameraCenter(viewmat: result.state.viewmats[$0])) },
               dst: camIdx.map { SIMD3<Double>(FrameSubsampler.cameraCenter(viewmat: scene.frames[$0].viewmat)) })
        {
            refinedState = applySimilarity(sim, to: result.state)
            let ang = acos(max(-1, min(1, (sim.R[0][0] + sim.R[1][1] + sim.R[2][2] - 1) / 2))) * 180 / .pi
            log(String(format: "BA→ARKit similarity: scale=%.4f rot=%.3f° trans=%.4f m (center RMS after align %.4f m)",
                       sim.s, ang, simd_length(sim.t), sim.rms))
        }
        // Cameras without observations do not move in BA, so left at ARKit they disagree with the
        // corrected neighbors (off by the drift correction). Apply the same correction W =
        // V_ARKitᵀ·V_BA (a rigid world transform) received by the temporally nearest observed
        // camera, placing it in the corrected world while keeping ARKit's local relative pose.
        let observedSorted = camIdx
        var carried = 0
        func carriedViewmat(_ i: Int) -> [Float] {
            guard !observedSorted.isEmpty else { return scene.frames[i].viewmat }
            var nb = observedSorted[0]
            for c in observedSorted where abs(c - i) < abs(nb - i) { nb = c }
            let Va = mat4(scene.frames[nb].viewmat), Vb = mat4(refinedState.viewmats[nb])
            let W = Va.inverse * Vb                       // world → world
            carried += 1
            return rowMajor(mat4(scene.frames[i].viewmat) * W)
        }
        var refinedFrames: [CapturedFrame] = []
        refinedFrames.reserveCapacity(n)
        for (i, f) in scene.frames.enumerated() {
            let vm = observedCams.contains(i) ? refinedState.viewmats[i] : carriedViewmat(i)
            refinedFrames.append(f.withViewmat(vm))
        }
        if carried > 0 {
            log("CaptureBA: carried neighbor correction to \(carried) unobserved camera(s)")
        }
        // BA points and raw ARKit points cover complementary regions, so both go into sparsePoints.
        // IGR does not compact points when dropping outlier observations, so pick only points with
        // remaining observations.
        let activePointSet: Set<Int> = Set(result.observations.map { $0.pointIndex })
        let baRefinedPoints: [SIMD3<Float>] = refinedState.points.enumerated().compactMap {
            (i, p) in activePointSet.contains(i) ? p : nil
        }
        let combinedSparse = baRefinedPoints + scene.sparsePoints
        log("CaptureBA refined: \(result.rounds.count) IGR rounds, " +
            "final obs \(result.observations.count) of \(observations.count), " +
            "BA-touched cams \(observedCams.count)/\(n) (rest restored to ARKit pose), " +
            "sparse points = BA \(baRefinedPoints.count) + ARKit \(scene.sparsePoints.count) = \(combinedSparse.count), " +
            "MVS depth-range = BA \(baRefinedPoints.count) only, " +
            "lidar = \(scene.lidarPoints.count) (passthrough)")
        if !distortion.isNone {
            refinedFrames = undistortFrames(refinedFrames, distortion: distortion, log: log)
        }
        // lidarPoints stay as unprojected with capture-time poses and intrinsics that ignore
        // distortion. The offset from BA corrections (cm-level) and distortion (a few px at the
        // image edge) is tolerated.
        return CapturedScene(
            frames: refinedFrames,
            sparsePoints: combinedSparse,
            baRefinedPoints: baRefinedPoints,
            lidarPoints: scene.lidarPoints,
            diagnostics: scene.diagnostics
        )
    }

    /// Replaces intrinsics with the undistorted ones (`LensUndistortion.undistortedIntrinsics`) and
    /// marks images to be undistorted on each use (so the captured images are not resident twice).
    /// Poses and feature observations (3D points in camera coordinates) remain valid.
    static func undistortFrames(
        _ frames: [CapturedFrame], distortion: RadialDistortion, log: (String) -> Void
    ) -> [CapturedFrame] {
        var scales: [Float] = []
        var fallback = 0
        let out: [CapturedFrame] = frames.map { f in
            // Found if the distortion estimate passed acceptance (corner correction ≤ 5%,
            // monotonic). If not, only undistort without changing K (edges become stretched edge
            // pixels, but the geometry stays consistent with BA).
            let target: Intrinsics
            if let k = LensUndistortion.undistortedIntrinsics(
                f.intrinsics, width: f.width, height: f.height, distortion: distortion) {
                target = k
            } else {
                target = f.intrinsics
                fallback += 1
            }
            scales.append(target.fx / f.intrinsics.fx)
            return f.undistorted(target: target, distortion: distortion)
        }
        log(String(format: "lens undistort: %d frame(s), focal scale %.4f–%.4f%@",
                   out.count, scales.min() ?? 1, scales.max() ?? 1,
                   fallback > 0 ? " (\(fallback) frame(s) kept original focal: no border-free crop)" : ""))
        return out
    }

    /// Similarity transform dst ≈ s·R·src + t (Umeyama 1991). nil if fewer than 3 points or
    /// degenerate (zero src variance).
    struct Similarity { let s: Double; let R: simd_double3x3; let t: SIMD3<Double>; let rms: Double }

    static func umeyamaSimilarity(src: [SIMD3<Double>], dst: [SIMD3<Double>]) -> Similarity? {
        let n = src.count
        guard n >= 3, n == dst.count else { return nil }
        let muS = src.reduce(SIMD3<Double>(repeating: 0), +) / Double(n)
        let muD = dst.reduce(SIMD3<Double>(repeating: 0), +) / Double(n)
        // Build cov = (1/n) Σ (d - μd)(s - μs)ᵀ as a column-major 3×3 (for LAPACK)
        var cov = [Double](repeating: 0, count: 9)
        var varS = 0.0
        for k in 0..<n {
            let a = src[k] - muS, b = dst[k] - muD
            varS += simd_dot(a, a)
            for c in 0..<3 { for r in 0..<3 { cov[c * 3 + r] += b[r] * a[c] } }
        }
        varS /= Double(n)
        guard varS > 1e-12 else { return nil }
        for k in 0..<9 { cov[k] /= Double(n) }
        // SVD: cov = U Σ Vᵀ
        var jobu = Int8(UInt8(ascii: "A")), jobvt = Int8(UInt8(ascii: "A"))
        var m: __CLPK_integer = 3, nn: __CLPK_integer = 3, lda: __CLPK_integer = 3
        var ldu: __CLPK_integer = 3, ldvt: __CLPK_integer = 3, info: __CLPK_integer = 0
        var sv = [Double](repeating: 0, count: 3)
        var u = [Double](repeating: 0, count: 9), vt = [Double](repeating: 0, count: 9)
        var lwork: __CLPK_integer = 64
        var work = [Double](repeating: 0, count: Int(lwork))
        var a = cov
        dgesvd_(&jobu, &jobvt, &m, &nn, &a, &lda, &sv, &u, &ldu, &vt, &ldvt, &work, &lwork, &info)
        guard info == 0 else { return nil }
        // column-major → simd (simd_double3x3 is built from column vectors)
        let U = simd_double3x3(columns: (SIMD3(u[0], u[1], u[2]), SIMD3(u[3], u[4], u[5]), SIMD3(u[6], u[7], u[8])))
        let VT = simd_double3x3(columns: (SIMD3(vt[0], vt[1], vt[2]), SIMD3(vt[3], vt[4], vt[5]), SIMD3(vt[6], vt[7], vt[8])))
        var S = simd_double3x3(diagonal: SIMD3(1, 1, 1))
        if simd_determinant(U) * simd_determinant(VT) < 0 { S[2][2] = -1 }
        let R = U * S * VT
        let s = (sv[0] * S[0][0] + sv[1] * S[1][1] + sv[2] * S[2][2]) / varS
        let t = muD - s * (R * muS)
        var se = 0.0
        for k in 0..<n { se += simd_length_squared(dst[k] - (s * (R * src[k]) + t)) }
        return Similarity(s: s, R: R, t: t, rms: (se / Double(n)).squareRoot())
    }

    /// Pose (row-major world→camera) and points after applying X' = s·R·X + t to world coordinates.
    /// Scaling camera coordinates by s leaves projection unchanged, so R_v' = R_v·Rᵀ, t_v' = s·t_v
    /// − R_v·Rᵀ·t.
    static func applySimilarity(_ sim: Similarity, to state: BundleAdjustment.State) -> BundleAdjustment.State {
        let RT = sim.R.transpose
        let viewmats: [[Float]] = state.viewmats.map { v in
            let Rv = simd_double3x3(rows: [
                SIMD3(Double(v[0]), Double(v[1]), Double(v[2])),
                SIMD3(Double(v[4]), Double(v[5]), Double(v[6])),
                SIMD3(Double(v[8]), Double(v[9]), Double(v[10])),
            ])
            let tv = SIMD3(Double(v[3]), Double(v[7]), Double(v[11]))
            let Rn = Rv * RT
            let tn = sim.s * tv - Rn * sim.t
            return [
                Float(Rn[0][0]), Float(Rn[1][0]), Float(Rn[2][0]), Float(tn.x),
                Float(Rn[0][1]), Float(Rn[1][1]), Float(Rn[2][1]), Float(tn.y),
                Float(Rn[0][2]), Float(Rn[1][2]), Float(Rn[2][2]), Float(tn.z),
                0, 0, 0, 1,
            ]
        }
        let points: [SIMD3<Float>] = state.points.map { p in
            let q = sim.s * (sim.R * SIMD3<Double>(p)) + sim.t
            return SIMD3<Float>(q)
        }
        return BundleAdjustment.State(viewmats: viewmats, points: points)
    }

    /// Row-major 16 elements → simd_double4x4.
    static func mat4(_ v: [Float]) -> simd_double4x4 {
        // Built row by row: the one-expression form takes Xcode 26's type checker too long.
        func row(_ r: Int) -> SIMD4<Double> {
            let b = r * 4
            return SIMD4<Double>(Double(v[b]), Double(v[b + 1]), Double(v[b + 2]), Double(v[b + 3]))
        }
        return simd_double4x4(rows: [row(0), row(1), row(2), row(3)])
    }
    static func rowMajor(_ m: simd_double4x4) -> [Float] {
        (0..<4).flatMap { r in (0..<4).map { c in Float(m[c][r]) } }
    }
}
