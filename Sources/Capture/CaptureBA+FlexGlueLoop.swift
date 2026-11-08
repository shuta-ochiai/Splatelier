// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Wide-angle loop closure with FlexGlue (experimental). The regular loop candidates need view
// directions within 45° and ORB matches, so passes that see a place from very different sides (the
// two faces of a double-sided sign, met at its ends) are never tied together, and ARKit's drift
// between them stays in the poses as a doubled structure. Here:
// - candidates: frame pairs at least `minGap` apart, view directions at least `minAngleDeg` apart,
//   whose frustums contain a Jaccard share >= `minShare` of the same ARKit points (occlusion is
//   ignored, which also tolerates ARKit's drift between the passes; the overlap of the points each
//   frame observed does not), largest share first, each frame in at most `perFrame` pairs, no two
//   pairs within `spacing` frames of each other on both sides, at most `maxPairs`;
// - keypoints: the strongest FAST corner of each cell of a grid (about `cells` per image): the
//   queries have to cover the whole image, the strongest corners overall bunch up on texture;
// - matching: FlexGlueMatcher; kept when the covisibility score is >= `minCovisibility` and a
//   relative pose within `maxRotationDeg` of ARKit's explains >= `minInliers` matches;
// - output: up to `maxPerPair` inlier matches per pair and the fitted pose, for a pose graph edge.

import CoreML
import Foundation
import MLX
import simd
import Trainer

extension CaptureBA {

    struct FlexGlueLoopParameters {
        var directory: URL
        var minShare: Float = 0.1
        var minAngleDeg: Float = 45
        var minGap = 30
        var perFrame = 2
        var spacing = 8
        var maxPairs = 150
        var maxDepth: Float = 4
        var cells = 2048
        var fastThreshold: Float = 0.02
        var minCovisibility: Float = 0.8
        var minInliers = 30
        var maxRotationDeg: Double = 5
        /// The pose is fitted within this of ARKit's, then kept only within `maxRotationDeg`.
        var searchRotationDeg: Double = 30
        var thresholdPx: Double = 1
        var maxPerPair = 80
    }

    struct FlexGlueLoopPair {
        let a: Int, b: Int
        let uvA: [SIMD2<Float>], uvB: [SIMD2<Float>]
        let fit: RelativePoseFit
        let priorT: SIMD3<Double>
    }

    /// The strongest FAST corner of each cell of a grid of about `cells` cells (pixel centers).
    static func fastGridKeypoints(luma: MLXArray, cells: Int, threshold: Float) -> [SIMD2<Float>] {
        let h = luma.dim(0), w = luma.dim(1)
        let c = max(4, Int((Double(h * w) / Double(cells)).squareRoot().rounded()))
        let hc = (h + c - 1) / c, wc = (w + c - 1) / c
        var score = FASTCorner.score(luma: luma, threshold: threshold)
        score = padded(score, widths: [.init((0, hc * c - h)), .init((0, wc * c - w))])
        let blocks = score.reshaped([hc, c, wc, c]).transposed(0, 2, 1, 3).reshaped([hc * wc, c * c])
        let best = argMax(blocks, axis: -1).asType(.int32), top = max(blocks, axis: -1)
        eval(best, top)
        let bi = best.asArray(Int32.self), tv = top.asArray(Int32.self)
        var out: [SIMD2<Float>] = []
        for k in 0..<(hc * wc) where tv[k] > 0 {
            let x = (k % wc) * c + Int(bi[k]) % c, y = (k / wc) * c + Int(bi[k]) / c
            out.append(SIMD2(Float(x) + 0.5, Float(y) + 0.5))
        }
        return out
    }

    /// Candidate pairs (a < b) as described at the top of the file, from each frame's world→camera
    /// viewmat, intrinsics and image size, and ARKit's points.
    static func flexGlueLoopCandidates(viewmats: [[Float]], intrinsics: [Intrinsics], sizes: [(width: Int, height: Int)],
                                       points: [SIMD3<Float>], parameters p: FlexGlueLoopParameters) -> [(a: Int, b: Int)] {
        let n = viewmats.count
        let words = (points.count + 63) / 64
        let visible: [[UInt64]] = (0..<n).map { f in
            var bits = [UInt64](repeating: 0, count: words)
            let v = viewmats[f], k = intrinsics[f]
            for (i, q) in points.enumerated() {
                let z = v[8] * q.x + v[9] * q.y + v[10] * q.z + v[11]
                guard z > 0.05, z < p.maxDepth else { continue }
                let u = k.fx * (v[0] * q.x + v[1] * q.y + v[2] * q.z + v[3]) / z + k.cx
                let r = k.fy * (v[4] * q.x + v[5] * q.y + v[6] * q.z + v[7]) / z + k.cy
                if u >= 0, r >= 0, u < Float(sizes[f].width), r < Float(sizes[f].height) { bits[i / 64] |= 1 << UInt64(i % 64) }
            }
            return bits
        }
        let counts = visible.map { $0.reduce(0) { $0 + $1.nonzeroBitCount } }
        let forwards = viewmats.map { v -> SIMD3<Float> in simd_normalize(SIMD3(v[8], v[9], v[10])) }
        let cosMax = cos(p.minAngleDeg * .pi / 180)
        var cands: [(share: Float, a: Int, b: Int)] = []
        for i in 0..<n where counts[i] > 0 {
            for j in (i + p.minGap)..<max(i + p.minGap, n) where counts[j] > 0 {
                if simd_dot(forwards[i], forwards[j]) > cosMax { continue }
                var both = 0
                for w in 0..<words { both += (visible[i][w] & visible[j][w]).nonzeroBitCount }
                let share = Float(both) / Float(counts[i] + counts[j] - both)
                if share >= p.minShare { cands.append((share, i, j)) }
            }
        }
        cands.sort { ($0.share, $1.a, $1.b) > ($1.share, $0.a, $0.b) }
        var used = [Int](repeating: 0, count: n)
        var out: [(a: Int, b: Int)] = []
        for c in cands where used[c.a] < p.perFrame && used[c.b] < p.perFrame {
            if p.spacing > 0, out.contains(where: { abs($0.a - c.a) < p.spacing && abs($0.b - c.b) < p.spacing }) { continue }
            out.append((c.a, c.b)); used[c.a] += 1; used[c.b] += 1
            if out.count >= p.maxPairs { break }
        }
        return out
    }

    /// Matches the candidates with FlexGlue and keeps the verified pairs. The descriptor runs on its
    /// own thread (Neural Engine) while this one prepares the next frames (MLX); the transformer
    /// runs on another (Neural Engine) while this one reads out the matches (MLX) and fits the poses.
    static func flexGlueLoopPairs(scene: CapturedScene, parameters p: FlexGlueLoopParameters,
                                  log: (String) -> Void) -> [FlexGlueLoopPair] {
        let start = Date()
        let candidates = flexGlueLoopCandidates(
            viewmats: scene.frames.map(\.viewmat), intrinsics: scene.frames.map(\.intrinsics),
            sizes: scene.frames.map { (width: $0.width, height: $0.height) }, points: scene.sparsePoints, parameters: p)
        guard !candidates.isEmpty else { log("wide-angle loop closure: no candidate pairs"); return [] }
        let matcher: FlexGlueMatcher
        do { matcher = try FlexGlueMatcher(directory: p.directory) } catch {
            log("wide-angle loop closure: FlexGlue models not loaded (\(error))"); return []
        }
        let loaded = Date()
        // MLX keeps freed buffers for reuse; cap that cache for this stage, which Core ML works beside
        let cacheLimit = MLX.Memory.cacheLimit
        MLX.Memory.cacheLimit = 256 * 1024 * 1024
        defer { MLX.Memory.cacheLimit = cacheLimit }

        final class Shared: @unchecked Sendable {
            let lock = NSLock()
            var features: [Int: Data] = [:]
            var tokens: [FlexGlueMatcher.Tokens?]
            init(_ n: Int) { tokens = .init(repeating: nil, count: n) }
        }
        let shared = Shared(candidates.count)

        // Features: inputs prepared here, described on a thread; kept as float16 (1.5 MB a frame)
        let frames = Set(candidates.flatMap { [$0.a, $0.b] }).sorted()
        var keypoints: [Int: [SIMD2<Float>]] = [:]
        do {
            final class Queue: @unchecked Sendable {
                let lock = NSLock(); var items: [(Int, MLMultiArray)] = []; var finished = false
            }
            let queue = Queue()
            let slots = DispatchSemaphore(value: 2), wake = DispatchSemaphore(value: 0), done = DispatchSemaphore(value: 0)
            Thread {
                while true {
                    wake.wait()
                    queue.lock.lock()
                    if queue.items.isEmpty && queue.finished { queue.lock.unlock(); break }
                    let (f, input) = queue.items.removeFirst()
                    queue.lock.unlock()
                    autoreleasepool {
                        let d = try? matcher.describe(input)
                        shared.lock.lock(); shared.features[f] = d; shared.lock.unlock()
                    }
                    slots.signal()
                }
                done.signal()
            }.start()
            for f in frames {
                autoreleasepool {
                    let image = scene.frames[f].imageFloat
                    keypoints[f] = fastGridKeypoints(luma: CensusTransform.rgbToLuma(image), cells: p.cells, threshold: p.fastThreshold)
                    if let input = try? matcher.descriptorInput(image) {
                        slots.wait()
                        queue.lock.lock(); queue.items.append((f, input)); queue.lock.unlock()
                        wake.signal()
                    }
                }
            }
            queue.lock.lock(); queue.finished = true; queue.lock.unlock()
            wake.signal()
            done.wait()
        }
        MLX.Memory.clearCache()
        let described = Date()

        // Transformer: two pairs ahead on a thread
        let slots = DispatchSemaphore(value: 2)
        let ready = candidates.map { _ in DispatchSemaphore(value: 0) }
        Thread {
            for (k, (a, b)) in candidates.enumerated() {
                slots.wait()
                autoreleasepool {
                    shared.lock.lock(); let da = shared.features[a], db = shared.features[b]; shared.lock.unlock()
                    guard let da, let db, let fa = try? matcher.features(da), let fb = try? matcher.features(db) else { return }
                    let t = try? matcher.transform(fa, fb)
                    shared.lock.lock(); shared.tokens[k] = t; shared.lock.unlock()
                }
                ready[k].signal()
            }
        }.start()

        var out: [FlexGlueLoopPair] = []
        var stopped = 0, lowCovisibility = 0, noFit = 0
        for (k, (a, b)) in candidates.enumerated() {
            ready[k].wait()
            shared.lock.lock(); let tokens = shared.tokens[k]; shared.tokens[k] = nil; shared.lock.unlock()
            slots.signal()
            guard let tokens else { continue }
            if tokens.stoppedEarly { stopped += 1; continue }
            guard tokens.covisibility >= p.minCovisibility else { lowCovisibility += 1; continue }
            let fA = scene.frames[a], fB = scene.frames[b]
            let r = autoreleasepool {
                matcher.readout(tokens, keypointsA: keypoints[a]!, sizeA: SIMD2(Float(fA.width), Float(fA.height)),
                                keypointsB: keypoints[b]!, sizeB: SIMD2(Float(fB.width), Float(fB.height)))
            }
            if k % 25 == 0 { MLX.Memory.clearCache() }
            let prior = relativePose(fA.viewmat, fB.viewmat)
            guard let fit = fitRelativePoseNearPrior(
                uvA: r.pointsA, uvB: r.pointsB, intrA: fA.intrinsics, intrB: fB.intrinsics,
                priorR: prior.R, priorT: prior.t, maxRotationDeg: p.searchRotationDeg, thresholdPx: p.thresholdPx,
                minInliers: p.minInliers, seed: UInt64(a * scene.frames.count + b)),
                  fit.rotationDeg <= p.maxRotationDeg else { noFit += 1; continue }
            // Evenly spaced subset of the inliers (deterministic)
            let step = max(1, fit.inliers.count / p.maxPerPair)
            let keep = stride(from: 0, to: fit.inliers.count, by: step).prefix(p.maxPerPair).map { fit.inliers[$0] }
            out.append(FlexGlueLoopPair(a: a, b: b, uvA: keep.map { r.pointsA[$0] }, uvB: keep.map { r.pointsB[$0] },
                                        fit: fit, priorT: prior.t))
        }
        log(String(format: "wide-angle loop closure: %d candidate pair(s) over %d frame(s) → %d kept (stopped early %d, "
                  + "covisibility < %.2f %d, no pose near ARKit's %d); models %.1fs, features %.1fs, matching %.1fs",
                   candidates.count, frames.count, out.count, stopped, p.minCovisibility, lowCovisibility, noFit,
                   loaded.timeIntervalSince(start), described.timeIntervalSince(loaded), Date().timeIntervalSince(described)))
        return out
    }
}
