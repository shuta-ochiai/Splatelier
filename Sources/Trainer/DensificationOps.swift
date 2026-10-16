// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX

/// Densification ops that rewrite `GaussianParamStore`. Follows gsplat's relocate / sample_add
/// (Apache-2.0). Index computation and opacity/scale correction run on the host; array updates go
/// through `updateParamsWithOptimizer` so parameters and Adam moments stay aligned.
extension Densification {

    /// Host-side parameters used by relocate / sampleAdd / buildCorrection.
    /// Fetched in one batch so each fire has a single sync barrier.
    struct HostSnapshot: Sendable {
        let opac: [Float]         // [N]   sigmoid(logitOpacities)
        let scalesFlat: [Float]   // [N*3] exp(logScales)
        let logitOpHost: [Float]  // [N]   raw logitOpacities
        let logScHost: [Float]    // [N*3] raw logScales
        let n: Int
    }

    /// Reads the 4 arrays to the host with a single eval. Call once at the start of a fire.
    static func snapshotHost(store: GaussianParamStore) -> HostSnapshot {
        let model = store.model
        let n = model.count
        let opacArr = model.opacities()
        let scalesArr = model.scales()
        eval(opacArr, scalesArr, model.logitOpacities, model.logScales)
        return HostSnapshot(
            opac: opacArr.asArray(Float.self),
            scalesFlat: scalesArr.asArray(Float.self),
            logitOpHost: model.logitOpacities.asArray(Float.self),
            logScHost: model.logScales.asArray(Float.self),
            n: n
        )
    }

    /// Removes dead Gaussians and compacts all parameters and Adam moments with the same keep index
    /// (unlike relocate, the total shrinks). dead = opacity ≤ minOpacity, non-finite / non-positive
    /// scale or opacity, and (if bloatScaleMultiple > 0) maxScale > median × multiple.
    /// `maxPruneFraction` caps deletions per fire to avoid collapsing coverage; at least
    /// `minRemaining` are kept.
    @discardableResult
    public static func pruneDead(
        store: GaussianParamStore,
        minOpacity: Float,
        bloatScaleMultiple: Float = 0,
        maxPruneFraction: Float = 0.25,
        minRemaining: Int = 1
    ) -> Int {
        let snapshot = snapshotHost(store: store)
        return pruneDead(
            store: store, snapshot: snapshot,
            minOpacity: minOpacity,
            bloatScaleMultiple: bloatScaleMultiple,
            maxPruneFraction: maxPruneFraction,
            minRemaining: minRemaining
        )
    }

    @discardableResult
    static func pruneDead(
        store: GaussianParamStore,
        snapshot: HostSnapshot,
        minOpacity: Float,
        bloatScaleMultiple: Float = 0,
        maxPruneFraction: Float = 0.25,
        minRemaining: Int = 1
    ) -> Int {
        let keep = pruneDeadKeepingIndices(
            store: store, snapshot: snapshot, minOpacity: minOpacity,
            bloatScaleMultiple: bloatScaleMultiple, maxPruneFraction: maxPruneFraction,
            minRemaining: minRemaining)
        return keep.map { snapshot.n - $0.count } ?? 0
    }

    /// Body of `pruneDead`. Returns the original indices (ascending) of the kept rows, or nil if
    /// nothing was removed. External per-row arrays (edge scores etc.) must be compacted with this
    /// index.
    static func pruneDeadKeepingIndices(
        store: GaussianParamStore,
        snapshot: HostSnapshot,
        minOpacity: Float,
        bloatScaleMultiple: Float = 0,
        maxPruneFraction: Float = 0.25,
        minRemaining: Int = 1
    ) -> [Int32]? {
        let n = snapshot.n
        guard n > Swift.max(1, minRemaining),
              maxPruneFraction.isFinite,
              maxPruneFraction > 0
        else { return nil }

        let opac = snapshot.opac
        let scalesFlat = snapshot.scalesFlat
        var scaleMax = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let sx = scalesFlat[i &* 3 &+ 0]
            let sy = scalesFlat[i &* 3 &+ 1]
            let sz = scalesFlat[i &* 3 &+ 2]
            scaleMax[i] = Swift.max(sx, Swift.max(sy, sz))
        }

        let bloatThreshold: Float
        if bloatScaleMultiple > 0 {
            let finitePositiveScales = scaleMax.filter { $0.isFinite && $0 > 0 }
            if finitePositiveScales.isEmpty {
                bloatThreshold = .infinity
            } else {
                let sortedScales = finitePositiveScales.sorted()
                let median = sortedScales[sortedScales.count / 2]
                bloatThreshold = median.isFinite && median > 0
                    ? bloatScaleMultiple * median
                    : .infinity
            }
        } else {
            bloatThreshold = .infinity
        }

        struct Candidate {
            let index: Int
            let priority: Float
        }

        var candidates: [Candidate] = []
        candidates.reserveCapacity(n)
        let safeMinOpacity = Swift.max(minOpacity, 1e-9)
        for i in 0..<n {
            let opa = opac[i]
            let scale = scaleMax[i]
            let invalid = !opa.isFinite || !scale.isFinite || scale <= 0
            let deadByOpacity = opa.isFinite && opa <= minOpacity
            let deadByBloat = scale.isFinite && scale > bloatThreshold
            guard invalid || deadByOpacity || deadByBloat else { continue }

            let priority: Float
            if invalid {
                priority = .greatestFiniteMagnitude
            } else if deadByBloat {
                let ratio = scale / Swift.max(bloatThreshold, 1e-12)
                priority = 10 + Swift.min(ratio, 1_000_000)
            } else {
                priority = Swift.max(0, (safeMinOpacity - opa) / safeMinOpacity)
            }
            candidates.append(Candidate(index: i, priority: priority))
        }
        guard !candidates.isEmpty else { return nil }

        candidates.sort {
            if $0.priority == $1.priority { return $0.index < $1.index }
            return $0.priority > $1.priority
        }

        let floorRemaining = Swift.max(1, minRemaining)
        let removableByFloor = Swift.max(0, n - floorRemaining)
        let fractionCap = maxPruneFraction >= 1
            ? n
            : Swift.max(1, Int(Float(n) * maxPruneFraction))
        let pruneCount = Swift.min(candidates.count, removableByFloor, fractionCap)
        guard pruneCount > 0 else { return nil }

        let pruneSet = Set(candidates.prefix(pruneCount).map(\.index))
        var keepIdx: [Int32] = []
        keepIdx.reserveCapacity(n - pruneCount)
        for i in 0..<n where !pruneSet.contains(i) {
            keepIdx.append(Int32(i))
        }
        guard !keepIdx.isEmpty else { return nil }

        let keepMLX = MLXArray(keepIdx, [keepIdx.count])
        store.updateParamsWithOptimizer(
            paramFn: { _, p in p[keepMLX] },
            momentFn: { _, m in m[keepMLX] }
        )
        return keepIdx
    }

    /// Moves dead Gaussians (opacity ≤ `minOpacity`) onto live Gaussians sampled proportional to
    /// opacity. Each sampled source is split with opacity/scale correction to preserve the
    /// composite, and the dead rows become its copies. Total count is unchanged.
    /// - Parameter bloatScaleMultiple: if > 0, maxScale > multiple × median also counts as dead,
    ///   relocating
    ///   huge blobs. 0 disables.
    @discardableResult
    public static func relocate<G: RandomNumberGenerator>(
        store: GaussianParamStore,
        binoms: [Float],
        minOpacity: Float,
        bloatScaleMultiple: Float = 0,
        maxRelocateFraction: Float = 1.0,
        using rng: inout G
    ) -> Int {
        let snapshot = snapshotHost(store: store)
        return relocate(store: store, snapshot: snapshot,
                        binoms: binoms, minOpacity: minOpacity,
                        bloatScaleMultiple: bloatScaleMultiple,
                        maxRelocateFraction: maxRelocateFraction, using: &rng)
    }

    /// Overload taking the snapshot from the start of the fire (one sync barrier per fire).
    @discardableResult
    static func relocate<G: RandomNumberGenerator>(
        store: GaussianParamStore,
        snapshot: HostSnapshot,
        binoms: [Float],
        minOpacity: Float,
        bloatScaleMultiple: Float = 0,
        maxRelocateFraction: Float = 1.0,
        using rng: inout G
    ) -> Int {
        let n = snapshot.n
        let opac = snapshot.opac
        let scalesFlat = snapshot.scalesFlat

        var scaleMax = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let sx = scalesFlat[i &* 3 &+ 0]
            let sy = scalesFlat[i &* 3 &+ 1]
            let sz = scalesFlat[i &* 3 &+ 2]
            scaleMax[i] = Swift.max(sx, Swift.max(sy, sz))
        }

        let bloatThreshold: Float
        if bloatScaleMultiple > 0 && n > 0 {
            let sortedScales = scaleMax.sorted()
            let median = sortedScales[n / 2]
            bloatThreshold = bloatScaleMultiple * median
        } else {
            bloatThreshold = .infinity
        }

        var deadIdx: [Int] = []
        var aliveIdx: [Int] = []
        for i in 0..<n {
            let deadByOpa = opac[i] <= minOpacity
            let deadByBloat = scaleMax[i] > bloatThreshold
            if deadByOpa || deadByBloat {
                deadIdx.append(i)
            } else {
                aliveIdx.append(i)
            }
        }
        guard !deadIdx.isEmpty, !aliveIdx.isEmpty else { return 0 }

        // Cap relocations per fire at maxRelocateFraction × N (relocating everything at once blows
        // up scale / n_isects and OOMs). Moves lowest opacity first; the rest wait for the next
        // fire. >= 1 is unlimited.
        if maxRelocateFraction < 1.0 {
            let cap = Swift.max(1, Int(Float(n) * maxRelocateFraction))
            if deadIdx.count > cap {
                deadIdx.sort { opac[$0] < opac[$1] }
                deadIdx = Array(deadIdx.prefix(cap))
            }
        }
        let nDead = deadIdx.count

        // Sample nDead live Gaussians proportional to opacity.
        let aliveWeights = aliveIdx.map { opac[$0] }
        let sampledLocal = multinomialSample(weights: aliveWeights, count: nDead, using: &rng)
        let sampledIdx = sampledLocal.map { aliveIdx[$0] }

        let (correctedLogitOp, correctedLogSc, maskMLX) =
            buildCorrection(snapshot: snapshot, sampledIdx: sampledIdx,
                            deadIdx: deadIdx,
                            binoms: binoms, minOpacity: minOpacity)

        // The gather index is identity; only dead rows point at their sampled source.
        var g = (0..<n).map { Int32($0) }
        for (j, d) in deadIdx.enumerated() { g[d] = Int32(sampledIdx[j]) }
        let gMLX = MLXArray(g, [n])

        store.updateParamsWithOptimizer(
            paramFn: { kind, p in
                switch kind {
                case .logitOpacities: return correctedLogitOp[gMLX]
                case .logScales: return correctedLogSc[gMLX]
                default: return p[gMLX]
                }
            },
            momentFn: { _, m in
                // Zero the moments of both the sampled (split) rows and the dead (teleported) rows
                // (keeps the old Adam state from running away at the teleport target).
                let shaped = maskMLX.reshaped([n] + Array(repeating: 1, count: m.ndim - 1))
                return m * shaped
            }
        )
        return nDead
    }

    /// Samples parents proportional to opacity, splits them with correction, and appends `count`
    /// rows at the end (moments 0).
    @discardableResult
    public static func sampleAdd<G: RandomNumberGenerator>(
        store: GaussianParamStore,
        count: Int,
        binoms: [Float],
        minOpacity: Float,
        using rng: inout G
    ) -> Int {
        guard count > 0 else { return 0 }
        let snapshot = snapshotHost(store: store)
        return sampleAdd(store: store, count: count, snapshot: snapshot,
                         binoms: binoms, minOpacity: minOpacity, using: &rng)
    }

    /// Overload taking a snapshot. A snapshot from before relocate is stale; refetch it before
    /// passing.
    @discardableResult
    static func sampleAdd<G: RandomNumberGenerator>(
        store: GaussianParamStore,
        count: Int,
        snapshot: HostSnapshot,
        binoms: [Float],
        minOpacity: Float,
        using rng: inout G
    ) -> Int {
        guard count > 0, snapshot.n > 0 else { return 0 }
        let sampledIdx = multinomialSample(weights: snapshot.opac, count: count, using: &rng)

        let (correctedLogitOp, correctedLogSc, _) =
            buildCorrection(snapshot: snapshot, sampledIdx: sampledIdx,
                            binoms: binoms, minOpacity: minOpacity)

        let sampledMLX = MLXArray(sampledIdx.map { Int32($0) }, [count])

        store.updateParamsWithOptimizer(
            paramFn: { kind, p in
                let src: MLXArray
                switch kind {
                case .logitOpacities: src = correctedLogitOp
                case .logScales: src = correctedLogSc
                default: src = p
                }
                return MLX.concatenated([src, src[sampledMLX]], axis: 0)
            },
            momentFn: { _, m in
                let tail = [count] + Array(m.shape.dropFirst())
                return MLX.concatenated([m, MLXArray.zeros(tail, dtype: m.dtype)], axis: 0)
            }
        )
        return count
    }

    /// ImprovedGS+ Long-Axis-Split (LAS). Splits each selected Gaussian into 2 at ±offset along its
    /// major axis (the existing row takes +, the new row takes - and is appended at the end).
    /// Major-axis logScale += log(alpha), minor axes +=
    /// log(secondaryScaleFactor), opacity ×= opacityFactor. Candidates are ordered by edge score
    /// `importanceScores` (used only when its length is N), otherwise by opacity × major-axis
    /// scale.
    @discardableResult
    public static func longAxisSplit(
        store: GaussianParamStore,
        count requestedCount: Int,
        importanceScores: [Float]? = nil,
        alpha: Float = 0.5,
        secondaryScaleFactor: Float = 0.85,
        opacityFactor: Float = 0.6,
        minOpacity: Float = 0.005,
        minSplitScale: Float = 0,
        minImportance: Float = 0,
        useImportanceThreshold: Bool = false
    ) -> Int {
        guard requestedCount > 0 else { return 0 }
        let model = store.model
        let n = model.count
        guard n > 0 else { return 0 }

        let opacArr = model.opacities()
        let scalesArr = model.scales()
        eval(model.means, model.quats, opacArr, scalesArr, model.logitOpacities, model.logScales)

        let meansHost = model.means.asArray(Float.self)
        let quatsHost = model.quats.asArray(Float.self)
        let opac = opacArr.asArray(Float.self)
        let scalesFlat = scalesArr.asArray(Float.self)
        let logitOpHost = model.logitOpacities.asArray(Float.self)
        let logScHost = model.logScales.asArray(Float.self)
        let edgeScores = (importanceScores?.count == n) ? importanceScores : nil

        struct Candidate {
            let idx: Int
            let principalAxis: Int
            let principalScale: Float
            let score: Float
        }

        var candidates: [Candidate] = []
        candidates.reserveCapacity(n)
        for i in 0..<n {
            guard opac[i] > minOpacity else { continue }
            let sx = scalesFlat[i * 3]
            let sy = scalesFlat[i * 3 + 1]
            let sz = scalesFlat[i * 3 + 2]
            let (axis, principal): (Int, Float)
            if sx >= sy && sx >= sz {
                (axis, principal) = (0, sx)
            } else if sy >= sz {
                (axis, principal) = (1, sy)
            } else {
                (axis, principal) = (2, sz)
            }
            let score: Float
            if let edgeScores {
                let edge = edgeScores[i]
                score = edge.isFinite ? edge : 0
            } else {
                score = opac[i] * principal
            }
            guard principal >= minSplitScale, score.isFinite, score > 0 else { continue }
            if useImportanceThreshold && score < minImportance { continue }
            candidates.append(Candidate(
                idx: i, principalAxis: axis, principalScale: principal, score: score
            ))
        }

        guard !candidates.isEmpty else { return 0 }
        candidates.sort {
            if $0.score == $1.score {
                if $0.principalScale == $1.principalScale { return $0.idx < $1.idx }
                return $0.principalScale > $1.principalScale
            }
            return $0.score > $1.score
        }
        let selected = Array(candidates.prefix(Swift.min(requestedCount, candidates.count)))
        let k = selected.count

        let safeAlpha = Swift.max(alpha, 1e-6)
        let safeSecondary = Swift.max(secondaryScaleFactor, 1e-6)
        let safeOpacityFactor = Swift.max(opacityFactor, 1e-6)
        let logAlpha = log(safeAlpha)
        let logSecondary = log(safeSecondary)
        let eps = Float.ulpOfOne

        // Updates to existing rows go to new*, added rows to added*; added rows are appended at the
        // end.
        var newMeans = meansHost
        var newLogSc = logScHost
        var newLogitOp = logitOpHost
        var addedMeans: [Float] = []
        addedMeans.reserveCapacity(k * 3)
        var addedLogSc: [Float] = []
        addedLogSc.reserveCapacity(k * 3)
        var addedLogitOp: [Float] = []
        addedLogitOp.reserveCapacity(k)

        for candidate in selected {
            let i = candidate.idx
            let axis = candidate.principalAxis
            let offsetMagnitude = candidate.principalScale * safeAlpha
            let column = rotationColumn(quat: quatsHost, index: i, axis: axis)

            let baseX = meansHost[i * 3]
            let baseY = meansHost[i * 3 + 1]
            let baseZ = meansHost[i * 3 + 2]
            let dx = column.0 * offsetMagnitude
            let dy = column.1 * offsetMagnitude
            let dz = column.2 * offsetMagnitude

            newMeans[i * 3] = baseX + dx
            newMeans[i * 3 + 1] = baseY + dy
            newMeans[i * 3 + 2] = baseZ + dz
            addedMeans += [baseX - dx, baseY - dy, baseZ - dz]

            let newOpacity = Swift.min(Swift.max(opac[i] * safeOpacityFactor, minOpacity), 1 - eps)
            newLogitOp[i] = logit(newOpacity)
            addedLogitOp.append(logit(newOpacity))

            for a in 0..<3 {
                let delta = (a == axis) ? logAlpha : logSecondary
                let v = logScHost[i * 3 + a] + delta
                newLogSc[i * 3 + a] = v
                addedLogSc.append(v)
            }
        }

        let selectedIdx = selected.map(\.idx)
        let selectedMLX = MLXArray(selectedIdx.map { Int32($0) }, [k])
        var resetMask = [Float](repeating: 1, count: n)
        for i in selectedIdx { resetMask[i] = 0 }
        let resetMaskMLX = MLXArray(resetMask, [n])

        store.updateParamsWithOptimizer(
            paramFn: { kind, p in
                let (base, rows): (MLXArray, MLXArray)
                switch kind {
                case .means:
                    (base, rows) = (MLXArray(newMeans, [n, 3]), MLXArray(addedMeans, [k, 3]))
                case .logScales:
                    (base, rows) = (MLXArray(newLogSc, [n, 3]), MLXArray(addedLogSc, [k, 3]))
                case .logitOpacities:
                    (base, rows) = (MLXArray(newLogitOp, [n]), MLXArray(addedLogitOp, [k]))
                default:
                    (base, rows) = (p, p[selectedMLX])
                }
                return MLX.concatenated([base, rows], axis: 0)
            },
            momentFn: { _, m in
                let shaped = resetMaskMLX.reshaped([n] + Array(repeating: 1, count: m.ndim - 1))
                let tail = [k] + Array(m.shape.dropFirst())
                return MLX.concatenated([m * shaped, MLXArray.zeros(tail, dtype: m.dtype)], axis: 0)
            }
        )
        return k
    }

    /// Shared by relocate / sampleAdd: writes the corrected opacity/scale of sampled (split) rows
    /// into full-length arrays. mask [N] is 0 for rows whose moments are zeroed, 1 otherwise.
    /// - Parameter deadIdx: if given, dead (teleported) rows also get mask 0 (for relocate).
    private static func buildCorrection(
        snapshot: HostSnapshot,
        sampledIdx: [Int],
        deadIdx: [Int]? = nil,
        binoms: [Float],
        minOpacity: Float
    ) -> (logitOp: MLXArray, logSc: MLXArray, mask: MLXArray) {
        let n = snapshot.n
        let opac = snapshot.opac
        let scalesHost = snapshot.scalesFlat

        // A sampled Gaussian is split into (sample count + 1) pieces.
        var counts: [Int: Int] = [:]
        for s in sampledIdx { counts[s, default: 0] += 1 }
        let ratios = sampledIdx.map { counts[$0]! + 1 }

        let sampledOpac = sampledIdx.map { opac[$0] }
        let sampledScales = sampledIdx.flatMap {
            [scalesHost[$0 * 3], scalesHost[$0 * 3 + 1], scalesHost[$0 * 3 + 2]]
        }
        let (newOp, newSc) = computeRelocation(
            opacities: sampledOpac, scalesFlat: sampledScales, ratios: ratios, binoms: binoms
        )
        let eps = Float.ulpOfOne

        var logitOpHost = snapshot.logitOpHost
        var logScHost = snapshot.logScHost
        var maskHost = [Float](repeating: 1, count: n)
        for (j, s) in sampledIdx.enumerated() {
            let clampedOp = Swift.min(Swift.max(newOp[j], minOpacity), 1 - eps)
            logitOpHost[s] = logit(clampedOp)
            logScHost[s * 3 + 0] = log(Swift.max(newSc[j * 3 + 0], 1e-9))
            logScHost[s * 3 + 1] = log(Swift.max(newSc[j * 3 + 1], 1e-9))
            logScHost[s * 3 + 2] = log(Swift.max(newSc[j * 3 + 2], 1e-9))
            maskHost[s] = 0
        }
        if let deadIdx {
            for d in deadIdx { maskHost[d] = 0 }
        }
        return (MLXArray(logitOpHost, [n]), MLXArray(logScHost, [n, 3]), MLXArray(maskHost, [n]))
    }

    /// MCMC Langevin position perturbation (every step). Shaped by the covariance; the opacity
    /// sigmoid gate moves only nearly dead Gaussians. `scaler` = lr · noiseLR.
    ///   noise = Σ · ( randn · σ(100·((1-o) - 0.995)) · scaler )
    public static func injectNoiseToPosition(store: GaussianParamStore, scaler: Float) {
        let model = store.model
        let n = model.count

        let opac = model.opacities()                                          // [N]
        let gate = 1 / (1 + MLX.exp(-100 * ((1 - opac) - 0.995)))             // [N]
        let base = MLXRandom.normal([n, 3]) * (scaler * gate).reshaped([n, 1])  // [N,3]
        let noise = covarianceTimes(quats: model.quats, scales: model.scales(), vectors: base)  // Σ·base

        model.means = model.means + noise
        eval(model.means)
    }

    /// Product `Σ·v` [N, 3] of the 3D covariance `Σ = R diag(s²) Rᵀ` with a vector. Rotation
    /// convention as in `gs_math.quat_to_rotmat`. With M = R·diag(s), Σ·v = M (Mᵀ v) is computed
    /// via elementwise products and 3-term sums, never building Σ [N, 3, 3]. A batched matmul over
    /// N 3×3 matrices suits small matrices poorly and took about 86 ms/step at 300k (Mac, MCMC
    /// per-step noise injection).
    static func covarianceTimes(quats: MLXArray, scales: MLXArray, vectors: MLXArray) -> MLXArray {
        let n = quats.shape[0]
        let m = scaledRotations(quats: quats, scales: scales)                  // [N,3,3] = R·diag(s)
        let t = (m * vectors.reshaped([n, 3, 1])).sum(axis: 1)                 // Mᵀ v   [N,3]
        return (m * t.reshaped([n, 1, 3])).sum(axis: 2)                        // M (Mᵀ v)
    }

    /// `R·diag(s)` [N, 3, 3] (Σ = M Mᵀ = R diag(s²) Rᵀ).
    private static func scaledRotations(quats: MLXArray, scales: MLXArray) -> MLXArray {
        let n = quats.shape[0]
        let norm = MLX.sqrt((quats * quats).sum(axis: 1)).reshaped([n, 1])
        let q = quats / norm
        let w = q[.ellipsis, 0], x = q[.ellipsis, 1], y = q[.ellipsis, 2], z = q[.ellipsis, 3]

        let r00 = 1 - 2 * (y * y + z * z)
        let r01 = 2 * (x * y - w * z)
        let r02 = 2 * (x * z + w * y)
        let r10 = 2 * (x * y + w * z)
        let r11 = 1 - 2 * (x * x + z * z)
        let r12 = 2 * (y * z - w * x)
        let r20 = 2 * (x * z - w * y)
        let r21 = 2 * (y * z + w * x)
        let r22 = 1 - 2 * (x * x + y * y)
        let r = MLX.stacked([r00, r01, r02, r10, r11, r12, r20, r21, r22], axis: 1)
            .reshaped([n, 3, 3])

        return r * scales.reshaped([n, 1, 3])
    }

    private static func rotationColumn(
        quat quatsHost: [Float],
        index i: Int,
        axis: Int
    ) -> (Float, Float, Float) {
        let base = i * 4
        var w = quatsHost[base]
        var x = quatsHost[base + 1]
        var y = quatsHost[base + 2]
        var z = quatsHost[base + 3]
        let norm = (w * w + x * x + y * y + z * z).squareRoot()
        if norm > 1e-12 {
            w /= norm
            x /= norm
            y /= norm
            z /= norm
        } else {
            w = 1
            x = 0
            y = 0
            z = 0
        }

        let r00 = 1 - 2 * (y * y + z * z)
        let r01 = 2 * (x * y - w * z)
        let r02 = 2 * (x * z + w * y)
        let r10 = 2 * (x * y + w * z)
        let r11 = 1 - 2 * (x * x + z * z)
        let r12 = 2 * (y * z - w * x)
        let r20 = 2 * (x * z - w * y)
        let r21 = 2 * (y * z + w * x)
        let r22 = 1 - 2 * (x * x + y * y)

        switch axis {
        case 0: return (r00, r10, r20)
        case 1: return (r01, r11, r21)
        default: return (r02, r12, r22)
        }
    }
}
