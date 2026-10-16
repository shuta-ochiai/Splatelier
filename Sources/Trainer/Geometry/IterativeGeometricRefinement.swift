// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Iterative Geometric Refinement (PocketGS Operator G 3c, end of Sec. III-A 2). 1 round = BA
// (`baIterationsPerRound` steps) → drop observations behind the camera → re-triangulate with GPU
// DLT → drop observations with reprojection error > τ_repro → drop points with max triangulation
// angle < τ_angle. Stops when nothing is removed or at `maxRounds`.

import Foundation
import simd
import MLX
import MLXFast

public enum IterativeGeometricRefinement {

    /// DLT triangulation (Jacobi 4×4 eigendecomposition, 1 thread per point).
    private static let dltKernel = SlangCompiledKernel.required("dlt_triangulate")

    public struct Parameters: Sendable {
        /// Max outer-loop rounds (no paper value).
        public var maxRounds: Int
        /// BA steps per round. Overrides `baParameters.iterations`.
        public var baIterationsPerRound: Int
        /// Observations with reprojection error (px) above this are dropped.
        public var reprojectionThresholdPx: Float
        /// Points whose max triangulation angle (rad) is below this are dropped (no paper value).
        public var minTriangulationAngleRadians: Float
        public var baParameters: BundleAdjustment.Parameters

        public init(
            maxRounds: Int = 3,
            baIterationsPerRound: Int = 5,
            reprojectionThresholdPx: Float = 2.0,
            minTriangulationAngleRadians: Float = 0.00873,    // 0.5°
            baParameters: BundleAdjustment.Parameters = .init(
                iterations: 10, freezeFirstCamera: true,
                scaleAwareGauge: true, huberThreshold: 2.0,
                lmInitialDamping: 1e-3, lmDecay: 0.5,
                adaptiveLM: true, dampingMode: .marquardt,
                // Weak constraint toward ARKit prior poses (σ 0.25° / 1 cm, pixel σ 1 px
                // equivalent). Keeps cameras with few observations from moving the wrong way and
                // the gauge (global rotation / scale) from drifting.
                priorRotationSigmaDeg: 0.25, priorPositionSigma: 0.01
            )
        ) {
            precondition(maxRounds >= 1)
            precondition(baIterationsPerRound >= 1)
            precondition(reprojectionThresholdPx > 0)
            precondition(minTriangulationAngleRadians >= 0)
            self.maxRounds = maxRounds
            self.baIterationsPerRound = baIterationsPerRound
            self.reprojectionThresholdPx = reprojectionThresholdPx
            self.minTriangulationAngleRadians = minTriangulationAngleRadians
            var p = baParameters
            p.iterations = baIterationsPerRound
            self.baParameters = p
        }
    }

    public struct Result: Sendable {
        public struct RoundRecord: Sendable {
            public let pointCount: Int
            public let observationCount: Int
            public let baResidualPx: Float
        }
        public let state: BundleAdjustment.State
        public let observations: [BundleAdjustment.Observation]
        public let rounds: [RoundRecord]
    }

    public static func refine(
        state initialState: BundleAdjustment.State,
        intrinsics: [Intrinsics],
        observations initialObservations: [BundleAdjustment.Observation],
        parameters: Parameters = Parameters(),
        log: (String) -> Void = { print($0) }
    ) -> Result {
        var state = initialState
        var observations = initialObservations
        var rounds: [Result.RoundRecord] = []

        for round in 0..<parameters.maxRounds {
            if observations.isEmpty {
                log("iter-refine: ran out of observations before round \(round), stopping")
                break
            }
            let baResult = BundleAdjustment.refine(
                state: state, intrinsics: intrinsics,
                observations: observations, parameters: parameters.baParameters,
                priorViewmats: initialState.viewmats
            )
            state = baResult.state
            let baResidual = baResult.residualNorms.last ?? 0

            let beforeBehind = observations.count
            observations = filterBehindCamera(state: state, observations: observations)
            let droppedBehind = beforeBehind - observations.count

            // Points with fewer than 2 observations keep their original position.
            let triangulated = retriangulate(
                state: state, intrinsics: intrinsics, observations: observations
            )

            let beforeRepro = observations.count
            let filteredObs = filterReprojection(
                state: BundleAdjustment.State(viewmats: state.viewmats, points: triangulated),
                intrinsics: intrinsics,
                observations: observations,
                thresholdPx: parameters.reprojectionThresholdPx
            )
            observations = filteredObs
            let droppedRepro = beforeRepro - observations.count

            let beforeAngle = (observations.map { $0.pointIndex }).reduce(into: Set<Int>()) { $0.insert($1) }.count
            observations = filterTriangulationAngle(
                points: triangulated, intrinsics: intrinsics, observations: observations,
                viewmats: state.viewmats,
                minAngleRadians: parameters.minTriangulationAngleRadians
            )
            let afterAngle = (observations.map { $0.pointIndex }).reduce(into: Set<Int>()) { $0.insert($1) }.count
            let droppedAngle = beforeAngle - afterAngle

            state = BundleAdjustment.State(viewmats: state.viewmats, points: triangulated)

            let activePoints = Set(observations.map { $0.pointIndex }).count
            log(String(format:
                "iter-refine round %d: BA residual %.3f px, points %d → %d, obs %d → %d (drop: behind=%d, repro=%d, angle=%d)",
                round, baResidual, initialState.points.count, activePoints,
                initialObservations.count, observations.count,
                droppedBehind, droppedRepro, droppedAngle
            ))
            rounds.append(Result.RoundRecord(
                pointCount: activePoints,
                observationCount: observations.count,
                baResidualPx: baResidual
            ))

            if droppedBehind == 0 && droppedRepro == 0 && droppedAngle == 0 {
                log("iter-refine: converged at round \(round)")
                break
            }
        }

        return Result(state: state, observations: observations, rounds: rounds)
    }

    // MARK: - Internal helpers

    /// Drops observations whose point is behind the camera (z ≤ 1e-3).
    static func filterBehindCamera(
        state: BundleAdjustment.State,
        observations: [BundleAdjustment.Observation]
    ) -> [BundleAdjustment.Observation] {
        observations.filter { obs in
            let P = state.points[obs.pointIndex]
            let vm = state.viewmats[obs.cameraIndex]
            let z = vm[8] * P.x + vm[9] * P.y + vm[10] * P.z + vm[11]
            return z > 1e-3
        }
    }

    /// Re-triangulates all points by multi-view DLT (smallest eigenvector of Σ AᵀA) on the GPU.
    /// Observations are sorted by pointIndex and passed to the kernel as per-point (start, count).
    /// Points with < 2 observations or degenerate (w≈0) keep their input values.
    public static func retriangulate(
        state: BundleAdjustment.State,
        intrinsics: [Intrinsics],
        observations: [BundleAdjustment.Observation]
    ) -> [SIMD3<Float>] {
        let n = state.points.count
        if n == 0 { return state.points }

        var countsArr = [Int32](repeating: 0, count: n)
        for obs in observations { countsArr[obs.pointIndex] += 1 }

        // Exclusive prefix sum of per-point observation counts
        var startsArr = [Int32](repeating: 0, count: n)
        for j in 1..<n { startsArr[j] = startsArr[j - 1] + countsArr[j - 1] }

        let sorted = observations.sorted { $0.pointIndex < $1.pointIndex }

        let nObs = sorted.count
        var camIdxArr = [Int32](repeating: 0, count: max(nObs, 1))
        var pixArr = [Float](repeating: 0, count: max(nObs * 2, 2))
        for (i, obs) in sorted.enumerated() {
            camIdxArr[i] = Int32(obs.cameraIndex)
            pixArr[i * 2 + 0] = obs.pixel.x
            pixArr[i * 2 + 1] = obs.pixel.y
        }

        let nCam = state.viewmats.count
        var vmFlat = [Float](repeating: 0, count: nCam * 16)
        for (i, vm) in state.viewmats.enumerated() {
            precondition(vm.count == 16, "viewmat must be 16 floats")
            for k in 0..<16 { vmFlat[i * 16 + k] = vm[k] }
        }
        var intrFlat = [Float](repeating: 0, count: nCam * 4)
        for (i, intr) in intrinsics.enumerated() {
            intrFlat[i * 4 + 0] = intr.fx
            intrFlat[i * 4 + 1] = intr.fy
            intrFlat[i * 4 + 2] = intr.cx
            intrFlat[i * 4 + 3] = intr.cy
        }

        var ptsFlat = [Float](repeating: 0, count: n * 3)
        for (i, p) in state.points.enumerated() {
            ptsFlat[i * 3 + 0] = p.x
            ptsFlat[i * 3 + 1] = p.y
            ptsFlat[i * 3 + 2] = p.z
        }

        let vmArr = MLXArray(vmFlat, [nCam * 16])
        let intrArrMLX = MLXArray(intrFlat, [nCam * 4])
        let camIdxMLX = MLXArray(camIdxArr, [max(nObs, 1)])
        let pixMLX = MLXArray(pixArr, [max(nObs * 2, 2)])
        let startsMLX = MLXArray(startsArr, [n])
        let countsMLX = MLXArray(countsArr, [n])
        let ptsInMLX = MLXArray(ptsFlat, [n * 3])
        let dimsMLX = MLXArray([Int32(n)], [1])

        let outputs = dltKernel(
            [vmArr, intrArrMLX, camIdxMLX, pixMLX, startsMLX, countsMLX, ptsInMLX, dimsMLX],
            grid: (n, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[n * 3]],
            outputDTypes: [.float32]
        )
        let outArr = outputs[0]
        eval(outArr)
        let outHost = outArr.asArray(Float.self)

        var result: [SIMD3<Float>] = []
        result.reserveCapacity(n)
        for i in 0..<n {
            result.append(SIMD3<Float>(outHost[i * 3 + 0], outHost[i * 3 + 1], outHost[i * 3 + 2]))
        }
        return result
    }

    /// Drops observations with reprojection error > `thresholdPx` or behind the camera. Residuals
    /// and the mask are computed on the GPU.
    static func filterReprojection(
        state: BundleAdjustment.State,
        intrinsics: [Intrinsics],
        observations: [BundleAdjustment.Observation],
        thresholdPx: Float
    ) -> [BundleAdjustment.Observation] {
        if observations.isEmpty { return observations }
        let t2 = thresholdPx * thresholdPx
        let (residualsFlat, zCam) = BundleAdjustment.computeResidualsGPU(
            state: state, intrinsics: intrinsics, observations: observations
        )
        let n = observations.count
        let res2 = residualsFlat.reshaped([n, 2])
        let sqMag = (res2 * res2).sum(axis: 1)
        let mask = (zCam .> Float(1e-3)) .&& (sqMag .<= t2)
        eval(mask)
        let maskHost = mask.asArray(Bool.self)
        var kept: [BundleAdjustment.Observation] = []
        kept.reserveCapacity(observations.count)
        for (i, obs) in observations.enumerated() where maskHost[i] {
            kept.append(obs)
        }
        return kept
    }

    /// Drops all observations of points whose max angle between observing camera-center pairs is
    /// below `minAngleRadians`.
    static func filterTriangulationAngle(
        points: [SIMD3<Float>],
        intrinsics: [Intrinsics],
        observations: [BundleAdjustment.Observation],
        viewmats: [[Float]],
        minAngleRadians: Float
    ) -> [BundleAdjustment.Observation] {
        let cosThreshold = cos(minAngleRadians)
        let camCenters = viewmats.map { FrameSubsampler.cameraCenter(viewmat: $0) }
        var byPoint: [[Int]] = Array(repeating: [], count: points.count)
        for obs in observations { byPoint[obs.pointIndex].append(obs.cameraIndex) }
        var keepPoint = [Bool](repeating: true, count: points.count)
        for j in 0..<points.count {
            let cams = byPoint[j]
            if cams.count < 2 { continue }   // can't judge, so keep
            let P = points[j]
            // Max pair angle = min cos between rays
            var minCos: Float = 1
            var rays: [SIMD3<Float>] = []
            rays.reserveCapacity(cams.count)
            for c in cams {
                let d = P - camCenters[c]
                let m = (d.x * d.x + d.y * d.y + d.z * d.z).squareRoot()
                if m < 1e-9 { continue }
                rays.append(d / m)
            }
            for a in 0..<rays.count {
                for b in (a + 1)..<rays.count {
                    let c = rays[a].x * rays[b].x + rays[a].y * rays[b].y + rays[a].z * rays[b].z
                    if c < minCos { minCos = c }
                }
            }
            if minCos > cosThreshold { keepPoint[j] = false }
        }
        return observations.filter { keepPoint[$0.pointIndex] }
    }

}
