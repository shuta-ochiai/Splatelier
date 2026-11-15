// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import simd

/// Mesh accuracy without ground-truth geometry, from the flat surfaces of man-made scenes: walls,
/// floors, table tops and shelf boards are planes, and where two of them meet the edge is sharp.
///
/// `detectPlanes` finds the large planar regions of a reference mesh (region growing over faces
/// whose smoothed normals agree, kept when the region is large in both directions, so a strip of
/// a gently curved surface is not taken for a plane). `evaluate` then measures any mesh of the same
/// scene against those regions, refitting each plane to that mesh's own faces inside the region:
///   plane RMS     distance of the faces to their plane (mm): waviness and noise of flat surfaces
///   normal RMS    angle between the faces' normals and the plane's (deg): bumpiness at the voxel scale
///   coverage      the mesh's face area in the region over the reference's: holes in flat surfaces
///   corner RMS    distance to the nearer of the two planes for faces within `cornerRadius` of the
///                 line where two planes meet (mm): rounded or chamfered edges
///   corner fill   share of the edge's length that has faces near it: gaps along edges
/// Measuring every mesh against the same regions keeps the numbers comparable between meshes.
public enum MeshPlanarity {

    public struct Parameters: Sendable {
        /// Neighbouring faces join a region when their smoothed normals differ by less than this (deg).
        public var neighbourAngle: Float = 12
        /// A face joins a region only within this angle of the region's plane normal (deg).
        public var planeAngle: Float = 25
        /// ... and within this distance of the region's plane (m). It also bounds the faces taken
        /// into a plane by `evaluate`.
        public var planeDistance: Float = 0.015
        /// Smallest region kept as a plane (m²).
        public var minArea: Float = 0.05
        /// Smallest extent of a kept region in both in-plane directions (m), from its spread.
        public var minExtent: Float = 0.2
        /// The regions' in-plane bounds are shrunk by this (m) before faces are taken in `evaluate`,
        /// so faces bending into a neighbouring surface are left to the corner measure.
        public var boundsMargin: Float = 0.02
        /// Planes whose normals are this close to perpendicular (deg either side of 90°) form corners.
        public var cornerAngleTolerance: Float = 30
        /// Faces within this distance of a corner's line are measured (m).
        public var cornerRadius: Float = 0.03
        /// Smallest edge length that both planes reach along a corner's line (m).
        public var minCornerLength: Float = 0.1
        public init() {}
    }

    /// A planar region of the reference mesh.
    public struct Plane: Sendable {
        public var normal: SIMD3<Float>
        public var center: SIMD3<Float>
        /// In-plane axes (orthonormal with `normal`).
        public var u: SIMD3<Float>
        public var v: SIMD3<Float>
        /// In-plane bounds of the region's faces (2nd to 98th percentile), relative to `center`.
        public var uRange: ClosedRange<Float>
        public var vRange: ClosedRange<Float>
        public var area: Float

        public func distance(_ p: SIMD3<Float>) -> Float { dot(p - center, normal) }
    }

    /// Two planes of the reference mesh that meet at an edge.
    public struct Corner: Sendable {
        public var planes: (Int, Int)
        /// Line parameter range along the edge, measured from `MeshPlanarity.line`'s point.
        public var tRange: ClosedRange<Float>
    }

    public struct PlaneResult: Sendable {
        public var area: Float
        public var rms: Float
        /// 1.4826 × the area-weighted median absolute distance: the noise level, robust to small
        /// objects on the surface that the RMS takes in.
        public var robustSigma: Float
        public var normalRMSDegrees: Float
        public var coverage: Float
    }

    public struct CornerResult: Sendable {
        public var rms: Float
        public var fill: Float
    }

    public struct Result: Sendable {
        /// nil where the mesh has no faces in the region (or for a corner, near the edge).
        public var planes: [PlaneResult?]
        public var corners: [CornerResult?]

        /// Area-weighted over the planes the mesh covers.
        public var planeRMS: Float { weighted(planes.compactMap { $0.map { ($0.rms, $0.area) } }) }
        public var planeRobustSigma: Float { weighted(planes.compactMap { $0.map { ($0.robustSigma, $0.area) } }) }
        public var normalRMSDegrees: Float { weighted(planes.compactMap { $0.map { ($0.normalRMSDegrees, $0.area) } }) }
        /// Mean over all planes (0 where a plane is missing).
        public var coverage: Float {
            planes.isEmpty ? .nan : planes.map { min(1, $0?.coverage ?? 0) }.reduce(0, +) / Float(planes.count)
        }
        public var cornerRMS: Float {
            let r = corners.compactMap { $0?.rms }
            return r.isEmpty ? .nan : r.reduce(0, +) / Float(r.count)
        }
        public var cornerFill: Float {
            corners.isEmpty ? .nan : corners.map { $0?.fill ?? 0 }.reduce(0, +) / Float(corners.count)
        }
    }

    // MARK: - Faces

    struct Faces {
        var centroid: [SIMD3<Float>]
        var normal: [SIMD3<Float>]
        var area: [Float]

        init(positions: [SIMD3<Float>], indices: [UInt32]) {
            let n = indices.count / 3
            centroid = []; normal = []; area = []
            centroid.reserveCapacity(n); normal.reserveCapacity(n); area.reserveCapacity(n)
            for t in 0 ..< n {
                let a = positions[Int(indices[3 * t])], b = positions[Int(indices[3 * t + 1])]
                let c = positions[Int(indices[3 * t + 2])]
                let cr = cross(b - a, c - a)
                let len = length(cr)
                centroid.append((a + b + c) / 3)
                normal.append(len > 0 ? cr / len : .zero)
                area.append(len / 2)
            }
        }
        var count: Int { area.count }
    }

    /// Area-weighted plane fit: centroid and the eigenvector of the scatter's smallest eigenvalue,
    /// oriented along `hint`. Also returns the in-plane axes (largest spread first) and their variances.
    static func fitPlane(_ faces: Faces, _ members: [Int], hint: SIMD3<Float>)
        -> (center: SIMD3<Float>, normal: SIMD3<Float>, u: SIMD3<Float>, v: SIMD3<Float>, spread: SIMD2<Float>)? {
        var w = 0.0
        var mean = SIMD3<Double>.zero
        for f in members {
            let a = Double(faces.area[f])
            w += a
            mean += a * SIMD3<Double>(faces.centroid[f])
        }
        guard w > 0 else { return nil }
        mean /= w
        var m = [Double](repeating: 0, count: 9)
        for f in members {
            let a = Double(faces.area[f])
            let d = SIMD3<Double>(faces.centroid[f]) - mean
            for i in 0 ..< 3 { for j in 0 ..< 3 { m[3 * i + j] += a * d[i] * d[j] } }
        }
        for i in 0 ..< 9 { m[i] /= w }
        let (values, vectors) = symmetricEigen3(m)
        // values ascending
        var n = SIMD3<Float>(vectors[0])
        if dot(n, hint) < 0 { n = -n }
        let u = SIMD3<Float>(vectors[2]), v = cross(n, u)
        return (SIMD3<Float>(mean), n, u, v, SIMD2(Float(values[2]), Float(values[1])))
    }

    // MARK: - Detection

    /// The large planar regions of `positions` / `indices` (largest first).
    public static func detectPlanes(positions: [SIMD3<Float>], indices: [UInt32],
                                    parameters p: Parameters = Parameters()) -> [Plane] {
        let faces = Faces(positions: positions, indices: indices)
        let fc = faces.count
        guard fc > 0 else { return [] }

        // Faces around each vertex (CSR).
        var start = [Int32](repeating: 0, count: positions.count + 1)
        for i in indices { start[Int(i) + 1] += 1 }
        for i in 0 ..< positions.count { start[i + 1] += start[i] }
        var fill = start
        var vertexFaces = [Int32](repeating: 0, count: indices.count)
        for (k, i) in indices.enumerated() {
            vertexFaces[Int(fill[Int(i)])] = Int32(k / 3)
            fill[Int(i)] += 1
        }
        func forEachNeighbour(_ f: Int, _ body: (Int) -> Void) {
            for c in 0 ..< 3 {
                let vtx = Int(indices[3 * f + c])
                for k in Int(start[vtx]) ..< Int(start[vtx + 1]) where Int(vertexFaces[k]) != f {
                    body(Int(vertexFaces[k]))
                }
            }
        }

        // Normals smoothed over the faces sharing a vertex (twice), so the voxel staircase of a
        // marching-cubes surface does not break a plane apart; and how much each face's
        // neighbourhood deviates from it (flattest faces seed first).
        var smooth = faces.normal
        for _ in 0 ..< 2 {
            var next = smooth
            for f in 0 ..< fc {
                var acc = smooth[f] * faces.area[f]
                forEachNeighbour(f) { g in acc += smooth[g] * faces.area[g] }
                let len = length(acc)
                next[f] = len > 0 ? acc / len : smooth[f]
            }
            smooth = next
        }
        var deviation = [Float](repeating: 0, count: fc)
        for f in 0 ..< fc {
            var worst: Float = 1
            forEachNeighbour(f) { g in worst = min(worst, dot(smooth[g], smooth[f])) }
            deviation[f] = 1 - worst
        }
        let order = (0 ..< fc).filter { faces.area[$0] > 0 }.sorted { deviation[$0] < deviation[$1] }

        let cosNeighbour = cos(p.neighbourAngle * .pi / 180)
        let cosPlane = cos(p.planeAngle * .pi / 180)
        var assigned = [Bool](repeating: false, count: fc)
        var planes: [Plane] = []
        var queue: [Int] = []
        for seed in order where !assigned[seed] {
            var members = [seed]
            assigned[seed] = true
            var normal = smooth[seed], center = faces.centroid[seed]
            var area = faces.area[seed], fittedArea = area
            queue.removeAll(keepingCapacity: true)
            queue.append(seed)
            var head = 0
            while head < queue.count {
                let f = queue[head]; head += 1
                forEachNeighbour(f) { g in
                    guard !assigned[g], faces.area[g] > 0,
                          dot(smooth[g], smooth[f]) > cosNeighbour,
                          dot(smooth[g], normal) > cosPlane,
                          abs(dot(faces.centroid[g] - center, normal)) < p.planeDistance else { return }
                    assigned[g] = true
                    members.append(g)
                    queue.append(g)
                    area += faces.area[g]
                }
                if area > fittedArea * 1.5, members.count >= 8,
                   let fit = fitPlane(faces, members, hint: normal) {
                    normal = fit.normal; center = fit.center; fittedArea = area
                }
            }
            // Faces of a region that is not kept stay assigned: regrowing it from each of them
            // would take quadratic time, and the flattest faces seed first, so they are rarely
            // part of a large plane.
            guard area >= p.minArea, let fit = fitPlane(faces, members, hint: normal) else { continue }
            // The extent of a uniform strip of width w has variance w²/12.
            let extent = (12 * fit.spread).squareRoot()
            guard extent.y >= p.minExtent else { continue }
            var us = members.map { dot(faces.centroid[$0] - fit.center, fit.u) }
            var vs = members.map { dot(faces.centroid[$0] - fit.center, fit.v) }
            us.sort(); vs.sort()
            func pct(_ a: [Float], _ q: Float) -> Float { a[min(a.count - 1, Int(q * Float(a.count)))] }
            planes.append(Plane(normal: fit.normal, center: fit.center, u: fit.u, v: fit.v,
                                uRange: pct(us, 0.02) ... pct(us, 0.98), vRange: pct(vs, 0.02) ... pct(vs, 0.98),
                                area: area))
        }
        return planes.sorted { $0.area > $1.area }
    }

    /// The point and direction of the line where two planes meet.
    public static func line(_ a: Plane, _ b: Plane) -> (point: SIMD3<Float>, direction: SIMD3<Float>)? {
        let d = cross(a.normal, b.normal)
        let len = length(d)
        guard len > 1e-3 else { return nil }
        let dir = d / len
        // Point on both planes, closest to the midpoint of the centers.
        let mid = (a.center + b.center) / 2
        // Solve n_a·x = n_a·c_a, n_b·x = n_b·c_b, dir·x = dir·mid.
        let m = simd_float3x3(rows: [a.normal, b.normal, dir])
        let rhs = SIMD3<Float>(dot(a.normal, a.center), dot(b.normal, b.center), dot(dir, mid))
        return (m.inverse * rhs, dir)
    }

    /// Corners of the reference planes: pairs near perpendicular whose regions both reach the
    /// line where they meet, over a common stretch of it.
    public static func detectCorners(planes: [Plane], positions: [SIMD3<Float>], indices: [UInt32],
                                     parameters p: Parameters = Parameters()) -> [Corner] {
        let faces = Faces(positions: positions, indices: indices)
        let regionFaces = planes.map { Self.members(of: $0, in: faces, parameters: p, margin: 0) }
        let maxCos = sin(p.cornerAngleTolerance * .pi / 180)
        let reach: Float = 0.04
        var corners: [Corner] = []
        for i in 0 ..< planes.count {
            for j in (i + 1) ..< planes.count {
                guard abs(dot(planes[i].normal, planes[j].normal)) < maxCos,
                      let l = line(planes[i], planes[j]) else { continue }
                func interval(_ k: Int) -> ClosedRange<Float>? {
                    var ts: [Float] = []
                    for f in regionFaces[k] {
                        let d = faces.centroid[f] - l.point
                        let t = dot(d, l.direction)
                        if length(d - t * l.direction) < reach { ts.append(t) }
                    }
                    guard ts.count >= 20 else { return nil }
                    ts.sort()
                    return ts[ts.count / 50] ... ts[ts.count * 49 / 50]
                }
                guard let a = interval(i), let b = interval(j) else { continue }
                let lo = max(a.lowerBound, b.lowerBound) + p.boundsMargin
                let hi = min(a.upperBound, b.upperBound) - p.boundsMargin
                guard hi - lo >= p.minCornerLength else { continue }
                corners.append(Corner(planes: (i, j), tRange: lo ... hi))
            }
        }
        return corners
    }

    /// Faces of `faces` near `plane` (distance, normal, in-plane bounds shrunk by `margin`).
    static func members(of plane: Plane, in faces: Faces, parameters p: Parameters, margin: Float,
                        refit: (center: SIMD3<Float>, normal: SIMD3<Float>)? = nil) -> [Int] {
        let center = refit?.center ?? plane.center, normal = refit?.normal ?? plane.normal
        let cosMax = cos(30 * Float.pi / 180)
        var out: [Int] = []
        for f in 0 ..< faces.count where faces.area[f] > 0 {
            let c = faces.centroid[f]
            guard abs(dot(c - center, normal)) < p.planeDistance,
                  abs(dot(faces.normal[f], normal)) > cosMax else { continue }
            let r = c - plane.center
            let u = dot(r, plane.u), v = dot(r, plane.v)
            guard u > plane.uRange.lowerBound + margin, u < plane.uRange.upperBound - margin,
                  v > plane.vRange.lowerBound + margin, v < plane.vRange.upperBound - margin else { continue }
            out.append(f)
        }
        return out
    }

    // MARK: - Evaluation

    /// Measures a mesh against the reference planes and corners (`detectPlanes` / `detectCorners`
    /// on the reference mesh). `referenceAreas` are the reference mesh's own face areas in each
    /// region (`regionAreas`), the denominators of coverage.
    public static func evaluate(positions: [SIMD3<Float>], indices: [UInt32], planes: [Plane],
                                corners: [Corner], referenceAreas: [Float],
                                parameters p: Parameters = Parameters()) -> Result {
        let faces = Faces(positions: positions, indices: indices)
        var fitted: [(center: SIMD3<Float>, normal: SIMD3<Float>)?] = []
        var planeResults: [PlaneResult?] = []
        for (k, plane) in planes.enumerated() {
            var refit: (center: SIMD3<Float>, normal: SIMD3<Float>)? = nil
            var sel: [Int] = []
            for _ in 0 ..< 3 {
                sel = members(of: plane, in: faces, parameters: p, margin: p.boundsMargin, refit: refit)
                guard sel.count >= 8, let fit = fitPlane(faces, sel, hint: plane.normal) else { sel = []; break }
                refit = (fit.center, fit.normal)
            }
            guard let r = refit, !sel.isEmpty else {
                fitted.append(nil); planeResults.append(nil); continue
            }
            var a = 0.0, se = 0.0, sa = 0.0
            var absDistances: [(Float, Float)] = []
            absDistances.reserveCapacity(sel.count)
            for f in sel {
                let w = Double(faces.area[f])
                let d = Double(dot(faces.centroid[f] - r.center, r.normal))
                let ang = Double(acos(min(1, abs(dot(faces.normal[f], r.normal)))))
                a += w; se += w * d * d; sa += w * ang * ang
                absDistances.append((Float(abs(d)), faces.area[f]))
            }
            absDistances.sort { $0.0 < $1.0 }
            var median: Float = 0, acc: Float = 0
            for (d, w) in absDistances {
                acc += w
                if Double(acc) >= a / 2 { median = d; break }
            }
            fitted.append(r)
            planeResults.append(PlaneResult(
                area: Float(a), rms: Float((se / a).squareRoot()), robustSigma: 1.4826 * median,
                normalRMSDegrees: Float((sa / a).squareRoot() * 180 / .pi),
                coverage: referenceAreas[k] > 0 ? Float(a) / referenceAreas[k] : .nan))
        }

        var cornerResults: [CornerResult?] = []
        for corner in corners {
            let (i, j) = corner.planes
            guard let ri = fitted[i], let rj = fitted[j], let refLine = line(planes[i], planes[j]) else {
                cornerResults.append(nil); continue
            }
            var pi = planes[i], pj = planes[j]
            pi.center = ri.center; pi.normal = ri.normal
            pj.center = rj.center; pj.normal = rj.normal
            guard let l = line(pi, pj) else { cornerResults.append(nil); continue }
            // The reference's stretch of the edge, projected onto this mesh's line.
            let ends = [corner.tRange.lowerBound, corner.tRange.upperBound].map {
                dot(refLine.point + $0 * refLine.direction - l.point, l.direction)
            }
            let tr = min(ends[0], ends[1]) ... max(ends[0], ends[1])
            let axis = normalize(cross(pi.normal, pj.normal))
            let bins = max(1, Int(((tr.upperBound - tr.lowerBound) / 0.01).rounded()))
            var hit = [Bool](repeating: false, count: bins)
            var a = 0.0, se = 0.0
            for f in 0 ..< faces.count where faces.area[f] > 0 {
                let d = faces.centroid[f] - l.point
                let t = dot(d, l.direction)
                guard tr.contains(t), length(d - t * l.direction) < p.cornerRadius,
                      abs(dot(faces.normal[f], axis)) < 0.5 else { continue }
                let e = Double(min(abs(pi.distance(faces.centroid[f])), abs(pj.distance(faces.centroid[f]))))
                let w = Double(faces.area[f])
                a += w; se += w * e * e
                hit[min(bins - 1, Int((t - tr.lowerBound) / 0.01))] = true
            }
            cornerResults.append(a > 0 ? CornerResult(rms: Float((se / a).squareRoot()),
                                                      fill: Float(hit.filter { $0 }.count) / Float(bins)) : nil)
        }
        return Result(planes: planeResults, corners: cornerResults)
    }

    /// The plane index of each face as `evaluate` would select it for the reference planes (-1: none),
    /// for drawing the regions.
    public static func faceRegions(positions: [SIMD3<Float>], indices: [UInt32], planes: [Plane],
                                   parameters p: Parameters = Parameters()) -> [Int] {
        let faces = Faces(positions: positions, indices: indices)
        var region = [Int](repeating: -1, count: faces.count)
        for (k, plane) in planes.enumerated() {
            for f in members(of: plane, in: faces, parameters: p, margin: p.boundsMargin) where region[f] < 0 {
                region[f] = k
            }
        }
        return region
    }

    /// For each face of `faceRegions`, its signed distance to the region's plane refitted to this
    /// mesh's faces (nan outside the regions), for drawing where flat surfaces bend.
    public static func faceResiduals(positions: [SIMD3<Float>], indices: [UInt32], planes: [Plane],
                                     parameters p: Parameters = Parameters()) -> [Float] {
        let faces = Faces(positions: positions, indices: indices)
        var out = [Float](repeating: .nan, count: faces.count)
        for plane in planes {
            let sel = members(of: plane, in: faces, parameters: p, margin: p.boundsMargin)
            guard sel.count >= 8, let fit = fitPlane(faces, sel, hint: plane.normal) else { continue }
            for f in sel where out[f].isNaN { out[f] = dot(faces.centroid[f] - fit.center, fit.normal) }
        }
        return out
    }

    /// The reference mesh's own face area in each region, as `evaluate` selects it.
    public static func regionAreas(positions: [SIMD3<Float>], indices: [UInt32], planes: [Plane],
                                   parameters p: Parameters = Parameters()) -> [Float] {
        let r = evaluate(positions: positions, indices: indices, planes: planes, corners: [],
                         referenceAreas: [Float](repeating: 1, count: planes.count), parameters: p)
        return r.planes.map { $0?.area ?? 0 }
    }
}

private func weighted(_ pairs: [(Float, Float)]) -> Float {
    let w = pairs.reduce(0) { $0 + $1.1 }
    return w > 0 ? pairs.reduce(0) { $0 + $1.0 * $1.1 } / w : .nan
}

/// Eigen-decomposition of a symmetric 3×3 matrix (row-major) by Jacobi rotations: eigenvalues in
/// ascending order with their unit eigenvectors.
func symmetricEigen3(_ m: [Double]) -> (values: [Double], vectors: [SIMD3<Double>]) {
    var a = m
    var v: [Double] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    for _ in 0 ..< 32 {
        let off = a[1] * a[1] + a[2] * a[2] + a[5] * a[5]
        if off < 1e-30 { break }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            let apq = a[3 * p + q]
            guard abs(apq) > 1e-300 else { continue }
            let theta = (a[3 * q + q] - a[3 * p + p]) / (2 * apq)
            let t = (theta >= 0 ? 1 : -1) / (abs(theta) + (theta * theta + 1).squareRoot())
            let c = 1 / (t * t + 1).squareRoot(), s = t * c
            for k in 0 ..< 3 {
                let akp = a[3 * k + p], akq = a[3 * k + q]
                a[3 * k + p] = c * akp - s * akq
                a[3 * k + q] = s * akp + c * akq
            }
            for k in 0 ..< 3 {
                let apk = a[3 * p + k], aqk = a[3 * q + k]
                a[3 * p + k] = c * apk - s * aqk
                a[3 * q + k] = s * apk + c * aqk
            }
            for k in 0 ..< 3 {
                let vkp = v[3 * k + p], vkq = v[3 * k + q]
                v[3 * k + p] = c * vkp - s * vkq
                v[3 * k + q] = s * vkp + c * vkq
            }
        }
    }
    let order = [0, 1, 2].sorted { a[4 * $0] < a[4 * $1] }
    return (order.map { a[4 * $0] }, order.map { SIMD3(v[$0], v[3 + $0], v[6 + $0]) })
}
