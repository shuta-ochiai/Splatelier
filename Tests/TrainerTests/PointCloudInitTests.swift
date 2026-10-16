// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import Testing
@testable import Trainer

@Test
func pointCloudInitBasics() {
    // Unit-spaced 3x3x3 grid, center z = 4.
    var posHost: [Float] = []
    var colHost: [Float] = []
    for x in 0..<3 {
        for y in 0..<3 {
            for z in 0..<3 {
                posHost.append(Float(x))
                posHost.append(Float(y))
                posHost.append(Float(z) + 4)
                colHost.append(Float(x) / 2)
                colHost.append(Float(y) / 2)
                colHost.append(Float(z) / 2)
            }
        }
    }
    let n = 27
    let positions = MLXArray(posHost, [n, 3])
    let colors = MLXArray(colHost, [n, 3])

    let model = GaussianModel.fromPointCloud(positions: positions, colors: colors, initOpacity: 0.1)
    #expect(model.count == n)

    let m = model.means.asArray(Float.self)
    #expect(abs(m[0] - 0) < 1e-6 && abs(m[2] - 4) < 1e-6)

    // quats are random in [0,1)^4 (gsplat convention, normalized inside the renderer). At least one
    // component is non-zero so it can be normalized.
    let q = model.quats.asArray(Float.self)
    for v in q.prefix(4) { #expect(v >= 0 && v < 1, "quat component \(v) outside [0,1)") }
    #expect(q.prefix(4).contains { $0 > 0 }, "quat all zero — would NaN on normalise")

    // opacity after sigmoid ≈ 0.1.
    let op = model.opacities().asArray(Float.self)
    #expect(abs(op[0] - 0.1) < 1e-3)

    // scale ≈ nearest-neighbor distance 1 (logScale ≈ 0). Generous tolerance.
    let s = model.scales().asArray(Float.self)
    for v in s.prefix(9) {
        #expect(v > 0.5 && v < 2.0, "init scale \(v) far from grid spacing")
    }
}

@Test
func pointCloudInitSubsamples() {
    let n = 100
    var posHost: [Float] = []
    for i in 0..<n {
        posHost.append(Float(i) * 0.1)
        posHost.append(0)
        posHost.append(4)
    }
    let positions = MLXArray(posHost, [n, 3])
    let colors = MLXArray([Float](repeating: 0.5, count: n * 3), [n, 3])

    let model = GaussianModel.fromPointCloud(
        positions: positions, colors: colors, maxPoints: 20
    )
    #expect(model.count == 20, "subsample count \(model.count)")
}

@Test
func evenSubsampleIndicesPicksExactCount() {
    // Even when total is 1-2× count (where an integer stride cannot give exactly count items),
    // the result has exactly count items, ascending, no duplicates, in range.
    for (total, count) in [(383_258, 300_000), (105_329, 100_000), (100, 20), (7, 7)] {
        let idx = GaussianModel.evenSubsampleIndices(total: total, count: count).asArray(Int32.self)
        #expect(idx.count == count)
        #expect(idx.first == 0 && Int(idx.last!) < total)
        #expect(zip(idx, idx.dropFirst()).allSatisfy { $0 < $1 })
    }
}

// MARK: - PocketGS Operator I (Sec. III-B)

/// `knnIndicesAndDistances` (8 points, k=3) matches a CPU reference (argsort of the full distance
/// matrix).
@Test
func knnIndicesAndDistancesMatchesHostArgsort() {
    let pts: [Float] = [
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        1, 1, 0,
        1, 0, 1,
        0, 1, 1,
        1, 1, 1,
    ]
    let n = 8
    let positions = MLXArray(pts, [n, 3])
    let (idxArr, distArr) = GaussianModel.knnIndicesAndDistances(positions, k: 3)
    #expect(idxArr.shape == [n, 3])
    #expect(distArr.shape == [n, 3])
    let idx = idxArr.asArray(Int32.self)
    let dist = distArr.asArray(Float.self)

    // CPU reference: argsort all N×N distances excluding self, take the top k.
    func host(_ i: Int) -> [(idx: Int, d: Float)] {
        var pairs: [(Int, Float)] = []
        for j in 0..<n where j != i {
            let dx = pts[i * 3 + 0] - pts[j * 3 + 0]
            let dy = pts[i * 3 + 1] - pts[j * 3 + 1]
            let dz = pts[i * 3 + 2] - pts[j * 3 + 2]
            pairs.append((j, (dx * dx + dy * dy + dz * dz).squareRoot()))
        }
        pairs.sort { $0.1 < $1.1 }
        return pairs.prefix(3).map { ($0.0, $0.1) }
    }
    for i in 0..<n {
        let expected = host(i)
        for k in 0..<3 {
            // With equidistant points the index order is not unique, so compare distances and only
            // check the index is not self.
            #expect(abs(dist[i * 3 + k] - expected[k].d) < 1e-5,
                "knn dist mismatch at i=\(i), k=\(k): got \(dist[i * 3 + k]), expected \(expected[k].d)")
            #expect(Int(idx[i * 3 + k]) != i, "knn returned self at i=\(i), k=\(k)")
        }
    }
}

/// `quatFromZToNormals` satisfies R(q)·ẑ ≈ n (within 1e-5) for any unit normal n.
@Test
func quatFromZToNormalsMapsZToN() {
    let normalsHost: [Float] = [
        0, 0, 1,             // n = ẑ → identity quat
        1, 0, 0,             // 90° about y
        0, 1, 0,             // -90° about x
        0, 0, -1,            // antipodal (fallback path)
        0.5, 0.5, 0.7071,    // general case
        -0.6, 0.7, 0.387298,
    ]
    let n = normalsHost.count / 3
    var unit: [Float] = []
    for i in 0..<n {
        let x = normalsHost[i * 3 + 0]
        let y = normalsHost[i * 3 + 1]
        let z = normalsHost[i * 3 + 2]
        let m = (x * x + y * y + z * z).squareRoot()
        unit += [x / m, y / m, z / m]
    }
    let nrm = MLXArray(unit, [n, 3])
    let quats = GaussianModel.quatFromZToNormals(nrm)
    let q = quats.asArray(Float.self)

    // R(q)·ẑ (third column of R) = (2(xz + wy), 2(yz - wx), w² + z² - x² - y²)
    for i in 0..<n {
        let w = q[i * 4 + 0], x = q[i * 4 + 1], y = q[i * 4 + 2], z = q[i * 4 + 3]
        let rx = 2 * (x * z + w * y)
        let ry = 2 * (y * z - w * x)
        let rz = w * w + z * z - x * x - y * y
        let expectedX = unit[i * 3 + 0]
        let expectedY = unit[i * 3 + 1]
        let expectedZ = unit[i * 3 + 2]
        #expect(abs(rx - expectedX) < 1e-5,
            "R(q)·ẑ x mismatch at i=\(i): got \(rx), expected \(expectedX)")
        #expect(abs(ry - expectedY) < 1e-5,
            "R(q)·ẑ y mismatch at i=\(i): got \(ry), expected \(expectedY)")
        #expect(abs(rz - expectedZ) < 1e-5,
            "R(q)·ẑ z mismatch at i=\(i): got \(rz), expected \(expectedZ)")
    }
}

/// `pca_normals_3x3.slang` (closed-form eigenvalues), via `fromPointCloudDisc`, gives the plane
/// normal for 3 axis-aligned planes, and for a spherical cloud gives normals that do not collapse
/// to the z-axis fallback.
@Test
func pcaNormalsKernelAlignsToPlanesAndSphereFallback() {
    // Extract the normal R(q)·ẑ from fromPointCloudDisc's quat.
    func extractNormals(_ posHost: [Float], _ count: Int) -> [SIMD3<Float>] {
        let pos = MLXArray(posHost, [count, 3])
        let col = MLXArray([Float](repeating: 0.5, count: count * 3), [count, 3])
        let model = GaussianModel.fromPointCloudDisc(
            positions: pos, colors: col,
            kN: 16, kS: 3, rNormal: 0.1, initOpacity: 0.1,
            initScale: 1.0, scaleCap: .infinity
        )
        let q = model.quats.asArray(Float.self)
        var normals: [SIMD3<Float>] = []
        normals.reserveCapacity(count)
        for i in 0..<count {
            let w = q[i*4 + 0], x = q[i*4 + 1], y = q[i*4 + 2], z = q[i*4 + 3]
            normals.append(SIMD3<Float>(
                2 * (x * z + w * y),
                2 * (y * z - w * x),
                w*w + z*z - x*x - y*y
            ))
        }
        return normals
    }
    func near(_ a: SIMD3<Float>, _ b: SIMD3<Float>, tol: Float = 0.15) -> Bool {
        let absDot = abs(a.x * b.x + a.y * b.y + a.z * b.z)
        return absDot > (1.0 - tol)
    }

    // (1) z = jitter ⇒ normal ≈ ±ẑ
    var posZ: [Float] = []
    for ix in 0..<8 { for iy in 0..<8 {
        posZ += [Float(ix), Float(iy), Float.random(in: -0.01...0.01)]
    }}
    let nZ = extractNormals(posZ, 64)
    var hitZ = 0
    for n in nZ where near(n, SIMD3<Float>(0, 0, 1)) { hitZ += 1 }
    #expect(hitZ >= 45, "expected ≥45/64 normals to align with ẑ, got \(hitZ)")

    // (2) y = jitter ⇒ normal ≈ ±ŷ
    var posY: [Float] = []
    for ix in 0..<8 { for iz in 0..<8 {
        posY += [Float(ix), Float.random(in: -0.01...0.01), Float(iz)]
    }}
    let nY = extractNormals(posY, 64)
    var hitY = 0
    for n in nY where near(n, SIMD3<Float>(0, 1, 0)) { hitY += 1 }
    #expect(hitY >= 45, "expected ≥45/64 normals to align with ŷ, got \(hitY)")

    // (3) x = jitter ⇒ normal ≈ ±x̂
    var posX: [Float] = []
    for iy in 0..<8 { for iz in 0..<8 {
        posX += [Float.random(in: -0.01...0.01), Float(iy), Float(iz)]
    }}
    let nX = extractNormals(posX, 64)
    var hitX = 0
    for n in nX where near(n, SIMD3<Float>(1, 0, 0)) { hitX += 1 }
    #expect(hitX >= 45, "expected ≥45/64 normals to align with x̂, got \(hitX)")

    // (4) points on a sphere ⇒ normals differ per point. Only check that the degenerate fallback
    // (0,0,1) is not returned.
    var posSphere: [Float] = []
    let n = 100
    for _ in 0..<n {
        var v: SIMD3<Float>
        var len: Float
        repeat {
            v = SIMD3<Float>(
                Float.random(in: -1...1),
                Float.random(in: -1...1),
                Float.random(in: -1...1)
            )
            len = (v.x * v.x + v.y * v.y + v.z * v.z).squareRoot()
        } while len < 0.3 || len > 1.0
        v = SIMD3<Float>(v.x / len, v.y / len, v.z / len)
        posSphere += [v.x, v.y, v.z]
    }
    let nS = extractNormals(posSphere, n)
    var zFallback = 0
    for nrm in nS where near(nrm, SIMD3<Float>(0, 0, 1), tol: 0.05) { zFallback += 1 }
    #expect(zFallback < n / 4,
        "too many normals collapsed to (0,0,1) fallback (\(zFallback)/\(n))")
}

/// PocketGS disc-like init produces anisotropic scales on an xy-plane cloud (normal ≈ ±ẑ, log s_x ≈
/// log s_y ≫ log s_z).
@Test
func pointCloudInitDiscLikeOnPlane() {
    // 8×8 = 64 points, xy spacing ~1. A thin band with small z noise so the normal estimate is well
    // determined.
    let side = 8
    let n = side * side
    var posHost: [Float] = []
    var colHost: [Float] = []
    var rng = TestLCG(seed: 1)
    for ix in 0..<side {
        for iy in 0..<side {
            let jitterX = (rng.next() - 0.5) * 0.05
            let jitterY = (rng.next() - 0.5) * 0.05
            let jitterZ = (rng.next() - 0.5) * 0.01  // plane thickness (normal direction)
            posHost.append(Float(ix) + jitterX)
            posHost.append(Float(iy) + jitterY)
            posHost.append(jitterZ)
            colHost += [0.5, 0.5, 0.5]
        }
    }
    let positions = MLXArray(posHost, [n, 3])
    let colors = MLXArray(colHost, [n, 3])

    let model = GaussianModel.fromPointCloudDisc(
        positions: positions, colors: colors,
        kN: 16, kS: 3, rNormal: 0.1, initOpacity: 0.1
    )
    #expect(model.count == n)

    let logS = model.logScales.asArray(Float.self)   // [N, 3] = (log s_t, log s_t, log s_n)
    let q = model.quats.asArray(Float.self)           // [N, 4]
    let op = model.opacities().asArray(Float.self)    // [N]

    var disclikeCount = 0
    var normalAlignedCount = 0
    var maxAbsLogStDiff: Float = 0
    for i in 0..<n {
        let lsx = logS[i * 3 + 0]
        let lsy = logS[i * 3 + 1]
        let lsz = logS[i * 3 + 2]
        // (1) the 2 tangent axes are about the same size
        maxAbsLogStDiff = max(maxAbsLogStDiff, abs(lsx - lsy))
        // (2) thin along the normal (r_normal = 0.1 → log difference ≈ 2.3)
        if (lsx - lsz) > 1.5 && (lsy - lsz) > 1.5 {
            disclikeCount += 1
        }
        // (3) R(q)·ẑ ≈ ±ẑ (normal perpendicular to the plane)
        let w = q[i * 4 + 0], x = q[i * 4 + 1], y = q[i * 4 + 2], z = q[i * 4 + 3]
        let rz = w * w + z * z - x * x - y * y
        if abs(rz) > 0.9 { normalAlignedCount += 1 }
    }
    // Border points have neighbors on one side only and their normals can tilt, so require 70%+
    // rather than all.
    #expect(disclikeCount >= Int(Float(n) * 0.7),
        "too few disc-like anisotropic points: \(disclikeCount)/\(n)")
    #expect(normalAlignedCount >= Int(Float(n) * 0.7),
        "too many points whose normal is not aligned with ±ẑ: \(normalAlignedCount)/\(n)")
    #expect(maxAbsLogStDiff < 1e-3,
        "log s_x and log s_y do not match (max diff \(maxAbsLogStDiff))")
    #expect(abs(op[0] - 0.1) < 1e-3, "opacity init mismatch: \(op[0])")
}

@Test
func pointCloudInitRendersAndTrains() throws {
    // A small point cloud yields a model the renderer can draw.
    let n = 8
    var posHost: [Float] = []
    for i in 0..<n {
        posHost.append(Float(i % 3) * 0.3 - 0.3)
        posHost.append(Float(i / 3) * 0.3 - 0.3)
        posHost.append(4.0)
    }
    let positions = MLXArray(posHost, [n, 3])
    let colors = MLXArray([Float](repeating: 0.6, count: n * 3), [n, 3])
    let model = GaussianModel.fromPointCloud(positions: positions, colors: colors)

    let renderer = try GaussianRenderer()
    let cam = Camera(
        viewmat: MLXArray(
            [Float(1), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [16]),
        intrinsics: MLXArray([Float(60), 60, 16, 16], [4]),
        width: 32, height: 32, tileSize: 16,
        background: MLXArray([Float(0), 0, 0], [3])
    )
    let (img, alpha, _) = renderer.forward(model: model, camera: cam)
    eval(img, alpha)
    #expect(img.shape == [32, 32, 3])
    let maxAlpha = alpha.asArray(Float.self).max() ?? 0
    #expect(maxAlpha > 0.0, "point-cloud-initialised model rendered nothing")
}

/// Grid K-nearest neighbors exactly match brute force in both index and distance. Mixes 2 planes +
/// a solid blob + duplicates at the same position + far isolated points (not settled within the
/// shell limit, so they fall back to brute force).
@Test
func knnGridMatchesBruteForce() {
    var lcg = TestLCG(seed: 2026)
    var pts: [Float] = []
    for _ in 0..<3000 { pts += [lcg.next() * 2, lcg.next() * 1.5, 0.5 * lcg.next() * 0.001] }      // nearly planar
    for _ in 0..<2000 { pts += [1.2 + 0.001 * lcg.next(), lcg.next(), lcg.next() * 2] }           // perpendicular plane
    for _ in 0..<800 { pts += [3 + 0.3 * lcg.next(), 0.3 * lcg.next(), 0.3 * lcg.next()] }        // solid blob
    for i in 0..<40 { pts += Array(pts[(i * 37 * 3)..<(i * 37 * 3 + 3)]) }                        // duplicates
    for i in 0..<5 { pts += [20 + Float(i) * 7, -15, 9] }                                         // isolated points
    let n = pts.count / 3
    #expect(n >= GaussianModel.knnGridMinPoints)
    let points = MLXArray(pts, [n, 3])
    let grid = GaussianModel.knnGrid(points, k: 16)!
    let brute = GaussianModel.knnBruteForce(points, k: 16)
    #expect(grid.indices.asArray(Int32.self) == brute.indices.asArray(Int32.self))
    #expect(grid.distances.asArray(Float.self) == brute.distances.asArray(Float.self))
}
