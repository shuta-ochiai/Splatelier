// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Gaussian opacity field: per-pixel 0.5 depth (`gof_depth`) and level-set extraction (`gof_carve_*`
// → Marching Cubes → binary search).

import Foundation
import MLX
import Testing
@testable import Trainer

/// One isotropic Gaussian in front of the camera: along a pixel ray the opacity is
/// o·exp(-½|t d - μ|²/s²) up to its peak, so the 0.5 depth is the near root of
/// |d|² t² - 2(d·μ) t + |μ|² + 2 s² ln(0.5 / o) = 0. Rays that miss the Gaussian never reach 0.5.
@Test
func gofDepthSingleGaussianMatchesAnalyticCrossing() throws {
    let mu = SIMD3<Float>(0.02, -0.01, 2)
    let s: Float = 0.1
    let o: Float = 0.9
    let model = GaussianModel(
        means: MLXArray([mu.x, mu.y, mu.z], [1, 3]),
        quats: MLXArray([Float(1), 0, 0, 0], [1, 4]),
        logScales: MLXArray([log(s), log(s), log(s)], [1, 3]),
        logitOpacities: MLXArray([logit(o)], [1]),
        sh0: MLXArray.zeros([1, 3]),
        shN: MLXArray.zeros([1, GaussianModel.shRestBands, 3]))
    let w = 64, h = 64
    let intr: Intrinsics = (fx: 100, fy: 100, cx: 32, cy: 32)
    let camera = Camera(
        viewmat: MLXArray([Float(1), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [16]),
        intrinsics: MLXArray([intr.fx, intr.fy, intr.cx, intr.cy], [4]),
        width: w, height: h, background: MLXArray.zeros([3]))
    let view = GaussianOpacityField.renderDepth(
        model: model, camera: camera, shDegree: 0, renderer: try GaussianRenderer(),
        parameters: .init(bisectionIterations: 20))

    func expected(_ x: Int, _ y: Int) -> Float? {
        let d = SIMD3<Float>((Float(x) + 0.5 - intr.cx) / intr.fx, (Float(y) + 0.5 - intr.cy) / intr.fy, 1)
        let a = (d * d).sum(), b = -2 * (d * mu).sum()
        let c = (mu * mu).sum() + 2 * s * s * log(0.5 / o)
        let disc = b * b - 4 * a * c
        return disc < 0 ? nil : (-b - disc.squareRoot()) / (2 * a)
    }
    var checked = 0
    for (x, y) in [(32, 32), (34, 31), (30, 35), (33, 29)] {
        guard let e = expected(x, y) else { Issue.record("pixel (\(x), \(y)) should reach 0.5"); continue }
        #expect(abs(view.depth[y * w + x] - e) < 1e-4, "pixel (\(x), \(y)): \(view.depth[y * w + x]) vs \(e)")
        checked += 1
    }
    #expect(checked == 4)
    // Far from the footprint (and at a pixel whose ray passes outside the 0.5 contour) nothing is hit.
    #expect(view.depth[0] < 0)
    #expect(expected(42, 32) == nil && view.depth[32 * w + 42] < 0)
}

/// The stochastic-solid profile (G³S eq. 12) on one isotropic Gaussian: with g(t) = o·exp(-½|t d - μ|²/s²)
/// peaking at p = g(t*), the transmittance is √(1 - g) before the peak and (1 - p)/√(1 - g) after it,
/// so it crosses 0.5 before the peak where g = 0.75 when p ≥ 0.75, after the peak where
/// g = 1 - 4(1 - p)² when 0.5 < p < 0.75, and never when p ≤ 0.5.
@Test
func gofDepthStochasticSolidMatchesAnalyticCrossing() throws {
    let mu = SIMD3<Float>(0.02, -0.01, 2)
    let s: Float = 0.1
    let w = 64, h = 64
    let intr: Intrinsics = (fx: 100, fy: 100, cx: 32, cy: 32)
    let camera = Camera(
        viewmat: MLXArray([Float(1), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [16]),
        intrinsics: MLXArray([intr.fx, intr.fy, intr.cx, intr.cy], [4]),
        width: w, height: h, background: MLXArray.zeros([3]))
    var before = 0, after = 0, missed = 0
    for o: Float in [0.9, 0.65] {
        let model = GaussianModel(
            means: MLXArray([mu.x, mu.y, mu.z], [1, 3]),
            quats: MLXArray([Float(1), 0, 0, 0], [1, 4]),
            logScales: MLXArray([log(s), log(s), log(s)], [1, 3]),
            logitOpacities: MLXArray([logit(o)], [1]),
            sh0: MLXArray.zeros([1, 3]),
            shN: MLXArray.zeros([1, GaussianModel.shRestBands, 3]))
        let view = GaussianOpacityField.renderDepth(
            model: model, camera: camera, shDegree: 0, renderer: try GaussianRenderer(),
            parameters: .init(bisectionIterations: 21, profile: .stochasticSolid))
        for (x, y) in [(32, 32), (34, 31), (30, 35), (36, 33), (38, 32), (40, 32)] {
            let d = SIMD3<Float>((Float(x) + 0.5 - intr.cx) / intr.fx, (Float(y) + 0.5 - intr.cy) / intr.fy, 1)
            let a = (d * d).sum(), dm = (d * mu).sum()
            let p = o * exp(-0.5 * ((mu * mu).sum() - dm * dm / a) / (s * s))
            let got = view.depth[y * w + x]
            guard p > 0.5 else {
                #expect(got < 0, "o \(o) pixel (\(x), \(y)): peak \(p) never reaches 0.5, got \(got)")
                missed += 1
                continue
            }
            // Roots of g(t) = target: |d|² t² - 2(d·μ) t + |μ|² + 2 s² ln(target / o) = 0.
            let target = p >= 0.75 ? Float(0.75) : 1 - 4 * (1 - p) * (1 - p)
            let c = (mu * mu).sum() + 2 * s * s * log(target / o)
            let disc = (dm * dm - a * c).squareRoot()
            let e = p >= 0.75 ? (dm - disc) / a : (dm + disc) / a
            if p >= 0.75 { before += 1 } else { after += 1 }
            #expect(abs(got - e) < 1e-4, "o \(o) pixel (\(x), \(y)), peak \(p): \(got) vs \(e)")
        }
    }
    #expect(before > 0 && after > 0 && missed > 0, "cases: before \(before), after \(after), missed \(missed)")
}

/// Depth maps of a sphere from cameras all around it: the extracted level set lies on the sphere,
/// and the binary search along the edges brings the vertices onto it.
@Test
func gofMeshOfSphereLiesOnTheSphere() {
    let radius: Float = 0.3
    let w = 192, h = 192
    let intr: Intrinsics = (fx: 180, fy: 180, cx: 96, cy: 96)
    var views: [CostVolumeMVS.DepthView] = []
    for ring in [-0.6, 0, 0.6] as [Float] {
        for k in 0..<10 {
            let az = Float(k) * 2 * .pi / 10 + ring
            let c = 1.2 * SIMD3<Float>(cos(ring) * cos(az), sin(ring), cos(ring) * sin(az))
            views.append(sphereDepthView(center: c, radius: radius, width: w, height: h, intrinsics: intr))
        }
    }
    func errors(searchIterations: Int) -> (mean: Float, max: Float)? {
        let params = GaussianOpacityField.MeshParameters(
            voxelSize: 0.01, bisectionIterations: searchIterations, minComponentTriangles: 0)
        guard let mesh = GaussianOpacityField.extractMesh(depthViews: views, parameters: params, log: { _ in }) else {
            return nil
        }
        #expect(mesh.triangleCount > 2000)
        let e = mesh.positions.map { abs(($0 * $0).sum().squareRoot() - radius) }
        return (e.reduce(0, +) / Float(e.count), e.max() ?? 0)
    }
    guard let coarse = errors(searchIterations: 0), let fine = errors(searchIterations: 8) else {
        Issue.record("no mesh extracted")
        return
    }
    #expect(fine.mean < 0.002, "mean error \(fine.mean)")
    #expect(fine.max < 0.01, "max error \(fine.max)")
    #expect(fine.mean < 0.5 * coarse.mean, "search \(fine.mean) vs midpoints \(coarse.mean)")
}

/// Ray-cast depth (camera z) of a sphere at the origin from a camera at `center` looking at it.
private func sphereDepthView(
    center c: SIMD3<Float>, radius: Float, width w: Int, height h: Int, intrinsics intr: Intrinsics
) -> CostVolumeMVS.DepthView {
    let z = -c / (c * c).sum().squareRoot()
    let upHint: SIMD3<Float> = abs(z.y) > 0.9 ? SIMD3(1, 0, 0) : SIMD3(0, 1, 0)
    var x = cross3(upHint, z)
    x /= (x * x).sum().squareRoot()
    let y = cross3(z, x)
    // Rows of R are the camera axes in world coordinates; t = -R c.
    let t = SIMD3<Float>(-(x * c).sum(), -(y * c).sum(), -(z * c).sum())
    let viewmat: [Float] = [x.x, x.y, x.z, t.x, y.x, y.y, y.z, t.y, z.x, z.y, z.z, t.z, 0, 0, 0, 1]
    var depth = [Float](repeating: -1, count: w * h)
    for py in 0..<h {
        for px in 0..<w {
            let dc = SIMD3<Float>((Float(px) + 0.5 - intr.cx) / intr.fx, (Float(py) + 0.5 - intr.cy) / intr.fy, 1)
            let dw = dc.x * x + dc.y * y + dc.z * z
            // |c + s·dw|² = r² with s the camera-z depth (dc.z = 1).
            let a = (dw * dw).sum(), b = 2 * (c * dw).sum(), cc = (c * c).sum() - radius * radius
            let disc = b * b - 4 * a * cc
            if disc >= 0 { depth[py * w + px] = (-b - disc.squareRoot()) / (2 * a) }
        }
    }
    return CostVolumeMVS.DepthView(
        width: w, height: h, depth: depth,
        rgb: [UInt8](repeating: 128, count: w * h * 3), viewmat: viewmat, intrinsics: intr)
}
