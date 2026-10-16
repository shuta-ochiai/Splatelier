// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import Trainer
import simd

// PLY for inspecting the seed point cloud + camera poses.
extension CaptureTraining {

    /// Writes a PLY for inspecting the seed point cloud + camera poses and returns the Gaussian
    /// count. The seed uses initial colors, and cameras are drawn as pyramids of elongated Gaussians
    /// (red, top image edge yellow), since the viewer can only draw Gaussians.
    @discardableResult
    static func writeSeedInspectionPLY(
        model: GaussianModel, frames: [CapturedFrame], sceneScale: Float, to url: URL
    ) throws -> Int {
        let c0 = GaussianModel.shC0
        // color = clamp(C0·sh0 + 0.5), so for a flat color c, sh0 = (c-0.5)/C0.
        func sh0(_ r: Float, _ g: Float, _ b: Float) -> [Float] {
            [(r - 0.5) / c0, (g - 0.5) / c0, (b - 0.5) / c0]
        }
        let camSH = sh0(1.00, 0.15, 0.10)   // camera frustum = red
        let camUpSH = sh0(1.00, 0.85, 0.10) // top image edge = yellow (makes roll/up-down explicit)

        let total = model.count
        let seedMeans = model.means.reshaped([total * 3]).asArray(Float.self)
        // The seed uses initial colors (shN≈0 at seed time, so the DC term alone sets the color).
        let seedSH0 = model.sh0.reshaped([total * 3]).asArray(Float.self)

        // Camera marker: pyramid (apex=center → rectangle in front). OpenCV convention, so image up
        // is -cameraY.
        let centers = cameraCenters(frames)
        let depth = 0.18 * Swift.max(sceneScale, 1e-4)   // distance from apex → front rectangle
        let halfW = depth * 0.50
        let halfH = depth * 0.36
        let lineRadius = depth * 0.012   // cross-section radius of the lines
        var camMeans: [Float] = []
        var camColors: [Float] = []
        var camQuats: [Float] = []
        var camLogScales: [Float] = []
        func unit(_ x: Float, _ y: Float, _ z: Float) -> SIMD3<Float> {
            let l = Swift.max(1e-6, (x * x + y * y + z * z).squareRoot())
            return SIMD3<Float>(x / l, y / l, z / l)
        }
        // Lines are drawn as elongated Gaussians laid along the edge. Shortest rotation aligning
        // the local x axis (scale[0]) to edge direction u:
        // q (w,x,y,z) = normalize(1+u.x, cross(x̂,u)) = normalize(1+u.x, 0, -u.z, u.y).
        func emitEdge(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ sh: [Float]) {
            let dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z
            let L = Swift.max(1e-6, (dx * dx + dy * dy + dz * dz).squareRoot())
            let ux = dx / L, uy = dy / L, uz = dz / L
            let qw: Float, qx: Float, qy: Float, qz: Float
            let w = 1 + ux
            if w < 1e-6 {                       // u ≈ -x̂: 180° rotation (about the y axis)
                (qw, qx, qy, qz) = (0, 0, 1, 0)
            } else {
                let nrm = Swift.max(1e-12, (w * w + uz * uz + uy * uy).squareRoot())
                (qw, qx, qy, qz) = (w / nrm, 0, -uz / nrm, uy / nrm)
            }
            // Overlap neighbors so they look like a continuous line.
            let spacing = lineRadius * 2.5
            let segs = Swift.min(16, Swift.max(3, Int((L / spacing).rounded())))
            let sigmaLong = (L / Float(segs)) * 0.75
            let lx = Foundation.log(Swift.max(1e-4, sigmaLong))
            let lr = Foundation.log(Swift.max(1e-4, lineRadius))
            for s in 0..<segs {
                let t = (Float(s) + 0.5) / Float(segs)
                camMeans += [a.x + dx * t, a.y + dy * t, a.z + dz * t]
                camColors += sh
                camQuats += [qw, qx, qy, qz]
                camLogScales += [lx, lr, lr]
            }
        }
        let apexScale = Foundation.log(Swift.max(1e-4, lineRadius * 2))
        for (i, ctr) in centers.enumerated() where i < frames.count {
            let m = frames[i].viewmat
            let right = unit(m[0], m[1], m[2])    // camera +X (image right)
            let fwd   = unit(m[8], m[9], m[10])   // camera +Z (view forward)
            let up    = -unit(m[4], m[5], m[6])   // -camera +Y (image up)
            let fc = ctr + fwd * depth
            let tr = fc + right * halfW + up * halfH
            let tl = fc - right * halfW + up * halfH
            let br = fc + right * halfW - up * halfH
            let bl = fc - right * halfW - up * halfH
            camMeans += [ctr.x, ctr.y, ctr.z]; camColors += camSH
            camQuats += [1, 0, 0, 0]; camLogScales += [apexScale, apexScale, apexScale]
            emitEdge(ctr, tr, camSH)
            emitEdge(ctr, tl, camSH)
            emitEdge(ctr, br, camSH)
            emitEdge(ctr, bl, camSH)
            emitEdge(tr, tl, camUpSH)   // top image edge
            emitEdge(br, bl, camSH)
            emitEdge(tr, br, camSH)
            emitEdge(tl, bl, camSH)
        }
        let camCount = camMeans.count / 3
        let n = total + camCount

        var means = seedMeans
        means += camMeans

        var sh0Flat: [Float] = []
        sh0Flat.reserveCapacity(n * 3)
        sh0Flat += seedSH0
        sh0Flat += camColors

        // Seed points are small and isotropic; camera markers use anisotropic scale stretched along
        // the edge.
        let ptScale = Foundation.log(Swift.max(1e-4, 0.004 * sceneScale))
        var logScales: [Float] = []
        logScales.reserveCapacity(n * 3)
        for _ in 0..<total { logScales += [ptScale, ptScale, ptScale] }
        logScales += camLogScales

        let fullLogit = logit(0.99)
        let logitOpac = [Float](repeating: fullLogit, count: n)
        var quats: [Float] = []
        quats.reserveCapacity(n * 4)
        for _ in 0..<total { quats += [1, 0, 0, 0] }
        quats += camQuats
        let shN = [Float](repeating: 0, count: n * GaussianModel.shRestBands * 3)

        let inspect = GaussianModel(
            means: MLXArray(means, [n, 3]),
            quats: MLXArray(quats, [n, 4]),
            logScales: MLXArray(logScales, [n, 3]),
            logitOpacities: MLXArray(logitOpac, [n]),
            sh0: MLXArray(sh0Flat, [n, 3]),
            shN: MLXArray(shN, [n, GaussianModel.shRestBands, 3])
        )
        eval(inspect.means, inspect.quats, inspect.logScales,
             inspect.logitOpacities, inspect.sh0, inspect.shN)
        // Cull disabled: keep distant seed/camera points out of the outlier cull.
        try PLYWriter.writeGaussianPLY(model: inspect, to: url, cull: .init(enabled: false))
        // sidecar: camera markers are the last camCount entries of the PLY. The viewer toggles
        // scene/camera display separately.
        let metaURL = url.deletingPathExtension().appendingPathExtension("seedmeta.json")
        let metaJSON = "{\"version\":1,\"cameraSplatCount\":\(camCount)}"
        try? metaJSON.data(using: .utf8)?.write(to: metaURL)
        return n
    }
}
