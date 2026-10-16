// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Finds the surface point under a measurement tap on the CPU. For splats, using the same 3D
// covariance as rendering (reading `SplatChunk`'s GPU buffers directly), it finds each Gaussian's
// peak response along the ray and its depth, composites front to back, and takes the depth where
// transmittance drops below 0.5 (median depth) as the surface. It uses the depth along the ray
// rather than the splat center's depth, so large flat splats seen at a grazing angle still hit the
// right surface position. The mesh is opaque, so its first hit triangle is used. With both shown,
// the nearer wins (rendering also hides splats behind the mesh).

#if os(iOS) || os(macOS)

import Metal
import MetalSplatter
import simd

/// Reference for reading the mesh layer's vertex / index buffers (shared storage) from the CPU.
/// Immutable after creation.
struct MeshRayGeometry: @unchecked Sendable {
    /// packed float3 × vertex count.
    let positions: MTLBuffer
    /// uint32 × 3 × triangle count.
    let indices: MTLBuffer
    let triangleCount: Int
}

/// Casts a ray against the visible layers. Chunk contents don't change after `addChunk` (the
/// locality sort finishes inside `addChunk`) and are only read alongside the GPU, so this may be
/// used from another thread.
struct SplatViewerRayCaster: Sendable {
    var splatChunks: [SplatChunk]
    var mesh: MeshRayGeometry?

    /// One Gaussian's contribution along the ray.
    struct SplatHit {
        /// Position of the peak response along the ray (distance from the ray origin).
        var t: Float
        var alpha: Float
    }

    /// - Parameters:
    ///   - direction: unit vector.
    ///   - pixelAngle: angular size of one pixel (rad). Adds spread equivalent to the render's 2D
    ///     dilation (0.3 px²) so splats smaller than 1 px on screen are hit just as they appear.
    /// - Returns: distance along the ray to the surface, or nil if there is nothing.
    func cast(origin: SIMD3<Float>, direction: SIMD3<Float>, pixelAngle: Float) -> Float? {
        var hits: [SplatHit] = []
        for chunk in splatChunks {
            Self.collectSplatHits(chunk, origin: origin, direction: direction, pixelAngle: pixelAngle, into: &hits)
        }
        let splatT = Self.surfaceDepth(hits)
        let meshT = mesh.flatMap { Self.firstTriangleHit($0, origin: origin, direction: direction) }
        switch (splatT, meshT) {
        case let (s?, m?): return min(s, m)
        case let (s?, nil): return s
        case let (nil, m?): return m
        case (nil, nil): return nil
        }
    }

    // MARK: Splats

    /// Minimum alpha considered when compositing (rendering also discards below 1/255).
    private static let minAlpha: Float = 1 / 255
    /// Mahalanobis distance² at which a Gaussian is truncated (3σ).
    private static let cutoffQ: Double = 9

    static func collectSplatHits(
        _ chunk: SplatChunk, origin o: SIMD3<Float>, direction d: SIMD3<Float>, pixelAngle: Float,
        into hits: inout [SplatHit]
    ) {
        let n = chunk.splatCount
        let values = chunk.splats.values
        let dilation = 0.3 * pixelAngle * pixelAngle
        let od = SIMD3<Double>(o), dd = SIMD3<Double>(d)
        for i in 0..<n {
            let splat = values[i]
            let opacity = splat.linearOpacity
            guard opacity >= minAlpha else { continue }
            let v = splat.worldPosition - o
            let tc = simd_dot(v, d)
            guard tc > SplatViewerCamera.near else { continue }
            let (upper, lower) = splat.covariance
            // A covariance stored in half precision is off by a relative 2⁻¹¹ per component, so the
            // thin-direction variance of a large thin splat can be buried in that error and lose
            // positive definiteness. Add the error bound (Frobenius ≤ 3·2⁻¹¹·max diagonal), the
            // subnormal step, and the render's dilation to the diagonal.
            let maxDiag = max(upper.x, lower.x, lower.z)
            let extra = dilation * tc * tc + 2e-3 * maxDiag + 1e-7
            // Anything whose center is more than 3σ from the ray (max eigenvalue ≤ trace) can't be
            // hit.
            let perp2 = simd_length_squared(v) - tc * tc
            let trace = upper.x + lower.x + lower.z + 3 * extra
            guard perp2 <= 9 * trace else { continue }

            // Inverse of Σ' = Σ + extra·I (via cofactors).
            let a = Double(upper.x + extra), b = Double(upper.y), c = Double(upper.z)
            let e = Double(lower.x + extra), f = Double(lower.y), g = Double(lower.z + extra)
            let i00 = e * g - f * f, i01 = c * f - b * g, i02 = b * f - c * e
            let det = a * i00 + b * i01 + c * i02
            guard det > 0, det.isFinite else { continue }
            let i11 = a * g - c * c, i12 = b * c - a * f, i22 = a * e - b * b
            func inv(_ x: SIMD3<Double>) -> SIMD3<Double> {
                SIMD3(i00 * x.x + i01 * x.y + i02 * x.z,
                      i01 * x.x + i11 * x.y + i12 * x.z,
                      i02 * x.x + i12 * x.y + i22 * x.z) / det
            }
            // Minimizer of Mahalanobis distance² q(t) = (x - μ)ᵀ Σ'⁻¹ (x - μ) along the ray
            // x(t) = o + t·d.
            let w = od - SIMD3<Double>(splat.worldPosition)
            let invD = inv(dd)
            let qa = simd_dot(dd, invD)
            guard qa > 0 else { continue }
            let qb = simd_dot(w, invD)
            let qc = simd_dot(w, inv(w))
            let q = max(0, qc - qb * qb / qa)
            guard q <= cutoffQ else { continue }
            let alpha = min(0.99, opacity * Float(exp(-0.5 * q)))
            guard alpha >= minAlpha else { continue }
            let t = Float(-qb / qa)
            guard t > SplatViewerCamera.near else { continue }
            hits.append(SplatHit(t: t, alpha: alpha))
        }
    }

    /// Depth where front-to-back transmittance drops below 0.5. For thin areas where it never drops
    /// below 0.5, if the composited opacity is at least `minCoverage`, the depth where it reaches
    /// half of that; otherwise treated as nothing.
    static func surfaceDepth(_ hits: [SplatHit], minCoverage: Float = 0.2) -> Float? {
        guard !hits.isEmpty else { return nil }
        let sorted = hits.sorted { $0.t < $1.t }
        let coverage = 1 - sorted.reduce(Float(1)) { $0 * (1 - $1.alpha) }
        guard coverage >= minCoverage else { return nil }
        let stop: Float = coverage >= 0.5 ? 0.5 : 1 - coverage * 0.5
        var transmittance: Float = 1
        for hit in sorted {
            transmittance *= 1 - hit.alpha
            if transmittance <= stop { return hit.t }
        }
        return sorted.last?.t
    }

    // MARK: Mesh

    /// First intersection with a two-sided triangle (Möller–Trumbore).
    static func firstTriangleHit(_ mesh: MeshRayGeometry, origin o: SIMD3<Float>, direction d: SIMD3<Float>) -> Float? {
        let p = mesh.positions.contents().bindMemory(to: Float.self, capacity: mesh.positions.length / 4)
        let idx = mesh.indices.contents().bindMemory(to: UInt32.self, capacity: mesh.triangleCount * 3)
        func vertex(_ k: UInt32) -> SIMD3<Float> {
            let j = Int(k) * 3
            return SIMD3(p[j], p[j + 1], p[j + 2])
        }
        var best = Float.infinity
        for tri in 0..<mesh.triangleCount {
            let v0 = vertex(idx[tri * 3]), v1 = vertex(idx[tri * 3 + 1]), v2 = vertex(idx[tri * 3 + 2])
            let e1 = v1 - v0, e2 = v2 - v0
            let pv = simd_cross(d, e2)
            let det = simd_dot(e1, pv)
            guard abs(det) > 1e-12 else { continue }
            let invDet = 1 / det
            let tv = o - v0
            let u = simd_dot(tv, pv) * invDet
            guard u >= 0, u <= 1 else { continue }
            let qv = simd_cross(tv, e1)
            let v = simd_dot(d, qv) * invDet
            guard v >= 0, u + v <= 1 else { continue }
            let t = simd_dot(e2, qv) * invDet
            if t > SplatViewerCamera.near, t < best { best = t }
        }
        return best.isFinite ? best : nil
    }
}

#endif // os(iOS) || os(macOS)
