// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// One frame's pose of the orbit camera. The render view / projection, the tap ray, and the screen
// projection of measurement points use the same formulas, so tapped points and drawn lines don't
// drift apart on screen. Also decides where a double tap zooms to.

#if os(iOS) || os(macOS)

import simd

struct SplatViewerCamera: Sendable {
    var azimuth: Float
    var elevation: Float
    var distance: Float
    var center: SIMD3<Float>
    var fovyRadians: Float
    /// Width / height of the render target.
    var aspect: Float

    static let near: Float = 0.05
    static let far: Float = 1000

    /// world→camera rotation (camera looks down -Z, +Y up).
    var orbitRotation: simd_float4x4 {
        rotation(radians: elevation, axis: SIMD3<Float>(1, 0, 0))
            * rotation(radians: azimuth, axis: SIMD3<Float>(0, 1, 0))
    }

    /// Look-at point to origin → azimuth rotation → elevation rotation → pull back by `distance`
    /// (same convention as the CLI viewer).
    var viewMatrix: simd_float4x4 {
        translation(0, 0, -distance) * orbitRotation * translation(-center.x, -center.y, -center.z)
    }

    var projectionMatrix: simd_float4x4 {
        perspectiveRH(fovyRadians: fovyRadians, aspect: aspect, near: Self.near, far: Self.far)
    }

    /// Camera center in world.
    var eye: SIMD3<Float> {
        center + axis(2) * distance
    }

    /// Camera axes in world (0: right, 1: up, 2: back). Columns of Rᵀ of the world→camera rotation
    /// R.
    func axis(_ i: Int) -> SIMD3<Float> {
        let c = orbitRotation.transpose[i]
        return SIMD3<Float>(c.x, c.y, c.z)
    }

    /// Camera forward in world.
    var forward: SIMD3<Float> { -axis(2) }

    /// World ray through a screen point (normalized coordinates, top-left 0 / bottom-right 1).
    /// Direction is a unit vector.
    func ray(normalized p: SIMD2<Float>) -> (origin: SIMD3<Float>, direction: SIMD3<Float>) {
        let tanHalf = tanf(fovyRadians * 0.5)
        let x = (2 * p.x - 1) * tanHalf * aspect
        let y = (1 - 2 * p.y) * tanHalf
        let direction = simd_normalize(axis(0) * x + axis(1) * y + forward)
        return (eye, direction)
    }

    /// Look-at point and distance after a double-tap zoom. `point` (the surface under the tap)
    /// becomes the new look-at point and the distance from the camera to it is divided by `zoom`.
    /// Orientation is unchanged, so `point` ends up at screen center. If `point` is nil (no
    /// surface), the look-at point stays and only the distance is divided.
    func focused(on point: SIMD3<Float>?, zoom: Float) -> (center: SIMD3<Float>, distance: Float) {
        guard let point else { return (center, max(Self.near, distance / zoom)) }
        return (point, max(Self.near, simd_distance(eye, point) / zoom))
    }

    /// Camera coordinates of a world point (visible points have z < 0).
    func cameraSpace(_ p: SIMD3<Float>) -> SIMD3<Float> {
        let q = viewMatrix * SIMD4<Float>(p, 1)
        return SIMD3<Float>(q.x, q.y, q.z)
    }

    /// Screen position (normalized coordinates) of a camera-space point. nil if in front of near.
    func project(cameraSpace q: SIMD3<Float>) -> SIMD2<Float>? {
        guard -q.z >= Self.near else { return nil }
        let tanHalf = tanf(fovyRadians * 0.5)
        let x = q.x / (-q.z * tanHalf * aspect)
        let y = q.y / (-q.z * tanHalf)
        return SIMD2<Float>((x + 1) * 0.5, (1 - y) * 0.5)
    }
}

// MARK: - Matrix helpers (right-handed, column-major)

func perspectiveRH(fovyRadians fovy: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
    let ys = 1 / tanf(fovy * 0.5)
    let xs = ys / aspect
    let zs = far / (near - far)
    return simd_float4x4(columns: (
        SIMD4<Float>(xs, 0, 0, 0),
        SIMD4<Float>(0, ys, 0, 0),
        SIMD4<Float>(0, 0, zs, -1),
        SIMD4<Float>(0, 0, zs * near, 0)
    ))
}

func translation(_ x: Float, _ y: Float, _ z: Float) -> simd_float4x4 {
    simd_float4x4(columns: (
        SIMD4<Float>(1, 0, 0, 0),
        SIMD4<Float>(0, 1, 0, 0),
        SIMD4<Float>(0, 0, 1, 0),
        SIMD4<Float>(x, y, z, 1)
    ))
}

func rotation(radians: Float, axis: SIMD3<Float>) -> simd_float4x4 {
    let a = normalize(axis)
    let ct = cosf(radians), st = sinf(radians), ci = 1 - ct
    let x = a.x, y = a.y, z = a.z
    return simd_float4x4(columns: (
        SIMD4<Float>(ct + x * x * ci, y * x * ci + z * st, z * x * ci - y * st, 0),
        SIMD4<Float>(x * y * ci - z * st, ct + y * y * ci, z * y * ci + x * st, 0),
        SIMD4<Float>(x * z * ci + y * st, y * z * ci - x * st, ct + z * z * ci, 0),
        SIMD4<Float>(0, 0, 0, 1)
    ))
}

#endif // os(iOS) || os(macOS)
