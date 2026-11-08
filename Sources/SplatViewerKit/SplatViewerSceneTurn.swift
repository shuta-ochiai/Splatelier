// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Standing a scene upright in the viewer. Splats made by another app can come Z-up, upside down or
// on their side, and then the orbit turns about the wrong axis. Orient turns the whole scene by
// quarter turns about the axes the camera sees it along (a turn on screen, or a tip toward or away),
// so a few taps bring the model's up to the screen's up. Only the view turns (`sceneRotation` of
// `SplatViewerCamera`); the file is left as it is, and the app keeps the turn beside it.

import simd

/// How the viewer turns a scene: a quaternion from the scene's coordinates to the viewer's (Y up).
/// Saved by the app beside the scene.
public struct SplatViewerSceneRotation: Codable, Equatable, Sendable {
    /// Quaternion (ix, iy, iz, r).
    public var rotation: SIMD4<Float>

    public init(_ rotation: simd_quatf) {
        self.rotation = rotation.vector
    }

    public var quaternion: simd_quatf {
        let q = simd_quatf(vector: rotation)
        let length = simd_length(q.vector)
        guard length.isFinite, length > 1e-6 else { return simd_quatf(ix: 0, iy: 0, iz: 0, r: 1) }
        return simd_quatf(vector: q.vector / length)
    }
}

/// The buttons of Orient. Each is a quarter turn of the whole scene about the point the camera
/// orbits, about the viewer's axis nearest to the one the camera sees it along, so the scene keeps
/// square to the viewer and four presses bring it back.
public enum SplatViewerSceneTurn: CaseIterable, Sendable {
    /// Counter-clockwise on screen.
    case rotateLeft
    /// Clockwise on screen.
    case rotateRight
    /// The top tips toward the camera.
    case tipForward
    /// The top tips away from the camera.
    case tipBack

    /// The scene rotation after this turn, seen from an orbit camera at `azimuth` and `elevation`.
    func applied(to rotation: simd_quatf, azimuth: Float, elevation: Float) -> simd_quatf {
        // The camera's axes in the viewer's coordinates: the orbit alone, without the scene's turn.
        let orbit = rotation3(radians: elevation, axis: SIMD3(1, 0, 0)) * rotation3(radians: azimuth, axis: SIMD3(0, 1, 0))
        let right = orbit.transpose.columns.0, back = orbit.transpose.columns.2
        let turn: simd_quatf
        switch self {
        case .rotateLeft: turn = simd_quatf(angle: .pi / 2, axis: Self.nearestAxis(back))
        case .rotateRight: turn = simd_quatf(angle: -.pi / 2, axis: Self.nearestAxis(back))
        case .tipForward: turn = simd_quatf(angle: .pi / 2, axis: Self.nearestAxis(right))
        case .tipBack: turn = simd_quatf(angle: -.pi / 2, axis: Self.nearestAxis(right))
        }
        return Self.squared(simd_normalize(turn * rotation))
    }

    /// The viewer's axis (±X, ±Y or ±Z) closest to `v`.
    static func nearestAxis(_ v: SIMD3<Float>) -> SIMD3<Float> {
        let a = simd_abs(v)
        let i = a.x >= a.y && a.x >= a.z ? 0 : (a.y >= a.z ? 1 : 2)
        var axis = SIMD3<Float>.zero
        axis[i] = v[i] >= 0 ? 1 : -1
        return axis
    }

    /// A rotation that is a whole number of quarter turns, with the rounding of repeated turns taken
    /// out (each entry of its matrix exactly -1, 0 or 1). Anything else is returned as it is.
    static func squared(_ q: simd_quatf) -> simd_quatf {
        let m = simd_float3x3(q)
        let r = simd_float3x3(m.columns.0.rounded(.toNearestOrEven), m.columns.1.rounded(.toNearestOrEven),
                              m.columns.2.rounded(.toNearestOrEven))
        for c in 0..<3 where simd_distance(m[c], r[c]) > 1e-3 { return q }
        guard abs(r.determinant - 1) < 1e-3 else { return q }
        return simd_normalize(simd_quatf(r))
    }
}

/// A rotation about a unit `axis` (right-handed).
private func rotation3(radians: Float, axis: SIMD3<Float>) -> simd_float3x3 {
    simd_float3x3(simd_quatf(angle: radians, axis: axis))
}
