// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Lasso crop. The loop is drawn on screen, so it is a 2D polygon, and it cuts through the scene
// from the front to the back: a splat is inside when its center, seen from the camera that is
// current when the loop is finished, lies inside the loop. The splats inside are selected (orange,
// like the floaters), and Apply crops them.
//
// Plain arithmetic, so it is unit-tested without a device.

#if os(iOS) || os(macOS)

import Foundation
import simd

enum SplatViewerLasso {
    /// A loop needs at least this many points to enclose anything.
    static let minimumPoints = 3
    /// Points closer than this to the last kept point (fraction of the shorter side of the view) are
    /// dropped while drawing, so a loop stays a few dozen edges long.
    static let minimumStep: Float = 0.004

    /// The loop counts as joined when its end comes back within this distance of its start, as a
    /// fraction of the view's height.
    static let joinDistance: Float = 0.06
    /// A loop this short can't have gone anywhere, so its end is not "back" at the start.
    static let joinMinimumPoints = 8

    /// Whether the end of the loop being drawn has come back to its start.
    /// `aspect` is the view's width over its height, so the distance is the same in both directions.
    static func isJoined(_ polygon: [SIMD2<Float>], aspect: Float) -> Bool {
        guard polygon.count >= joinMinimumPoints, let first = polygon.first, let last = polygon.last else {
            return false
        }
        return simd_length((last - first) * SIMD2(aspect, 1)) <= joinDistance
    }

    /// Whether the loop encloses something: three points that are not on one line.
    static func isUsable(_ polygon: [SIMD2<Float>]) -> Bool {
        polygon.count >= minimumPoints && abs(signedArea(polygon)) > 1e-6
    }

    /// Twice the signed area of the polygon (shoelace).
    static func signedArea(_ polygon: [SIMD2<Float>]) -> Float {
        var sum: Float = 0
        for i in polygon.indices {
            let a = polygon[i], b = polygon[(i + 1) % polygon.count]
            sum += a.x * b.y - b.x * a.y
        }
        return sum / 2
    }

    /// Whether `point` is inside the polygon (even-odd rule).
    static func contains(_ polygon: [SIMD2<Float>], _ point: SIMD2<Float>) -> Bool {
        var inside = false
        var j = polygon.count - 1
        for i in polygon.indices {
            let a = polygon[i], b = polygon[j]
            if (a.y > point.y) != (b.y > point.y),
               point.x < (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x {
                inside.toggle()
            }
            j = i
        }
        return inside
    }

    /// The test for one splat, for a loop drawn in normalized screen coordinates and the camera
    /// that views it: whether the splat's center is inside the loop. A splat behind the camera or
    /// closer than its near plane has no place on screen and is outside.
    static func test(
        polygon normalized: [SIMD2<Float>], camera: SplatViewerCamera
    ) -> (_ position: SIMD3<Float>) -> Bool {
        var low = normalized[0], high = normalized[0]
        for p in normalized {
            low = simd_min(low, p)
            high = simd_max(high, p)
        }
        let view = camera.viewMatrix
        return { position in
            let q = view * SIMD4<Float>(position, 1)
            guard let n = camera.project(cameraSpace: SIMD3(q.x, q.y, q.z)),
                  n.x >= low.x, n.x <= high.x, n.y >= low.y, n.y <= high.y
            else { return false }
            return contains(normalized, n)
        }
    }

    /// The loop with `point` added, unless it is too close to the last point to matter.
    /// `aspect` is the view's width over its height, so the distance is the same in both directions.
    static func appending(_ point: SIMD2<Float>, to polygon: [SIMD2<Float>], aspect: Float) -> [SIMD2<Float>] {
        guard let last = polygon.last else { return [point] }
        let d = (point - last) * SIMD2(aspect, 1)
        guard simd_length(d) >= minimumStep else { return polygon }
        return polygon + [point]
    }
}

#endif // os(iOS) || os(macOS)
