// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Handles of a selected CAD model: its bounding box outlined like the Crop box and a full ring round
// each of its own axes that turns it about that axis (through its center), as the Crop box's rings
// do. The rings are small, keep their size on screen and carry arrows that show they turn. They sit
// up and to the right of the model on screen, off it, kept inside the view: the model itself is
// dragged to move it (direct manipulation, as Apple's AR guidelines and RealityKit's gestures do), so
// a finger meant to move it doesn't catch a ring.

import simd

/// A grab point of the selected CAD model.
enum SplatViewerCADHandle: Equatable {
    /// The model itself, which moves it in the screen plane.
    case move
    /// A point on the ring round the model's own `axis`, `angle` radians round it.
    case ring(axis: Int, angle: Float)
}

struct SplatViewerCADGizmo {
    /// The model's bounding box, placed and turned with it.
    let box: SplatViewerCropRegion
    /// Radius of the rings (see `ringRadius(for:camera:)`).
    let ringRadius: Float
    /// Center of the rings, in the world, at the depth of the model's center.
    let anchor: SIMD3<Float>

    /// The rings' diameter as a share of the view's height at the model's center.
    static let ringScreenShare: Float = 0.3

    /// A ring radius that keeps the rings the same size on screen as the camera moves.
    static func ringRadius(for box: SplatViewerCropRegion, camera: SplatViewerCamera) -> Float {
        let depth = max(simd_dot(box.center - camera.eye, camera.forward), 1e-3)
        let viewHeight = 2 * depth * tanf(camera.fovyRadians / 2)
        return ringScreenShare / 2 * viewHeight
    }

    /// `anchor` nil places the rings beside the model on screen (`screenAnchor`); pass the one in use
    /// to keep the rings in place while one of them is dragged.
    init(box: SplatViewerCropRegion, camera: SplatViewerCamera, anchor: SIMD3<Float>? = nil) {
        let radius = Self.ringRadius(for: box, camera: camera)
        self.init(box: box, ringRadius: radius, anchor: anchor ?? Self.screenAnchor(of: box, camera: camera))
    }

    init(box: SplatViewerCropRegion, ringRadius: Float, anchor: SIMD3<Float>) {
        self.box = box
        self.ringRadius = ringRadius
        self.anchor = anchor
    }

    /// Where the rings go: just past the top right of the model's outline on screen, pulled back
    /// inside the view (a ring's width from each edge) when that would leave it, at the depth of the
    /// model's center. A model that fills the view gets its rings over its top right corner.
    static func screenAnchor(of box: SplatViewerCropRegion, camera: SplatViewerCamera) -> SIMD3<Float> {
        let corners: [SIMD2<Float>] = (0..<8).compactMap { k in
            let h = box.halfExtent
            let offset = SIMD3<Float>(k & 1 == 0 ? -h.x : h.x, k & 2 == 0 ? -h.y : h.y, k & 4 == 0 ? -h.z : h.z)
            return camera.project(cameraSpace: camera.cameraSpace(box.center + box.rotation.act(offset)))
        }
        guard let right = corners.map(\.x).max(), let top = corners.map(\.y).min() else { return box.center }
        // A ring's radius in normalized units: the share is of the view's height.
        let r = SIMD2<Float>(ringScreenShare / 2 / max(camera.aspect, 1e-3), ringScreenShare / 2)
        let lo = r * 1.2, hi = SIMD2<Float>(repeating: 1) - r * 1.2
        let wanted = SIMD2<Float>(right + r.x * 1.1, top - r.y * 1.1)
        let point = simd_clamp(wanted, lo, simd_max(lo, hi))
        let ray = camera.ray(normalized: point)
        let depth = max(simd_dot(box.center - camera.eye, camera.forward), 1e-3)
        return ray.origin + ray.direction * (depth / max(simd_dot(ray.direction, camera.forward), 1e-3))
    }

    /// World position of the point on the ring round the box's own `axis` at `angle`. The ring lies in
    /// the plane of the other two axes, and an increasing angle turns the right-handed way about
    /// `axis` (as the Crop box's rings).
    func ringPoint(axis: Int, angle: Float) -> SIMD3<Float> {
        anchor + (direction((axis + 1) % 3) * cosf(angle) + direction((axis + 2) % 3) * sinf(angle)) * ringRadius
    }

    private func direction(_ axis: Int) -> SIMD3<Float> {
        var u = SIMD3<Float>.zero
        u[axis] = 1
        return box.rotation.act(u)
    }

    static let ringSteps = 72
    /// The two arrowheads on each ring: either end of a short arc, pointing away from each other
    /// round the ring, to show that it turns either way.
    static let arrows: [(angle: Float, sense: Float)] = [(.pi / 4 + 0.35, 1), (.pi / 4 - 0.35, -1)]

    /// The whole ring round the box's own `axis`, as angles and world points (the last closes it).
    func ring(axis: Int) -> [(angle: Float, point: SIMD3<Float>)] {
        (0...Self.ringSteps).map { k in
            let angle = 2 * Float.pi * Float(k) / Float(Self.ringSteps)
            return (angle, ringPoint(axis: axis, angle: angle))
        }
    }

    /// The model after a two-finger twist (or a trackpad rotation) of `radians`, clockwise on screen
    /// positive: it turns about the vertical, the way the fingers turn as seen from the camera.
    func twisted(by radians: Float, camera: SplatViewerCamera) -> SplatViewerCropRegion {
        let up = camera.up
        // Seen from above, clockwise is a negative turn about +Y; from below it is the other way.
        let sense: Float = simd_dot(camera.forward, up) <= 0 ? -1 : 1
        var turned = box
        turned.rotation = simd_normalize(simd_quatf(angle: sense * radians, axis: up) * box.rotation)
        return turned
    }

    /// The ring under a touch (normalized coordinates), within `ringReach` pixels. Seen edge-on a
    /// ring is a line; it still turns the model when dragged along it. nil leaves the touch to the
    /// model or the camera.
    func ring(
        at touch: SIMD2<Float>, camera: SplatViewerCamera, viewSize: SIMD2<Float>, ringReach: Float
    ) -> SplatViewerCADHandle? {
        let target = touch * viewSize
        var best: (handle: SplatViewerCADHandle, distance: Float)?
        for axis in 0..<3 {
            let points = ring(axis: axis)
            let screen = points.map { camera.project(cameraSpace: camera.cameraSpace($0.point)).map { $0 * viewSize } }
            for k in 0..<Self.ringSteps {
                guard let a = screen[k], let b = screen[k + 1] else { continue }
                let ab = b - a
                let t = min(max(simd_dot(target - a, ab) / max(simd_dot(ab, ab), 1e-6), 0), 1)
                let distance = simd_distance(target, a + ab * t)
                if distance < (best?.distance ?? .infinity) {
                    let angle = points[k].angle + (points[k + 1].angle - points[k].angle) * t
                    best = (.ring(axis: axis, angle: angle), distance)
                }
            }
        }
        guard let best, best.distance <= ringReach else { return nil }
        return best.handle
    }

    /// The box turned `angle` radians about its own `axis` (right-handed, through its center).
    func turned(about axis: Int, by angle: Float) -> SplatViewerCropRegion {
        var turned = box
        turned.rotation = simd_normalize(simd_quatf(angle: angle, axis: direction(axis)) * box.rotation)
        return turned
    }

    /// How far dragging a ring by `pixels` turns the model about the ring's axis, in radians: the arc
    /// the finger covers along the grabbed point's tangent over the ring's radius. 0 for the model
    /// itself or when it can't be measured.
    func turnAngle(
        _ handle: SplatViewerCADHandle, byPixels d: SIMD2<Float>,
        camera: SplatViewerCamera, viewSize: SIMD2<Float>
    ) -> Float {
        guard case let .ring(axis, angle) = handle else { return 0 }
        let p = ringPoint(axis: axis, angle: angle)
        let tangent = simd_cross(direction(axis), p - anchor)
        guard simd_length(tangent) > 1e-6,
              let column = screenColumn(at: p, along: simd_normalize(tangent), camera: camera, viewSize: viewSize)
        else { return 0 }
        return simd_dot(d, column) / simd_dot(column, column) / ringRadius
    }

    /// The box after dragging `handle` by `pixels` on screen, worked as on the Crop box: the model
    /// moves in the screen plane at the depth of its center, and a ring's grabbed point follows the
    /// finger round the ring (the model turns about its center by the arc the finger covers over the
    /// ring's radius).
    func dragged(
        _ handle: SplatViewerCADHandle, byPixels d: SIMD2<Float>,
        camera: SplatViewerCamera, viewSize: SIMD2<Float>
    ) -> SplatViewerCropRegion {
        switch handle {
        case .move:
            return box.dragged(.center, byPixels: d, camera: camera, viewSize: viewSize)
        case let .ring(axis, _):
            return turned(about: axis, by: turnAngle(handle, byPixels: d, camera: camera, viewSize: viewSize))
        }
    }

    /// Screen movement in pixels for a unit world movement of `at` along `u`, as for the Crop box.
    private func screenColumn(
        at: SIMD3<Float>, along u: SIMD3<Float>, camera: SplatViewerCamera, viewSize: SIMD2<Float>
    ) -> SIMD2<Float>? {
        let eps = max(camera.distance, 1e-3) * 1e-3
        guard let p0 = camera.project(cameraSpace: camera.cameraSpace(at)),
              let p1 = camera.project(cameraSpace: camera.cameraSpace(at + u * eps))
        else { return nil }
        let column = (p1 - p0) * viewSize / eps
        return simd_length_squared(column) > 1e-6 ? column : nil
    }

    /// What the overlay draws: the box outline, the three rings and their arrows.
    func frame(camera: SplatViewerCamera) -> SplatCropFrame {
        var frame = SplatCropFrame.make(region: box, camera: camera)
        frame.handles = []
        frame.moveHandles = []
        frame.isSelection = true
        func project(_ p: SIMD3<Float>) -> SIMD2<Float>? {
            camera.project(cameraSpace: camera.cameraSpace(p))
        }
        frame.rings = (0..<3).map { axis in
            let points = ring(axis: axis).map(\.point)
            return zip(points, points.dropFirst()).compactMap { a, b in
                guard let clipped = SplatMeasureProjection.clipToNear(camera.cameraSpace(a), camera.cameraSpace(b)),
                      let pa = camera.project(cameraSpace: clipped.0),
                      let pb = camera.project(cameraSpace: clipped.1)
                else { return nil }
                return SplatCropFrame.Segment(a: pa, b: pb)
            }
        }
        // Each arrowhead points along the ring the way it turns the model.
        frame.ringArrows = (0..<3).flatMap { axis in
            Self.arrows.compactMap { arrow -> SplatCropFrame.Arrow? in
                guard let at = project(ringPoint(axis: axis, angle: arrow.angle)),
                      let ahead = project(ringPoint(axis: axis, angle: arrow.angle + 0.05 * arrow.sense)),
                      simd_length(ahead - at) > 1e-6
                else { return nil }
                return SplatCropFrame.Arrow(axis: axis, at: at, direction: simd_normalize(ahead - at))
            }
        }
        return frame
    }
}

/// The pull of floors and tables on a dragged CAD model: it moves freely, and when its foot comes
/// within reach of a floor-like surface just under or over it (the mesh made from the capture, or
/// the splats) it settles onto it, as a magnet would, tilted to lie flat on it (the splats may lean).
/// Dragging further pulls it off again, back to the way it was turned.
enum SplatViewerCADSurface {
    /// How upright a surface must be to stand on: the cosine of its steepest allowed slope (~45°).
    static let minimumUp: Float = 0.7
    /// The most a model is tilted to lie flat on a floor. ARKit's gravity keeps a capture's floors
    /// within a few degrees of level, so a floor that reads steeper is taken as a misreading of the
    /// splats and the model only settles on it, upright.
    static let maximumTilt: Float = 10 * .pi / 180
    /// How close the foot must come to a surface to settle on it, in pixels at the model's depth.
    static let snapPixels: Float = 30

    /// The model's foot: under its center, level with its lowest corner as it now stands.
    static func foot(of box: SplatViewerCropRegion) -> SIMD3<Float> {
        let h = box.halfExtent
        let lowest = (0..<8).map { k -> Float in
            let offset = SIMD3<Float>(k & 1 == 0 ? -h.x : h.x, k & 2 == 0 ? -h.y : h.y, k & 4 == 0 ? -h.z : h.z)
            return (box.center + box.rotation.act(offset)).y
        }.min() ?? box.center.y
        return SIMD3(box.center.x, lowest, box.center.z)
    }

    /// Normal of the surface through hits around a point, in order round it (some may be missing),
    /// turned toward `eye`: the area-weighted normal of the polygon they make (Newell's method), which
    /// evens out the bumps of a splat surface. nil when fewer than three hits are there or they are
    /// in a line.
    static func normal(around hits: [SIMD3<Float>?], toward eye: SIMD3<Float>) -> SIMD3<Float>? {
        let points = hits.compactMap { $0 }
        guard points.count >= 3 else { return nil }
        var n = SIMD3<Float>.zero
        for (k, a) in points.enumerated() {
            let b = points[(k + 1) % points.count]
            n += SIMD3((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y))
        }
        let length = simd_length(n)
        guard length > 1e-9 else { return nil }
        let unit = n / length
        let middle = points.reduce(.zero, +) / Float(points.count)
        return simd_dot(unit, eye - middle) >= 0 ? unit : -unit
    }

    /// Where to look for a surface under the foot: straight down from `reach` above it, at the foot
    /// and at eight points round it `spread` away, in order (for the surface's slope). The search
    /// stops `reach` below the foot, so only surfaces within reach count.
    static func probes(under foot: SIMD3<Float>, reach: Float, spread: Float) -> [SIMD3<Float>] {
        let top = foot + SIMD3(0, reach, 0)
        return [top] + (0..<8).map { k in
            let a = Float(k) * .pi / 4
            return top + SIMD3(cosf(a), 0, -sinf(a)) * spread
        }
    }

    /// A floor-like surface: a point on it and its upward normal.
    struct Floor: Equatable {
        var point: SIMD3<Float>
        var normal: SIMD3<Float>

        /// Height of the surface's plane above (x, z).
        func height(atX x: Float, z: Float) -> Float {
            point.y - (normal.x * (x - point.x) + normal.z * (z - point.z)) / normal.y
        }

        /// The floor turned about the origin.
        func rotated(by q: simd_quatf) -> Floor {
            Floor(point: q.act(point), normal: q.act(normal))
        }
    }

    /// The floor-like surface the probes found (hits from straight down, the first at the foot), or
    /// nil when there is none within reach or it is too steep to stand on.
    static func floor(hits: [SIMD3<Float>?], foot: SIMD3<Float>, reach: Float) -> Floor? {
        guard let center = hits.first ?? nil, abs(center.y - foot.y) <= reach else { return nil }
        let up = SIMD3<Float>(0, 1, 0)
        guard let normal = normal(around: Array(hits.dropFirst()), toward: center + up), normal.y >= minimumUp
        else { return nil }
        return Floor(point: center, normal: normal)
    }

    /// Which of the box's own axes points most upward as it now stands, and which way: the one that
    /// becomes the floor's normal when it settles (a model laid on its side stays on its side).
    static func upAxis(of box: SplatViewerCropRegion) -> (axis: Int, sign: Float) {
        let up = SIMD3<Float>(0, 1, 0)
        var best = (axis: 1, sign: Float(1), dot: -Float.infinity)
        for axis in 0..<3 {
            var e = SIMD3<Float>.zero
            e[axis] = 1
            let d = simd_dot(box.rotation.act(e), up)
            if abs(d) > best.dot { best = (axis, d >= 0 ? 1 : -1, abs(d)) }
        }
        return (best.axis, best.sign)
    }

    /// The freely dragged box settled onto `floor` when its foot is within `reach` of it: tilted
    /// (about its center, by the least turn) so its upward axis follows the floor's normal (or stays
    /// upright when that leans more than `maximumTilt`), then lowered or raised until the middle of
    /// its bottom lies on the floor. Otherwise the box as dragged.
    static func settled(_ box: SplatViewerCropRegion, on found: Floor?, reach: Float) -> SplatViewerCropRegion {
        guard var floor = found, abs(foot(of: box).y - floor.point.y) <= reach else { return box }
        let up = SIMD3<Float>(0, 1, 0)
        if acosf(min(1, floor.normal.y)) > maximumTilt { floor.normal = up }
        let (axis, sign) = upAxis(of: box)
        var e = SIMD3<Float>.zero
        e[axis] = sign
        var settled = box
        settled.rotation = simd_normalize(simd_quatf(from: box.rotation.act(e), to: floor.normal) * box.rotation)
        let bottom = settled.center - floor.normal * box.halfExtent[axis]
        settled.center.y += floor.height(atX: bottom.x, z: bottom.z) - bottom.y
        return settled
    }
}

extension SplatViewerCropRegion {
    /// The region turned about the origin: into the viewer's coordinates by the scene's rotation, or
    /// back by its inverse.
    func rotated(by q: simd_quatf) -> SplatViewerCropRegion {
        var turned = self
        turned.center = q.act(center)
        turned.rotation = simd_normalize(q * rotation)
        return turned
    }
}

/// The turns offered while a CAD model is selected, as buttons: a quarter turn about the vertical
/// either way, laid on its side, and back to the way it was imported.
public enum SplatViewerCADTurn: CaseIterable, Sendable {
    /// A quarter turn about the vertical, counter-clockwise seen from above.
    case left
    /// A quarter turn about the vertical, clockwise seen from above.
    case right
    /// Laid on its side: what is up now goes to the right as the camera sees it.
    case layDown
    /// Back to the way it was imported: upright and facing as in the CAD file.
    case resetRotation

    /// The box after this turn, about its center. `camera` gives the viewer's up (turned with the
    /// scene) and the side `layDown` falls to.
    func applied(to box: SplatViewerCropRegion, camera: SplatViewerCamera) -> SplatViewerCropRegion {
        let up = camera.up
        let turn: simd_quatf
        switch self {
        case .left:
            turn = simd_quatf(angle: .pi / 2, axis: up)
        case .right:
            turn = simd_quatf(angle: -.pi / 2, axis: up)
        case .layDown:
            var right = camera.axis(0)
            right -= up * simd_dot(right, up)
            guard simd_length(right) > 1e-4 else { return box }
            // Up turns onto the level right, about the level direction the camera looks along.
            turn = simd_quatf(from: up, to: simd_normalize(right))
        case .resetRotation:
            // Standing as in the CAD file, in the viewer.
            var reset = box
            reset.rotation = camera.sceneRotation.inverse
            return reset
        }
        var turned = box
        turned.rotation = simd_normalize(turn * box.rotation)
        return turned
    }

}

/// Which touches take hold of a CAD model. Small or far away it is hard to hit exactly, so a touch
/// on or near the box around it counts too, as Apple's AR guidelines advise for objects that are
/// hard to touch precisely; once selected, anywhere in its (orange) box does. A model that fills much
/// of the view is only taken by touching it, so the camera can still be moved around it.
enum SplatViewerCADReach {
    /// Margin round the box on screen that still counts, as a share of the view's shorter side.
    static let margin: Float = 0.04
    /// An unselected model smaller than this share of the view (its box's longer side) is taken by a
    /// touch near its box; a larger one only by a touch on it.
    static let smallShare: Float = 0.25

    /// The box's outline on screen: the bounds of its corners (normalized coordinates), or nil when a
    /// corner is behind the camera.
    static func screenBounds(of box: SplatViewerCropRegion, camera: SplatViewerCamera) -> (lo: SIMD2<Float>, hi: SIMD2<Float>)? {
        var lo = SIMD2<Float>(repeating: .infinity), hi = SIMD2<Float>(repeating: -.infinity)
        let h = box.halfExtent
        for k in 0..<8 {
            let offset = SIMD3<Float>(k & 1 == 0 ? -h.x : h.x, k & 2 == 0 ? -h.y : h.y, k & 4 == 0 ? -h.z : h.z)
            guard let p = camera.project(cameraSpace: camera.cameraSpace(box.center + box.rotation.act(offset)))
            else { return nil }
            lo = simd_min(lo, p)
            hi = simd_max(hi, p)
        }
        return (lo, hi)
    }

    /// Whether a touch (normalized) takes hold of the model: `onModel` is whether it lands on the
    /// model itself.
    static func takes(
        _ touch: SIMD2<Float>, box: SplatViewerCropRegion, selected: Bool, onModel: Bool,
        camera: SplatViewerCamera, viewSize: SIMD2<Float>
    ) -> Bool {
        if onModel { return true }
        guard let bounds = screenBounds(of: box, camera: camera) else { return false }
        let pixelsLo = bounds.lo * viewSize, pixelsHi = bounds.hi * viewSize
        let shorter = min(viewSize.x, viewSize.y)
        if !selected, max(pixelsHi.x - pixelsLo.x, pixelsHi.y - pixelsLo.y) >= smallShare * shorter {
            return false
        }
        let pad = SIMD2<Float>(repeating: margin * shorter)
        let p = touch * viewSize
        return all(p .>= pixelsLo - pad) && all(p .<= pixelsHi + pad)
    }
}
