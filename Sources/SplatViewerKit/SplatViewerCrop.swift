// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Crop tool. A box or a sphere in world coordinates keeps or removes the foreground splats inside
// it. Its handles are dragged on screen: each handle moves along the world direction whose screen
// movement matches the finger, so the handle stays under it. Applying a crop only filters indices
// into the loaded splats, so undo costs no copies. Dragging is screen-space math, so it is
// unit-tested without a device.

#if os(iOS) || os(macOS)

import Foundation
import SplatIO
import SwiftUI
import simd

enum SplatViewerCropShape: String, CaseIterable, Identifiable {
    case box
    case sphere
    /// A loop drawn on screen (see `SplatViewerLasso`). It has no region in the world.
    case lasso

    var id: String { rawValue }

    var title: String {
        switch self {
        case .box: "Box"
        case .sphere: "Sphere"
        case .lasso: "Lasso"
        }
    }

    var systemImage: String {
        switch self {
        case .box: "cube"
        case .sphere: "circle"
        case .lasso: "lasso"
        }
    }
}

/// A grab point of the crop region.
enum SplatViewerCropHandle: Equatable {
    /// Middle of one box face, along the box's own axis (0...2) and which side (+1 or -1).
    case face(axis: Int, side: Float)
    /// Center of the box or the sphere, which moves the whole region.
    case center
    /// A point on the ring around one of the box's axes, which turns the box about that axis.
    /// `angle` is where on the ring it was grabbed, in the box's own frame.
    case ring(axis: Int, angle: Float)
    /// Sphere surface on the camera's right, which drags the radius.
    case radius
}

struct SplatViewerCropRegion: Equatable {
    var shape: SplatViewerCropShape
    var center: SIMD3<Float>
    /// Box half sizes along its own X, Y and Z.
    var halfExtent: SIMD3<Float>
    /// Orientation of the box. A sphere has none.
    var rotation = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    /// Sphere radius.
    var radius: Float

    /// Smallest size a handle can shrink the region to (meters for capture data).
    static let minimumSize: Float = 0.01

    func contains(_ p: SIMD3<Float>) -> Bool {
        switch shape {
        case .box:
            let d = simd_abs(rotation.inverse.act(p - center))
            return d.x <= halfExtent.x && d.y <= halfExtent.y && d.z <= halfExtent.z
        case .sphere:
            return simd_distance_squared(p, center) <= radius * radius
        case .lasso:
            return false
        }
    }

    /// The kept splats inside the region, as indices into `kept`. `position` gives a splat's center
    /// by its index in the loaded splats.
    func insideIndices(kept: [UInt32], position: (UInt32) -> SIMD3<Float>) -> Set<Int32> {
        var inside = Set<Int32>()
        for i in kept.indices where contains(position(kept[i])) { inside.insert(Int32(i)) }
        return inside
    }

    /// The same region as the other shape. A box becomes a sphere with its largest half size; a
    /// sphere becomes a box with its radius on every side.
    func converted(to shape: SplatViewerCropShape) -> SplatViewerCropRegion {
        guard shape != self.shape else { return self }
        switch shape {
        case .lasso:
            // The region keeps its box or sphere, so coming back from Lasso finds it as it was.
            return self
        case .box:
            return SplatViewerCropRegion(
                shape: .box, center: center, halfExtent: SIMD3(repeating: radius), radius: radius)
        case .sphere:
            let largest = max(halfExtent.x, max(halfExtent.y, halfExtent.z))
            return SplatViewerCropRegion(
                shape: .sphere, center: center, halfExtent: halfExtent, radius: largest)
        }
    }

    /// Box around the 2nd–98th percentile of each axis, so a few stray splats don't inflate it.
    /// The sphere with the same center and the largest half size.
    static func initial(shape: SplatViewerCropShape, positions: [SIMD3<Float>]) -> SplatViewerCropRegion {
        guard !positions.isEmpty else {
            return SplatViewerCropRegion(shape: shape, center: .zero, halfExtent: SIMD3(repeating: 1), radius: 1)
        }
        var lo = SIMD3<Float>.zero
        var hi = SIMD3<Float>.zero
        for axis in 0..<3 {
            let values = positions.map { $0[axis] }.sorted()
            lo[axis] = values[Int(0.02 * Float(values.count - 1))]
            hi[axis] = values[Int(0.98 * Float(values.count - 1))]
        }
        let box = SplatViewerCropRegion(
            shape: .box, center: (lo + hi) / 2, halfExtent: pointwiseMax(SIMD3(repeating: minimumSize), (hi - lo) / 2),
            radius: 1)
        return box.converted(to: shape)
    }

    /// Handles to draw and grab, with their world positions. The camera decides where the sphere's
    /// radius handle sits.
    func handles(camera: SplatViewerCamera) -> [(handle: SplatViewerCropHandle, position: SIMD3<Float>)] {
        switch shape {
        case .box:
            let faces = (0..<3).flatMap { axis in
                [Float(1), -1].map { side in
                    (SplatViewerCropHandle.face(axis: axis, side: side),
                     center + axisDirection(axis) * (side * halfExtent[axis]))
                }
            }
            return faces + [(.center, center)]
        case .sphere:
            return [
                (.center, center),
                (.radius, center + camera.axis(0) * radius),
            ]
        case .lasso:
            return []
        }
    }

    /// The region after dragging `handle` by `pixels` on screen. `viewSize` is the drawable size in
    /// pixels, which the projection needs.
    func dragged(
        _ handle: SplatViewerCropHandle, byPixels d: SIMD2<Float>,
        camera: SplatViewerCamera, viewSize: SIMD2<Float>
    ) -> SplatViewerCropRegion {
        var region = self
        switch (shape, handle) {
        case let (.box, .face(axis, side)):
            let direction = axisDirection(axis)
            let at = center + direction * (side * halfExtent[axis])
            guard let column = screenColumn(at: at, along: direction, camera: camera, viewSize: viewSize)
            else { return self }
            let m = simd_dot(d, column) / simd_dot(column, column)
            // The opposite face stays where it is; only the grabbed face moves. `lo` and `hi` are
            // offsets from the center along the box's axis.
            var lo = -halfExtent[axis]
            var hi = halfExtent[axis]
            if side > 0 {
                hi += m
                hi = max(hi, lo + 2 * Self.minimumSize)
            } else {
                lo += m
                lo = min(lo, hi - 2 * Self.minimumSize)
            }
            region.center = center + direction * ((lo + hi) / 2)
            region.halfExtent[axis] = (hi - lo) / 2
        case (.box, .center), (.sphere, .center):
            // Moves the center in the screen plane: the camera's right and up, solved by least
            // squares so the center follows the drag.
            let right = camera.axis(0), up = camera.axis(1)
            guard let c0 = screenColumn(at: center, along: right, camera: camera, viewSize: viewSize),
                  let c1 = screenColumn(at: center, along: up, camera: camera, viewSize: viewSize)
            else { return self }
            let a = simd_dot(c0, c0), b = simd_dot(c0, c1), c = simd_dot(c1, c1)
            let det = a * c - b * b
            guard abs(det) > 1e-6 else { return self }
            let r0 = simd_dot(c0, d), r1 = simd_dot(c1, d)
            let m0 = (c * r0 - b * r1) / det
            let m1 = (a * r1 - b * r0) / det
            region.center = center + right * m0 + up * m1
        case let (.box, .ring(axis, angle)):
            // The grabbed point of the ring follows the finger along the ring: the box turns about
            // the axis by the arc the finger covers, divided by the radius of the ring.
            let around = axisDirection(axis)
            let p = ringPoint(axis: axis, angle: angle)
            let tangent = simd_cross(around, p - center)
            guard simd_length(tangent) > 1e-6,
                  let column = screenColumn(
                    at: p, along: simd_normalize(tangent), camera: camera, viewSize: viewSize)
            else { return self }
            let m = simd_dot(d, column) / simd_dot(column, column)
            let turn = simd_quatf(angle: m / ringRadius, axis: around)
            region.rotation = simd_normalize(turn * rotation)
        case (.sphere, .radius):
            let right = camera.axis(0)
            let at = center + right * radius
            guard let column = screenColumn(at: at, along: right, camera: camera, viewSize: viewSize)
            else { return self }
            let m = simd_dot(d, column) / simd_dot(column, column)
            region.radius = max(Self.minimumSize, radius + m)
        default:
            return self
        }
        return region
    }

    /// Radius of the rings: the half diagonal, so they clear the whole box.
    var ringRadius: Float { max(simd_length(halfExtent), Self.minimumSize) }

    /// World position of the point on the ring around `axis` at `angle`. The ring lies in the plane
    /// of the other two axes, and an increasing angle turns the right-handed way about `axis`.
    func ringPoint(axis: Int, angle: Float) -> SIMD3<Float> {
        let u = axisDirection((axis + 1) % 3), v = axisDirection((axis + 2) % 3)
        return center + (u * cosf(angle) + v * sinf(angle)) * ringRadius
    }

    /// The part of a ring that is drawn and grabbed: angles and world points. It is fixed on the
    /// box, around the direction of its +X +Y +Z corner, so the three arcs gather at that corner,
    /// stay put when the camera moves and turn with the box.
    func ringArc(axis: Int) -> [(angle: Float, point: SIMD3<Float>)] {
        (0...Self.ringArcSteps).map { k in
            let angle = Self.ringArcMiddle + (2 * Float(k) / Float(Self.ringArcSteps) - 1) * Self.ringArcHalfSpan
            return (angle, ringPoint(axis: axis, angle: angle))
        }
    }

    /// Where on each ring the arc is centered: halfway between the two other axes.
    static let ringArcMiddle: Float = .pi / 4

    /// Half the angle one ring's arc covers.
    static let ringArcHalfSpan: Float = 0.6
    static let ringArcSteps = 16

    /// The ring arc nearest to a touch (normalized coordinates) and how far it is, in pixels. nil
    /// for anything but a box, or when no arc is on screen.
    func nearestRing(
        to touch: SIMD2<Float>, camera: SplatViewerCamera, viewSize: SIMD2<Float>
    ) -> (handle: SplatViewerCropHandle, distance: Float)? {
        guard shape == .box else { return nil }
        let target = touch * viewSize
        var best: (handle: SplatViewerCropHandle, distance: Float)?
        for axis in 0..<3 {
            let arc = ringArc(axis: axis)
            let points = arc.map { camera.project(cameraSpace: camera.cameraSpace($0.point)).map { $0 * viewSize } }
            for k in 0..<Self.ringArcSteps {
                guard let a = points[k], let b = points[k + 1] else { continue }
                let ab = b - a
                let t = min(max(simd_dot(target - a, ab) / max(simd_dot(ab, ab), 1e-6), 0), 1)
                let distance = simd_distance(target, a + ab * t)
                if distance < (best?.distance ?? .infinity) {
                    let angle = arc[k].angle + (arc[k + 1].angle - arc[k].angle) * t
                    best = (.ring(axis: axis, angle: angle), distance)
                }
            }
        }
        return best
    }

    /// World direction of one of the box's axes.
    private func axisDirection(_ axis: Int) -> SIMD3<Float> {
        var u = SIMD3<Float>.zero
        u[axis] = 1
        return rotation.act(u)
    }

    /// Screen movement in pixels for a unit world movement of `at` along `u`. nil if it can't be
    /// measured (e.g. the point is behind the near plane, or `u` points straight at the camera).
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
}

/// Which splats are left after the crops applied so far. Holds indices into the foreground splats as
/// loaded, so undo and restore only swap index arrays.
struct SplatViewerCropEdits {
    enum Result: Equatable {
        case applied
        /// The region removes nothing.
        case unchanged
        /// The crop would leave no splats.
        case emptied
    }

    private(set) var kept: [UInt32]
    private var history: [[UInt32]] = []

    init(count: Int) {
        kept = (0..<count).map(UInt32.init)
    }

    var isEdited: Bool { !history.isEmpty }
    var canUndo: Bool { !history.isEmpty }

    /// Applies a crop: keeps the splats for which `keep` is true.
    mutating func apply(_ keep: (UInt32) -> Bool) -> Result {
        let next = kept.filter(keep)
        guard !next.isEmpty else { return .emptied }
        guard next.count != kept.count else { return .unchanged }
        history.append(kept)
        kept = next
        return .applied
    }

    mutating func undo() {
        guard let previous = history.popLast() else { return }
        kept = previous
    }

    /// Back to every splat as loaded.
    mutating func restoreAll() {
        guard let first = history.first else { return }
        kept = first
        history = []
    }
}

/// Screen positions of the crop region's outline and handles. Recomputed on every draw, like the
/// measurement lines, so the overlay follows the camera.
struct SplatCropFrame: Equatable {
    struct Segment: Equatable {
        /// Normalized screen coordinates (top-left 0 / bottom-right 1).
        var a: SIMD2<Float>
        var b: SIMD2<Float>
    }

    var segments: [Segment] = []
    var handles: [SIMD2<Float>] = []
    /// The handle that moves the region, drawn apart from the others (box and sphere alike).
    var moveHandles: [SIMD2<Float>] = []
    /// The arcs of the three rings that turn the box, one per box axis, as outline segments.
    var rings: [[Segment]] = []

    nonisolated static func make(region: SplatViewerCropRegion, camera: SplatViewerCamera) -> SplatCropFrame {
        var frame = SplatCropFrame()
        for (a, b) in outline(of: region) {
            let ca = camera.cameraSpace(a), cb = camera.cameraSpace(b)
            guard let clipped = SplatMeasureProjection.clipToNear(ca, cb),
                  let pa = camera.project(cameraSpace: clipped.0),
                  let pb = camera.project(cameraSpace: clipped.1)
            else { continue }
            frame.segments.append(Segment(a: pa, b: pb))
        }
        if region.shape == .box {
            frame.rings = (0..<3).map { axis in
                let points = region.ringArc(axis: axis).map(\.point)
                return zip(points, points.dropFirst()).compactMap { a, b in
                    guard let clipped = SplatMeasureProjection.clipToNear(
                        camera.cameraSpace(a), camera.cameraSpace(b)),
                          let pa = camera.project(cameraSpace: clipped.0),
                          let pb = camera.project(cameraSpace: clipped.1)
                    else { return nil }
                    return Segment(a: pa, b: pb)
                }
            }
        }
        for item in region.handles(camera: camera) {
            guard let p = camera.project(cameraSpace: camera.cameraSpace(item.position)) else { continue }
            if item.handle == .center { frame.moveHandles.append(p) } else { frame.handles.append(p) }
        }
        return frame
    }

    /// World segments of the outline: 12 box edges, or 3 great circles of the sphere.
    nonisolated private static func outline(of region: SplatViewerCropRegion) -> [(SIMD3<Float>, SIMD3<Float>)] {
        switch region.shape {
        case .box:
            let h = region.halfExtent, c = region.center
            let corners: [SIMD3<Float>] = (0..<8).map { i in
                c + SIMD3(
                    (i & 1 == 0 ? -1 : 1) * h.x,
                    (i & 2 == 0 ? -1 : 1) * h.y,
                    (i & 4 == 0 ? -1 : 1) * h.z)
            }.map { region.center + region.rotation.act($0 - region.center) }
            // Corners that differ in one bit are joined by an edge.
            var edges: [(SIMD3<Float>, SIMD3<Float>)] = []
            for i in 0..<8 {
                for bit in [1, 2, 4] where i & bit == 0 {
                    edges.append((corners[i], corners[i | bit]))
                }
            }
            return edges
        case .sphere:
            let steps = 48
            var edges: [(SIMD3<Float>, SIMD3<Float>)] = []
            // Great circles in the XY, XZ and YZ planes.
            for (u, v) in [(0, 1), (0, 2), (1, 2)] {
                func point(_ t: Float) -> SIMD3<Float> {
                    var p = SIMD3<Float>.zero
                    p[u] = cos(t) * region.radius
                    p[v] = sin(t) * region.radius
                    return region.center + p
                }
                for k in 0..<steps {
                    let t0 = 2 * Float.pi * Float(k) / Float(steps)
                    let t1 = 2 * Float.pi * Float(k + 1) / Float(steps)
                    edges.append((point(t0), point(t1)))
                }
            }
            return edges
        case .lasso:
            return []
        }
    }
}

/// Outline and handles of the crop region on screen, observed by the overlay.
@MainActor
final class SplatCropProjection: ObservableObject {
    @Published private(set) var frame = SplatCropFrame()
    /// The Lasso loop being drawn, in normalized screen coordinates. It lives on the screen, not in
    /// the world, and is empty once the finger lifts and the loop has become a selection.
    @Published private(set) var lasso: [SIMD2<Float>] = []

    func setLasso(_ points: [SIMD2<Float>]) {
        lasso = points
    }

    func update(_ newFrame: SplatCropFrame) {
        if newFrame != frame { frame = newFrame }
    }
}

/// Draws the crop outline and handles. Touches pass through to the MTKView below.
struct SplatCropOverlay: View {
    @ObservedObject var projection: SplatCropProjection

    /// Colours of the rings around the box's X, Y and Z axes.
    private static let ringColors: [Color] = [.red, .green, .blue]

    var body: some View {
        Canvas { context, size in
            func point(_ p: SIMD2<Float>) -> CGPoint {
                CGPoint(x: CGFloat(p.x) * size.width, y: CGFloat(p.y) * size.height)
            }
            var outline = Path()
            for segment in projection.frame.segments {
                outline.move(to: point(segment.a))
                outline.addLine(to: point(segment.b))
            }
            let lasso = projection.lasso
            // Once the end meets the start the loop closes and takes the colour of the selection.
            let joined = SplatViewerLasso.isJoined(lasso, aspect: Float(size.width / max(size.height, 1)))
            if let first = lasso.first {
                outline.move(to: point(first))
                for p in lasso.dropFirst() { outline.addLine(to: point(p)) }
                if joined {
                    outline.closeSubpath()
                    context.fill(outline, with: .color(SplatViewerFloaterHighlight.swiftUIColor.opacity(0.18)))
                }
            }
            context.stroke(outline, with: .color(.black.opacity(0.5)), lineWidth: 3)
            context.stroke(
                outline, with: .color(joined ? SplatViewerFloaterHighlight.swiftUIColor : .white),
                lineWidth: joined ? 2.5 : 1.5)
            for (axis, segments) in projection.frame.rings.enumerated() {
                var ring = Path()
                for segment in segments {
                    ring.move(to: point(segment.a))
                    ring.addLine(to: point(segment.b))
                }
                context.stroke(ring, with: .color(.black.opacity(0.5)), lineWidth: 4)
                context.stroke(ring, with: .color(Self.ringColors[axis]), lineWidth: 2.5)
            }
            for h in projection.frame.handles {
                let c = point(h)
                let dot = Path(ellipseIn: CGRect(x: c.x - 9, y: c.y - 9, width: 18, height: 18))
                context.fill(dot, with: .color(.white))
                context.stroke(dot, with: .color(.black.opacity(0.6)), lineWidth: 1.5)
            }
            for h in projection.frame.moveHandles {
                let c = point(h)
                let dot = Path(ellipseIn: CGRect(x: c.x - 9, y: c.y - 9, width: 18, height: 18))
                context.fill(dot, with: .color(.cyan))
                context.stroke(dot, with: .color(.black.opacity(0.6)), lineWidth: 1.5)
            }
        }
        .allowsHitTesting(false)
    }
}

/// Writes splats as a PLY the same way the file was read (degree of the first splat). Runs off the
/// main actor, so the splat arrays are only read.
/// The formats an edited splat file can be shared in, besides the PLY the edit writes. The app
/// supplies the conversions, since the exporters live outside the viewer.
public struct SplatViewerCropExport: Sendable {
    public struct Format: Sendable, Identifiable {
        public let title: String
        /// File extension without the dot, e.g. "spz" or "compressed.ply".
        public let fileExtension: String
        public var id: String { fileExtension }

        public init(title: String, fileExtension: String) {
            self.title = title
            self.fileExtension = fileExtension
        }
    }

    public let formats: [Format]
    /// Converts the edited PLY to the given extension and returns the file to share.
    public let convert: @Sendable (_ plyURL: URL, _ fileExtension: String) async throws -> URL

    public init(
        formats: [Format],
        convert: @escaping @Sendable (_ plyURL: URL, _ fileExtension: String) async throws -> URL
    ) {
        self.formats = formats
        self.convert = convert
    }
}

nonisolated func writeEditedPLY(_ points: [SplatPoint], name: String) async throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(name)-edited.ply")
    try? FileManager.default.removeItem(at: url)
    let degree = points.first?.color.shDegree ?? .sh0
    let writer = try SplatPLYSceneWriter(toFileAtPath: url.path)
    try await writer.start(sphericalHarmonicDegree: UInt(degree.rawValue), pointCount: points.count)
    try await writer.write(points)
    try await writer.close()
    return url
}

#endif // os(iOS) || os(macOS)
