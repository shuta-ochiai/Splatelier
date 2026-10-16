// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Measurement (distance between 2 points) display. Measurement points are kept in world
// coordinates; on every draw the renderer projects them to the screen with the same camera and
// hands them to `SplatMeasureProjection`. Lines and labels are overlaid on the splats with a
// SwiftUI Canvas (always in front). Capture data coordinates are ARKit world (meters), so distances
// are true scale as is.

#if os(iOS) || os(macOS)

import SwiftUI
import simd

/// On-screen positions of measurement lines. They change every frame during camera moves, so they
/// are kept apart from `SplatViewerModel` and only the overlay Canvas is invalidated.
@MainActor
final class SplatMeasureProjection: ObservableObject {
    struct Segment: Equatable {
        /// On-screen endpoints (normalized coordinates, top-left 0 / bottom-right 1, after clipping
        /// at near).
        var a: SIMD2<Float>
        var b: SIMD2<Float>
        var label: String
    }

    struct Frame: Equatable {
        var segments: [Segment] = []
        /// Visible measurement points.
        var points: [SIMD2<Float>] = []
    }

    @Published private(set) var frame = Frame()

    func update(_ newFrame: Frame) {
        if newFrame != frame { frame = newFrame }
    }

    /// Projects measurement points (each pair is one line; with an odd count the last is just a
    /// start point) with `camera`.
    nonisolated static func frame(points: [SIMD3<Float>], camera: SplatViewerCamera) -> Frame {
        let cam = points.map(camera.cameraSpace)
        var frame = Frame()
        frame.points = cam.compactMap(camera.project(cameraSpace:))
        for k in stride(from: 0, to: points.count - 1, by: 2) {
            guard let clipped = clipToNear(cam[k], cam[k + 1]),
                  let pa = camera.project(cameraSpace: clipped.0),
                  let pb = camera.project(cameraSpace: clipped.1)
            else { continue }
            frame.segments.append(
                Segment(a: pa, b: pb, label: formatDistance(simd_distance(points[k], points[k + 1]))))
        }
        return frame
    }

    /// Clips a camera-space segment to the front side of the near plane (clips slightly beyond near
    /// so projection doesn't flip it).
    nonisolated static func clipToNear(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> (SIMD3<Float>, SIMD3<Float>)? {
        let plane = SplatViewerCamera.near * 1.01
        let fa = -a.z - plane, fb = -b.z - plane
        if fa < 0 && fb < 0 { return nil }
        if fa >= 0 && fb >= 0 { return (a, b) }
        let cut = a + (b - a) * (fa / (fa - fb))
        return fa < 0 ? (cut, b) : (a, cut)
    }

    /// Distance display (mm below 1 cm, cm below 1 m, m otherwise).
    nonisolated static func formatDistance(_ meters: Float) -> String {
        if meters < 0.01 {
            String(format: "%.1f mm", meters * 1000)
        } else if meters < 1 {
            String(format: "%.1f cm", meters * 100)
        } else {
            String(format: "%.2f m", meters)
        }
    }
}

/// Overlay of measurement lines, points and distance labels. Touches pass through to the MTKView
/// below.
struct SplatMeasureOverlay: View {
    @ObservedObject var projection: SplatMeasureProjection

    var body: some View {
        Canvas { context, size in
            let frame = projection.frame
            func point(_ p: SIMD2<Float>) -> CGPoint {
                CGPoint(x: CGFloat(p.x) * size.width, y: CGFloat(p.y) * size.height)
            }
            for segment in frame.segments {
                var path = Path()
                path.move(to: point(segment.a))
                path.addLine(to: point(segment.b))
                context.stroke(path, with: .color(.black.opacity(0.55)), lineWidth: 4)
                context.stroke(path, with: .color(.yellow), lineWidth: 2)
            }
            for p in frame.points {
                let c = point(p)
                let dot = Path(ellipseIn: CGRect(x: c.x - 5, y: c.y - 5, width: 10, height: 10))
                context.fill(dot, with: .color(.yellow))
                context.stroke(dot, with: .color(.black.opacity(0.7)), lineWidth: 1.5)
            }
            for segment in frame.segments {
                let a = point(segment.a), b = point(segment.b)
                let mid = CGPoint(x: (a.x + b.x) / 2, y: (a.y + b.y) / 2)
                let text = context.resolve(
                    Text(segment.label)
                        .font(.caption.monospacedDigit().weight(.semibold))
                        .foregroundStyle(Color.black))
                let textSize = text.measure(in: size)
                let rect = CGRect(
                    x: mid.x - textSize.width / 2 - 7, y: mid.y - textSize.height / 2 - 3,
                    width: textSize.width + 14, height: textSize.height + 6)
                context.fill(Capsule().path(in: rect), with: .color(.yellow))
                context.draw(text, at: mid)
            }
        }
        .allowsHitTesting(false)
    }
}

#endif // os(iOS) || os(macOS)
