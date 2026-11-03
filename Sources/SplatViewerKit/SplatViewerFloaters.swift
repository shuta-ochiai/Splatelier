// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Floater tool. Selecting by hand and detecting are separate algorithms that only share the
// selection they add to.
//
// By hand (`SplatViewerFloaterBrush`): a tap selects the splats within a brush radius of the first
// thing under the finger, whatever they are joined to, and a tap on the selection lets them go.
// Many floaters are joined to the scene by faint splats, so a brush reaches them where clusters
// can't.
//
// Detect (`SplatViewerFloaterDetector`): groups the foreground splats into clusters of touching
// voxels. Each splat marks the voxels reached by its center and its largest radius (capped at one
// voxel), and occupied voxels that share a face, an edge or a corner join one cluster. The largest
// cluster is the scene, and the small clusters near it are floaters. Clusters far outside the scene
// are never detected, so a background shell that a file didn't split off is left alone (the viewer
// already keeps the split-off background out of the foreground).
//
// While Selection size is being changed, a circle shows the brush on screen: around the last tapped
// point (or the look-at point) at its real size there.
//
// Both are plain arithmetic, so they are unit-tested without a device.

#if os(iOS) || os(macOS)

import Foundation
import SplatIO
import SwiftUI
import simd

/// Selecting by hand.
enum SplatViewerFloaterBrush {
    /// Brush radius as a fraction of the scene's size (the diagonal of its 2nd–98th percentile box).
    static let sizeRange: ClosedRange<Float> = 0.005...0.1
    static let defaultSize: Float = 0.03

    /// Splats with their center within `radius` of `center`, as indices into `positions`.
    static func splats(within radius: Float, of center: SIMD3<Float>, positions: [SIMD3<Float>]) -> [Int32] {
        let r2 = radius * radius
        return positions.indices.compactMap { simd_distance_squared(positions[$0], center) <= r2 ? Int32($0) : nil }
    }

    /// The selection after a tap that brushed `splats`. Brushing what is mostly selected already
    /// (the highlight) lets those splats go; anything else adds them.
    static func toggled(_ selection: Set<Int32>, by splats: [Int32]) -> Set<Int32> {
        let selected = splats.lazy.filter { selection.contains($0) }.count
        return selected * 2 > splats.count ? selection.subtracting(splats) : selection.union(splats)
    }

    /// Diagonal of the 2nd–98th percentile box of the splats.
    static func sceneSize(of positions: [SIMD3<Float>]) -> Float {
        let (lo, hi) = percentileBox(positions)
        return simd_length(hi - lo)
    }
}

/// Detect.
struct SplatViewerFloaterDetector: Sendable {
    /// Voxels along the diagonal of the 2nd–98th percentile box of the splats.
    static let divisions: Float = 256
    /// Detect takes clusters with fewer splats than this fraction of all of them.
    static let floaterFraction: Float = 0.01
    /// Detect takes clusters whose centroid is inside the scene's 2nd–98th percentile box grown on
    /// every side by this fraction of its longest side (a flat scene still gets room above it).
    static let sceneMargin: Float = 0.5

    /// Edge of a voxel in world units (meters for capture data).
    let voxelSize: Float
    /// Cluster of each splat, in the order the positions were given.
    let clusterOfSplat: [Int32]
    /// Number of splats in each cluster.
    let clusterSizes: [Int]
    /// Mean splat position of each cluster.
    let clusterCentroids: [SIMD3<Float>]
    /// The scene: the cluster with the most splats. nil when there are no splats.
    let largest: Int32?
    /// 2nd–98th percentile box of the scene's splats.
    let sceneLow: SIMD3<Float>
    let sceneHigh: SIMD3<Float>

    /// - Parameters:
    ///   - radii: Largest radius of each splat (1σ), same order as `positions`.
    ///   - voxelSize: nil picks it from the extent of the positions (see `divisions`).
    init(positions: [SIMD3<Float>], radii: [Float], voxelSize: Float? = nil) {
        precondition(positions.count == radii.count)
        let size = voxelSize ?? Self.voxelSize(for: positions)
        self.voxelSize = size

        // Occupied voxels, numbered in the order first seen.
        var voxelIndex: [UInt64: Int32] = [:]
        voxelIndex.reserveCapacity(positions.count)
        var voxels: [SIMD3<Int32>] = []
        func occupy(_ v: SIMD3<Int32>) -> Int32 {
            let key = Self.key(v)
            if let i = voxelIndex[key] { return i }
            let i = Int32(voxels.count)
            voxelIndex[key] = i
            voxels.append(v)
            return i
        }
        var centerVoxel = [Int32](repeating: 0, count: positions.count)
        for (i, p) in positions.enumerated() {
            let c = Self.voxel(p, size)
            centerVoxel[i] = occupy(c)
            // A splat larger than a voxel reaches its neighbours, so it bridges a sparse surface.
            let r = min(max(radii[i], 0), size)
            let lo = Self.voxel(p - r, size), hi = Self.voxel(p + r, size)
            guard lo != hi else { continue }
            for x in lo.x...hi.x {
                for y in lo.y...hi.y {
                    for z in lo.z...hi.z { _ = occupy(SIMD3(x, y, z)) }
                }
            }
        }

        // Union–find over the voxels. Each neighbour pair is visited once, from its lower side.
        var parent = (0..<Int32(voxels.count)).map { $0 }
        func find(_ a: Int32) -> Int32 {
            var r = a
            while parent[Int(r)] != r { r = parent[Int(r)] }
            var a = a
            while parent[Int(a)] != r {
                let next = parent[Int(a)]
                parent[Int(a)] = r
                a = next
            }
            return r
        }
        for (i, v) in voxels.enumerated() {
            for offset in Self.forwardNeighbours {
                guard let j = voxelIndex[Self.key(v &+ offset)] else { continue }
                let ri = find(Int32(i)), rj = find(j)
                if ri != rj { parent[Int(max(ri, rj))] = min(ri, rj) }
            }
        }

        // Clusters numbered 0... in the order their first splat appears.
        var clusterOfRoot: [Int32: Int32] = [:]
        var sizes: [Int] = []
        var sums: [SIMD3<Float>] = []
        var clusterOfSplat = [Int32](repeating: 0, count: positions.count)
        for (i, p) in positions.enumerated() {
            let root = find(centerVoxel[i])
            let cluster: Int32
            if let c = clusterOfRoot[root] {
                cluster = c
            } else {
                cluster = Int32(sizes.count)
                clusterOfRoot[root] = cluster
                sizes.append(0)
                sums.append(.zero)
            }
            clusterOfSplat[i] = cluster
            sizes[Int(cluster)] += 1
            sums[Int(cluster)] += p
        }
        self.clusterOfSplat = clusterOfSplat
        self.clusterSizes = sizes
        self.clusterCentroids = zip(sums, sizes).map { $0 / Float($1) }
        let largest = sizes.indices.max { sizes[$0] < sizes[$1] }.map(Int32.init)
        self.largest = largest

        if let largest {
            let scene = positions.indices.filter { clusterOfSplat[$0] == largest }.map { positions[$0] }
            (sceneLow, sceneHigh) = percentileBox(scene)
        } else {
            (sceneLow, sceneHigh) = (.zero, .zero)
        }
    }

    /// Clusters of the kept foreground splats, in the order of `kept`.
    init(splats: [SplatPoint], kept: [UInt32]) {
        self.init(
            positions: kept.map { splats[Int($0)].position },
            radii: kept.map { splats[Int($0)].scale.asLinearFloat.max() })
    }

    /// Small clusters near the scene, the floaters.
    var floaterClusters: [Int32] {
        let total = clusterSizes.reduce(0, +)
        let margin = SIMD3(repeating: (sceneHigh - sceneLow).max() * Self.sceneMargin)
        let low = sceneLow - margin, high = sceneHigh + margin
        return clusterSizes.indices.compactMap { i in
            let c = clusterCentroids[i]
            guard Int32(i) != largest,
                  Float(clusterSizes[i]) < Self.floaterFraction * Float(total),
                  all(c .>= low), all(c .<= high)
            else { return nil }
            return Int32(i)
        }
    }

    /// Splats in the floater clusters, as indices in the order the positions were given.
    var floaterSplats: [Int32] {
        let floaters = Set(floaterClusters)
        return clusterOfSplat.indices.compactMap { floaters.contains(clusterOfSplat[$0]) ? Int32($0) : nil }
    }

    /// The voxel size that puts `divisions` voxels along the diagonal of the 2nd–98th percentile box.
    static func voxelSize(for positions: [SIMD3<Float>]) -> Float {
        let (lo, hi) = percentileBox(positions)
        let diagonal = simd_length(hi - lo)
        return diagonal.isFinite && diagonal > 0 ? diagonal / divisions : 0.01
    }

    // MARK: Voxels

    /// 13 of the 26 neighbours: the ones after the voxel in x, then y, then z order.
    private static let forwardNeighbours: [SIMD3<Int32>] = {
        var offsets: [SIMD3<Int32>] = []
        for x in Int32(-1)...1 {
            for y in Int32(-1)...1 {
                for z in Int32(-1)...1 where (x, y, z) > (0, 0, 0) {
                    offsets.append(SIMD3(x, y, z))
                }
            }
        }
        return offsets
    }()

    /// 21 bits per axis, so voxel coordinates within ±2²⁰ stay distinct.
    private static let keyBias: Int32 = 1 << 20

    private static func voxel(_ p: SIMD3<Float>, _ size: Float) -> SIMD3<Int32> {
        // Room for the neighbour offsets on either side.
        let limit = Float(keyBias - 3)
        let clamped = simd_clamp(p / size, SIMD3(repeating: -limit), SIMD3(repeating: limit))
        return SIMD3<Int32>(clamped, rounding: .down)
    }

    private static func key(_ v: SIMD3<Int32>) -> UInt64 {
        let u = SIMD3<UInt64>(truncatingIfNeeded: v &+ keyBias)
        return u.x | (u.y << 21) | (u.z << 42)
    }

}

/// How the selection is drawn.
enum SplatViewerFloaterHighlight {
    /// Selected splats are drawn in a flat orange and made at least this opaque, so faint floaters
    /// show clearly.
    static let color = SIMD3<Float>(1, 0.42, 0.1)
    static let opacity: Float = 0.6

    /// The splat drawn as selected. The SH bands are kept in number (a chunk wants them alike) but
    /// zeroed, so the colour doesn't shift with the view.
    static func apply(to point: SplatPoint) -> SplatPoint {
        var p = point
        var coefficients = point.color.asSphericalHarmonicFloat
        if coefficients.isEmpty { coefficients = [.zero] }
        coefficients[0] = (color - 0.5) * SplatPoint.Color.INV_SH_C0
        for i in coefficients.indices.dropFirst() { coefficients[i] = .zero }
        p.color = .sphericalHarmonicFloat(coefficients)
        p.opacity = .linearFloat(max(point.opacity.asLinearFloat, opacity))
        return p
    }
}

/// Where the brush circle sits on screen. Recomputed on every draw while it is shown, like the crop
/// outline, so it follows the camera.
struct SplatBrushFrame: Equatable {
    /// Normalized screen coordinates (top-left 0 / bottom-right 1).
    var center: SIMD2<Float>
    /// Radius as a fraction of the view height.
    var radius: Float

    var isOnScreen: Bool { all(center .>= 0) && all(center .<= 1) }

    /// A brush of world radius `radius` around `anchor`, as `camera` sees it. nil if `anchor` is
    /// behind the camera.
    nonisolated static func make(anchor: SIMD3<Float>, radius: Float, camera: SplatViewerCamera) -> SplatBrushFrame? {
        let q = camera.cameraSpace(anchor)
        guard let center = camera.project(cameraSpace: q) else { return nil }
        // The view height spans 2·depth·tan(fovy/2) world units at the anchor's depth.
        return SplatBrushFrame(center: center, radius: radius / (2 * -q.z * tanf(camera.fovyRadians * 0.5)))
    }
}

/// The brush circle on screen, observed by the overlay. nil hides it.
@MainActor
final class SplatBrushProjection: ObservableObject {
    @Published private(set) var frame: SplatBrushFrame?

    func update(_ newFrame: SplatBrushFrame?) {
        if newFrame != frame { frame = newFrame }
    }
}

/// Draws the brush circle in the highlight colour. It grows in when shown and springs to each new
/// size. Touches pass through to the MTKView below.
struct SplatBrushOverlay: View {
    @ObservedObject var projection: SplatBrushProjection

    private static let tint: Color = {
        let c = SplatViewerFloaterHighlight.color
        return Color(red: Double(c.x), green: Double(c.y), blue: Double(c.z))
    }()

    var body: some View {
        GeometryReader { geometry in
            if let frame = projection.frame {
                let d = CGFloat(frame.radius) * geometry.size.height * 2
                Circle()
                    .fill(Self.tint.opacity(0.15))
                    .overlay(Circle().stroke(Self.tint, lineWidth: 2))
                    .frame(width: d, height: d)
                    .position(
                        x: CGFloat(frame.center.x) * geometry.size.width,
                        y: CGFloat(frame.center.y) * geometry.size.height)
                    .transition(.scale(scale: 0.5).combined(with: .opacity))
            }
        }
        .animation(.spring(duration: 0.25), value: projection.frame?.radius)
        .animation(.easeOut(duration: 0.2), value: projection.frame == nil)
        .allowsHitTesting(false)
    }
}

/// 2nd–98th percentile box of the positions, so a few strays don't stretch it.
private func percentileBox(_ positions: [SIMD3<Float>]) -> (SIMD3<Float>, SIMD3<Float>) {
    guard !positions.isEmpty else { return (.zero, .zero) }
    var lo = SIMD3<Float>.zero, hi = SIMD3<Float>.zero
    for axis in 0..<3 {
        let values = positions.map { $0[axis] }.sorted()
        lo[axis] = values[Int(0.02 * Float(values.count - 1))]
        hi[axis] = values[Int(0.98 * Float(values.count - 1))]
    }
    return (lo, hi)
}

#endif // os(iOS) || os(macOS)
