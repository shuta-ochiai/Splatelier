// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Viewer floater tool: the brush that selects by hand, the detector (clusters of touching voxels and
// what Detect selects), the front-depth ray test a tap uses, and the highlight.

import Foundation
import SplatIO
import Testing
import simd
@testable import SplatViewerKit

/// A 1 m square of splats on the floor, 1 cm apart: the scene.
private func floor() -> [SIMD3<Float>] {
    (0..<100).flatMap { i in (0..<100).map { j in SIMD3(Float(i) * 0.01, 0, Float(j) * 0.01) } }
}

/// A small blob of splats around `center`, 5 mm apart.
private func blob(at center: SIMD3<Float>, side: Int = 3) -> [SIMD3<Float>] {
    (0..<side).flatMap { i in
        (0..<side).flatMap { j in
            (0..<side).map { k in center + SIMD3(Float(i), Float(j), Float(k)) * 0.005 }
        }
    }
}

private func detector(_ positions: [SIMD3<Float>], radius: Float = 0.001) -> SplatViewerFloaterDetector {
    SplatViewerFloaterDetector(
        positions: positions, radii: Array(repeating: radius, count: positions.count), voxelSize: 0.02)
}

// MARK: Brush

/// The brush takes the splats within its radius of the tapped point, whatever they are joined to.
@Test
func brushSelectsSplatsWithinItsRadius() {
    let positions: [SIMD3<Float>] = [SIMD3(0, 0, 0), SIMD3(0.05, 0, 0), SIMD3(0.2, 0, 0)]
    #expect(SplatViewerFloaterBrush.splats(within: 0.1, of: SIMD3(0.01, 0, 0), positions: positions) == [0, 1])
    #expect(SplatViewerFloaterBrush.splats(within: 0.1, of: SIMD3(1, 0, 0), positions: positions).isEmpty)
}

/// Brushing new splats adds them; brushing what is mostly selected lets those splats go.
@Test
func brushOnTheSelectionLetsItGo() {
    let added = SplatViewerFloaterBrush.toggled([1], by: [2, 3])
    #expect(added == [1, 2, 3])
    #expect(SplatViewerFloaterBrush.toggled(added, by: [2, 3, 4]) == [1])
    // Half selected is not mostly selected, so it adds.
    #expect(SplatViewerFloaterBrush.toggled([2], by: [2, 3]) == [2, 3])
}

/// The scene size is the diagonal of the 2nd–98th percentile box, so a stray splat doesn't stretch
/// the brush.
@Test
func brushScalesWithTheSceneNotTheStrays() {
    let scene = floor()
    #expect(abs(SplatViewerFloaterBrush.sceneSize(of: scene) - 0.96 * sqrt(2)) < 0.03)
    #expect(abs(SplatViewerFloaterBrush.sceneSize(of: scene + [SIMD3(50, 50, 50)]) - 0.96 * sqrt(2)) < 0.03)
}

// MARK: Detect

/// The floor is one cluster, the largest; a blob floating above it is another, and Detect takes it.
@Test
func detectSelectsASmallBlobAboveTheScene() {
    let scene = floor()
    let floater = blob(at: SIMD3(0.5, 0.3, 0.5))
    let d = detector(scene + floater)
    let sceneCluster = d.clusterOfSplat[0]
    #expect(d.largest == sceneCluster)
    #expect(Set(d.clusterOfSplat[..<scene.count]) == [sceneCluster])
    let floaterClusters = Set(d.clusterOfSplat[scene.count...])
    #expect(floaterClusters.count == 1)
    #expect(d.floaterClusters == Array(floaterClusters))
    #expect(d.floaterSplats == Array(Int32(scene.count)..<Int32(scene.count + floater.count)))
}

/// A cluster far outside the scene, like a background shell left in the file, is not detected.
@Test
func detectLeavesClustersFarFromTheSceneAlone() {
    let scene = floor()
    let far = blob(at: SIMD3(30, 5, 30))
    let d = detector(scene + far)
    #expect(d.clusterOfSplat[scene.count] != d.largest)
    #expect(d.floaterSplats.isEmpty)
}

/// Clusters with as many splats as a real part of the scene are not detected.
@Test
func detectLeavesLargeClustersAlone() {
    let scene = floor()
    let part = blob(at: SIMD3(0.5, 0.3, 0.5), side: 6)
    #expect(Float(part.count) > SplatViewerFloaterDetector.floaterFraction * Float(scene.count + part.count))
    #expect(detector(scene + part).floaterSplats.isEmpty)
}

/// Splats larger than the gap between them reach across it and join one cluster.
@Test
func largeSplatsBridgeAGap() {
    let left = SIMD3<Float>(0, 0, 0), right = SIMD3<Float>(0.07, 0, 0)
    #expect(detector([left, right], radius: 0.001).clusterSizes.count == 2)
    #expect(detector([left, right], radius: 0.02).clusterSizes.count == 1)
}

@Test
func noSplatsMakeNoClusters() {
    let d = detector([])
    #expect(d.largest == nil)
    #expect(d.floaterSplats.isEmpty)
}

// MARK: Tap and highlight

/// The front depth stops at the first faint splats; the surface depth goes on to the opaque one.
@Test
func frontDepthStopsAtTheFirstFaintSplats() {
    let hits = [
        SplatViewerRayCaster.SplatHit(t: 3, alpha: 0.9),
        SplatViewerRayCaster.SplatHit(t: 1, alpha: 0.04),
        SplatViewerRayCaster.SplatHit(t: 1.1, alpha: 0.04),
    ]
    #expect(SplatViewerRayCaster.frontDepth(hits, opacity: 0.05) == 1.1)
    #expect(SplatViewerRayCaster.surfaceDepth(hits) == 3)
    #expect(SplatViewerRayCaster.frontDepth(hits, opacity: 0.99) == nil)
}

/// The highlight keeps the number of SH coefficients, zeroes the view-dependent ones and makes a
/// faint splat opaque enough to see.
@Test
func highlightKeepsTheShapeOfTheSplat() {
    let point = SplatPoint(
        position: SIMD3(1, 2, 3),
        color: .sphericalHarmonicFloat(Array(repeating: SIMD3(0.1, 0.2, 0.3), count: 4)),
        opacity: .linearFloat(0.05),
        scale: .linearFloat(SIMD3(repeating: 0.01)),
        rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1))
    let h = SplatViewerFloaterHighlight.apply(to: point)
    let sh = h.color.asSphericalHarmonicFloat
    #expect(sh.count == 4)
    #expect(sh.dropFirst().allSatisfy { $0 == .zero })
    #expect(simd_length(h.color.asSRGBFloat - SplatViewerFloaterHighlight.color) < 0.01)
    #expect(h.opacity.asLinearFloat == SplatViewerFloaterHighlight.opacity)
    #expect(h.position == point.position)
}

/// The brush circle sits on the anchor's screen position, and its radius is the world radius over
/// the view height at the anchor's depth. Behind the camera there is no circle.
@Test
func brushCircleMatchesTheViewAtItsDepth() {
    let camera = SplatViewerCamera(
        azimuth: 0, elevation: 0, distance: 5, center: .zero, fovyRadians: 1, aspect: 1.5)
    let frame = SplatBrushFrame.make(anchor: .zero, radius: 0.2, camera: camera)
    #expect(frame.map { simd_distance($0.center, SIMD2(0.5, 0.5)) < 1e-5 } == true)
    #expect(frame.map { abs($0.radius - 0.2 / (2 * 5 * tanf(0.5))) < 1e-5 } == true)
    #expect(frame?.isOnScreen == true)
    #expect(SplatBrushFrame.make(anchor: camera.eye + camera.axis(2), radius: 0.2, camera: camera) == nil)
}

/// The tint pulls the colour toward the orange but keeps the splat's opacity and its other bands.
@Test
func tintKeepsOpacityAndBands() {
    let point = SplatPoint(
        position: SIMD3(1, 2, 3),
        color: .sphericalHarmonicFloat([SIMD3(0.5, 0.5, 0.5), SIMD3(0.1, 0.2, 0.3)]),
        opacity: .linearFloat(0.9),
        scale: .linearFloat(SIMD3(repeating: 0.01)),
        rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1))
    let t = SplatViewerFloaterHighlight.tint(point)
    #expect(t.opacity.asLinearFloat == point.opacity.asLinearFloat)
    #expect(t.color.asSphericalHarmonicFloat.dropFirst() == point.color.asSphericalHarmonicFloat.dropFirst())
    let own = point.color.asSRGBFloat, tinted = t.color.asSRGBFloat
    let orange = SplatViewerFloaterHighlight.color
    // Between its own colour and the orange, not either end.
    #expect(simd_length(tinted - orange) < simd_length(own - orange))
    #expect(simd_length(tinted - own) > 0.05)
    #expect(simd_length(tinted - orange) > 0.05)
}
