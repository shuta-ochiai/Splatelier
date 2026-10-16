// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Live AR overlay during capture. ARSCNView does not take over session.delegate, so it can coexist
// with ARKitCaptureSession's capture delegate.
//
// When the live point cloud is MVS, a translucent white is laid over the camera feed and the splats
// of captured areas are drawn on top in their real colors (CaptureSurfelRenderer, drawn directly
// with Metal in SceneKit's render pass). Color fills in over the white guide where you have
// captured. Splats are larger and blurrier where an area was only captured from far away, and
// sharpen as you capture it up close.
//
// For feature points / LiDAR, coverage splats are drawn as point primitives. Before drawing the
// points, only their depth is written, hiding points that lie behind points of a nearer surface
// (the back side of an object, surfaces behind an object).

#if os(iOS)

import SwiftUI
import ARKit
import SceneKit
import Capture

struct ARCaptureView: UIViewRepresentable {
    let captureSession: ARKitCaptureSession

    func makeCoordinator() -> Coordinator {
        Coordinator(captureSession: captureSession)
    }

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView(frame: .zero)
        view.scene = SCNScene()
        view.rendersContinuously = true
        view.automaticallyUpdatesLighting = true
        // Share the in-progress capture session (this view does not run/pause it).
        if let session = captureSession.arSession {
            view.session = session
        }
        view.delegate = context.coordinator
        context.coordinator.attach(to: view)
        return view
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}

    static func dismantleUIView(_ uiView: ARSCNView, coordinator: Coordinator) {
        // ARKitCaptureSession owns the session, so do not pause it. Only detach rendering.
        uiView.delegate = nil
    }

    final class Coordinator: NSObject, ARSCNViewDelegate {
        private let captureSession: ARKitCaptureSession
        private let pointNode = SCNNode()
        /// Depth-only occluder points (`makeOccluderGeometry`). Drawn before pointNode.
        private let occluderNode = SCNNode()
        private var lastRevision = -1
        private var lastUpdate: TimeInterval = 0

        /// Surfel display (white + splats) for MVS.
        private let usesSurfels: Bool
        private let washNode = CaptureSurfelRenderer.makeWashNode(opacity: Coordinator.washOpacity)
        /// Created on the first update on the render thread (needs SceneKit's Metal device).
        private var surfelRenderer: CaptureSurfelRenderer?
        private var lastBuildEye = SIMD3<Float>(repeating: .infinity)

        /// Opacity of the white laid over the camera feed.
        static let washOpacity: Float = 0.6
        /// Splats are sorted back-to-front from the camera position at packing time, so repack once
        /// the camera moves more than this (m).
        static let resortDistance: Float = 0.1
        /// Minimum interval between splat repacks (seconds).
        static let surfelUpdateInterval: TimeInterval = 0.15

        /// Radius of the occluder points (relative to the display voxel edge). Points are drawn as
        /// circles and shrink on screen by the amount they are pushed back, so use 1 voxel to leave
        /// no holes between diagonal neighbors (0.71 voxel).
        static let occluderRadiusVoxels: Float = 1
        /// How far to push the occluder points back along the view ray (relative to the display
        /// voxel edge). Points farther than this behind the nearest point are hidden. If too small,
        /// points of the same surface seen at a grazing angle are hidden by their nearer neighbors.
        /// (2026-10-09 Mac offscreen render, radius 1 voxel: points behind a nearer surface at
        /// 0.3-2 m with 2/6 cm voxels all disappear; points of the same surface remain at 93-97%
        /// with a 15° angle between surface and view ray, 36-43% at 10°. With 2 voxels, 50% or less
        /// at 15°)
        static let occlusionDepthVoxels: Float = 3
        /// Minimum observation count for a voxel to be used as an occluder. Voxels observed once
        /// (isolated points floating off the surface) would hide a wide area behind them from up
        /// front, so they are not used for occlusion (they are still displayed).
        static let occluderMinObservations = 2

        init(captureSession: ARKitCaptureSession) {
            self.captureSession = captureSession
            self.usesSurfels = captureSession.liveReconstructionMode == .mvs
        }

        @MainActor
        func attach(to view: ARSCNView) {
            if usesSurfels { return }   // White is attached to the camera (updateSurfels); splats are drawn in didRenderScene
            occluderNode.renderingOrder = -1
            view.scene.rootNode.addChildNode(occluderNode)
            view.scene.rootNode.addChildNode(pointNode)
        }

        // Called every frame, so throttle point cloud updates to ~5 Hz.
        func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
            if usesSurfels {
                updateSurfels(renderer, time: time)
                return
            }
            guard time - lastUpdate > 0.2 else { return }
            lastUpdate = time
            let revision = captureSession.liveCoverageRevision
            guard revision != lastRevision else { return }
            lastRevision = revision
            let splats = captureSession.liveCoverageSplats
            pointNode.geometry = Self.makePointGeometry(splats)
            occluderNode.geometry = Self.makeOccluderGeometry(splats)
        }

        /// Attach the white to the camera; when new surfels arrive or the camera moves, dispatch a
        /// splat repack on a separate queue.
        private func updateSurfels(_ renderer: SCNSceneRenderer, time: TimeInterval) {
            if washNode.parent == nil, let pov = renderer.pointOfView { pov.addChildNode(washNode) }
            if surfelRenderer == nil, let device = renderer.device {
                surfelRenderer = CaptureSurfelRenderer(device: device)
            }
            guard let surfelRenderer, !surfelRenderer.isUpdating, time - lastUpdate > Self.surfelUpdateInterval,
                  let eye = renderer.pointOfView?.simdWorldPosition else { return }
            let revision = captureSession.liveSurfelRevision
            guard revision != lastRevision || simd_distance(eye, lastBuildEye) > Self.resortDistance else { return }
            lastUpdate = time
            lastRevision = revision
            lastBuildEye = eye
            surfelRenderer.update(captureSession.liveSurfels, eye: eye)
        }

        func renderer(_ renderer: SCNSceneRenderer, didRenderScene scene: SCNScene, atTime time: TimeInterval) {
            guard usesSurfels else { return }
            surfelRenderer?.draw(in: renderer)
        }

        /// Build a point-primitive SCNGeometry from the coverage splats.
        static func makePointGeometry(_ splats: [CaptureCoverageSplat]) -> SCNGeometry? {
            guard !splats.isEmpty else { return nil }
            let vertices = splats.map {
                SCNVector3($0.position.x, $0.position.y, $0.position.z)
            }
            let source = SCNGeometrySource(vertices: vertices)

            var colors = [Float](repeating: 0, count: splats.count * 4)
            for (i, splat) in splats.enumerated() {
                // Yellow → cyan by observation count.
                let t = min(1, log1p(Float(splat.observationCount)) / log1p(Float(8)))
                let rgb = SIMD3<Float>(1.0 - 0.78 * t, 0.74 + 0.26 * t, 0.12 + 0.78 * t)
                colors[i * 4 + 0] = rgb.x
                colors[i * 4 + 1] = rgb.y
                colors[i * 4 + 2] = rgb.z
                colors[i * 4 + 3] = splat.opacity
            }
            let colorData = colors.withUnsafeBufferPointer { Data(buffer: $0) }
            let colorSource = SCNGeometrySource(
                data: colorData,
                semantic: .color,
                vectorCount: splats.count,
                usesFloatComponents: true,
                componentsPerVector: 4,
                bytesPerComponent: MemoryLayout<Float>.size,
                dataOffset: 0,
                dataStride: MemoryLayout<Float>.size * 4
            )

            // pointSize is in world units, so 18 always sticks to the cap and points render at 24
            // px regardless of distance.
            let element = pointElement(count: splats.count)
            element.pointSize = 18
            element.minimumPointScreenSpaceRadius = 4
            element.maximumPointScreenSpaceRadius = 24

            let geometry = SCNGeometry(sources: [source, colorSource], elements: [element])
            let material = SCNMaterial()
            material.diffuse.contents = UIColor.white
            material.lightingModel = .constant
            material.isDoubleSided = true
            material.blendMode = .alpha
            material.readsFromDepthBuffer = true
            material.writesToDepthBuffer = false
            geometry.firstMaterial = material
            return geometry
        }

        /// Geometry that writes only the points' depth. Display points have a constant on-screen
        /// size, so gaps open between them up close, but these cover the surface without gaps at
        /// voxel size (world units). They are written pushed back along the view ray by
        /// `occlusionDepthVoxels`, so they hide only points behind that, not points of the same
        /// surface.
        static func makeOccluderGeometry(_ splats: [CaptureCoverageSplat]) -> SCNGeometry? {
            let occluders = splats.filter { $0.observationCount >= occluderMinObservations }
            guard let voxel = occluders.first?.voxelSizeMeters else { return nil }
            let source = SCNGeometrySource(vertices: occluders.map {
                SCNVector3($0.position.x, $0.position.y, $0.position.z)
            })
            let element = pointElement(count: occluders.count)
            element.pointSize = CGFloat(voxel * occluderRadiusVoxels)
            element.minimumPointScreenSpaceRadius = 1
            element.maximumPointScreenSpaceRadius = 511

            let geometry = SCNGeometry(sources: [source], elements: [element])
            let material = SCNMaterial()
            material.lightingModel = .constant
            material.colorBufferWriteMask = []
            material.readsFromDepthBuffer = true
            material.writesToDepthBuffer = true
            material.shaderModifiers = [.geometry: occluderShaderModifier]
            material.setValue(NSNumber(value: voxel * occlusionDepthVoxels), forKey: "depthOffset")
            geometry.firstMaterial = material
            return geometry
        }

        /// Push vertices away from the viewpoint by `depthOffset` (m).
        static let occluderShaderModifier = """
        #pragma arguments
        float depthOffset;
        #pragma body
        float3 eye = (scn_node.inverseModelViewTransform * float4(0.0, 0.0, 0.0, 1.0)).xyz;
        _geometry.position.xyz += normalize(_geometry.position.xyz - eye) * depthOffset;
        """

        static func pointElement(count: Int) -> SCNGeometryElement {
            let indices = Array(Int32(0)..<Int32(count))
            let indexData = indices.withUnsafeBufferPointer { Data(buffer: $0) }
            return SCNGeometryElement(
                data: indexData,
                primitiveType: .point,
                primitiveCount: count,
                bytesPerIndex: MemoryLayout<Int32>.size
            )
        }
    }
}

#endif // os(iOS)
