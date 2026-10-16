// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Builds a vertex-colored mesh via depth maps → TSDF fusion → Marching Cubes, from either the trained
// Gaussians (their surface depth; flat where training used the geometric regularization) or a
// capture's geometry (`CaptureGeometry`: intake → BA → MVS, shared with training: the same depth as
// the MVS point cloud that seeds training).

import Foundation
import MLX
import Trainer

public enum CaptureMeshing {

    public struct Parameters: Sendable {
        public var tsdf: TSDFFusion.Parameters
        /// Observations required to emit a surface (all 8 cube corners). Too few produces spurious
        /// surfaces where observations are thin.
        public var minWeight: Float
        /// Drop connected components (floating fragments) with fewer triangles than this. 0
        /// disables.
        public var minComponentTriangles: Int

        public init(
            tsdf: TSDFFusion.Parameters = TSDFFusion.Parameters(),
            minWeight: Float = 3,
            minComponentTriangles: Int = 200
        ) {
            precondition(minWeight > 0 && minComponentTriangles >= 0)
            self.tsdf = tsdf
            self.minWeight = minWeight
            self.minComponentTriangles = minComponentTriangles
        }
    }

    /// Builds a mesh from trained Gaussians: each camera's surface depth (where the Gaussians'
    /// transmittance reaches 0.5, `GaussianOpacityField.surfaceDepthViews`, by default the
    /// stochastic solid the geometric regularization trains) fused as a TSDF. The grid is placed
    /// from depths within `depthClipFactor` × the median depth (far background would coarsen the
    /// voxels; 0 = no limit). nil if no surface remains.
    public static func reconstruct(
        model: GaussianModel, cameras: [Camera], shDegree: Int,
        parameters: Parameters = Parameters(),
        depth: GaussianOpacityField.DepthParameters = .init(profile: .stochasticSolid),
        depthClipFactor: Float = 2.5,
        log: (String) -> Void = { print($0) }
    ) throws -> TriangleMesh? {
        let views = try GaussianOpacityField.surfaceDepthViews(
            model: model, cameras: cameras, shDegree: shDegree, parameters: depth, log: log)
        var parameters = parameters
        if depthClipFactor > 0, let median = GaussianOpacityField.medianDepth(views) {
            parameters.tsdf.allocationMaxDepth = depthClipFactor * median
        }
        return mesh(fromDepthViews: views, parameters: parameters, log: log)
    }

    /// Builds a mesh from a capture's geometry. nil if MVS produced no depth / no surface remains.
    public static func reconstruct(
        geometry: CaptureGeometry,
        parameters: Parameters = Parameters(),
        log: (String) -> Void = { print($0) }
    ) -> TriangleMesh? {
        let views: [CostVolumeMVS.DepthView]
        do {
            views = try geometry.depthViews()
        } catch {
            log("[mesh] could not read the MVS depth maps: \(error)")
            return nil
        }
        return mesh(fromDepthViews: views, parameters: parameters, log: log)
    }

    /// Builds a mesh from depth maps (TSDF fusion → Marching Cubes → fragment removal).
    public static func mesh(
        fromDepthViews views: [CostVolumeMVS.DepthView],
        parameters: Parameters = Parameters(),
        log: (String) -> Void = { print($0) }
    ) -> TriangleMesh? {
        guard !views.isEmpty else {
            log("[mesh] no depth maps")
            return nil
        }
        let t0 = Date()
        guard let volume = TSDFFusion.integrate(views: views, parameters: parameters.tsdf, log: log) else {
            return nil
        }
        let t1 = Date()
        let raw = MarchingCubes.extract(volume: volume, minWeight: parameters.minWeight)
        let mesh = raw.removingSmallComponents(minTriangles: parameters.minComponentTriangles)
        log(String(format:
            "[mesh] %d vertices, %d triangles (marching cubes %d, dropped %d in components < %d tris); "
            + "TSDF %.2fs, extract %.2fs",
            mesh.vertexCount, mesh.triangleCount, raw.triangleCount,
            raw.triangleCount - mesh.triangleCount, parameters.minComponentTriangles,
            t1.timeIntervalSince(t0), Date().timeIntervalSince(t1)))
        return mesh.triangleCount > 0 ? mesh : nil
    }
}
