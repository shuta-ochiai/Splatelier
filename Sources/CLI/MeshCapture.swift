// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// The mesh of `train-capture` with MESH_OBJ, as the app's Generate mesh (OBJ, vertex colors): after
// training with the geometric regularization, the trained Gaussians' surface depth → TSDF fusion →
// Marching Cubes; otherwise (GEO_REG=0, or `gs-mesh`) training's MVS depth.
//
// env:
//   MESH_VOXEL=<m>            voxel edge length (default 0.005)
//   MESH_TRUNC=<voxels>       truncation distance (in voxels, default 4)
//   MESH_MIN_WEIGHT=<n>       observations needed to emit a surface (default 3)
//   MESH_MIN_COMPONENT=<n>    drop floating pieces with fewer triangles than this (default 200, 0
//                             disables)
//   MESH_MAX_VOXELS=<n>       cap on allocated voxels; above it the voxel size grows automatically
//                             (default 12M as in the app; the MESH_HYBRID / MESH_GOF / MESH_GS_TSDF
//                             experiments below default to 32M)

import Foundation
import MLX
import Trainer
import Capture

/// Builds the mesh with the MESH_* env knobs and writes it as OBJ: from `session`'s Gaussians when
/// it trained with the geometric regularization (as the app), else from `geometry`'s MVS depth maps
/// without those of the frames in `excluding` (view matrices; the views held out for evaluation).
/// false if no surface remains.
func writeCaptureMesh(
    geometry: CaptureGeometry, session: TrainingDriver.ResumableTrainingSession? = nil, sourceName: String,
    to outputOBJ: URL, excluding: Set<[Float]> = [], log: (String) -> Void
) throws -> Bool {
    let env = ProcessInfo.processInfo.environment
    var parameters = CaptureTraining.AppPreset.meshParameters
    if let v = env["MESH_VOXEL"].flatMap({ Float($0) }) { parameters.tsdf.voxelSize = v }
    if let v = env["MESH_TRUNC"].flatMap({ Float($0) }) { parameters.tsdf.truncationVoxels = v }
    if let v = env["MESH_MAX_VOXELS"].flatMap({ Int($0) }) { parameters.tsdf.maxVoxels = v }
    if let v = env["MESH_MIN_WEIGHT"].flatMap({ Float($0) }) { parameters.minWeight = v }
    if let v = env["MESH_MIN_COMPONENT"].flatMap({ Int($0) }) { parameters.minComponentTriangles = v }

    let t0 = Date()
    let source: String
    let built: TriangleMesh?
    if let session, session.geometryRegularization != nil {
        source = "Gaussian surface depth"
        built = try CaptureMeshing.reconstruct(
            model: session.trainer.model, cameras: session.views.map(\.camera), shDegree: session.maxShDegree,
            parameters: parameters, log: log)
    } else {
        source = "MVS depth"
        let views = try geometry.depthViews().filter { !excluding.contains($0.viewmat) }
        built = CaptureMeshing.mesh(fromDepthViews: views, parameters: parameters, log: log)
    }
    guard let mesh = built else { return false }
    try MeshOBJWriter.write(
        mesh, to: outputOBJ,
        comment: "source: \(sourceName) (\(source) → TSDF → marching cubes)\n"
            + "voxel \(parameters.tsdf.voxelSize) m, trunc \(parameters.tsdf.truncationVoxels) voxels, "
            + "minWeight \(parameters.minWeight), minComponent \(parameters.minComponentTriangles)")
    log(String(format: "[timing] mesh (TSDF+extract+OBJ write) = %.2fs", Date().timeIntervalSince(t0)))
    print("mesh: wrote \(mesh.vertexCount) vertices, \(mesh.triangleCount) triangles to \(outputOBJ.path)")
    return true
}

// Meshes from the trained Gaussians (train-capture with MESH_HYBRID / MESH_GOF / MESH_GS_TSDF). All
// use the per-view depth at which the Gaussian opacity field along each pixel ray reaches 0.5.
//
// env:
//   MESH_HYBRID=<path.obj>    MVS depth TSDF, with the Gaussian depth filling where MVS left the grid
//                             unobserved (MESH_* knobs of the MVS mesh; MESH_HYBRID_TRUST=<n> MVS
//                             observations for full trust, default 3; MESH_FILL_FRACTION=<f> voxels
//                             the fill may add, as a fraction of MESH_MAX_VOXELS, default 0.25)
//   MESH_GOF=<path.obj>       level set of the opacity field (GOF idea, GaussianOpacityField)
//   MESH_GS_TSDF=<path.obj>   TSDF fusion of the Gaussian depth maps alone
//   GS_CONSISTENCY=0          fill with the raw Gaussian depth instead of its multi-view consistent pixels
//   GS_DEPTH_CLIP=<f>         TSDF grids are placed from depths <= f × the median Gaussian depth
//                             (default 2.5, 0 = no limit)
//   GOF_RES=<s>               depth rendering resolution relative to the training views (default 0.5)
//   GOF_VIEWS=<n>             use at most n training views, evenly spaced (default all)
//   GOF_DEPTH_ITERS=<n>       bisection steps of the per-pixel 0.5 depth (default 12 = 4 passes)
//   GS_DEPTH_PROFILE=<names>  per-Gaussian transmittance: gof (opacity field) or solid (stochastic
//                             solid, G³S, default as in the app). A comma-separated list (gof,solid)
//                             writes each mesh once per profile from the same training, named
//                             <path>_<profile>.obj
//   GOF_VOXEL=<m>             sampling grid voxel (default 0.005)
//   GOF_BAND=<voxels>         allocated band around the depth samples (default 2)
//   GOF_MIN_OBS=<n>           views that must observe a cube (default 3)
//   GOF_MAX_CARVE=<n>         views allowed to see a point in front of their surface (default 0 = GOF)
//   GOF_TOL=<voxels>          depth tolerance before a view carves, in voxels (default 0)
//   GOF_TOL_REL=<f>           relative depth tolerance (default 0)
//   GOF_SEARCH_ITERS=<n>      binary-search steps along the edges (default 8)
//   GOF_MIN_COMPONENT=<n>     drop components with fewer triangles (default 200)
//   GOF_MIN_SUPPORT=<n>       views whose surface depth must back a vertex (default 1)

/// Writes the meshes requested by MESH_HYBRID / MESH_GOF / MESH_GS_TSDF from trained Gaussians seen
/// from `cameras` (and, for MESH_HYBRID, the MVS depth of `geometry`), once per GS_DEPTH_PROFILE.
/// Returns false if a requested mesh had no surface.
func writeGaussianMeshes(
    model: GaussianModel, cameras: [Camera], shDegree: Int, geometry: CaptureGeometry,
    sourceName: String, excluding: Set<[Float]> = [], log: (String) -> Void
) throws -> Bool {
    let env = ProcessInfo.processInfo.environment
    let outputs = ["MESH_HYBRID", "MESH_GOF", "MESH_GS_TSDF"].map { env[$0].map { URL(fileURLWithPath: $0) } }
    guard outputs.contains(where: { $0 != nil }) else { return true }
    let profiles: [String: GaussianOpacityField.TransmittanceProfile] = [
        "gof": .opacityField, "solid": .stochasticSolid,
    ]
    let names = (env["GS_DEPTH_PROFILE"] ?? "solid").split(separator: ",").map(String.init)
    var ok = true
    for name in names {
        guard let profile = profiles[name] else {
            log("GS_DEPTH_PROFILE: unknown profile \(name) (gof, solid)")
            ok = false
            continue
        }
        // One profile keeps the given paths; several get the profile name before the extension.
        let urls = outputs.map { url in
            url.map { names.count == 1 ? $0 : $0.deletingLastPathComponent().appendingPathComponent(
                "\($0.deletingPathExtension().lastPathComponent)_\(name).\($0.pathExtension)") }
        }
        ok = try writeGaussianMeshes(
            model: model, cameras: cameras, shDegree: shDegree, geometry: geometry, sourceName: sourceName,
            profile: profile, profileName: name, hybridOBJ: urls[0], gofOBJ: urls[1], tsdfOBJ: urls[2],
            excluding: excluding, log: log) && ok
    }
    return ok
}

private func writeGaussianMeshes(
    model: GaussianModel, cameras: [Camera], shDegree: Int, geometry: CaptureGeometry, sourceName: String,
    profile: GaussianOpacityField.TransmittanceProfile, profileName: String,
    hybridOBJ: URL?, gofOBJ: URL?, tsdfOBJ: URL?, excluding: Set<[Float]>, log: (String) -> Void
) throws -> Bool {
    let env = ProcessInfo.processInfo.environment
    var depth = GaussianOpacityField.DepthParameters(profile: profile)
    if let v = env["GOF_RES"].flatMap({ Float($0) }) { depth.resolutionScale = v }
    if let v = env["GOF_VIEWS"].flatMap({ Int($0) }) { depth.maxViews = v }
    if let v = env["GOF_DEPTH_ITERS"].flatMap({ Int($0) }) { depth.bisectionIterations = v }
    let t0 = Date()
    let views = try GaussianOpacityField.surfaceDepthViews(
        model: model, cameras: cameras, shDegree: shDegree, parameters: depth, log: log)
    log(String(format: "[timing] gaussian surface depth (%@) = %.2fs", profileName, Date().timeIntervalSince(t0)))
    let clipFactor = env["GS_DEPTH_CLIP"].flatMap { Float($0) } ?? 2.5
    let maxDepth = clipFactor > 0 ? clipFactor * (GaussianOpacityField.medianDepth(views) ?? .infinity) : .infinity

    /// MESH_* knobs of the MVS mesh, with the grid placed from depths <= maxDepth.
    func tsdfMeshParameters() -> CaptureMeshing.Parameters {
        var tsdf = TSDFFusion.Parameters()
        if let v = env["MESH_VOXEL"].flatMap({ Float($0) }) { tsdf.voxelSize = v }
        if let v = env["MESH_TRUNC"].flatMap({ Float($0) }) { tsdf.truncationVoxels = v }
        if let v = env["MESH_MAX_VOXELS"].flatMap({ Int($0) }) { tsdf.maxVoxels = v }
        tsdf.allocationMaxDepth = maxDepth
        var parameters = CaptureMeshing.Parameters(tsdf: tsdf)
        if let v = env["MESH_MIN_WEIGHT"].flatMap({ Float($0) }) { parameters.minWeight = v }
        if let v = env["MESH_MIN_COMPONENT"].flatMap({ Int($0) }) { parameters.minComponentTriangles = v }
        return parameters
    }

    var ok = true
    if let url = hybridOBJ {
        var parameters = tsdfMeshParameters()
        if let v = env["MESH_FILL_FRACTION"].flatMap({ Float($0) }) { parameters.tsdf.fillVoxelFraction = v }
        let trust = env["MESH_HYBRID_TRUST"].flatMap { Float($0) } ?? 3
        let t1 = Date()
        let mvsViews = try geometry.depthViews().filter { !excluding.contains($0.viewmat) }
        // GS_CONSISTENCY=0 fills with the raw Gaussian depth (default: only multi-view consistent pixels).
        let fill = env["GS_CONSISTENCY"] == "0" ? views : GaussianOpacityField.consistentDepthViews(views, log: log)
        if let volume = TSDFFusion.integrateHybrid(
            primary: mvsViews, fill: fill, primaryTrustObservations: trust, parameters: parameters.tsdf, log: log) {
            let raw = MarchingCubes.extract(volume: volume, minWeight: parameters.minWeight)
            let mesh = raw.removingSmallComponents(minTriangles: parameters.minComponentTriangles)
            try MeshOBJWriter.write(
                mesh, to: url,
                comment: "source: \(sourceName) (MVS depth TSDF, holes filled from the Gaussian 0.5-opacity depth)\n"
                    + "voxel \(volume.voxelSize) m, trunc \(parameters.tsdf.truncationVoxels) voxels, MVS trust \(trust)")
            log(String(format: "[timing] hybrid mesh = %.2fs", Date().timeIntervalSince(t1)))
            print("hybrid mesh: wrote \(mesh.vertexCount) vertices, \(mesh.triangleCount) triangles to \(url.path)")
        } else {
            log("hybrid mesh: no surface reconstructed")
            ok = false
        }
    }
    if let url = gofOBJ {
        var p = GaussianOpacityField.MeshParameters()
        if let v = env["GOF_VOXEL"].flatMap({ Float($0) }) { p.voxelSize = v }
        if let v = env["GOF_BAND"].flatMap({ Float($0) }) { p.bandVoxels = v }
        if let v = env["GOF_MIN_OBS"].flatMap({ Float($0) }) { p.minObservations = v }
        if let v = env["GOF_MAX_CARVE"].flatMap({ Int($0) }) { p.maxCarvingViews = v }
        if let v = env["GOF_TOL"].flatMap({ Float($0) }) { p.depthToleranceVoxels = v }
        if let v = env["GOF_TOL_REL"].flatMap({ Float($0) }) { p.depthToleranceRelative = v }
        if let v = env["GOF_SEARCH_ITERS"].flatMap({ Int($0) }) { p.bisectionIterations = v }
        if let v = env["GOF_MIN_COMPONENT"].flatMap({ Int($0) }) { p.minComponentTriangles = v }
        if let v = env["GOF_MIN_SUPPORT"].flatMap({ Float($0) }) { p.minSupportViews = v }
        p.allocationDepthFactor = clipFactor
        let t1 = Date()
        if let mesh = GaussianOpacityField.extractMesh(depthViews: views, parameters: p, log: log) {
            try MeshOBJWriter.write(
                mesh, to: url,
                comment: "source: \(sourceName) (Gaussian opacity field level set)\n"
                    + "requested voxel \(p.voxelSize) m, minObservations \(p.minObservations), "
                    + "maxCarvingViews \(p.maxCarvingViews), tolerance \(p.depthToleranceVoxels) voxels")
            log(String(format: "[timing] GOF mesh = %.2fs", Date().timeIntervalSince(t1)))
            print("gof mesh: wrote \(mesh.vertexCount) vertices, \(mesh.triangleCount) triangles to \(url.path)")
        } else {
            log("gof mesh: no surface reconstructed")
            ok = false
        }
    }
    if let url = tsdfOBJ {
        let parameters = tsdfMeshParameters()
        let t1 = Date()
        if let mesh = CaptureMeshing.mesh(fromDepthViews: views, parameters: parameters, log: log) {
            try MeshOBJWriter.write(
                mesh, to: url,
                comment: "source: \(sourceName) (Gaussian 0.5-opacity depth → TSDF → marching cubes)\n"
                    + "voxel \(parameters.tsdf.voxelSize) m, trunc \(parameters.tsdf.truncationVoxels) voxels")
            log(String(format: "[timing] GS-depth TSDF mesh = %.2fs", Date().timeIntervalSince(t1)))
            print("gs tsdf mesh: wrote \(mesh.vertexCount) vertices, \(mesh.triangleCount) triangles to \(url.path)")
        } else {
            log("gs tsdf mesh: no surface reconstructed")
            ok = false
        }
    }
    return ok
}

/// `gs-mesh <dataset.valset> <trained.ply>`: the meshes of MESH_OBJ / MESH_HYBRID / MESH_GOF /
/// MESH_GS_TSDF from Gaussians trained earlier (train-capture's output), without training. Intake,
/// BA and MVS are redone from the valset (deterministic up to float rounding, so the poses match the
/// training's). The Gaussian depth is rendered from the app's number of training views, evenly
/// spaced, at the frames' resolution (the training's final one; GOF_VIEWS / GOF_RES override). Background splats listed in
/// `<base>.background.json` are left out.
func runGaussianMeshCommand(valsetURL: URL, plyURL: URL, log: (String) -> Void) throws {
    let env = ProcessInfo.processInfo.environment
    var model = try PLYReader.readGaussianPLY(url: plyURL)
    let sidecar = plyURL.deletingPathExtension().appendingPathExtension("background.json")
    if let data = try? Data(contentsOf: sidecar),
       let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
       let count = json["backgroundSplatCount"] as? Int {
        model.environmentCount = Swift.min(count, model.count)
    }
    log("gs-mesh: \(model.count) Gaussians (\(model.environmentCount) background) from \(plyURL.lastPathComponent)")
    let scene = try CaptureDatasetStore.load(from: valsetURL)
    let geometry = CaptureTraining.prepareGeometry(scene: scene, log: log)
    // With EVAL_SPLIT (or EVAL_HOLDOUT), the training views of that run (so the held-out views stay
    // unseen, as in train-capture's meshes); otherwise the app's number of views, evenly spaced.
    let excluded: Set<[Float]>
    let picked: [CapturedFrame]
    if let split = try EvaluationSplit.frames(geometry: geometry, defaultEvery: 0) {
        picked = split.training
        excluded = Set(split.heldOut.map(\.viewmat))
    } else {
        let frames = geometry.scene.frames
        let viewCount = env["GOF_VIEWS"].flatMap { Int($0) }
            ?? CaptureTraining.recommendedTrainingViewCap(frameCount: frames.count)
        picked = viewCount > 0 && viewCount < frames.count
            ? (0..<viewCount).map { frames[$0 * frames.count / viewCount] } : frames
        excluded = []
    }
    if let meshOBJ = env["MESH_OBJ"].map({ URL(fileURLWithPath: $0) }),
       !(try writeCaptureMesh(geometry: geometry, sourceName: valsetURL.lastPathComponent, to: meshOBJ,
                              excluding: excluded, log: log)) {
        log("mesh: no surface reconstructed")
    }
    // GOF_RES is applied by writeGaussianMeshes.
    let background = MLXArray.zeros([3])
    let cameras = picked.map { f in
        Camera(
            viewmat: MLXArray(f.viewmat, [16]),
            intrinsics: MLXArray([f.intrinsics.fx, f.intrinsics.fy, f.intrinsics.cx, f.intrinsics.cy], [4]),
            width: f.width, height: f.height, background: background)
    }
    _ = try writeGaussianMeshes(
        model: model, cameras: cameras, shDegree: GaussianModel.maxShDegree, geometry: geometry,
        sourceName: valsetURL.lastPathComponent, excluding: excluded, log: log)
}
