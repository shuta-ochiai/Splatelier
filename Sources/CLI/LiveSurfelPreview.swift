// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// REPLAY_MODES=surfel of `live-mvs-replay`: checks on the Mac how the in-capture display looks as
// "blurry splats that sharpen as you get closer" (CaptureSurfelMap). Feeds valset keyframes in
// capture order to a StreamingMVS configured like the live in-capture point cloud (lookahead 0 /
// fusionDelay 0, ARKit poses, feature points in arrival order) and, at a few points along the way,
// renders the current surfels as Gaussians with the training renderer.
//
// Output (outDir):
//   surfel_sheet.png  one row = one moment during capture. From left:
//                       camera image | camera image with translucent white over it and splats drawn
//                       on top in their own colors (models the in-capture AR display: color fills
//                       in over the white guide where you have captured) | splats only, from a
//                       fixed viewpoint mid-capture
//   surfel.ply         surfels at end of capture (3DGS format, viewable in the viewer)
//
// env:
//   (defaults are the app's, CaptureSurfelMap())
//   SURFEL_MIN=0.005     edge length of the finest cell (m)
//   SURFEL_LEVELS=7      number of levels (cell = SURFEL_MIN × 2^L)
//   SURFEL_CPP=1         edge of the cell a point goes into = pixel footprint on the surface × this
//   SURFEL_HIDE=1        don't emit a cell with at least this many refined children (of 8)
//   SURFEL_MINPTS=2      don't emit cells with fewer points than this
//   SURFEL_SIGMA=<s>     splat std dev = cell edge × this (in-plane, default
//                        CaptureSurfelMap.splatSigmaPerCell)
//   SURFEL_FLAT=0.2      std dev along the normal = in-plane × this
//   SURFEL_SHOTS=6       number of intermediate moments (SURFEL_SHOT_FRAMES=<k,k,...> picks the
//                        keyframes instead)
//   SURFEL_RENDER=480    long side of the render (px)
//   SURFEL_WASH=0.6      opacity of the white overlaid on the camera image
//   SURFEL_VIDEO=<n>     every n keyframes, write the same 3 images as a sheet row to
//                        outDir/frames/NNNN.png (for making a video with ffmpeg, e.g.
//                        `ffmpeg -framerate 8 -i frames/%04d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2"
//                        -pix_fmt yuv420p out.mp4`)
//   SURFEL_SCENEKIT=1    (macOS) at each intermediate moment, render offscreen the same way as the
//                        app (SceneKit + CaptureSurfelRenderer) and write scenekit_sheet.png next
//                        to the training renderer's overlay
//   SURFEL_CARVE=1       remove coarse splats floating in front of surfaces captured up close (0
//                        keeps them)
//   SURFEL_MAXLEVEL=<L>  (for investigation) don't draw surfels coarser than this level
//   SURFEL_LOG=1         (for investigation) print per-keyframe point count, pixel footprint and
//                        level used
//   SURFEL_FINAL=1       compare the live MVS with the training MVS (LiveVsFinalMVS.swift): rows
//                        become camera image | live display | live MVS points | training MVS
//                        points (BA camera, undistorted image), both added in capture order.
//                        SURFEL_POINT_SIGMA=<m> draws every point that size instead of the size of
//                        the pixel that saw it

import Foundation
import MLX
import Trainer
import Capture
import simd
#if os(macOS) && canImport(SceneKit)
import AppKit
import SceneKit
#endif

func runSurfelPreviewReplay(
    frames: [CapturedFrame], images: [MLXArray], intrinsics intr: [Intrinsics], viewmats: [[Float]],
    parameters: CostVolumeMVS.Parameters, sparse: [SIMD3<Float>], causal: Bool,
    newFeaturePoints: (Int) -> [SIMD3<Float>], finalMVS: FinalMVSPoints? = nil, outDir: URL
) throws {
    let env = ProcessInfo.processInfo.environment
    func f(_ k: String, _ d: Float) -> Float { env[k].flatMap { Float($0) } ?? d }
    func i(_ k: String, _ d: Int) -> Int { env[k].flatMap { Int($0) } ?? d }
    let app = CaptureSurfelMap()
    var map = CaptureSurfelMap(
        finestCellMeters: f("SURFEL_MIN", app.finestCellMeters), levelCount: i("SURFEL_LEVELS", app.levelCount),
        cellsPerPixel: f("SURFEL_CPP", app.cellsPerPixel),
        hideWhenRefinedChildren: i("SURFEL_HIDE", app.hideWhenRefinedChildren),
        minPoints: i("SURFEL_MINPTS", app.minPoints))
    let sigma = f("SURFEL_SIGMA", CaptureSurfelMap.splatSigmaPerCell), flat = f("SURFEL_FLAT", 0.2), wash = f("SURFEL_WASH", 0.6)
    let shots = max(1, i("SURFEL_SHOTS", 6))
    let renderLong = i("SURFEL_RENDER", 480)
    let maxLevel = i("SURFEL_MAXLEVEL", Int.max)
    let perFrameLog = env["SURFEL_LOG"] != nil
    let carve = env["SURFEL_CARVE"] != "0"
    let videoEvery = i("SURFEL_VIDEO", 0)
    let framesDir = outDir.appendingPathComponent("frames")
    if videoEvery > 0 { try FileManager.default.createDirectory(at: framesDir, withIntermediateDirectories: true) }

    let livePoints = PointCloudAccumulator(), finalPoints = PointCloudAccumulator()
    var finalFrame: CapturedFrame?

    let engine = StreamingMVS(
        parameters: .init(mvs: parameters, lookahead: 0, fusionDelay: 0), retainsCloud: false)
    engine.emitsSurfels = true
    if !causal { engine.addSparsePoints(sparse) }

    let renderer = try GaussianRenderer()
    let longSide = max(frames[0].width, frames[0].height)
    let r = max(1, (longSide + renderLong - 1) / renderLong)
    func camera(_ k: Int) -> Camera {
        let it = frames[k].intrinsics
        return Camera(
            viewmat: MLXArray(frames[k].viewmat, [16]),
            intrinsics: MLXArray([it.fx / Float(r), it.fy / Float(r), it.cx / Float(r), it.cy / Float(r)], [4]),
            width: frames[k].width / r, height: frames[k].height / r,
            background: MLXArray([Float(0), 0, 0], [3]))
    }
    func cameraFor(_ frame: CapturedFrame) -> Camera {
        let it = frame.intrinsics
        return Camera(
            viewmat: MLXArray(frame.viewmat, [16]),
            intrinsics: MLXArray([it.fx / Float(r), it.fy / Float(r), it.cx / Float(r), it.cy / Float(r)], [4]),
            width: frame.width / r, height: frame.height / r,
            background: MLXArray([Float(0), 0, 0], [3]))
    }
    func render(_ model: GaussianModel?, _ k: Int) -> (MLXArray, MLXArray) {
        let cam = camera(k)
        guard let model else {
            return (MLXArray.zeros([cam.height, cam.width, 3]), MLXArray.zeros([cam.height, cam.width]))
        }
        let (rgb, alpha, _) = renderer.forward(model: model, camera: cam)
        return (MLX.clip(rgb, min: 0, max: 1), alpha)
    }
    let fixedView = frames.count / 2
    // SURFEL_SHOT_FRAMES=<k,k,...>: draw these keyframes (0-based) instead of evenly spaced ones.
    let shotFrames = env["SURFEL_SHOT_FRAMES"].map { Set($0.split(separator: ",").compactMap { Int($0) }) }
        ?? Set((1...shots).map { frames.count * $0 / shots - 1 })

    var rows: [MLXArray] = []
    let sceneKitCompare = env["SURFEL_SCENEKIT"] != nil
    var sceneKitRows: [MLXArray] = []
    var mvsSeconds = 0.0, surfelSeconds = 0.0, insertSeconds = 0.0, snapshotSeconds = 0.0, carveSeconds = 0.0
    var points = 0, fallback = 0, carved = 0, snapshots = 0
    for k in frames.indices {
        if causal { engine.addSparsePoints(newFeaturePoints(k)) }
        let t0 = Date()
        let step = engine.addFrame(image: images[k], viewmat: viewmats[k], intrinsics: intr[k])
        mvsSeconds += Date().timeIntervalSince(t0) - step.surfelSeconds
        surfelSeconds += step.surfelSeconds
        fallback += step.fallbackNormals
        precondition(step.normals.count == step.positions.count && step.pixelSizes.count * 3 == step.positions.count)
        let tc = Date()
        if carve {
            for d in step.fusedDepths {
                carved += map.carve(depth: d.depth, width: d.width, height: d.height,
                                    intrinsics: intr[d.frame], viewmat: viewmats[d.frame])
            }
        }
        carveSeconds += Date().timeIntervalSince(tc)
        let t1 = Date()
        var offset = 0
        for (fr, n) in zip(step.fusedFrames, step.fusedPointCounts) where n > 0 {
            let center = FrameSubsampler.cameraCenter(viewmat: viewmats[fr])
            if perFrameLog {
                let ps = step.pixelSizes[offset..<(offset + n)].sorted()
                print(String(format: "[surfel frame] %d: %d points, pixel size p10/p50/p90 = %.1f/%.1f/%.1f mm",
                             fr, n, 1000 * ps[n / 10], 1000 * ps[n / 2], 1000 * ps[n * 9 / 10]))
            }
            for j in offset..<(offset + n) {
                map.insert(
                    position: SIMD3(step.positions[j * 3], step.positions[j * 3 + 1], step.positions[j * 3 + 2]),
                    normal: SIMD3(step.normals[j * 3], step.normals[j * 3 + 1], step.normals[j * 3 + 2]),
                    color: SIMD3(step.colors[j * 3], step.colors[j * 3 + 1], step.colors[j * 3 + 2]),
                    pixelSize: step.pixelSizes[j], cameraCenter: center, frame: fr)
            }
            offset += n
        }
        points += offset
        insertSeconds += Date().timeIntervalSince(t1)
        if let finalMVS {
            livePoints.append(PointBatch(
                positions: step.positions, colors: step.colors, pixelSizes: step.pixelSizes))
            if let batch = finalMVS.pointsByFrame[k] { finalPoints.append(batch) }
            finalFrame = finalMVS.framesByIndex[k] ?? finalFrame
        }

        let isShot = shotFrames.contains(k)
        let isVideo = videoEvery > 0 && (k % videoEvery == 0 || k == frames.count - 1)
        guard isShot || isVideo else { continue }
        let t2 = Date()
        let surfels = map.snapshot().filter { $0.level <= maxLevel }
        snapshotSeconds += Date().timeIntervalSince(t2)
        snapshots += 1
        let model = surfelModel(surfels, sigma: sigma, flat: flat)
        let cam = StreamingMVS.boxDownscale(frames[k].image, factor: r)
        let (now, nowAlpha) = render(model, k)
        let (fixed, _) = render(model, fixedView)
        let h = now.shape[0], w = now.shape[1]
        let camCrop = cam[0..<h, 0..<w]
        let washed = (1 - nowAlpha.expandedDimensions(axis: 2)) * ((1 - wash) * camCrop + wash)
        let overlay = now + washed
        var row = MLX.concatenated(
            [camCrop, overlay, fixed[0..<h, 0..<w]], axis: 1)
        if finalMVS != nil {
            func pointPanel(_ model: GaussianModel?, _ camera: Camera, _ image: MLXArray) -> MLXArray {
                let bg = (1 - wash) * StreamingMVS.boxDownscale(image, factor: r)[0..<h, 0..<w] + wash
                guard let model else { return bg }
                let (rgb, alpha, _) = renderer.forward(model: model, camera: camera)
                return MLX.clip(rgb, min: 0, max: 1)[0..<h, 0..<w]
                    + (1 - alpha[0..<h, 0..<w].expandedDimensions(axis: 2)) * bg
            }
            let pointSigma = env["SURFEL_POINT_SIGMA"].flatMap { Float($0) }
            let live = pointPanel(livePoints.model(fixedSigma: pointSigma), camera(k), frames[k].image)
            let final = finalFrame.map {
                pointPanel(finalPoints.model(fixedSigma: pointSigma), cameraFor($0), $0.image)
            }
                ?? MLXArray.zeros([h, w, 3])
            row = MLX.concatenated([camCrop, overlay, live, final], axis: 1)
        }
        if isVideo {
            try ImagePNG.write(row, to: framesDir.appendingPathComponent(String(format: "%04d.png", k / videoEvery)))
        }
        guard isShot else { continue }
        rows.append(row)
        #if os(macOS)
        if sceneKitCompare {
            let sk = try renderSurfelsWithSceneKit(
                surfels, frame: frames[k], downscale: r, background: camCrop, wash: wash,
                scratch: outDir.appendingPathComponent("scenekit_tmp.png"))
            sceneKitRows.append(MLX.concatenated([overlay, sk], axis: 1))
        }
        #endif
        let levelHist = (0..<map.levelCount).map { l in surfels.filter { $0.level == l }.count }
        print(String(format: "[surfel] frame %d/%d: %d surfels shown (by level %@), cells %d",
                     k + 1, frames.count, surfels.count,
                     levelHist.map(String.init).joined(separator: "/") as NSString,
                     map.cellCounts.reduce(0, +)))
        if k == frames.count - 1, let model {
            try PLYWriter.writeGaussianPLY(model: model, to: outDir.appendingPathComponent("surfel.ply"))
        }
    }
    try ImagePNG.write(MLX.concatenated(rows, axis: 0), to: outDir.appendingPathComponent("surfel_sheet.png"))
    if !sceneKitRows.isEmpty {
        try ImagePNG.write(MLX.concatenated(sceneKitRows, axis: 0),
                           to: outDir.appendingPathComponent("scenekit_sheet.png"))
    }
    let kf = Double(frames.count)
    print(String(format: "[surfel] %d points, normal fallback %.1f%%, carved %d cells; per keyframe: MVS %.1f ms, "
        + "normals %.2f ms, carve %.2f ms, map insert %.2f ms; snapshot %.1f ms avg",
        points, 100 * Double(fallback) / Double(max(1, points)), carved, 1000 * mvsSeconds / kf,
        1000 * surfelSeconds / kf, 1000 * carveSeconds / kf, 1000 * insertSeconds / kf,
        1000 * snapshotSeconds / Double(max(1, snapshots))))
}

/// Turns a surfel into a Gaussian flattened along its normal (in-plane std dev = cell edge ×
/// `sigma`, opacity = `Surfel.splatOpacity`, color = the cell color).
func surfelModel(_ surfels: [CaptureSurfelMap.Surfel], sigma: Float, flat: Float) -> GaussianModel? {
    guard !surfels.isEmpty else { return nil }
    let n = surfels.count
    var means = [Float](), quats = [Float](), scales = [Float](), logits = [Float](), sh0 = [Float]()
    means.reserveCapacity(n * 3); quats.reserveCapacity(n * 4); scales.reserveCapacity(n * 3)
    logits.reserveCapacity(n); sh0.reserveCapacity(n * 3)
    for s in surfels {
        means += [s.position.x, s.position.y, s.position.z]
        // Shortest rotation taking the z axis to the normal (w, x, y, z). If exactly opposite, 180°
        // about x.
        let nz = s.normal
        let q = nz.z < -0.9999 ? SIMD4<Float>(0, 1, 0, 0)
            : simd_normalize(SIMD4<Float>(1 + nz.z, -nz.y, nz.x, 0))
        quats += [q.x, q.y, q.z, q.w]
        let sd = s.cellMeters * sigma
        scales += [log(sd), log(sd), log(sd * flat)]
        let a = s.splatOpacity
        logits.append(log(a / (1 - a)))
        let c = s.color
        sh0 += [(c.x - 0.5) / GaussianModel.shC0, (c.y - 0.5) / GaussianModel.shC0,
                (c.z - 0.5) / GaussianModel.shC0]
    }
    return GaussianModel(
        means: MLXArray(means, [n, 3]), quats: MLXArray(quats, [n, 4]),
        logScales: MLXArray(scales, [n, 3]), logitOpacities: MLXArray(logits, [n]),
        sh0: MLXArray(sh0, [n, 3]), shN: MLXArray.zeros([n, GaussianModel.shRestBands, 3]))
}

#if os(macOS) && canImport(SceneKit)
/// Renders the same way as the app's capture screen (SceneKit camera image + translucent white +
/// `CaptureSurfelRenderer`) offscreen with SCNRenderer from the keyframe camera; returns
/// `[h, w, 3]` (0..1).
func renderSurfelsWithSceneKit(
    _ surfels: [CaptureSurfelMap.Surfel], frame: CapturedFrame, downscale r: Int, background: MLXArray,
    wash: Float, scratch: URL
) throws -> MLXArray {
    let h = background.shape[0], w = background.shape[1]
    let it = frame.intrinsics
    let fx = it.fx / Float(r), fy = it.fy / Float(r), cx = it.cx / Float(r), cy = it.cy / Float(r)
    let near: Float = 0.01, far: Float = 200
    // Turn OpenCV intrinsics (y down) into the projection of a SceneKit camera looking down -z with
    // y up.
    let proj = simd_float4x4(rows: [
        SIMD4(2 * fx / Float(w), 0, 1 - 2 * cx / Float(w), 0),
        SIMD4(0, 2 * fy / Float(h), 2 * cy / Float(h) - 1, 0),
        SIMD4(0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)),
        SIMD4(0, 0, -1, 0),
    ])
    let camera = SCNCamera()
    camera.projectionTransform = SCNMatrix4(proj)
    let cameraNode = SCNNode()
    cameraNode.camera = camera
    // ARKit's camera-to-world (y up, looking down -z) has the same orientation as a SceneKit
    // camera.
    cameraNode.simdTransform = PoseConversion.cameraToWorld(fromViewmat: frame.viewmat)
    cameraNode.addChildNode(CaptureSurfelRenderer.makeWashNode(opacity: wash))

    let scene = SCNScene()
    try ImagePNG.write(background, to: scratch)
    scene.background.contents = NSImage(contentsOf: scratch)
    scene.rootNode.addChildNode(cameraNode)
    let device = MTLCreateSystemDefaultDevice()!
    guard let surfelRenderer = CaptureSurfelRenderer(device: device) else { return MLXArray.zeros([h, w, 3]) }
    let t0 = Date()
    surfelRenderer.updateSynchronously(surfels, eye: cameraNode.simdWorldPosition)
    print(String(format: "[surfel scenekit] %d surfels: sort + pack %.1f ms", surfels.count,
                 1000 * Date().timeIntervalSince(t0)))
    let renderer = SCNRenderer(device: device, options: nil)
    renderer.scene = scene
    renderer.pointOfView = cameraNode
    let drawDelegate = SurfelDrawDelegate(surfelRenderer)
    renderer.delegate = drawDelegate
    defer { withExtendedLifetime(drawDelegate) {} }
    // Shader compile and pipeline creation happen on another thread and splats aren't drawn until
    // they finish, so re-render until they are drawn.
    var image = renderer.snapshot(atTime: 0, with: CGSize(width: w, height: h), antialiasingMode: .none)
    for _ in 0..<100 where surfelRenderer.drawnFrames == 0 {
        Thread.sleep(forTimeInterval: 0.02)
        image = renderer.snapshot(atTime: 0, with: CGSize(width: w, height: h), antialiasingMode: .none)
    }
    guard let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        return MLXArray.zeros([h, w, 3])
    }
    // Re-render the result into sRGB RGBA8 and read it (so it doesn't depend on SCNRenderer's
    // output color space/scale).
    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    let ctx = CGContext(data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                        space: CGColorSpace(name: CGColorSpace.sRGB)!,
                        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    let rgb = (0..<(w * h)).flatMap { i in (0..<3).map { Float(rgba[i * 4 + $0]) / 255 } }
    return MLXArray(rgb, [h, w, 3])
}

/// As in the app's ARCaptureView, draw the splats once the scene has finished drawing.
private final class SurfelDrawDelegate: NSObject, SCNSceneRendererDelegate {
    let surfels: CaptureSurfelRenderer
    init(_ surfels: CaptureSurfelRenderer) { self.surfels = surfels }
    func renderer(_ renderer: SCNSceneRenderer, didRenderScene scene: SCNScene, atTime time: TimeInterval) {
        surfels.draw(in: renderer)
    }
}
#endif
