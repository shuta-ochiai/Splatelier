// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX

/// A training view (camera and downscaled target image).
public struct TrainView {
    public let camera: Camera
    /// Kept resident as [H, W, 3] uint8 (0..255) (lossless since it comes from 8-bit; 1/4 of
    /// float32). Use `imageFloat` where float is needed.
    public let image: MLXArray
    /// Float [0,1] version. Converted on each use; do not cache it (converting every view to float
    /// brings resident memory back to 4x).
    public var imageFloat: MLXArray {
        image.dtype == .uint8 ? image.asType(.float32) / 255.0 : image
    }
    public let name: String
    /// Relative sampling weight [0, 1]. Scaling the loss is canceled by Adam (`m/√v`), so
    /// low-confidence frames are down-weighted through sampling frequency instead.
    public let weight: Float
    /// This view's MVS depth map, for the geometric regularization (nil without one).
    public var mvsDepth: DepthTarget?

    public init(camera: Camera, image: MLXArray, name: String, weight: Float = 1.0, mvsDepth: DepthTarget? = nil) {
        self.camera = camera
        self.image = image
        self.name = name
        self.weight = weight
        self.mvsDepth = mvsDepth
    }
}

/// Helper that builds training views from a COLMAP scene (image loading and intrinsics scaling live
/// here to keep the loader pure).
public enum ColmapTraining {

    /// Loads each camera's image with the long side at most `maxDimension` and scales intrinsics to
    /// match the resolution. Views whose image is missing or unreadable are skipped.
    public static func makeViews(
        scene: ColmapScene,
        imagesDirectory: URL,
        maxDimension: Int,
        tileSize: Int = 16,
        background: MLXArray
    ) -> [TrainView] {
        var views: [TrainView] = []
        for cam in scene.cameras {
            let url = imagesDirectory.appendingPathComponent(cam.imageName)
            guard FileManager.default.fileExists(atPath: url.path) else { continue }
            guard let image = try? ImageLoader.loadRGB(url: url, maxDimension: maxDimension) else {
                continue
            }
            let h = image.shape[0]
            let w = image.shape[1]
            // Rounding can shift the x/y ratio, so scale per axis.
            let sx = Float(w) / Float(cam.width)
            let sy = Float(h) / Float(cam.height)
            let intrinsics = MLXArray(
                [cam.fx * sx, cam.fy * sy, cam.cx * sx, cam.cy * sy], [4]
            )
            let camera = Camera(
                viewmat: MLXArray(cam.viewmat, [16]),
                intrinsics: intrinsics,
                width: w, height: h, tileSize: tileSize,
                background: background
            )
            // TrainView.image is contractually uint8 (loadRGB returns float [0,1]).
            let imageU8 = (MLX.clip(image, min: 0.0, max: 1.0) * 255.0 + 0.5).asType(.uint8)
            eval(camera.viewmat, camera.intrinsics, imageU8)
            views.append(TrainView(camera: camera, image: imageU8, name: cam.imageName))
        }
        return views
    }
}

/// Weighted sampler over training views (cumulative weights + binary search). Uniform if all
/// weights are equal.
public struct ViewSampler: Sendable {
    let count: Int
    /// Cumulative weights. nil means uniform.
    let cdf: [Float]?

    init(views: [TrainView]) {
        count = views.count
        let weights = views.map { Swift.max(0, $0.weight) }
        let total = weights.reduce(0, +)
        guard total > 0, let first = weights.first,
              weights.contains(where: { $0 != first }) else { cdf = nil; return }
        var cdf = [Float](repeating: 0, count: weights.count)
        var acc: Float = 0
        for i in weights.indices { acc += weights[i]; cdf[i] = acc }
        self.cdf = cdf
    }

    func sample<G: RandomNumberGenerator>(using rng: inout G) -> Int {
        guard let cdf, let total = cdf.last, total > 0 else {
            return Int.random(in: 0..<count, using: &rng)
        }
        let r = Float.random(in: 0..<total, using: &rng)
        var lo = 0, hi = cdf.count - 1
        while lo < hi {
            let mid = (lo + hi) / 2
            if cdf[mid] <= r { lo = mid + 1 } else { hi = mid }
        }
        return lo
    }
}
