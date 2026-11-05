// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX

/// A small picture of what the model looks like right now, for the app to show while it trains.
public struct TrainingPreviewFrame: Sendable {
    public let width: Int
    public let height: Int
    /// Row-major RGBA8, alpha 255.
    public let rgba: [UInt8]
    /// Row-major world→camera of the view it is drawn from (OpenCV), so the app can turn the
    /// picture (in sensor orientation) upright.
    public let viewmat: [Float]

    /// Seconds between two previews (the first one comes with the first step).
    public static let interval: TimeInterval = 3

    /// Which of the training views the `n`th preview (from 0) is drawn from. It walks through up to
    /// six views spread over the capture, and stays on each for two previews, so the picture is
    /// seen to sharpen before the angle changes.
    static func viewIndex(forPreview n: Int, viewCount: Int) -> Int {
        guard viewCount > 1 else { return 0 }
        let slots = Swift.min(6, viewCount)
        let slot = (n / 2) % slots
        return Swift.min(viewCount - 1, slot * viewCount / slots + viewCount / (2 * slots))
    }

    /// Renders `camera` with the current model at about `maxWidth` pixels wide, so it costs a
    /// fraction of a training step. Call it between steps (the model is not changing then).
    static func render(
        trainer: GaussianTrainer, camera: Camera, shDegree: Int, maxWidth: Int = 720
    ) -> TrainingPreviewFrame? {
        let factor = Swift.max(1, (camera.width + maxWidth - 1) / maxWidth)
        let small = factor > 1 ? TrainingDriver.coarseCamera(camera, factor: factor) : camera
        guard small.width > 0, small.height > 0 else { return nil }
        let image = trainer.renderForEval(camera: small, shDegree: shDegree)   // [H, W, 3] in [0, 1]
        let bytes = (MLX.clip(image, min: 0, max: 1) * 255).asType(.uint8).asArray(UInt8.self)
        let pixels = small.width * small.height
        guard bytes.count == pixels * 3 else { return nil }
        var rgba = [UInt8](repeating: 255, count: pixels * 4)
        for p in 0..<pixels {
            rgba[p * 4] = bytes[p * 3]
            rgba[p * 4 + 1] = bytes[p * 3 + 1]
            rgba[p * 4 + 2] = bytes[p * 3 + 2]
        }
        return TrainingPreviewFrame(
            width: small.width, height: small.height, rgba: rgba,
            viewmat: camera.viewmat.asArray(Float.self))
    }
}
