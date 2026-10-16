// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX

extension Densification {
    /// Image-space importance map for the ImprovedGS+ edge score. 5x5 blur → Sobel → NMS, then
    /// normalized by the median of positive values.
    public struct EdgeImportanceMap: Sendable {
        public let width: Int
        public let height: Int
        public let scores: [Float]

        public init(width: Int, height: Int, scores: [Float]) {
            precondition(width > 0 && height > 0, "edge map dimensions must be positive")
            precondition(scores.count == width * height, "scores must be width*height")
            self.width = width
            self.height = height
            self.scores = scores
        }

        public func sampleMax(x: Float, y: Float, radius: Int) -> Float {
            guard x.isFinite, y.isFinite else { return 0 }
            // Pixel centers are at i + 0.5 (same convention as the rasterizer), so the index is
            // floor.
            let cx = Int(x.rounded(.down))
            let cy = Int(y.rounded(.down))
            if cx < 0 || cx >= width || cy < 0 || cy >= height { return 0 }
            let r = Swift.max(0, radius)
            let x0 = Swift.max(0, cx - r)
            let x1 = Swift.min(width - 1, cx + r)
            let y0 = Swift.max(0, cy - r)
            let y1 = Swift.min(height - 1, cy + r)
            var best: Float = 0
            for yy in y0...y1 {
                let row = yy * width
                for xx in x0...x1 {
                    let s = scores[row + xx]
                    if s > best { best = s }
                }
            }
            return best
        }

        /// The map whose value at each pixel is `sampleMax` with `radius` there, so a window lookup
        /// becomes one read: `windowMax(radius: r).sampleMax(x:y:radius: 0)` equals
        /// `sampleMax(x:y:radius: r)` bit for bit. Row pass then column pass over the same clamped
        /// window, with the same `> best` (from 0) comparison.
        public func windowMax(radius: Int) -> EdgeImportanceMap {
            let r = Swift.max(0, radius)
            let w = width, h = height
            var rows = [Float](repeating: 0, count: w * h)
            var out = [Float](repeating: 0, count: w * h)
            scores.withUnsafeBufferPointer { src in
                rows.withUnsafeMutableBufferPointer { dst in
                    for y in 0..<h {
                        let row = y * w
                        for x in 0..<w {
                            var best: Float = 0
                            for xx in Swift.max(0, x - r)...Swift.min(w - 1, x + r) {
                                let v = src[row + xx]
                                if v > best { best = v }
                            }
                            dst[row + x] = best
                        }
                    }
                }
            }
            rows.withUnsafeBufferPointer { src in
                out.withUnsafeMutableBufferPointer { dst in
                    for y in 0..<h {
                        let y0 = Swift.max(0, y - r), y1 = Swift.min(h - 1, y + r)
                        for x in 0..<w {
                            var best: Float = 0
                            for yy in y0...y1 {
                                let v = src[yy * w + x]
                                if v > best { best = v }
                            }
                            dst[y * w + x] = best
                        }
                    }
                }
            }
            return EdgeImportanceMap(width: w, height: h, scores: out)
        }
    }

    public static func edgeImportanceMap(image: MLXArray) -> EdgeImportanceMap {
        precondition(image.shape.count == 3 && image.shape[2] == 3, "image must be [H, W, 3]")
        eval(image)
        return edgeImportanceMap(
            rgb: image.asArray(Float.self),
            width: image.shape[1],
            height: image.shape[0]
        )
    }

    public static func edgeImportanceMap(
        rgb: [Float],
        width: Int,
        height: Int
    ) -> EdgeImportanceMap {
        precondition(width > 0 && height > 0, "image dimensions must be positive")
        precondition(rgb.count == width * height * 3, "rgb must be width*height*3")

        var gray = [Float](repeating: 0, count: width * height)
        for i in 0..<(width * height) {
            let r = rgb[i * 3 + 0]
            let g = rgb[i * 3 + 1]
            let b = rgb[i * 3 + 2]
            gray[i] = 0.299 * r + 0.587 * g + 0.114 * b
        }

        let blurred = gaussianBlur5x5(gray, width: width, height: height)
        let (magnitude, orientation) = sobelMagnitudeOrientation(
            blurred, width: width, height: height
        )
        let nms = nonMaximumSuppression(
            magnitude: magnitude, orientation: orientation,
            width: width, height: height
        )
        let normalized = medianNormalizePositive(nms)
        return EdgeImportanceMap(width: width, height: height, scores: normalized)
    }

    /// `stepToMapScale` maps step pixel coordinates to edge-map pixel coordinates. The edge map is
    /// built at full resolution, so it is 2 for a coarse 1/2 step and 1.5 for fine 2/3 (the cache's
    /// means2d / radii / conics are at the step's resolution).
    public static func edgeImportanceScores(
        edgeMap: EdgeImportanceMap,
        cache: GaussianRenderer.ForwardCache,
        stepToMapScale: Float = 1
    ) -> [Float] {
        let n = cache.means2d.shape[0]
        guard n > 0 else { return [] }
        eval(cache.means2d, cache.radii, cache.depths, cache.conics, cache.opacities)
        return edgeImportanceScores(
            edgeMap: edgeMap,
            means2d: cache.means2d.asArray(Float.self),
            radii: cache.radii.asArray(Int32.self),
            depths: cache.depths.asArray(Float.self),
            conics: cache.conics.asArray(Float.self),
            opacities: cache.opacities.asArray(Float.self),
            stepToMapScale: stepToMapScale
        )
    }

    public static func edgeImportanceScores(
        edgeMap: EdgeImportanceMap,
        means2d: [Float],
        radii: [Int32],
        depths: [Float],
        conics: [Float],
        opacities: [Float],
        stepToMapScale: Float = 1
    ) -> [Float] {
        let n = depths.count
        guard n > 0 else { return [] }
        precondition(means2d.count == n * 2, "means2d must be [N, 2]")
        precondition(radii.count == n * 2, "radii must be [N, 2]")
        precondition(conics.count == n * 3, "conics must be [N, 3]")
        precondition(opacities.count == n, "opacities must be [N]")
        precondition(stepToMapScale > 0 && stepToMapScale.isFinite, "stepToMapScale must be positive")

        let alphaThreshold: Float = 1.0 / 255.0
        // Evaluate the footprint in step pixels and map only the sample position to edge-map
        // pixels. The max window covers one step pixel.
        let k = stepToMapScale
        let window = Swift.max(1, Int(k.rounded(.up)))
        // Up to 25 samples per Gaussian each took a (2·window+1)² max; one pass over the map makes
        // each sample a single read.
        let windowed = edgeMap.windowMax(radius: window)
        var scores = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let rx = Int(radii[i * 2 + 0])
            let ry = Int(radii[i * 2 + 1])
            let opacity = opacities[i]
            guard depths[i] > 0, rx > 0, ry > 0, opacity > alphaThreshold else { continue }

            let cx = means2d[i * 2 + 0]
            let cy = means2d[i * 2 + 1]
            let conicBase = i * 3
            let a = conics[conicBase + 0]
            let b = conics[conicBase + 1]
            let c = conics[conicBase + 2]
            guard cx.isFinite, cy.isFinite, a.isFinite, b.isFinite, c.isFinite else { continue }

            var best: Float = 0
            let xs = footprintOffsets(radius: rx)
            let ys = footprintOffsets(radius: ry)
            for dy in ys {
                for dx in xs {
                    let sigma = 0.5 * (a * dx * dx + c * dy * dy) + b * dx * dy
                    guard sigma >= 0, sigma.isFinite else { continue }
                    let alpha = Swift.min(0.99, opacity * exp(-sigma))
                    guard alpha >= alphaThreshold else { continue }
                    let edge = windowed.sampleMax(x: (cx + dx) * k, y: (cy + dy) * k, radius: 0)
                    if edge > best { best = edge }
                }
            }
            scores[i] = best
        }
        return scores
    }

    private static func footprintOffsets(radius: Int) -> [Float] {
        let r = Swift.max(1, radius)
        if r == 1 { return [-1, 0, 1] }
        let half = Swift.max(1, r / 2)
        return [Float(-r), Float(-half), 0, Float(half), Float(r)]
    }

    private static func gaussianBlur5x5(
        _ input: [Float],
        width: Int,
        height: Int
    ) -> [Float] {
        let kernel: [Float] = [1, 4, 6, 4, 1]
        let norm: Float = 16
        var tmp = [Float](repeating: 0, count: width * height)
        var out = [Float](repeating: 0, count: width * height)

        for y in 0..<height {
            for x in 0..<width {
                var acc: Float = 0
                for k in 0..<5 {
                    let xx = clampIndex(x + k - 2, upper: width)
                    acc += kernel[k] * input[y * width + xx]
                }
                tmp[y * width + x] = acc / norm
            }
        }

        for y in 0..<height {
            for x in 0..<width {
                var acc: Float = 0
                for k in 0..<5 {
                    let yy = clampIndex(y + k - 2, upper: height)
                    acc += kernel[k] * tmp[yy * width + x]
                }
                out[y * width + x] = acc / norm
            }
        }
        return out
    }

    private static func sobelMagnitudeOrientation(
        _ input: [Float],
        width: Int,
        height: Int
    ) -> (magnitude: [Float], orientation: [Float]) {
        var magnitude = [Float](repeating: 0, count: width * height)
        var orientation = [Float](repeating: 0, count: width * height)
        guard width >= 3, height >= 3 else { return (magnitude, orientation) }

        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                let i00 = input[(y - 1) * width + (x - 1)]
                let i01 = input[(y - 1) * width + x]
                let i02 = input[(y - 1) * width + (x + 1)]
                let i10 = input[y * width + (x - 1)]
                let i12 = input[y * width + (x + 1)]
                let i20 = input[(y + 1) * width + (x - 1)]
                let i21 = input[(y + 1) * width + x]
                let i22 = input[(y + 1) * width + (x + 1)]

                let gx = -i00 + i02 - 2 * i10 + 2 * i12 - i20 + i22
                let gy = -i00 - 2 * i01 - i02 + i20 + 2 * i21 + i22
                let idx = y * width + x
                magnitude[idx] = (gx * gx + gy * gy).squareRoot()
                orientation[idx] = atan2(gy, gx)
            }
        }
        return (magnitude, orientation)
    }

    private static func nonMaximumSuppression(
        magnitude: [Float],
        orientation: [Float],
        width: Int,
        height: Int
    ) -> [Float] {
        var out = [Float](repeating: 0, count: width * height)
        guard width >= 3, height >= 3 else { return out }

        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                let idx = y * width + x
                let angle = orientation[idx] * 180 / Float.pi
                let a = angle < 0 ? angle + 180 : angle
                let m = magnitude[idx]

                let q: Float
                let r: Float
                if a < 22.5 || a >= 157.5 {
                    q = magnitude[y * width + (x + 1)]
                    r = magnitude[y * width + (x - 1)]
                } else if a < 67.5 {
                    q = magnitude[(y + 1) * width + (x - 1)]
                    r = magnitude[(y - 1) * width + (x + 1)]
                } else if a < 112.5 {
                    q = magnitude[(y + 1) * width + x]
                    r = magnitude[(y - 1) * width + x]
                } else {
                    q = magnitude[(y - 1) * width + (x - 1)]
                    r = magnitude[(y + 1) * width + (x + 1)]
                }

                if m >= q && m >= r {
                    out[idx] = m
                }
            }
        }
        return out
    }

    private static func medianNormalizePositive(_ values: [Float]) -> [Float] {
        let positives = values.filter { $0.isFinite && $0 > 0 }.sorted()
        guard !positives.isEmpty else {
            return [Float](repeating: 0, count: values.count)
        }
        let median = positives[positives.count / 2]
        let denom = Swift.max(median, 1e-6)
        return values.map { value in
            guard value.isFinite, value > 0 else { return 0 }
            return value / denom
        }
    }

    private static func clampIndex(_ value: Int, upper: Int) -> Int {
        Swift.min(Swift.max(value, 0), upper - 1)
    }
}
