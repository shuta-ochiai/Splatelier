// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// FlexGlue matching (cvg/FlexGlue, MIT; its descriptor is a DINOv3 ConvNeXt-B under the DINOv3
// License) for wide-angle loop closure. Dense features per frame and the self/cross transformer per
// pair run as Core ML models on the Neural Engine; the correspondence head of the last layer runs in
// MLX with the trained weights. Keypoints only say where to ask: each one gets a predicted point in
// the other image, read back to check the cycle, as FlexGlue.match does for keypoints in both images
// (no refinement, no sparse mode).
//
// The transformer (6 of the checkpoint's 9 layers, rewritten for the Neural Engine: channels-first
// tokens, 1×1 convolutions, per-head attention) is split after layer 4: a pair whose covisibility
// there is low is not run through the rest. Model files, made by scripts/flexglue/export_models.py:
// - flexglue_descriptor.mlpackage: image [1, 3, 384, 512] in 0...1 → features [1, 256, 48, 64]
// - flexglue_transformer_a.mlpackage: f0, f1 → x0, x1 [1, 512, 1, 3072], covis [1] (layers 1-4)
// - flexglue_transformer_b.mlpackage: y0, y1 → x0, x1, covis (layers 5-6)
// - flexglue_head.safetensors: the correspondence head after layer 6.
// Each may be there compiled (.mlmodelc, as bundled with the app) instead of as a package.

import CoreML
import Foundation
import MLX
import MLXFast

public final class FlexGlueMatcher: @unchecked Sendable {
    public static let inputHeight = 384, inputWidth = 512
    static let gridH = inputHeight / 8, gridW = inputWidth / 8
    static let modelNames = ["flexglue_descriptor", "flexglue_transformer_a", "flexglue_transformer_b"]
    static let headFile = "flexglue_head.safetensors"

    /// Whether `directory` holds the models (compiled or as packages).
    public static func modelsPresent(in directory: URL) -> Bool {
        let fm = FileManager.default
        return modelNames.allSatisfy { name in
            ["mlmodelc", "mlpackage"].contains { fm.fileExists(atPath: directory.appendingPathComponent("\(name).\($0)").path) }
        } && fm.fileExists(atPath: directory.appendingPathComponent(headFile).path)
    }

    public struct Result {
        public let covisibility: Float
        /// Matches of both directions: point in A, point in B (pixels of each frame), confidence.
        public let pointsA: [SIMD2<Float>]
        public let pointsB: [SIMD2<Float>]
        public let confidence: [Float]
    }

    private let descriptor: MLModel
    private let transformerA: MLModel
    private let transformerB: MLModel
    private let head: [String: MLXArray]
    /// A pair whose covisibility after the first part is below this skips the second.
    public let earlyStopCovisibility: Float
    public let filterThreshold: Float
    public let maxCyclePx: Float

    /// Loads the models from `directory`. A package (.mlpackage) is compiled once into
    /// `compiledDirectory`: Core ML reuses its Neural Engine build only for a model loaded from the
    /// same place, and compiling takes seconds.
    public init(directory: URL, compiledDirectory: URL? = nil, earlyStopCovisibility: Float = 0.6,
                filterThreshold: Float = 0.1, maxCyclePx: Float = 3) throws {
        let fm = FileManager.default
        let cache = compiledDirectory ?? fm.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("FlexGlue")
        func load(_ name: String) throws -> MLModel {
            let c = MLModelConfiguration(); c.computeUnits = .cpuAndNeuralEngine
            var url = directory.appendingPathComponent("\(name).mlmodelc")
            if !fm.fileExists(atPath: url.path) {
                url = cache.appendingPathComponent("\(name).mlmodelc")
                if !fm.fileExists(atPath: url.path) {
                    let temp = try MLModel.compileModel(at: directory.appendingPathComponent("\(name).mlpackage"))
                    try fm.createDirectory(at: cache, withIntermediateDirectories: true)
                    try fm.moveItem(at: temp, to: url)
                }
            }
            return try Self.loadModel(at: url, configuration: c)
        }
        descriptor = try load(Self.modelNames[0])
        transformerA = try load(Self.modelNames[1])
        transformerB = try load(Self.modelNames[2])
        head = try MLX.loadArrays(url: directory.appendingPathComponent(Self.headFile)).mapValues { $0.asType(.float32) }
        self.earlyStopCovisibility = earlyStopCovisibility
        self.filterThreshold = filterThreshold
        self.maxCyclePx = maxCyclePx
    }

    /// Loads a compiled model, waiting for Core ML's asynchronous loader: an encrypted model (as the
    /// app bundles them when built with a model key) loads only that way, fetching its key from
    /// Apple on first use. Not on the main thread.
    static func loadModel(at url: URL, configuration: MLModelConfiguration) throws -> MLModel {
        final class Box: @unchecked Sendable { var result: Swift.Result<MLModel, Error>? }
        let box = Box(), done = DispatchSemaphore(value: 0)
        MLModel.load(contentsOf: url, configuration: configuration) { result in
            box.result = result
            done.signal()
        }
        done.wait()
        return try box.result!.get()
    }

    /// The elements of a Core ML array in row-major order. Core ML pads rows (strides): a last
    /// dimension of 60 float16 values, for one, is laid out in rows of 64.
    static func contiguous(_ a: MLMultiArray) -> [Float] {
        a.dataType == .float16 ? contiguous(a, as: Float16.self) : contiguous(a, as: Float.self)
    }

    private static func contiguous<T: BinaryFloatingPoint>(_ a: MLMultiArray, as: T.Type) -> [Float] {
        let shape = a.shape.map(\.intValue), strides = a.strides.map(\.intValue)
        let n = shape.reduce(1, *)
        var packed = [Int](repeating: 1, count: shape.count)
        for d in stride(from: shape.count - 2, through: 0, by: -1) { packed[d] = packed[d + 1] * shape[d + 1] }
        if packed == strides || n <= 1 {
            let src = a.dataPointer.bindMemory(to: T.self, capacity: n)
            return (0..<n).map { Float(src[$0]) }
        }
        let extent = zip(shape, strides).map { ($0 - 1) * $1 }.reduce(0, +) + 1
        let src = a.dataPointer.bindMemory(to: T.self, capacity: extent)
        var out = [Float](repeating: 0, count: n)
        var index = [Int](repeating: 0, count: shape.count)
        for k in 0..<n {
            var offset = 0
            for d in 0..<shape.count { offset += index[d] * strides[d] }
            out[k] = Float(src[offset])
            var d = shape.count - 1
            while d >= 0 { index[d] += 1; if index[d] < shape[d] { break }; index[d] = 0; d -= 1 }
        }
        return out
    }

    // MARK: Descriptor

    /// Bilinear resize of an image [H, W, C] to [h, w, C], as torch's interpolate with
    /// align_corners=False and no antialiasing.
    static func resize(_ image: MLXArray, h: Int, w: Int) -> MLXArray {
        func axis(_ n: Int, _ m: Int) -> (MLXArray, MLXArray, MLXArray) {
            let scale = Float(n) / Float(m)
            var i0 = [Int32](), i1 = [Int32](), lam = [Float]()
            for o in 0..<m {
                let src = max(0, (Float(o) + 0.5) * scale - 0.5)
                let f = Int(src)
                i0.append(Int32(min(f, n - 1))); i1.append(Int32(min(f + 1, n - 1))); lam.append(src - Float(f))
            }
            return (MLXArray(i0), MLXArray(i1), MLXArray(lam))
        }
        let (y0, y1, ly) = axis(image.dim(0), h), (x0, x1, lx) = axis(image.dim(1), w)
        let rows = take(image, y0, axis: 0) * (1 - ly.reshaped([h, 1, 1])) + take(image, y1, axis: 0) * ly.reshaped([h, 1, 1])
        return take(rows, x0, axis: 1) * (1 - lx.reshaped([1, w, 1])) + take(rows, x1, axis: 1) * lx.reshaped([1, w, 1])
    }

    /// The descriptor's input for a frame (image [H, W, 3], RGB in 0...1). MLX, so on the thread
    /// that runs the MLX work.
    public func descriptorInput(_ image: MLXArray) throws -> MLMultiArray {
        let h = Self.inputHeight, w = Self.inputWidth
        let values = Self.resize(image.asType(.float32), h: h, w: w).transposed(2, 0, 1).asArray(Float.self)
        let a = try MLMultiArray(shape: [1, 3, NSNumber(value: h), NSNumber(value: w)], dataType: .float32)
        values.withUnsafeBufferPointer { src in
            a.dataPointer.bindMemory(to: Float.self, capacity: values.count).update(from: src.baseAddress!, count: values.count)
        }
        return a
    }

    /// Dense features of a descriptor input as float16 bytes (Core ML; any thread).
    public func describe(_ input: MLMultiArray) throws -> Data {
        let out = try descriptor.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": input]))
        let h = Self.contiguous(out.featureValue(for: "features")!.multiArrayValue!).map { Float16($0) }
        return h.withUnsafeBytes { Data($0) }
    }

    /// float16 features (`describe`) as a float32 model input.
    public func features(_ d: Data) throws -> MLMultiArray {
        let n = d.count / 2
        let a = try MLMultiArray(shape: [1, 256, NSNumber(value: Self.gridH), NSNumber(value: Self.gridW)], dataType: .float32)
        precondition(a.count == n, "feature size")
        let dst = a.dataPointer.bindMemory(to: Float.self, capacity: n)
        d.withUnsafeBytes { raw in
            let src = raw.bindMemory(to: Float16.self)
            for i in 0..<n { dst[i] = Float(src[i]) }
        }
        return a
    }

    // MARK: Transformer

    /// The transformer's output for a pair: tokens of both images (nil when stopped early) and
    /// the covisibility score.
    public struct Tokens: @unchecked Sendable {
        let x0: MLMultiArray?, x1: MLMultiArray?
        public let covisibility: Float
        public var stoppedEarly: Bool { x0 == nil }
    }

    /// The transformer of a pair (Core ML; any thread, one call at a time).
    public func transform(_ featuresA: MLMultiArray, _ featuresB: MLMultiArray) throws -> Tokens {
        func run(_ m: MLModel, _ inputs: [String: MLMultiArray]) throws -> (MLMultiArray, MLMultiArray, Float) {
            let out = try m.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs))
            return (out.featureValue(for: "x0")!.multiArrayValue!, out.featureValue(for: "x1")!.multiArrayValue!,
                    out.featureValue(for: "covis")!.multiArrayValue![0].floatValue)
        }
        let (y0, y1, ca) = try run(transformerA, ["f0": featuresA, "f1": featuresB])
        if ca < earlyStopCovisibility { return Tokens(x0: nil, x1: nil, covisibility: ca) }
        let (x0, x1, c) = try run(transformerB, ["y0": y0, "y1": y1])
        return Tokens(x0: x0, x1: x1, covisibility: c)
    }

    /// [1, D, 1, S] → [S, D] float32.
    static func tokens(_ a: MLMultiArray) -> MLXArray {
        let n = a.count, s = gridH * gridW
        return MLXArray(contiguous(a)).reshaped([n / s, s]).T.asType(.float32)
    }

    // MARK: Correspondence head

    private func linear(_ name: String, _ x: MLXArray) -> MLXArray {
        matmul(x, head[name + ".weight"]!.T) + head[name + ".bias"]!
    }

    private func mlp(_ name: String, _ x: MLXArray) -> MLXArray {   // Linear, LayerNorm, ReLU, Linear
        let h = MLXFast.layerNorm(linear(name + ".0", x), weight: head[name + ".1.weight"]!, bias: head[name + ".1.bias"]!, eps: 1e-5)
        return linear(name + ".3", maximum(h, 0))
    }

    private static func normalize(_ x: MLXArray) -> MLXArray {
        x / maximum(sqrt(sum(x * x, axis: -1, keepDims: true)), 1e-12)
    }

    /// Bilinear samples (grid_sample, border padding, align_corners=False) of features [h*w, D] at
    /// normalized points [N, 2].
    static func sample(_ f: MLXArray, _ p: MLXArray, h: Int, w: Int) -> MLXArray {
        let px = clip(((p[0..., 0] + 1) * Float(w) - 1) / 2, min: 0, max: Float(w - 1))
        let py = clip(((p[0..., 1] + 1) * Float(h) - 1) / 2, min: 0, max: Float(h - 1))
        let fx = floor(px), fy = floor(py)
        let ax = (px - fx).expandedDimensions(axis: -1), ay = (py - fy).expandedDimensions(axis: -1)
        let x0 = fx.asType(.int32), y0 = fy.asType(.int32)
        let x1 = minimum(x0 + 1, Int32(w - 1)), y1 = minimum(y0 + 1, Int32(h - 1))
        func g(_ y: MLXArray, _ x: MLXArray) -> MLXArray { take(f, y * Int32(w) + x, axis: 0) }
        let top = g(y0, x0) * (1 - ax) + g(y0, x1) * ax
        let bottom = g(y1, x0) * (1 - ax) + g(y1, x1) * ax
        return top * (1 - ay) + bottom * ay
    }

    /// The head on normalized points [N, 2] of image i: predicted normalized points in j and
    /// confidence (matchability × window mass), as MatchHead.match_points.
    func correspond(points: MLXArray, cfI: MLXArray, cfJ: MLXArray) -> (MLXArray, MLXArray) {
        let h = Self.gridH, w = Self.gridW, k = min(512, h * w)
        let search = Self.normalize(linear("to_search", cfJ))                     // [n, 64]
        let fq = Self.sample(cfI, points, h: h, w: w)                              // [N, D]
        let query = Self.normalize(linear("to_search", fq))
        let logits = matmul(query, search.T) * head["temperature"]!                // [N, n]
        let nq = points.dim(0)
        // 1.5-entmax threshold from the top k scores (sorted, descending)
        let topV = -sorted(-top(logits, k: k, axis: -1), axis: -1)
        let vmax = topV[0..., 0..<1]
        let ys = (topV - vmax) / 2
        let rho = MLXArray((1...k).map(Float.init))
        let mean = cumsum(ys, axis: -1) / rho, meanSq = cumsum(ys * ys, axis: -1) / rho
        let tau = mean - sqrt(maximum((1 - rho * (meanSq - mean * mean)) / rho, 0))
        let support = sum((tau .<= ys).asType(.int32), axis: -1, keepDims: true)
        let t = takeAlong(tau, support - 1, axis: -1).squeezed(axis: -1)
        // expected position over the 3×3 window around the maximum
        let idx = argMax(logits, axis: -1).asType(.int32)
        let iy = floorDivide(idx, Int32(w)), ix = idx % Int32(w)
        var wsum = MLXArray.zeros([nq]), mass = MLXArray.zeros([nq])
        var sx = MLXArray.zeros([nq]), sy = MLXArray.zeros([nq])
        let vm = vmax.squeezed(axis: -1)
        for dy in Int32(-1)...1 { for dx in Int32(-1)...1 {
            let yy = iy + dy, xx = ix + dx
            let inside = ((yy .>= 0) .&& (yy .< Int32(h)) .&& (xx .>= 0) .&& (xx .< Int32(w))).asType(.float32)
            let cy = clip(yy, min: 0, max: h - 1), cx = clip(xx, min: 0, max: w - 1)
            let l = takeAlong(logits, (cy * Int32(w) + cx).expandedDimensions(axis: -1), axis: -1).squeezed(axis: -1)
            let pr = maximum((l - vm) / 2 - t, 0)
            let p2 = pr * pr
            wsum = wsum + p2
            mass = mass + p2 * inside
            sx = sx + p2 * ((cx.asType(.float32) + 0.5) / Float(w) * 2 - 1)
            sy = sy + p2 * ((cy.asType(.float32) + 0.5) / Float(h) * 2 - 1)
        } }
        var p = stacked([sx / wsum, sy / wsum], axis: -1)
        // sub-cell offset
        let fm = Self.sample(cfJ, p, h: h, w: w)
        let x = concatenated([mlp("offset_head.query_encoder", fq), mlp("offset_head.match_encoder", fm)], axis: -1)
        let ol = mlp("offset_head.head", x).reshaped([nq, 2, 18])[0..., 0..., 0..<17]
        let bins = MLXArray((0..<17).map(Float.init))
        let delta = (sum(softmax(ol, axis: -1) * bins, axis: -1) / 16 - 0.5) * 2
        p = p + delta * MLXArray([2 / Float(w), 2 / Float(h)])
        let matchability = sigmoid(linear("matchability", fq).squeezed(axis: -1))
        return (p, matchability * mass)
    }

    // MARK: Pair

    /// Matches between frames A and B from the transformer's output and keypoints (pixels, origin
    /// at the top-left image corner) in images of `sizeA`, `sizeB` (width, height). MLX; none when
    /// the pair stopped early.
    public func readout(_ tokens: Tokens, keypointsA: [SIMD2<Float>], sizeA: SIMD2<Float>,
                        keypointsB: [SIMD2<Float>], sizeB: SIMD2<Float>) -> Result {
        guard let t0 = tokens.x0, let t1 = tokens.x1 else {
            return Result(covisibility: tokens.covisibility, pointsA: [], pointsB: [], confidence: [])
        }
        let cf = [Self.tokens(t0), Self.tokens(t1)]
        let resized = MLXArray([Float(Self.inputWidth), Float(Self.inputHeight)])
        let kps = [keypointsA, keypointsB], sizes = [sizeA, sizeB]
        var pa: [SIMD2<Float>] = [], pb: [SIMD2<Float>] = [], conf: [Float] = []
        for i in 0..<2 where !kps[i].isEmpty {
            let j = 1 - i
            let kp = MLXArray(kps[i].flatMap { [$0.x, $0.y] }).reshaped([kps[i].count, 2])
            let pn = kp / MLXArray([sizes[i].x, sizes[i].y]) * 2 - 1
            let (p, c) = correspond(points: pn, cfI: cf[i], cfJ: cf[j])
            let (back, bc) = correspond(points: p, cfI: cf[j], cfJ: cf[i])
            let err = sqrt(sum(square((back - pn) / 2 * resized), axis: -1))     // descriptor-input pixels of i
            let confidence = sqrt(c * bc)
            let pr = (p + 1) / 2 * resized                                          // descriptor-input pixels of j
            let kr = (pn + 1) / 2 * resized
            let inside = all((pr .> 0) .&& (pr .< resized), axis: -1) .&& all((kr .> 0) .&& (kr .< resized), axis: -1)
            let valid = inside .&& (err .< maxCyclePx) .&& (confidence .> filterThreshold)
            let pj = (p + 1) / 2 * MLXArray([sizes[j].x, sizes[j].y])
            eval(valid, pj, confidence)
            let v = valid.asArray(Bool.self), q = pj.asArray(Float.self), cc = confidence.asArray(Float.self)
            for (m, ok) in v.enumerated() where ok {
                let a = kps[i][m], b = SIMD2<Float>(q[2 * m], q[2 * m + 1])
                if i == 0 { pa.append(a); pb.append(b) } else { pa.append(b); pb.append(a) }
                conf.append(cc[m])
            }
        }
        return Result(covisibility: tokens.covisibility, pointsA: pa, pointsB: pb, confidence: conf)
    }
}
