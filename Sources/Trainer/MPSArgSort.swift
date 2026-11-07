// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import Metal
import MetalPerformanceShadersGraph
import MLX

/// `argSort` of 1-D int32 keys with MPSGraph, for a GPU where MLX's sort is wrong (the A14's: its
/// shader compiler miscompiles the `short`-indexed threadgroup loops of MLX's sort kernels).
/// MPSGraph sorts correctly there, and 1M keys take about 9 ms on an A14 against 70 ms on its CPU.
final class MPSArgSort: @unchecked Sendable {
    /// nil when Metal or MPSGraph is unavailable, or when MPSGraph's own sort fails the check too.
    static let shared: MPSArgSort? = {
        guard let sorter = MPSArgSort(), sorter.passesSelfTest() else { return nil }
        return sorter
    }()

    /// Inputs are padded up to a multiple of this, so a few graph shapes are compiled and reused
    /// rather than one per step (the pair count changes every step; a new shape costs a compile,
    /// about 7 ms for 300k keys on an A14).
    static let sizeQuantum = 1 << 16

    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let graph = MPSGraph()
    private let input: MPSGraphTensor
    private let output: MPSGraphTensor
    private let lock = NSLock()

    private init?() {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            return nil
        }
        self.device = device
        self.queue = queue
        input = graph.placeholder(shape: nil, dataType: .int32, name: nil)
        output = graph.argSort(input, axis: 0, name: nil)
    }

    /// Indices that sort `keys` ([n] int32) ascending.
    func argSort(_ keys: MLXArray) -> MLXArray {
        let n = keys.size
        guard n > 0 else { return MLXArray.zeros([0], dtype: .int32) }
        let padded = (n + Self.sizeQuantum - 1) / Self.sizeQuantum * Self.sizeQuantum
        let input = padded == n
            ? keys.reshaped([n])
            : MLX.concatenated([keys.reshaped([n]), MLXArray.full([padded - n], values: MLXArray(Int32.max))])
        lock.lock()
        defer { lock.unlock() }
        var order = [Int32](repeating: 0, count: padded)
        // MPSGraph hands back its buffers autoreleased (the input's wrapper, the result and its
        // ndarray). A caller without a pool of its own, such as the thumbnail pass that renders
        // every view after training, kept every call's buffers until it ended: 236 views on an
        // iPad Air (4th generation) went past its memory limit. Drain them here, every call.
        let sorted: Bool = autoreleasepool {
            guard let buffer = input.asMTLBuffer(device: device, noCopy: true)
                    ?? input.asMTLBuffer(device: device, noCopy: false) else {
                return false
            }
            let data = MPSGraphTensorData(buffer, shape: [NSNumber(value: padded)], dataType: .int32)
            let result = graph.run(with: queue, feeds: [self.input: data], targetTensors: [output],
                                   targetOperations: nil)
            result[output]?.mpsndarray().readBytes(&order, strideBytes: nil)
            withExtendedLifetime(input) {}
            return true
        }
        guard sorted else { return Self.cpuArgSort(keys) }
        // The padding sorts last, but a real key may equal it, so drop the padding by index.
        if padded != n { order.removeAll { Int($0) >= n } }
        return MLXArray(order, [n])
    }

    /// The same order on the CPU: each key packed with its index, so one sort of 64-bit values
    /// gives it.
    static func cpuArgSort(_ keys: MLXArray) -> MLXArray {
        let k = keys.asArray(Int32.self)
        var packed = [UInt64](repeating: 0, count: k.count)
        for i in 0..<k.count { packed[i] = UInt64(UInt32(bitPattern: k[i])) << 32 | UInt64(i) }
        packed.sort()
        return MLXArray(packed.map { Int32(truncatingIfNeeded: $0 & 0xffff_ffff) }, [k.count])
    }

    private func passesSelfTest() -> Bool {
        var rng = SystemRandomNumberGenerator()
        for n in [1000, 70_000] {
            let keys = (0..<n).map { _ in Int32(truncatingIfNeeded: rng.next() & 0x7fff_ffff) }
            let order = argSort(MLXArray(keys, [n])).asArray(Int32.self)
            guard ArgSortCheck.sorts(keys, by: order) else { return false }
        }
        return true
    }
}

/// Whether `order` is a permutation of `keys`' indices that puts them in ascending order.
enum ArgSortCheck {
    static func sorts(_ keys: [Int32], by order: [Int32]) -> Bool {
        let n = keys.count
        guard order.count == n else { return false }
        var seen = [Bool](repeating: false, count: n)
        for i in 0..<n {
            let b = Int(order[i])
            guard b >= 0, b < n, !seen[b] else { return false }
            seen[b] = true
            if i > 0, keys[Int(order[i - 1])] > keys[b] { return false }
        }
        return true
    }
}
