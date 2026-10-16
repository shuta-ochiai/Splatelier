// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Fuses multiple seeds (MVS / BA sparse / LiDAR) into one dense prior by a per-voxel
// confidence-weighted average (PocketGS Operator G, Sec. III-A "fused into P").

import Foundation
import MLX

public enum SeedFusion {
    /// Merges the points in each voxel into one by confidence-weighted averaging of position and
    /// color.
    /// - Parameters:
    ///   - clouds: (positions [N_i, 3], colors [N_i, 3], confidences [N_i]) for each source
    ///   - voxelSize: Voxel edge length (m)
    public static func fuse(
        clouds: [(positions: MLXArray, colors: MLXArray, confidences: MLXArray)],
        voxelSize: Float
    ) -> (positions: MLXArray, colors: MLXArray) {
        var pos: [Float] = []
        var col: [Float] = []
        var conf: [Float] = []
        for cloud in clouds {
            eval(cloud.positions, cloud.colors, cloud.confidences)
            pos += cloud.positions.asArray(Float.self)
            col += cloud.colors.asArray(Float.self)
            conf += cloud.confidences.asArray(Float.self)
        }
        let n = conf.count
        guard n > 0 else {
            return (positions: MLXArray.zeros([0, 3]), colors: MLXArray.zeros([0, 3]))
        }
        precondition(pos.count == n * 3 && col.count == n * 3,
                     "positions/colors/confidences count mismatch across sources")

        // Pack the voxel index into an Int64 key, 21 bits per axis (same convention as
        // CostVolumeMVS's voxel dedup).
        let bias: Int64 = 1 << 20
        // XOR the key with a per-call random value before inserting into the Dictionary. MVS points
        // arrive in the iteration order (bucket order) of the voxel-dedup Dictionary (8.4M
        // buckets). Swift seeds the hash with the storage address, so if the 4.2M-bucket table
        // grown here is allocated at the same address as the just-freed dedup table, the hashes
        // match: the input walks the smaller table twice, the second pass lands on buckets already
        // 63% full, and the probe clusters keep growing until the cost is quadratic in the point
        // count (same shape as Swift issue SR-3268). 2026-10-09
        // Tile: MVS 5.33M + sparse 140K took 117 s on device, 177-234 s reproduced on Mac → 1 s
        // with the xor.
        let salt = Int64.random(in: .min ... .max)
        var sums: [Int64: (px: Double, py: Double, pz: Double,
                            cr: Double, cg: Double, cb: Double, wsum: Double)] = [:]
        sums.reserveCapacity(n / 4)
        let invSize = 1.0 / voxelSize
        for i in 0..<n {
            let x = pos[i * 3 + 0], y = pos[i * 3 + 1], z = pos[i * 3 + 2]
            let vx = Int64((x * invSize).rounded(.down))
            let vy = Int64((y * invSize).rounded(.down))
            let vz = Int64((z * invSize).rounded(.down))
            if abs(vx) >= bias || abs(vy) >= bias || abs(vz) >= bias { continue }
            let key = (((vx + bias) << 42) | ((vy + bias) << 21) | (vz + bias)) ^ salt
            // Floor the weight so a voxel with only confidence-0 points does not get a zero weight
            // sum.
            let w = Double(Swift.max(conf[i], 1e-4))
            var entry = sums[key] ?? (0, 0, 0, 0, 0, 0, 0)
            entry.px += Double(x) * w; entry.py += Double(y) * w; entry.pz += Double(z) * w
            entry.cr += Double(col[i * 3 + 0]) * w
            entry.cg += Double(col[i * 3 + 1]) * w
            entry.cb += Double(col[i * 3 + 2]) * w
            entry.wsum += w
            sums[key] = entry
        }
        var outPos: [Float] = []
        var outCol: [Float] = []
        outPos.reserveCapacity(sums.count * 3)
        outCol.reserveCapacity(sums.count * 3)
        for (_, e) in sums {
            let invW = 1.0 / e.wsum
            outPos += [Float(e.px * invW), Float(e.py * invW), Float(e.pz * invW)]
            outCol += [Float(e.cr * invW), Float(e.cg * invW), Float(e.cb * invW)]
        }
        let m = sums.count
        if m == 0 {
            return (positions: MLXArray.zeros([0, 3]), colors: MLXArray.zeros([0, 3]))
        }
        return (positions: MLXArray(outPos, [m, 3]), colors: MLXArray(outCol, [m, 3]))
    }
}
