// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX

/// Reads a binary little-endian 3DGS `.ply` into a `GaussianModel` (the inverse of ``PLYWriter``).
/// Properties are looked up by name, so standard 3DGS PLYs load too. f_dc → sh0, f_rest (optional)
/// → shN.
public enum PLYReader {

    enum Error: Swift.Error, CustomStringConvertible {
        case parse(String)
        var description: String {
            switch self {
            case .parse(let s): return "PLY parse error: \(s)"
            }
        }
    }

    public static func readGaussianPLY(url: URL) throws -> GaussianModel {
        let data = try Data(contentsOf: url)

        let marker = Data("end_header\n".utf8)
        guard let r = data.range(of: marker) else {
            throw Error.parse("no end_header")
        }
        let header = String(decoding: data[data.startIndex..<r.lowerBound], as: UTF8.self)
        let bodyStart = r.upperBound

        guard header.contains("format binary_little_endian") else {
            throw Error.parse("only binary_little_endian is supported")
        }

        var vertexCount = 0
        var props: [String] = []
        for line in header.split(separator: "\n") {
            let parts = line.split(separator: " ")
            if parts.count >= 3, parts[0] == "element", parts[1] == "vertex" {
                vertexCount = Int(parts[2]) ?? 0
            } else if parts.count >= 3, parts[0] == "property" {
                // Only float32 is supported (the writer emits nothing else).
                guard parts[1] == "float" || parts[1] == "float32" else {
                    throw Error.parse("non-float property \(parts.last.map(String.init) ?? "?")")
                }
                props.append(String(parts[2]))
            }
        }
        guard vertexCount > 0, !props.isEmpty else { throw Error.parse("empty/invalid header") }

        var index: [String: Int] = [:]
        for (i, p) in props.enumerated() { index[p] = i }
        func idx(_ name: String) throws -> Int {
            guard let i = index[name] else { throw Error.parse("missing property \(name)") }
            return i
        }
        let ix = try idx("x"), iy = try idx("y"), iz = try idx("z")
        let idc0 = try idx("f_dc_0"), idc1 = try idx("f_dc_1"), idc2 = try idx("f_dc_2")
        let iop = try idx("opacity")
        let is0 = try idx("scale_0"), is1 = try idx("scale_1"), is2 = try idx("scale_2")
        let ir0 = try idx("rot_0"), ir1 = try idx("rot_1"), ir2 = try idx("rot_2"), ir3 = try idx("rot_3")

        let stride = props.count
        let need = bodyStart + vertexCount * stride * 4
        guard data.count >= need else { throw Error.parse("truncated body") }

        let floats: [Float] = data[bodyStart...].withUnsafeBytes { raw in
            let f = raw.bindMemory(to: Float32.self)
            return Array(f[0 ..< vertexCount * stride])
        }

        // f_rest is channel-major and optional. To read PLYs of any degree (including degree-3 with
        // 45 columns), derive the file's coefficient count from the column count and take only the
        // first restBands. The channel stride is fileBands.
        let restBands = GaussianModel.shRestBands
        var fileRestTotal = 0
        while index["f_rest_\(fileRestTotal)"] != nil { fileRestTotal += 1 }
        let fileBands = fileRestTotal / 3                 // coefficients per channel in the file
        let usedBands = Swift.min(restBands, fileBands)
        // restIdx[ch*restBands + band] = column in the file (nil if absent).
        var restIdx = [Int?](repeating: nil, count: restBands * 3)
        for ch in 0..<3 {
            for band in 0..<usedBands {
                restIdx[ch * restBands + band] = index["f_rest_\(ch * fileBands + band)"]
            }
        }

        let n = vertexCount
        var means = [Float](repeating: 0, count: n * 3)
        var sh0 = [Float](repeating: 0, count: n * 3)
        var shN = [Float](repeating: 0, count: n * restBands * 3)
        var logScales = [Float](repeating: 0, count: n * 3)
        var logitOpac = [Float](repeating: 0, count: n)
        var quats = [Float](repeating: 0, count: n * 4)
        for i in 0..<n {
            let b = i * stride
            means[i * 3 + 0] = floats[b + ix]
            means[i * 3 + 1] = floats[b + iy]
            means[i * 3 + 2] = floats[b + iz]
            sh0[i * 3 + 0] = floats[b + idc0]
            sh0[i * 3 + 1] = floats[b + idc1]
            sh0[i * 3 + 2] = floats[b + idc2]
            // channel-major → band-major [N, restBands, 3]
            for ch in 0..<3 {
                for band in 0..<restBands {
                    if let col = restIdx[ch * restBands + band] {
                        shN[i * restBands * 3 + band * 3 + ch] = floats[b + col]
                    }
                }
            }
            logScales[i * 3 + 0] = floats[b + is0]
            logScales[i * 3 + 1] = floats[b + is1]
            logScales[i * 3 + 2] = floats[b + is2]
            logitOpac[i] = floats[b + iop]
            quats[i * 4 + 0] = floats[b + ir0]
            quats[i * 4 + 1] = floats[b + ir1]
            quats[i * 4 + 2] = floats[b + ir2]
            quats[i * 4 + 3] = floats[b + ir3]
        }

        return GaussianModel(
            means: MLXArray(means, [n, 3]),
            quats: MLXArray(quats, [n, 4]),
            logScales: MLXArray(logScales, [n, 3]),
            logitOpacities: MLXArray(logitOpac, [n]),
            sh0: MLXArray(sh0, [n, 3]),
            shN: MLXArray(shN, [n, restBands, 3])
        )
    }
}
