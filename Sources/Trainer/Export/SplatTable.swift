// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX

/// Gaussians as flat CPU arrays in standard 3DGS PLY space, the common input of every export
/// writer. Values are raw like the PLY: opacity is a logit, scales are logs, f_dc = sh0 (RGB = 0.5 +
/// C0·f_dc), quats are normalized (w, x, y, z). `shRest` is channel-major like PLY f_rest: each row
/// holds `restCoeffs` red coefficients, then green, then blue.
///
/// The last `backgroundCount` rows are the baked background (environment sphere). Writers that
/// reorder rows keep them at the tail, so the `<base>.background.json` sidecar stays valid.
public struct SplatTable: Sendable {
    public private(set) var count: Int
    public var positions: [Float]       // [N*3]
    public var logScales: [Float]       // [N*3]
    public var quats: [Float]           // [N*4] (w, x, y, z), normalized
    public var logitOpacities: [Float]  // [N]
    public var sh0: [Float]             // [N*3]
    public var shRest: [Float]          // [N*restCoeffs*3], channel-major
    /// Higher-order SH degree, 0...3.
    public private(set) var shDegree: Int
    public var backgroundCount: Int

    /// Higher-order coefficients per channel for `shDegree` (0, 3, 8, 15).
    public var restCoeffs: Int { Self.restCoeffs(degree: shDegree) }

    public static func restCoeffs(degree: Int) -> Int { [0, 3, 8, 15][degree] }

    public init(
        positions: [Float], logScales: [Float], quats: [Float], logitOpacities: [Float],
        sh0: [Float], shRest: [Float], shDegree: Int, backgroundCount: Int = 0
    ) {
        let n = logitOpacities.count
        precondition((0...3).contains(shDegree), "shDegree must be 0...3")
        precondition(positions.count == n * 3 && logScales.count == n * 3 && sh0.count == n * 3,
                     "positions/logScales/sh0 must be [N*3]")
        precondition(quats.count == n * 4, "quats must be [N*4]")
        precondition(shRest.count == n * Self.restCoeffs(degree: shDegree) * 3,
                     "shRest must be [N*restCoeffs*3]")
        precondition((0...n).contains(backgroundCount), "backgroundCount out of range")
        self.count = n
        self.positions = positions
        self.logScales = logScales
        self.quats = quats
        self.logitOpacities = logitOpacities
        self.sh0 = sh0
        self.shRest = shRest
        self.shDegree = shDegree
        self.backgroundCount = backgroundCount
    }

    /// Activated opacity of row `i`.
    @inline(__always) public func opacity(_ i: Int) -> Float {
        1 / (1 + expf(-logitOpacities[i]))
    }

    /// Rows in a new order (`order` is a full permutation of 0..<count). The caller keeps the
    /// background tail in place; see ``tailPreservingOrder(_:)``.
    public func permuted(_ order: [Int]) -> SplatTable {
        precondition(order.count == count, "order must be a full permutation")
        let rc = restCoeffs * 3
        var t = self
        for (dst, src) in order.enumerated() {
            for k in 0..<3 {
                t.positions[dst * 3 + k] = positions[src * 3 + k]
                t.logScales[dst * 3 + k] = logScales[src * 3 + k]
                t.sh0[dst * 3 + k] = sh0[src * 3 + k]
            }
            for k in 0..<4 { t.quats[dst * 4 + k] = quats[src * 4 + k] }
            t.logitOpacities[dst] = logitOpacities[src]
            for k in 0..<rc { t.shRest[dst * rc + k] = shRest[src * rc + k] }
        }
        return t
    }

    /// Applies `sort` separately to the foreground rows and the background rows, so the
    /// background stays at the tail. `sort` returns a permutation of the indices it is given.
    public func tailPreservingOrder(_ sort: ([Int]) -> [Int]) -> [Int] {
        let fg = count - backgroundCount
        return sort(Array(0..<fg)) + sort(Array(fg..<count))
    }

    /// Drops SH bands above `degree` (no-op when already at or below it).
    public func reducingSH(to degree: Int) -> SplatTable {
        guard degree < shDegree else { return self }
        let from = restCoeffs, to = Self.restCoeffs(degree: degree)
        var rest = [Float](repeating: 0, count: count * to * 3)
        for i in 0..<count {
            for ch in 0..<3 {
                for c in 0..<to {
                    rest[(i * 3 + ch) * to + c] = shRest[(i * 3 + ch) * from + c]
                }
            }
        }
        var t = self
        t.shRest = rest
        t.shDegree = degree
        return t
    }
}

// MARK: - From a trained model

extension SplatTable {

    /// The model's rows that pass `cull`, with the cull breakdown. The environment rows (the last
    /// `model.environmentCount`) are exempt from every cull except non-finite values, and become
    /// the table's background.
    public static func fromModel(
        _ model: GaussianModel, cull: PLYWriter.CullOptions = PLYWriter.CullOptions()
    ) -> (table: SplatTable, stats: PLYWriter.CullStats) {
        let n = model.count
        let bands = GaussianModel.shRestBands
        let means = model.means.asArray(Float.self)
        let sh0 = model.sh0.asArray(Float.self)            // [N*3]
        let shN = model.shN.asArray(Float.self)            // [N*bands*3], band-major
        let logScales = model.logScales.asArray(Float.self)
        let logitOpac = model.logitOpacities.asArray(Float.self)
        let quats = model.quats.asArray(Float.self)

        // Keep condition: finite, opacity > minOpacity, max scale < maxScale, coordinates <
        // maxAbsCoord. The environment sphere is an intentional far shell, so it is exempt from all
        // culls except NaN (fixed thresholds would wrongly cut it).
        let fgN = Swift.max(0, n - Swift.max(0, model.environmentCount))
        var keep = [Bool](repeating: true, count: n)
        var nNonFinite = 0, nLowOpac = 0, nGiant = 0, nOutlier = 0
        if cull.enabled {
            for i in 0..<n {
                var finite = true
                for k in 0..<3 where !means[i * 3 + k].isFinite { finite = false; break }
                if finite { for k in 0..<3 where !sh0[i * 3 + k].isFinite { finite = false; break } }
                if finite { for k in 0..<3 where !logScales[i * 3 + k].isFinite { finite = false; break } }
                if finite, !logitOpac[i].isFinite { finite = false }
                if finite { for k in 0..<4 where !quats[i * 4 + k].isFinite { finite = false; break } }
                if finite {
                    for k in 0..<(bands * 3) where !shN[i * bands * 3 + k].isFinite { finite = false; break }
                }
                if !finite { keep[i] = false; nNonFinite += 1; continue }

                // The environment sphere is exempt from the remaining culls.
                if i >= fgN { continue }

                let opac = 1.0 / (1.0 + expf(-logitOpac[i]))
                if opac <= cull.minOpacity { keep[i] = false; nLowOpac += 1; continue }

                let s0 = expf(logScales[i * 3 + 0])
                let s1 = expf(logScales[i * 3 + 1])
                let s2 = expf(logScales[i * 3 + 2])
                let sMax = Swift.max(s0, Swift.max(s1, s2))
                if sMax >= cull.maxScale { keep[i] = false; nGiant += 1; continue }

                if abs(means[i * 3 + 0]) >= cull.maxAbsCoord
                    || abs(means[i * 3 + 1]) >= cull.maxAbsCoord
                    || abs(means[i * 3 + 2]) >= cull.maxAbsCoord {
                    keep[i] = false; nOutlier += 1; continue
                }
            }
        }

        let kept = keep.reduce(0) { $0 + ($1 ? 1 : 0) }
        let keptEnvironment = keep[fgN...].reduce(0) { $0 + ($1 ? 1 : 0) }
        let stats = PLYWriter.CullStats(
            total: n, kept: kept,
            droppedNonFinite: nNonFinite,
            droppedLowOpacity: nLowOpac,
            droppedGiantScale: nGiant,
            droppedOutlierPosition: nOutlier,
            keptEnvironment: keptEnvironment
        )

        var tPos = [Float](), tScale = [Float](), tQuat = [Float](), tOpac = [Float]()
        var tSh0 = [Float](), tRest = [Float]()
        tPos.reserveCapacity(kept * 3); tScale.reserveCapacity(kept * 3)
        tQuat.reserveCapacity(kept * 4); tOpac.reserveCapacity(kept)
        tSh0.reserveCapacity(kept * 3); tRest.reserveCapacity(kept * bands * 3)
        for i in 0..<n where keep[i] {
            for k in 0..<3 {
                tPos.append(means[i * 3 + k])
                tScale.append(logScales[i * 3 + k])
                tSh0.append(sh0[i * 3 + k])
            }
            tOpac.append(logitOpac[i])
            let qw = quats[i * 4 + 0], qx = quats[i * 4 + 1]
            let qy = quats[i * 4 + 2], qz = quats[i * 4 + 3]
            let norm = (qw * qw + qx * qx + qy * qy + qz * qz).squareRoot()
            let inv = norm > 0 ? 1 / norm : 1
            tQuat += [qw * inv, qx * inv, qy * inv, qz * inv]
            // band-major [bands, 3] → channel-major
            for ch in 0..<3 {
                for band in 0..<bands { tRest.append(shN[i * bands * 3 + band * 3 + ch]) }
            }
        }
        let table = SplatTable(
            positions: tPos, logScales: tScale, quats: tQuat, logitOpacities: tOpac,
            sh0: tSh0, shRest: tRest, shDegree: GaussianModel.maxShDegree,
            backgroundCount: keptEnvironment
        )
        return (table, stats)
    }
}

// MARK: - From a PLY file

extension SplatTable {

    enum ReadError: Swift.Error, CustomStringConvertible {
        case parse(String)
        var description: String {
            switch self {
            case .parse(let s): return "PLY parse error: \(s)"
            }
        }
    }

    /// Reads a binary little-endian 3DGS `.ply` with all its SH bands (unlike ``PLYReader``, which
    /// keeps only the model's degree). The background count comes from the `<base>.background.json`
    /// sidecar when present. Quats are normalized.
    public static func readPLY(url: URL) throws -> SplatTable {
        let data = try Data(contentsOf: url)
        guard let r = data.range(of: Data("end_header\n".utf8)) else {
            throw ReadError.parse("no end_header")
        }
        let header = String(decoding: data[data.startIndex..<r.lowerBound], as: UTF8.self)
        guard header.contains("format binary_little_endian") else {
            throw ReadError.parse("only binary_little_endian is supported")
        }

        var vertexCount = 0
        var props: [String] = []
        var inVertex = false
        for line in header.split(separator: "\n") {
            let parts = line.split(separator: " ")
            if parts.count >= 3, parts[0] == "element" {
                inVertex = parts[1] == "vertex"
                if inVertex { vertexCount = Int(parts[2]) ?? 0 }
            } else if inVertex, parts.count >= 3, parts[0] == "property" {
                guard parts[1] == "float" || parts[1] == "float32" else {
                    throw ReadError.parse("non-float property \(parts.last.map(String.init) ?? "?")")
                }
                props.append(String(parts[2]))
            }
        }
        guard vertexCount > 0, !props.isEmpty else { throw ReadError.parse("empty/invalid header") }

        var index: [String: Int] = [:]
        for (i, p) in props.enumerated() { index[p] = i }
        func idx(_ name: String) throws -> Int {
            guard let i = index[name] else { throw ReadError.parse("missing property \(name)") }
            return i
        }
        let ipos = [try idx("x"), try idx("y"), try idx("z")]
        let idc = [try idx("f_dc_0"), try idx("f_dc_1"), try idx("f_dc_2")]
        let isc = [try idx("scale_0"), try idx("scale_1"), try idx("scale_2")]
        let irot = [try idx("rot_0"), try idx("rot_1"), try idx("rot_2"), try idx("rot_3")]
        let iop = try idx("opacity")

        var restTotal = 0
        while index["f_rest_\(restTotal)"] != nil { restTotal += 1 }
        guard let fileDegree = [0, 9, 24, 45].firstIndex(of: restTotal) else {
            throw ReadError.parse("unsupported f_rest count \(restTotal)")
        }
        let irest = try (0..<restTotal).map { try idx("f_rest_\($0)") }

        let stride = props.count
        let bodyStart = r.upperBound
        guard data.count >= bodyStart + vertexCount * stride * 4 else {
            throw ReadError.parse("truncated body")
        }
        let floats: [Float] = data[bodyStart...].withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float32.self)[0..<(vertexCount * stride)])
        }

        let n = vertexCount
        var pos = [Float](repeating: 0, count: n * 3)
        var scl = [Float](repeating: 0, count: n * 3)
        var dc = [Float](repeating: 0, count: n * 3)
        var quat = [Float](repeating: 0, count: n * 4)
        var opac = [Float](repeating: 0, count: n)
        var rest = [Float](repeating: 0, count: n * restTotal)
        for i in 0..<n {
            let b = i * stride
            for k in 0..<3 {
                pos[i * 3 + k] = floats[b + ipos[k]]
                scl[i * 3 + k] = floats[b + isc[k]]
                dc[i * 3 + k] = floats[b + idc[k]]
            }
            var q = irot.map { floats[b + $0] }
            let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).squareRoot()
            if norm > 0 { q = q.map { $0 / norm } }
            for k in 0..<4 { quat[i * 4 + k] = q[k] }
            opac[i] = floats[b + iop]
            for k in 0..<restTotal { rest[i * restTotal + k] = floats[b + irest[k]] }
        }

        let background = Self.readBackgroundSidecar(siblingOf: url).map { Swift.min($0, n) } ?? 0
        let table = SplatTable(
            positions: pos, logScales: scl, quats: quat, logitOpacities: opac,
            sh0: dc, shRest: rest, shDegree: fileDegree, backgroundCount: background
        )
        // PLYWriter zero-pads the model's degree to degree 3; drop bands that are zero in every row.
        return table.reducingSH(to: table.effectiveSHDegree)
    }

    /// The lowest degree whose dropped bands are all exactly zero.
    var effectiveSHDegree: Int {
        let rc = restCoeffs
        var degree = shDegree
        while degree > 0 {
            let lo = Self.restCoeffs(degree: degree - 1)
            var allZero = true
            rows: for i in 0..<count {
                for ch in 0..<3 {
                    for c in lo..<rc where shRest[(i * 3 + ch) * rc + c] != 0 { allZero = false; break rows }
                }
            }
            if !allZero { break }
            degree -= 1
        }
        return degree
    }

    /// `backgroundSplatCount` from the `<base>.background.json` next to `url`, or nil.
    static func readBackgroundSidecar(siblingOf url: URL) -> Int? {
        let metaURL = url.deletingPathExtension().appendingPathExtension("background.json")
        guard let data = try? Data(contentsOf: metaURL),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let count = obj["backgroundSplatCount"] as? Int, count > 0
        else { return nil }
        return count
    }
}
