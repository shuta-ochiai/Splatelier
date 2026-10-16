// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX

/// Writes a trained `GaussianModel` to a standard 3DGS binary little-endian `.ply` (read by
/// MetalSplatter and others). The model is stored in the same raw space as the PLY (opacity as
/// logit, scale_* as log), and f_dc = sh0 (RGB = 0.5 + C0·f_dc). f_rest is channel-major as in
/// gsplat (shN [N, 3, 3] is band-major, so it is transposed). Quats are normalized and written in
/// (w, x, y, z) order (so orientation is right even for readers that don't normalize).
public enum PLYWriter {

    /// Number of higher-order SH coefficients in the model (3, since degree 1).
    static let modelRestBands = GaussianModel.shRestBands

    /// Number of higher-order coefficients written to the PLY. MetalSplatter's reader accepts only
    /// 0 or 45 f_rest, so this is always degree-3's 15 (= 45 f_rest), zero-filled after the model's
    /// coefficients.
    static let exportRestBands = 15

    static let properties: [String] = {
        var p = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
        for i in 0..<(exportRestBands * 3) { p.append("f_rest_\(i)") }
        p += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        return p
    }()

    /// Cull thresholds before export. In addition to the gsplat exporter's opacity > 1/255, drops
    /// NaNs, huge discs and far outliers (left in, they drag the viewer's auto-fit to an extreme
    /// bbox).
    public struct CullOptions: Sendable, Equatable {
        /// Excluded if sigmoid(opacity) is at or below this (gsplat default 1/255).
        public var minOpacity: Float
        /// Excludes huge discs whose max scale is at or above this (scene units).
        public var maxScale: Float
        /// Excludes outliers whose absolute coordinate is at or above this (scene units).
        public var maxAbsCoord: Float
        /// false disables culling (for debugging).
        public var enabled: Bool

        public init(
            enabled: Bool = true,
            minOpacity: Float = 1.0 / 255.0,
            maxScale: Float = 3.0,
            maxAbsCoord: Float = 50.0
        ) {
            self.enabled = enabled
            self.minOpacity = minOpacity
            self.maxScale = maxScale
            self.maxAbsCoord = maxAbsCoord
        }
    }

    /// Breakdown of rows dropped at export.
    public struct CullStats: Sendable, Equatable {
        public var total: Int
        public var kept: Int
        public var droppedNonFinite: Int
        public var droppedLowOpacity: Int
        public var droppedGiantScale: Int
        public var droppedOutlierPosition: Int
        /// Environment rows written (they stay last in the PLY, since rows keep their order).
        public var keptEnvironment: Int
    }

    @discardableResult
    public static func writeGaussianPLY(
        model: GaussianModel,
        to url: URL,
        cull: CullOptions = CullOptions()
    ) throws -> CullStats {
        let n = model.count
        let means = model.means.asArray(Float.self)
        let sh0 = model.sh0.asArray(Float.self)            // [N*3]
        let shN = model.shN.asArray(Float.self)            // [N*modelRestBands*3], band-major
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
                    for k in 0..<(modelRestBands * 3)
                    where !shN[i * modelRestBands * 3 + k].isFinite { finite = false; break }
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

        let kept = cull.enabled ? keep.reduce(0) { $0 + ($1 ? 1 : 0) } : n
        let keptEnvironment = cull.enabled ? keep[fgN...].reduce(0) { $0 + ($1 ? 1 : 0) } : n - fgN
        let stats = CullStats(
            total: n, kept: kept,
            droppedNonFinite: nNonFinite,
            droppedLowOpacity: nLowOpac,
            droppedGiantScale: nGiant,
            droppedOutlierPosition: nOutlier,
            keptEnvironment: keptEnvironment
        )
        // The breakdown is returned; logging is up to the caller.

        var header = "ply\n"
        header += "format binary_little_endian 1.0\n"
        header += "element vertex \(kept)\n"
        for p in properties { header += "property float \(p)\n" }
        header += "end_header\n"

        var data = Data(header.utf8)
        data.reserveCapacity(header.utf8.count + kept * properties.count * 4)

        let restOffset = 9
        let opacOffset = restOffset + exportRestBands * 3   // 9 + 45 = 54
        var rec = [Float](repeating: 0, count: properties.count)
        for i in 0..<n where keep[i] {
            rec[0] = means[i * 3 + 0]
            rec[1] = means[i * 3 + 1]
            rec[2] = means[i * 3 + 2]
            rec[3] = 0; rec[4] = 0; rec[5] = 0  // normals (unused)
            rec[6] = sh0[i * 3 + 0]
            rec[7] = sh0[i * 3 + 1]
            rec[8] = sh0[i * 3 + 2]
            // f_rest is channel-major. Each channel puts the model's coefficients first, the rest
            // 0.
            for ch in 0..<3 {
                for band in 0..<exportRestBands {
                    let v: Float = band < modelRestBands
                        ? shN[i * modelRestBands * 3 + band * 3 + ch]
                        : 0
                    rec[restOffset + ch * exportRestBands + band] = v
                }
            }
            rec[opacOffset] = logitOpac[i]
            rec[opacOffset + 1] = logScales[i * 3 + 0]
            rec[opacOffset + 2] = logScales[i * 3 + 1]
            rec[opacOffset + 3] = logScales[i * 3 + 2]
            let qw = quats[i * 4 + 0], qx = quats[i * 4 + 1]
            let qy = quats[i * 4 + 2], qz = quats[i * 4 + 3]
            let norm = (qw * qw + qx * qx + qy * qy + qz * qz).squareRoot()
            let inv = norm > 0 ? 1 / norm : 1
            rec[opacOffset + 4] = qw * inv
            rec[opacOffset + 5] = qx * inv
            rec[opacOffset + 6] = qy * inv
            rec[opacOffset + 7] = qz * inv
            rec.withUnsafeBytes { data.append(contentsOf: $0) }
        }

        try data.write(to: url)
        return stats
    }

    /// Writes `<base>.background.json` next to a PLY: the number of rows at the end of the PLY that
    /// are the background (the bg-SH baked into an environment sphere). The viewer shows
    /// them as a layer it can hide. With 0 the file is removed, so rewriting a PLY never leaves a
    /// stale count behind.
    public static func writeBackgroundSidecar(splatCount: Int, siblingOf url: URL) throws {
        let metaURL = url.deletingPathExtension().appendingPathExtension("background.json")
        guard splatCount > 0 else {
            if FileManager.default.fileExists(atPath: metaURL.path) {
                try FileManager.default.removeItem(at: metaURL)
            }
            return
        }
        let json = "{\"version\":1,\"backgroundSplatCount\":\(splatCount)}"
        try Data(json.utf8).write(to: metaURL, options: .atomic)
    }
}
