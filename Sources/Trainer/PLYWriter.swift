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

    /// `yUpWorld`: the model is in a Y-up world (a capture, trained in ARKit's), so it is written
    /// turned 180° about Z to the usual 3DGS axes, with `PLYAxes.turnedComment` in the header
    /// (`PLYReader` turns it back).
    @discardableResult
    public static func writeGaussianPLY(
        model: GaussianModel,
        to url: URL,
        cull: CullOptions = CullOptions(),
        yUpWorld: Bool = false
    ) throws -> CullStats {
        // The breakdown is returned; logging is up to the caller.
        let (table, stats) = SplatTable.fromModel(model, cull: cull)
        if yUpWorld {
            try write(table.turnedAboutZ(), to: url, comments: [PLYAxes.turnedComment])
        } else {
            try write(table, to: url)
        }
        return stats
    }

    /// Writes `table` as a standard 3DGS PLY. f_rest is zero-padded to degree 3 whatever the
    /// table's degree, for MetalSplatter. `comments` are whole header lines (`comment …`).
    public static func write(_ table: SplatTable, to url: URL, comments: [String] = []) throws {
        let n = table.count
        let tableRest = table.restCoeffs

        var header = "ply\n"
        header += "format binary_little_endian 1.0\n"
        for comment in comments { header += comment + "\n" }
        header += "element vertex \(n)\n"
        for p in properties { header += "property float \(p)\n" }
        header += "end_header\n"

        var data = Data(header.utf8)
        data.reserveCapacity(header.utf8.count + n * properties.count * 4)

        let restOffset = 9
        let opacOffset = restOffset + exportRestBands * 3   // 9 + 45 = 54
        var rec = [Float](repeating: 0, count: properties.count)
        for i in 0..<n {
            rec[0] = table.positions[i * 3 + 0]
            rec[1] = table.positions[i * 3 + 1]
            rec[2] = table.positions[i * 3 + 2]
            rec[3] = 0; rec[4] = 0; rec[5] = 0  // normals (unused)
            rec[6] = table.sh0[i * 3 + 0]
            rec[7] = table.sh0[i * 3 + 1]
            rec[8] = table.sh0[i * 3 + 2]
            // f_rest is channel-major. Each channel puts the table's coefficients first, the rest
            // 0.
            for ch in 0..<3 {
                for band in 0..<exportRestBands {
                    let v: Float = band < tableRest
                        ? table.shRest[(i * 3 + ch) * tableRest + band]
                        : 0
                    rec[restOffset + ch * exportRestBands + band] = v
                }
            }
            rec[opacOffset] = table.logitOpacities[i]
            rec[opacOffset + 1] = table.logScales[i * 3 + 0]
            rec[opacOffset + 2] = table.logScales[i * 3 + 1]
            rec[opacOffset + 3] = table.logScales[i * 3 + 2]
            for k in 0..<4 { rec[opacOffset + 4 + k] = table.quats[i * 4 + k] }
            rec.withUnsafeBytes { data.append(contentsOf: $0) }
        }

        try data.write(to: url)
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
