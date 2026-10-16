// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import Testing
@testable import Trainer

/// Minimal binary PLY parser for round-trip checks. Returns the header property names and float
/// records.
private func parseBinaryPLY(_ data: Data) -> (props: [String], count: Int, records: [[Float]])? {
    guard let headerEnd = data.range(of: Data("end_header\n".utf8)) else { return nil }
    let headerData = data.subdata(in: 0..<headerEnd.upperBound)
    guard let header = String(data: headerData, encoding: .utf8) else { return nil }

    var props: [String] = []
    var count = 0
    for line in header.split(separator: "\n") {
        if line.hasPrefix("element vertex ") {
            count = Int(line.dropFirst("element vertex ".count)) ?? 0
        } else if line.hasPrefix("property float ") {
            props.append(String(line.dropFirst("property float ".count)))
        }
    }
    let stride = props.count
    var records: [[Float]] = []
    records.reserveCapacity(count)
    var offset = headerEnd.upperBound
    let bytes = [UInt8](data)
    for _ in 0..<count {
        var rec = [Float](repeating: 0, count: stride)
        for j in 0..<stride {
            var bits: UInt32 = 0
            for b in 0..<4 { bits |= UInt32(bytes[offset + j * 4 + b]) << (8 * b) }
            rec[j] = Float(bitPattern: bits)
        }
        records.append(rec)
        offset += stride * 4
    }
    return (props, count, records)
}

@Test
func plyWriterRoundTrips() throws {
    let n = 5
    let model = GaussianModel(
        means: MLXArray(
            [Float(0.1), 0.2, 4.0,
             -0.3, 0.5, 3.5,
             0.0, -0.4, 5.0,
             0.2, 0.2, 4.2,
             -0.1, 0.0, 3.9],
            [n, 3]),
        quats: MLXArray(
            [Float(1), 0, 0, 0,
             0.92388, 0, 0, 0.38268,   // 45° about z (nearly unit)
             0.7071, 0.7071, 0, 0,
             1, 0.1, 0.2, 0.3,         // deliberately unnormalized
             0.5, 0.5, 0.5, 0.5],
            [n, 4]),
        logScales: MLXArray(
            [Float](repeating: -1.5, count: n * 3), [n, 3]),
        logitOpacities: MLXArray([logit(0.2), logit(0.5), logit(0.9), logit(0.7), logit(0.3)], [n]),
        sh0: MLXArray(
            [Float(0.9), 0.1, 0.2,
             0.0, 0.8, 0.4,
             0.3, 0.3, 1.0,
             0.5, 0.5, 0.5,
             0.7, 0.2, 0.6],
            [n, 3]),
        // All distinct non-zero values so the channel-major order of f_rest can be checked.
        shN: MLXArray(
            (0..<(n * GaussianModel.shRestBands * 3)).map { Float($0) * 0.013 - 1.0 }, [n, GaussianModel.shRestBands, 3])
    )
    eval(model.means, model.quats, model.logScales, model.logitOpacities, model.sh0, model.shN)

    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("gs_\(UUID().uuidString).ply")
    defer { try? FileManager.default.removeItem(at: url) }
    try PLYWriter.writeGaussianPLY(model: model, to: url)

    let data = try Data(contentsOf: url)
    guard let (props, count, recs) = parseBinaryPLY(data) else {
        Issue.record("failed to parse written PLY")
        return
    }
    #expect(props == PLYWriter.properties)
    #expect(count == n)

    let means = model.means.asArray(Float.self)
    let sh0 = model.sh0.asArray(Float.self)
    let shN = model.shN.asArray(Float.self)
    let logScales = model.logScales.asArray(Float.self)
    let logitOpac = model.logitOpacities.asArray(Float.self)
    let quats = model.quats.asArray(Float.self)

    // The PLY uses the degree-3 layout for viewer compatibility (45 f_rest, channel stride 15). The
    // model's degree-1 coefficients (band < modelRestBands) are real values; the rest is 0 padding.
    let modelRestBands = GaussianModel.shRestBands
    let exportRestBands = PLYWriter.exportRestBands   // 15
    let restOff = 9
    let opOff = restOff + exportRestBands * 3   // 9 + 45 = 54 (degree-3 layout)
    for i in 0..<n {
        let r = recs[i]
        // Positions unchanged.
        #expect(abs(r[0] - means[i * 3 + 0]) < 1e-6)
        #expect(abs(r[2] - means[i * 3 + 2]) < 1e-6)
        // f_dc = sh0 as is.
        for c in 0..<3 {
            #expect(abs(r[6 + c] - sh0[i * 3 + c]) < 1e-5, "sh0 mismatch at \(i),\(c)")
        }
        // f_rest is channel-major: r[9 + ch*exportRestBands + band].
        for ch in 0..<3 {
            for band in 0..<exportRestBands {
                let got = r[restOff + ch * exportRestBands + band]
                let want = band < modelRestBands ? shN[i * modelRestBands * 3 + band * 3 + ch] : 0
                #expect(abs(got - want) < 1e-5, "f_rest mismatch at \(i),ch\(ch),band\(band)")
            }
        }
        // opacity / scale are raw values.
        #expect(abs(r[opOff] - logitOpac[i]) < 1e-6)
        #expect(abs(r[opOff + 1] - logScales[i * 3 + 0]) < 1e-6)
        // Rotation is the normalized model quat.
        let qw = quats[i * 4 + 0], qx = quats[i * 4 + 1]
        let qy = quats[i * 4 + 2], qz = quats[i * 4 + 3]
        let inv = 1 / (qw * qw + qx * qx + qy * qy + qz * qz).squareRoot()
        #expect(abs(r[opOff + 4] - qw * inv) < 1e-5)
        #expect(abs(r[opOff + 5] - qx * inv) < 1e-5)
        #expect(abs(r[opOff + 6] - qy * inv) < 1e-5)
        #expect(abs(r[opOff + 7] - qz * inv) < 1e-5)
        // normal is 0.
        #expect(r[3] == 0 && r[4] == 0 && r[5] == 0)
    }
}

// MARK: - Pre-export cull

/// Splats that each trip one of the 4 default `CullOptions` conditions (non-finite means /
/// sigmoid(logit) ≤ 1/255 / max exp(logScale) ≥ 3.0 / |mean| ≥ 50.0) are dropped and only the 1
/// clean one remains. Giant discs and stray positions break SuperSplat's auto-fit (equivalent to
/// gsplat exporter.py).
@Test
func plyWriterCullsOutliersAndLowOpacity() throws {
    let n = 5
    let model = GaussianModel(
        means: MLXArray(
            [Float.nan,  0.0, 0.0,    // splat 0: NaN means → culled as non-finite
             0.0,        0.0, 0.0,    // splat 1: low opacity
             1.0,        1.0, 1.0,    // splat 2: giant scale
             1000.0,    -2.0, 5.0,    // splat 3: outside bbox
             0.5,        0.5, 0.5],   // splat 4: clean → kept
            [n, 3]),
        quats: MLXArray(
            [Float(1), 0, 0, 0,
             1, 0, 0, 0,
             1, 0, 0, 0,
             1, 0, 0, 0,
             1, 0, 0, 0],
            [n, 4]),
        logScales: MLXArray(
            [Float(-1.5), -1.5, -1.5,
             -1.5,        -1.5, -1.5,
             logf(5.0),   logf(5.0), logf(5.0),              // splat 2: exp = 5.0 ≥ 3.0
             -1.5,        -1.5, -1.5,
             -1.5,        -1.5, -1.5],
            [n, 3]),
        logitOpacities: MLXArray(
            [logit(0.5),
             logit(0.001),        // splat 1: sigmoid = 0.001 < 1/255
             logit(0.5),
             logit(0.5),
             logit(0.5)],
            [n]),
        sh0: MLXArray((0..<(n * 3)).map { Float($0) * 0.1 }, [n, 3]),
        shN: MLXArray((0..<(n * GaussianModel.shRestBands * 3)).map { Float($0) * 0.01 }, [n, GaussianModel.shRestBands, 3])
    )
    eval(model.means, model.quats, model.logScales, model.logitOpacities, model.sh0, model.shN)

    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("gs_cull_\(UUID().uuidString).ply")
    defer { try? FileManager.default.removeItem(at: url) }
    let stats = try PLYWriter.writeGaussianPLY(model: model, to: url)

    #expect(stats.total == 5)
    #expect(stats.kept == 1, "expected 1 kept (splat 4), got \(stats.kept)")
    #expect(stats.droppedNonFinite == 1, "expected 1 NaN drop, got \(stats.droppedNonFinite)")
    #expect(stats.droppedLowOpacity == 1, "expected 1 low-opac drop, got \(stats.droppedLowOpacity)")
    #expect(stats.droppedGiantScale == 1, "expected 1 giant drop, got \(stats.droppedGiantScale)")
    #expect(stats.droppedOutlierPosition == 1, "expected 1 outlier drop, got \(stats.droppedOutlierPosition)")

    let data = try Data(contentsOf: url)
    guard let (_, count, recs) = parseBinaryPLY(data) else {
        Issue.record("failed to parse culled PLY")
        return
    }
    #expect(count == 1)
    #expect(recs.count == 1)
    // The survivor is splat 4 = means (0.5, 0.5, 0.5)
    #expect(abs(recs[0][0] - 0.5) < 1e-5)
    #expect(abs(recs[0][1] - 0.5) < 1e-5)
    #expect(abs(recs[0][2] - 0.5) < 1e-5)
}

/// The environment sphere (trailing environmentCount splats) is exempt from the bbox / giant scale
/// / low opacity cull; only the NaN guard applies. The fixed thresholds (maxAbsCoord=50,
/// maxScale=3) do not follow scene_scale and would cut the intentional distant shell
/// (R=24×scene_scale) as outliers. fg is culled as usual.
@Test
func plyWriterExemptsEnvironmentSphereFromOutlierCull() throws {
    let n = 5
    // fgN = n - environmentCount(3) = 2 → splats 0,1 are fg / splats 2,3,4 are env.
    let model = GaussianModel(
        means: MLXArray(
            [0.5,       0.5, 0.5,    // splat 0: fg clean → kept
             1000.0,   -2.0, 5.0,    // splat 1: fg outside bbox → dropped
             1000.0,    1.0, 1.0,    // splat 2: env outside bbox → kept (exempt)
             0.0,       0.0, 0.0,    // splat 3: env giant scale → kept (exempt)
             Float.nan, 0.0, 0.0],   // splat 4: env NaN → dropped (NaN guard applies)
            [n, 3]),
        quats: MLXArray([Float](repeating: 0, count: n * 4).enumerated().map {
            $0.offset % 4 == 0 ? 1 : 0
        }, [n, 4]),
        logScales: MLXArray(
            [Float(-1.5), -1.5, -1.5,
             -1.5,        -1.5, -1.5,
             -1.5,        -1.5, -1.5,
             logf(5.0),   logf(5.0), logf(5.0),   // splat 3 (env): exp=5 ≥ 3 but exempt
             -1.5,        -1.5, -1.5],
            [n, 3]),
        logitOpacities: MLXArray([Float](repeating: logit(0.5), count: n), [n]),
        sh0: MLXArray((0..<(n * 3)).map { Float($0) * 0.1 }, [n, 3]),
        shN: MLXArray((0..<(n * GaussianModel.shRestBands * 3)).map { Float($0) * 0.01 }, [n, GaussianModel.shRestBands, 3])
    )
    model.environmentCount = 3
    eval(model.means, model.quats, model.logScales, model.logitOpacities, model.sh0, model.shN)

    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("gs_env_exempt_\(UUID().uuidString).ply")
    defer { try? FileManager.default.removeItem(at: url) }
    let stats = try PLYWriter.writeGaussianPLY(model: model, to: url)

    #expect(stats.total == 5)
    // kept: splat 0 (fg clean) + splat 2 (env bbox) + splat 3 (env giant) = 3
    #expect(stats.kept == 3, "expected 3 kept (fg clean + 2 env-exempt), got \(stats.kept)")
    #expect(stats.droppedOutlierPosition == 1, "only the fg bbox outlier drops, got \(stats.droppedOutlierPosition)")
    #expect(stats.droppedGiantScale == 0, "env giant scale must be exempt, got \(stats.droppedGiantScale)")
    #expect(stats.droppedNonFinite == 1, "NaN env still drops, got \(stats.droppedNonFinite)")
    #expect(stats.keptEnvironment == 2, "env rows written = 3 - NaN, got \(stats.keptEnvironment)")
}

/// The background sidecar names the trailing background rows, and a count of 0 removes it.
@Test
func plyWriterBackgroundSidecar() throws {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("gs_bg_\(UUID().uuidString).ply")
    let metaURL = url.deletingPathExtension().appendingPathExtension("background.json")
    defer { try? FileManager.default.removeItem(at: metaURL) }

    try PLYWriter.writeBackgroundSidecar(splatCount: 6000, siblingOf: url)
    let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: metaURL)) as? [String: Any]
    #expect(meta?["version"] as? Int == 1)
    #expect(meta?["backgroundSplatCount"] as? Int == 6000)

    try PLYWriter.writeBackgroundSidecar(splatCount: 0, siblingOf: url)
    #expect(!FileManager.default.fileExists(atPath: metaURL.path))
}

/// `CullOptions(enabled: false)` writes every splat (raw export path for debugging).
@Test
func plyWriterCullDisabledKeepsEverything() throws {
    let n = 3
    let model = GaussianModel(
        means: MLXArray([Float(10000), 0, 0,
                         0, 0, 0,
                         0, 0, 0], [n, 3]),
        quats: MLXArray([Float(1), 0, 0, 0,
                         1, 0, 0, 0,
                         1, 0, 0, 0], [n, 4]),
        logScales: MLXArray([Float(-1.5), -1.5, -1.5,
                             -1.5, -1.5, -1.5,
                             -1.5, -1.5, -1.5], [n, 3]),
        logitOpacities: MLXArray([logit(0.001), logit(0.5), logit(0.5)], [n]),
        sh0: MLXArray((0..<(n * 3)).map { _ in Float(0) }, [n, 3]),
        shN: MLXArray((0..<(n * GaussianModel.shRestBands * 3)).map { _ in Float(0) }, [n, GaussianModel.shRestBands, 3])
    )
    eval(model.means, model.quats, model.logScales, model.logitOpacities, model.sh0, model.shN)

    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("gs_raw_\(UUID().uuidString).ply")
    defer { try? FileManager.default.removeItem(at: url) }
    let stats = try PLYWriter.writeGaussianPLY(
        model: model, to: url, cull: .init(enabled: false)
    )
    #expect(stats.total == 3 && stats.kept == 3)
}
