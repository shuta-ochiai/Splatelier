// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import Testing
@testable import Trainer

private func plyVertexCount(fromHeader header: String) -> Int? {
    for line in header.split(separator: "\n") where line.hasPrefix("element vertex ") {
        return Int(line.dropFirst("element vertex ".count))
    }
    return nil
}

/// End-to-end integration test on COLMAP bear. Skipped unless `MST_COLMAP_DIR` is set. Both run
/// full training, so running them concurrently overflows GPU memory and Metal aborts →
/// `.serialized`. Run: `TEST_RUNNER_MST_COLMAP_DIR=$PWD/data/bear make test`
@Suite(.serialized)
struct ColmapBearIntegrationTests {

/// A short multi-view L1+SSIM training run lowers the mean loss.
@Test
func colmapTrainingReducesLoss() throws {
    guard let dirString = ProcessInfo.processInfo.environment["MST_COLMAP_DIR"] else {
        return
    }
    let dir = URL(fileURLWithPath: dirString)
    let plyURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("bear_trained_\(UUID().uuidString).ply")
    defer { try? FileManager.default.removeItem(at: plyURL) }

    let result = try TrainingDriver.trainColmap(
        sceneDirectory: dir,
        iterations: 80,
        maxImageDimension: 192,    // small to keep the test fast
        maxPoints: 8_000,
        meansLR: 1.6e-4,
        logEvery: 20,
        outputPLY: plyURL
    )

    #expect(result.gaussianCount > 0, "no gaussians initialised")
    #expect(result.viewCount > 0, "no training views built")
    #expect(result.lossHistory.count == 80)

    // A PLY with a binary 3DGS header is written. Read 512 bytes, which holds the whole header.
    #expect(FileManager.default.fileExists(atPath: plyURL.path), "no PLY written")
    let plyHead = try Data(contentsOf: plyURL).prefix(512)
    let headStr = String(decoding: plyHead, as: UTF8.self)
    #expect(headStr.hasPrefix("ply\nformat binary_little_endian"), "unexpected PLY header")
    #expect(headStr.contains("f_dc_0"), "PLY missing SH dc property")
    let writtenCount = try #require(plyVertexCount(fromHeader: headStr), "PLY missing vertex count")
    #expect(writtenCount > 0, "PLY wrote no vertices")
    #expect(writtenCount <= result.gaussianCount,
            "PLY vertex count must not exceed model count: \(writtenCount) > \(result.gaussianCount)")

    // Per-iter loss is noisy due to random views, so compare means over the first/last window.
    let history = result.lossHistory
    let window = 15
    let firstAvg = history.prefix(window).reduce(0, +) / Float(window)
    let lastAvg = history.suffix(window).reduce(0, +) / Float(window)
    print("colmap training: first-\(window) avg \(firstAvg), last-\(window) avg \(lastAvg) "
        + "(\(result.gaussianCount) gaussians, \(result.viewCount) views)")
    #expect(lastAvg < firstAvg, "loss did not decrease: \(firstAvg) -> \(lastAvg)")
}

/// MCMC densification: starting from a subsampled point cloud, the Gaussian count grows past the
/// initial cap.
@Test
func colmapTrainingWithMCMCGrowsGaussians() throws {
    guard let dirString = ProcessInfo.processInfo.environment["MST_COLMAP_DIR"] else {
        return
    }
    let dir = URL(fileURLWithPath: dirString)
    let plyURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("bear_mcmc_\(UUID().uuidString).ply")
    defer { try? FileManager.default.removeItem(at: plyURL) }

    let initialCap = 4_000
    let strategy = MCMCStrategy(
        capMax: 8_000, noiseLR: 5e5,
        refineStartIter: 10, refineStopIter: 80, refineEvery: 20,
        minOpacity: 0.005, growthFactor: 1.1, seed: 0
    )
    let result = try TrainingDriver.trainColmap(
        sceneDirectory: dir,
        iterations: 100,
        maxImageDimension: 192,
        maxPoints: initialCap,
        meansLR: 1.6e-4,
        logEvery: 20,
        strategy: strategy,
        opacityReg: 0.01,
        scaleReg: 0.01,
        outputPLY: plyURL
    )

    #expect(result.gaussianCount > initialCap,
            "MCMC should grow beyond the initial \(initialCap), got \(result.gaussianCount)")
    #expect(FileManager.default.fileExists(atPath: plyURL.path), "no PLY written")
}

}   // ColmapBearIntegrationTests
