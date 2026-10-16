// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import Testing
@testable import Trainer

/// Integration test that loads a real COLMAP reconstruction. Skipped unless `MST_COLMAP_DIR` (a
/// scene dir with `sparse/0/*.bin` and `images/`) is set. Run:
/// `TEST_RUNNER_MST_COLMAP_DIR=$PWD/data/bear make test`
@Test
func colmapRealDatasetLoads() throws {
    guard let dirString = ProcessInfo.processInfo.environment["MST_COLMAP_DIR"] else {
        return
    }
    let dir = URL(fileURLWithPath: dirString)

    let scene = try ColmapLoader.load(directory: dir)

    #expect(!scene.cameras.isEmpty, "no cameras parsed")
    let n = scene.pointPositions.shape[0]
    #expect(n > 0, "empty point cloud")
    #expect(scene.pointColors.shape == [n, 3])

    // Focal lengths are positive and the principal point is inside the frame.
    for cam in scene.cameras.prefix(5) {
        #expect(cam.fx > 0 && cam.fy > 0, "non-positive focal for \(cam.imageName)")
        #expect(cam.cx > 0 && cam.cx < Float(cam.width), "cx out of frame for \(cam.imageName)")
        #expect(cam.cy > 0 && cam.cy < Float(cam.height), "cy out of frame for \(cam.imageName)")
        #expect(cam.viewmat[15] == 1)
    }

    // Point cloud colors are normalized to [0, 1].
    let colors = scene.pointColors.asArray(Float.self)
    let maxC = colors.max() ?? 0
    let minC = colors.min() ?? 0
    #expect(minC >= 0 && maxC <= 1, "colors out of [0,1]: [\(minC), \(maxC)]")

    // The first camera's image size matches the camera's width/height.
    let imagesDir = dir.appendingPathComponent("images")
    if let first = scene.cameras.first {
        let imgURL = imagesDir.appendingPathComponent(first.imageName)
        if FileManager.default.fileExists(atPath: imgURL.path) {
            let img = try ImageLoader.loadRGB(url: imgURL)
            #expect(img.shape.count == 3 && img.shape[2] == 3)
            #expect(img.shape[0] == first.height, "image H \(img.shape[0]) != camera H \(first.height)")
            #expect(img.shape[1] == first.width, "image W \(img.shape[1]) != camera W \(first.width)")
        }
    }

    print("COLMAP real dataset: \(scene.cameras.count) cameras, \(n) points, "
        + "first cam \(scene.cameras.first?.imageName ?? "?") "
        + "\(scene.cameras.first?.width ?? 0)x\(scene.cameras.first?.height ?? 0) "
        + "fx=\(scene.cameras.first?.fx ?? 0)")
}
