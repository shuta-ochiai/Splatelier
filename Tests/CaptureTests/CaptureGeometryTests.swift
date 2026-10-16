// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import Trainer
import Testing
@testable import Capture

@Test
func geometryKeepsDepthMapsOnDiskExactly() throws {
    // The MVS depth maps are written to a temporary file as they come (never all resident), read
    // back as they were for meshing, and the file goes with the geometry.
    var views: [CostVolumeMVS.DepthView] = []
    for i in 0..<3 {
        let offset = Float(i)
        let depth: [Float] = (0..<20).map { Float($0) * 0.1 + offset - 0.5 }
        let rgb: [UInt8] = (0..<60).map { UInt8(($0 * 7 + i) % 256) }
        let viewmat: [Float] = (0..<16).map { Float($0) * 0.25 + offset }
        views.append(CostVolumeMVS.DepthView(
            width: 5, height: 4, depth: depth, rgb: rgb, viewmat: viewmat,
            intrinsics: (fx: 1.5, fy: 2.5, cx: 3.5, cy: 4.5 + offset)))
    }
    var file: DepthMapFile? = try DepthMapFile()
    for v in views { file!.append(v) }
    try file!.finishWriting()
    let url = file!.url
    let empty = CapturedScene(frames: [], sparsePoints: [])
    var geometry: CaptureGeometry? = CaptureGeometry(
        capturedFrameCount: 0, scene: empty, weightByTimestamp: nil, sceneScale: 1, mvsVoxelSize: nil,
        seedCloud: nil, depthMaps: file)
    file = nil

    let back = try geometry!.depthViews()
    #expect(back.count == views.count)
    for (a, b) in zip(views, back) {
        #expect(a.width == b.width)
        #expect(a.height == b.height)
        #expect(a.depth == b.depth)
        #expect(a.rgb == b.rgb)
        #expect(a.viewmat == b.viewmat)
        let ia = [a.intrinsics.fx, a.intrinsics.fy, a.intrinsics.cx, a.intrinsics.cy]
        let ib = [b.intrinsics.fx, b.intrinsics.fy, b.intrinsics.cx, b.intrinsics.cy]
        #expect(ia == ib)
    }

    geometry = nil
    #expect(!FileManager.default.fileExists(atPath: url.path))
}
