// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import Testing
@testable import Trainer

private func makeCamerasBin() -> Data {
    var w = BinaryWriter()
    w.writeUInt64(1)            // num_cameras
    w.writeUInt32(1)           // camera_id
    w.writeInt32(1)            // model_id = PINHOLE
    w.writeUInt64(640)         // width
    w.writeUInt64(480)         // height
    w.writeFloat64(500.0)      // fx
    w.writeFloat64(510.0)      // fy
    w.writeFloat64(320.0)      // cx
    w.writeFloat64(240.0)      // cy
    return w.data
}

private func makeImagesBin() -> Data {
    var w = BinaryWriter()
    w.writeUInt64(2)            // num_images

    // Image 1: identity rotation, translation (0, 0, 5)
    w.writeUInt32(1)
    w.writeFloat64(1.0); w.writeFloat64(0.0); w.writeFloat64(0.0); w.writeFloat64(0.0)  // qvec
    w.writeFloat64(0.0); w.writeFloat64(0.0); w.writeFloat64(5.0)                       // tvec
    w.writeUInt32(1)            // camera_id
    w.writeCString("img_001.png")
    w.writeUInt64(2)            // num_points2D
    w.writeFloat64(100.0); w.writeFloat64(200.0); w.writeUInt64(1)
    w.writeFloat64(150.0); w.writeFloat64(250.0); w.writeUInt64(2)

    // Image 2: 90° rotation about z (qw=cos45, qz=sin45), translation (1, 2, 6)
    let c = (Double.pi / 4).magnitude
    let qw = Foundation.cos(c), qz = Foundation.sin(c)
    w.writeUInt32(2)
    w.writeFloat64(qw); w.writeFloat64(0.0); w.writeFloat64(0.0); w.writeFloat64(qz)
    w.writeFloat64(1.0); w.writeFloat64(2.0); w.writeFloat64(6.0)
    w.writeUInt32(1)
    w.writeCString("img_000.png")   // sorts before img_001 (checks ordering)
    w.writeUInt64(0)            // num_points2D

    return w.data
}

private func makePoints3DBin() -> Data {
    var w = BinaryWriter()
    w.writeUInt64(3)           // num_points

    // Point 1
    w.writeUInt64(1)
    w.writeFloat64(0.5); w.writeFloat64(-0.5); w.writeFloat64(2.0)
    w.writeUInt8(255); w.writeUInt8(0); w.writeUInt8(0)
    w.writeFloat64(0.3)        // error
    w.writeUInt64(1)           // track length
    w.writeUInt32(1); w.writeUInt32(0)

    // Point 2
    w.writeUInt64(2)
    w.writeFloat64(-1.0); w.writeFloat64(1.0); w.writeFloat64(3.0)
    w.writeUInt8(0); w.writeUInt8(255); w.writeUInt8(0)
    w.writeFloat64(0.2)
    w.writeUInt64(0)           // track length

    // Point 3
    w.writeUInt64(3)
    w.writeFloat64(0.0); w.writeFloat64(0.0); w.writeFloat64(4.0)
    w.writeUInt8(0); w.writeUInt8(0); w.writeUInt8(128)
    w.writeFloat64(0.1)
    w.writeUInt64(2)
    w.writeUInt32(1); w.writeUInt32(1)
    w.writeUInt32(2); w.writeUInt32(0)

    return w.data
}

private func writeFixture() throws -> URL {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("colmap_test_\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    try makeCamerasBin().write(to: dir.appendingPathComponent("cameras.bin"))
    try makeImagesBin().write(to: dir.appendingPathComponent("images.bin"))
    try makePoints3DBin().write(to: dir.appendingPathComponent("points3D.bin"))
    return dir
}

@Test
func colmapLoaderParsesCamerasImagesPoints() throws {
    let dir = try writeFixture()
    defer { try? FileManager.default.removeItem(at: dir) }

    let scene = try ColmapLoader.load(directory: dir)

    // Sorted by image name.
    #expect(scene.cameras.count == 2)
    #expect(scene.cameras[0].imageName == "img_000.png")
    #expect(scene.cameras[1].imageName == "img_001.png")

    // img_001: identity rotation + tvec (0,0,5), PINHOLE.
    let cam1 = scene.cameras[1]
    #expect(cam1.fx == 500.0)
    #expect(cam1.fy == 510.0)
    #expect(cam1.cx == 320.0)
    #expect(cam1.cy == 240.0)
    #expect(cam1.width == 640)
    #expect(cam1.height == 480)
    // rotation block = I
    #expect(abs(cam1.viewmat[0] - 1) < 1e-6)
    #expect(abs(cam1.viewmat[5] - 1) < 1e-6)
    #expect(abs(cam1.viewmat[10] - 1) < 1e-6)
    // translation column
    #expect(abs(cam1.viewmat[3] - 0) < 1e-6)
    #expect(abs(cam1.viewmat[7] - 0) < 1e-6)
    #expect(abs(cam1.viewmat[11] - 5) < 1e-6)
    // bottom row
    #expect(cam1.viewmat[12] == 0 && cam1.viewmat[13] == 0 && cam1.viewmat[14] == 0)
    #expect(cam1.viewmat[15] == 1)

    // img_000: qvec=(cos45,0,0,sin45) → R = [[0,-1,0],[1,0,0],[0,0,1]], tvec = (1,2,6).
    let cam0 = scene.cameras[0]
    #expect(abs(cam0.viewmat[0] - 0) < 1e-5)   // r00
    #expect(abs(cam0.viewmat[1] - (-1)) < 1e-5) // r01
    #expect(abs(cam0.viewmat[4] - 1) < 1e-5)   // r10
    #expect(abs(cam0.viewmat[5] - 0) < 1e-5)   // r11
    #expect(abs(cam0.viewmat[10] - 1) < 1e-5)  // r22
    #expect(abs(cam0.viewmat[3] - 1) < 1e-6)   // tx
    #expect(abs(cam0.viewmat[7] - 2) < 1e-6)   // ty
    #expect(abs(cam0.viewmat[11] - 6) < 1e-6)  // tz

    // point cloud
    #expect(scene.pointPositions.shape == [3, 3])
    #expect(scene.pointColors.shape == [3, 3])
    let pos = scene.pointPositions.asArray(Float.self)
    #expect(abs(pos[0] - 0.5) < 1e-6)
    #expect(abs(pos[1] - (-0.5)) < 1e-6)
    #expect(abs(pos[2] - 2.0) < 1e-6)
    let col = scene.pointColors.asArray(Float.self)
    #expect(abs(col[0] - 1.0) < 1e-6)       // r = 255/255
    #expect(abs(col[1] - 0.0) < 1e-6)
    #expect(abs(col[3 + 1] - 1.0) < 1e-6)   // g of point 2 = 255/255
    #expect(abs(col[6 + 2] - 128.0 / 255.0) < 1e-6)  // b of point 3 = 128/255
}

@Test
func colmapLoaderFindsSparseSubdirectory() throws {
    // Finds the bins under <dir>/sparse/0/ automatically.
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("colmap_sparse_\(UUID().uuidString)")
    let model = root.appendingPathComponent("sparse/0")
    try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try makeCamerasBin().write(to: model.appendingPathComponent("cameras.bin"))
    try makeImagesBin().write(to: model.appendingPathComponent("images.bin"))
    try makePoints3DBin().write(to: model.appendingPathComponent("points3D.bin"))

    let scene = try ColmapLoader.load(directory: root)
    #expect(scene.cameras.count == 2)
    #expect(scene.pointPositions.shape == [3, 3])
}

@Test
func colmapLoaderThrowsOnMissingFiles() {
    let empty = FileManager.default.temporaryDirectory
        .appendingPathComponent("colmap_empty_\(UUID().uuidString)")
    try? FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: empty) }

    #expect(throws: ColmapLoader.Error.self) {
        _ = try ColmapLoader.load(directory: empty)
    }
}
