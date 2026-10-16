// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX

/// One COLMAP view (world-to-camera transform + pinhole intrinsics + image name).
public struct ColmapCamera: Sendable {
    public let imageName: String
    /// Row-major 4x4 world-to-camera (compatible with `Camera.viewmat`).
    public let viewmat: [Float]
    public let fx: Float
    public let fy: Float
    public let cx: Float
    public let cy: Float
    public let width: Int
    public let height: Int
}

/// A COLMAP reconstruction (cameras sorted by image name + sparse point cloud for Gaussian init).
/// Not Sendable because it holds MLXArrays.
public struct ColmapScene {
    public let cameras: [ColmapCamera]
    public let pointPositions: MLXArray   // [N, 3] Float32, world coordinates
    public let pointColors: MLXArray      // [N, 3] Float32 in [0, 1]
}

public enum ColmapLoader {

    public enum Error: Swift.Error, CustomStringConvertible {
        case missingFile(String)
        case unsupportedCameraModel(Int32)
        case parse(String)

        public var description: String {
            switch self {
            case .missingFile(let f): return "COLMAP file not found: \(f)"
            case .unsupportedCameraModel(let id): return "Unsupported COLMAP camera model id \(id)"
            case .parse(let m): return "COLMAP parse error: \(m)"
            }
        }
    }

    // MARK: - Entry point

    /// Looks for `*.bin` in `directory` → `directory/sparse/0` → `directory/sparse`, in that order,
    /// and reads it.
    public static func load(directory: URL) throws -> ColmapScene {
        let base = try locateModelDirectory(directory)

        let camerasData = try readFile(base, "cameras.bin")
        let imagesData = try readFile(base, "images.bin")
        let pointsData = try readFile(base, "points3D.bin")

        let intrinsics = try parseCameras(camerasData)
        let images = try parseImages(imagesData)
        let (positions, colors) = try parsePoints3D(pointsData)

        var cameras: [ColmapCamera] = []
        cameras.reserveCapacity(images.count)
        for img in images {
            guard let intr = intrinsics[img.cameraID] else {
                throw Error.parse("image \(img.name) references unknown camera id \(img.cameraID)")
            }
            let r = qvecToRotation(img.qw, img.qx, img.qy, img.qz)
            let viewmat: [Float] = [
                r[0], r[1], r[2], Float(img.tx),
                r[3], r[4], r[5], Float(img.ty),
                r[6], r[7], r[8], Float(img.tz),
                0, 0, 0, 1,
            ]
            let cam = ColmapCamera(
                imageName: img.name,
                viewmat: viewmat,
                fx: intr.fx, fy: intr.fy, cx: intr.cx, cy: intr.cy,
                width: intr.width, height: intr.height
            )
            cameras.append(cam)
        }
        cameras.sort { $0.imageName < $1.imageName }

        let n = positions.count / 3
        let positionsMLX = n > 0 ? MLXArray(positions, [n, 3]) : MLXArray.zeros([0, 3], dtype: .float32)
        let colorsMLX = n > 0 ? MLXArray(colors, [n, 3]) : MLXArray.zeros([0, 3], dtype: .float32)

        return ColmapScene(
            cameras: cameras,
            pointPositions: positionsMLX,
            pointColors: colorsMLX
        )
    }

    // MARK: - File lookup

    private static func locateModelDirectory(_ directory: URL) throws -> URL {
        let candidates = [
            directory,
            directory.appendingPathComponent("sparse/0"),
            directory.appendingPathComponent("sparse"),
        ]
        let fm = FileManager.default
        for c in candidates {
            let camerasPath = c.appendingPathComponent("cameras.bin").path
            if fm.fileExists(atPath: camerasPath) { return c }
        }
        throw Error.missingFile("cameras.bin (searched \(candidates.map { $0.path }))")
    }

    private static func readFile(_ base: URL, _ name: String) throws -> Data {
        let url = base.appendingPathComponent(name)
        guard let data = FileManager.default.contents(atPath: url.path) else {
            throw Error.missingFile(url.path)
        }
        return data
    }

    // MARK: - cameras.bin

    struct CameraParams {
        let fx: Float
        let fy: Float
        let cx: Float
        let cy: Float
        let width: Int
        let height: Int
    }

    /// Param count per camera model id. Distortion coefficients are only skipped, never used.
    private static func cameraModelParamCount(_ modelID: Int32) throws -> Int {
        switch modelID {
        case 0: return 3   // SIMPLE_PINHOLE: f, cx, cy
        case 1: return 4   // PINHOLE: fx, fy, cx, cy
        case 2: return 4   // SIMPLE_RADIAL: f, cx, cy, k
        case 3: return 5   // RADIAL: f, cx, cy, k1, k2
        case 4: return 8   // OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
        case 5: return 8   // OPENCV_FISHEYE
        case 6: return 12  // FULL_OPENCV
        case 7: return 5   // FOV
        case 8: return 4   // SIMPLE_RADIAL_FISHEYE
        case 9: return 5   // RADIAL_FISHEYE
        case 10: return 12 // THIN_PRISM_FISHEYE
        default: throw Error.unsupportedCameraModel(modelID)
        }
    }

    /// Whether the params are a single focal length (`f, cx, cy, ...`) or `fx, fy, cx, cy, ...`.
    private static func cameraModelHasSharedFocal(_ modelID: Int32) -> Bool {
        switch modelID {
        case 0, 2, 3, 8, 9: return true   // SIMPLE_* / RADIAL / *_RADIAL_FISHEYE
        default: return false
        }
    }

    static func parseCameras(_ data: Data) throws -> [UInt32: CameraParams] {
        var reader = BinaryReader(data)
        let numCameras = try reader.readUInt64()
        var result: [UInt32: CameraParams] = [:]
        for _ in 0..<numCameras {
            let cameraID = try reader.readUInt32()
            let modelID = try reader.readInt32()
            let width = Int(try reader.readUInt64())
            let height = Int(try reader.readUInt64())
            let numParams = try cameraModelParamCount(modelID)
            var params: [Double] = []
            params.reserveCapacity(numParams)
            for _ in 0..<numParams { params.append(try reader.readFloat64()) }

            let fx: Float
            let fy: Float
            let cx: Float
            let cy: Float
            if cameraModelHasSharedFocal(modelID) {
                fx = Float(params[0]); fy = Float(params[0])
                cx = Float(params[1]); cy = Float(params[2])
            } else {
                fx = Float(params[0]); fy = Float(params[1])
                cx = Float(params[2]); cy = Float(params[3])
            }
            result[cameraID] = CameraParams(
                fx: fx, fy: fy, cx: cx, cy: cy, width: width, height: height
            )
        }
        return result
    }

    // MARK: - images.bin

    struct ImageRecord {
        let id: UInt32
        let qw: Double, qx: Double, qy: Double, qz: Double
        let tx: Double, ty: Double, tz: Double
        let cameraID: UInt32
        let name: String
    }

    static func parseImages(_ data: Data) throws -> [ImageRecord] {
        var reader = BinaryReader(data)
        let numImages = try reader.readUInt64()
        var records: [ImageRecord] = []
        records.reserveCapacity(Int(numImages))
        for _ in 0..<numImages {
            let id = try reader.readUInt32()
            let qw = try reader.readFloat64()
            let qx = try reader.readFloat64()
            let qy = try reader.readFloat64()
            let qz = try reader.readFloat64()
            let tx = try reader.readFloat64()
            let ty = try reader.readFloat64()
            let tz = try reader.readFloat64()
            let cameraID = try reader.readUInt32()
            let name = try reader.readCString()
            // One 2D feature = (float64 x, float64 y, uint64 point3D_id).
            let numPoints2D = try reader.readUInt64()
            for _ in 0..<numPoints2D {
                _ = try reader.readFloat64()
                _ = try reader.readFloat64()
                _ = try reader.readUInt64()
            }
            records.append(ImageRecord(
                id: id, qw: qw, qx: qx, qy: qy, qz: qz,
                tx: tx, ty: ty, tz: tz, cameraID: cameraID, name: name
            ))
        }
        return records
    }

    // MARK: - points3D.bin

    static func parsePoints3D(_ data: Data) throws -> (positions: [Float], colors: [Float]) {
        var reader = BinaryReader(data)
        let numPoints = try reader.readUInt64()
        var positions: [Float] = []
        var colors: [Float] = []
        positions.reserveCapacity(Int(numPoints) * 3)
        colors.reserveCapacity(Int(numPoints) * 3)
        for _ in 0..<numPoints {
            _ = try reader.readUInt64()                  // point3D_id
            let x = try reader.readFloat64()
            let y = try reader.readFloat64()
            let z = try reader.readFloat64()
            let r = try reader.readUInt8()
            let g = try reader.readUInt8()
            let b = try reader.readUInt8()
            _ = try reader.readFloat64()                 // reprojection error
            let trackLength = try reader.readUInt64()
            for _ in 0..<trackLength {
                _ = try reader.readUInt32()              // image_id
                _ = try reader.readUInt32()              // point2D_idx
            }
            positions.append(Float(x))
            positions.append(Float(y))
            positions.append(Float(z))
            colors.append(Float(r) / 255.0)
            colors.append(Float(g) / 255.0)
            colors.append(Float(b) / 255.0)
        }
        return (positions, colors)
    }

    // MARK: - Quaternions (COLMAP / Eigen Hamilton convention)

    /// `qvec = (qw, qx, qy, qz)` → row-major 3x3 (world-to-camera).
    /// Same as `Eigen::Quaterniond(w, x, y, z).toRotationMatrix()`.
    static func qvecToRotation(_ qw: Double, _ qx: Double, _ qy: Double, _ qz: Double) -> [Float] {
        let norm = (qw * qw + qx * qx + qy * qy + qz * qz).squareRoot()
        let safe = norm > 0 ? norm : 1
        let w = qw / safe, x = qx / safe, y = qy / safe, z = qz / safe
        let r00 = 1 - 2 * (y * y + z * z)
        let r01 = 2 * (x * y - z * w)
        let r02 = 2 * (x * z + y * w)
        let r10 = 2 * (x * y + z * w)
        let r11 = 1 - 2 * (x * x + z * z)
        let r12 = 2 * (y * z - x * w)
        let r20 = 2 * (x * z - y * w)
        let r21 = 2 * (y * z + x * w)
        let r22 = 1 - 2 * (x * x + y * y)
        return [r00, r01, r02, r10, r11, r12, r20, r21, r22].map { Float($0) }
    }
}
