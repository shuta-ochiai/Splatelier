// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Which way up a 3DGS PLY is. 3DGS grew out of COLMAP, whose world follows its cameras (OpenCV: +Y
// down), so a PLY made elsewhere has +Y pointing into the ground, and Y-up viewers (SuperSplat and
// PlayCanvas, glTF, Scaniverse's exports) turn it 180° about Z to stand it up. A capture is trained
// in ARKit's world, gravity-aligned with +Y up, so its PLY is written turned 180° about Z to match
// the others, with a header comment saying so; reading it back for training turns it again. PLYs
// written before that have no comment and stay Y-up; their `<base>.preview.json` gives them away.
//
// A 180° turn about Z is its own inverse: x and y change sign, a quaternion q becomes r·q with
// r = (w 0, x 0, y 0, z 1), and each real SH coefficient Y_l^m is multiplied by (-1)^m.

import Foundation

public enum PLYAxes {
    /// Header comment of a PLY turned 180° about Z from its Y-up world. The viewer
    /// (`SplatViewerFileAxes`) looks for the same line.
    public static let turnedComment = "comment splatelier: turned 180 degrees about z from a y-up world"

    /// The header (everything before `end_header`) of a PLY, read from its first bytes.
    public static func header(of url: URL) -> String? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        var data = Data()
        let marker = Data("end_header".utf8)
        while data.count < 1 << 20 {
            guard let chunk = try? handle.read(upToCount: 16 << 10), !chunk.isEmpty else { break }
            data.append(chunk)
            if let r = data.range(of: marker) { return String(decoding: data[..<r.lowerBound], as: UTF8.self) }
        }
        return nil
    }

    /// The header carries `turnedComment`.
    public static func isTurned(header: String) -> Bool {
        header.split(separator: "\n").contains { $0.trimmingCharacters(in: .whitespaces) == turnedComment }
    }

    /// The PLY at `url` is a capture written Y-up, before captures were turned: no `turnedComment`,
    /// and a `<base>.preview.json` beside it (only capture training writes one).
    public static func isLegacyYUp(_ url: URL) -> Bool {
        guard url.pathExtension.lowercased() == "ply", let header = header(of: url), !isTurned(header: header)
        else { return false }
        let preview = url.deletingPathExtension().appendingPathExtension("preview.json")
        return FileManager.default.fileExists(atPath: preview.path)
    }

    /// The sign a 180° turn about Z gives the higher-order SH coefficient `c` (0 for l = 1, m = -1;
    /// then in order of l, and of m within it): (-1)^m.
    @inline(__always) public static func restSign(_ c: Int) -> Float {
        let l = Int(Double(c + 1).squareRoot())
        let m = c + 1 - l * l - l
        return m % 2 == 0 ? 1 : -1
    }

    /// A unit quaternion (w, x, y, z) after the 180° turn about Z: r·q = (-z, -y, x, w).
    @inline(__always) public static func turnedQuat(_ w: Float, _ x: Float, _ y: Float, _ z: Float)
        -> (Float, Float, Float, Float) {
        (-z, -y, x, w)
    }
}

extension SplatTable {
    /// The table turned 180° about Z: Y-up to the usual 3DGS PLY axes, or back.
    public func turnedAboutZ() -> SplatTable {
        var t = self
        let rc = restCoeffs
        let signs = (0..<rc).map(PLYAxes.restSign)
        for i in 0..<count {
            t.positions[i * 3] = -positions[i * 3]
            t.positions[i * 3 + 1] = -positions[i * 3 + 1]
            let q = PLYAxes.turnedQuat(quats[i * 4], quats[i * 4 + 1], quats[i * 4 + 2], quats[i * 4 + 3])
            t.quats[i * 4] = q.0; t.quats[i * 4 + 1] = q.1; t.quats[i * 4 + 2] = q.2; t.quats[i * 4 + 3] = q.3
            for ch in 0..<3 {
                for c in 0..<rc where signs[c] < 0 {
                    t.shRest[(i * 3 + ch) * rc + c] = -shRest[(i * 3 + ch) * rc + c]
                }
            }
        }
        return t
    }
}

extension SplatTable {
    /// ``readPLY(url:)`` in the usual 3DGS axes: a capture PLY written Y-up before captures were
    /// turned (`PLYAxes.isLegacyYUp`) is turned on the way in, so exports of it stand up elsewhere.
    public static func readPLYInUsualAxes(url: URL) throws -> SplatTable {
        let table = try readPLY(url: url)
        return PLYAxes.isLegacyYUp(url) ? table.turnedAboutZ() : table
    }
}
