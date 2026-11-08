// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Which way up a splats file is. 3DGS files keep COLMAP's camera-like axes, +Y pointing into the
// ground, and Y-up viewers turn them 180° about Z to stand them up (SuperSplat, PlayCanvas). SplatIO
// reads .spz and .splat into the same axes as a PLY. Splatelier writes its captures that way too,
// turned from ARKit's Y-up world, with a header comment saying so (Trainer's `PLYAxes`); only a
// capture PLY written before then is Y-up, and its `<base>.preview.json` gives it away.
//
// The viewer keeps everything in the file's coordinates and turns the view instead
// (`SplatViewerModel.fileRotation`). What else comes with a capture is in its Y-up world and is
// brought into the file's coordinates on loading: the mesh (OBJ) here, the source view of
// `preview.json` in `SplatViewerPreviewCamera`.

import Foundation
import simd

enum SplatViewerFileAxes {
    /// Header comment of a PLY turned 180° about Z from its Y-up world (Trainer's
    /// `PLYAxes.turnedComment`, which writes it).
    static let turnedComment = "comment splatelier: turned 180 degrees about z from a y-up world"

    /// A half turn about Z.
    static let halfTurnZ = simd_quatf(ix: 0, iy: 0, iz: 1, r: 0)

    /// The turn that stands the splats at `url` upright in the viewer: a half turn about Z, except for
    /// a Splatelier capture PLY written Y-up (no `turnedComment`, a `<base>.preview.json` beside it),
    /// and for no splats at all.
    static func uprightRotation(forSplatsAt url: URL?) -> simd_quatf {
        let identity = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
        guard let url else { return identity }
        if url.pathExtension.lowercased() == "ply" {
            let turned = header(of: url).map(isTurned) ?? false
            let preview = url.deletingPathExtension().appendingPathExtension("preview.json")
            if !turned, FileManager.default.fileExists(atPath: preview.path) { return identity }
        }
        return halfTurnZ
    }

    static func isTurned(header: String) -> Bool {
        header.split(separator: "\n").contains { $0.trimmingCharacters(in: .whitespaces) == turnedComment }
    }

    /// The header (everything before `end_header`) of a PLY, read from its first bytes.
    static func header(of url: URL) -> String? {
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
}

extension LoadedMesh {
    /// The mesh turned about the origin by `q`.
    func rotated(by q: simd_quatf) -> LoadedMesh {
        var turned = self
        for v in 0..<vertexCount {
            let p = q.act(SIMD3(positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2]))
            turned.positions[v * 3] = p.x; turned.positions[v * 3 + 1] = p.y; turned.positions[v * 3 + 2] = p.z
            guard normals.count == positions.count else { continue }
            let n = q.act(SIMD3(normals[v * 3], normals[v * 3 + 1], normals[v * 3 + 2]))
            turned.normals[v * 3] = n.x; turned.normals[v * 3 + 1] = n.y; turned.normals[v * 3 + 2] = n.z
        }
        let bounds = LoadedMesh.bounds(turned.positions)
        turned.center = bounds.center
        turned.radius = bounds.radius
        return turned
    }
}
