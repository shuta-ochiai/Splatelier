// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// The "Gaussian splats" entry of the viewer's Share menu: the trained PLY as it is, or converted
// to a smaller / other format when picked.

import Foundation
import SplatViewerKit
import Trainer

enum SplatExportShare {
    /// Formats offered besides the PLY itself, in menu order.
    private static let convertedFormats: [SplatExportFormat] = [.compressedPly, .spz, .splat, .glb, .sog, .usda]

    /// The Share entry for the capture's GS file. Only a PLY can be converted (it is read back in
    /// full); anything else, such as an imported .spz, is shared as it is.
    static func item(for url: URL, title: String = "Gaussian splats") -> SplatViewerShareItem {
        guard url.pathExtension.lowercased() == "ply" else {
            return SplatViewerShareItem(title: title, systemImage: "cube.fill", url: url)
        }
        // A capture written Y-up before captures were turned goes out turned, like newer ones.
        var options = PLYAxes.isLegacyYUp(url)
            ? [SplatViewerShareItem.Option(title: "PLY (as trained)") { try await export(url, as: .ply) }]
            : [SplatViewerShareItem.Option(title: "PLY (as trained)", url: url)]
        for format in convertedFormats {
            options.append(SplatViewerShareItem.Option(title: format.displayName) {
                try await export(url, as: format)
            })
        }
        return SplatViewerShareItem(title: title, systemImage: "cube.fill", options: options)
    }

    /// Every format a capture's splats can be shared in, PLY (as trained) first.
    static let formats: [SplatExportFormat] = [.ply] + convertedFormats

    /// The picker's wording: the format and the file extension it gives.
    static func label(for format: SplatExportFormat) -> String {
        switch format {
        case .ply: return "PLY as trained (.ply)"
        default: return "\(format.displayName) (.\(format.fileExtension))"
        }
    }

    /// A fresh temporary directory for one share (the share sheet reads the files from there).
    /// Earlier exports are deleted first.
    static func makeExportDirectory() throws -> URL {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("splat-export", isDirectory: true)
        try? fm.removeItem(at: root)
        let dir = root.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// The splats file to share in `format`: the file itself for the PLY as trained (unless it was
    /// written Y-up, before captures were turned), and for anything that is not a PLY (an imported
    /// .spz, say); otherwise a converted copy in `dir`.
    static func file(for url: URL, as format: SplatExportFormat, in dir: URL) async throws -> URL {
        if url.pathExtension.lowercased() != "ply" { return url }
        if format == .ply, !PLYAxes.isLegacyYUp(url) { return url }
        return try await convert(url, as: format, into: dir)
    }

    /// Converts on a background thread and returns the file in `dir`.
    private static func convert(_ plyURL: URL, as format: SplatExportFormat, into dir: URL) async throws -> URL {
        try await Task.detached(priority: .userInitiated) {
            let out = format.url(replacingExtensionOf: dir.appendingPathComponent(plyURL.lastPathComponent))
            // readPLY already drops the zero padding above the trained SH degree.
            let table = try SplatTable.readPLYInUsualAxes(url: plyURL)
            try SplatExporter.write(table, format: format, to: out)
            return out
        }.value
    }

    /// The cropped splats' Share options: the PLY (see the viewer) and the same formats as above.
    static let cropExport = SplatViewerCropExport(
        formats: convertedFormats.map { SplatViewerCropExport.Format(title: $0.displayName, fileExtension: $0.fileExtension) },
        convert: { plyURL, fileExtension in
            guard let format = convertedFormats.first(where: { $0.fileExtension == fileExtension }) else {
                throw CocoaError(.fileWriteUnknown)
            }
            return try await export(plyURL, as: format)
        })

    /// For the viewer's Share menu: a conversion made when the option is picked.
    private static func export(_ plyURL: URL, as format: SplatExportFormat) async throws -> URL {
        try await convert(plyURL, as: format, into: try makeExportDirectory())
    }
}
