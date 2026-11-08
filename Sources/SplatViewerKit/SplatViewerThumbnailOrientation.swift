// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Which way to turn a capture's picture (a frame, or a render from a training view) so it stands
// upright. Pictures are in sensor orientation, so a portrait capture lies on its side and a
// landscape one is already upright (or upside down). The view's pose says where world up (+Y)
// falls in the image.

import Foundation

public enum SplatViewerThumbnailOrientation {
    /// Clockwise quarter turns (0...3) that put world up at the top of the thumbnail saved next to
    /// `url` (`<base>.png`), read from its `<base>.preview.json`. nil if that file is missing or
    /// unreadable.
    public static func clockwiseQuarterTurns(siblingOf url: URL) -> Int? {
        struct Metadata: Decodable { let viewmat: [Float] }
        let metadataURL = url.deletingPathExtension().appendingPathExtension("preview.json")
        guard let data = try? Data(contentsOf: metadataURL),
              let metadata = try? JSONDecoder().decode(Metadata.self, from: data)
        else { return nil }
        return clockwiseQuarterTurns(viewmat: metadata.viewmat)
    }

    /// The same for an image in sensor orientation taken from a row-major world→camera matrix
    /// (OpenCV: +X right, +Y down in the image; world +Y up, as ARKit's). nil if world up is
    /// (nearly) along the optical axis, where the image has no clear up.
    public static func clockwiseQuarterTurns(viewmat m: [Float]) -> Int? {
        guard m.count == 16 else { return nil }
        // World +Y in camera coordinates is the rotation's column 1; its image-plane part.
        let x = m[1], y = m[5]
        guard x.isFinite, y.isFinite, x * x + y * y > 1e-4 else { return nil }
        if abs(y) >= abs(x) {
            return y <= 0 ? 0 : 2        // up toward the image top: as is; toward the bottom: 180°
        }
        return x < 0 ? 1 : 3             // up toward the left edge: clockwise; right: counter-clockwise
    }
}
