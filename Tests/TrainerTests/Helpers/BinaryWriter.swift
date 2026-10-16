// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation

/// Minimal little-endian writer for synthesizing COLMAP `.bin` fixtures inside tests.
struct BinaryWriter {
    private(set) var bytes: [UInt8] = []

    mutating func writeUInt8(_ v: UInt8) { bytes.append(v) }

    mutating func writeUInt32(_ v: UInt32) {
        for i in 0..<4 { bytes.append(UInt8((v >> (8 * i)) & 0xff)) }
    }

    mutating func writeInt32(_ v: Int32) { writeUInt32(UInt32(bitPattern: v)) }

    mutating func writeUInt64(_ v: UInt64) {
        for i in 0..<8 { bytes.append(UInt8((v >> (8 * i)) & 0xff)) }
    }

    mutating func writeFloat32(_ v: Float) { writeUInt32(v.bitPattern) }
    mutating func writeFloat64(_ v: Double) { writeUInt64(v.bitPattern) }

    mutating func writeCString(_ s: String) {
        bytes.append(contentsOf: Array(s.utf8))
        bytes.append(0)
    }

    var data: Data { Data(bytes) }
}
