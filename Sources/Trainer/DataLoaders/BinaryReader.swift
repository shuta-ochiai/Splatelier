// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation

/// Read cursor for COLMAP `.bin` (little-endian). Assembles values byte by byte to be
/// alignment-independent.
struct BinaryReader {
    let data: [UInt8]
    private(set) var offset: Int = 0

    init(_ data: Data) {
        self.data = [UInt8](data)
    }

    var isAtEnd: Bool { offset >= data.count }
    var remaining: Int { data.count - offset }

    enum Error: Swift.Error, CustomStringConvertible {
        case outOfBounds(needed: Int, remaining: Int)
        var description: String {
            switch self {
            case .outOfBounds(let needed, let remaining):
                return "BinaryReader out of bounds: needed \(needed) bytes, \(remaining) remaining"
            }
        }
    }

    private mutating func require(_ n: Int) throws {
        if remaining < n { throw Error.outOfBounds(needed: n, remaining: remaining) }
    }

    mutating func readUInt8() throws -> UInt8 {
        try require(1)
        defer { offset += 1 }
        return data[offset]
    }

    mutating func readUInt32() throws -> UInt32 {
        try require(4)
        var v: UInt32 = 0
        for i in 0..<4 { v |= UInt32(data[offset + i]) << (8 * i) }
        offset += 4
        return v
    }

    mutating func readInt32() throws -> Int32 {
        Int32(bitPattern: try readUInt32())
    }

    mutating func readUInt64() throws -> UInt64 {
        try require(8)
        var v: UInt64 = 0
        for i in 0..<8 { v |= UInt64(data[offset + i]) << (8 * i) }
        offset += 8
        return v
    }

    mutating func readInt64() throws -> Int64 {
        Int64(bitPattern: try readUInt64())
    }

    mutating func readFloat32() throws -> Float {
        Float(bitPattern: try readUInt32())
    }

    mutating func readFloat64() throws -> Double {
        Double(bitPattern: try readUInt64())
    }

    /// NUL-terminated UTF-8 string (COLMAP image name).
    mutating func readCString() throws -> String {
        var bytes: [UInt8] = []
        while true {
            let b = try readUInt8()
            if b == 0 { break }
            bytes.append(b)
        }
        return String(decoding: bytes, as: UTF8.self)
    }
}
