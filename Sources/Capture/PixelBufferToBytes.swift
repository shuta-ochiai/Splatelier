// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// ARFrame.capturedImage (420YpCbCr8BiPlanarFullRange) → downscale + RGB UInt8 [H*W*3]. Accumulating
// float32 triggers jetsam OOM, so UInt8 is kept. CoreImage (Metal) does YCbCr→RGB and downscaling
// in one pass. The YCbCr matrix is auto-selected from the buffer's attachments. The input color
// space is declared as sRGB. capturedImage is tagged transfer=ITU_R_709_2, and by default the
// 709→sRGB tone conversion lifts the midtones and makes training images look washed out. Kept in
// sensor orientation without rotation (to match ARCamera.intrinsics). Non-iOS is a stub that throws
// unavailable (the symbols remain).

import Foundation

public enum PixelBufferToBytesError: Error, Sendable {
    case unavailableOnPlatform
    case poolCreationFailed(OSStatus)
    case renderFailed(String)
}

public final class PixelBufferToBytes {

    /// Output for one frame. `bytes.count == width * height * 3`.
    public struct Output {
        public let bytes: [UInt8]
        public let width: Int
        public let height: Int
    }

    public init(targetMaxDim: Int) throws {
        #if canImport(ARKit) && os(iOS)
        try self.iOSImpl = IOSImpl(targetMaxDim: targetMaxDim)
        #else
        _ = targetMaxDim
        self.iOSImpl = nil
        #endif
    }

    /// Converts a CVPixelBuffer. Takes `Any` so macOS does not import CVPixelBuffer.
    public func convert(_ pixelBuffer: Any) throws -> Output {
        #if canImport(ARKit) && os(iOS)
        return try iOSImpl!.convert(pixelBuffer)
        #else
        throw PixelBufferToBytesError.unavailableOnPlatform
        #endif
    }

    /// Formats the capture buffer's color tags (format / matrix / primaries / transfer) as one line
    /// (for logs). convert declares the input as sRGB, preventing the midtone lift from the
    /// transfer=709 tag.
    public static func colorTagDescription(_ pixelBuffer: Any) -> String {
        #if canImport(ARKit) && os(iOS)
        let pb = pixelBuffer as! CVPixelBuffer
        func tag(_ key: CFString) -> String {
            CVBufferCopyAttachment(pb, key, nil).map { "\($0)" } ?? "none"
        }
        let fmt = CVPixelBufferGetPixelFormatType(pb)
        let fourCC = String(bytes: [24, 16, 8, 0].map { UInt8((fmt >> UInt32($0)) & 0xFF) },
                            encoding: .ascii) ?? "\(fmt)"
        return "format=\(fourCC) matrix=\(tag(kCVImageBufferYCbCrMatrixKey)) "
            + "primaries=\(tag(kCVImageBufferColorPrimariesKey)) "
            + "transfer=\(tag(kCVImageBufferTransferFunctionKey))"
        #else
        _ = pixelBuffer
        return "unavailable"
        #endif
    }

    #if canImport(ARKit) && os(iOS)
    private let iOSImpl: IOSImpl?
    #else
    private let iOSImpl: Void?
    #endif
}

#if canImport(ARKit) && os(iOS)

import ARKit
import CoreImage
import CoreVideo
import Metal

private final class IOSImpl {
    private let ciContext: CIContext
    /// Color space of the input (after YCbCr→RGB) and output. Kept identical so no tone conversion
    /// occurs (see file header).
    private let srgb = CGColorSpace(name: CGColorSpace.sRGB)!
    private let targetMaxDim: Int
    private var pool: CVPixelBufferPool?
    private var poolSize: (w: Int, h: Int) = (0, 0)

    init(targetMaxDim: Int) throws {
        precondition(targetMaxDim > 0, "targetMaxDim must be positive")
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw PixelBufferToBytesError.renderFailed("no Metal device")
        }
        // Do not pass NSNull as workingColorSpace. The YCbCr→RGB conversion gets skipped and
        // luminance breaks.
        self.ciContext = CIContext(mtlDevice: device, options: [
            .cacheIntermediates: false,
        ])
        self.targetMaxDim = targetMaxDim
    }

    func convert(_ pixelBufferAny: Any) throws -> PixelBufferToBytes.Output {
        let pixelBuffer = pixelBufferAny as! CVPixelBuffer
        let srcW = CVPixelBufferGetWidth(pixelBuffer)
        let srcH = CVPixelBufferGetHeight(pixelBuffer)
        let longSide = max(srcW, srcH)
        let scale: CGFloat = longSide > targetMaxDim
            ? CGFloat(targetMaxDim) / CGFloat(longSide) : 1.0
        let dstW = max(1, Int((CGFloat(srcW) * scale).rounded()))
        let dstH = max(1, Int((CGFloat(srcH) * scale).rounded()))

        // Declare the input as sRGB to stop the tone conversion from the 709 transfer tag (see file
        // header).
        var ci = CIImage(cvPixelBuffer: pixelBuffer, options: [.colorSpace: srgb])
        if scale != 1.0 {
            ci = ci.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        }
        let dst = try obtainBuffer(width: dstW, height: dstH)
        ciContext.render(
            ci, to: dst,
            bounds: CGRect(x: 0, y: 0, width: dstW, height: dstH),
            colorSpace: srgb
        )

        CVPixelBufferLockBaseAddress(dst, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(dst, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(dst) else {
            throw PixelBufferToBytesError.renderFailed("no base address")
        }
        let stride = CVPixelBufferGetBytesPerRow(dst)
        let basePtr = base.assumingMemoryBound(to: UInt8.self)
        var rgb = [UInt8](repeating: 0, count: dstW * dstH * 3)
        rgb.withUnsafeMutableBufferPointer { dstBuf in
            for y in 0..<dstH {
                let rowSrc = basePtr.advanced(by: y * stride)
                let rowDst = dstBuf.baseAddress!.advanced(by: y * dstW * 3)
                for x in 0..<dstW {
                    let srcOff = x * 4
                    let dstOff = x * 3
                    // 32BGRA → RGB
                    rowDst[dstOff + 0] = rowSrc[srcOff + 2]
                    rowDst[dstOff + 1] = rowSrc[srcOff + 1]
                    rowDst[dstOff + 2] = rowSrc[srcOff + 0]
                }
            }
        }
        // TODO: could be replaced with vImageConvert_BGRA8888toRGB888.
        return PixelBufferToBytes.Output(bytes: rgb, width: dstW, height: dstH)
    }

    private func obtainBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
        if pool == nil || poolSize != (width, height) {
            let attrs: [CFString: Any] = [
                kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey: width,
                kCVPixelBufferHeightKey: height,
                kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
            ]
            var p: CVPixelBufferPool?
            let s = CVPixelBufferPoolCreate(nil, nil, attrs as CFDictionary, &p)
            guard s == kCVReturnSuccess, let pool = p else {
                throw PixelBufferToBytesError.poolCreationFailed(s)
            }
            self.pool = pool
            self.poolSize = (width, height)
        }
        var buf: CVPixelBuffer?
        let s = CVPixelBufferPoolCreatePixelBuffer(nil, pool!, &buf)
        guard s == kCVReturnSuccess, let buf else {
            throw PixelBufferToBytesError.poolCreationFailed(s)
        }
        return buf
    }
}

#endif
