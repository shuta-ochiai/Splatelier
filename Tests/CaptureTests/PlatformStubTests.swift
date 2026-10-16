// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Behavior of the ARKitCaptureSession stub on non-iOS (macOS / visionOS).

#if !os(iOS)

import Foundation
import Testing
@testable import Capture

@Test
func arkitCaptureSessionUnavailableOnMacOS() {
    let session = ARKitCaptureSession()
    do {
        try session.start()
        Issue.record("start() should throw on macOS")
    } catch let CaptureError.unavailableOnPlatform {
        // Expected behavior
    } catch {
        Issue.record("expected .unavailableOnPlatform, got \(error)")
    }
}

@Test
func arkitCaptureSessionStateIsIdleOnMacOS() {
    let session = ARKitCaptureSession()
    if case .idle = session.state {
        // Expected behavior
    } else {
        Issue.record("expected .idle, got \(session.state)")
    }
}

@Test
func arkitCaptureSessionDiagnosticsDefaultsOnMacOS() {
    let session = ARKitCaptureSession()
    let d = session.diagnostics
    #expect(d.keptFrameCount == 0)
    #expect(!d.relocalizationOccurred)
    #expect(d.anchorUpdateCount == 0)
}

@Test
func pixelBufferToBytesUnavailableOnMacOS() throws {
    let conv = try PixelBufferToBytes(targetMaxDim: 256)
    do {
        _ = try conv.convert(NSObject())
        Issue.record("convert() should throw on macOS")
    } catch let PixelBufferToBytesError.unavailableOnPlatform {
        // Expected behavior
    } catch {
        Issue.record("expected .unavailableOnPlatform, got \(error)")
    }
}

@Test
func captureConfigurationDefaultsAreSane() {
    let cfg = CaptureConfiguration()
    #expect(cfg.minMotionMeters == 0.05)   // PocketGS τ_d
    #expect(cfg.maxFrames == 300)
    #expect(cfg.targetMaxDim == 768)       // PocketGS Sec III-A
    #expect(cfg.settleSeconds == 0.5)
    #expect(cfg.requireNormalTracking)
    #expect(cfg.lockExposureAndWhiteBalance)          // AE/AWB lock ON by default (prevents color drift)
    #expect(cfg.exposureLockWarmupSeconds == 1.0)
}

#endif
