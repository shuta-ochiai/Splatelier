// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation

/// Optimization run mode by thermal state (reported to the UI via `onThermalThrottle`).
///  - throttled: thermalState=.serious. Keeps going with sleeps between steps to lower the duty
///    cycle.
///  - paused:    thermalState=.critical. Pauses until cooled down, then resumes automatically.
public enum ThermalThrottleState: Sendable, Equatable {
    case normal
    case throttled
    case paused
}

/// How the PocketGS Operator T replay cache is used.
///  - off: Always coop backward. Fast but runs hot.
///  - on: Always replay backward. Slower (+26% on device) but memory-bandwidth-bound, so lower
///    power and less prone to thermal slowdown.
public enum ReplayCacheMode: String, Sendable {
    case off
    case on

    /// Sets up the renderer at training start and logs it.
    func configure(_ renderer: GaussianRenderer, replayKMax: Int, log: (String) -> Void) {
        renderer.replayKMax = replayKMax
        switch self {
        case .off:
            renderer.useReplayCache = false
            log("rasterizer: PocketGS Operator T replay cache disabled")
        case .on:
            renderer.useReplayCache = true
            log("rasterizer: PocketGS Operator T replay cache enabled (K_max=\(replayKMax))")
        }
    }
}

/// Thermal guard before each training step (iOS). On critical, pauses until cooled and resumes
/// automatically; on serious, sleeps for the previous step time × `seriousThrottleFactor` to lower
/// the duty cycle, preventing critical / an OS thermal kill. No-op on macOS.
struct ThermalGovernor {
    let seriousThrottleFactor: Float
    let onChange: ((ThermalThrottleState) -> Void)?
    private var notified: ThermalThrottleState = .normal
    private var seriousLogged = false

    init(seriousThrottleFactor: Float, onChange: ((ThermalThrottleState) -> Void)?) {
        self.seriousThrottleFactor = seriousThrottleFactor
        self.onChange = onChange
    }

    private var thermalState: ProcessInfo.ThermalState {
        TrainingDriver.simulatedThermalState ?? ProcessInfo.processInfo.thermalState
    }

    private mutating func notify(_ s: ThermalThrottleState) {
        guard s != notified else { return }
        notified = s
        onChange?(s)
    }

    mutating func beforeStep(iteration it: Int, lastStepSeconds: Double, log: (String) -> Void) {
        if thermalState == .critical {
            notify(.paused)
            log("[thermal] critical → too hot, pausing optimization (waiting for cool-down, iter \(it))")
            var waited = 0.0
            while thermalState == .critical {
                Thread.sleep(forTimeInterval: 1.0)
                waited += 1.0
            }
            log(String(format: "[thermal] cooled down (waited %.0fs) → resuming at iter %d", waited, it))
        }
        // Re-evaluate after the critical cool-down wait (it is often serious after leaving
        // critical).
        if seriousThrottleFactor > 0
            && thermalState == .serious
            && lastStepSeconds > 0 {
            notify(.throttled)
            Thread.sleep(forTimeInterval: lastStepSeconds * Double(seriousThrottleFactor))
            if !seriousLogged {
                log(String(format:
                    "[thermal] serious → slowing down with a lower duty cycle (sleep %.0f%% / step, from iter %d). Prevents critical.",
                    Double(seriousThrottleFactor) * 100, it))
                seriousLogged = true
            }
        } else if thermalState != .critical {
            // Back to normal speed. Reset so the log is emitted again if it becomes serious again.
            notify(.normal)
            seriousLogged = false
        }
    }
}
