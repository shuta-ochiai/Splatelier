// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX

/// Per-stage profiler for the training loop (always on, negligible overhead). The first window is
/// warmup including kernel JIT, so the second becomes the baseline and later windows report % vs
/// baseline (over +5% suggests thermal throttling).
final class TrainingProfiler {
    private struct Bucket {
        var totalMs: Double = 0
        var count: Int = 0
        mutating func add(_ ms: Double) { totalMs += ms; count += 1 }
        var mean: Double { count > 0 ? totalMs / Double(count) : 0 }
    }

    private static let stageOrder = [
        "param_activations",
        "project_fwd", "tile_binning", "sh_eval_fwd", "rasterize_fwd",
        "loss_eval", "loss_item",
        "rasterize_bwd", "project_bwd", "sh_eval_bwd",
        "geo_depth_fwd", "geo_normals", "geo_loss", "geo_depth_bwd", "geo_normal_bwd", "geo_chain",
        "adam_enqueue"
    ]

    private var step = Bucket()
    private var postStep = Bucket()
    private var densifyFire = Bucket()
    private var lastDensifyDiagnostic: String?
    private var stages: [String: Bucket] = [:]
    private var baselineMs: Double?
    private var baselineRange: (start: Int, end: Int) = (0, 0)
    private var windowIndex = 0
    /// GPU energy (only when `TrainingDriver.gpuEnergyReader` exists). The cumulative value when
    /// the previous window closed, and totals over all windows / windows excluding warmup.
    private var energyMark: (joules: Double, time: Date)?
    private var energyTotal = (joules: 0.0, seconds: 0.0, iters: 0)
    private var energySteady = (joules: 0.0, iters: 0)

    /// Points the renderer / trainer stage timing hooks at this profiler. Call `uninstall` when
    /// done.
    func install() {
        GaussianRenderer.profileHook = { [unowned self] stage, ms in
            self.stages[stage, default: Bucket()].add(ms)
        }
        GaussianTrainer.profileHook = { [unowned self] stage, ms in
            self.stages[stage, default: Bucket()].add(ms)
        }
        if let joules = TrainingDriver.gpuEnergyReader?() {
            energyMark = (joules, Date())
        }
    }

    func uninstall() {
        GaussianRenderer.profileHook = nil
        GaussianTrainer.profileHook = nil
    }

    /// `densifyDiagnostic` is the strategy's diagnostic when densify fired on this step (the window
    /// report includes the latest one).
    func record(stepMs: Double, postStepMs: Double, densifyDiagnostic: String?) {
        step.add(stepMs)
        postStep.add(postStepMs)
        if let densifyDiagnostic {
            densifyFire.add(postStepMs)
            lastDensifyDiagnostic = densifyDiagnostic
        }
    }

    /// Closes the window at iter `it` and returns a report (nil if the window has no steps). Large
    /// postStep = densify is heavy; a monotonic rise vs baseline = thermal; growing mem peak = pool
    /// not released.
    func closeWindow(at it: Int) -> String? {
        guard step.count > 0 else { return nil }
        let bucketStart = it - step.count + 1
        let iterMs = step.mean + postStep.mean
        let warmupTag = windowIndex == 0 ? " (warmup)" : ""
        var baselineNote = ""
        if windowIndex == 1 {
            baselineMs = iterMs
            baselineRange = (start: bucketStart, end: it)
        } else if let base = baselineMs, windowIndex > 1 {
            let delta = (iterMs - base) / base * 100
            let tag: String
            if delta > 5 { tag = " (thermal?)" }
            else if delta < -2 { tag = " (cool)" }
            else { tag = "" }
            baselineNote = String(format: " vs baseline[%d..%d] %+.1f%%%@",
                                  baselineRange.start, baselineRange.end, delta, tag)
        }
        var lines: [String] = []
        lines.append(String(format:
            "  prof[%d..%d]%@ iter=%.1fms (step=%.1f postStep=%.2f)%@ | %@",
            bucketStart, it, warmupTag, iterMs, step.mean, postStep.mean,
            baselineNote as NSString, Self.memoryLine() as NSString))
        let stageParts = Self.stageOrder.compactMap { name -> String? in
            guard let b = stages[name], b.count > 0 else { return nil }
            return "\(name)=\(String(format: "%.1f", b.mean))"
        }
        let stageSum = Self.stageOrder.reduce(0.0) { $0 + (stages[$1]?.mean ?? 0) }
        lines.append(String(format:
            "       stages: %@ (sum=%.1f)",
            stageParts.joined(separator: " ") as NSString, stageSum))
        if densifyFire.count > 0 {
            lines.append(String(format:
                "       densify-fires n=%d avg=%.1fms | last %@",
                densifyFire.count, densifyFire.mean, (lastDensifyDiagnostic ?? "") as NSString))
        }
        if let line = energyLine(iters: step.count) {
            lines.append(line)
        }

        step = Bucket()
        postStep = Bucket()
        densifyFire = Bucket()
        lastDensifyDiagnostic = nil
        stages.removeAll(keepingCapacity: true)
        windowIndex += 1
        return lines.joined(separator: "\n")
    }

    /// Returns GPU energy since the previous window, per iter and as average power.
    private func energyLine(iters: Int) -> String? {
        guard let mark = energyMark, let joules = TrainingDriver.gpuEnergyReader?() else { return nil }
        let now = Date()
        let dJ = joules - mark.joules
        let dt = now.timeIntervalSince(mark.time)
        energyMark = (joules, now)
        energyTotal.joules += dJ
        energyTotal.seconds += dt
        energyTotal.iters += iters
        if windowIndex > 0 {
            energySteady.joules += dJ
            energySteady.iters += iters
        }
        return String(format: "       gpu energy=%.0fmJ/iter power=%.1fW",
                      dJ / Double(iters) * 1000, dt > 0 ? dJ / dt : 0)
    }

    /// GPU energy for the whole training run. The first window includes kernel JIT, so per-iter
    /// excluding it is also reported.
    func energySummary() -> String? {
        guard energyTotal.iters > 0 else { return nil }
        var s = String(format: "[energy] GPU %.1fJ / %d iter = %.0fmJ/iter, avg %.1fW",
                       energyTotal.joules, energyTotal.iters,
                       energyTotal.joules / Double(energyTotal.iters) * 1000,
                       energyTotal.seconds > 0 ? energyTotal.joules / energyTotal.seconds : 0)
        if energySteady.iters > 0 {
            s += String(format: " (excluding warmup %.0fmJ/iter)",
                        energySteady.joules / Double(energySteady.iters) * 1000)
        }
        return s
    }

    /// iOS has no temperature API, so thermalState stands in. Together with free memory it tells
    /// thermal slowdown apart from OOM.
    private static func memoryLine() -> String {
        let snap = MLX.Memory.snapshot()
        let thermal: String
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: thermal = "nominal"
        case .fair: thermal = "fair"
        case .serious: thermal = "serious"
        case .critical: thermal = "critical"
        @unknown default: thermal = "unknown"
        }
        #if os(iOS)
        let availNote = String(format: " avail=%dMB", os_proc_available_memory() / (1024 * 1024))
        #else
        let availNote = ""
        #endif
        return String(format:
            "mem active=%dMB peak=%dMB thermal=%@%@",
            snap.activeMemory / (1024 * 1024),
            snap.peakMemory / (1024 * 1024),
            thermal as NSString, availNote as NSString)
    }
}
