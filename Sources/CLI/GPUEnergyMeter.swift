// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Reads Apple Silicon GPU energy consumption from IOReport's "Energy Model" group (no sudo needed).
// Heat = energy, so this is used to compare thermal A/B runs by "GPU energy per iter". IOReport is
// a private API, so this is Mac-CLI only (not in the app). It is loaded via dlopen, so where it
// cannot be read this returns nil and training runs as usual. Values are system-wide cumulative
// totals, so other processes using the GPU get mixed in (check beforehand with `idleWatts`).

#if os(macOS)

import Foundation

final class GPUEnergyMeter: @unchecked Sendable {
    private typealias CopyChannelsInGroup = @convention(c)
        (CFString, CFString?, UInt64, UInt64, UInt64) -> Unmanaged<CFDictionary>?
    private typealias CreateSubscription = @convention(c)
        (UnsafeMutableRawPointer?, CFMutableDictionary,
         UnsafeMutablePointer<Unmanaged<CFMutableDictionary>?>, UInt64, CFTypeRef?) -> OpaquePointer?
    private typealias CreateSamples = @convention(c)
        (OpaquePointer, CFMutableDictionary, CFTypeRef?) -> Unmanaged<CFDictionary>?
    private typealias SimpleGetIntegerValue = @convention(c) (CFDictionary, Int32) -> Int64
    private typealias ChannelGetString = @convention(c) (CFDictionary) -> Unmanaged<CFString>?

    private let subscription: OpaquePointer
    private let subscribed: CFMutableDictionary
    private let createSamples: CreateSamples
    private let getInteger: SimpleGetIntegerValue
    private let channelName: ChannelGetString
    private let unitLabel: ChannelGetString
    private let lock = NSLock()

    /// nil when IOReport cannot be read (different OS / no channel).
    init?() {
        guard let lib = dlopen("/usr/lib/libIOReport.dylib", RTLD_NOW),
              let copyPtr = dlsym(lib, "IOReportCopyChannelsInGroup"),
              let subPtr = dlsym(lib, "IOReportCreateSubscription"),
              let samplesPtr = dlsym(lib, "IOReportCreateSamples"),
              let intPtr = dlsym(lib, "IOReportSimpleGetIntegerValue"),
              let namePtr = dlsym(lib, "IOReportChannelGetChannelName"),
              let unitPtr = dlsym(lib, "IOReportChannelGetUnitLabel")
        else { return nil }
        let copyChannels = unsafeBitCast(copyPtr, to: CopyChannelsInGroup.self)
        let createSubscription = unsafeBitCast(subPtr, to: CreateSubscription.self)
        createSamples = unsafeBitCast(samplesPtr, to: CreateSamples.self)
        getInteger = unsafeBitCast(intPtr, to: SimpleGetIntegerValue.self)
        channelName = unsafeBitCast(namePtr, to: ChannelGetString.self)
        unitLabel = unsafeBitCast(unitPtr, to: ChannelGetString.self)

        guard let channels = copyChannels("Energy Model" as CFString, nil, 0, 0, 0)?.takeRetainedValue(),
              let desired = CFDictionaryCreateMutableCopy(nil, 0, channels)
        else { return nil }
        var subbed: Unmanaged<CFMutableDictionary>?
        guard let sub = createSubscription(nil, desired, &subbed, 0, nil),
              let subbedDict = subbed?.takeRetainedValue()
        else { return nil }
        subscription = sub
        subscribed = subbedDict
        guard joules() != nil else { return nil }
    }

    /// Cumulative GPU energy consumed since boot (J).
    func joules() -> Double? {
        lock.lock()
        defer { lock.unlock() }
        guard let samples = createSamples(subscription, subscribed, nil)?.takeRetainedValue(),
              let items = (samples as NSDictionary)["IOReportChannels"] as? [NSDictionary]
        else { return nil }
        for item in items {
            let dict = item as CFDictionary
            guard channelName(dict)?.takeUnretainedValue() as String? == "GPU Energy" else { continue }
            let value = Double(getInteger(dict, 0))
            switch unitLabel(dict)?.takeUnretainedValue() as String? {
            case "nJ": return value * 1e-9
            case "uJ", "µJ": return value * 1e-6
            case "mJ": return value * 1e-3
            default: return nil
            }
        }
        return nil
    }

    /// Waits `seconds` and measures average GPU power (W). Call before training to spot other
    /// processes using the GPU.
    func idleWatts(seconds: Double = 1.0) -> Double? {
        guard let j0 = joules() else { return nil }
        let t0 = Date()
        Thread.sleep(forTimeInterval: seconds)
        guard let j1 = joules() else { return nil }
        return (j1 - j0) / Date().timeIntervalSince(t0)
    }
}

#endif
