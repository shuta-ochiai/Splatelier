// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation

/// Shared pieces of the densification strategies. Numbers follow gsplat (Apache-2.0) and
/// "3D Gaussian Splatting as Markov Chain Monte Carlo" (Kheradmand et al., 2024).
public enum Densification {

    /// Max copy count handled by the relocation correction (gsplat's `n_max`). ratio is clamped to
    /// [1, nMax].
    public static let nMax = 51

    /// Binomial table `table[n * nMax + k] = C(n, k)` (0 for k > n). Same layout as gsplat's
    /// `binoms`.
    public static func binomialTable(nMax: Int = nMax) -> [Float] {
        var table = [Float](repeating: 0, count: nMax * nMax)
        for n in 0..<nMax {
            var c = 1.0
            for k in 0...n {
                table[n * nMax + k] = Float(c)
                c = c * Double(n - k) / Double(k + 1)
            }
        }
        return table
    }

    /// Opacity/scale correction when one Gaussian is represented by `ratio` overlapping copies
    /// (MCMC paper eq. 9). Preserves the composite contribution:
    ///   o_new = 1 - (1 - o)^(1/n)
    ///   s_new = (o / denom) · s, denom = Σ_{i=1..n} Σ_{k=0..i-1} C(i-1,k) (-1)^k / √(k+1) ·
    ///   o_new^(k+1)
    /// Only a few rows at refine time, so host computation suffices.
    ///
    /// - Parameters:
    ///   - opacities: [M] activated opacities.
    ///   - scalesFlat: [M*3] linear scales (row-major).
    ///   - ratios: [M] copy counts (clamped to at least 1).
    /// - Returns: New opacities [M] and new linear scales [M*3].
    public static func computeRelocation(
        opacities: [Float],
        scalesFlat: [Float],
        ratios: [Int],
        binoms: [Float],
        nMax: Int = nMax
    ) -> (newOpacities: [Float], newScalesFlat: [Float]) {
        let m = opacities.count
        precondition(scalesFlat.count == m * 3, "scalesFlat must be [M*3]")
        precondition(ratios.count == m, "ratios must be [M]")

        var newOpacities = [Float](repeating: 0, count: m)
        var newScales = [Float](repeating: 0, count: m * 3)

        for idx in 0..<m {
            let n = min(max(ratios[idx], 1), nMax)
            let op = opacities[idx]
            let newOp = 1 - pow(1 - op, 1 / Float(n))
            newOpacities[idx] = newOp

            var denom: Float = 0
            for i in 1...n {
                for k in 0...(i - 1) {
                    let binCoeff = binoms[(i - 1) * nMax + k]
                    let sign: Float = (k % 2 == 0) ? 1 : -1
                    let term = (sign / Float(k + 1).squareRoot()) * pow(newOp, Float(k + 1))
                    denom += binCoeff * term
                }
            }
            let coeff = op / denom
            for a in 0..<3 {
                newScales[idx * 3 + a] = coeff * scalesFlat[idx * 3 + a]
            }
        }
        return (newOpacities, newScales)
    }

    /// Draws `count` indices with replacement, proportional to weight (host inverse CDF + binary
    /// search, equivalent to gsplat's `torch.multinomial`). Uniform if all weights are 0.
    public static func multinomialSample<G: RandomNumberGenerator>(
        weights: [Float], count: Int, using rng: inout G
    ) -> [Int] {
        let n = weights.count
        precondition(n > 0, "weights must be non-empty")
        var cum = [Float](repeating: 0, count: n)
        var acc: Float = 0
        for i in 0..<n {
            acc += Swift.max(0, weights[i])
            cum[i] = acc
        }
        var out = [Int](repeating: 0, count: count)
        guard acc > 0 else {
            for i in 0..<count { out[i] = Int(rng.next() % UInt64(n)) }
            return out
        }
        for i in 0..<count {
            let u = Double(rng.next() >> 11) * (1.0 / 9_007_199_254_740_992.0)  // [0,1)
            let r = Float(u) * acc
            var lo = 0, hi = n - 1
            while lo < hi {
                let mid = (lo + hi) / 2
                if cum[mid] <= r { lo = mid + 1 } else { hi = mid }
            }
            out[i] = lo
        }
        return out
    }
}

/// Deterministic RNG for host-side densification sampling. Seeded from the training seed so runs
/// are reproducible.
public struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64
    public init(seed: UInt64) { self.state = seed }
    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
