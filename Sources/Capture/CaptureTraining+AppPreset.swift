// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import Trainer

extension CaptureTraining {
    /// The app's training configuration, fixed by a parameter sweep. `train`'s defaults are these
    /// values (with `MVSMode` for the seed), so the app only passes what changes per run, and CLI
    /// `train-capture` uses them as the defaults that its env variables override.
    public enum AppPreset {
        /// Gaussian cap (the app's picker default). Caps both the initial count and growth.
        public static let defaultGaussianCap = 300_000
        /// Initial Gaussian count for `cap`: 5/6 of it (250k of 300k), and Hybrid's edge splits grow
        /// it to the cap during the run. This matches a full-cap start in PSNR at slightly lower GPU
        /// energy; starting at 2/3 does not grow to the cap and loses PSNR.
        public static func initialGaussianCap(cap: Int) -> Int { cap * 5 / 6 }
        /// Opacity and scale regularization weight, a tenth of gsplat MCMC's 0.01: at 0.01 it held
        /// most Gaussians near transparent through a short run, so the scene was drawn by many faint
        /// layers and looked hazy.
        public static let regularization: Float = 0.001
        /// Opacity of the seed Gaussians. Starting opaque (rather than 3DGS's 0.1) lets a short run
        /// settle on opaque surfaces instead of stacks of faint discs.
        public static let seedOpacity: Float = 0.5
        /// Weight λ of the D-SSIM term. Above 3DGS's 0.2 it keeps more of the local contrast and
        /// edges, at about 0.1 dB of held-out PSNR.
        public static let lambdaSSIM: Float = 0.5
        /// Fragment slots per pixel of the replay cache (a margin covering the overlap at the center
        /// pixel). The replay cache itself is always on as a thermal measure: its backward is
        /// slower than coop backward but memory-bandwidth bound and low power.
        public static let replayKMax = 96
        /// Fraction of iterations in the coarse phase.
        public static let coarseToFineFraction: Float = 0.5
        /// Coarse-phase downscale relative to the capture: 1/3 = 360×270, half the fine resolution.
        public static let coarseToFineFactor = 3
        /// Fine-phase resolution: 2/3 of the capture = 720×540. It matches full resolution in PSNR
        /// at well under half the energy.
        public static let fineResample: (num: Int, den: Int) = (2, 3)
        /// Geometric regularization (G³S surface depth with normal consistency and the MVS depth,
        /// every 8 iterations from 30% of training), so the Gaussians' surface depth gives a flat
        /// TSDF mesh (generated after training in the app), at a small cost in GPU energy and held-out PSNR.
        public static let geometryRegularization: GeometryRegularization? = GeometryRegularization()
        /// A gain per frame on the render (exposure and white balance), fitted from 30% of training
        /// on. Frames shot before the exposure lock, or under a changing light, then do not pull
        /// the scene's colours; with a locked exposure the gains stay near 1.
        public static let photometricGains: PhotometricGainConfig? = PhotometricGainConfig()
        /// The app's mesh after training: 5mm voxels, at most 12 million allocated (≈ 145MB, to fit device memory;
        /// beyond that the voxel size is widened automatically).
        public static let meshParameters = CaptureMeshing.Parameters(
            tsdf: TSDFFusion.Parameters(voxelSize: 0.005, maxVoxels: 12_000_000, voxelsPerDispatch: 1_000_000))

        /// When Hybrid densifies (its IGS+ part).
        public struct Schedule: Sendable, Equatable {
            public let refineStartIter: Int
            public let refineStopIter: Int
            public let refineEvery: Int

            public var approximateFireCount: Int {
                guard refineEvery > 0, refineStopIter > refineStartIter else { return 0 }
                return max(1, (refineStopIter - refineStartIter) / refineEvery)
            }
        }

        public static func schedule(iterations: Int) -> Schedule {
            let horizon = max(1, iterations)
            let start = max(100, horizon / 5)
            let stop = max(start + 1, (horizon * 3) / 4)
            // IGS+ LAS is deterministic and edge-guided, so it uses fewer, higher-confidence splits
            // than MCMC.
            let targetFireCount = min(16, max(8, horizon / 500 + 6))
            return Schedule(
                refineStartIter: start, refineStopIter: stop,
                refineEvery: max(25, max(1, stop - start) / targetFireCount))
        }

        /// Hybrid: IGS+ LAS growth + MCMC dead prune. The budget freed by pruning goes to edges, and
        /// bloatScaleMultiple also cleans up bloated blobs. The prune takes the faint (opacity ≤ 0.15)
        /// Gaussians larger than the median, which veil the scene, and keeps the faint small ones
        /// that draw fine detail such as text; minOpacity also keeps faint Gaussians from being
        /// split and floors the split halves' opacity. With the SH background, the coarse phase
        /// at half the fine resolution and 4 iter/frame, it beats MCMC + environment sphere in PSNR
        /// at lower energy.
        public static func strategy(iterations: Int, cap: Int) -> IGSPlusStrategy {
            let s = schedule(iterations: iterations)
            return IGSPlusStrategy(
                capMax: cap,
                refineStartIter: s.refineStartIter,
                refineStopIter: s.refineStopIter,
                refineEvery: s.refineEvery,
                minOpacity: 0.15,
                pruneDeadGaussians: true,
                bloatScaleMultiple: 100,
                faintPruneScaleMultiple: 1)
        }
    }
}
