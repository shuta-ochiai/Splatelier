// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "splatelier",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
        .visionOS(.v2),
    ],
    products: [
        .library(
            name: "Trainer",
            targets: ["Trainer"]
        ),
        .library(
            name: "Capture",
            targets: ["Capture"]
        ),
        // On-device SwiftUI viewer for trained PLYs. Wraps MetalSplatter's
        // real-time renderer so the app target depends on a single thin product
        // (keeps MetalSplatter out of the training library and tests).
        .library(
            name: "SplatViewerKit",
            targets: ["SplatViewerKit"]
        ),
        .executable(
            name: "splatelier",
            targets: ["CLI"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.25.3"),
        // Real-time Gaussian-splat viewer (MIT). Used by the CLI's `view` subcommand,
        // the live-training window and SplatViewerKit, not the training library.
        // Vendored v1.0.1 with a patch that blends in sRGB space on non-sRGB targets
        // (matches training compositing); see Vendor/MetalSplatter/VENDORED.md.
        .package(path: "Vendor/MetalSplatter"),
    ],
    targets: [
        .target(
            name: "Trainer",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLinalg", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            resources: [
                .copy("Resources/Slang"),
            ]
        ),
        .executableTarget(
            name: "CLI",
            dependencies: [
                "Trainer",
                // To reload a validation dataset (.valset = CapturedScene) on the Mac and retrain
                // it (the `train-capture` command). Capture guards ARKit with #if canImport, so it
                // builds on macOS too (as a no-op).
                "Capture",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MetalSplatter", package: "MetalSplatter"),
                .product(name: "SplatIO", package: "MetalSplatter"),
            ]
        ),
        // ARKit-based on-device capture (iOS-only). The real code is guarded by `#if
        // canImport(ARKit)`, so it still compiles for macOS / visionOS (as a no-op). Ingests the
        // PocketGS Sec. III-A inputs (RGB + coarse pose) from ARSession and builds an in-memory
        // scene equivalent to ColmapScene so it can be passed to TrainingDriver.
        .target(
            name: "Capture",
            dependencies: [
                "Trainer",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        // Thin SwiftUI + MetalKit viewer built on MetalSplatter's SplatRenderer.
        // Loads any SplatIO-readable file (our .ply, plus .splat/.spz) and shows
        // it with an orbit camera. iOS/macOS only (gated in-source); no MLX dep.
        .target(
            name: "SplatViewerKit",
            dependencies: [
                .product(name: "MetalSplatter", package: "MetalSplatter"),
                .product(name: "SplatIO", package: "MetalSplatter"),
            ]
        ),
        .testTarget(
            name: "TrainerTests",
            dependencies: ["Trainer"]
        ),
        .testTarget(
            name: "CaptureTests",
            dependencies: ["Capture", "Trainer"]
        ),
        .testTarget(
            name: "SplatViewerKitTests",
            dependencies: [
                "SplatViewerKit", "Trainer",
                .product(name: "MetalSplatter", package: "MetalSplatter"),
                .product(name: "SplatIO", package: "MetalSplatter"),
            ]
        ),
    ],
    swiftLanguageModes: [.v6]
)
