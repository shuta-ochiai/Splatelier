# Splatelier

<p align="center">
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
  <a href="https://github.com/shuta-ochiai/Splatelier/tags"><img alt="Version" src="https://img.shields.io/github/v/tag/shuta-ochiai/Splatelier?label=version"></a>
  <img alt="Platform: iOS 18+" src="https://img.shields.io/badge/platform-iOS%2018%2B-blue">
  <img alt="Swift 6.0" src="https://img.shields.io/badge/Swift-6.0-orange?logo=swift&logoColor=white">
</p>

**Capture, train and view 3D Gaussian Splats entirely on your iPhone.**

Splatelier (from *splat* + *atelier*) turns a short walk around an object or a room into a 3D
Gaussian Splatting scene. Capture, camera pose refinement, training and viewing all run on the
device: no server, no upload, no desktop GPU.

Under the hood, the trainer is written in Swift on [MLX-Swift](https://github.com/ml-explore/mlx-swift),
and the differentiable rendering kernels are written in [Slang](https://github.com/shader-slang/slang)
and compiled to Metal.

> **Status:** early release. The app is built from source with Xcode and is not on the App Store.
> Expect rough edges, and expect settings and file formats to change.

## Features

- **Guided capture**: ARKit images and poses, blur and exposure filtering while you capture, and a
  live splat preview that shows which parts of the scene are covered.
- **On-device training**: bundle adjustment refines the ARKit poses, multi-view stereo provides dense
  seed points, and the Gaussians are trained on the iPhone's GPU. Training slows down when the device
  gets hot and pauses when iOS reports a critical thermal state. A finished model can be trained
  further.
- **Built-in viewer**: orbit the result, hide the background, and measure distances between two
  points.
- **Export**: standard 3DGS PLY files that common splat viewers can open, and a mesh (OBJ) generated
  from the trained scene. Files are available in the Files app.
- **Mac tools**: retrain a capture from the app on a Mac, train and evaluate on COLMAP scenes, and
  inspect PLY files from the command line.

## Requirements

- An iPhone or iPad with iOS 18 or later. ARKit capture and training need a real device. LiDAR
  is not required.
- An Apple Silicon Mac with macOS 15 or later and Xcode 16 (iOS 18 SDK) to build the app.
- An Apple account added to Xcode as a development team, to sign the app.

## Install the app

1. Clone the repository and open the Xcode project:

   ```sh
   git clone https://github.com/shuta-ochiai/Splatelier.git
   cd Splatelier
   open App/Splatelier.xcodeproj
   ```

2. In the **Splatelier** target's *Signing & Capabilities*, choose your team and change the bundle
   identifier to one your account can use. To keep these settings when the project is regenerated,
   set `DEVELOPMENT_TEAM` and `PRODUCT_BUNDLE_IDENTIFIER` in `App/project.yml` as well (see
   [Editing the Xcode project](#editing-the-xcode-project)).
3. Turn on Developer Mode on the device (*Settings → Privacy & Security → Developer Mode*).
4. Select the **Splatelier** scheme and your device, and run. The scheme builds in Release, because
   training in Debug is much slower.

## Using the app

1. In the **Capture** tab, tap **Start capture** and walk slowly around the subject. Keep it in view
   and cover it from several heights.
2. Tap **Stop**, then **Train this capture**. Choose the number of iterations and the Gaussian cap,
   and tap **Start training**.
3. When training finishes, tap **View** to open the result. From there you can **Train more**,
   **Generate mesh** or **Export**. Trained models are listed in the **Library** tab and in the Files
   app.

Tips: even lighting and a textured subject give the best results. Shiny, transparent and plain
white surfaces are hard to reconstruct.

## Mac command-line tool

The Swift package also builds a command-line tool, `splatelier`.

```sh
make build                                                   # swift build
make test                                                    # package tests via xcodebuild
scripts/run-release.sh train-capture <dataset.valset> out.ply
```

Main commands (run `splatelier` without arguments for the full usage):

```text
train <colmap_scene_dir> <output.ply> [iterations] ...    train on a COLMAP scene
eval | compare | render                                   evaluate on a COLMAP scene
train-capture <dataset.valset> <output.ply> [iterations]  retrain an app capture with the app's
                                                          settings (MESH_OBJ=<path> also writes a mesh)
view | capture | stats | cull                             viewer and PLY tools
```

- A capture can be saved from the app as a `.valset` dataset and shared to the Mac.
- `EVAL_HOLDOUT=N` with `train-capture` holds out every Nth frame and logs PSNR.
- Train and measure with release builds. `scripts/run-release.sh` builds release, adds MLX's
  metallib and runs the tool.
- GPU tests and training need Xcode's toolchain (`xcodebuild`), not plain `swift test`, because of
  MLX's Metal shaders.
- The viewer commands (`view`, `capture`) need `scripts/build-metalsplatter-metallib.sh [debug|release]`
  first.
- The COLMAP integration tests take a local scene:
  `TEST_RUNNER_MST_COLMAP_DIR=$PWD/data/bear make test`.

## Development

### Editing the Xcode project

The Xcode project is generated from `App/project.yml` with [XcodeGen](https://github.com/yonaskolb/XcodeGen)
and checked in. Make project changes in `project.yml` and run `cd App && xcodegen generate`. Anything
set only in the `.xcodeproj` is lost on regeneration.

### Slang kernels

```sh
./scripts/install_slang.sh
./scripts/compile_slang.sh          # regenerate
./scripts/compile_slang.sh --check  # check that the checked-in output is current
```

The generated `.metal` and `.refl.json` files are checked in under `Sources/Trainer/Resources/Slang`,
so Slang is only needed to edit `.slang` kernels. Generation is deterministic, so comment-only edits
leave them unchanged.

### How it works

- **Poses and seed points**: bundle adjustment over ORB and ARKit feature tracks, and dense seed
  points from multi-view stereo (plane sweep + SGM).
- **Training**: PocketGS disc-like initialization and replay cache; a hybrid of MCMC and ImprovedGS+
  long-axis split for densification; a directional spherical-harmonics background; fused SSIM;
  coarse-to-fine; thermal throttling; continued training with the Adam state kept.
- **Output**: a standard 3DGS PLY (SH degree 1, written in the degree-3 layout so common viewers read
  it) with the background baked into a far shell of Gaussians, and a TSDF mesh.
- **Viewer**: [MetalSplatter](https://github.com/scier/MetalSplatter) with local patches: blending that
  matches training, stochastic transparency while the camera moves, a mesh overlay, hiding the
  background and distance measurement. See [VENDORED.md](Vendor/MetalSplatter/VENDORED.md).

### Repository layout

```text
App/                       iOS app; project.yml is the source of truth
Sources/Trainer/           renderer, trainer, densification, BA and MVS, PLY I/O
Sources/Capture/           ARKit capture, training from captures, .valset storage
Sources/SplatViewerKit/    SwiftUI + MetalKit viewer
Sources/CLI/               command-line tool
slang/                     Slang kernel sources
Vendor/MetalSplatter/      vendored MetalSplatter with local patches
Tests/                     unit and GPU tests
```

## Acknowledgments

Splatelier is built on [mlx-swift](https://github.com/ml-explore/mlx-swift) (MIT) and
[Slang](https://github.com/shader-slang/slang), and its viewer is a modified copy of
[MetalSplatter](https://github.com/scier/MetalSplatter) (MIT, see
[VENDORED.md](Vendor/MetalSplatter/VENDORED.md)).

We referred to the code of these projects while writing our own:

- [gsplat](https://github.com/nerfstudio-project/gsplat) (Apache-2.0): the numerical reference for
  the rasterizer, projection, spherical harmonics, Adam, MCMC densification and the PLY layout
- [splatfacto-w](https://github.com/KevinXu02/splatfacto-w) (Apache-2.0): the directional SH
  background and the alpha-loss mask
- [VkSplat](https://github.com/harry7557558/vksplat) (Apache-2.0): the 32-bit tile and depth sort key
- [libSGM](https://github.com/fixstars/libSGM) (Apache-2.0): splitting SGM aggregation across a SIMD
  group, and the ratio of the default penalties
- [COLMAP](https://github.com/colmap/colmap) (BSD-3-Clause): the radial distortion model, and the
  binary reconstruction format
- [SuperSplat](https://github.com/playcanvas/supersplat) and the
  [PlayCanvas Engine](https://github.com/playcanvas/engine) (MIT): stochastic transparency in the
  viewer

Main papers:

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079)
- [PocketGS: On-Device Training of 3D Gaussian Splatting for High Perceptual Modeling](https://arxiv.org/abs/2601.17354):
  disc-like initialization, replay cache, geometry prior
- [3D Gaussian Splatting as Markov Chain Monte Carlo](https://arxiv.org/abs/2404.09591): densification
- [ImprovedGS+](https://arxiv.org/abs/2603.08661): long-axis split
- [Splatfacto-W](https://arxiv.org/abs/2407.12306): directional SH background
- [Semi-Global Matching](https://doi.org/10.1109/TPAMI.2007.1166) and Mobile3DRecon (ISMAR 2020):
  multi-view stereo
- [Bundle Adjustment — A Modern Synthesis](https://doi.org/10.1007/3-540-44480-7_21)

<details>
<summary>All papers whose methods Splatelier implements</summary>

| Method | Paper |
|---|---|
| 3D Gaussian Splatting | Kerbl et al. 2023, ACM TOG 42(4) |
| 3D Gaussian Splatting as MCMC | Kheradmand et al. 2024 |
| EWA splatting (viewer's 2D covariance) | Zwicker et al. 2001, IEEE Visualization |
| PocketGS | arXiv 2601.17354 |
| ImprovedGS+ | arXiv 2603.08661 |
| Mobile3DRecon | Yang et al. 2020, ISMAR |
| Splatfacto-W | Xu et al. 2024 |
| Space-sweep multi-view matching | Collins 1996 |
| Semi-Global Matching | Hirschmüller 2005, 2008 |
| Census transform | Zabih & Woodfill 1994; Fife & Archibald 2013 |
| Stereo confidence measures | Hu & Mordohai 2012 |
| Bundle adjustment | Triggs et al. 2000; Solà et al. 2018 |
| FAST, BRIEF, ORB | Rosten & Drummond 2006; Calonder et al. 2010; Rublee et al. 2011 |
| Ratio test | Lowe 2004 |
| Similarity transform alignment | Umeyama 1991 |
| TSDF fusion | Curless & Levoy 1996 |
| Surfel fusion | Keller et al. 2013, 3DV |
| Marching Cubes | Lorensen & Cline 1987 |
| Gaussian Opacity Fields | Yu et al. 2024, ACM TOG 43(6) |
| Sorted Opacity Fields | Radl et al. 2025, SIGGRAPH Asia |
| Geometry-Grounded Gaussian Splatting | Zhang et al. 2026, arXiv 2601.17835 |
| SSIM | Wang et al. 2004 |
| Spherical harmonics constants | Sloan 2008 |
| Closed-form 3×3 eigenvalues | Smith 1961 |
| Blur metric (edge width) | Marziliano et al. 2002 |

</details>

## License

Apache License 2.0. See [LICENSE](LICENSE). Third-party components keep their own licenses.
