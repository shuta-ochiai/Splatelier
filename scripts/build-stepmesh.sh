#!/bin/zsh
# Builds native/stepmesh (STEP to USDZ through cadmpeg, Rust) into StepMesh.xcframework, the static
# library SplatViewerKit links for STEP import. Slices: macOS arm64, iOS arm64, iOS Simulator arm64.
#
# Needs the Rust toolchain named below with the aarch64-apple-ios and aarch64-apple-ios-sim targets:
#   rustup toolchain install 1.97.1 && rustup +1.97.1 target add aarch64-apple-ios aarch64-apple-ios-sim
#
# usage: scripts/build-stepmesh.sh
#   STEPMESH_TARGETS="aarch64-apple-darwin aarch64-apple-ios-sim" builds only those slices (CI, which
#   builds for the simulator only), and STEPMESH_PROFILE=ci builds without LTO (Cargo.toml).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CRATE="$ROOT/native/stepmesh"
TOOLCHAIN="${STEPMESH_RUST_TOOLCHAIN:-1.97.1}"
PROFILE="${STEPMESH_PROFILE:-release}"
OUT="$CRATE/StepMesh.xcframework"
TARGETS=(${=STEPMESH_TARGETS:-aarch64-apple-darwin aarch64-apple-ios aarch64-apple-ios-sim})

# Match the package's platform floors so the linker does not warn about newer object files.
export MACOSX_DEPLOYMENT_TARGET=15.0
export IPHONEOS_DEPLOYMENT_TARGET=18.0

cd "$CRATE"
for target in $TARGETS; do
    echo "[stepmesh] cargo build --profile $PROFILE --target $target"
    cargo "+$TOOLCHAIN" build --profile "$PROFILE" --lib --target "$target"
done

HEADERS="$CRATE/target/xcframework-headers"
rm -rf "$HEADERS" "$OUT"
mkdir -p "$HEADERS"
cp "$CRATE/include/stepmesh.h" "$HEADERS/"
cat > "$HEADERS/module.modulemap" <<'MAP'
module StepMesh {
    header "stepmesh.h"
    export *
}
MAP

args=()
for target in $TARGETS; do
    args+=(-library "$CRATE/target/$target/$PROFILE/libstepmesh.a" -headers "$HEADERS")
done
xcodebuild -create-xcframework $args -output "$OUT"
echo "[stepmesh] wrote $OUT"
