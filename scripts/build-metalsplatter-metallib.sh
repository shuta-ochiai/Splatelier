#!/bin/zsh
# Compiles the vendored MetalSplatter shaders into default.metallib and places it in the SwiftPM resource bundle.
#
# Problem: `swift build` does not compile .metal files; it only copies the sources into
# MetalSplatter_MetalSplatter.bundle. Creating a SplatRenderer from the CLI (view / capture)
# then fails in makeDefaultLibrary(bundle:). Xcode builds (the app) compile them automatically, so this is not needed there.
#
# usage: scripts/build-metalsplatter-metallib.sh [debug|release]   (default release)
set -euo pipefail

CONFIG="${1:-release}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/Vendor/MetalSplatter/MetalSplatter/Resources"
BUNDLE="$ROOT/.build/arm64-apple-macosx/$CONFIG/MetalSplatter_MetalSplatter.bundle"

if [ ! -d "$BUNDLE" ]; then
    echo "[metallib] $BUNDLE not found. Run swift build -c $CONFIG first" >&2
    exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
airs=()
for f in "$SRC"/*.metal; do
    air="$TMP/$(basename "$f" .metal).air"
    xcrun -sdk macosx metal -c -I "$SRC" "$f" -o "$air"
    airs+=("$air")
done
xcrun -sdk macosx metallib "${airs[@]}" -o "$BUNDLE/default.metallib"
echo "[metallib] wrote $BUNDLE/default.metallib" >&2
