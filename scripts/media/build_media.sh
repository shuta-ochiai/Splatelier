#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Renders the app's animations and writes them where the app bundles them:
#
#   launch  App/Splatelier/launch.mp4          shown after the OS launch screen
#   guides  App/Splatelier/Guides/guide_*.mp4  capture tips on the Capture screen
#
# Usage: scripts/media/build_media.sh [launch|guides|all]   (default: all)
#
# Needs python3 with numpy and Pillow, and ffmpeg. The frames are rendered into a temporary
# directory. Encoding is H.264, no audio, tagged BT.709 limited range; the scaler's accurate
# rounding keeps the flat background within 1-2 levels of the launch screen color (#161B46).
# Regenerate the Xcode project (cd App && xcodegen generate) only when a file is added or removed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")/App/Splatelier"
PYTHON="${PYTHON:-python3}"
TARGET="${1:-all}"

case "$TARGET" in launch|guides|all) ;; *) echo "usage: $0 [launch|guides|all]" >&2; exit 2 ;; esac
command -v ffmpeg >/dev/null || { echo "ffmpeg not found (brew install ffmpeg)" >&2; exit 1; }
"$PYTHON" -c "import numpy, PIL" 2>/dev/null \
    || { echo "$PYTHON needs numpy and Pillow (pip install numpy pillow)" >&2; exit 1; }
"$SCRIPT_DIR/fetch_fonts.sh" >/dev/null

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# encode <frames dir> <fps> <crf> <output.mp4>
encode() {
    ffmpeg -v error -y -framerate "$2" -i "$1/f%04d.png" -an \
        -vf "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p" \
        -c:v libx264 -preset slow -crf "$3" \
        -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv \
        -movflags +faststart "$4"
    echo "wrote $4 ($(du -h "$4" | cut -f1))"
}

if [ "$TARGET" = launch ] || [ "$TARGET" = all ]; then
    mkdir -p "$WORK/launch"
    "$PYTHON" "$SCRIPT_DIR/render_launch.py" "$WORK/launch"
    encode "$WORK/launch" 60 12 "$APP_DIR/launch.mp4"
fi

if [ "$TARGET" = guides ] || [ "$TARGET" = all ]; then
    mkdir -p "$APP_DIR/Guides"
    for scene in slow turn angles pivot loop light subjects cool; do
        mkdir -p "$WORK/$scene"
        "$PYTHON" "$SCRIPT_DIR/render_guides.py" "$WORK/$scene" "$scene"
        encode "$WORK/$scene" 60 16 "$APP_DIR/Guides/guide_$scene.mp4"
    done
fi
