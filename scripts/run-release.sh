#!/usr/bin/env bash
# Wrapper that runs the release-build CLI on the Mac.
#
# Problem: `swift build -c release` does not emit MLX's metallib (mlx.metallib) into the release
# products directory (the debug build does; SwiftPM's placement of Metal
# compile output depends on the config). MLX reads mlx.metallib next to the binary at runtime,
# so without it the CLI dies with "Failed to load the default metallib".
#
# Fix: build release, and if the metallib is missing, copy it from the debug build (which generates it)
# before exec'ing the CLI. Arguments and environment variables are passed through to the CLI unchanged.
#
# Usage:
#   scripts/run-release.sh train-capture foo.valset /tmp/out.ply 1
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[run-release] building release…" >&2
swift build -c release >&2
REL_BIN="$(swift build -c release --show-bin-path)"

if [ ! -f "$REL_BIN/mlx.metallib" ]; then
    echo "[run-release] mlx.metallib missing in release; building debug to source it…" >&2
    swift build >&2
    DBG_BIN="$(swift build --show-bin-path)"
    if [ -f "$DBG_BIN/mlx.metallib" ]; then
        cp "$DBG_BIN/mlx.metallib" "$REL_BIN/"
        echo "[run-release] copied mlx.metallib → $REL_BIN/" >&2
    else
        echo "[run-release] WARNING: mlx.metallib not found in debug build ($DBG_BIN);" >&2
        echo "[run-release]          MLX will likely fail to load its metallib." >&2
    fi
fi

exec "$REL_BIN/splatelier" "$@"
