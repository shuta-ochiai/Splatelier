#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Downloads a Slang compiler release into ./tools/slang-<version>/ and prints
# the path to slangc. Idempotent: skips download if already present.
#
# Override the version: SLANG_VERSION=2026.9.1 ./scripts/install_slang.sh

set -euo pipefail

SLANG_VERSION="${SLANG_VERSION:-2026.9.1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TOOLS_DIR="$PROJECT_ROOT/tools"
SLANG_DIR="$TOOLS_DIR/slang-$SLANG_VERSION"

if [ -x "$SLANG_DIR/bin/slangc" ]; then
    echo "slangc already installed: $SLANG_DIR/bin/slangc"
    exit 0
fi

ARCH="$(uname -m)"
case "$ARCH" in
    arm64|aarch64) SLANG_ARCH="aarch64" ;;
    x86_64) SLANG_ARCH="x86_64" ;;
    *) echo "Unsupported architecture: $ARCH" >&2; exit 1 ;;
esac

ARCHIVE="slang-${SLANG_VERSION}-macos-${SLANG_ARCH}.tar.gz"
URL="https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/${ARCHIVE}"

mkdir -p "$TOOLS_DIR"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading: $URL"
curl -fsSL "$URL" -o "$TMP/$ARCHIVE"

echo "Extracting to: $SLANG_DIR"
mkdir -p "$SLANG_DIR"
tar -xzf "$TMP/$ARCHIVE" -C "$SLANG_DIR"

if [ ! -x "$SLANG_DIR/bin/slangc" ]; then
    echo "Extraction did not produce bin/slangc; archive layout may have changed." >&2
    ls -la "$SLANG_DIR" >&2
    exit 1
fi

echo "Installed slangc $SLANG_VERSION at: $SLANG_DIR/bin/slangc"
"$SLANG_DIR/bin/slangc" -v 2>&1 | head -3 || true
