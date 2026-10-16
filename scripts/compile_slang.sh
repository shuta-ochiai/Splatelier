#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Generates (.metal, .refl.json) for each compute entry in ./slang/*.slang
# into ./Sources/Trainer/Resources/Slang/.
# Entries are picked up from `void <name>(` following `[shader("compute")]`. gs_math.slang has no entry.
# # No #line directives are emitted (keeps output machine-independent, so comment-only edits produce no diff).
# Stale outputs whose source is gone are deleted.
#
#  ./scripts/compile_slang.sh          regenerate
#  ./scripts/compile_slang.sh --check  check that regenerated output matches the committed output (writes nothing)

set -euo pipefail

SLANG_VERSION="${SLANG_VERSION:-2026.9.1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SLANGC="$PROJECT_ROOT/tools/slang-$SLANG_VERSION/bin/slangc"
SRC_DIR="$PROJECT_ROOT/slang"
OUT_DIR="$PROJECT_ROOT/Sources/Trainer/Resources/Slang"

CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1

if [ ! -x "$SLANGC" ]; then
    echo "slangc not found at $SLANGC" >&2
    echo "Run: ./scripts/install_slang.sh" >&2
    exit 1
fi

GEN_DIR="$(mktemp -d)"
trap 'rm -rf "$GEN_DIR"' EXIT

discover_entries() {
    awk '
        /^[[:space:]]*\/\// { next }
        /\[shader\("compute"\)\]/ { armed = 1; next }
        armed && /void[[:space:]]+[A-Za-z_][A-Za-z0-9_]*[[:space:]]*\(/ {
            match($0, /void[[:space:]]+[A-Za-z_][A-Za-z0-9_]*/)
            name = substr($0, RSTART, RLENGTH)
            sub(/^void[[:space:]]+/, "", name)
            print name
            armed = 0
        }
    ' "$1"
}

shopt -s nullglob
for src in "$SRC_DIR"/*.slang; do
    base=$(basename "$src" .slang)
    [ "$base" = "gs_math" ] && continue

    entries=$(discover_entries "$src")
    if [ -z "$entries" ]; then
        echo "warning: no compute entries found in $base.slang; skipping" >&2
        continue
    fi

    while IFS= read -r entry; do
        [ -z "$entry" ] && continue
        echo "  $base.slang :: $entry"
        "$SLANGC" "$src" \
            -I "$SRC_DIR" \
            -target metal \
            -line-directive-mode none \
            -entry "$entry" \
            -o "$GEN_DIR/$entry.metal" \
            -reflection-json "$GEN_DIR/$entry.refl.json"
    done <<< "$entries"
done

if [ "$CHECK" = 1 ]; then
    if diff -r -q "$GEN_DIR" "$OUT_DIR"; then
        echo "Slang outputs are up to date."
    else
        echo "Slang outputs differ from sources. Run ./scripts/compile_slang.sh" >&2
        exit 1
    fi
    exit 0
fi

mkdir -p "$OUT_DIR"
for f in "$OUT_DIR"/*.metal "$OUT_DIR"/*.refl.json; do
    [ -e "$GEN_DIR/$(basename "$f")" ] || { echo "  removing stale $(basename "$f")"; rm "$f"; }
done
cp "$GEN_DIR"/* "$OUT_DIR"/
echo "Slang compilation done. Outputs in $OUT_DIR"
