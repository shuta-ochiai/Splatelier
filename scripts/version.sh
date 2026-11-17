#!/usr/bin/env bash
# The release version lives in two files: Splatelier.version (the Swift package: the CLI and the
# version a trained PLY records) and MARKETING_VERSION in App/Config/Version.xcconfig (the app's
# CFBundleShortVersionString). This script writes both and checks that they agree.
#
#   scripts/version.sh                 print the version
#   scripts/version.sh set 0.7.0       write 0.7.0 to both files
#   scripts/version.sh check [v0.7.0]  fail unless both files agree (and match the tag, if given)
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
swift_file="$root/Sources/Trainer/Splatelier.swift"
xcconfig="$root/App/Config/Version.xcconfig"

swift_version() {
    sed -n 's/^ *public static let version = "\([^"]*\)".*/\1/p' "$swift_file"
}

xcconfig_version() {
    sed -n 's/^MARKETING_VERSION = \(.*\)$/\1/p' "$xcconfig"
}

# Replaces the lines matching $2 in file $1 with sed expression $3 (a temporary file, since BSD
# and GNU sed take -i differently), and fails unless exactly one line matched.
rewrite() {
    local file="$1" pattern="$2" expr="$3"
    [ "$(grep -c "$pattern" "$file")" = 1 ] || { echo "error: expected one '$pattern' line in $file" >&2; exit 1; }
    sed "$expr" "$file" > "$file.tmp"
    mv "$file.tmp" "$file"
}

case "${1:-}" in
"")
    swift_version
    ;;
set)
    new="${2:-}"
    [[ "$new" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "usage: $0 set MAJOR.MINOR.PATCH" >&2; exit 2; }
    rewrite "$swift_file" '^ *public static let version = "' \
        "s/^\( *public static let version = \"\)[^\"]*\"/\1$new\"/"
    rewrite "$xcconfig" '^MARKETING_VERSION = ' "s/^MARKETING_VERSION = .*/MARKETING_VERSION = $new/"
    "$0" check
    ;;
check)
    s="$(swift_version)" x="$(xcconfig_version)"
    [ -n "$s" ] || { echo "error: no version in $swift_file" >&2; exit 1; }
    if [ "$s" != "$x" ]; then
        echo "error: Splatelier.version is '$s' but Version.xcconfig's MARKETING_VERSION is '$x'" >&2
        echo "       (scripts/version.sh set X.Y.Z writes both)" >&2
        exit 1
    fi
    if [ -n "${2:-}" ] && [ "${2#v}" != "$s" ]; then
        echo "error: the tag is $2 but the version in the code is $s" >&2
        exit 1
    fi
    echo "version $s${2:+ (tag $2)}"
    ;;
*)
    echo "usage: $0 [set X.Y.Z | check [vX.Y.Z]]" >&2
    exit 2
    ;;
esac
