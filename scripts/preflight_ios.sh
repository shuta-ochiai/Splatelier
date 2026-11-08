#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Preflight checks before deploying to an iPhone 16 Pro. Checks settings / Xcode / connection / code
# state together to clear blockers up front.

set -uo pipefail   # -e dropped: each check runs independently

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XCODEPROJ="$PROJECT_DIR/App/Splatelier.xcodeproj"
PASS=0
WARN=0
FAIL=0

ok()    { echo "  ✅ $*"; PASS=$((PASS + 1)); }
warn()  { echo "  ⚠️  $*"; WARN=$((WARN + 1)); }
fail()  { echo "  ❌ $*"; FAIL=$((FAIL + 1)); }
info()  { echo "  ℹ️  $*"; }

echo "=== iPhone 16 Pro deployment preflight ==="
echo ""

# 1. Xcode version
echo "[1/8] Xcode"
xcode_ver=$(xcodebuild -version 2>&1 | head -1)
if [[ -n "$xcode_ver" && "$xcode_ver" == Xcode* ]]; then
    ver_num="${xcode_ver#Xcode }"
    major="${ver_num%%.*}"
    if [[ "$major" -ge 16 ]]; then
        ok "$xcode_ver"
    else
        warn "$xcode_ver — Xcode 16+ recommended (iOS 18 SDK required)"
    fi
else
    fail "xcodebuild does not run: $xcode_ver"
fi

# 2. iOS SDK
echo "[2/8] iOS SDK"
if ios_sdk=$(xcrun --sdk iphoneos --show-sdk-version 2>/dev/null); then
    if [[ "${ios_sdk%%.*}" -ge 18 ]]; then
        ok "iOS SDK $ios_sdk"
    else
        fail "iOS SDK $ios_sdk — iOS 18+ required (Package.swift specifies .iOS(.v18))"
    fi
else
    fail "iOS SDK not installed"
fi

# 3. Project file
echo "[3/8] Xcode project"
if [[ -d "$XCODEPROJ" ]]; then
    ok "$XCODEPROJ"
else
    fail "$XCODEPROJ not found"
fi

# 4-5. Team and bundle ID, from App/Config/Local.xcconfig (ignored by git; see the .example)
LOCAL_XCCONFIG="$PROJECT_DIR/App/Config/Local.xcconfig"
local_setting() {
    grep -E "^[[:space:]]*$1[[:space:]]*=" "$LOCAL_XCCONFIG" 2>/dev/null | tail -1 \
        | sed -E 's/^[^=]*=[[:space:]]*//; s/[[:space:]]*$//'
}
echo "[4/8] Bundle ID"
bid=$(local_setting PRODUCT_BUNDLE_IDENTIFIER)
if [[ ! -f "$LOCAL_XCCONFIG" ]]; then
    fail "App/Config/Local.xcconfig not found"
    info "cp App/Config/Local.xcconfig.example App/Config/Local.xcconfig, then set your team and bundle ID in it"
elif [[ -z "$bid" || "$bid" == "com.example.splatelier" ]]; then
    fail "PRODUCT_BUNDLE_IDENTIFIER is not set to your own in App/Config/Local.xcconfig"
else
    ok "Bundle ID = $bid"
fi

# 5. DEVELOPMENT_TEAM (automatic signing fails if empty)
echo "[5/8] Signing team"
team=$(local_setting DEVELOPMENT_TEAM)
if [[ -z "$team" || "$team" == "ABCDE12345" ]]; then
    warn "DEVELOPMENT_TEAM is not set in App/Config/Local.xcconfig"
    info "The 10-character team ID is shown in Xcode → Settings → Accounts"
else
    ok "DEVELOPMENT_TEAM = $team"
fi

# 6. Connected devices
echo "[6/8] Connected device"
if devices=$(xcrun devicectl list devices 2>/dev/null | grep -E "iPhone|connected" | grep -v "unavailable"); then
    if echo "$devices" | grep -qi "iPhone 16 Pro"; then
        ok "iPhone 16 Pro detected"
        echo "$devices" | grep -i "iPhone 16 Pro" | sed 's/^/      /'
    elif echo "$devices" | grep -qi "iPhone"; then
        warn "iPhone connected but not a 16 Pro:"
        echo "$devices" | grep -i "iPhone" | sed 's/^/      /'
    else
        warn "No iPhone connected (check USB or wired connection)"
    fi
else
    info "devicectl not used — check the connection in Xcode's Devices and Simulators (⌘⇧⌥ 2)"
fi

# 7. Camera/ARKit in Info.plist
echo "[7/8] Info.plist (Camera + ARKit)"
plist="$PROJECT_DIR/App/Splatelier/Info.plist"
if grep -q "NSCameraUsageDescription" "$plist" 2>/dev/null; then
    ok "NSCameraUsageDescription present"
else
    fail "NSCameraUsageDescription missing from Info.plist → crash when ARKit starts"
fi
if grep -q "<string>arkit</string>" "$plist" 2>/dev/null; then
    ok "UIRequiredDeviceCapabilities = arkit"
else
    warn "arkit missing from UIRequiredDeviceCapabilities"
fi

# 8. Local package resolution
echo "[8/8] Swift Package resolution"
build_out=$(cd "$PROJECT_DIR" && swift build --target Trainer 2>&1 | tail -3)
if echo "$build_out" | grep -qE "Build (complete|of target)"; then
    ok "macOS swift build OK (packages resolved)"
else
    warn "macOS swift build has problems: $(echo "$build_out" | tr '\n' ' ')"
    info "Open Xcode once and run File → Packages → Resolve Package Versions"
fi

echo ""
echo "=== Result: $PASS passed, $WARN warnings, $FAIL failures ==="
if [[ $FAIL -gt 0 ]]; then
    echo "Blockers found. Fix the failures above."
    exit 1
elif [[ $WARN -gt 0 ]]; then
    echo "Warnings found. Check manually in the Xcode UI."
    exit 0
else
    echo "Preflight OK. Press ⌘+R in Xcode and the app should launch on the iPhone 16 Pro."
    exit 0
fi
