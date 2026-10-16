# splatelier convenience targets.
#
# SwiftPM (command-line `swift test`) cannot build mlx-swift's Metal shaders,
# so anything that exercises the GPU has to go through xcodebuild. Plain
# compilation and the CLI run fine under `swift build` / `swift run`.
# Override DEVELOPER_DIR if your Xcode lives elsewhere.

DEVELOPER_DIR ?= /Applications/Xcode.app/Contents/Developer
export DEVELOPER_DIR

SCHEME ?= splatelier-Package
DESTINATION ?= platform=macOS

.PHONY: build test test-cli run clean

build:
	swift build

# Full test suite via xcodebuild (required for any MLX GPU kernels).
# Filtered output: only swift-testing markers, errors, and the final verdict.
test:
	@bash -c 'set -o pipefail; \
		xcodebuild test -scheme $(SCHEME) -destination "$(DESTINATION)" 2>&1 \
		| grep -E "✘|✔|◇|TEST (SUCCEEDED|FAILED)|^[^ ].*error: "'

# Lightweight smoke (build only, no GPU). For Cmd-line iteration.
test-cli:
	swift test --filter versionStringFormat

run:
	swift run splatelier

clean:
	swift package clean
	rm -rf .build
	rm -rf $(HOME)/Library/Developer/Xcode/DerivedData/splatelier-*
