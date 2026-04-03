#!/usr/bin/env bash
# Phase 1: ns-3.40 native build on macOS (CMake + ninja + clang).
# Sources live outside this repo; override location with FSO_TOOLS_DIR.
set -euo pipefail

FSO_TOOLS_DIR="${FSO_TOOLS_DIR:-$HOME/fso-tools}"
NS3_DIR="$FSO_TOOLS_DIR/ns-3-dev"
NS3_VERSION="ns-3.40"

command -v brew >/dev/null || { echo "Homebrew required: https://brew.sh"; exit 1; }
brew list cmake &>/dev/null || brew install cmake
brew list ninja &>/dev/null || brew install ninja
brew list python@3.11 &>/dev/null || brew install python@3.11

# ns-3.40's ./ns3 wrapper breaks on Python >= 3.14 (argparse change), and
# ns3-ai's Python bindings want <= 3.11, so the whole ns-3 toolchain is
# pinned to 3.11.
NS3_PYTHON="$(brew --prefix python@3.11)/bin/python3.11"

mkdir -p "$FSO_TOOLS_DIR"
if [ ! -d "$NS3_DIR" ]; then
    git clone --branch "$NS3_VERSION" --depth 1 https://gitlab.com/nsnam/ns-3-dev.git "$NS3_DIR"
fi

# ns-3.40's custom pair operator== is ambiguous under newer libc++ (fixed
# upstream after 3.41); backport by deleting it. Idempotent.
PATCH="$(cd "$(dirname "$0")" && pwd)/patches/ns3-3.40-libcxx-pair-eq.patch"
if git -C "$NS3_DIR" apply --check "$PATCH" 2>/dev/null; then
    git -C "$NS3_DIR" apply "$PATCH"
fi

cd "$NS3_DIR"
"$NS3_PYTHON" ./ns3 configure --build-profile=optimized --enable-examples --enable-tests \
    -- -DPython_EXECUTABLE="$NS3_PYTHON" -DPython3_EXECUTABLE="$NS3_PYTHON"
"$NS3_PYTHON" ./ns3 build

"$NS3_PYTHON" ./ns3 run hello-simulator
echo "OK: $NS3_VERSION built at $NS3_DIR"
