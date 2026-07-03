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

mkdir -p "$FSO_TOOLS_DIR"
if [ ! -d "$NS3_DIR" ]; then
    git clone --branch "$NS3_VERSION" --depth 1 https://gitlab.com/nsnam/ns-3-dev.git "$NS3_DIR"
fi

cd "$NS3_DIR"
./ns3 configure --build-profile=optimized --enable-examples --enable-tests
./ns3 build

./ns3 run hello-simulator
echo "OK: $NS3_VERSION built at $NS3_DIR"
