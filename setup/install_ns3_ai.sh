#!/usr/bin/env bash
# Phase 1: ns3-ai contrib module + macOS shared-memory smoke test.
# ns3-ai is Linux-first; this script is the early compatibility check
# flagged in plan.md. Run install_ns3.sh first.
set -euo pipefail

FSO_TOOLS_DIR="${FSO_TOOLS_DIR:-$HOME/fso-tools}"
NS3_DIR="$FSO_TOOLS_DIR/ns-3-dev"

[ -d "$NS3_DIR" ] || { echo "ns-3 not found at $NS3_DIR — run install_ns3.sh first"; exit 1; }

brew list protobuf &>/dev/null || brew install protobuf
brew list pybind11 &>/dev/null || brew install pybind11

if [ ! -d "$NS3_DIR/contrib/ai" ]; then
    git clone https://github.com/hust-diangroup/ns3-ai.git "$NS3_DIR/contrib/ai"
fi

cd "$NS3_DIR"
./ns3 configure --build-profile=optimized --enable-examples --enable-tests
./ns3 build ai

echo "OK: ns3-ai module built — run an example (e.g. contrib/ai/examples) to confirm"
echo "shared memory works end-to-end on macOS before relying on it for Phase 4."
