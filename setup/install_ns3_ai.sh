#!/usr/bin/env bash
# Phase 1: ns3-ai contrib module + macOS shared-memory smoke test.
# ns3-ai is Linux-first; this script is the early compatibility check
# flagged in plan.md. Run install_ns3.sh first.
set -euo pipefail

FSO_TOOLS_DIR="${FSO_TOOLS_DIR:-$HOME/fso-tools}"
NS3_DIR="$FSO_TOOLS_DIR/ns-3-dev"

[ -d "$NS3_DIR" ] || { echo "ns-3 not found at $NS3_DIR — run install_ns3.sh first"; exit 1; }

brew list boost &>/dev/null || brew install boost
brew list protobuf &>/dev/null || brew install protobuf
brew list pybind11 &>/dev/null || brew install pybind11

NS3_PYTHON="$(brew --prefix python@3.11)/bin/python3.11"

if [ ! -d "$NS3_DIR/contrib/ai" ]; then
    git clone https://github.com/hust-diangroup/ns3-ai.git "$NS3_DIR/contrib/ai"
fi

# The multi-bss example needs ns-3.41's MakeEnumAccessor<T>; skip it on 3.40
# (hust-diangroup/ns3-ai#112).
sed -i '' 's|^add_subdirectory(multi-bss)|# add_subdirectory(multi-bss)  # requires ns-3.41|' \
    "$NS3_DIR/contrib/ai/examples/CMakeLists.txt"

cd "$NS3_DIR"
"$NS3_PYTHON" ./ns3 configure --build-profile=optimized --enable-examples --enable-tests \
    -- -DPython_EXECUTABLE="$NS3_PYTHON" -DPython3_EXECUTABLE="$NS3_PYTHON"
"$NS3_PYTHON" ./ns3 build ai

VENV_DIR="$FSO_TOOLS_DIR/ns3ai-venv"
[ -d "$VENV_DIR" ] || "$NS3_PYTHON" -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install -q -e "$NS3_DIR/contrib/ai/python_utils"
"$VENV_DIR/bin/pip" install -q -e "$NS3_DIR/contrib/ai/model/gym-interface/py"

echo "OK: ns3-ai module built, python packages in $VENV_DIR"
echo "Run a shared-memory example (contrib/ai/examples) to confirm it works"
echo "end-to-end on macOS before relying on it for Phase 4."
