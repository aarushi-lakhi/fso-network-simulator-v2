#!/usr/bin/env bash
# Phase 1: check every tool the simulator needs; prints PASS/FAIL per item.
set -uo pipefail

FSO_TOOLS_DIR="${FSO_TOOLS_DIR:-$HOME/fso-tools}"
NS3_DIR="$FSO_TOOLS_DIR/ns-3-dev"
fail=0

check() {
    local name="$1"; shift
    if "$@" &>/dev/null; then
        echo "PASS  $name"
    else
        echo "FAIL  $name"
        fail=1
    fi
}

NS3_PYTHON="$(brew --prefix python@3.11 2>/dev/null)/bin/python3.11"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

check "python3 >= 3.10"      python3 -c 'import sys; assert sys.version_info >= (3, 10)'
check "prototype venv deps"  "$REPO_ROOT/prototype/.venv/bin/python" -c 'import numpy, scipy, matplotlib'
check "GNU Radio 3.10"       python3 -c 'from gnuradio import gr; assert gr.version().startswith("3.10")'
check "cmake"                cmake --version
check "ninja"                ninja --version
check "python3.11 (ns-3 toolchain)" "$NS3_PYTHON" --version
check "ns-3.40 checkout"     sh -c "git -C '$NS3_DIR' tag --points-at HEAD | grep -qx ns-3.40"
check "ns-3 built"           test -d "$NS3_DIR/build"
check "hello-simulator runs" "$NS3_PYTHON" "$NS3_DIR/ns3" run hello-simulator
check "ns3-ai module"        test -d "$NS3_DIR/contrib/ai"
check "ns3-ai python pkgs"   "$FSO_TOOLS_DIR/ns3ai-venv/bin/python" -c 'import ns3ai_utils'

exit $fail
