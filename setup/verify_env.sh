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

check "python3 >= 3.10"      python3 -c 'import sys; assert sys.version_info >= (3, 10)'
check "numpy/scipy/matplotlib" python3 -c 'import numpy, scipy, matplotlib'
check "GNU Radio 3.10"       python3 -c 'from gnuradio import gr; assert gr.version().startswith("3.10")'
check "cmake"                cmake --version
check "ninja"                ninja --version
check "ns-3.40 checkout"     git -C "$NS3_DIR" describe --tags --exact-match
check "ns-3 built"           test -d "$NS3_DIR/build"
check "hello-simulator runs" "$NS3_DIR/ns3" run hello-simulator
check "ns3-ai module"        test -d "$NS3_DIR/contrib/ai"

exit $fail
