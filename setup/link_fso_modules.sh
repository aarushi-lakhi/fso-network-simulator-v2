#!/usr/bin/env bash
# Phase 4a: link this repo's ns-3 pieces into the ns-3 tree and build them.
# Symlinks ns3-fso-channel -> contrib/fso-channel and ns3-rl-router/sim ->
# scratch/fso-rl-env, then reconfigures and builds. Idempotent; pass
# --unlink to remove the symlinks and restore the default tree (ai only).
set -euo pipefail

FSO_TOOLS_DIR="${FSO_TOOLS_DIR:-$HOME/fso-tools}"
NS3_DIR="$FSO_TOOLS_DIR/ns-3-dev"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
NS3_PYTHON="$(brew --prefix python@3.11)/bin/python3.11"

[ -d "$NS3_DIR" ] || { echo "ns-3 not found at $NS3_DIR — run install_ns3.sh first"; exit 1; }
[ -d "$NS3_DIR/contrib/ai" ] || { echo "ns3-ai not found — run install_ns3_ai.sh first"; exit 1; }

reconfigure() {
    cd "$NS3_DIR"
    "$NS3_PYTHON" ./ns3 configure --build-profile=optimized --enable-examples --enable-tests \
        -- -DPython_EXECUTABLE="$NS3_PYTHON" -DPython3_EXECUTABLE="$NS3_PYTHON"
}

if [ "${1:-}" = "--unlink" ]; then
    rm -f "$NS3_DIR/contrib/fso-channel" "$NS3_DIR/scratch/fso-rl-env"
    reconfigure
    # Build the ai module like install_ns3_ai.sh does; a full './ns3 build'
    # would trip over ns3-ai's bundled rate-control example, which needs a
    # newer wifi API than ns-3.40 ships.
    "$NS3_PYTHON" ./ns3 build ai
    echo "OK: symlinks removed, tree restored to contrib = ai only"
    exit 0
fi

ln -sfn "$REPO_DIR/ns3-fso-channel" "$NS3_DIR/contrib/fso-channel"
ln -sfn "$REPO_DIR/ns3-rl-router/sim" "$NS3_DIR/scratch/fso-rl-env"

reconfigure
"$NS3_PYTHON" ./ns3 build fso-channel fso-rl-env

echo "OK: fso-channel and fso-rl-env linked and built"
echo "Smoke test: source $FSO_TOOLS_DIR/ns3ai-venv/bin/activate &&"
echo "            python $REPO_DIR/ns3-rl-router/sim/check_env.py --steps 10"
