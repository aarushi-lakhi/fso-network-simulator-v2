#!/usr/bin/env bash
# Phase 1: GNU Radio 3.10 on macOS via Homebrew.
set -euo pipefail

command -v brew >/dev/null || { echo "Homebrew required: https://brew.sh"; exit 1; }

brew list gnuradio &>/dev/null || brew install gnuradio

python3 - <<'EOF'
from gnuradio import gr
version = gr.version()
print(f"GNU Radio {version}")
assert version.startswith("3.10"), f"Expected GNU Radio 3.10.x, got {version}"
EOF

echo "OK: GNU Radio installed and importable"
