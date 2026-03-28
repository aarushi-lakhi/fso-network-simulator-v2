# Environment setup (macOS)

Installs the Phase 1 toolchain: GNU Radio 3.10, ns-3.40, and the ns3-ai contrib module.
ns-3 sources go to `~/fso-tools/ns-3-dev` (override with `FSO_TOOLS_DIR`).

Run in order:

```bash
./install_gnuradio.sh   # brew install gnuradio, verify import
./install_ns3.sh        # clone + build ns-3.40, run hello-simulator
./install_ns3_ai.sh     # add ns3-ai contrib module (Linux-first — macOS check)
./verify_env.sh         # PASS/FAIL summary of everything above
```

If ns3-ai's shared memory misbehaves on macOS, the fallback is a Docker or Lima
Ubuntu 22.04 environment (see plan.md decision log, 2026-07-03).
