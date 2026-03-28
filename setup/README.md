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

## Python versions

Two Pythons by design: GNU Radio uses Homebrew's default `python3`, while the ns-3
toolchain is pinned to `python@3.11` — ns-3.40's `./ns3` wrapper breaks on Python ≥ 3.14
and ns3-ai's bindings want ≤ 3.11. ns3-ai's Python packages live in `~/fso-tools/ns3ai-venv`.

## ns3-ai on macOS (researched 2026-07-03)

- macOS is officially supported by ns3-ai's current `main` branch (the 2023 rewrite);
  its IPC is boost shared memory plus its own atomic spin-wait semaphore — no POSIX
  semaphores, so the usual macOS blocker doesn't apply. Expect one busy core while blocked.
- Correct clone location is `contrib/ai` (the old version used `contrib/ns3-ai`).
- On ns-3.40 the bundled `multi-bss` example doesn't compile (needs ns-3.41's
  `MakeEnumAccessor<T>`); `install_ns3_ai.sh` comments it out
  ([ns3-ai#112](https://github.com/hust-diangroup/ns3-ai/issues/112)).
- Caveat: no independent Apple Silicon reports exist — only the maintainer's own macOS
  testing. If shared memory misbehaves here, the proven fallback is Docker or Lima
  Ubuntu 22.04 with ns-3 + the agent in the same container (see plan.md decision log).
