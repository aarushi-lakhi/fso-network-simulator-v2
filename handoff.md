# FSO Network Simulator — Agent Handoff Document

> **Purpose:** continuity notes for picking this project up in a new session or on a new
> machine. Read alongside `plan.md` (phase tracker, decisions log, conventions) and
> `README.md` (architecture, results, reproduction). This file carries only what those
> two don't: working style, environment quirks, and the open threads.
>
> Last updated: 2026-07-17 — **all seven phases complete**, `main` == `dev` at the
> phase 7 release (PR #24).

---

## Project State in One Paragraph

The simulator is finished and self-documenting: Gamma-Gamma turbulence math validated in
Python (`prototype/`), mirrored into a GNU Radio fading block (`gr-fso-turbulence/`) and
a custom ns-3 channel module (`ns3-fso-channel/`, including Phase 6's correlated-fading
process), bridged to a PyTorch PPO agent through ns3-ai (`ns3-rl-router/`), and
benchmarked across three studies (`benchmarks/results/README.md`). The scientific arc
ended with a sharp negative result: adaptation is provably profitable in the Phase 7
environment (a scripted greedy-PER rule beats the best static route 8/10 episodes) and
PPO still collapses to constant-route policies — the bottleneck is the optimizer, not
the environment.

## Working Style (unchanged — follow strictly)

- **Never push without explicit approval; before any push/PR give a step-by-step
  breakdown** (commands, commit messages, PR title/description) and wait for a yes.
- Feature branches → PR into `dev`; **no squash merges**; small intentional commits and
  PRs (stacked branches welcome). Cut `main` only from a `release/*` branch — never with
  `dev` as the PR head (GitHub auto-delete once nuked `dev`; see plan.md).
- Conventional Commits, **no `Co-Authored-By` trailers**. PR descriptions: a few casual
  notes, no headers/bolding/text walls. Sparse, purposeful code comments.
- `plan.md` is the living source of truth — update the tracker and decisions log as
  things change. Report results honestly, including losses.

## Environment (macOS, Apple Silicon)

- Toolchain lives outside the repo at `~/fso-tools/` (ns-3.40 + ns3-ai; see `setup/`).
  `setup/verify_env.sh` checks everything (11 PASS = healthy).
- The ns-3 side is pinned to **python@3.11** (`./ns3` breaks on 3.14; ns3-ai bindings
  are 3.11). GNU Radio uses Homebrew's default python. Activate the agent venv
  (3.11, `ns3-rl-router` requirements + the two ns3ai editable packages) for anything
  that spawns ns-3 — `env python3` must resolve to 3.11.
- `setup/link_fso_modules.sh` links the repo's ns-3 modules into the tree;
  `--unlink` restores it (then build only the `ai` target — contrib/ai's rate-control
  example has a known pre-existing build failure on ns-3.40).
- Hard-won pitfalls: ns3-ai allows **one Experiment per process** (run env-owning
  phases in subprocesses); upstream `Ns3Env.reset()` drops settings (our `FsoNs3Env`
  fixes it — always use it); the default shell is zsh, which does **not** word-split
  unquoted variables; long study runs should persist results incrementally.
- GitHub remote: `https://github.com/aarushi-lakhi/fso-network-simulator-v2.git`.
  CI (GitHub Actions) runs the hermetic test suites + ruff on every PR.

## Open Threads (a new chapter, not unfinished business)

1. **Why does on-policy PPO collapse to constant routes under ~25% return noise, and
   what fixes it?** Cheap experiments with the existing harness: advantage/reward
   normalization variants; behavior-cloning the scripted greedy-PER teacher then
   fine-tuning with PPO; off-policy methods (DQN/SAC-discrete). Each is a small
   `agent/` branch plus one `--study` cell.
2. PHY-in-the-loop: wire the GNU Radio block's output into the ns-3 channel for true
   cross-layer simulation (currently the layers share parameters, not samples).
