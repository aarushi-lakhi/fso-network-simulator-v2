# FSO Network Simulator — Agent Handoff Document

> **Purpose:** continuity notes for picking this project up in a new session or on a new
> machine. Read alongside `plan.md` (phase tracker, decisions log, conventions) and
> `README.md` (architecture, results, reproduction). This file carries only what those
> two don't: working style, environment quirks, and the open threads.
>
> Last updated: 2026-07-18 — **all ten phases complete**, `main` == `dev` at the
> phase 10 release (PR #35). For the full history, read `RETROSPECTIVE.md`.

---

## Project State in One Paragraph

The simulator is finished and self-documenting: Gamma-Gamma turbulence math validated in
Python (`prototype/`), mirrored into a GNU Radio fading block (`gr-fso-turbulence/`) and
a custom ns-3 channel module (`ns3-fso-channel/`) — all three layers including the
correlated (copula AR(1)) fading model — bridged to PyTorch agents through ns3-ai
(`ns3-rl-router/`: PPO, Double DQN, behavior cloning, a scripted greedy-PER teacher),
and benchmarked across six shared-seed studies (`benchmarks/results/README.md`). The
research arc *resolved*: after PPO merely tied the best static route (phase 5), five
controlled studies cornered the cause through environment (6), optimizer (7–8), and
optimizer family (9) to the observation itself (10) — appending the held route to the
observation let a behavior-cloned policy match the scripted teacher and beat the best
static route significantly, the program's confirmed-hypothesis ending.

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

The big question — why RL collapsed and what fixes it — was ANSWERED by phases 8–10
(see `RETROSPECTIVE.md`): the winning recipe is route-aware observations + imitation
from the greedy-PER teacher, optionally DQN-fine-tuned. What remains open:

1. **TCP's return noise defeats every optimizer tried** (PPO's entropy collapse and
   DQN's TD-erasure both persist even with route-aware observations). Untested levers:
   variance reduction via a "weather luck" baseline, distributional RL, larger batch /
   multi-episode returns.
2. **Nothing finds switching from scratch** — exploration is a wall independent of
   retention. Untested: intrinsic motivation, scheduled teacher mixing (DAgger-style).
3. **PHY-in-the-loop**: wire the GNU Radio block's output into the ns-3 channel for
   true cross-layer simulation (currently the layers share parameters, not samples).

Each is an afternoon-sized experiment with the existing `--study` harness.
