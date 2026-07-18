# FSO Network Simulator — Project Plan

> **Living document.** Update this file as phases complete, decisions change, or scope shifts.
> Last updated: 2026-07-17 — all phases complete; see the tracker and `README.md`

---

## Project Overview

A cross-layer Free-Space Optical (FSO) network simulator bridging physical-layer atmospheric turbulence
modeling (GNU Radio) with network-layer adaptive routing (ns-3 + Deep Reinforcement Learning).

### High-Level Goals

- **PHY layer:** Custom GNU Radio OOT module implementing Gamma-Gamma fading to simulate atmospheric
  signal degradation
- **Channel bridge:** Custom ns-3 `PropagationLossModel` consuming Gamma-Gamma statistics to model
  FSO link quality realistically
- **Network layer:** Cross-layer ns-3 + PyTorch simulation trained via ns3-ai (shared memory interface)
- **Control plane:** Deep RL agent (PPO) that proactively reroutes mesh traffic upon detecting
  physical-layer signal variance
- **Efficiency goal:** Minimize packet loss and eliminate redundant retransmissions during
  high-turbulence conditions

---

## Repository Structure (actual, as shipped)

```
fso-network-simulator-v2/
├── plan.md                        ← this file
├── handoff.md                     # session/machine continuity notes
├── README.md                      # architecture, results, reproduction
├── LICENSE                        # GPL-2.0
├── .github/workflows/ci.yml      # hermetic tests + ruff on every PR
│
├── prototype/                     # Phase 2a/6a — pure Python math validation
│   ├── gamma_gamma.py             # Gamma-Gamma model + correlated (copula AR(1)) sampler
│   ├── turbulence_plots.py        # fading traces, BER, scintillation, correlation plots
│   ├── plots/                     # committed publication plots
│   └── tests/                     # 64 tests (39 core + 25 correlated)
│
├── gr-fso-turbulence/             # Phase 2b — GNU Radio OOT module
│   ├── python/fso_turbulence/     # fso_fading_channel block + QA tests
│   ├── grc/                       # GRC block definition
│   └── examples/fso_fading_demo.grc
│
├── ns3-fso-channel/               # Phase 3/6b — custom ns-3 contrib module
│   ├── model/                     # GammaGammaFsoLossModel + CorrelatedGammaGammaFading
│   ├── helper/                    # FsoTopologyHelper (fading → p2p error-rate bridge)
│   ├── test/                      # two ns-3 test suites
│   └── examples/fso-5node-mesh.cc
│
├── ns3-rl-router/                 # Phase 4/7 — DRL routing agent
│   ├── sim/                       # fso-rl-env.cc (ns3-ai Gym env), check_env, shims
│   ├── agent/                     # PPO, actor-critic, frame stack, train/eval, 36 tests
│   ├── config/sim_config.yaml
│   └── requirements.txt
│
├── benchmarks/                    # Phases 5/6c/7c — studies and results
│   ├── run_benchmark.py           # orchestrator (--study turbulence|correlated|adaptation)
│   ├── parse_traces.py / plot_results.py
│   ├── aodv/fso-aodv-baseline.cc  # classical-routing baseline scenario
│   ├── tests/                     # hermetic parsing tests
│   └── results/                   # committed CSVs, plots, findings README
│
└── setup/                         # Phase 1 — macOS toolchain scripts
    ├── install_{gnuradio,ns3,ns3_ai}.sh, link_fso_modules.sh, verify_env.sh
    └── patches/                   # ns-3.40 libc++ compat patch
```

---

## Branch Strategy

### Permanent Branches

| Branch | Purpose |
|--------|---------|
| `main` | Stable, demo-ready code only. **Never commit directly.** Merge via PR only. |
| `dev` | Integration branch. All features merge here first. `main` gets a cut when a phase is fully working. |

> ⚠️ **Cutting `main`:** never open the release PR with `dev` itself as the head branch —
> GitHub's auto-delete-head-branches setting deletes the PR head on merge, which nuked
> `dev` after PR #12 (restored from `main`). Instead branch `release/<name>` off `dev`
> and PR that into `main`.

### Branch Naming Convention

Format: **`<type>/<short-kebab-description>`**

Branch names describe *what the code does*, not when you're doing it (no phase numbers — the plan can change, branch names live forever in `git log`).

| Type | When to use |
|------|------------|
| `feat/` | New functionality |
| `fix/` | Bug fix on existing code |
| `chore/` | Tooling, build system, environment setup |
| `docs/` | Documentation only |
| `test/` | Tests with no production code changes |
| `refactor/` | Code restructure, no behavior change |
| `experiment/` | Exploratory / proof-of-concept (may never merge) |

### Planned Branches for This Project

```
main
└── dev
    ├── chore/dev-environment          # Phase 1: macOS GNU Radio + ns-3 install scripts + verification
    ├── feat/gamma-gamma-sampler       # Phase 2a: Python prototype — Gamma-Gamma RNG, BER curves
    ├── feat/gr-fso-fading-block       # Phase 2b: GNU Radio OOT module (depends on gamma-gamma-sampler)
    ├── feat/ns3-fso-propagation-model # Phase 3: custom PropagationLossModel in C++
    ├── feat/ns3-ai-gym-interface      # Phase 4a: ns3-ai shared memory bridge
    ├── feat/ppo-routing-agent         # Phase 4b: PyTorch PPO agent (can prototype in parallel with 4a)
    └── feat/benchmark-suite           # Phase 5: comparison scripts + plots
```

> **Parallel agent rule:** branches that touch different directories are safe to run simultaneously.
> e.g., `feat/gr-fso-fading-block` (only `gr-fso-turbulence/`) and
> `feat/ns3-fso-propagation-model` (only `ns3-fso-channel/`) never conflict.

### Branch Lifecycle

```bash
# 1. Always branch from dev (make sure it's up to date first)
git checkout dev
git pull origin dev
git checkout -b feat/gamma-gamma-sampler

# 2. Commit often with meaningful messages (see Commit Messages section)
git add prototype/gamma_gamma.py
git commit -m "feat(gamma-gamma): implement Gamma-Gamma RNG via product-of-Gammas method"

# 3. Keep your branch current with dev to avoid big merge conflicts later
git fetch origin
git rebase origin/dev        # preferred over merge — keeps history linear

# 4. When ready: push, open PR → dev, get review (even self-review — read the diff!)
git push origin feat/gamma-gamma-sampler
# PR title should match branch: "feat: gamma-gamma sampler"
# Merge strategy: merge commit or rebase-merge — NO squash merges (keep the real commits)
# Keep PRs and commits small and intentional; stack branches off other feature
# branches when needed to keep each PR reviewable
# PR descriptions: a few straightforward notes, casual tone — no headers/bolding/text walls
# NEVER merge directly — always open a PR so there's a review record

# 5. After merge, delete the remote branch (GitHub does this automatically if configured)
git branch -d feat/gamma-gamma-sampler
```

### Push Policy

> ⚠️ **Always review and test locally before pushing anything.**
> No force-pushes to `dev` or `main`. Ever.
> If a commit needs fixing after push, open a new `fix/` branch — don't rewrite history on shared branches.

### Parallel Agent Workflow

When two workstreams are active simultaneously (e.g., GNU Radio block + ns-3 channel model),
each runs in its own branch with no shared files — safe to work in parallel:

```
Agent 1: feature/phase2-gnuradio-block   (touches only gr-fso-turbulence/)
Agent 2: feature/phase3-ns3-fso-channel  (touches only ns3-fso-channel/)
```

Both agents PR into `dev` independently. Zero merge conflicts by design (separate directories).

---

## Coding Practices

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description>

[optional body]
```

| Type | When to use |
|------|------------|
| `feat` | New functionality |
| `fix` | Bug fix |
| `math` | Mathematical model changes (custom type for this project) |
| `test` | Adding or updating tests |
| `docs` | Documentation only |
| `refactor` | Code restructure, no behavior change |
| `perf` | Performance improvement |
| `chore` | Build system, deps, tooling |

**Examples:**
```
feat(gamma-gamma): add Rytov variance calculator with C2n sweep
math(fso-channel): implement Beer-Lambert atmospheric extinction term
test(prototype): verify scintillation index matches theoretical SI = 1/α + 1/β + 1/(αβ)
fix(gnuradio-block): correct buffer stride in work() for complex IQ input
```

### Python Standards

- **Style:** PEP 8. Enforced via `ruff` (faster than flake8/black combo).
- **Type hints:** All public functions must have type annotations.
- **Docstrings:** Google-style docstrings on every module, class, and public function.
- **Testing:** `pytest`. Target ≥80% coverage on `prototype/` and `agent/`.
- **Requirements:** Pin versions in `requirements.txt`. Use virtual environments (never global pip).

```python
# Example: typed + documented function
def gamma_gamma_sample(
    alpha: float,
    beta: float,
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate Gamma-Gamma distributed irradiance samples.

    Models atmospheric scintillation as the product of two independent
    Gamma-distributed random variables (Andrews & Phillips, 2005).

    Args:
        alpha: Large-scale scintillation parameter (> 0).
        beta: Small-scale scintillation parameter (> 0).
        n_samples: Number of irradiance samples to generate.
        rng: Optional NumPy random generator for reproducibility.

    Returns:
        Array of shape (n_samples,) with Gamma-Gamma distributed values,
        normalized so E[I] = 1.

    Raises:
        ValueError: If alpha or beta are not positive.
    """
```

### C++ Standards (ns-3 and GNU Radio)

- Follow existing ns-3 coding style (enforced by `utils/check-style-clang-format.py`).
- Header guards: `#ifndef NS3_GAMMA_GAMMA_FSO_LOSS_MODEL_H` style.
- Doxygen comments on all public methods.
- No raw pointers where smart pointers are available.
- Const-correctness: mark all read-only methods `const`.

### Testing Strategy

| Layer | Framework | What to test |
|-------|-----------|-------------|
| Python prototype | `pytest` | PDF shape, SI formula, edge cases (α=β, strong turbulence) |
| GNU Radio block | GRC flowgraph + Python unit | Buffer passthrough, energy conservation under fading |
| ns-3 channel model | ns-3 test suite | PDR vs. C²_n matches theoretical BER curves |
| DRL agent | pytest + tensorboard | Reward convergence, no NaN in gradients, action entropy |

### What Gets Committed

**Always commit:**
- Source code (`.py`, `.cc`, `.h`, `.yml`, `.grc`)
- Test files
- `requirements.txt`, `CMakeLists.txt`
- Plot scripts and generated plots (`benchmarks/results/*.png`)
- This `plan.md`

**Never commit:**
- Build artifacts (`build/`, `__pycache__/`, `*.pyc`, `*.so`, `*.o`)
- Large trace files (`*.pcap`, `*.tr`) — too large, generate locally
- Virtual environments (`venv/`, `.venv/`)
- ns-3 build directory

### `.gitignore` (to be created in Phase 1)

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/
*.egg-info/
dist/
.pytest_cache/
.ruff_cache/

# C++ / CMake
build/
*.o
*.so
*.a
CMakeCache.txt
CMakeFiles/

# ns-3
ns-3-dev/build/

# GNU Radio
*.pyc

# Simulation outputs (large — generate locally)
*.pcap
*.tr
*.xml

# OS
.DS_Store
Thumbs.db
```

---

## Phase Tracker

| Phase | Branch | Status | Notes |
|-------|--------|--------|-------|
| 1 — Environment Setup | `chore/dev-environment` | ✅ Complete | Installed + verified on macOS 2026-07-03 (`setup/verify_env.sh` 11/11); ns3-ai shared memory confirmed working natively on Apple Silicon |
| 2a — Python Prototype | `feat/gamma-gamma-sampler` | ✅ Complete | Merged to `dev` via PR #1 (2026-07-03); 39 tests, 99% coverage |
| 2b — GNU Radio Block | `feat/gr-fso-fading-block` | ✅ Complete | Merged to `dev` via PR #3 (2026-07-03); 6 QA tests green |
| 3 — ns-3 FSO Channel | `feat/ns3-fso-propagation-model` | ✅ Complete | Merged to `dev` via PR #4 (2026-07-03); test suite green, 5-node mesh demo works |
| 4a — ns3-ai Interface | `feat/ns3-ai-gym-interface` | ✅ Complete | Merged via PR #6 (2026-07-03); random-action episodes verified over shared memory |
| 4b — PPO Agent | `feat/ppo-routing-agent` | ✅ Complete | Merged via PR #7 (2026-07-03); 28 tests, 97% coverage. Integration merged via PR #8: PPO beats random by ~41% on the live mesh (PDR 0.724 vs 0.670, C²ₙ=10⁻¹³) |
| 5 — Benchmarks | `feat/benchmark-suite` | ✅ Complete | Merged via PR #10 (2026-07-03). PPO converges to the optimal (best-static) policy; beats random by 41%, AODV by 15%. Full findings in `benchmarks/results/README.md` |
| 6a — Correlated sampler (math) | `math/correlated-fading-sampler` | ✅ Complete | Merged via PR #16 (2026-07-07); SI identity holds on correlated chains within 1% |
| 6b — ns-3 correlated fading | `feat/correlated-fso-fading` | ✅ Complete | Merged via PR #17 (2026-07-07); τ=0 bit-identical to i.i.d., 9/9 tests |
| 6c — Correlated-fading study | `feat/correlated-fading-study` | ✅ Complete | Answer: no — memory is necessary but not sufficient. PPO stays constant-route even at τ=500 ms / 50 ms steps; PER observation fixed route *selection*, not switching. Analysis in `benchmarks/results/README.md` |
| CI | `chore/github-actions-ci` | ✅ Complete | Merged via PR #20 (2026-07-16); hermetic tests + ruff run on every PR, badge on the README |
| 7a — Adaptation-friendly environment | `feat/disjoint-routes-tcp` | ✅ Complete | Merged via PR #22 (2026-07-16); probe confirmed the per-episode best route flips on the disjoint mesh |
| 7b — Policy memory | `feat/frame-stacked-obs` | ✅ Complete | Folded into 7c: `FlatFrameStack` wrapper (8 frames, 7 tests) — collapsed to constant-route exactly like plain PPO |
| 7c — Adaptation study | `feat/adaptation-study` | ✅ Complete | Verdict: adaptation is now provably profitable (scripted greedy-PER beats best-static 8/10 in both correlated cells) and PPO still can't find it — collapses to constant routes, entropy ≈0.005 nats. Bottleneck moved from environment to optimizer. Analysis in `benchmarks/results/README.md` |
| 8 — Imitation-then-RL | `feat/imitation-study` | ✅ Complete | Verdict: PPO **destroys** the cloned switching policy in both correlated cells (UDP: collapses to constant route 0, byte-identical all seeds; TCP: BC started *better* than best-static and fine-tuning walked it below its own init). Exploration ruled out — the on-policy gradient under this return variance is the bottleneck. Bonus finding: a stateless clone matches the teacher's fade-dodging exactly but not its hysteresis (the held route isn't observable). Trajectory + analysis in `benchmarks/results/README.md` |
| 9 — Off-policy learning | `feat/offpolicy-study` | ✅ Complete | The 2×2 verdict is split, and it refines the conviction: DQN-scratch collapses like PPO but onto the *correct* route; DQN-BC under UDP is the **only quadrant where switching survives** (18/ep stable over 80k steps, +3.5 pp PDR vs best-static, reward tie); under TCP it's destroyed faster than PPO. Binding constraint = return-noise-to-action-gap ratio, not on-vs-off-policy; the scripted teacher stays unbeaten. The teacher's unobservable hysteresis state remains the standing hypothesis (Phase 10: put the held route in the observation). Analysis in `benchmarks/results/README.md` |
| 10 — Route-aware observation | `feat/held-route-obs` | 🔄 In progress | The standing-hypothesis test: append the current route (one-hot) to the observation so hold-vs-switch becomes expressible. If BC can now clone the teacher's hysteresis, the flap tax disappears — and a learned policy may finally beat best-static |

**Status legend:** 🔲 Not started · 🔄 In progress · ✅ Complete · ⏸ Blocked

---

## Key Technical Decisions (log changes here)

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-05-26 | Use `ns3-ai` instead of `ns3-gym` | ns3-gym is abandoned (~2021); ns3-ai uses shared memory (10-100x faster), has active PyTorch examples |
| 2026-05-26 | Use PPO as baseline RL algorithm | Stable training, widely understood, strong baseline; switch to SAC if continuous action space needed |
| 2026-05-26 | Omit `gr-osmosdr` from install | SDR hardware package — irrelevant without physical hardware |
| 2026-05-26 | Target GNU Radio 3.10 on Ubuntu 22.04 | Avoid version fragmentation; 3.10 ships with Ubuntu 22.04 apt |
| 2026-05-26 | FSO topology uses PointToPoint, not Wi-Fi mesh | FSO is point-to-point laser, not broadcast RF — wrong propagation abstraction |
| 2026-07-03 | Dev environment moved from Windows 11 + WSL2 to macOS (Apple Silicon) | New laptop is a Mac. GNU Radio 3.10 installs via Homebrew/conda; ns-3.40 builds natively with CMake + clang. Supersedes the Ubuntu 22.04/WSL2 parts of the 2026-05-26 decision (GNU Radio 3.10 target unchanged). Risk: ns3-ai shared memory is Linux-first — verify on macOS in Phase 1; fallback is Docker/Lima Ubuntu 22.04 |
| 2026-07-03 | No `Co-Authored-By` trailers on commits | User preference; supersedes earlier co-author policy in handoff.md |
| 2026-07-03 | ns3-ai natively on macOS — Docker/Lima fallback not needed | a-plus-b shared-memory example verified end-to-end on Apple Silicon. ns-3 toolchain pinned to python@3.11 (ns-3.40's `./ns3` breaks on 3.14; ns3-ai bindings want ≤3.11); ns-3.40 needs a small libc++ compat patch, shipped in `setup/patches/` |
| 2026-07-03 | Phase 5 baselines: best-static route, random routing, and AODV — not HWMP | HWMP is 802.11s-specific and the topology is PointToPoint (per the 2026-05-26 decision), so it can't apply. AODV runs at the IP layer and works over p2p links; together with best-fixed-route and random, it gives classical-reactive, oracle-ish, and floor baselines for the trained PPO policy |
| 2026-07-03 | Phase 6 (extension): temporally correlated fading via Gaussian copula AR(1) applied per Gamma component | Phase 5 showed i.i.d. 1 ms block fading makes hold-the-best-route optimal — RL had nothing to exploit. The copula construction (correlated normal → Φ → Gamma quantile, per component with separate large/small-scale coherence times) preserves the exact Gamma-Gamma marginal, so all Phase 2 validation still holds while coherence becomes tunable. 6c may add slow per-link C²ₙ drift (OU process) as the minutes-scale "weather" signal and/or faster decision steps |
| 2026-07-13 | Phase 7 attacks adaptation *profitability* via environment changes first, policy memory second | Phase 6's mechanism finding: switching never pays because routes share links (one fade epoch degrades several routes at once) and UDP loss is linear in drop count. So 7a changes the environment — link-disjoint routes (a fade on one route leaves alternatives genuinely clean) and TCP traffic (drops compound through congestion control, making fade-dodging pay non-linearly) — before touching the policy. Frame-stacking (7b) only if 7a alone doesn't separate PPO from best-static. Same shared-seed study protocol as 6c |
| 2026-07-17 | Phase 8: imitation-then-RL is the first PPO-collapse experiment | Cheapest decisive probe of the Phase 7 question. Behavior-cloning the scripted greedy-PER teacher yields a switching policy without any exploration; PPO then fine-tunes from that initialization. If fine-tuning *improves or holds* the policy, exploration was the bottleneck; if it *degrades* it back to constant-route, the on-policy gradient itself is. Key hazard to control: a freshly-initialized critic can destroy a good BC policy in the first updates — warm up the value head on frozen-policy rollouts before unfreezing |
| 2026-07-17 | Phase 9: Double DQN completes the {PPO, DQN} × {scratch, BC-init} 2×2 | Phase 8 convicted the on-policy gradient: PPO destroys a working switching policy it was handed. Off-policy Q-learning estimates per-action values by averaging over a replay buffer instead of the last on-policy batch — exactly the property that should resist the ~25% return noise that collapses PPO. Two arms per cell: DQN from scratch (can off-policy *find* the policy?) and DQN Q-network initialized from the Phase 8 BC checkpoints (can it *keep* it?). Same cells, seeds, and paired protocol as Phase 8; three of the four 2×2 quadrants are already measured |
| 2026-07-18 | Phase 10 tests the surviving hypothesis: the held route joins the observation | Every learned policy across phases 7–9 failed to express the teacher's hysteresis, and the reward gaps were arithmetically the flap bill — because the currently-held route isn't observable, hold and switch are indistinguishable to a stateless policy. Fix: a `--routeInObs` flag appends a 4-dim one-hot of the current route (obs 28→32), default off so all earlier studies stay reproducible. Experiment: re-run BC / bc-ppo / dqn-bc (+ dqn-scratch) with route-aware observations on the two correlated cells. Success criterion: BC matches the teacher (≈46 switches, its reward), and the best RL fine-tune finally beats best-static significantly |

---

## The Core Math (reference)

### Gamma-Gamma Irradiance PDF

```
f(I; α, β) = [2(αβ)^((α+β)/2)] / [Γ(α)Γ(β)]
             × I^((α+β)/2 - 1)
             × K_{α-β}(2√(αβI))
```

- `K_v` = modified Bessel function of the second kind (`scipy.special.kv`)
- `Γ` = gamma function (`scipy.special.gamma`)
- Sampling method: product of two independent Gamma RVs (faster than inverting the PDF)

### Rytov Variance (turbulence regime)

```
σ²_R = 1.23 × C²_n × k^(7/6) × L^(11/6)
```

- `C²_n` = refractive index structure parameter (turbulence strength)
  - Weak: `C²_n ≈ 10⁻¹⁷ m^(-2/3)`
  - Moderate: `C²_n ≈ 10⁻¹⁵ m^(-2/3)`
  - Strong: `C²_n ≈ 10⁻¹³ m^(-2/3)`
- `k = 2π/λ` (wavenumber; λ ≈ 1550 nm for telecom FSO)
- `L` = link distance (meters)

### Scintillation Index (validation target)

```
SI = E[I²] / E[I]² - 1 = 1/α + 1/β + 1/(αβ)
```

Use this identity to validate your Gamma-Gamma sampler in tests.

### α, β from Rytov variance (plane wave approximation)

```
α = [exp(0.49 σ²_R / (1 + 1.11 σ_R^(12/5))^(7/6)) - 1]^(-1)
β = [exp(0.51 σ²_R / (1 + 0.69 σ_R^(12/5))^(5/6)) - 1]^(-1)
```

### Beer-Lambert Atmospheric Extinction (ns-3 channel model)

```
L_atm(d) = exp(-σ_ext × d)
```

- `σ_ext` = extinction coefficient (visibility-dependent; fog: ~0.1 km⁻¹, clear: ~0.01 km⁻¹)

---

## Resources

- [Andrews & Phillips — Laser Beam Propagation through Random Media (2005)](https://spie.org/Publications/Book/684}
- [ns3-ai GitHub](https://github.com/hust-diangroup/ns3-ai)
- [GNU Radio gr-modtool docs](https://wiki.gnuradio.org/index.php/OutOfTreeModules)
- [ns-3 PropagationLossModel API](https://www.nsnam.org/docs/release/3.40/doxygen/propagation-loss-model_8h.html)
- [Stable Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- Conventional Commits spec: https://www.conventionalcommits.org/
