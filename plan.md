# FSO Network Simulator — Project Plan

> **Living document.** Update this file as phases complete, decisions change, or scope shifts.
> Last updated: 2026-07-03

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

## Repository Structure (target)

```
fso-network-simulator/
├── plan.md                        ← this file
├── .gitignore
├── README.md
│
├── prototype/                     # Phase 2 — pure Python math validation
│   ├── gamma_gamma.py             # Gamma-Gamma RNG + PDF implementation
│   ├── turbulence_plots.py        # visualization: fading traces, BER curves
│   ├── requirements.txt
│   └── tests/
│       └── test_gamma_gamma.py
│
├── gr-fso-turbulence/             # Phase 2 — GNU Radio OOT module
│   ├── CMakeLists.txt
│   ├── python/
│   │   └── fso_turbulence/
│   │       └── fso_fading_channel.py
│   ├── grc/
│   │   └── fso_turbulence_fso_fading_channel.block.yml
│   └── examples/
│       └── fso_fading_demo.grc    # GRC flowgraph: tone → fading → oscilloscope
│
├── ns3-fso-channel/               # Phase 3 — custom ns-3 propagation model
│   ├── model/
│   │   ├── gamma-gamma-fso-loss-model.h
│   │   └── gamma-gamma-fso-loss-model.cc
│   ├── helper/
│   │   └── fso-topology-helper.h/.cc
│   └── examples/
│       └── fso-5node-mesh.cc
│
├── ns3-rl-router/                 # Phase 4 — DRL routing agent
│   ├── sim/
│   │   └── fso-rl-env.cc          # ns-3 simulation + ns3-ai interface
│   ├── agent/
│   │   ├── ppo_agent.py
│   │   ├── network.py             # Actor-Critic neural net
│   │   └── train.py
│   ├── config/
│   │   └── sim_config.yaml
│   └── requirements.txt
│
└── benchmarks/                    # Phase 5 — results and comparison
    ├── run_benchmark.py
    ├── parse_traces.py
    ├── plot_results.py
    └── results/                   # .gitignore'd large trace files; plots committed
        └── .gitkeep
```

---

## Branch Strategy

### Permanent Branches

| Branch | Purpose |
|--------|---------|
| `main` | Stable, demo-ready code only. **Never commit directly.** Merge via PR only. |
| `dev` | Integration branch. All features merge here first. `main` gets a cut when a phase is fully working. |

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
# Merge strategy: squash-merge (clean history on dev) or rebase-merge (preserve commits)
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
| 1 — Environment Setup | `chore/dev-environment` | 🔲 Not started | macOS: GNU Radio 3.10 (Homebrew/conda) + ns-3.40 (native CMake/clang); verify ns3-ai shared memory works on macOS early |
| 2a — Python Prototype | `feat/gamma-gamma-sampler` | 🔄 In progress | Code + 39 tests + plots done, verified on macOS (2026-07-03); PR → `dev` pending |
| 2b — GNU Radio Block | `feat/gr-fso-fading-block` | 🔲 Not started | Depends on Phase 1 + 2a |
| 3 — ns-3 FSO Channel | `feat/ns3-fso-propagation-model` | 🔲 Not started | Depends on Phase 1 + 2a (for params) |
| 4a — ns3-ai Interface | `feat/ns3-ai-gym-interface` | 🔲 Not started | Depends on Phase 3 |
| 4b — PPO Agent | `feat/ppo-routing-agent` | 🔲 Not started | Can prototype in parallel with 4a |
| 5 — Benchmarks | `feat/benchmark-suite` | 🔲 Not started | Depends on Phase 4 |

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
