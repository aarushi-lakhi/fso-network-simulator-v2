# FSO Network Simulator — Agent Handoff Document

> **Purpose:** Captures conversation context, decisions, preferences, and current state for
> seamless continuity when picking up this project on a new machine or in a new session.
> Read this alongside `plan.md`, which has the full technical plan and phase tracker.
>
> Last updated: 2026-07-01

---

## Project in One Paragraph

Building a cross-layer Free-Space Optical (FSO) network simulator. The physics layer is a
custom GNU Radio OOT module that implements Gamma-Gamma atmospheric turbulence fading. Those
physical-layer statistics feed into a custom ns-3 propagation loss model (a piece the original
plan was missing). On top of that sits a Deep Reinforcement Learning agent (PyTorch PPO via
ns3-ai) that reroutes mesh traffic proactively when it detects signal variance, minimising
packet loss and eliminating redundant retransmissions. The user's goals are equally portfolio
and academic — it needs to be both technically rigorous and visually impressive on GitHub.

---

## Current Git State

```
origin/main      ← stable; two commits (init + branch naming fix)
origin/dev       ← integration branch; mirrors main
feat/gamma-gamma-sampler  ← LOCAL ONLY, not yet pushed to origin
```

**Commits on `feat/gamma-gamma-sampler` (ahead of dev by 2):**
```
eaa0728  test(gamma-gamma): full pytest suite with SI identity validation
42ebc00  feat(gamma-gamma): implement Gamma-Gamma atmospheric turbulence model
f2e0b79  docs(plan): update branch naming to industry-standard convention   ← on main/dev
78db0b7  chore: initialize repo with plan and gitignore                     ← on main/dev
```

**Action needed on new machine:**
```bash
git clone https://github.com/aarushi-lakhi/fso-network-simulator.git
cd fso-network-simulator

# The feature branch is local-only on the old laptop — it needs to be pushed first,
# OR the user can cherry-pick / re-push after cloning.
# If the user pushed before switching: git checkout feat/gamma-gamma-sampler
# If not yet pushed: the two commits above need to be recreated or pushed from the old machine.
```

---

## What Has Been Built: Phase 2a (Complete, Not Yet PR'd)

All files live in `prototype/`.

### Files Created

| File | Purpose |
|------|---------|
| `prototype/gamma_gamma.py` | Core math — Gamma-Gamma model, BER analysis |
| `prototype/turbulence_plots.py` | Three publication-quality plots |
| `prototype/tests/test_gamma_gamma.py` | 30 pytest tests |
| `prototype/requirements.txt` | numpy, scipy, matplotlib, pytest, pytest-cov |
| `prototype/tests/__init__.py` | Makes tests/ a package |

### What `gamma_gamma.py` Implements

- **`TurbulenceParams`** — validated dataclass (C²_n, wavelength, distance); raises `ValueError` on bad input
- **`rytov_variance(C2n, wavelength, distance)`** — σ²_R = 1.23 × C²_n × k^(7/6) × L^(11/6)
- **`alpha_beta_from_rytov(sigma2_R)`** — Andrews & Phillips (2005) Eqs. 8.16–8.17
- **`gamma_gamma_sample(alpha, beta, n_samples, rng)`** — product-of-Gammas, E[I]=1 normalised
- **`scintillation_index(alpha, beta)`** — closed-form: 1/α + 1/β + 1/(αβ)
- **`empirical_scintillation_index(samples)`** — for sampler validation
- **`ber_ook_fading(snr_db_range, alpha, beta, ...)`** — Monte Carlo average BER
- **`ber_awgn_baseline(snr_db_range)`** — analytical AWGN reference

### What `turbulence_plots.py` Generates

Run `python turbulence_plots.py` from `prototype/` to write to `prototype/plots/`:
1. `fading_traces.png` — irradiance time-series for weak/moderate/strong turbulence
2. `ber_vs_snr.png` — BER vs SNR, fading vs AWGN baseline
3. `scintillation_map.png` — SI vs C²_n across multiple link distances

### Key Test (the mathematical validation)

`test_scintillation_index_identity` in `test_gamma_gamma.py` — verifies the sampler
produces distributions where the empirical SI matches the closed-form SI = 1/α + 1/β + 1/(αβ)
within 5% on 500k samples. If this test passes, the Gamma-Gamma math is correct.

### How to Run

```bash
cd prototype/
python3 -m venv .venv && source .venv/bin/activate   # Linux/WSL2
pip install -r requirements.txt

# Tests
pytest tests/ -v --cov=gamma_gamma --cov-report=term-missing

# Plots
python turbulence_plots.py
```

---

## Next Steps (in order)

1. **Push `feat/gamma-gamma-sampler`** — from old machine before switching, or recreate from
   the two commits above:
   ```bash
   git push origin feat/gamma-gamma-sampler
   ```

2. **Run the test suite** on the new machine in WSL2 — confirm all 30 tests pass.

3. **Run `turbulence_plots.py`** — review the three plots visually to confirm the fading
   traces, BER curves, and scintillation map look physically correct.

4. **Open PR: `feat/gamma-gamma-sampler` → `dev`** — user reviews the diff, squash-merge.

5. **Start Phase 1 (`chore/dev-environment`)** — WSL2 + GNU Radio 3.10 + ns-3.40 setup.
   This can be worked on in parallel with step 3/4.

6. **Start Phase 2b (`feat/gr-fso-fading-block`)** — GNU Radio OOT block; depends on
   Phase 1 environment being set up and Phase 2a being merged.

---

## Working Style & Preferences (Important for the Next Agent)

- **Never push to remote without explicit user approval.** The user reviews and tests locally
  first. Ask before every push — even if they've approved it once before, ask again.

- **Never push to `main` or `dev` directly.** All changes go through feature branches → PR.

- **Branch naming convention:** `<type>/<descriptive-name>` (e.g. `feat/gamma-gamma-sampler`,
  `chore/dev-environment`). No phase numbers in branch names — they describe what the code
  does, not when.

- **Commit message style:** Conventional Commits — `<type>(<scope>): <description>`.
  Always add `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>` trailer.

- **Two agents in parallel:** Safe when working in different directories. The plan defines
  which branches touch which directories — check before running concurrent agents.

- **`plan.md` is the living source of truth.** Update the phase tracker table and the
  "Key Technical Decisions" log whenever something changes.

---

## Key Corrections Made to the Original Gemini Plan

These are important context — the user started with a Gemini-generated plan that had gaps.
Don't revert to the original plan's approach on these items.

| Original Plan | Corrected Approach | Why |
|--------------|-------------------|-----|
| Used `ns3-gym` | Use `ns3-ai` | ns3-gym abandoned ~2021; ns3-ai uses shared memory (10-100x faster), actively maintained |
| 802.11s Wi-Fi mesh topology | Custom `PointToPointHelper` FSO topology | FSO is point-to-point laser, not broadcast RF — wrong propagation model |
| No ns-3 FSO channel model | Add Phase 3: `GammaGammaFsoLossModel` custom `PropagationLossModel` | ns-3 has no FSO model; without this, the two phases are disconnected |
| Included `gr-osmosdr` | Omit entirely | SDR hardware package; irrelevant without physical hardware attached |
| `feature/phase<N>-<name>` branch naming | `<type>/<descriptive-name>` | Industry standard; phase numbers in names rot as the plan changes |
| No explicit PHY→network bridge | Phase 2a outputs parameterise Phase 3 | Without this link, GNU Radio and ns-3 are two unrelated simulations |

---

## The Architecture (What Makes This Coherent)

```
Phase 2a: Python Prototype
  gamma_gamma.py → fading coefficient time-series, BER curves, α/β/C²_n tables
        │
        │ (parameters exported)
        ▼
Phase 2b: GNU Radio OOT Block (gr-fso-turbulence)
  fso_fading_channel.py → multiplies incoming IQ samples by Gamma-Gamma coefficients
        │
        │ (same physical parameters)
        ▼
Phase 3: ns-3 Custom Propagation Model
  GammaGammaFsoLossModel → C++ PropagationLossModel consuming Gamma-Gamma stats
  + Beer-Lambert atmospheric extinction: L_atm = exp(-σ_ext × d)
  + PointToPoint FSO 5-node topology (NOT Wi-Fi mesh)
        │
        ▼
Phase 4: DRL Routing Agent
  ns3-ai shared-memory interface → PyTorch PPO agent
  State: per-link SNR, drop rate, scintillation index, queue depth
  Action: next-hop selection
  Reward: -dropped_packets - latency + energy_saved - route_flapping
        │
        ▼
Phase 5: Benchmarks
  Trained PPO vs AODV vs HWMP across weak/moderate/strong C²_n sweep
```

---

## Core Math Reference (quick lookup)

**Gamma-Gamma PDF:**
```
f(I; α, β) = [2(αβ)^((α+β)/2)] / [Γ(α)Γ(β)] × I^((α+β)/2 - 1) × K_{α-β}(2√(αβI))
```

**Rytov variance:** `σ²_R = 1.23 × C²_n × k^(7/6) × L^(11/6)`  where `k = 2π/λ`

**α, β from σ²_R (plane wave):**
```
α = 1 / [exp(0.49 σ²_R / (1 + 1.11 σ_R^(12/5))^(7/6)) - 1]
β = 1 / [exp(0.51 σ²_R / (1 + 0.69 σ_R^(12/5))^(5/6)) - 1]
```

**Scintillation index identity (sampler validation):**  `SI = 1/α + 1/β + 1/(αβ)`

**C²_n regimes:** Weak ≈ 10⁻¹⁷, Moderate ≈ 10⁻¹⁵, Strong ≈ 10⁻¹³ m^(-2/3)

---

## Environment Notes

- Development environment: **WSL2 (Ubuntu 22.04) is required** for GNU Radio and ns-3.
  The Python prototype (`prototype/`) runs on native Windows Python too.
- Target: **GNU Radio 3.10** (ships with Ubuntu 22.04 apt — no version pinning needed).
- Target: **ns-3.40** (pin with `git checkout ns-3.40` after cloning).
- User is on **Windows 11** (moving from a Dell Latitude 5330 to a new laptop).
- GitHub remote: `https://github.com/aarushi-lakhi/fso-network-simulator.git`
