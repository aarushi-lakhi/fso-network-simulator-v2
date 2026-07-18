# Project Retrospective — FSO Network Simulator

> The full story of this project: what we built in each phase, why we built it, the
> tradeoffs we weighed, what we learned, and the incidents worth remembering. Written
> as a memory aid — read `README.md` for the polished summary and
> `benchmarks/results/README.md` for the study data; read this to remember *how it
> actually went*.

The one-paragraph arc: we set out to build a portfolio-quality cross-layer FSO
simulator where a deep RL agent routes around atmospheric turbulence. We built exactly
that — and when the honest benchmark showed the RL agent merely *tying* the best
static route, the project turned into a five-study research program that hunted the
reason through the environment, the optimizer, the optimizer family, and finally the
observation itself, ending with a confirmed hypothesis and a learned policy that beats
every baseline. Ten phases, 35 PRs, six shared-seed studies, ~250 tests.

---

## Phase 0 (pre-history): fixing the original plan

The project began with a Gemini-generated plan that had real gaps. Before writing any
code we corrected it, and every correction mattered later:

- **ns3-gym → ns3-ai.** ns3-gym was abandoned around 2021; ns3-ai's shared-memory
  interface is 10–100× faster and actively maintained. (Later phases lived inside this
  interface daily — the right call.)
- **802.11s Wi-Fi mesh → PointToPoint links.** FSO is a laser, not broadcast RF. This
  forced Phase 3's most interesting design problem (see below) and killed HWMP as a
  baseline in Phase 5 (it's 802.11s-specific).
- **Added the missing bridge phase.** The original plan had GNU Radio and ns-3 as two
  unrelated simulations. We added Phase 3 (a custom `PropagationLossModel`) so one
  physics model drives everything.
- **Branch naming without phase numbers** — branches describe what code does, because
  plans change (they did: the plan said 5 phases; we shipped 10).

Also from this era: the project migrated from Windows 11/WSL2 to macOS (Apple Silicon)
between planning and execution, which shaped all of Phase 1.

## Phase 2a — the math prototype (built first, deliberately)

**What:** `prototype/gamma_gamma.py` — Rytov variance, α/β closed forms (Andrews &
Phillips), a product-of-Gammas sampler normalized to E[I]=1, Monte-Carlo OOK BER.

**Why first:** pure Python, no toolchain risk, and it established the validation
anchor used by every later layer: the scintillation-index identity
`SI = 1/α + 1/β + 1/(αβ)`. If a sampler's empirical SI matches the closed form, the
math is right — we used this same test in Python, C++, and the GNU Radio block.

**What we learned immediately:** two of the original tests were physically wrong, and
the *code* was right. One demanded σ²_R > 5 for "strong turbulence" at 1 km where the
closed form gives ≈1.99 (strong fluctuations begin at σ²_R > 1); the other expected
BER ≈ 0.5 at 0 dB SNR where ½·erfc(√½) ≈ 0.1587 is correct (BER → 0.5 only as
SNR → −∞). Hand-verifying the math before trusting tests became a habit.

## Phase 1 — the macOS toolchain (where the environment fought back)

**What:** `setup/` scripts installing GNU Radio 3.10 (Homebrew), building ns-3.40, and
adding ns3-ai — plus `verify_env.sh` as the health check.

**Tradeoff considered:** ns3-ai is Linux-first, and we had a Docker/Lima Ubuntu
fallback planned. Research showed its IPC uses boost shared memory plus its own
atomic spin-wait semaphore — *not* POSIX semaphores, sidestepping the classic macOS
blocker — and the a-plus-b example verified end-to-end natively on Apple Silicon.
Native won; the fallback was never needed.

**Incidents that became institutional knowledge:**
- ns-3.40's `./ns3` wrapper crashes on Python ≥3.14 (argparse change) → the entire
  ns-3 toolchain is pinned to python@3.11, which conveniently matched ns3-ai's needs.
- ns-3.40's test suite doesn't compile under current Apple libc++ (its custom
  `std::pair` `operator==` became ambiguous once libc++ shipped heterogeneous pair
  comparison) → we backported the upstream fix as `setup/patches/`.
- ns3-ai's first build races its own protobuf codegen → the install script retries.

## Phase 2b — the GNU Radio fading block

**What:** `gr-fso-turbulence/` — a numpy-only sync block multiplying the complex field
by √I so signal *power* follows the Gamma-Gamma irradiance statistics, holding each
coefficient for a coherence window. GRC block definition + demo flowgraph + QA tests.

**Design choice worth remembering:** amplitude ×√I, not ×I — photodetectors respond
to intensity and the Gamma-Gamma PDF describes intensity, not field. Also: no scipy
dependency, because GNU Radio's Homebrew python doesn't have it (this constraint bites
again spectacularly in the Phase 10 era — see "correlated fading parity" below).

## Phase 3 — the ns-3 FSO channel (the architecture puzzle)

**What:** `ns3-fso-channel/` — `GammaGammaFsoLossModel` (Beer-Lambert extinction +
Gamma-Gamma fading via ns-3's `GammaRandomVariable`) and `FsoTopologyHelper`.

**The puzzle:** PointToPoint channels don't consume `PropagationLossModel` at all —
that abstraction belongs to Wi-Fi/spectrum channels. Naively attaching the loss model
would have done nothing. The helper bridges the gap: it periodically draws the fading
state, computes instantaneous SNR from a link budget, maps it through the OOK BER
formula to a packet error rate, and installs that on a `RateErrorModel` per direction.
Fading physics → packet drops, on a link type that was never designed for it.

**Validated by:** an ns-3 test suite (Beer-Lambert exactness, unit-mean fading, the SI
identity again, distance monotonicity) and a 5-node mesh demo where PDR degraded from
1.0 (weak turbulence) to ~0.74 (strong) — the physics visibly reaching the network layer.

## Phase 4 — the cross-layer RL pipeline

**What:** `ns3-rl-router/` — the ns3-ai Gym environment (28-dim observation: per-link
SNR margin, drop rate, scintillation index, queue depth; `Discrete(4)` route actions
applied via static routing; reward −drops −delay −flapping −energy) and a from-scratch
PyTorch PPO (GAE, clipped surrogate, entropy bonus) proven on a toy env before ever
touching ns-3.

**Why PPO from scratch instead of stable-baselines:** the point was understanding and
portfolio value — and it paid off later, when the research phases needed surgical
access to entropy, KL, and update internals to diagnose the collapse.

**The bug that mattered:** upstream ns3-ai's `Ns3Env.reset()` silently re-runs ns-3
*without your settings* — from episode 2 onward everything reverts to C++ defaults and
every episode reuses one RNG run. Our `FsoNs3Env` subclass fixes it. Without catching
this, every study in the project would have been quietly invalid.

**First headline:** trained PPO beat random routing by 41% (PDR 0.724 vs 0.670),
800 episodes in ~8 minutes on the laptop.

## Phase 5 — the benchmark that changed the project

**What:** `benchmarks/` — PPO vs best-static / random / AODV across the C²ₙ sweep,
10 shared-seed episodes per cell (every policy faces identical fading realizations —
this protocol carried through all six studies).

**Tradeoff:** HWMP (from the original plan) can't run on PointToPoint; we substituted
AODV (IP-layer, works over p2p) as the classical-reactive baseline, plus best-static
as the oracle-ish bar and random as the floor.

**The finding that redirected everything:** PPO converged to *exactly* the best static
route — byte-identical per seed. Not a failure: under i.i.d. 1 ms block fading with
100 ms decisions, hold-the-best-route *is* the optimal policy, and PPO found it. But
it meant RL had nothing to exploit — and the honest write-up of that fact spawned
Phases 6–10. Side findings: moderate turbulence was indistinguishable from weak at our
18 dB link margin; AODV is competitive when calm but worst under stress (its control
packets fade too); and the strong-turbulence checkpoint transferred to calm regimes
*better* than training in them (flat reward landscapes give bad gradients).

## Phase 6 — correlated fading (memory is necessary…)

**What:** temporal correlation via a **Gaussian copula AR(1) applied per Gamma
component** — latent normal AR(1) → Φ → Gamma quantile — chosen because it preserves
the *exact* Gamma-Gamma marginal (so all Phase 2 validation still holds) while making
coherence time a knob. Implemented three times: Python reference (25 tests), C++
(`CorrelatedGammaGammaFading`, bit-exact at τ=0, boost `gamma_p_inv`), and later the
GNU Radio block.

**The 6c study verdict: no.** Even at τ=500 ms with 50 ms decision steps (10 decisions
per fade epoch), PPO stayed constant-route. The iteration *did* fix something real:
switching the observation from empirical drop counts to the loss model's instantaneous
per-link PER made PPO find the *right* constant route (it had converged to a wrong one
at τ=100/20 ms). Better perception improved route selection, not route switching.
Mechanism identified: per-episode reward variance ~25% of the mean vs a small switching
margin — a memoryless policy rationally collapses to the best constant action.
"The correlated channel makes adaptation *possible*, not *profitable at this margin*."

## Phase 7 — making adaptation provably profitable (…but not sufficient)

**What:** attacked the margin mechanism with environment changes: a topology whose
four routes are pairwise **link-disjoint** (the direct 0→3 link deliberately longer,
so d^(11/6) Rytov scaling makes relays genuinely compete — no route dominates a
priori), and **TCP traffic** (drops compound through congestion control, so dodging a
fade epoch pays non-linearly). A probe confirmed the per-episode best route now flips
between seeds — the adaptation signal finally existed.

**The sharpest negative result of the project:** a two-line scripted rule
(**greedy-PER**: pick the route whose links currently have the lowest error rate, with
hysteresis) beat the best static route on 8/10 episodes in both correlated cells
(+13.8 pp PDR under UDP, +75% goodput under TCP) — using *exactly the observation the
RL agent sees*. And PPO, even frame-stacked at double budget, still collapsed to a
constant route with entropy ≈0.005 nats. The bottleneck formally moved from the
environment to the optimizer. (Also memorable: TCP's bimodal failure — constant-route
policies get PDR≈0 episodes when their route is faded at handshake time.)

## Phase 8 — imitation-then-RL (convicting the on-policy gradient)

**The question, precisely:** can't *find* the switching policy, or can't *keep* it?
Behavior-clone the teacher (so no exploration needed), warm up the critic on
frozen-policy rollouts (a fresh critic's garbage advantages can destroy a good policy
in the first updates — we called this hazard before running), then PPO fine-tune.

**Verdict: can't keep.** PPO destroyed the cloned policy in both cells — under TCP
from an initialization that already beat every static route. The three-panel
trajectory plot (entropy 0.65→0.005, KL-from-BC drifting to 5.5 nats, switches 80→0)
is the single best figure of the project: you watch the destruction happen.

**The accidental discovery that set up Phase 10:** BC validation accuracy capped at
~0.63 — structurally. The teacher's hysteresis depends on which route it's currently
holding, and *the held route wasn't in the observation*. The clone matched the
teacher's fade-dodging exactly (PDR 0.913 vs 0.914) but over-switched (73 vs 46/ep),
and the reward gap was arithmetically exactly the flap bill: (73.3−46)×5 ≈ the
measured Δ. A stateless policy could imitate the *what* but not the *when-to-hold*.

## Phase 9 — Double DQN (completing the 2×2)

**Why:** off-policy value learning averages per-action estimates over a replay buffer
instead of the last on-policy batch — exactly the property that should resist the
return noise. Two arms: from scratch, and Q-network initialized from the BC checkpoint
(with an argmax-invariant return-scale bias offset, and low starting ε so exploration
didn't wash out the very policy we were testing retention of).

**Verdict: split, and it refined the conviction.** DQN-scratch collapsed like PPO but
onto the *correct* constant route (replay averaging fixes route selection). DQN-BC
under UDP became the **only quadrant in the whole {PPO,DQN}×{scratch,BC} grid where
switching survived** — stable ~18 switches/ep across all 80k steps, +3.5 pp PDR over
best-static, reward a statistical tie. Under TCP it was destroyed *faster* than PPO
(Q-gap crushed within ~500 gradient steps — TD-erasure). Conclusion: the binding
constraint is the **return-noise-to-action-gap ratio**, not on-policy vs off-policy
per se; and the teacher remained unbeaten, with its unobservable hysteresis state as
the last hypothesis standing.

## Phase 10 — route-aware observation (the payoff)

**The test:** append a 4-dim one-hot of the currently-held route to the observation
(28→32, behind a default-off flag so every earlier study stays reproducible). Success
criteria stated *before* running: BC should break the 0.63 cap and match the teacher,
and the best fine-tune should finally beat best-static significantly.

**It did.** BC accuracy jumped to 0.92+. `bc-route` became the first learned policy to
match the teacher (41 vs 46 switches/ep, reward statistically tied) and to **beat the
best static route significantly** (UDP: +99.9 reward, 7/0/3 paired, p≈0.03).
`dqn-bc-route` converted Phase 9's tie into a borderline win (+74.3, 8/2, p≈0.05).
Honest remainder: TCP's noise still defeats every optimizer (both failure modes were
noise problems, not information problems), and nothing finds switching from scratch.
Final ranking where learning works: **teacher ≈ bc-route > dqn-bc-route > best-static**.

The five-study arc in one line: *environment → optimizer → optimizer family →
observation → confirmed.* Each phase eliminated exactly one hypothesis with a
controlled, shared-seed, paired-comparison experiment.

---

## Infrastructure milestones along the way

- **CI** (GitHub Actions): hermetic test suites + ruff on every PR. Setting it up
  surfaced a pytest collision (two packages named `tests`) and the first-ever ruff run
  over `prototype/` (three lint nits, one of which was a test drawing 500k samples it
  never inspected).
- **Correlated-fading parity for the GNU Radio block**: scipy isn't available in GNU
  Radio's python, so the Gamma quantile function was implemented in pure numpy
  (Numerical Recipes incomplete-gamma + Halley inversion), validated to max relative
  error **2.35×10⁻¹²** against a hardcoded scipy reference table — with τ=0 verified
  *bit-exact* against the pre-change block. All three layers now share the full
  correlation model, not just the marginal.
- **The audit** (post-Phase 7): a full staleness sweep that rewrote `handoff.md` from a
  Phase 2a time capsule into a real continuity doc, fixed every stale count and date,
  and confirmed zero TODOs and complete study artifacts.

## War stories (the things that only live in memory otherwise)

- **GitHub auto-delete nuked `dev`.** The first `dev` → `main` release PR used `dev`
  itself as the head branch; on merge, GitHub's auto-delete-head-branches removed our
  permanent integration branch. Restored from `main`, and the rule was written into
  plan.md: releases go through short-lived `release/*` branches, never `dev` directly.
- **One ns3-ai Experiment per process, ever** — the shm segment is a C++ function-local
  static. Every study runs env-owning phases in subprocesses; forgetting this throws
  `boost::interprocess` errors on the second env in a process.
- **zsh doesn't word-split unquoted variables.** An afternoon was spent believing a
  `cd` was being stripped from commands when actually `$cfg` containing
  `"--topology disjoint"` was reaching argparse as a single token.
- **Long-running study agents stall; detached studies survive.** The pattern that
  emerged: orchestrators persist results incrementally (CSV after every cell), long
  runs launch via nohup, and progress is polled with foreground loops — so a stalled
  supervisor never costs more than the time since the last checkpoint.
- **The Phase 8 TCP cell briefly reused a 1000-step smoke checkpoint** (cache
  pollution) — caught and retrained before analysis. Checkpoint caching needs
  `--retrain` discipline.

## The working conventions that made it work

Small intentional commits and PRs (35 PRs, most under ~10 commits); no squash merges;
Conventional Commits with a custom `math` type; casual few-note PR descriptions;
review-then-push with an explicit breakdown every time; parallel agents only on
disjoint directories; `plan.md` as the living source of truth with a decisions log
(every pivot has a dated rationale); and — the one that mattered most — **reporting
losses as prominently as wins**. The project's credibility rests on Phases 5–9 being
negative results published with full data.

## What's deliberately still open

1. **TCP's return noise defeats every optimizer tried.** Variance-reduction ideas
   (baselines that subtract episode "weather luck", distributional RL) are untested.
2. **Nothing finds switching from scratch** — exploration remains a wall independent
   of retention; the winning recipe is imitation-first.
3. **PHY-in-the-loop**: the GNU Radio block and ns-3 share parameters, not samples —
   wiring actual IQ-derived fading into the channel would close the last gap between
   the layers.

Every one of these is an afternoon-sized experiment with the existing harness.
