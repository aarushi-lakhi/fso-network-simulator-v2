# Benchmark results (Phases 5–8)

PPO routing vs classical baselines on the 5-node FSO mesh (one 2 Mbps UDP
flow 0→3, 7 Gamma-Gamma faded links, 100-step episodes at 0.1 s/step).
Every policy was evaluated on the same 10 held-out ns-3 run numbers
(seeds 100–109), so all policies face identical fading realisations.
Values are mean ± std over the 10 episodes.

## Headline table

| regime (C²ₙ) | policy | reward | PDR | delay [ms] | PHY drops |
|---|---|---|---|---|---|
| weak (1e-17) | PPO (per-regime) | −80.9 ± 0.0 | 1.000 ± 0.000 | 0.268 ± 0.000 | 0.0 |
| weak | PPO (1e-13 ckpt) | −55.6 ± 0.0 | 1.000 ± 0.000 | 0.179 ± 0.000 | 0.0 |
| weak | **best static (route 0)** | **−50.6 ± 0.0** | 1.000 ± 0.000 | 0.179 ± 0.000 | 0.0 |
| weak | random | −427.7 ± 22.0 | 1.000 ± 0.000 | 0.202 ± 0.003 | 0.0 |
| weak | AODV | −55.3 ± 4.8 | 0.983 ± 0.011 | 1.095 ± 1.397 | 0.0 |
| moderate (1e-15) | PPO (per-regime) | −80.9 ± 0.0 | 1.000 ± 0.000 | 0.268 ± 0.000 | 0.0 |
| moderate | PPO (1e-13 ckpt) | −55.6 ± 0.0 | 1.000 ± 0.000 | 0.179 ± 0.000 | 0.0 |
| moderate | **best static (route 0)** | **−50.6 ± 0.0** | 1.000 ± 0.000 | 0.179 ± 0.000 | 0.0 |
| moderate | random | −427.7 ± 22.0 | 1.000 ± 0.000 | 0.202 ± 0.003 | 0.0 |
| moderate | AODV | −55.3 ± 4.8 | 0.983 ± 0.011 | 1.095 ± 1.397 | 0.0 |
| strong (1e-13) | **PPO (per-regime)** | **−730.6 ± 16.1** | **0.724 ± 0.007** | 0.179 ± 0.000 | **675.0 ± 16.1** |
| strong | PPO (1e-13 ckpt) | −730.6 ± 16.1 | 0.724 ± 0.007 | 0.179 ± 0.000 | 675.0 ± 16.1 |
| strong | best static (route 2) | −730.6 ± 16.1 | 0.724 ± 0.007 | 0.179 ± 0.000 | 675.0 ± 16.1 |
| strong | random | −1233.0 ± 26.6 | 0.670 ± 0.008 | 0.200 ± 0.003 | 805.3 ± 19.8 |
| strong | AODV | −858.4 ± 51.5 | 0.664 ± 0.024 | 1.194 ± 1.610 | 807.4 ± 51.4 |

Full per-episode data: `raw_results.csv` (includes each static route
separately); aggregates: `summary.csv`; charts: `plots/`.

## Reading the results honestly

* **Weak and moderate regimes are indistinguishable.** With the default
  link budget (18 dB SNR margin at 940 m) the scintillation index at
  C²ₙ ≤ 1e-15 is too small to corrupt packets: zero PHY drops for every
  policy, PDR = 1 for all in-band routers. Only the reward's energy /
  flap / delay shaping separates policies there.
* **PPO ties the best static route under strong turbulence — it does
  not beat it.** Both the per-regime policy and the Phase 4 checkpoint
  converge to "hold route 2 (0-4-3) and never flap", which is exactly
  the best fixed route (identical rewards on every seed). Fading here
  is i.i.d. block fading (1 ms coherence vs 100 ms decision steps), so
  there is no persistent link-quality signal to exploit by rerouting
  mid-episode; learning the best route and holding it *is* the optimal
  stationary policy, and PPO finds it. PPO's margin over random
  (+41% reward, +5.4 pp PDR) comes from that.
* **Fresh PPO is slightly *worse* than the transferred checkpoint in
  the drop-free regimes.** With no drops, the reward landscape is
  almost flat (route differences are worth ~25 reward of 80) and the
  20k/40k-step runs settled on the 3-hop route 3 (−80.9) instead of a
  2-hop route (−55.6). The strong-trained checkpoint transfers down to
  calm regimes better than training in them, because strong turbulence
  gives the gradient signal a reason to prefer short, reliable routes.
* **AODV is competitive when calm, weakest when stressed.** It loses
  ~2% PDR at startup (route discovery buffers the first packets — also
  visible as its large delay variance) and under strong turbulence its
  control packets face the same fading as data, so it re-routes
  erratically and lands below even the best static route (PDR 0.664 vs
  0.724, with 3× the reward variance).

## Phase 6: the correlated-fading study

Phase 5's conclusion was that i.i.d. 1 ms block fading gives RL nothing to
exploit. Phase 6 added tunable coherence times (Gaussian copula AR(1),
exact Gamma-Gamma marginal preserved) and asked the follow-up question:
**does PPO beat the best static route once the channel has memory?**

All at strong turbulence (C²ₙ = 10⁻¹³), same shared-seed protocol
(`correlated_raw.csv` / `correlated_summary.csv`, plots in
`plots/correlated_*.png`):

| coherence config | policy | reward | PDR |
|---|---|---|---|
| i.i.d. (control) | PPO / best-static (tie) | −730.2 ± 22.4 | 0.724 ± 0.009 |
| τ_L/τ_S = 100/20 ms | PPO (original obs) | −814.3 ± 97.1 | 0.690 ± 0.040 |
| τ 100/20 ms | PPO (PER obs) / best-static (tie) | −717.0 ± 77.1 | 0.730 ± 0.032 |
| τ 500/100 ms | PPO / best-static (tie) | −729.5 ± 183.7 | 0.724 ± 0.075 |
| τ 500/100 ms, 50 ms steps | PPO (PER obs) / best-static (tie) | −728.8 ± 177.3 | 0.725 ± 0.073 |
| τ 500/100 ms, 50 ms steps | PPO (160k steps, entropy 0.03) | −840.1 ± 189.4 | 0.679 ± 0.078 |

**The honest verdict: no.** Every converged PPO policy is a *constant-route*
policy — the paired per-episode comparison at the strongest correlation
point shows PPO byte-identical to a fixed route on all 10 episodes, zero
route switches, even with 50 ms decision steps (10 decisions per
large-scale fade epoch), a cleaner observation, and an entropy/budget bump.

What the iteration *did* fix: with the original drop-count observation,
PPO at τ 100/20 ms converged to the **wrong** route (−814 vs −717). Giving
the observation the loss model's instantaneous per-link PER instead made it
find the optimal route reliably in every config. Better perception improved
route *selection*, not route *switching*.

Why switching never emerges, mechanically: the per-episode reward variance
under correlation is huge (std ~180, ~25% of the mean) while the expected
gain from a single well-timed switch is small — fade epochs hit all links,
routes share nodes, and the flap penalty plus the transient drops during a
switch eat much of the margin. A memoryless MLP policy facing that
signal-to-noise ratio rationally collapses onto the best constant action.
The correlated channel makes adaptation *possible*, not *profitable at
this margin*.

What would give adaptation a real edge (written as future work at the
time; **all of it was implemented and tested in Phase 7 below**): a
recurrent policy or frame-stacked observations so fade trajectories are
visible, not just snapshots; disjoint routes so a fade on one route leaves
an alternative genuinely clean; TCP traffic, where drops compound through
congestion control and dodging a fade epoch pays non-linearly; and a lower
flap penalty during exploration.

## Phase 7: the adaptation study

Phase 6 ended with a mechanism finding: switching never paid because the
pentagon's routes share links (one fade epoch degrades several routes at
once) and UDP loss is linear in drop count. Phase 7a changed both
conditions — a topology whose four 0→3 routes are pairwise
**link-disjoint** (fade epochs independent per route) and optional TCP
traffic (drops compound through congestion control) — and this study
asks the headline question one more time: **does PPO beat the best
static route when adaptation is genuinely profitable?**

Three configs, all at strong turbulence (C²ₙ = 10⁻¹³) on the disjoint
topology, same shared-seed protocol (eval seeds 100–109;
`adaptation_raw.csv` / `adaptation_summary.csv`, plots in
`plots/adaptation_*.png`). The correlated cells run 50 ms decision
steps: a probe of the linkPer observation on this topology (held route,
3 episodes × 7 links) measured a lag-1 autocorrelation across steps of
0.46 at 50 ms vs 0.28 at 100 ms — the disjoint links are longer than
the pentagon's, so the channel decorrelates faster per step — and 50 ms
gives ~10 decisions per large-scale fade epoch (τ_L = 500 ms) instead
of 5. 200 steps keep the 10 s episode; training and evaluation share
each cell's step settings. PPO trains fresh per cell (80k steps, reward
plateaued by the second fifth of every run; seed 42, disjoint from the
eval seeds).

| config | policy | reward | PDR | goodput [Mbps] | switches/ep |
|---|---|---|---|---|---|
| i.i.d. + UDP (control) | PPO / best-static route 0 (tie) | −522.3 ± 21.2 | 0.796 ± 0.009 | — | 0 |
| i.i.d. + UDP | greedy-PER (scripted) | −850.2 ± 28.7 | 0.760 ± 0.007 | — | 44.6 |
| i.i.d. + UDP | AODV | −548.9 ± 26.1 | 0.793 ± 0.015 | — | — |
| τ 500/100 ms + UDP | PPO (≡ route 3) | −618.0 ± 165.7 | 0.770 ± 0.068 | — | 1.0 |
| τ 500/100 ms + UDP | PPO 8-frame stack, 160k (≡ route 1) | −748.0 ± 109.6 | 0.717 ± 0.045 | — | 1.0 |
| τ 500/100 ms + UDP | best static (route 0) | −572.9 ± 140.1 | 0.776 ± 0.057 | — | 0 |
| τ 500/100 ms + UDP | **greedy-PER (scripted)** | **−484.5 ± 113.8** | **0.914 ± 0.026** | — | 46.0 |
| τ 500/100 ms + UDP | AODV | −606.2 ± 145.7 | 0.769 ± 0.059 | — | — |
| τ 500/100 ms + TCP | PPO (≡ route 2) | −2120.2 ± 404.0 | 0.730 ± 0.302 | 0.274 | 1.0 |
| τ 500/100 ms + TCP | PPO 8-frame stack, 160k (≡ route 3) | −2173.5 ± 381.5 | 0.585 ± 0.406 | 0.229 | 1.0 |
| τ 500/100 ms + TCP | best static (route 0) | −1897.6 ± 453.2 | 0.870 ± 0.163 | 0.451 | 0 |
| τ 500/100 ms + TCP | **greedy-PER (scripted)** | **−1726.8 ± 569.6** | **0.928 ± 0.083** | **0.788** | 46.0 |
| τ 500/100 ms + TCP | AODV | −1843.7 ± 423.8 | 0.888 ± 0.151 | 0.502 | — |

Paired per-episode analysis (same seeds, PPO minus best-static;
`parse_traces.py --study adaptation --paired`):

| config | reward Δ | PDR Δ | W/T/L | PPO switches | identical to |
|---|---|---|---|---|---|
| i.i.d. + UDP | 0.0 ± 0.0 | +0.000 | 0/10/0 | 0 | route 0 |
| τ 500/100 + UDP | −45.1 ± 231.9 | −0.006 | 5/0/5 | 1.0 (max 1) | route 3 |
| τ 500/100 + TCP | −222.6 ± 400.3 | −0.141 | 1/0/9 | 1.0 (max 1) | route 2 |

**The verdict: no.** PPO does not beat the best static route in any
config — it ties the control exactly (as it should: under i.i.d. fading
holding the best route is optimal, and PPO finds it) and *loses* both
correlated cells. Every trained policy, in every cell, is a
constant-route policy: its per-episode rewards are byte-identical to
one static route on all 10 seeds, and its single "switch" is the first
step onto that route. Which route it locks onto is a training-seed
lottery (route 3, then 1, then 2 across runs) — under correlated
fading the per-episode return variance (std ≈ 150–450, up to ~25% of
the mean) swamps the between-route mean differences, so whichever
route looked best in the early rollouts wins and the policy entropy
collapses onto it (measured H ≈ 0.005 nats vs 1.386 uniform, max
softmax probability 0.9995, the same action at all 600 probed steps —
the policy is state-independent).

The environment is no longer the excuse — that is the study's sharpest
result. The scripted greedy-PER baseline (hold the route, switch when
another route's summed link PER is 0.1 better — a *memoryless* rule
over the same observation PPO sees) beats best-static by +88.4 reward
(8 wins / 2 losses paired) and +13.8 pp PDR in the UDP cell, and by
+170.8 reward (8/2), +5.8 pp PDR, and +75% goodput (0.788 vs
0.451 Mbps) under TCP, switching ~46 times per 200-step episode and
paying every flap penalty. It loses in the i.i.d. control (−850 vs
−522), exactly as it should: switching on white noise is pure cost.
Adaptation on the disjoint topology is *possible and profitable*, and
TCP amplifies the payoff non-linearly, as the Phase 7 design predicted.
Even an episode-level oracle that merely picks the best fixed route per
seed would gain +78 reward over best-static in the UDP cell (the
winner flips 5/5 between routes 0 and 3 across seeds).

Bounded iteration performed, per the plan: constant-route check
(positive, all cells), entropy inspection (fully collapsed, above), and
frame-stacked observations (7b: `FlatFrameStack`, 8 frames = 400 ms of
history, 160k-step budget — twice the compute). The stacked variant
collapsed identically, just onto different (worse) routes. The 100 ms
decision step was not re-run: greedy-PER wins at 50 ms using the same
observation, so decision rate and observability are demonstrably not
the blocker, and Phase 6 already showed both step times collapse the
same way on the pentagon.

So the bottleneck has moved. Phase 6 said the *environment* gives
switching no margin; Phase 7 built the margin and showed a trivial
reactive controller collects it, while PPO — with the right
observation, enough decisions per fade epoch, memory, and double
budget — cannot escape the constant-route local optimum. The failure
is the *optimizer*, not the information: on-policy gradients under
this return variance rationally trade exploration for the safest
constant action long before a switching policy's advantage becomes
statistically visible to them. What that points to next (out of scope
here): variance-reduced training (much longer horizons, many parallel
episodes per update, or seed-paired advantage baselines), imitation
warm-start from greedy-PER, or reward shaping that pays for correct
switches directly rather than through the episode return.

Two honest footnotes. First, AODV — dismissed since Phase 5 as the
weakest classical baseline — beats PPO in both correlated cells and
nearly matches best-static under TCP (−1843.7): its route discovery is
crude, but it *does* react, and under TCP its control-packet losses are
absorbed by retransmission. Reactivity per se is worth more than
learned route selection here. Second, under TCP the constant-route
policies' PDR distributions are bimodal (PPO std 0.302, three episodes
with PDR ≈ 0): when the locked route's link is deep-faded at episode
start, the TCP handshake itself can fail and nothing flows for the
whole episode — a failure mode adaptive policies simply do not have.

## Phase 8: the imitation-then-RL study

Phase 7 ended with the bottleneck moved onto the optimizer: adaptation
is provably profitable (the scripted greedy-PER teacher collects the
margin) and PPO trained from scratch always collapses to a constant
route. Phase 8 separates the two remaining hypotheses. Behavior-clone
the teacher — a switching policy obtained *without any exploration* —
then fine-tune it with PPO. If fine-tuning holds or improves the
switching policy, exploration (finding it) was the bottleneck; if it
degrades the policy back to a constant route, the on-policy gradient
itself is.

Setup (same two correlated cells, seeds, and env settings as Phase 7,
so the ppo / best-static / greedy-per rows below are the committed
`adaptation_raw.csv` measurements, not re-runs): per cell, 25 teacher
episodes on training seeds 42–66 (5 000 (obs, action) pairs), a
cross-entropy fit of the actor head with a 5-episode held-out split
(`bc`), an 8 000-step value warmup on rollouts from the frozen BC
policy — the fresh critic would otherwise feed garbage advantages to
the first updates — then an 80 000-step PPO fine-tune (`bc-ppo`, the
7c budget class, identical hyperparameters). Machinery:
`agent/teacher.py`, `agent/imitation.py`,
`run_benchmark.py --study imitation`.

**BC fidelity.** Validation action-match accuracy is 0.626 (UDP) /
0.631 (TCP) against a 0.372 majority-class baseline — capped well
below 1.0 for a structural reason: the teacher's hysteresis state (the
currently-held route) is not in the observation, so a stateless MLP
cannot express the hold rule, only the argmin-PER preference.
Behaviorally the clone is nonetheless a faithful fade-dodger: it
matches the teacher's PDR (0.913 vs 0.914 UDP) and goodput (0.799 vs
0.788 Mbps TCP) and reproduces its PHY-drop profile (213 vs 210
drops/ep UDP), but switches ~73 times per episode where the teacher
switches 46 — and the reward gap to the teacher is almost exactly the
flap-penalty bill for those extra switches ((73.3 − 46.0) × 5 ≈ 137 vs
a measured Δ of 134 UDP / 124 TCP). BC lands between best-static and
the teacher in the TCP cell (beats best-static 6/4 paired, +47.2) and
just below best-static in the UDP cell (4/6, −45.0, the flap tax)
while delivering +13.7 pp PDR there.

| config | policy | reward | PDR | goodput [Mbps] | switches/ep |
|---|---|---|---|---|---|
| τ 500/100 ms + UDP | PPO scratch (7c, ≡ route 3) | −618.0 ± 165.7 | 0.770 ± 0.068 | — | 1.0 |
| τ 500/100 ms + UDP | **bc** (clone of teacher) | −618.0 ± 117.4 | **0.913 ± 0.029** | — | 73.3 |
| τ 500/100 ms + UDP | **bc-ppo** (fine-tuned, ≡ route 0) | −572.9 ± 140.1 | 0.776 ± 0.057 | — | 0.0 |
| τ 500/100 ms + UDP | best static (route 0) | −572.9 ± 140.1 | 0.776 ± 0.057 | — | 0 |
| τ 500/100 ms + UDP | greedy-PER (teacher) | **−484.5 ± 113.8** | 0.914 ± 0.026 | — | 46.0 |
| τ 500/100 ms + TCP | PPO scratch (7c, ≡ route 2) | −2120.2 ± 404.0 | 0.730 ± 0.302 | 0.274 | 1.0 |
| τ 500/100 ms + TCP | **bc** (clone of teacher) | −1850.4 ± 548.8 | 0.918 ± 0.112 | **0.799** | 73.5 |
| τ 500/100 ms + TCP | **bc-ppo** (fine-tuned) | −1925.5 ± 444.1 | 0.869 ± 0.163 | 0.462 | 8.2 |
| τ 500/100 ms + TCP | best static (route 0) | −1897.6 ± 453.2 | 0.870 ± 0.163 | 0.451 | 0 |
| τ 500/100 ms + TCP | greedy-PER (teacher) | **−1726.8 ± 569.6** | **0.928 ± 0.083** | 0.788 | 46.0 |

Paired per-episode analysis (shared seeds;
`parse_traces.py --study imitation --paired`):

| comparison | config | reward Δ | PDR Δ | W/T/L | notes |
|---|---|---|---|---|---|
| bc − best-static | UDP | −45.0 ± 139.5 | +0.137 | 4/0/6 | the flap tax |
| bc − best-static | TCP | +47.2 ± 298.0 | +0.047 | 6/0/4 | profitable start |
| bc-ppo − best-static | UDP | 0.0 ± 0.0 | +0.000 | 0/10/0 | byte-identical to route 0 |
| bc-ppo − best-static | TCP | −27.9 ± 28.0 | −0.001 | 1/0/9 | |
| bc-ppo − bc | UDP | +45.0 ± 139.5 | −0.137 | 6/0/4 | switching traded for reward |
| bc-ppo − bc | TCP | −75.1 ± 308.9 | −0.048 | 4/0/6 | strictly degraded |

**The verdict: fine-tuning destroys the switching policy in both
cells — the on-policy gradient itself is the bottleneck, not
exploration.** The trajectory (`plots/imitation_trajectory.png`;
per-update CSVs `imitation_trajectory_*.csv` — this trajectory is the
result) shows it happening. The BC policy starts at entropy ≈ 0.65
nats with ~80 greedy switches per episode. In the UDP cell PPO drains
both monotonically: by update ~40 (20k steps) switching has halved, by
update ~93 (46k steps) the policy is fully collapsed — entropy < 0.01
(the Phase 7c signature), zero switches, KL from the BC policy
climbing to ~5.5 nats — and it stays there for the remaining 40k
steps. Evaluation confirms total collapse: bc-ppo's per-episode
rewards are byte-identical to static route 0 on all 10 seeds. The TCP
cell resists longer (the switching payoff is larger): the decay is
non-monotonic with partial recoveries near updates 105–130, but it
too reaches the collapsed state by update ~145, and its end-of-run
twitches leave a nearly-constant policy (8.2 switches/ep) that loses
to plain best-static 9/10 episodes.

Two honest readings of the UDP cell. First, the collapse there is
locally reward-rational: the hysteresis-free clone pays ~137/episode
in flap penalties, so best-static (−572.9) really is better than the
BC start (−618.0), and PPO cashes that in (+45, 6/4 vs bc) — it even
collapses onto the *correct* constant route (route 0, the true
best-static, where scratch runs lottery onto routes 1–3). But a
strictly better policy sat one hysteresis away: the teacher (−484.5)
switches 27 fewer times over the same fades the clone already dodges,
and gradient descent on the switching side of policy space could
plausibly have found it (switch less when margins are small) — instead
the gradient exited the switching regime entirely and never returned:
once entropy is drained, no gradient path leads back out. Second, the
TCP cell removes even that excuse: there the BC start was already
*better* than best-static (+47.2, 6/4) and the teacher better still
(+170.8, 8/2), yet fine-tuning still walked the policy down to the
constant-route attractor (−75.1 vs its own initialisation, 4/6). PPO
did not fail to *find* the switching policy — it was handed one, with
a warmed-up critic, and dismantled it. Under this return variance
(per-episode std 140–570, larger than every between-policy margin),
the surviving-action-probability dynamics of the on-policy gradient
point one way: toward the constant route.

What this pins down for the write-up: exploration is ruled out as the
binding constraint; the fix must change the gradient's information,
not the starting point — variance reduction (seed-paired advantage
baselines, many-episode updates, much longer horizons), rewards that
pay for correct switches directly, or off-policy/imitation-regularized
objectives (e.g. a KL leash to the teacher, which the KL panel shows
would bind early: the policy leaves the BC neighbourhood within ~10
updates of unfreezing).

Reproduction: BC datasets and intermediate artifacts live in
`results/checkpoints/` (gitignored; the four bc/bc-ppo policy
checkpoints *are* committed there, ~55–160 KB each).
`python run_benchmark.py --study imitation` regenerates everything
(datasets, BC, warmup + fine-tune, eval; ~11 min); per-cell re-runs
via `--regime`, full retrain via `--retrain`.

## Reproducing

```bash
./setup/link_fso_modules.sh          # symlink + build fso-channel,
                                     # fso-rl-env, fso-aodv-baseline
cd ns3-rl-router
"$(brew --prefix python@3.11)/bin/python3.11" -m venv agent/.venv
agent/.venv/bin/pip install -r requirements.txt \
    -e ~/fso-tools/ns-3-dev/contrib/ai/python_utils \
    -e ~/fso-tools/ns-3-dev/contrib/ai/model/gym-interface/py
source agent/.venv/bin/activate      # env python3 must be this 3.11
cd ../benchmarks
python run_benchmark.py              # ~25 min (training dominates)
python parse_traces.py               # prints table, writes summary.csv
python plot_results.py               # writes plots/*.png
```

The Phase 6 study: `python run_benchmark.py --study correlated` (then
`parse_traces.py --study correlated` and `plot_results.py --study
correlated`); `--coherence` re-runs a single coherence config.

The Phase 7 study: `python run_benchmark.py --study adaptation` (then
`parse_traces.py --study adaptation [--paired]` and `plot_results.py
--study adaptation`); `--regime` re-runs a single config. The ppo-stack
variant trains at 160k steps (~9 min per correlated cell); everything
else matches the Phase 6 budgets.

The Phase 8 study: `python run_benchmark.py --study imitation` (then
`parse_traces.py --study imitation [--paired]` and `plot_results.py
--study imitation`). The parser and plots pull the Phase 7 baselines
from the committed `adaptation_raw.csv` for comparison; only the
bc/bc-ppo rows are measured by this study.

`run_benchmark.py --quick` smoke-tests the pipeline in ~1 min;
`--regime`/`--policy` re-run a single cell (rows are replaced in place).
Per-regime PPO checkpoints are cached in `checkpoints/` (gitignored;
delete or pass `--retrain` to retrain). Training budgets: 20k steps
(weak), 40k (moderate), 80k (strong), ~8 min for the strong regime on
an M-series MacBook; the full study is ~10 min of evaluation on top.
