# Benchmark results (Phases 5 & 6)

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

What would give adaptation a real edge (future work, in rough order of
expected payoff): a recurrent policy or frame-stacked observations so fade
trajectories are visible, not just snapshots; disjoint routes so a fade on
one route leaves an alternative genuinely clean; TCP traffic, where drops
compound through congestion control and dodging a fade epoch pays
non-linearly; and a lower flap penalty during exploration.

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

`run_benchmark.py --quick` smoke-tests the pipeline in ~1 min;
`--regime`/`--policy` re-run a single cell (rows are replaced in place).
Per-regime PPO checkpoints are cached in `checkpoints/` (gitignored;
delete or pass `--retrain` to retrain). Training budgets: 20k steps
(weak), 40k (moderate), 80k (strong), ~8 min for the strong regime on
an M-series MacBook; the full study is ~10 min of evaluation on top.
