"""Benchmark orchestrator: PPO vs classical routing baselines.

Three studies share this orchestrator:

* ``--study turbulence`` (Phase 5, default) sweeps the three turbulence
  regimes (C2n weak 1e-17, moderate 1e-15, strong 1e-13) with i.i.d.
  block fading; results go to results/raw_results.csv.
* ``--study correlated`` (Phase 6) fixes strong turbulence (C2n 1e-13)
  and sweeps fading coherence times (COHERENCE_CONFIGS) to test whether
  PPO can exploit channel memory; results go to
  results/correlated_raw.csv with the coherence config name in the
  ``regime`` column.
* ``--study adaptation`` (Phase 7) fixes strong turbulence on the
  link-disjoint topology and sweeps the environment conditions that
  Phase 6 identified as blocking profitable route switching
  (ADAPTATION_CONFIGS: i.i.d. UDP control, correlated fading + UDP,
  correlated fading + TCP); results go to results/adaptation_raw.csv.
  Rows additionally carry goodput/retx (TCP) and the number of
  within-episode route switches.
* ``--study imitation`` (Phase 8) reuses the two correlated adaptation
  cells and runs the imitation-then-RL pipeline per cell: collect a
  greedy-PER teacher dataset on training seeds, behavior-clone it,
  value-warmup + PPO fine-tune (agent/imitation.py), then evaluate the
  ``bc`` and ``bc-ppo`` policies on the shared eval seeds into
  results/imitation_raw.csv. The baselines those rows are compared
  against (ppo, statics, greedy-per) come from the committed Phase 7
  adaptation_raw.csv — same seeds, same settings, not re-run.
* ``--study offpolicy`` (Phase 9) reuses the same two correlated cells
  and trains Double DQN per cell twice — ``dqn-scratch`` (fresh
  Q-network) and ``dqn-bc`` (Q-network initialised from the committed
  Phase 8 BC checkpoint, low initial epsilon) — via agent/dqn.py, then
  evaluates both greedy policies on the shared eval seeds into
  results/offpolicy_raw.csv. Baselines (ppo, statics, greedy-per, bc,
  bc-ppo) again come from the committed Phase 7/8 CSVs, not re-runs.
* ``--study routeaware`` (Phase 10) re-runs the decisive Phase 8/9 arms
  with the env's ``routeInObs`` flag on (a one-hot of the currently
  held route joins the observation, 28 -> 32 dims), testing the
  surviving hypothesis: every learned policy failed to express the
  teacher's hysteresis because hold-vs-switch was unobservable. Per
  cell: collect a route-aware teacher dataset (verifying the one-hot
  matches the teacher's held route on every step), clone it
  (``bc-route``), value-warmup + PPO fine-tune (``bc-ppo-route``,
  Phase 8 protocol), and train Double DQN from scratch
  (``dqn-scratch-route``) and from the fresh BC checkpoint
  (``dqn-bc-route``, Phase 9 protocol); results go to
  results/routeaware_raw.csv. The 28-dim counterparts and baselines
  come from the committed Phase 7/8/9 CSVs, not re-runs.

Policies compared per sweep point on the 5-node FSO mesh:

* ``ppo``          — PPO trained fresh per regime (checkpoint cached under
                     results/checkpoints/, reused unless --retrain).
* ``ppo-transfer`` — the committed Phase 4 checkpoint (trained at 1e-13)
                     evaluated cross-regime for a generalization datapoint.
* ``static-0..3``  — each fixed route of the 0->3 flow; parse_traces.py
                     reports the best one per regime as ``best-static``.
* ``random``       — uniform random route choice each step.
* ``aodv``         — ns-3 AODV via the fso-aodv-baseline scratch program.

All policies are evaluated on the same fixed ns-3 run numbers
(seed, seed+1, ...) so they face identical fading realisations. Raw
per-episode rows are written to the study's raw CSV; rows for
(regime, policy) pairs re-run later replace the old ones in place.

Process model: ns3-ai's shared-memory interface is once-per-process (the
segment lives in a C++ function-local static), so every phase that owns
a gym env runs in its own subprocess — training shells out to the
Phase 4 train.py CLI, and env evaluation re-invokes this script with
--eval-worker, which prints one ``FSO-ROW <csv>`` line per episode.

Prerequisites: modules linked and built (setup/link_fso_modules.sh) and
the agent venv active (`env python3` must resolve to Python 3.11).

Typical usage:
    $ python run_benchmark.py                     # full Phase 5 study
    $ python run_benchmark.py --quick             # 2-episode smoke run
    $ python run_benchmark.py --regime strong --policy aodv
    $ python run_benchmark.py --study correlated  # full Phase 6 study
    $ python run_benchmark.py --study correlated --coherence tau500-100
    $ python run_benchmark.py --study adaptation  # full Phase 7 study
    $ python run_benchmark.py --study adaptation --regime disjoint-tau500-tcp
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

_REPO_DIR = Path(__file__).resolve().parent.parent
_AGENT_DIR = _REPO_DIR / "ns3-rl-router" / "agent"
_SIM_DIR = _REPO_DIR / "ns3-rl-router" / "sim"
for _dir in (str(_AGENT_DIR), str(_SIM_DIR)):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

from teacher import (  # noqa: E402
    DEFAULT_MARGIN as GREEDY_MARGIN,
    DISJOINT_ROUTE_LINKS,
    PENTAGON_ROUTE_LINKS as ROUTE_LINKS,
    GreedyPerTeacher,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RAW_CSV = RESULTS_DIR / "raw_results.csv"
CORRELATED_RAW_CSV = RESULTS_DIR / "correlated_raw.csv"
ADAPTATION_RAW_CSV = RESULTS_DIR / "adaptation_raw.csv"
IMITATION_RAW_CSV = RESULTS_DIR / "imitation_raw.csv"
OFFPOLICY_RAW_CSV = RESULTS_DIR / "offpolicy_raw.csv"
ROUTEAWARE_RAW_CSV = RESULTS_DIR / "routeaware_raw.csv"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
PHASE4_CHECKPOINT = _AGENT_DIR / "checkpoints" / "ns3_ppo.pt"
DEFAULT_SIM_CONFIG = _REPO_DIR / "ns3-rl-router" / "config" / "sim_config.yaml"

REGIMES: dict[str, str] = {"weak": "1e-17", "moderate": "1e-15", "strong": "1e-13"}

# Per-regime training budgets [env steps]. Phase 4's strong-regime reward
# plateaued by ~20k of 80k steps; calmer regimes have a flatter reward
# landscape (few or no drops) and need less exploration, so budgets scale
# with turbulence while keeping total training around the Phase 4 wall time.
TRAIN_STEPS: dict[str, int] = {"weak": 20_000, "moderate": 40_000, "strong": 80_000}

POLICIES = ("ppo", "ppo-transfer", "static-0", "static-1", "static-2", "static-3",
            "random", "aodv")

# Phase 6 iteration variants: ppo-per is PPO trained after the env's
# drop-rate observation was replaced by the per-link PER (the correlated
# channel state, visible for off-route links too); the first-pass "ppo"
# rows of the correlated study predate that observation change.
# ppo-per-ent additionally trains with entropy_coef 0.03 and a 160k-step
# budget (trained via the train.py CLI, see results/README.md; this
# orchestrator only evaluates its checkpoint). greedy-per is a scripted
# reactive baseline: hold the current route, switch to the route with the
# lowest summed link PER when it beats the current one by GREEDY_MARGIN.
# ppo-stack is the Phase 7b policy-memory variant: PPO over
# FRAME_STACK_K stacked observations (train and eval both wrap the env
# in FlatFrameStack).
ALL_POLICIES = ("ppo", "ppo-per", "ppo-per-ent", "ppo-stack", "greedy-per",
                "bc", "bc-ppo", "dqn-scratch", "dqn-bc",
                "bc-route", "bc-ppo-route", "dqn-scratch-route",
                "dqn-bc-route", *POLICIES[1:])

# Route tables (ROUTE_LINKS/DISJOINT_ROUTE_LINKS) and the greedy-per
# hysteresis margin are imported from agent/teacher.py, the shared home
# of the scripted greedy-PER policy since Phase 8.


@dataclass(frozen=True)
class CoherenceConfig:
    """One environment sweep point (Phase 6 coherence, Phase 7 adaptation).

    Attributes:
        coherence_large: Large-scale fading coherence time (ns-3 Time
            string; "0ms" means i.i.d. block fading).
        coherence_small: Small-scale fading coherence time (same format).
        step_time_s: Decision interval override [s]; None keeps the
            sim_config default (0.1 s).
        episode_steps: Decision steps per episode override; None keeps
            the sim_config default (100). Scale it with 1/step_time_s so
            every sweep point simulates the same 10 s episode.
        topology: Mesh layout override ("pentagon"/"disjoint"); None
            keeps the sim_config default (pentagon).
        traffic_protocol: Transport override of the 0->3 flow
            ("udp"/"tcp"); None keeps the sim_config default (udp).
        route_in_obs: Route-aware observation override ("true"/"false",
            Phase 10: a current-route one-hot joins the observation);
            None keeps the sim_config default (false).
    """

    coherence_large: str
    coherence_small: str
    step_time_s: str | None = None
    episode_steps: str | None = None
    topology: str | None = None
    traffic_protocol: str | None = None
    route_in_obs: str | None = None


# Phase 6 coherence sweep, all at strong turbulence (C2n = CORRELATED_C2N).
# "iid" is the tau=0 control replicating the Phase 5 strong regime. A probe
# of the dropRate observation's lag-1 autocorrelation under a held route
# (3 episodes x 2 links) measured: iid ~0.0; tau 100/20 ms 0.20 at 0.1 s
# steps but 0.29 at 0.05 s; tau 500/100 ms 0.42 at 0.1 s steps. So the
# tau100-20 point runs 0.05 s decision steps (200 steps = same 10 s episode)
# to keep the channel state observable across steps; tau500-100 keeps the
# Phase 5 step time for direct comparability with the strong regime.
COHERENCE_CONFIGS: dict[str, CoherenceConfig] = {
    "iid": CoherenceConfig("0ms", "0ms"),
    "tau100-20": CoherenceConfig("100ms", "20ms", "0.05", "200"),
    "tau500-100": CoherenceConfig("500ms", "100ms"),
    # Iteration cell: same channel as tau500-100 but 50 ms decision steps
    # (PER-state lag-1 autocorr 0.60 vs 0.42 across steps) so the agent
    # can react within a fade epoch; 200 steps keep the 10 s episode.
    "tau500-100-step50": CoherenceConfig("500ms", "100ms", "0.05", "200"),
}

CORRELATED_C2N = "1e-13"
CORRELATED_TRAIN_STEPS = 80_000

# ppo-transfer is a Phase 5 cross-regime datapoint; it adds nothing here.
CORRELATED_POLICIES = tuple(p for p in ALL_POLICIES if p != "ppo-transfer")

# Phase 7 adaptation study: strong turbulence on the link-disjoint
# topology (routes share no links, so fade epochs are independent per
# route). The iid cell is the control where holding the best route is
# provably optimal; the tau500 cells add channel memory (UDP) and then
# non-linear loss compounding (TCP). Both correlated cells use 50 ms
# decision steps: a probe of the linkPer observation on the disjoint
# topology at tau 500/100 ms (held route, 3 episodes x 7 links)
# measured a lag-1 autocorrelation across steps of 0.46 at 50 ms vs
# 0.28 at 100 ms — the disjoint links are longer than the pentagon's,
# so the channel decorrelates faster per step and the finer step is
# needed to keep the state observable — and 50 ms gives the agent ~10
# decisions per large-scale fade epoch (tau_L = 500 ms) instead of 5.
# 200 steps keep the 10 s episode; train and eval share each cell's
# step settings.
ADAPTATION_CONFIGS: dict[str, CoherenceConfig] = {
    "disjoint-iid-udp": CoherenceConfig(
        "0ms", "0ms", topology="disjoint", traffic_protocol="udp"),
    "disjoint-tau500-udp": CoherenceConfig(
        "500ms", "100ms", "0.05", "200",
        topology="disjoint", traffic_protocol="udp"),
    "disjoint-tau500-tcp": CoherenceConfig(
        "500ms", "100ms", "0.05", "200",
        topology="disjoint", traffic_protocol="tcp"),
}

ADAPTATION_TRAIN_STEPS = 80_000
ADAPTATION_POLICIES = ("ppo", "ppo-stack", "static-0", "static-1", "static-2",
                       "static-3", "greedy-per", "random", "aodv")

# Phase 7b iteration: the 80k "ppo" runs converged to constant-route
# policies with fully collapsed entropy (H ~ 0.005 nats vs 1.386
# uniform), so ppo-stack gives the policy memory — 8 stacked
# observations cover 400 ms, most of a tau_L = 500 ms fade epoch at
# 50 ms steps — and double the budget (the input is 8x wider). The
# stacked variant is only meaningful where the channel has memory; the
# orchestrator skips it in the iid control cell.
FRAME_STACK_K = 8
VARIANT_TRAIN_STEPS = {"ppo-stack": 160_000}
STACK_SKIP_REGIMES = ("disjoint-iid-udp",)

# Phase 8 imitation study: same two correlated cells as Phase 7 (the
# i.i.d. control is pointless here — the teacher provably loses on
# white noise), same eval seeds and env settings, so imitation_raw.csv
# rows are directly comparable with the committed adaptation_raw.csv
# baselines. Per cell: 25 teacher episodes on training seeds 42..66
# (5000 pairs at 200 steps/episode), BC, 8k-step value warmup on the
# frozen BC policy, 80k-step PPO fine-tune (the 7c budget class).
IMITATION_CONFIGS: dict[str, CoherenceConfig] = {
    name: ADAPTATION_CONFIGS[name]
    for name in ("disjoint-tau500-udp", "disjoint-tau500-tcp")
}
IMITATION_POLICIES = ("bc", "bc-ppo")
BC_DATASET_EPISODES = 25
BC_EPOCHS = 40
BC_WARMUP_STEPS = 8_000
IMITATION_TRAIN_STEPS = 80_000

# Phase 9 off-policy study: the same two correlated cells, completing
# the {PPO, DQN} x {scratch, BC-init} 2x2. Per cell, Double DQN trains
# twice (agent/dqn.py): dqn-scratch explores from a fresh Q-network
# (epsilon 1.0 -> 0.05 over the first half of the budget), dqn-bc
# initialises the Q-network from the committed Phase 8 BC checkpoint
# and keeps epsilon LOW from the start (0.05 -> 0.01) so random actions
# do not wash the handed policy out of the replay buffer. The BC-init
# Q-values are the BC logits shifted by a constant (argmax-invariant)
# offset onto the return scale: mean step reward of the committed bc
# eval rows / (1 - gamma). Budget matches Phase 8's fine-tune.
OFFPOLICY_CONFIGS = IMITATION_CONFIGS
OFFPOLICY_POLICIES = ("dqn-scratch", "dqn-bc")
OFFPOLICY_TRAIN_STEPS = 80_000
DQN_GAMMA = 0.99
DQN_SCRATCH_EPS = ("1.0", "0.05", 40_000)
DQN_BC_EPS = ("0.05", "0.01", 20_000)

# Phase 10 route-aware study: the same two correlated cells with the
# env's routeInObs flag on (obs 28 -> 32: a one-hot of the currently
# held route). The four arms re-run the decisive Phase 8/9 protocols on
# the widened observation — bc-route (clone the teacher; its hysteresis
# state is now observable), bc-ppo-route (value warmup + PPO fine-tune,
# Phase 8 budgets), dqn-scratch-route and dqn-bc-route (Double DQN,
# Phase 9 epsilon schedules; the BC init comes from this study's fresh
# 32-dim bc-route checkpoint). The dqn-bc-route Q-offset is derived
# from the collected dataset's teacher episode rewards (the committed
# Phase 9 recipe used the bc eval rows, which for the route-aware clone
# do not exist before training; the teacher's reward is the scale the
# clone is expected to land on).
ROUTEAWARE_CONFIGS: dict[str, CoherenceConfig] = {
    name: CoherenceConfig(**{**asdict(config), "route_in_obs": "true"})
    for name, config in IMITATION_CONFIGS.items()
}
ROUTEAWARE_POLICIES = ("bc-route", "bc-ppo-route", "dqn-scratch-route",
                       "dqn-bc-route")
ROUTEAWARE_TRAIN_STEPS = 80_000

CSV_FIELDS = ("regime", "c2n", "policy", "episode", "sim_seed", "reward",
              "drops", "tx_pkts", "rx_pkts", "pdr", "mean_delay_ms",
              "goodput_mbps", "retx", "switches")

ROW_PREFIX = "FSO-ROW "


@dataclass
class EpisodeRow:
    """One evaluation episode of one policy in one regime.

    Attributes:
        regime: Turbulence regime name (weak/moderate/strong).
        c2n: Refractive index structure parameter [m^-2/3].
        policy: Policy identifier (see POLICIES).
        episode: Episode index within the evaluation run.
        sim_seed: ns-3 run number the episode used.
        reward: Total (undiscounted) episode reward.
        drops: PHY packets lost to fading, all links, whole episode.
        tx_pkts: Flow packets sent by the source.
        rx_pkts: Flow packets delivered to the sink.
        pdr: Flow packet delivery ratio (0 when nothing sent).
        mean_delay_ms: Packet-weighted mean end-to-end delay [ms].
        goodput_mbps: Mean per-step application goodput [Mbps]; 0 for
            UDP episodes (the field is TCP-only).
        retx: TCP data segments retransmitted over the episode; 0 for UDP.
        switches: Route changes within the episode (steps whose route
            differs from the previous step's; the episode starts on
            route 0). 0 for AODV, whose routing is not step-observable.
    """

    regime: str
    c2n: str
    policy: str
    episode: int
    sim_seed: int
    reward: float
    drops: int
    tx_pkts: int
    rx_pkts: int
    pdr: float
    mean_delay_ms: float
    goodput_mbps: float = 0.0
    retx: int = 0
    switches: int = 0

    def to_csv_line(self) -> str:
        """Serialise as a raw CSV data line (CSV_FIELDS order)."""
        values = asdict(self)
        return ",".join(str(values[f]) for f in CSV_FIELDS)

    @classmethod
    def from_csv_line(cls, line: str) -> EpisodeRow:
        """Parse a line produced by :meth:`to_csv_line`.

        Args:
            line: Comma-separated values in CSV_FIELDS order.

        Returns:
            The parsed row.
        """
        parts = dict(zip(CSV_FIELDS, line.split(",")))
        return cls(regime=parts["regime"], c2n=parts["c2n"], policy=parts["policy"],
                   episode=int(parts["episode"]), sim_seed=int(parts["sim_seed"]),
                   reward=float(parts["reward"]), drops=int(parts["drops"]),
                   tx_pkts=int(parts["tx_pkts"]), rx_pkts=int(parts["rx_pkts"]),
                   pdr=float(parts["pdr"]), mean_delay_ms=float(parts["mean_delay_ms"]),
                   goodput_mbps=float(parts.get("goodput_mbps", 0.0)),
                   retx=int(parts.get("retx", 0)),
                   switches=int(parts.get("switches", 0)))


def _metrics_from_steps(step_fields: list[dict[str, str]],
                        reward_key: str | None = None) -> dict:
    """Fold per-step key=value dicts into episode metrics.

    Args:
        step_fields: One dict per step with drops/txPkts/rxPkts/meanDelayMs
            (and optionally a reward field).
        reward_key: Key holding the per-step reward; None if the caller
            accumulates reward itself.

    Returns:
        Dict with reward (0 if no reward_key), drops, tx_pkts, rx_pkts,
        pdr, packet-weighted mean_delay_ms, goodput_mbps (step mean; 0
        without TCP fields), retx, and switches (route changes across
        steps; 0 without route fields).
    """
    reward_total = 0.0
    drops = tx = rx = retx = switches = 0
    delay_weighted_sum = 0.0
    goodput_sum = 0.0
    route = "0"  # both env and sim start every episode on route 0
    for fields in step_fields:
        step_rx = int(fields.get("rxPkts", 0))
        drops += int(fields.get("drops", 0))
        tx += int(fields.get("txPkts", 0))
        rx += step_rx
        delay_weighted_sum += float(fields.get("meanDelayMs", 0.0)) * step_rx
        goodput_sum += float(fields.get("goodputMbps", 0.0))
        retx += int(fields.get("retx", 0))
        step_route = fields.get("route", route)
        if step_route != route:
            switches += 1
            route = step_route
        if reward_key is not None:
            reward_total += float(fields.get(reward_key, 0.0))
    return {
        "reward": reward_total,
        "drops": drops,
        "tx_pkts": tx,
        "rx_pkts": rx,
        "pdr": rx / tx if tx else 0.0,
        "mean_delay_ms": delay_weighted_sum / rx if rx else 0.0,
        "goodput_mbps": goodput_sum / len(step_fields) if step_fields else 0.0,
        "retx": retx,
        "switches": switches,
    }


# ---------------------------------------------------------------------------
# Worker: owns the single gym env this process is allowed to create
# ---------------------------------------------------------------------------


def run_env_episode(env, seed: int, action_fn: Callable[[np.ndarray], int]) -> dict:
    """Roll one episode of the gym env under the given action function.

    Args:
        env: The FSO ns3 environment.
        seed: ns-3 run number for this episode.
        action_fn: Maps the current observation to a route action.

    Returns:
        Dict with reward, drops, tx_pkts, rx_pkts, pdr, mean_delay_ms.
    """
    from eval_policy import parse_info

    obs, _ = env.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float32)
    reward_total = 0.0
    steps: list[dict[str, str]] = []
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(action_fn(obs))
        obs = np.asarray(obs, dtype=np.float32)
        reward_total += float(reward)
        steps.append(parse_info(info))
        done = terminated or truncated
    metrics = _metrics_from_steps(steps)
    metrics["reward"] = reward_total
    return metrics


def _load_agent(checkpoint: Path, obs_dim: int, n_actions: int):
    """Load a greedy-evaluation PPO agent from a checkpoint.

    Args:
        checkpoint: Path to the .pt checkpoint file.
        obs_dim: Flattened observation dimension.
        n_actions: Number of discrete actions.

    Returns:
        The loaded PPOAgent.
    """
    from ppo_agent import PPOAgent

    agent = PPOAgent(obs_dim, n_actions)
    agent.load(checkpoint)
    return agent


def _action_fn_for(policy: str, env, seed: int,
                   route_links: tuple = ROUTE_LINKS) -> Callable[[np.ndarray], int]:
    """Build the observation->action function for an env-driven policy.

    Args:
        policy: One of ppo, ppo-transfer, static-K, greedy-per, random.
        env: The gym env (for space sizes; already frame-stacked for
            the ppo-stack variant).
        seed: RNG seed for the random policy.
        route_links: Per-route link indices of the active topology
            (for greedy-per).

    Returns:
        Callable mapping an observation to a discrete action.
    """
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)
    if policy == "ppo-transfer":
        return _load_agent(PHASE4_CHECKPOINT, obs_dim, n_actions).act_greedy
    if policy.startswith("dqn"):
        from dqn import DQNAgent

        variant, regime = policy.split(":", 1)
        agent = DQNAgent(obs_dim, n_actions)
        agent.load(CHECKPOINTS_DIR / f"{variant.replace('-', '_')}_{regime}.pt")
        return agent.act_greedy
    if policy.startswith(("ppo", "bc")):
        variant, regime = policy.split(":", 1)
        checkpoint = CHECKPOINTS_DIR / f"{variant.replace('-', '_')}_{regime}.pt"
        return _load_agent(checkpoint, obs_dim, n_actions).act_greedy
    if policy.startswith("static-"):
        route = int(policy.split("-", 1)[1])
        return lambda _obs: route
    if policy == "greedy-per":
        # One teacher per evaluation run: as before Phase 8's refactor,
        # the held route deliberately persists across episodes.
        return GreedyPerTeacher(route_links, margin=GREEDY_MARGIN).act
    if policy == "random":
        rng = np.random.default_rng(seed)
        return lambda _obs: int(rng.integers(n_actions))
    raise ValueError(f"unknown env policy: {policy}")


def eval_worker(args: argparse.Namespace) -> None:
    """Evaluate all env-driven policies of one sweep point in this process.

    Creates the single allowed gym env, rolls the shared seed set for
    each policy, and prints one ``FSO-ROW <csv>`` line per episode on
    stdout (progress goes to stderr).

    Args:
        args: Parsed CLI namespace (regime, policies, episodes, seed,
            and optional c2n/coherence/step-time env overrides).
    """
    from ns3_env import make_ns3_env

    regime = args.regime[0]
    c2n = args.c2n or REGIMES[regime]
    policies = args.policy
    env = make_ns3_env(str(Path(args.sim_config).resolve()), c2n=c2n, seed=args.seed,
                       coherence_large=args.coherence_large,
                       coherence_small=args.coherence_small,
                       step_time_s=args.step_time,
                       episode_steps=args.episode_steps,
                       topology=args.topology,
                       traffic_protocol=args.traffic_protocol,
                       route_in_obs=args.route_in_obs)
    route_links = (DISJOINT_ROUTE_LINKS if args.topology == "disjoint"
                   else ROUTE_LINKS)
    stacked_env = None
    try:
        for policy in policies:
            worker_policy = policy
            if policy != "ppo-transfer" and policy.startswith(("ppo", "bc",
                                                               "dqn")):
                worker_policy = f"{policy}:{regime}"
            policy_env = env
            if policy.startswith("ppo-stack"):
                if stacked_env is None:
                    from frame_stack import FlatFrameStack

                    stacked_env = FlatFrameStack(env, FRAME_STACK_K)
                policy_env = stacked_env
            action_fn = _action_fn_for(worker_policy, policy_env, args.seed,
                                       route_links)
            for ep in range(args.episodes):
                metrics = run_env_episode(policy_env, args.seed + ep, action_fn)
                row = EpisodeRow(regime=regime, c2n=c2n, policy=policy, episode=ep,
                                 sim_seed=args.seed + ep, **metrics)
                print(ROW_PREFIX + row.to_csv_line(), flush=True)
                print(f"[{regime}] {policy} ep{ep}: reward={metrics['reward']:.1f} "
                      f"pdr={metrics['pdr']:.3f} "
                      f"delay={metrics['mean_delay_ms']:.3f}ms",
                      file=sys.stderr, flush=True)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Orchestrator: everything below runs env-owning phases in subprocesses
# ---------------------------------------------------------------------------


def _coherence_cli_args(coherence: CoherenceConfig | None,
                        step_flag: str = "--step-time") -> list[str]:
    """Build the env-override CLI arguments for a coherence sweep point.

    Args:
        coherence: The sweep point; None (turbulence study) adds nothing.
        step_flag: Flag name for the decision interval (train.py and the
            eval worker both use ``--step-time``).

    Returns:
        Argument list to append to a train.py or eval-worker command.
    """
    if coherence is None:
        return []
    args = ["--coherence-large", coherence.coherence_large,
            "--coherence-small", coherence.coherence_small]
    if coherence.step_time_s is not None:
        args += [step_flag, coherence.step_time_s]
    if coherence.episode_steps is not None:
        args += ["--episode-steps", coherence.episode_steps]
    if coherence.topology is not None:
        args += ["--topology", coherence.topology]
    if coherence.traffic_protocol is not None:
        args += ["--traffic-protocol", coherence.traffic_protocol]
    if coherence.route_in_obs is not None:
        args += ["--route-in-obs", coherence.route_in_obs]
    return args


def train_regime_policy(regime: str, c2n: str, total_steps: int,
                        checkpoint: Path, train_seed: int,
                        coherence: CoherenceConfig | None = None,
                        frame_stack: int = 1) -> None:
    """Train a fresh PPO policy for one sweep point via the train.py CLI.

    Runs in a subprocess because the ns3-ai shared-memory interface can
    only be created once per process.

    Args:
        regime: Sweep point name (for logging only).
        c2n: C2n value the env is trained at.
        total_steps: Environment steps of training.
        checkpoint: Where the trained weights are saved.
        train_seed: Global training seed (also the first episode's run
            number; disjoint from the evaluation seed range).
        coherence: Optional coherence sweep point (correlated study).
        frame_stack: Observation stack depth (1 = no stacking).

    Raises:
        subprocess.CalledProcessError: If training fails.
    """
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{regime}] training PPO for {total_steps} steps at C2n={c2n} ...",
          flush=True)
    rewards_csv = checkpoint.with_name(checkpoint.stem + "_rewards.csv")
    subprocess.run(
        [sys.executable, str(_AGENT_DIR / "train.py"), "--env", "ns3",
         "--c2n", c2n, "--total-steps", str(total_steps),
         "--rollout-steps", "500", "--seed", str(train_seed),
         "--checkpoint-path", str(checkpoint),
         "--rewards-csv", str(rewards_csv),
         "--frame-stack", str(frame_stack),
         *_coherence_cli_args(coherence)],
        cwd=_AGENT_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def _run_agent_stage(cmd: list[str], label: str) -> str:
    """Run one imitation stage (agent/imitation.py) in a subprocess.

    Args:
        cmd: Full command line (already includes sys.executable).
        label: Stage name for error messages.

    Returns:
        The stage's stdout.

    Raises:
        RuntimeError: If the stage exited non-zero.
    """
    proc = subprocess.run(cmd, cwd=_AGENT_DIR, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-4000:])
        raise RuntimeError(f"{label} failed (rc={proc.returncode})")
    return proc.stdout


def run_imitation_pipeline(regime: str, c2n: str, coherence: CoherenceConfig,
                           args: argparse.Namespace) -> None:
    """Collect + BC + fine-tune one imitation cell (Phase 8).

    Each stage runs in its own subprocess (collect and finetune own a
    gym env); existing artifacts are reused unless --retrain. The BC
    learning curve, fine-tune trajectory, and fine-tune episode rewards
    are written to results/ as committed CSVs.

    Args:
        regime: Imitation cell name (an ADAPTATION_CONFIGS key).
        c2n: Turbulence strength of the cell.
        coherence: Env sweep point of the cell.
        args: Orchestrator CLI namespace (quick/retrain/train knobs).
    """
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    imitation_py = str(_AGENT_DIR / "imitation.py")
    dataset = CHECKPOINTS_DIR / f"bc_dataset_{regime}.npz"
    bc_ckpt = CHECKPOINTS_DIR / f"bc_{regime}.pt"
    bc_ppo_ckpt = CHECKPOINTS_DIR / f"bc_ppo_{regime}.pt"
    env_args = ["--c2n", c2n, *_coherence_cli_args(coherence)]

    if args.retrain or not dataset.exists():
        episodes = 2 if args.quick else BC_DATASET_EPISODES
        print(f"[{regime}] collecting {episodes} teacher episodes ...", flush=True)
        out = _run_agent_stage(
            [sys.executable, imitation_py, "collect", "--out", str(dataset),
             "--episodes", str(episodes), "--seed", str(args.train_seed),
             *env_args],
            f"{regime} collect")
        print("\n".join(line for line in out.splitlines()
                        if line.startswith("[collect] wrote")
                        or line.startswith("[collect] teacher")), flush=True)
    else:
        print(f"[{regime}] reusing dataset {dataset}")

    if args.retrain or not bc_ckpt.exists():
        epochs = 5 if args.quick else BC_EPOCHS
        print(f"[{regime}] behavior cloning ({epochs} epochs) ...", flush=True)
        out = _run_agent_stage(
            [sys.executable, imitation_py, "bc", "--dataset", str(dataset),
             "--checkpoint", str(bc_ckpt), "--epochs", str(epochs),
             "--metrics-csv", str(RESULTS_DIR / f"imitation_bc_{regime}.csv")],
            f"{regime} bc")
        print("\n".join(line for line in out.splitlines()
                        if line.startswith("[bc]")), flush=True)
    else:
        print(f"[{regime}] reusing BC checkpoint {bc_ckpt}")

    if args.retrain or not bc_ppo_ckpt.exists():
        warmup = 500 if args.quick else BC_WARMUP_STEPS
        total = 1000 if args.quick else (args.train_steps
                                         or IMITATION_TRAIN_STEPS)
        print(f"[{regime}] value warmup ({warmup}) + PPO fine-tune ({total}) ...",
              flush=True)
        out = _run_agent_stage(
            [sys.executable, imitation_py, "finetune",
             "--bc-checkpoint", str(bc_ckpt),
             "--checkpoint", str(bc_ppo_ckpt),
             "--trajectory-csv",
             str(RESULTS_DIR / f"imitation_trajectory_{regime}.csv"),
             "--rewards-csv",
             str(RESULTS_DIR / f"imitation_finetune_rewards_{regime}.csv"),
             "--warmup-steps", str(warmup), "--total-steps", str(total),
             "--rollout-steps", "500", "--seed", str(args.train_seed),
             *env_args],
            f"{regime} finetune")
        print("\n".join(out.splitlines()[-2:]), flush=True)
    else:
        print(f"[{regime}] reusing fine-tuned checkpoint {bc_ppo_ckpt}")


def bc_q_offset(regime: str, gamma: float = DQN_GAMMA) -> float:
    """Return-scale Q-value offset for the BC-initialised DQN of a cell.

    The BC policy's logits are O(1) while returns are O(-10^2..-10^3);
    a constant added to every Q(s, a) is argmax-invariant, so shifting
    the loaded logits by the BC policy's own measured value scale —
    mean step reward / (1 - gamma), from the committed Phase 8
    imitation_raw.csv eval rows — lets TD refine the value surface
    instead of rebuilding it through the wrong scale.

    Args:
        regime: Imitation cell name (must have bc rows in the CSV).
        gamma: Discount factor of the DQN.

    Returns:
        The offset (a large negative number for these cells).

    Raises:
        RuntimeError: If the committed CSV has no bc rows for the cell.
    """
    with open(IMITATION_RAW_CSV, newline="", encoding="utf-8") as fp:
        rows = [r for r in csv.DictReader(fp)
                if r["regime"] == regime and r["policy"] == "bc"]
    if not rows:
        raise RuntimeError(f"no committed bc rows for {regime} in "
                           f"{IMITATION_RAW_CSV}")
    steps = int(OFFPOLICY_CONFIGS[regime].episode_steps or "200")
    mean_reward = np.mean([float(r["reward"]) for r in rows])
    return float(mean_reward / steps / (1.0 - gamma))


def run_offpolicy_pipeline(regime: str, c2n: str, coherence: CoherenceConfig,
                           args: argparse.Namespace) -> None:
    """Train the dqn-scratch and dqn-bc arms of one cell (Phase 9).

    Each arm trains in its own subprocess (agent/dqn.py owns a gym
    env); existing checkpoints are reused unless --retrain. The dqn-bc
    arm requires the committed Phase 8 BC checkpoint of the cell. Each
    run's trajectory (switches, Q-gap, epsilon, TD loss per 500-step
    window) and episode rewards are written to results/ as CSVs.

    Args:
        regime: Cell name (an OFFPOLICY_CONFIGS key).
        c2n: Turbulence strength of the cell.
        coherence: Env sweep point of the cell.
        args: Orchestrator CLI namespace (quick/retrain/train knobs).
    """
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    dqn_py = str(_AGENT_DIR / "dqn.py")
    env_args = ["--c2n", c2n, *_coherence_cli_args(coherence)]
    total = 1000 if args.quick else (args.train_steps or OFFPOLICY_TRAIN_STEPS)
    scale = total / OFFPOLICY_TRAIN_STEPS

    for arm in OFFPOLICY_POLICIES:
        checkpoint = CHECKPOINTS_DIR / f"{arm.replace('-', '_')}_{regime}.pt"
        if not args.retrain and checkpoint.exists():
            print(f"[{regime}] reusing checkpoint {checkpoint}")
            continue
        eps_start, eps_end, eps_decay = (DQN_BC_EPS if arm == "dqn-bc"
                                         else DQN_SCRATCH_EPS)
        cmd = [sys.executable, dqn_py, "train",
               "--checkpoint", str(checkpoint),
               "--trajectory-csv",
               str(RESULTS_DIR / f"offpolicy_trajectory_{arm}_{regime}.csv"),
               "--rewards-csv",
               str(RESULTS_DIR / f"offpolicy_rewards_{arm}_{regime}.csv"),
               "--total-steps", str(total),
               "--eps-start", eps_start, "--eps-end", eps_end,
               "--eps-decay-steps", str(max(1, int(eps_decay * scale))),
               "--seed", str(args.train_seed), *env_args]
        if arm == "dqn-bc":
            bc_ckpt = CHECKPOINTS_DIR / f"bc_{regime}.pt"
            if not bc_ckpt.exists():
                raise RuntimeError(
                    f"missing committed Phase 8 BC checkpoint {bc_ckpt} "
                    f"(needed to initialise {arm})")
            cmd += ["--bc-checkpoint", str(bc_ckpt),
                    "--q-offset", f"{bc_q_offset(regime):.6g}"]
        print(f"[{regime}] training {arm} for {total} steps "
              f"(eps {eps_start} -> {eps_end}) ...", flush=True)
        out = _run_agent_stage(cmd, f"{regime} {arm}")
        print("\n".join(out.splitlines()[-2:]), flush=True)


def dataset_q_offset(dataset: Path, episode_steps: int,
                     gamma: float = DQN_GAMMA) -> float:
    """Return-scale Q-value offset from a collected teacher dataset.

    The route-aware analogue of :func:`bc_q_offset`: the fresh 32-dim
    BC clone has no committed eval rows before training, so the offset
    comes from the teacher episode rewards stored in the BC dataset —
    the reward scale the clone is expected to land on. Any constant is
    argmax-invariant; only the order of magnitude matters.

    Args:
        dataset: BC dataset (.npz with an ``episode_rewards`` array).
        episode_steps: Decision steps per episode of the cell.
        gamma: Discount factor of the DQN.

    Returns:
        Mean teacher step reward / (1 - gamma).
    """
    rewards = np.load(dataset)["episode_rewards"]
    return float(np.mean(rewards) / episode_steps / (1.0 - gamma))


def run_routeaware_pipeline(regime: str, c2n: str, coherence: CoherenceConfig,
                            args: argparse.Namespace) -> None:
    """Train the four route-aware arms of one cell (Phase 10).

    Repeats the Phase 8 imitation pipeline (collect / bc / value-warmup
    + PPO fine-tune) and the Phase 9 DQN arms with the env's routeInObs
    flag on, so every stage sees the 32-dim observation. The collect
    stage verifies on every step that the observation's route one-hot
    equals the teacher's held route. Each stage runs in its own
    subprocess; existing artifacts are reused unless --retrain.

    Args:
        regime: Cell name (a ROUTEAWARE_CONFIGS key).
        c2n: Turbulence strength of the cell.
        coherence: Env sweep point of the cell (route_in_obs="true").
        args: Orchestrator CLI namespace (quick/retrain/train knobs).
    """
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    imitation_py = str(_AGENT_DIR / "imitation.py")
    dqn_py = str(_AGENT_DIR / "dqn.py")
    dataset = CHECKPOINTS_DIR / f"bc_dataset_route_{regime}.npz"
    bc_ckpt = CHECKPOINTS_DIR / f"bc_route_{regime}.pt"
    bc_ppo_ckpt = CHECKPOINTS_DIR / f"bc_ppo_route_{regime}.pt"
    env_args = ["--c2n", c2n, *_coherence_cli_args(coherence)]

    if args.retrain or not dataset.exists():
        episodes = 2 if args.quick else BC_DATASET_EPISODES
        print(f"[{regime}] collecting {episodes} route-aware teacher episodes "
              f"...", flush=True)
        out = _run_agent_stage(
            [sys.executable, imitation_py, "collect", "--out", str(dataset),
             "--episodes", str(episodes), "--seed", str(args.train_seed),
             *env_args],
            f"{regime} collect")
        print("\n".join(line for line in out.splitlines()
                        if line.startswith(("[collect] wrote",
                                            "[collect] teacher",
                                            "[collect] route-onehot"))),
              flush=True)
    else:
        print(f"[{regime}] reusing dataset {dataset}")

    if args.retrain or not bc_ckpt.exists():
        epochs = 5 if args.quick else BC_EPOCHS
        print(f"[{regime}] behavior cloning ({epochs} epochs) ...", flush=True)
        out = _run_agent_stage(
            [sys.executable, imitation_py, "bc", "--dataset", str(dataset),
             "--checkpoint", str(bc_ckpt), "--epochs", str(epochs),
             "--metrics-csv", str(RESULTS_DIR / f"routeaware_bc_{regime}.csv")],
            f"{regime} bc")
        print("\n".join(line for line in out.splitlines()
                        if line.startswith("[bc]")), flush=True)
    else:
        print(f"[{regime}] reusing BC checkpoint {bc_ckpt}")

    total = 1000 if args.quick else (args.train_steps or ROUTEAWARE_TRAIN_STEPS)
    if args.retrain or not bc_ppo_ckpt.exists():
        warmup = 500 if args.quick else BC_WARMUP_STEPS
        print(f"[{regime}] value warmup ({warmup}) + PPO fine-tune ({total}) "
              f"...", flush=True)
        out = _run_agent_stage(
            [sys.executable, imitation_py, "finetune",
             "--bc-checkpoint", str(bc_ckpt),
             "--checkpoint", str(bc_ppo_ckpt),
             "--trajectory-csv",
             str(RESULTS_DIR / f"routeaware_trajectory_bc-ppo-route_{regime}.csv"),
             "--rewards-csv",
             str(RESULTS_DIR / f"routeaware_rewards_bc-ppo-route_{regime}.csv"),
             "--warmup-steps", str(warmup), "--total-steps", str(total),
             "--rollout-steps", "500", "--seed", str(args.train_seed),
             *env_args],
            f"{regime} finetune")
        print("\n".join(out.splitlines()[-2:]), flush=True)
    else:
        print(f"[{regime}] reusing fine-tuned checkpoint {bc_ppo_ckpt}")

    scale = total / ROUTEAWARE_TRAIN_STEPS
    for arm in ("dqn-scratch-route", "dqn-bc-route"):
        checkpoint = CHECKPOINTS_DIR / f"{arm.replace('-', '_')}_{regime}.pt"
        if not args.retrain and checkpoint.exists():
            print(f"[{regime}] reusing checkpoint {checkpoint}")
            continue
        eps_start, eps_end, eps_decay = (DQN_BC_EPS if arm == "dqn-bc-route"
                                         else DQN_SCRATCH_EPS)
        cmd = [sys.executable, dqn_py, "train",
               "--checkpoint", str(checkpoint),
               "--trajectory-csv",
               str(RESULTS_DIR / f"routeaware_trajectory_{arm}_{regime}.csv"),
               "--rewards-csv",
               str(RESULTS_DIR / f"routeaware_rewards_{arm}_{regime}.csv"),
               "--total-steps", str(total),
               "--eps-start", eps_start, "--eps-end", eps_end,
               "--eps-decay-steps", str(max(1, int(eps_decay * scale))),
               "--seed", str(args.train_seed), *env_args]
        if arm == "dqn-bc-route":
            steps = int(coherence.episode_steps or "200")
            cmd += ["--bc-checkpoint", str(bc_ckpt),
                    "--q-offset", f"{dataset_q_offset(dataset, steps):.6g}"]
        print(f"[{regime}] training {arm} for {total} steps "
              f"(eps {eps_start} -> {eps_end}) ...", flush=True)
        out = _run_agent_stage(cmd, f"{regime} {arm}")
        print("\n".join(out.splitlines()[-2:]), flush=True)


def eval_env_policies(regime: str, policies: list[str], episodes: int,
                      seed: int, sim_config: str, c2n: str | None = None,
                      coherence: CoherenceConfig | None = None) -> list[EpisodeRow]:
    """Evaluate env-driven policies for one sweep point in a worker subprocess.

    Args:
        regime: Sweep point to evaluate (regime or coherence config name).
        policies: Env-driven policy names (no aodv).
        episodes: Episodes per policy.
        seed: First episode's ns-3 run number.
        sim_config: Path to sim_config.yaml.
        c2n: C2n override; None derives it from the regime name.
        coherence: Optional coherence sweep point (correlated study).

    Returns:
        Parsed episode rows from the worker's FSO-ROW lines.

    Raises:
        RuntimeError: If the worker produced no rows.
        subprocess.CalledProcessError: If the worker failed.
    """
    cmd = [sys.executable, str(Path(__file__).resolve()), "--eval-worker",
           "--regime", regime, "--episodes", str(episodes),
           "--seed", str(seed), "--sim-config", sim_config,
           *_coherence_cli_args(coherence)]
    if c2n is not None:
        cmd += ["--c2n", c2n]
    for policy in policies:
        cmd += ["--policy", policy]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    rows = [EpisodeRow.from_csv_line(line[len(ROW_PREFIX):])
            for line in proc.stdout.splitlines() if line.startswith(ROW_PREFIX)]
    if proc.returncode != 0 or not rows:
        sys.stderr.write(proc.stderr[-4000:])
        raise RuntimeError(
            f"eval worker for {regime} {policies} failed (rc={proc.returncode}, "
            f"{len(rows)} rows)")
    return rows


def run_aodv_episode(settings: dict[str, str], seed: int, ns3_path: str) -> dict:
    """Run one fso-aodv-baseline episode and parse its FSO-BENCH lines.

    Args:
        settings: fso-rl-env style settings (flapPenalty is dropped;
            fso-aodv-baseline shares the remaining arguments).
        seed: ns-3 run number for this episode.
        ns3_path: ns-3 root directory containing the built program.

    Returns:
        Same metrics dict as :func:`run_env_episode`.

    Raises:
        RuntimeError: If the program produced no FSO-BENCH lines.
    """
    args = {k: v for k, v in settings.items() if k != "flapPenalty"}
    args["simSeed"] = str(seed)
    arg_str = " ".join(f"--{k}={v}" for k, v in args.items())
    proc = subprocess.run(
        [sys.executable, "./ns3", "run", "--no-build", f"fso-aodv-baseline {arg_str}"],
        cwd=ns3_path,
        capture_output=True,
        text=True,
        check=True,
    )
    steps = [dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
             for line in proc.stdout.splitlines()
             if line.startswith("FSO-BENCH step=")]
    if not steps:
        raise RuntimeError(f"no FSO-BENCH output from fso-aodv-baseline:\n{proc.stdout}")
    return _metrics_from_steps(steps, reward_key="reward")


def merge_rows(existing: list[dict], new_rows: list[EpisodeRow]) -> list[dict]:
    """Replace re-run (regime, policy) groups in the raw CSV rows.

    Args:
        existing: Rows already in the CSV, as string dicts.
        new_rows: Freshly measured episode rows.

    Returns:
        Combined row list in canonical (regime, policy, episode) order.
    """
    replaced = {(row.regime, row.policy) for row in new_rows}
    kept = [r for r in existing if (r["regime"], r["policy"]) not in replaced]
    combined = kept + [{k: str(v) for k, v in asdict(row).items()} for row in new_rows]
    regime_order = {name: i
                    for i, name in enumerate([*REGIMES, *COHERENCE_CONFIGS,
                                              *ADAPTATION_CONFIGS])}
    policy_order = {name: i for i, name in enumerate(ALL_POLICIES)}
    combined.sort(key=lambda r: (regime_order.get(r["regime"], 99),
                                 policy_order.get(r["policy"], 99),
                                 int(r["episode"])))
    return combined


def write_raw_csv(path: Path, rows: list[dict]) -> None:
    """Write the raw results CSV.

    Args:
        path: Output CSV path.
        rows: Row dicts with CSV_FIELDS keys.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse orchestrator/worker CLI arguments.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("turbulence", "correlated",
                                            "adaptation", "imitation",
                                            "offpolicy", "routeaware"),
                        default="turbulence",
                        help="turbulence: Phase 5 C2n sweep (default); "
                             "correlated: Phase 6 coherence-time sweep; "
                             "adaptation: Phase 7 disjoint-topology study; "
                             "imitation: Phase 8 imitation-then-RL study; "
                             "offpolicy: Phase 9 Double DQN study; "
                             "routeaware: Phase 10 route-in-observation study")
    parser.add_argument("--regime", action="append",
                        choices=(sorted(REGIMES) + sorted(COHERENCE_CONFIGS)
                                 + sorted(ADAPTATION_CONFIGS)),
                        help="restrict to this sweep point "
                             "(repeatable; default all)")
    parser.add_argument("--coherence", action="append",
                        choices=sorted(COHERENCE_CONFIGS),
                        help="restrict --study correlated to this coherence "
                             "config (repeatable; default all)")
    parser.add_argument("--policy", action="append", choices=ALL_POLICIES,
                        help="restrict to this policy (repeatable; default all)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="evaluation episodes per policy per regime")
    parser.add_argument("--seed", type=int, default=100,
                        help="ns-3 run number of the first evaluation episode")
    parser.add_argument("--train-seed", type=int, default=42,
                        help="training seed (and first training run number)")
    parser.add_argument("--train-steps", type=int, default=None,
                        help="override the per-cell PPO training budget "
                             "[env steps]")
    parser.add_argument("--retrain", action="store_true",
                        help="retrain PPO even if a regime checkpoint exists")
    parser.add_argument("--quick", action="store_true",
                        help="smoke mode: 2 episodes, 1000 training steps")
    parser.add_argument("--sim-config", type=str, default=str(DEFAULT_SIM_CONFIG),
                        help="path to sim_config.yaml")
    parser.add_argument("--eval-worker", action="store_true",
                        help=argparse.SUPPRESS)
    # Worker-only env overrides, set by the orchestrator
    parser.add_argument("--c2n", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--coherence-large", type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--coherence-small", type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--step-time", type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--episode-steps", type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--topology", type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--traffic-protocol", type=str, default=None,
                        dest="traffic_protocol", help=argparse.SUPPRESS)
    parser.add_argument("--route-in-obs", type=str, default=None,
                        dest="route_in_obs", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    """CLI entry point: run the selected slice of the benchmark study."""
    args = parse_args()
    if args.eval_worker:
        eval_worker(args)
        return

    correlated = args.study == "correlated"
    adaptation = args.study == "adaptation"
    imitation = args.study == "imitation"
    offpolicy = args.study == "offpolicy"
    routeaware = args.study == "routeaware"
    if routeaware:
        selected = args.regime or list(ROUTEAWARE_CONFIGS)
        cells = [(name, CORRELATED_C2N, ROUTEAWARE_CONFIGS[name])
                 for name in ROUTEAWARE_CONFIGS if name in selected]
        base_policies = ROUTEAWARE_POLICIES
        raw_csv = ROUTEAWARE_RAW_CSV
        train_steps = {}
    elif offpolicy:
        selected = args.regime or list(OFFPOLICY_CONFIGS)
        cells = [(name, CORRELATED_C2N, OFFPOLICY_CONFIGS[name])
                 for name in OFFPOLICY_CONFIGS if name in selected]
        base_policies = OFFPOLICY_POLICIES
        raw_csv = OFFPOLICY_RAW_CSV
        train_steps = {}
    elif imitation:
        selected = args.regime or list(IMITATION_CONFIGS)
        cells = [(name, CORRELATED_C2N, IMITATION_CONFIGS[name])
                 for name in IMITATION_CONFIGS if name in selected]
        base_policies = IMITATION_POLICIES
        raw_csv = IMITATION_RAW_CSV
        train_steps = {}
    elif correlated:
        selected = args.coherence or list(COHERENCE_CONFIGS)
        cells = [(name, CORRELATED_C2N, COHERENCE_CONFIGS[name])
                 for name in COHERENCE_CONFIGS if name in selected]
        base_policies = CORRELATED_POLICIES
        raw_csv = CORRELATED_RAW_CSV
        train_steps = {name: 1000 if args.quick else CORRELATED_TRAIN_STEPS
                       for name, _, _ in cells}
    elif adaptation:
        selected = args.regime or list(ADAPTATION_CONFIGS)
        cells = [(name, CORRELATED_C2N, ADAPTATION_CONFIGS[name])
                 for name in ADAPTATION_CONFIGS if name in selected]
        base_policies = ADAPTATION_POLICIES
        raw_csv = ADAPTATION_RAW_CSV
        budget = args.train_steps or ADAPTATION_TRAIN_STEPS
        train_steps = {name: 1000 if args.quick else budget
                       for name, _, _ in cells}
    else:
        regimes = sorted(args.regime or list(REGIMES), key=list(REGIMES).index)
        cells = [(regime, REGIMES[regime], None) for regime in regimes]
        base_policies = POLICIES
        raw_csv = RAW_CSV
        train_steps = dict.fromkeys(TRAIN_STEPS, 1000) if args.quick else TRAIN_STEPS

    policies = [p for p in base_policies if p in (args.policy or base_policies)]
    episodes = 2 if args.quick else args.episodes

    from ns3ai_shim import DEFAULT_NS3_PATH, load_flat_yaml, ns3_settings

    sim_config = str(Path(args.sim_config).resolve())
    config = load_flat_yaml(sim_config)
    existing: list[dict] = []
    if raw_csv.exists():
        with open(raw_csv, newline="", encoding="utf-8") as fp:
            existing = list(csv.DictReader(fp))

    new_rows: list[EpisodeRow] = []
    timings: list[tuple[str, str, float]] = []
    study_start = time.monotonic()

    for regime, c2n, coherence in cells:
        detail = "" if coherence is None else (
            f", tau {coherence.coherence_large}/{coherence.coherence_small}")
        print(f"\n=== {regime} (C2n={c2n}{detail}) ===", flush=True)

        cell_policies = [p for p in policies
                         if not (p == "ppo-stack" and regime in STACK_SKIP_REGIMES)]
        if imitation:
            start = time.monotonic()
            run_imitation_pipeline(regime, c2n, coherence, args)
            timings.append((regime, "imitation-pipeline",
                            time.monotonic() - start))
        if offpolicy:
            start = time.monotonic()
            run_offpolicy_pipeline(regime, c2n, coherence, args)
            timings.append((regime, "offpolicy-pipeline",
                            time.monotonic() - start))
        if routeaware:
            start = time.monotonic()
            run_routeaware_pipeline(regime, c2n, coherence, args)
            timings.append((regime, "routeaware-pipeline",
                            time.monotonic() - start))
        train_variants = [] if (imitation or offpolicy or routeaware) else [
            p for p in cell_policies
            if p.startswith("ppo") and p != "ppo-transfer"]
        for variant in train_variants:
            checkpoint = (CHECKPOINTS_DIR /
                          f"{variant.replace('-', '_')}_{regime}.pt")
            if args.retrain or not checkpoint.exists():
                steps = train_steps[regime]
                if not args.quick:
                    steps = args.train_steps or VARIANT_TRAIN_STEPS.get(
                        variant, steps)
                start = time.monotonic()
                train_regime_policy(regime, c2n, steps,
                                    checkpoint, args.train_seed, coherence,
                                    frame_stack=(FRAME_STACK_K
                                                 if variant == "ppo-stack" else 1))
                elapsed = time.monotonic() - start
                timings.append((regime, f"{variant}-train", elapsed))
                print(f"[{regime}] {variant} training done in {elapsed:.0f}s",
                      flush=True)
            else:
                print(f"[{regime}] reusing checkpoint {checkpoint}")

        env_policies = [p for p in cell_policies if p != "aodv"]
        if env_policies:
            start = time.monotonic()
            rows = eval_env_policies(regime, env_policies, episodes,
                                     args.seed, sim_config,
                                     c2n=None if coherence is None else c2n,
                                     coherence=coherence)
            new_rows.extend(rows)
            timings.append((regime, "env-eval", time.monotonic() - start))
            for row in rows:
                print(f"[{regime}] {row.policy} ep{row.episode}: "
                      f"reward={row.reward:.1f} pdr={row.pdr:.3f} "
                      f"delay={row.mean_delay_ms:.3f}ms", flush=True)

        if "aodv" in policies:
            settings = ns3_settings(config, args.seed)
            settings["c2n"] = c2n
            if coherence is not None:
                settings["coherenceLarge"] = coherence.coherence_large
                settings["coherenceSmall"] = coherence.coherence_small
                if coherence.step_time_s is not None:
                    settings["stepTime"] = coherence.step_time_s
                if coherence.episode_steps is not None:
                    settings["episodeSteps"] = coherence.episode_steps
                if coherence.topology is not None:
                    settings["topology"] = coherence.topology
                if coherence.traffic_protocol is not None:
                    settings["trafficProtocol"] = coherence.traffic_protocol
            start = time.monotonic()
            for ep in range(episodes):
                metrics = run_aodv_episode(settings, args.seed + ep, DEFAULT_NS3_PATH)
                new_rows.append(EpisodeRow(regime=regime, c2n=c2n, policy="aodv",
                                           episode=ep, sim_seed=args.seed + ep,
                                           **metrics))
                print(f"[{regime}] aodv ep{ep}: reward={metrics['reward']:.1f} "
                      f"pdr={metrics['pdr']:.3f} "
                      f"delay={metrics['mean_delay_ms']:.3f}ms", flush=True)
            timings.append((regime, "aodv", time.monotonic() - start))

        # Persist after every sweep point so a crash never loses finished work
        write_raw_csv(raw_csv, merge_rows(existing, new_rows))

    rows = merge_rows(existing, new_rows)
    write_raw_csv(raw_csv, rows)
    print(f"\nwrote {len(rows)} rows ({len(new_rows)} new) to {raw_csv}")

    print("\nwall time breakdown:")
    for regime, phase, seconds in timings:
        print(f"  {regime:9s} {phase:12s} {seconds:7.1f}s")
    print(f"  total: {time.monotonic() - study_start:.1f}s")


if __name__ == "__main__":
    main()
