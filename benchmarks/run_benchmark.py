"""Phase 5 benchmark orchestrator: PPO vs classical routing baselines.

Runs the full comparison study over the three turbulence regimes
(C2n weak 1e-17, moderate 1e-15, strong 1e-13) on the 5-node FSO mesh:

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
per-episode rows are written to results/raw_results.csv; rows for
(regime, policy) pairs re-run later replace the old ones in place.

Process model: ns3-ai's shared-memory interface is once-per-process (the
segment lives in a C++ function-local static), so every phase that owns
a gym env runs in its own subprocess — training shells out to the
Phase 4 train.py CLI, and env evaluation re-invokes this script with
--eval-worker, which prints one ``FSO-ROW <csv>`` line per episode.

Prerequisites: modules linked and built (setup/link_fso_modules.sh) and
the agent venv active (`env python3` must resolve to Python 3.11).

Typical usage:
    $ python run_benchmark.py                     # full study
    $ python run_benchmark.py --quick             # 2-episode smoke run
    $ python run_benchmark.py --regime strong --policy aodv
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

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RAW_CSV = RESULTS_DIR / "raw_results.csv"
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

CSV_FIELDS = ("regime", "c2n", "policy", "episode", "sim_seed", "reward",
              "drops", "tx_pkts", "rx_pkts", "pdr", "mean_delay_ms")

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
                   pdr=float(parts["pdr"]), mean_delay_ms=float(parts["mean_delay_ms"]))


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
        pdr, and packet-weighted mean_delay_ms.
    """
    reward_total = 0.0
    drops = tx = rx = 0
    delay_weighted_sum = 0.0
    for fields in step_fields:
        step_rx = int(fields.get("rxPkts", 0))
        drops += int(fields.get("drops", 0))
        tx += int(fields.get("txPkts", 0))
        rx += step_rx
        delay_weighted_sum += float(fields.get("meanDelayMs", 0.0)) * step_rx
        if reward_key is not None:
            reward_total += float(fields.get(reward_key, 0.0))
    return {
        "reward": reward_total,
        "drops": drops,
        "tx_pkts": tx,
        "rx_pkts": rx,
        "pdr": rx / tx if tx else 0.0,
        "mean_delay_ms": delay_weighted_sum / rx if rx else 0.0,
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


def _action_fn_for(policy: str, env, seed: int) -> Callable[[np.ndarray], int]:
    """Build the observation->action function for an env-driven policy.

    Args:
        policy: One of ppo, ppo-transfer, static-K, random.
        env: The gym env (for space sizes).
        seed: RNG seed for the random policy.

    Returns:
        Callable mapping an observation to a discrete action.
    """
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)
    if policy == "ppo-transfer":
        return _load_agent(PHASE4_CHECKPOINT, obs_dim, n_actions).act_greedy
    if policy.startswith("ppo"):
        regime = policy.split(":", 1)[1]
        checkpoint = CHECKPOINTS_DIR / f"ppo_{regime}.pt"
        return _load_agent(checkpoint, obs_dim, n_actions).act_greedy
    if policy.startswith("static-"):
        route = int(policy.split("-", 1)[1])
        return lambda _obs: route
    if policy == "random":
        rng = np.random.default_rng(seed)
        return lambda _obs: int(rng.integers(n_actions))
    raise ValueError(f"unknown env policy: {policy}")


def eval_worker(args: argparse.Namespace) -> None:
    """Evaluate all env-driven policies of one regime in this process.

    Creates the single allowed gym env, rolls the shared seed set for
    each policy, and prints one ``FSO-ROW <csv>`` line per episode on
    stdout (progress goes to stderr).

    Args:
        args: Parsed CLI namespace (regime, policies, episodes, seed, ...).
    """
    from ns3_env import make_ns3_env

    regime = args.regime[0]
    c2n = REGIMES[regime]
    policies = args.policy
    env = make_ns3_env(str(Path(args.sim_config).resolve()), c2n=c2n, seed=args.seed)
    try:
        for policy in policies:
            worker_policy = f"ppo:{regime}" if policy == "ppo" else policy
            action_fn = _action_fn_for(worker_policy, env, args.seed)
            for ep in range(args.episodes):
                metrics = run_env_episode(env, args.seed + ep, action_fn)
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


def train_regime_policy(regime: str, c2n: str, total_steps: int,
                        checkpoint: Path, train_seed: int) -> None:
    """Train a fresh PPO policy for one regime via the train.py CLI.

    Runs in a subprocess because the ns3-ai shared-memory interface can
    only be created once per process.

    Args:
        regime: Regime name (for logging only).
        c2n: C2n value the env is trained at.
        total_steps: Environment steps of training.
        checkpoint: Where the trained weights are saved.
        train_seed: Global training seed (also the first episode's run
            number; disjoint from the evaluation seed range).

    Raises:
        subprocess.CalledProcessError: If training fails.
    """
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{regime}] training PPO for {total_steps} steps at C2n={c2n} ...",
          flush=True)
    subprocess.run(
        [sys.executable, str(_AGENT_DIR / "train.py"), "--env", "ns3",
         "--c2n", c2n, "--total-steps", str(total_steps),
         "--rollout-steps", "500", "--seed", str(train_seed),
         "--checkpoint-path", str(checkpoint)],
        cwd=_AGENT_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def eval_env_policies(regime: str, policies: list[str], episodes: int,
                      seed: int, sim_config: str) -> list[EpisodeRow]:
    """Evaluate env-driven policies for one regime in a worker subprocess.

    Args:
        regime: Regime to evaluate.
        policies: Env-driven policy names (no aodv).
        episodes: Episodes per policy.
        seed: First episode's ns-3 run number.
        sim_config: Path to sim_config.yaml.

    Returns:
        Parsed episode rows from the worker's FSO-ROW lines.

    Raises:
        RuntimeError: If the worker produced no rows.
        subprocess.CalledProcessError: If the worker failed.
    """
    cmd = [sys.executable, str(Path(__file__).resolve()), "--eval-worker",
           "--regime", regime, "--episodes", str(episodes),
           "--seed", str(seed), "--sim-config", sim_config]
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
    regime_order = {name: i for i, name in enumerate(REGIMES)}
    policy_order = {name: i for i, name in enumerate(POLICIES)}
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
    parser.add_argument("--regime", action="append", choices=sorted(REGIMES),
                        help="restrict to this regime (repeatable; default all)")
    parser.add_argument("--policy", action="append", choices=POLICIES,
                        help="restrict to this policy (repeatable; default all)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="evaluation episodes per policy per regime")
    parser.add_argument("--seed", type=int, default=100,
                        help="ns-3 run number of the first evaluation episode")
    parser.add_argument("--train-seed", type=int, default=42,
                        help="training seed (and first training run number)")
    parser.add_argument("--retrain", action="store_true",
                        help="retrain PPO even if a regime checkpoint exists")
    parser.add_argument("--quick", action="store_true",
                        help="smoke mode: 2 episodes, 1000 training steps")
    parser.add_argument("--sim-config", type=str, default=str(DEFAULT_SIM_CONFIG),
                        help="path to sim_config.yaml")
    parser.add_argument("--eval-worker", action="store_true",
                        help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    """CLI entry point: run the selected slice of the benchmark study."""
    args = parse_args()
    if args.eval_worker:
        eval_worker(args)
        return

    regimes = sorted(args.regime or list(REGIMES), key=list(REGIMES).index)
    policies = [p for p in POLICIES if p in (args.policy or POLICIES)]
    episodes = 2 if args.quick else args.episodes
    train_steps = dict.fromkeys(TRAIN_STEPS, 1000) if args.quick else TRAIN_STEPS

    from ns3ai_shim import DEFAULT_NS3_PATH, load_flat_yaml, ns3_settings

    sim_config = str(Path(args.sim_config).resolve())
    config = load_flat_yaml(sim_config)
    existing: list[dict] = []
    if RAW_CSV.exists():
        with open(RAW_CSV, newline="", encoding="utf-8") as fp:
            existing = list(csv.DictReader(fp))

    new_rows: list[EpisodeRow] = []
    timings: list[tuple[str, str, float]] = []
    study_start = time.monotonic()

    for regime in regimes:
        c2n = REGIMES[regime]
        print(f"\n=== regime {regime} (C2n={c2n}) ===", flush=True)

        if "ppo" in policies:
            checkpoint = CHECKPOINTS_DIR / f"ppo_{regime}.pt"
            if args.retrain or not checkpoint.exists():
                start = time.monotonic()
                train_regime_policy(regime, c2n, train_steps[regime],
                                    checkpoint, args.train_seed)
                elapsed = time.monotonic() - start
                timings.append((regime, "ppo-train", elapsed))
                print(f"[{regime}] training done in {elapsed:.0f}s", flush=True)
            else:
                print(f"[{regime}] reusing checkpoint {checkpoint}")

        env_policies = [p for p in policies if p != "aodv"]
        if env_policies:
            start = time.monotonic()
            rows = eval_env_policies(regime, env_policies, episodes,
                                     args.seed, sim_config)
            new_rows.extend(rows)
            timings.append((regime, "env-eval", time.monotonic() - start))
            for row in rows:
                print(f"[{regime}] {row.policy} ep{row.episode}: "
                      f"reward={row.reward:.1f} pdr={row.pdr:.3f} "
                      f"delay={row.mean_delay_ms:.3f}ms", flush=True)

        if "aodv" in policies:
            settings = ns3_settings(config, args.seed)
            settings["c2n"] = c2n
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

    rows = merge_rows(existing, new_rows)
    write_raw_csv(RAW_CSV, rows)
    print(f"\nwrote {len(rows)} rows ({len(new_rows)} new) to {RAW_CSV}")

    print("\nwall time breakdown:")
    for regime, phase, seconds in timings:
        print(f"  {regime:9s} {phase:12s} {seconds:7.1f}s")
    print(f"  total: {time.monotonic() - study_start:.1f}s")


if __name__ == "__main__":
    main()
