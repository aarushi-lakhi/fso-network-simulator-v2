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
for _dir in (str(_AGENT_DIR),):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

from eval_policy import parse_info  # noqa: E402
from ns3_env import DEFAULT_CONFIG_PATH, make_ns3_env  # noqa: E402
from ns3ai_shim import DEFAULT_NS3_PATH, load_flat_yaml, ns3_settings  # noqa: E402
from ppo_agent import PPOAgent  # noqa: E402
from train import TrainConfig, train  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RAW_CSV = RESULTS_DIR / "raw_results.csv"
PHASE4_CHECKPOINT = _AGENT_DIR / "checkpoints" / "ns3_ppo.pt"

REGIMES: dict[str, str] = {"weak": "1e-17", "moderate": "1e-15", "strong": "1e-13"}

# Per-regime training budgets [env steps]. Phase 4 showed the strong-regime
# reward plateaus by ~20k steps; weaker regimes converge faster still
# (fewer drops -> less reward variance), so they get smaller budgets.
TRAIN_STEPS: dict[str, int] = {"weak": 20_000, "moderate": 40_000, "strong": 80_000}

POLICIES = ("ppo", "ppo-transfer", "static-0", "static-1", "static-2", "static-3",
            "random", "aodv")

CSV_FIELDS = ("regime", "c2n", "policy", "episode", "sim_seed", "reward",
              "drops", "tx_pkts", "rx_pkts", "pdr", "mean_delay_ms")


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


def run_env_episode(env, seed: int, action_fn: Callable[[np.ndarray], int]) -> dict:
    """Roll one episode of the gym env under the given action function.

    Args:
        env: The FSO ns3 environment.
        seed: ns-3 run number for this episode.
        action_fn: Maps the current observation to a route action.

    Returns:
        Dict with reward, drops, tx_pkts, rx_pkts, pdr, mean_delay_ms.
    """
    obs, _ = env.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float32)
    reward_total = 0.0
    drops = tx = rx = 0
    delay_weighted_sum = 0.0
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(action_fn(obs))
        obs = np.asarray(obs, dtype=np.float32)
        reward_total += float(reward)
        fields = parse_info(info)
        step_rx = int(fields.get("rxPkts", 0))
        drops += int(fields.get("drops", 0))
        tx += int(fields.get("txPkts", 0))
        rx += step_rx
        delay_weighted_sum += float(fields.get("meanDelayMs", 0.0)) * step_rx
        done = terminated or truncated
    return {
        "reward": reward_total,
        "drops": drops,
        "tx_pkts": tx,
        "rx_pkts": rx,
        "pdr": rx / tx if tx else 0.0,
        "mean_delay_ms": delay_weighted_sum / rx if rx else 0.0,
    }


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
    reward_total = 0.0
    drops = tx = rx = 0
    delay_weighted_sum = 0.0
    steps = 0
    for line in proc.stdout.splitlines():
        if not line.startswith("FSO-BENCH step="):
            continue
        fields = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
        step_rx = int(fields["rxPkts"])
        reward_total += float(fields["reward"])
        drops += int(fields["drops"])
        tx += int(fields["txPkts"])
        rx += step_rx
        delay_weighted_sum += float(fields["meanDelayMs"]) * step_rx
        steps += 1
    if steps == 0:
        raise RuntimeError(f"no FSO-BENCH output from fso-aodv-baseline:\n{proc.stdout}")
    return {
        "reward": reward_total,
        "drops": drops,
        "tx_pkts": tx,
        "rx_pkts": rx,
        "pdr": rx / tx if tx else 0.0,
        "mean_delay_ms": delay_weighted_sum / rx if rx else 0.0,
    }


def load_agent(checkpoint: Path, obs_dim: int = 28, n_actions: int = 4) -> PPOAgent:
    """Load a greedy-evaluation PPO agent from a checkpoint.

    Args:
        checkpoint: Path to the .pt checkpoint file.
        obs_dim: Flattened observation dimension.
        n_actions: Number of discrete actions.

    Returns:
        The loaded agent.
    """
    agent = PPOAgent(obs_dim, n_actions)
    agent.load(checkpoint)
    return agent


def train_regime_policy(
    regime: str,
    c2n: str,
    total_steps: int,
    checkpoint: Path,
    train_seed: int,
) -> None:
    """Train a fresh PPO policy for one turbulence regime.

    Args:
        regime: Regime name (for logging only).
        c2n: C2n value the env is trained at.
        total_steps: Environment steps of training.
        checkpoint: Where the trained weights are saved.
        train_seed: Global training seed (also the first episode's run
            number; disjoint from the evaluation seed range).
    """
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(
        total_steps=total_steps,
        rollout_steps=500,
        seed=train_seed,
        checkpoint_path=str(checkpoint),
    )
    print(f"[{regime}] training PPO for {total_steps} steps at C2n={c2n} ...")
    start = time.monotonic()
    result = train(config, env_factory=lambda: make_ns3_env(c2n=c2n, seed=train_seed))
    elapsed = time.monotonic() - start
    rewards = result.episode_rewards
    head = float(np.mean(rewards[: max(1, len(rewards) // 10)]))
    tail = float(np.mean(rewards[-max(1, len(rewards) // 10):]))
    print(f"[{regime}] trained in {elapsed:.0f}s: {len(rewards)} episodes, "
          f"mean reward first 10% {head:.1f} -> last 10% {tail:.1f}")


def action_fn_for(policy: str, agent: PPOAgent | None,
                  rng: np.random.Generator, n_actions: int) -> Callable[[np.ndarray], int]:
    """Build the observation->action function for an env-driven policy.

    Args:
        policy: One of ppo, ppo-transfer, static-K, random.
        agent: Loaded PPO agent for the ppo policies.
        rng: RNG for the random policy.
        n_actions: Size of the action space.

    Returns:
        Callable mapping an observation to a discrete action.
    """
    if policy in ("ppo", "ppo-transfer"):
        assert agent is not None
        return agent.act_greedy
    if policy.startswith("static-"):
        route = int(policy.split("-", 1)[1])
        return lambda _obs: route
    if policy == "random":
        return lambda _obs: int(rng.integers(n_actions))
    raise ValueError(f"unknown env policy: {policy}")


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


def main() -> None:
    """CLI entry point: run the selected slice of the benchmark study."""
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
    parser.add_argument("--sim-config", type=str, default=DEFAULT_CONFIG_PATH,
                        help="path to sim_config.yaml")
    args = parser.parse_args()

    regimes = args.regime or list(REGIMES)
    policies = args.policy or list(POLICIES)
    episodes = 2 if args.quick else args.episodes
    train_steps = dict.fromkeys(TRAIN_STEPS, 1000) if args.quick else TRAIN_STEPS

    sim_config = str(Path(args.sim_config).resolve())
    config = load_flat_yaml(sim_config)
    checkpoints_dir = RESULTS_DIR / "checkpoints"
    raw_csv = RAW_CSV
    existing: list[dict] = []
    if raw_csv.exists():
        with open(raw_csv, newline="", encoding="utf-8") as fp:
            existing = list(csv.DictReader(fp))

    new_rows: list[EpisodeRow] = []
    timings: list[tuple[str, str, float]] = []
    study_start = time.monotonic()

    for regime in sorted(regimes, key=list(REGIMES).index):
        c2n = REGIMES[regime]
        print(f"\n=== regime {regime} (C2n={c2n}) ===")

        if "ppo" in policies:
            checkpoint = checkpoints_dir / f"ppo_{regime}.pt"
            if args.retrain or not checkpoint.exists():
                start = time.monotonic()
                train_regime_policy(regime, c2n, train_steps[regime],
                                    checkpoint, args.train_seed)
                timings.append((regime, "ppo-train", time.monotonic() - start))
            else:
                print(f"[{regime}] reusing checkpoint {checkpoint}")

        env_policies = [p for p in policies if p != "aodv"]
        if env_policies:
            env = make_ns3_env(sim_config, c2n=c2n, seed=args.seed)
            try:
                n_actions = int(env.action_space.n)
                for policy in env_policies:
                    agent = None
                    if policy == "ppo":
                        agent = load_agent(checkpoints_dir / f"ppo_{regime}.pt")
                    elif policy == "ppo-transfer":
                        agent = load_agent(PHASE4_CHECKPOINT)
                    rng = np.random.default_rng(args.seed)
                    action_fn = action_fn_for(policy, agent, rng, n_actions)
                    start = time.monotonic()
                    for ep in range(episodes):
                        metrics = run_env_episode(env, args.seed + ep, action_fn)
                        new_rows.append(EpisodeRow(regime=regime, c2n=c2n,
                                                   policy=policy, episode=ep,
                                                   sim_seed=args.seed + ep,
                                                   **metrics))
                        print(f"[{regime}] {policy} ep{ep}: "
                              f"reward={metrics['reward']:.1f} "
                              f"pdr={metrics['pdr']:.3f} "
                              f"delay={metrics['mean_delay_ms']:.3f}ms")
                    timings.append((regime, policy, time.monotonic() - start))
            finally:
                env.close()

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
                      f"delay={metrics['mean_delay_ms']:.3f}ms")
            timings.append((regime, "aodv", time.monotonic() - start))

    rows = merge_rows(existing, new_rows)
    write_raw_csv(raw_csv, rows)
    print(f"\nwrote {len(rows)} rows ({len(new_rows)} new) to {raw_csv}")

    print("\nwall time breakdown:")
    for regime, phase, seconds in timings:
        print(f"  {regime:9s} {phase:12s} {seconds:7.1f}s")
    print(f"  total: {time.monotonic() - study_start:.1f}s")


if __name__ == "__main__":
    main()
