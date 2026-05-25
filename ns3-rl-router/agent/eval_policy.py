"""Evaluate a routing policy on the real ns3-ai FSO environment.

Runs N episodes with either a random policy or a greedy PPO checkpoint
policy, using deterministic per-episode ns-3 run numbers (seed, seed+1,
...) so different policies can be compared on identical fading
realisations. Prints per-episode reward and delivery stats parsed from
the environment's info string, then a mean +/- std summary.

Typical usage (agent venv activated, modules linked and built):
    $ python eval_policy.py --episodes 10 --seed 100 --c2n 1e-13
    $ python eval_policy.py --episodes 10 --seed 100 --c2n 1e-13 \\
          --checkpoint checkpoints/ns3_ppo.pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np

from ns3_env import DEFAULT_CONFIG_PATH, make_ns3_env
from ppo_agent import PPOAgent


@dataclass
class EpisodeStats:
    """Aggregated outcome of one evaluation episode.

    Attributes:
        reward: Total (undiscounted) episode reward.
        drops: PHY packets lost to fading, summed over all links/steps.
        tx_packets: Flow packets sent by the source.
        rx_packets: Flow packets delivered to the sink.
    """

    reward: float
    drops: int
    tx_packets: int
    rx_packets: int

    @property
    def pdr(self) -> float:
        """Packet delivery ratio of the 0->3 flow (0 when nothing sent)."""
        return self.rx_packets / self.tx_packets if self.tx_packets else 0.0


def parse_info(info: dict) -> dict[str, str]:
    """Parse the env's ``key=value`` info string into a dict.

    Args:
        info: Gymnasium info dict; the ns3 env stores its extra info
            under the "info" key as e.g. "step=3 route=1 drops=2 ...".

    Returns:
        Mapping of key to raw string value (empty if absent).
    """
    text = info.get("info", "")
    if not isinstance(text, str):
        return {}
    return dict(kv.split("=", 1) for kv in text.split() if "=" in kv)


def run_episode(
    env: gym.Env,
    seed: int,
    agent: PPOAgent | None,
    rng: np.random.Generator,
) -> EpisodeStats:
    """Roll one episode to completion under the given policy.

    Args:
        env: The FSO ns3 environment.
        seed: ns-3 run number for this episode.
        agent: Greedy policy source; None selects uniform random actions.
        rng: RNG for the random policy.

    Returns:
        EpisodeStats for the episode.
    """
    obs, _ = env.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float32)
    total_reward = 0.0
    drops = tx = rx = 0
    done = False
    while not done:
        if agent is None:
            action = int(rng.integers(env.action_space.n))
        else:
            action = agent.act_greedy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        obs = np.asarray(obs, dtype=np.float32)
        total_reward += float(reward)
        fields = parse_info(info)
        drops += int(fields.get("drops", 0))
        tx += int(fields.get("txPkts", 0))
        rx += int(fields.get("rxPkts", 0))
        done = terminated or truncated
    return EpisodeStats(total_reward, drops, tx, rx)


def main() -> None:
    """CLI entry point: evaluate a policy and print per-episode stats."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10,
                        help="number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=100,
                        help="ns-3 run number of the first episode")
    parser.add_argument("--c2n", type=str, default=None,
                        help="C2n override [m^-2/3], e.g. 1e-13")
    parser.add_argument("--coherence-large", type=str, default=None,
                        help="large-scale fading coherence time, e.g. 500ms "
                             "(0ms = i.i.d.)")
    parser.add_argument("--coherence-small", type=str, default=None,
                        help="small-scale fading coherence time, e.g. 100ms "
                             "(0ms = i.i.d.)")
    parser.add_argument("--step-time", type=str, default=None,
                        help="decision interval [s], e.g. 0.05")
    parser.add_argument("--episode-steps", type=int, default=None,
                        help="decision steps per episode")
    parser.add_argument("--topology", type=str, default=None,
                        choices=("pentagon", "disjoint"),
                        help="mesh layout (see sim/README.md)")
    parser.add_argument("--traffic-protocol", type=str, default=None,
                        choices=("udp", "tcp"), dest="traffic_protocol",
                        help="transport of the 0->3 flow")
    parser.add_argument("--sim-config", type=str, default=DEFAULT_CONFIG_PATH,
                        help="path to sim_config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="PPO checkpoint (.pt); omit for a random policy")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
    sim_config = str(Path(args.sim_config).resolve())

    env = make_ns3_env(sim_config, c2n=args.c2n, seed=args.seed,
                       coherence_large=args.coherence_large,
                       coherence_small=args.coherence_small,
                       step_time_s=args.step_time,
                       episode_steps=args.episode_steps,
                       topology=args.topology,
                       traffic_protocol=args.traffic_protocol)
    agent = None
    if checkpoint is not None:
        obs_dim = int(np.prod(env.observation_space.shape))
        n_actions = int(env.action_space.n)
        agent = PPOAgent(obs_dim, n_actions)
        agent.load(checkpoint)

    policy = "random" if agent is None else f"checkpoint {checkpoint}"
    print(f"evaluating {policy} for {args.episodes} episodes "
          f"(seeds {args.seed}..{args.seed + args.episodes - 1})")

    rng = np.random.default_rng(args.seed)
    episodes: list[EpisodeStats] = []
    try:
        for ep in range(args.episodes):
            stats = run_episode(env, args.seed + ep, agent, rng)
            episodes.append(stats)
            print(f"episode {ep:2d} (simSeed={args.seed + ep}): "
                  f"reward={stats.reward:9.3f} drops={stats.drops:4d} "
                  f"tx={stats.tx_packets:4d} rx={stats.rx_packets:4d} "
                  f"pdr={stats.pdr:.3f}")
    finally:
        env.close()

    rewards = np.array([e.reward for e in episodes])
    drops = np.array([e.drops for e in episodes])
    pdrs = np.array([e.pdr for e in episodes])
    print(f"\npolicy: {policy}")
    print(f"episodes:       {len(episodes)}")
    print(f"episode reward: {rewards.mean():.3f} +/- {rewards.std():.3f}")
    print(f"drops/episode:  {drops.mean():.1f} +/- {drops.std():.1f}")
    print(f"PDR:            {pdrs.mean():.3f} +/- {pdrs.std():.3f}")


if __name__ == "__main__":
    main()
