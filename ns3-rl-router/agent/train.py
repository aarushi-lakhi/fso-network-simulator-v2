"""
Training loop for the PPO routing agent.

Drives any Gymnasium-style environment through on-policy rollouts and
PPO updates, with TensorBoard logging and checkpointing. The env is
supplied as a factory so the ns3-ai Gym environment can be plugged in
at integration time without touching this file; by default the
self-contained toy routing env is used.

Typical usage:
    $ python train.py --total-steps 50000 --seed 42
    $ python train.py --config ../config/train_config.yaml
    $ tensorboard --logdir runs/

Programmatic:
    >>> result = train(TrainConfig(total_steps=10_000))
    >>> result.episode_rewards[-5:]
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import yaml

from ppo_agent import PPOAgent, PPOConfig, RolloutBuffer
from toy_env import ToyFsoRoutingEnv

EnvFactory = Callable[[], gym.Env]


@dataclass
class TrainConfig:
    """Configuration for a training run.

    Attributes:
        total_steps: Total environment steps to train for.
        rollout_steps: Steps collected per PPO update.
        learning_rate: Adam step size.
        gamma: Discount factor.
        gae_lambda: GAE λ.
        clip_epsilon: PPO clipping range.
        value_coef: Value loss weight.
        entropy_coef: Entropy bonus weight.
        max_grad_norm: Gradient clipping threshold.
        n_epochs: Optimisation epochs per rollout.
        minibatch_size: SGD minibatch size.
        hidden_sizes: Actor/critic hidden-layer widths.
        seed: Global RNG seed (torch, numpy, env). None disables seeding.
        log_dir: TensorBoard run directory. None disables logging.
        checkpoint_path: Where to save the final checkpoint. None disables.
        checkpoint_every: Save a checkpoint every N updates (0 = final only).
        device: Torch device string.
    """

    total_steps: int = 50_000
    rollout_steps: int = 512
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    minibatch_size: int = 64
    hidden_sizes: tuple[int, ...] = (64, 64)
    seed: int | None = None
    log_dir: str | None = None
    checkpoint_path: str | None = None
    checkpoint_every: int = 0
    device: str = "cpu"

    def ppo_config(self) -> PPOConfig:
        """Build the PPOConfig subset of this training config."""
        return PPOConfig(
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_epsilon=self.clip_epsilon,
            value_coef=self.value_coef,
            entropy_coef=self.entropy_coef,
            max_grad_norm=self.max_grad_norm,
            n_epochs=self.n_epochs,
            minibatch_size=self.minibatch_size,
            hidden_sizes=tuple(self.hidden_sizes),
        )


@dataclass
class TrainResult:
    """Artifacts of a completed training run.

    Attributes:
        agent: The trained PPO agent.
        episode_rewards: Total reward of each completed episode, in order.
        update_metrics: Per-update PPO metrics (losses, entropy, KL, ...).
    """

    agent: PPOAgent
    episode_rewards: list[float] = field(default_factory=list)
    update_metrics: list[dict[str, float]] = field(default_factory=list)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Args:
        seed: Seed value applied to all three generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str | Path) -> TrainConfig:
    """Load a TrainConfig from a YAML file.

    Unknown keys are rejected so typos in config files fail loudly.

    Args:
        path: Path to a YAML mapping of TrainConfig field names to values.

    Returns:
        Populated TrainConfig.

    Raises:
        ValueError: If the file contains keys not present on TrainConfig.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    valid = {f.name for f in fields(TrainConfig)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
    if "hidden_sizes" in raw:
        raw["hidden_sizes"] = tuple(raw["hidden_sizes"])
    return TrainConfig(**raw)


def train(
    config: TrainConfig,
    env_factory: EnvFactory = ToyFsoRoutingEnv,
    agent: PPOAgent | None = None,
) -> TrainResult:
    """Train a PPO agent on the environment produced by env_factory.

    Args:
        config: Training hyperparameters and run settings.
        env_factory: Zero-argument callable returning a Gymnasium env with
            a flattened Box observation space and Discrete action space.
            Defaults to the built-in toy FSO routing env.
        agent: Optional pre-built agent (e.g. loaded from a checkpoint to
            resume training). Built fresh from config when omitted.

    Returns:
        TrainResult with the trained agent and logged learning curves.
    """
    if config.seed is not None:
        set_global_seed(config.seed)

    env = env_factory()
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)

    if agent is None:
        agent = PPOAgent(obs_dim, n_actions, config.ppo_config(), config.device)
    if config.seed is not None:
        agent.seed(config.seed)

    writer = None
    if config.log_dir is not None:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(config.log_dir)

    buffer = RolloutBuffer(config.rollout_steps, obs_dim)
    result = TrainResult(agent=agent)

    obs, _ = env.reset(seed=config.seed)
    episode_reward = 0.0
    global_step = 0
    update_idx = 0
    n_updates = config.total_steps // config.rollout_steps

    for update_idx in range(1, n_updates + 1):
        for _ in range(config.rollout_steps):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            buffer.add(obs, action, log_prob, reward, terminated, value)
            episode_reward += reward
            global_step += 1

            if terminated or truncated:
                result.episode_rewards.append(episode_reward)
                if writer is not None:
                    writer.add_scalar("rollout/episode_reward", episode_reward, global_step)
                episode_reward = 0.0
                next_obs, _ = env.reset()
            obs = next_obs

        last_value = 0.0 if terminated else agent.predict_value(obs)
        metrics = agent.update(buffer, last_value)
        result.update_metrics.append(metrics)

        if writer is not None:
            for key, val in metrics.items():
                writer.add_scalar(f"train/{key}", val, global_step)

        if (
            config.checkpoint_path is not None
            and config.checkpoint_every > 0
            and update_idx % config.checkpoint_every == 0
        ):
            agent.save(config.checkpoint_path)

    if config.checkpoint_path is not None:
        agent.save(config.checkpoint_path)
    if writer is not None:
        writer.close()
    env.close()
    return result


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    """Build a TrainConfig from CLI arguments, optionally seeded by YAML.

    CLI flags override values from --config.

    Args:
        argv: Argument list; None uses sys.argv.

    Returns:
        Fully resolved TrainConfig.
    """
    parser = argparse.ArgumentParser(description="Train the PPO routing agent")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config) if args.config else TrainConfig()
    for name in (
        "total_steps",
        "rollout_steps",
        "learning_rate",
        "gamma",
        "entropy_coef",
        "seed",
        "log_dir",
        "checkpoint_path",
        "device",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    return config


def main() -> None:
    """CLI entry point: train on the toy env and print a reward summary."""
    config = parse_args()
    result = train(config)
    rewards = result.episode_rewards
    if rewards:
        head = np.mean(rewards[: max(1, len(rewards) // 10)])
        tail = np.mean(rewards[-max(1, len(rewards) // 10):])
        print(f"episodes: {len(rewards)}")
        print(f"mean episode reward — first 10%: {head:.3f}, last 10%: {tail:.3f}")


if __name__ == "__main__":
    main()
