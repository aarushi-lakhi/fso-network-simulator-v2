"""
Proximal Policy Optimization (PPO) for FSO mesh route selection.

Implements the clipped-surrogate PPO variant (Schulman et al., 2017)
with Generalized Advantage Estimation (Schulman et al., 2016):

    L^CLIP(θ) = E_t[ min(r_t(θ) Â_t, clip(r_t(θ), 1−ε, 1+ε) Â_t) ]
    L(θ)      = −L^CLIP + c_v · L^VF − c_e · H[π_θ]

The agent is environment-agnostic: it consumes flattened Box
observations and produces Discrete actions, matching both the toy
routing env and the ns3-ai Gym env that replaces it at integration.

Typical usage:
    >>> agent = PPOAgent(obs_dim=20, n_actions=5, config=PPOConfig())
    >>> action, logprob, value = agent.select_action(obs)
    >>> metrics = agent.update(buffer, last_value=0.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn

from network import ActorCritic


@dataclass
class PPOConfig:
    """Hyperparameters for PPO training.

    Attributes:
        learning_rate: Adam step size.
        gamma: Discount factor for future rewards.
        gae_lambda: GAE bias-variance trade-off parameter λ.
        clip_epsilon: PPO surrogate clipping range ε.
        value_coef: Weight c_v of the value loss term.
        entropy_coef: Weight c_e of the entropy bonus.
        max_grad_norm: Global gradient-norm clipping threshold.
        n_epochs: Optimisation epochs per rollout.
        minibatch_size: Samples per SGD minibatch.
        hidden_sizes: Actor/critic trunk widths.
    """

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


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE(λ) advantages and discounted returns.

    Recursion (Schulman et al., 2016, Eq. 16):
        δ_t = r_t + γ V(s_{t+1}) (1 − done_t) − V(s_t)
        Â_t = δ_t + γ λ (1 − done_t) Â_{t+1}

    where done_t = 1 means the episode terminated at step t, cutting
    the bootstrap. Returns are advantages + values (TD(λ) targets).

    Args:
        rewards: Shape (T,) rewards r_t.
        values: Shape (T,) value estimates V(s_t).
        dones: Shape (T,) episode-termination flags (0.0 or 1.0).
        last_value: Bootstrap value V(s_T) for the state after the rollout.
        gamma: Discount factor.
        gae_lambda: GAE λ.

    Returns:
        Tuple (advantages, returns), each of shape (T,).
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        gae = delta + gamma * gae_lambda * not_done * gae
        advantages[t] = gae
    return advantages, advantages + values


class RolloutBuffer:
    """Fixed-capacity storage for one on-policy rollout.

    Stores transitions step by step, then computes GAE targets once the
    rollout is complete and serves shuffled minibatches for optimisation.
    """

    def __init__(self, capacity: int, obs_dim: int) -> None:
        """Allocate storage.

        Args:
            capacity: Number of transitions per rollout.
            obs_dim: Flattened observation dimensionality.
        """
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    @property
    def full(self) -> bool:
        """Whether the buffer holds a complete rollout."""
        return self.pos == self.capacity

    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        """Append one transition.

        Args:
            obs: Observation s_t, shape (obs_dim,).
            action: Discrete action a_t.
            log_prob: log π(a_t | s_t) under the behaviour policy.
            reward: Reward r_t.
            done: Whether the episode terminated at this step.
            value: Critic estimate V(s_t).

        Raises:
            IndexError: If the buffer is already full.
        """
        if self.full:
            raise IndexError("RolloutBuffer is full; call reset() first")
        i = self.pos
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.values[i] = value
        self.pos += 1

    def finalize(self, last_value: float, gamma: float, gae_lambda: float) -> None:
        """Compute advantages and returns for the stored rollout.

        Args:
            last_value: Bootstrap value for the state following the rollout.
            gamma: Discount factor.
            gae_lambda: GAE λ.
        """
        self.advantages, self.returns = compute_gae(
            self.rewards[: self.pos],
            self.values[: self.pos],
            self.dones[: self.pos],
            last_value,
            gamma,
            gae_lambda,
        )

    def minibatches(
        self, batch_size: int, rng: np.random.Generator
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield shuffled minibatches of the rollout as tensors.

        Advantages are normalised per rollout (zero mean, unit variance).

        Args:
            batch_size: Samples per minibatch.
            rng: NumPy generator controlling the shuffle.

        Yields:
            Dicts with keys: obs, actions, log_probs, advantages, returns.
        """
        n = self.pos
        adv = self.advantages[:n]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        indices = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "obs": torch.as_tensor(self.obs[idx]),
                "actions": torch.as_tensor(self.actions[idx]),
                "log_probs": torch.as_tensor(self.log_probs[idx]),
                "advantages": torch.as_tensor(adv[idx]),
                "returns": torch.as_tensor(self.returns[idx]),
            }

    def reset(self) -> None:
        """Mark the buffer empty for the next rollout."""
        self.pos = 0


class PPOAgent:
    """Clipped-surrogate PPO agent over an ActorCritic network."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        config: PPOConfig | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialise network, optimiser, and update RNG.

        Args:
            obs_dim: Flattened observation dimensionality.
            n_actions: Number of discrete actions.
            config: Hyperparameters; defaults to PPOConfig().
            device: Torch device string.
        """
        self.config = config or PPOConfig()
        self.device = torch.device(device)
        self.network = ActorCritic(obs_dim, n_actions, self.config.hidden_sizes).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.config.learning_rate
        )
        self._rng = np.random.default_rng()

    def seed(self, seed: int) -> None:
        """Seed the minibatch-shuffling RNG.

        Args:
            seed: Seed value.
        """
        self._rng = np.random.default_rng(seed)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> tuple[int, float, float]:
        """Sample an action from the current policy.

        Args:
            obs: Observation of shape (obs_dim,).

        Returns:
            Tuple (action, log_prob, value) for storage in the rollout buffer.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        logits, value = self.network(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(
            value.item()
        )

    @torch.no_grad()
    def predict_value(self, obs: np.ndarray) -> float:
        """Return the critic's value estimate for one observation.

        Args:
            obs: Observation of shape (obs_dim,).

        Returns:
            Scalar V(s) estimate, used to bootstrap GAE at rollout end.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        _, value = self.network(obs_t)
        return float(value.item())

    @torch.no_grad()
    def act_greedy(self, obs: np.ndarray) -> int:
        """Return the argmax action (deterministic policy for evaluation).

        Args:
            obs: Observation of shape (obs_dim,).

        Returns:
            Index of the most probable action.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        logits, _ = self.network(obs_t)
        return int(logits.argmax(dim=-1).item())

    def update(self, buffer: RolloutBuffer, last_value: float) -> dict[str, float]:
        """Run PPO minibatch epochs over one rollout.

        Args:
            buffer: Filled rollout buffer.
            last_value: Bootstrap value V(s_T) for GAE.

        Returns:
            Mean training metrics over all minibatches: policy_loss,
            value_loss, entropy, approx_kl, clip_fraction, grad_norm.
        """
        cfg = self.config
        buffer.finalize(last_value, cfg.gamma, cfg.gae_lambda)

        metrics: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "grad_norm": [],
        }

        for _ in range(cfg.n_epochs):
            for batch in buffer.minibatches(cfg.minibatch_size, self._rng):
                logits, values = self.network(batch["obs"].to(self.device))
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(batch["actions"].to(self.device))
                entropy = dist.entropy().mean()

                advantages = batch["advantages"].to(self.device)
                ratio = torch.exp(log_probs - batch["log_probs"].to(self.device))
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon)
                policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

                value_loss = nn.functional.mse_loss(
                    values, batch["returns"].to(self.device)
                )

                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.network.parameters(), cfg.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (batch["log_probs"].to(self.device) - log_probs).mean()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_epsilon).float().mean()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.item())
                metrics["approx_kl"].append(approx_kl.item())
                metrics["clip_fraction"].append(clip_frac.item())
                metrics["grad_norm"].append(float(grad_norm))

        buffer.reset()
        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def save(self, path: str | Path) -> None:
        """Save network and optimiser state to a checkpoint file.

        Args:
            path: Destination file path (created/overwritten).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "obs_dim": self.network.obs_dim,
                "n_actions": self.network.n_actions,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """Restore network and optimiser state from a checkpoint file.

        Args:
            path: Checkpoint file written by save().

        Raises:
            ValueError: If the checkpoint's dimensions do not match this agent.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        if (
            ckpt["obs_dim"] != self.network.obs_dim
            or ckpt["n_actions"] != self.network.n_actions
        ):
            raise ValueError(
                f"checkpoint dims (obs={ckpt['obs_dim']}, act={ckpt['n_actions']}) "
                f"do not match agent (obs={self.network.obs_dim}, "
                f"act={self.network.n_actions})"
            )
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
