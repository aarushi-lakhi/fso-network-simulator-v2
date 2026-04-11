"""
Toy FSO routing environment for validating the PPO agent.

A miniature stand-in for the ns3-ai Gym environment with identical
interface shapes: flattened Box observation of per-route link metrics
(SNR, drop rate, scintillation index, queue depth) and a Discrete
action choosing one of K candidate routes.

Each route carries a latent quality q_k ∈ [0, 1] that drifts as a
mean-reverting random walk — mimicking slowly evolving atmospheric
turbulence. The observed metrics are noisy functions of q_k, and the
reward mirrors the real environment's structure:

    r = −drops − latency − flap_penalty·[route changed] + energy_bonus

An agent that reads the per-route metrics and tracks the currently
best route earns substantially more than a random or static policy,
so learning progress here certifies the PPO implementation before
ns-3 integration.

Typical usage:
    >>> env = ToyFsoRoutingEnv(n_routes=4)
    >>> obs, info = env.reset(seed=42)
    >>> obs, reward, terminated, truncated, info = env.step(0)
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

#: Observed metrics per route: [snr, drop_rate, scintillation, queue_depth].
FEATURES_PER_ROUTE: int = 4


class ToyFsoRoutingEnv(gym.Env):
    """K-route stochastic routing MDP with drifting link qualities.

    Attributes:
        observation_space: Box of shape (n_routes * 4,), float32.
        action_space: Discrete(n_routes) — index of the route to use.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_routes: int = 4,
        episode_length: int = 64,
        drift_rate: float = 0.15,
        drift_noise: float = 0.1,
        obs_noise: float = 0.02,
        flap_penalty: float = 0.05,
    ) -> None:
        """Configure the environment.

        Args:
            n_routes: Number of candidate routes K. Must be > 1.
            episode_length: Steps per episode before truncation.
            drift_rate: Mean-reversion speed of route qualities.
            drift_noise: Std of the per-step quality random walk.
            obs_noise: Std of measurement noise on observed metrics.
            flap_penalty: Reward penalty applied when the chosen route
                differs from the previous step's route.

        Raises:
            ValueError: If n_routes or episode_length are out of range.
        """
        super().__init__()
        if n_routes <= 1:
            raise ValueError(f"n_routes must be > 1, got {n_routes}")
        if episode_length <= 0:
            raise ValueError(f"episode_length must be positive, got {episode_length}")

        self.n_routes = n_routes
        self.episode_length = episode_length
        self.drift_rate = drift_rate
        self.drift_noise = drift_noise
        self.obs_noise = obs_noise
        self.flap_penalty = flap_penalty

        obs_dim = n_routes * FEATURES_PER_ROUTE
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(n_routes)

        self._quality = np.zeros(n_routes)
        self._quality_mean = np.zeros(n_routes)
        self._queue = np.zeros(n_routes)
        self._prev_action: int | None = None
        self._step_count = 0

    def _observe(self) -> np.ndarray:
        """Build the noisy flattened observation from latent state."""
        rng = self.np_random
        q = self._quality
        noise = lambda: rng.normal(0.0, self.obs_noise, self.n_routes)  # noqa: E731

        snr = q + noise()  # normalised SNR: higher is better
        drop_rate = (1.0 - q) * 0.5 + noise()
        scintillation = 0.1 + (1.0 - q) * 0.9 + noise()
        queue_depth = self._queue + noise()

        obs = np.stack([snr, drop_rate, scintillation, queue_depth], axis=1)
        return obs.reshape(-1).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode with freshly drawn route qualities.

        Args:
            seed: Optional RNG seed for reproducibility.
            options: Unused; present for Gymnasium API compatibility.

        Returns:
            Tuple (observation, info) per the Gymnasium API.
        """
        super().reset(seed=seed)
        rng = self.np_random
        self._quality_mean = rng.uniform(0.2, 0.9, self.n_routes)
        self._quality = self._quality_mean + rng.normal(0.0, 0.05, self.n_routes)
        self._quality = np.clip(self._quality, 0.0, 1.0)
        self._queue = rng.uniform(0.0, 0.3, self.n_routes)
        self._prev_action = None
        self._step_count = 0
        return self._observe(), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Route traffic over the chosen route and advance link dynamics.

        Args:
            action: Route index in [0, n_routes).

        Returns:
            Tuple (observation, reward, terminated, truncated, info).
            info contains the latent quality of the chosen route and the
            index of the currently optimal route.
        """
        action = int(action)
        rng = self.np_random
        q = self._quality[action]

        drops = (1.0 - q) * 0.5
        latency = 0.1 + 0.3 * self._queue[action]
        flapped = self._prev_action is not None and action != self._prev_action
        energy_bonus = 0.02

        reward = float(
            -drops - latency - self.flap_penalty * flapped + energy_bonus
        )

        # Mean-reverting drift of route qualities (turbulence evolution)
        self._quality += self.drift_rate * (self._quality_mean - self._quality)
        self._quality += rng.normal(0.0, self.drift_noise, self.n_routes)
        self._quality = np.clip(self._quality, 0.0, 1.0)

        # Chosen route's queue fills slightly; idle routes drain
        self._queue *= 0.9
        self._queue[action] = min(self._queue[action] + 0.05, 1.0)

        self._prev_action = action
        self._step_count += 1
        truncated = self._step_count >= self.episode_length
        info = {
            "chosen_quality": q,
            "best_route": int(np.argmax(self._quality)),
        }
        return self._observe(), reward, False, truncated, info
