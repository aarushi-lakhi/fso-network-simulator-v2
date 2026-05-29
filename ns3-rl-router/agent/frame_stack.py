"""Flat frame-stacking observation wrapper (Phase 7b policy memory).

Concatenates the last ``k`` observations of a flat Box environment into
one flat observation, oldest first, so an MLP policy sees the recent
trajectory of the channel state instead of a single snapshot. On reset
the stack is filled with ``k`` copies of the initial observation.

Unlike ``gymnasium.wrappers.FrameStackObservation`` this keeps the
observation flat (shape ``[k * n]`` instead of ``[k, n]``), matching
what the PPO agent's flattening expects, and has no other behaviour.

Typical usage:
    >>> env = FlatFrameStack(make_ns3_env(...), k=8)
    >>> obs, _ = env.reset(seed=42)   # shape (8 * 28,)
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np


class FlatFrameStack(gym.Wrapper):
    """Stack the last ``k`` flat observations into one flat observation.

    Attributes:
        k: Number of observations stacked.
    """

    def __init__(self, env: gym.Env, k: int) -> None:
        """Wrap ``env`` so observations carry ``k`` steps of history.

        Args:
            env: Environment with a flat (1-D) Box observation space.
            k: Number of observations to stack (>= 1).

        Raises:
            ValueError: If ``k`` < 1 or the observation space is not a
                1-D Box.
        """
        super().__init__(env)
        if k < 1:
            raise ValueError(f"stack depth must be >= 1, got {k}")
        space = env.observation_space
        if not isinstance(space, gym.spaces.Box) or len(space.shape) != 1:
            raise ValueError(f"expected a 1-D Box observation space, got {space}")
        self.k = k
        self._frames: deque[np.ndarray] = deque(maxlen=k)
        self.observation_space = gym.spaces.Box(
            low=np.tile(space.low, k),
            high=np.tile(space.high, k),
            dtype=space.dtype,
        )

    def _stacked(self) -> np.ndarray:
        """Return the current stack as one flat array, oldest first."""
        return np.concatenate(list(self._frames))

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the env and fill the stack with the initial observation.

        Args:
            **kwargs: Passed through to the wrapped env's reset.

        Returns:
            Tuple (stacked observation, info).
        """
        obs, info = self.env.reset(**kwargs)
        frame = np.asarray(obs, dtype=self.observation_space.dtype)
        self._frames.clear()
        for _ in range(self.k):
            self._frames.append(frame)
        return self._stacked(), info

    def step(self, action: Any) -> tuple[np.ndarray, Any, bool, bool, dict[str, Any]]:
        """Step the env and append the new observation to the stack.

        Args:
            action: Passed through to the wrapped env's step.

        Returns:
            Tuple (stacked observation, reward, terminated, truncated,
            info) per the Gymnasium API.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(np.asarray(obs, dtype=self.observation_space.dtype))
        return self._stacked(), reward, terminated, truncated, info
