"""Hermetic tests for the FlatFrameStack wrapper (toy env, no ns-3)."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frame_stack import FlatFrameStack  # noqa: E402
from toy_env import ToyFsoRoutingEnv  # noqa: E402


class _CountingEnv(gym.Env):
    """Flat Box env whose observation is [step, step] for easy checks."""

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(-1e6, 1e6, shape=(2,),
                                                dtype=np.float64)
        self.action_space = gym.spaces.Discrete(2)
        self._step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return np.zeros(2), {}

    def step(self, action):
        self._step += 1
        obs = np.full(2, float(self._step))
        return obs, 0.0, False, self._step >= 5, {"info": "route=0"}


def test_observation_space_is_tiled() -> None:
    env = FlatFrameStack(_CountingEnv(), k=3)
    assert env.observation_space.shape == (6,)


def test_reset_fills_stack_with_initial_observation() -> None:
    env = FlatFrameStack(_CountingEnv(), k=3)
    obs, _ = env.reset()
    assert obs.shape == (6,)
    np.testing.assert_array_equal(obs, np.zeros(6))


def test_step_appends_newest_frame_last() -> None:
    env = FlatFrameStack(_CountingEnv(), k=3)
    env.reset()
    obs, _, _, _, _ = env.step(0)   # frames: 0, 0, 1
    np.testing.assert_array_equal(obs, [0, 0, 0, 0, 1, 1])
    obs, _, _, _, _ = env.step(0)   # frames: 0, 1, 2
    np.testing.assert_array_equal(obs, [0, 0, 1, 1, 2, 2])
    obs, _, _, _, _ = env.step(0)   # frames: 1, 2, 3
    np.testing.assert_array_equal(obs, [1, 1, 2, 2, 3, 3])


def test_reset_clears_previous_episode_history() -> None:
    env = FlatFrameStack(_CountingEnv(), k=2)
    env.reset()
    env.step(0)
    obs, _ = env.reset()
    np.testing.assert_array_equal(obs, np.zeros(4))


def test_reward_done_info_pass_through() -> None:
    env = FlatFrameStack(_CountingEnv(), k=2)
    env.reset()
    for _ in range(4):
        _, reward, terminated, truncated, info = env.step(0)
        assert reward == 0.0 and not terminated and not truncated
        assert info == {"info": "route=0"}
    _, _, _, truncated, _ = env.step(0)
    assert truncated


def test_k_one_is_identity_on_observations() -> None:
    env = FlatFrameStack(_CountingEnv(), k=1)
    obs, _ = env.reset()
    np.testing.assert_array_equal(obs, np.zeros(2))
    obs, _, _, _, _ = env.step(0)
    np.testing.assert_array_equal(obs, [1, 1])


def test_rejects_bad_stack_depth_and_space() -> None:
    with pytest.raises(ValueError):
        FlatFrameStack(_CountingEnv(), k=0)

    class _ImageEnv(_CountingEnv):
        def __init__(self) -> None:
            super().__init__()
            self.observation_space = gym.spaces.Box(0, 1, shape=(2, 2))

    with pytest.raises(ValueError):
        FlatFrameStack(_ImageEnv(), k=2)


def test_wraps_the_toy_routing_env() -> None:
    env = FlatFrameStack(ToyFsoRoutingEnv(), k=4)
    obs, _ = env.reset(seed=7)
    base = np.prod(env.env.observation_space.shape)
    assert obs.shape == (4 * base,)
    obs, _, _, _, _ = env.step(int(env.action_space.sample()))
    assert obs.shape == (4 * base,)
