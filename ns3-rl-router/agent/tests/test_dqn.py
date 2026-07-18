"""Hermetic tests for the Double DQN machinery.

Test strategy:
    - ReplayBuffer: round-trip of stored transitions, circular eviction,
      sampling from the live region only.
    - linear_epsilon: endpoints, midpoint, degenerate schedule.
    - double_dqn_targets: hand-computed example where the online argmax
      and the target max disagree (the Double DQN distinction).
    - target sync: hard copy, and updates leave the target untouched
      until the next sync.
    - train_step: TD loss decreases on a synthetic fixed-point problem.
    - checkpoint round-trip: online + target restored bit-identical.
    - load_bc_policy: greedy actions equal the BC policy's argmax, the
      q_offset shifts values without changing the argmax, and the
      target network starts synced to the loaded weights.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dqn import (
    DQNAgent,
    QNetwork,
    ReplayBuffer,
    double_dqn_targets,
    linear_epsilon,
    q_gaps,
)
from imitation import BCConfig, split_by_episode, train_bc
from ppo_agent import PPOAgent

SEED = 42
OBS_DIM = 8
N_ACTIONS = 4


class TestReplayBuffer:
    def test_round_trip(self):
        buf = ReplayBuffer(capacity=16, obs_dim=OBS_DIM, seed=SEED)
        obs = np.arange(OBS_DIM, dtype=np.float32)
        buf.add(obs, 2, -1.5, obs + 1, done=True)
        assert len(buf) == 1
        batch = buf.sample(4)
        assert torch.equal(batch["obs"][0], torch.as_tensor(obs))
        assert batch["actions"][0] == 2
        assert batch["rewards"][0] == pytest.approx(-1.5)
        assert torch.equal(batch["next_obs"][0], torch.as_tensor(obs + 1))
        assert batch["dones"][0] == 1.0

    def test_eviction_overwrites_oldest(self):
        buf = ReplayBuffer(capacity=4, obs_dim=1, seed=SEED)
        for i in range(6):
            buf.add(np.array([float(i)]), i % N_ACTIONS, float(i),
                    np.array([float(i + 1)]), done=False)
        assert len(buf) == 4
        stored = sorted(float(o) for o in buf.obs[:, 0])
        assert stored == [2.0, 3.0, 4.0, 5.0]

    def test_sample_only_from_live_region(self):
        buf = ReplayBuffer(capacity=100, obs_dim=1, seed=SEED)
        buf.add(np.array([7.0]), 1, 0.0, np.array([7.0]), done=False)
        batch = buf.sample(32)
        assert torch.all(batch["obs"] == 7.0)
        assert torch.all(batch["actions"] == 1)

    def test_sample_empty_raises(self):
        buf = ReplayBuffer(capacity=4, obs_dim=1)
        with pytest.raises(ValueError, match="empty replay buffer"):
            buf.sample(1)


class TestLinearEpsilon:
    def test_endpoints_and_midpoint(self):
        assert linear_epsilon(0, 1.0, 0.05, 1000) == pytest.approx(1.0)
        assert linear_epsilon(500, 1.0, 0.05, 1000) == pytest.approx(0.525)
        assert linear_epsilon(1000, 1.0, 0.05, 1000) == pytest.approx(0.05)
        assert linear_epsilon(5000, 1.0, 0.05, 1000) == pytest.approx(0.05)

    def test_degenerate_schedule_returns_end(self):
        assert linear_epsilon(0, 1.0, 0.05, 0) == pytest.approx(0.05)


class TestDoubleDqnTargets:
    def test_hand_computed_example(self):
        # Online net prefers action 1 at s'; target net would prefer
        # action 0. Double DQN must evaluate the *online* argmax (1)
        # under the *target* net: y = r + gamma * Q_target(s', 1).
        rewards = torch.tensor([2.0, 3.0])
        dones = torch.tensor([0.0, 1.0])
        next_q_online = torch.tensor([[1.0, 5.0, 0.0],
                                      [9.0, 0.0, 0.0]])
        next_q_target = torch.tensor([[10.0, 4.0, 0.0],
                                      [7.0, 0.0, 0.0]])
        targets = double_dqn_targets(rewards, dones, next_q_online,
                                     next_q_target, gamma=0.5)
        # row 0: 2 + 0.5 * 4 = 4 (NOT 2 + 0.5 * 10 = 7, the vanilla max)
        # row 1: done cuts the bootstrap -> 3
        assert torch.allclose(targets, torch.tensor([4.0, 3.0]))

    def test_differs_from_vanilla_max(self):
        rewards = torch.zeros(1)
        dones = torch.zeros(1)
        next_q_online = torch.tensor([[0.0, 1.0]])
        next_q_target = torch.tensor([[5.0, 2.0]])
        target = double_dqn_targets(rewards, dones, next_q_online,
                                    next_q_target, gamma=1.0)
        assert target.item() == pytest.approx(2.0)  # vanilla would say 5.0


class TestTargetSync:
    def test_sync_copies_and_updates_do_not_leak(self):
        torch.manual_seed(SEED)
        agent = DQNAgent(OBS_DIM, N_ACTIONS)
        agent.seed(SEED)
        obs = torch.randn(8, OBS_DIM)
        assert torch.equal(agent.network(obs), agent.target(obs))

        buf = ReplayBuffer(64, OBS_DIM, seed=SEED)
        rng = np.random.default_rng(SEED)
        for _ in range(64):
            o = rng.uniform(0, 1, OBS_DIM).astype(np.float32)
            buf.add(o, int(rng.integers(N_ACTIONS)), 1.0, o, done=False)
        target_before = [p.detach().clone() for p in agent.target.parameters()]
        agent.train_step(buf.sample(32))
        for before, after in zip(target_before, agent.target.parameters()):
            assert torch.equal(before, after)
        assert not torch.equal(agent.network(obs), agent.target(obs))

        agent.sync_target()
        assert torch.equal(agent.network(obs), agent.target(obs))


class TestTrainStep:
    def test_td_loss_decreases_on_synthetic_data(self):
        # Terminal one-step transitions with reward = f(obs): the TD
        # target is the reward itself, a fixed regression problem the
        # Q-network must fit, so the Huber loss must fall.
        torch.manual_seed(SEED)
        agent = DQNAgent(OBS_DIM, N_ACTIONS)
        agent.seed(SEED)
        rng = np.random.default_rng(SEED)
        buf = ReplayBuffer(512, OBS_DIM, seed=SEED)
        for _ in range(512):
            o = rng.uniform(0, 1, OBS_DIM).astype(np.float32)
            action = int(rng.integers(N_ACTIONS))
            buf.add(o, action, float(o[action]), o, done=True)
        losses = [agent.train_step(buf.sample(64)) for _ in range(300)]
        assert np.mean(losses[-20:]) < np.mean(losses[:20]) / 2


class TestCheckpoint:
    def test_round_trip_bit_identical(self, tmp_path):
        torch.manual_seed(SEED)
        agent = DQNAgent(OBS_DIM, N_ACTIONS)
        agent.seed(SEED)
        buf = ReplayBuffer(64, OBS_DIM, seed=SEED)
        rng = np.random.default_rng(SEED)
        for _ in range(64):
            o = rng.uniform(0, 1, OBS_DIM).astype(np.float32)
            buf.add(o, int(rng.integers(N_ACTIONS)), float(o[0]), o,
                    done=False)
        agent.train_step(buf.sample(32))  # desync online from target
        path = tmp_path / "dqn.pt"
        agent.save(path)

        restored = DQNAgent(OBS_DIM, N_ACTIONS)
        restored.load(path)
        obs = torch.randn(16, OBS_DIM)
        assert torch.equal(agent.network(obs), restored.network(obs))
        assert torch.equal(agent.target(obs), restored.target(obs))

    def test_dimension_mismatch_rejected(self, tmp_path):
        agent = DQNAgent(OBS_DIM, N_ACTIONS)
        path = tmp_path / "dqn.pt"
        agent.save(path)
        other = DQNAgent(OBS_DIM + 1, N_ACTIONS)
        with pytest.raises(ValueError, match="do not match"):
            other.load(path)


class TestBcLoading:
    def _bc_agent(self) -> PPOAgent:
        """Behavior-clone a synthetic argmin teacher into a PPOAgent."""
        rng = np.random.default_rng(SEED)
        obs = rng.uniform(0, 1, size=(2000, OBS_DIM)).astype(np.float32)
        actions = obs[:, :N_ACTIONS].argmin(axis=1).astype(np.int64)
        episode_ids = np.repeat(np.arange(20), 100)
        train_idx, val_idx = split_by_episode(episode_ids, 0.2, rng)
        torch.manual_seed(SEED)
        agent = PPOAgent(OBS_DIM, N_ACTIONS)
        train_bc(agent.network, obs, actions, train_idx, val_idx,
                 BCConfig(epochs=20, learning_rate=3e-3, seed=SEED))
        return agent

    def test_greedy_policy_equals_bc_argmax(self, tmp_path):
        bc = self._bc_agent()
        path = tmp_path / "bc.pt"
        bc.save(path)

        torch.manual_seed(SEED + 1)
        agent = DQNAgent(OBS_DIM, N_ACTIONS)
        agent.load_bc_policy(path)
        obs = np.random.default_rng(SEED).uniform(
            0, 1, size=(64, OBS_DIM)).astype(np.float32)
        logits, _ = bc.network(torch.as_tensor(obs))
        bc_actions = logits.argmax(dim=-1).numpy()
        dqn_actions = np.array([agent.act_greedy(o) for o in obs])
        assert np.array_equal(bc_actions, dqn_actions)

    def test_q_offset_shifts_values_argmax_invariant(self, tmp_path):
        bc = self._bc_agent()
        path = tmp_path / "bc.pt"
        bc.save(path)

        torch.manual_seed(SEED + 1)
        plain = DQNAgent(OBS_DIM, N_ACTIONS)
        plain.load_bc_policy(path, q_offset=0.0)
        torch.manual_seed(SEED + 1)
        shifted = DQNAgent(OBS_DIM, N_ACTIONS)
        shifted.load_bc_policy(path, q_offset=-309.0)

        obs = torch.randn(32, OBS_DIM)
        q_plain = plain.network(obs)
        q_shifted = shifted.network(obs)
        assert torch.allclose(q_shifted, q_plain - 309.0, atol=1e-4)
        assert torch.equal(q_plain.argmax(dim=-1), q_shifted.argmax(dim=-1))

    def test_target_synced_after_load(self, tmp_path):
        bc = self._bc_agent()
        path = tmp_path / "bc.pt"
        bc.save(path)
        agent = DQNAgent(OBS_DIM, N_ACTIONS)
        agent.load_bc_policy(path, q_offset=-100.0)
        obs = torch.randn(8, OBS_DIM)
        assert torch.equal(agent.network(obs), agent.target(obs))

    def test_dimension_mismatch_rejected(self, tmp_path):
        bc = PPOAgent(OBS_DIM + 2, N_ACTIONS)
        path = tmp_path / "bc.pt"
        bc.save(path)
        agent = DQNAgent(OBS_DIM, N_ACTIONS)
        with pytest.raises(ValueError, match="do not match"):
            agent.load_bc_policy(path)


class TestRouteAwareObsDims:
    """Loader behaviour across the Phase 10 obs change (28 -> 32 dims)."""

    PLAIN, AWARE = 28, 32

    def test_route_aware_checkpoint_round_trips(self, tmp_path):
        path = tmp_path / "bc_route.pt"
        PPOAgent(self.AWARE, N_ACTIONS).save(path)
        agent = PPOAgent(self.AWARE, N_ACTIONS)
        agent.load(path)
        dqn = DQNAgent(self.AWARE, N_ACTIONS)
        dqn.load_bc_policy(path, q_offset=-300.0)
        assert dqn.network.obs_dim == self.AWARE

    def test_plain_checkpoint_rejected_by_route_aware_agents(self, tmp_path):
        path = tmp_path / "bc.pt"
        PPOAgent(self.PLAIN, N_ACTIONS).save(path)
        with pytest.raises(ValueError, match="do not match"):
            PPOAgent(self.AWARE, N_ACTIONS).load(path)
        with pytest.raises(ValueError, match="do not match"):
            DQNAgent(self.AWARE, N_ACTIONS).load_bc_policy(path)

    def test_route_aware_checkpoint_rejected_by_plain_agents(self, tmp_path):
        path = tmp_path / "bc_route.pt"
        PPOAgent(self.AWARE, N_ACTIONS).save(path)
        with pytest.raises(ValueError, match="do not match"):
            PPOAgent(self.PLAIN, N_ACTIONS).load(path)
        with pytest.raises(ValueError, match="do not match"):
            DQNAgent(self.PLAIN, N_ACTIONS).load_bc_policy(path)


class TestQGaps:
    def test_known_network_gap(self):
        net = QNetwork(2, 3, hidden_sizes=())
        with torch.no_grad():
            net.layers[0].weight.zero_()
            net.layers[0].bias.copy_(torch.tensor([1.0, 4.0, 2.5]))
        mean_gap, max_gap = q_gaps(net, torch.zeros(5, 2))
        assert mean_gap == pytest.approx(1.5)
        assert max_gap == pytest.approx(1.5)
