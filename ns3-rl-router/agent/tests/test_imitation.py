"""Hermetic tests for the imitation (BC + warmup) machinery.

Test strategy:
    - split_by_episode: whole episodes only, both splits non-empty.
    - train_bc: cross-entropy decreases and validation accuracy is high
      on a synthetic linearly-separable teacher; critic untouched.
    - checkpoint round-trip: a BC-trained policy saved through
      PPOAgent.save loads back bit-identical.
    - count_route_switches: matches the benchmark's eval-time metric.
    - mean_kl_from_reference: zero against itself, positive otherwise.
    - fit_value: value loss decreases while the actor stays frozen.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from imitation import (
    BCConfig,
    count_route_switches,
    evaluate_actions,
    fit_value,
    greedy_actions,
    mean_entropy,
    mean_kl_from_reference,
    split_by_episode,
    train_bc,
)
from network import ActorCritic
from ppo_agent import PPOAgent, RolloutBuffer

SEED = 42
OBS_DIM = 8
N_ACTIONS = 4


def synthetic_teacher_dataset(
    n_episodes: int = 20, steps: int = 100, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observations whose teacher action is a deterministic function.

    The teacher picks the argmin of the first N_ACTIONS observation
    features — the same shape of rule as the greedy-PER teacher, and
    exactly learnable by an MLP.
    """
    rng = np.random.default_rng(seed)
    obs = rng.uniform(0, 1, size=(n_episodes * steps, OBS_DIM)).astype(np.float32)
    actions = obs[:, :N_ACTIONS].argmin(axis=1).astype(np.int64)
    episode_ids = np.repeat(np.arange(n_episodes), steps)
    return obs, actions, episode_ids


class TestSplitByEpisode:
    def test_episodes_do_not_straddle_splits(self):
        _, _, episode_ids = synthetic_teacher_dataset()
        rng = np.random.default_rng(SEED)
        train_idx, val_idx = split_by_episode(episode_ids, 0.25, rng)
        train_eps = set(episode_ids[train_idx])
        val_eps = set(episode_ids[val_idx])
        assert train_eps.isdisjoint(val_eps)
        assert len(train_idx) + len(val_idx) == len(episode_ids)
        assert len(val_eps) == 5  # 25% of 20

    def test_at_least_one_validation_episode(self):
        episode_ids = np.repeat(np.arange(2), 10)
        rng = np.random.default_rng(SEED)
        _, val_idx = split_by_episode(episode_ids, 0.01, rng)
        assert len(val_idx) == 10

    def test_rejects_single_episode(self):
        with pytest.raises(ValueError, match="at least two episodes"):
            split_by_episode(np.zeros(10, dtype=np.int64), 0.2,
                             np.random.default_rng(SEED))


class TestTrainBC:
    def test_loss_decreases_and_accuracy_high(self):
        obs, actions, episode_ids = synthetic_teacher_dataset()
        rng = np.random.default_rng(SEED)
        train_idx, val_idx = split_by_episode(episode_ids, 0.2, rng)
        torch.manual_seed(SEED)
        net = ActorCritic(OBS_DIM, N_ACTIONS)
        result = train_bc(net, obs, actions, train_idx, val_idx,
                          BCConfig(epochs=60, learning_rate=3e-3, seed=SEED))
        assert result.train_losses[-1] < result.train_losses[0] / 2
        assert result.val_losses[-1] < result.val_losses[0]
        assert result.val_accuracy > 0.9

    def test_critic_parameters_untouched(self):
        obs, actions, episode_ids = synthetic_teacher_dataset(n_episodes=4)
        rng = np.random.default_rng(SEED)
        train_idx, val_idx = split_by_episode(episode_ids, 0.25, rng)
        torch.manual_seed(SEED)
        net = ActorCritic(OBS_DIM, N_ACTIONS)
        critic_before = [p.detach().clone() for p in net.critic.parameters()]
        actor_before = [p.detach().clone() for p in net.actor.parameters()]
        train_bc(net, obs, actions, train_idx, val_idx,
                 BCConfig(epochs=3, seed=SEED))
        for before, after in zip(critic_before, net.critic.parameters()):
            assert torch.equal(before, after)
        assert any(not torch.equal(b, a)
                   for b, a in zip(actor_before, net.actor.parameters()))

    def test_checkpoint_round_trip(self, tmp_path):
        obs, actions, episode_ids = synthetic_teacher_dataset(n_episodes=4)
        rng = np.random.default_rng(SEED)
        train_idx, val_idx = split_by_episode(episode_ids, 0.25, rng)
        torch.manual_seed(SEED)
        agent = PPOAgent(OBS_DIM, N_ACTIONS)
        train_bc(agent.network, obs, actions, train_idx, val_idx,
                 BCConfig(epochs=3, seed=SEED))
        path = tmp_path / "bc.pt"
        agent.save(path)

        restored = PPOAgent(OBS_DIM, N_ACTIONS)
        restored.load(path)
        obs_t = torch.as_tensor(obs[:32])
        logits_a, value_a = agent.network(obs_t)
        logits_b, value_b = restored.network(obs_t)
        assert torch.equal(logits_a, logits_b)
        assert torch.equal(value_a, value_b)

    def test_evaluate_actions_perfect_on_memorised_labels(self):
        torch.manual_seed(SEED)
        net = ActorCritic(OBS_DIM, N_ACTIONS)
        obs = torch.eye(OBS_DIM)[:N_ACTIONS]
        labels = greedy_actions(net, obs)
        loss, acc = evaluate_actions(net, obs, torch.as_tensor(labels))
        assert acc == 1.0
        assert loss > 0.0


class TestSwitchCounting:
    def test_counts_first_step_off_route_zero(self):
        actions = np.array([1, 1, 1])
        dones = np.zeros(3)
        assert count_route_switches(actions, dones) == 1

    def test_counts_changes_within_episode(self):
        actions = np.array([0, 2, 2, 3, 0])
        dones = np.zeros(5)
        assert count_route_switches(actions, dones) == 3

    def test_done_resets_to_initial_route(self):
        # Two episodes: [1, 1] then [1]; each episode starts on route 0
        actions = np.array([1, 1, 1])
        dones = np.array([0.0, 1.0, 0.0])
        assert count_route_switches(actions, dones) == 2

    def test_constant_route_zero_never_switches(self):
        actions = np.zeros(10, dtype=np.int64)
        dones = np.zeros(10)
        dones[4] = 1.0
        assert count_route_switches(actions, dones) == 0


class TestPolicyDiagnostics:
    def test_kl_zero_against_itself(self):
        torch.manual_seed(SEED)
        net = ActorCritic(OBS_DIM, N_ACTIONS)
        obs = torch.randn(16, OBS_DIM)
        assert mean_kl_from_reference(net, net, obs) == pytest.approx(0.0)

    def test_kl_positive_for_different_policies(self):
        torch.manual_seed(SEED)
        ref = ActorCritic(OBS_DIM, N_ACTIONS)
        net = ActorCritic(OBS_DIM, N_ACTIONS)
        with torch.no_grad():
            for p in net.actor.parameters():
                p.add_(torch.randn_like(p))
        obs = torch.randn(16, OBS_DIM)
        assert mean_kl_from_reference(ref, net, obs) > 0.0

    def test_entropy_bounded_by_uniform(self):
        torch.manual_seed(SEED)
        net = ActorCritic(OBS_DIM, N_ACTIONS)
        obs = torch.randn(16, OBS_DIM)
        h = mean_entropy(net, obs)
        assert 0.0 < h <= float(np.log(N_ACTIONS)) + 1e-6


class TestFitValue:
    def _filled_buffer(self, agent: PPOAgent, steps: int = 256) -> tuple:
        rng = np.random.default_rng(SEED)
        buffer = RolloutBuffer(steps, OBS_DIM)
        obs = rng.uniform(0, 1, size=(steps, OBS_DIM)).astype(np.float32)
        # Reward is a fixed function of the observation: learnable values
        rewards = obs[:, 0].astype(np.float32)
        for t in range(steps):
            _, _, value = agent.select_action(obs[t])
            buffer.add(obs[t], 0, 0.0, float(rewards[t]),
                       done=(t % 64 == 63), value=value)
        return buffer, obs

    def test_value_loss_decreases_actor_frozen(self):
        torch.manual_seed(SEED)
        agent = PPOAgent(OBS_DIM, N_ACTIONS)
        agent.seed(SEED)
        critic_opt = torch.optim.Adam(agent.network.critic.parameters(), lr=1e-3)
        actor_before = [p.detach().clone()
                        for p in agent.network.actor.parameters()]
        losses = []
        for _ in range(6):
            buffer, _ = self._filled_buffer(agent)
            losses.append(fit_value(agent, buffer, last_value=0.0,
                                    optimizer=critic_opt))
        assert losses[-1] < losses[0]
        for before, after in zip(actor_before, agent.network.actor.parameters()):
            assert torch.equal(before, after)

    def test_buffer_reset_after_fit(self):
        torch.manual_seed(SEED)
        agent = PPOAgent(OBS_DIM, N_ACTIONS)
        agent.seed(SEED)
        critic_opt = torch.optim.Adam(agent.network.critic.parameters(), lr=1e-3)
        buffer, _ = self._filled_buffer(agent, steps=64)
        fit_value(agent, buffer, last_value=0.0, optimizer=critic_opt)
        assert buffer.pos == 0
