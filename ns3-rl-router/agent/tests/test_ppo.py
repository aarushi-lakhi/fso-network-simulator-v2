"""
Tests for the PPO routing agent.

Test strategy:
    - Correctness: GAE matches a hand-computed example; checkpoint
      round-trips produce identical policies; seeded rollouts reproduce.
    - Stability: no NaN/inf in losses, gradients, or parameters across
      many updates; entropy stays finite and positive.
    - Learning: on the toy FSO routing env, mean episode reward improves
      significantly over a short training run and entropy decreases as
      the policy commits.

The learning test is the load-bearing one: it certifies the full
rollout → GAE → clipped-surrogate update pipeline end to end before
the ns3-ai environment is attached.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from network import ActorCritic
from ppo_agent import PPOAgent, RolloutBuffer, compute_gae
from toy_env import ToyFsoRoutingEnv
from train import TrainConfig, load_config, parse_args, train

SEED = 42


def _short_train_config(**overrides: object) -> TrainConfig:
    """Toy-env training config small enough to finish in seconds."""
    defaults = dict(
        total_steps=16_384,
        rollout_steps=512,
        learning_rate=1e-3,
        entropy_coef=0.01,
        seed=SEED,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class TestActorCritic:
    def test_output_shapes(self) -> None:
        net = ActorCritic(obs_dim=16, n_actions=4)
        logits, value = net(torch.randn(8, 16))
        assert logits.shape == (8, 4)
        assert value.shape == (8,)

    def test_single_observation_is_batched(self) -> None:
        net = ActorCritic(obs_dim=16, n_actions=4)
        logits, value = net(torch.randn(16))
        assert logits.shape == (1, 4)
        assert value.shape == (1,)

    def test_initial_policy_near_uniform(self) -> None:
        net = ActorCritic(obs_dim=16, n_actions=4)
        dist = net.action_distribution(torch.randn(32, 16))
        max_entropy = np.log(4)
        assert dist.entropy().mean().item() > 0.99 * max_entropy

    def test_rejects_bad_dimensions(self) -> None:
        with pytest.raises(ValueError):
            ActorCritic(obs_dim=0, n_actions=4)
        with pytest.raises(ValueError):
            ActorCritic(obs_dim=16, n_actions=1)


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------


class TestGAE:
    def test_matches_hand_computed_example(self) -> None:
        # T=3, gamma=0.9, lambda=0.8, no terminals, bootstrap V(s_3)=0.2
        rewards = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        values = np.array([0.5, 0.4, 0.3], dtype=np.float32)
        dones = np.zeros(3, dtype=np.float32)
        gamma, lam = 0.9, 0.8

        # delta_2 = 1.0 + 0.9*0.2 - 0.3            = 0.88
        # delta_1 = 0.0 + 0.9*0.3 - 0.4            = -0.13
        # delta_0 = 1.0 + 0.9*0.4 - 0.5            = 0.86
        # A_2 = 0.88
        # A_1 = -0.13 + 0.72*0.88                  = 0.5036
        # A_0 = 0.86 + 0.72*0.5036                 = 1.222592
        expected_adv = np.array([1.222592, 0.5036, 0.88])

        adv, ret = compute_gae(rewards, values, dones, 0.2, gamma, lam)
        np.testing.assert_allclose(adv, expected_adv, rtol=1e-5)
        np.testing.assert_allclose(ret, expected_adv + values, rtol=1e-5)

    def test_done_cuts_bootstrap_and_recursion(self) -> None:
        rewards = np.array([1.0, 2.0], dtype=np.float32)
        values = np.array([0.5, 0.5], dtype=np.float32)
        dones = np.array([1.0, 0.0], dtype=np.float32)
        gamma, lam = 0.9, 0.8

        # delta_1 = 2.0 + 0.9*3.0 - 0.5 = 4.2 (bootstraps into last_value)
        # delta_0 = 1.0 - 0.5 = 0.5 (done: no bootstrap, no recursion)
        adv, _ = compute_gae(rewards, values, dones, 3.0, gamma, lam)
        np.testing.assert_allclose(adv, [0.5, 4.2], rtol=1e-6)

    def test_lambda_zero_reduces_to_td_error(self) -> None:
        rng = np.random.default_rng(SEED)
        rewards = rng.normal(size=10).astype(np.float32)
        values = rng.normal(size=10).astype(np.float32)
        dones = np.zeros(10, dtype=np.float32)
        last_value = 0.7

        adv, _ = compute_gae(rewards, values, dones, last_value, 0.99, 0.0)
        next_values = np.append(values[1:], last_value)
        td_errors = rewards + 0.99 * next_values - values
        np.testing.assert_allclose(adv, td_errors, rtol=1e-5)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------


class TestRolloutBuffer:
    def test_fills_and_rejects_overflow(self) -> None:
        buf = RolloutBuffer(capacity=4, obs_dim=3)
        obs = np.zeros(3, dtype=np.float32)
        for _ in range(4):
            buf.add(obs, 0, 0.0, 1.0, False, 0.5)
        assert buf.full
        with pytest.raises(IndexError):
            buf.add(obs, 0, 0.0, 1.0, False, 0.5)
        buf.reset()
        assert not buf.full

    def test_minibatches_cover_all_samples_once(self) -> None:
        buf = RolloutBuffer(capacity=10, obs_dim=2)
        for i in range(10):
            buf.add(np.full(2, i, dtype=np.float32), i % 3, 0.0, float(i), False, 0.0)
        buf.finalize(last_value=0.0, gamma=0.99, gae_lambda=0.95)

        seen = []
        for batch in buf.minibatches(4, np.random.default_rng(SEED)):
            seen.extend(batch["obs"][:, 0].tolist())
        assert sorted(seen) == list(range(10))

    def test_advantages_normalised_per_rollout(self) -> None:
        buf = RolloutBuffer(capacity=32, obs_dim=2)
        rng = np.random.default_rng(SEED)
        for _ in range(32):
            buf.add(np.zeros(2, np.float32), 0, 0.0, rng.normal(), False, rng.normal())
        buf.finalize(last_value=0.0, gamma=0.99, gae_lambda=0.95)

        adv = np.concatenate(
            [b["advantages"].numpy() for b in buf.minibatches(8, rng)]
        )
        assert abs(adv.mean()) < 1e-6
        assert abs(adv.std() - 1.0) < 1e-2


# ---------------------------------------------------------------------------
# Toy environment
# ---------------------------------------------------------------------------


class TestToyEnv:
    def test_gymnasium_interface_shapes(self) -> None:
        env = ToyFsoRoutingEnv(n_routes=4)
        obs, info = env.reset(seed=SEED)
        assert obs.shape == env.observation_space.shape == (16,)
        assert obs.dtype == np.float32

        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == (16,)
        assert isinstance(reward, float)
        assert not terminated
        assert "best_route" in info

    def test_truncates_at_episode_length(self) -> None:
        env = ToyFsoRoutingEnv(n_routes=3, episode_length=5)
        env.reset(seed=SEED)
        for i in range(5):
            _, _, _, truncated, _ = env.step(0)
        assert truncated

    def test_best_route_beats_worst_route(self) -> None:
        env = ToyFsoRoutingEnv(n_routes=4, flap_penalty=0.0)
        env.reset(seed=SEED)
        best_total, worst_total = 0.0, 0.0
        for _ in range(200):
            best = int(np.argmax(env._quality))
            worst = int(np.argmin(env._quality))
            _, r_best, _, truncated, _ = env.step(best)
            best_total += r_best
            if truncated:
                env.reset()
            _, r_worst, _, truncated, _ = env.step(worst)
            worst_total += r_worst
            if truncated:
                env.reset()
        assert best_total > worst_total

    def test_seeded_reset_reproducible(self) -> None:
        env1, env2 = ToyFsoRoutingEnv(), ToyFsoRoutingEnv()
        obs1, _ = env1.reset(seed=SEED)
        obs2, _ = env2.reset(seed=SEED)
        np.testing.assert_array_equal(obs1, obs2)


# ---------------------------------------------------------------------------
# Agent: checkpointing and reproducibility
# ---------------------------------------------------------------------------


class TestCheckpointing:
    def test_save_load_identical_policy_outputs(self, tmp_path) -> None:
        agent = PPOAgent(obs_dim=16, n_actions=4)
        path = tmp_path / "ckpt.pt"
        agent.save(path)

        restored = PPOAgent(obs_dim=16, n_actions=4)
        restored.load(path)

        obs = torch.randn(32, 16)
        with torch.no_grad():
            logits_a, values_a = agent.network(obs)
            logits_b, values_b = restored.network(obs)
        torch.testing.assert_close(logits_a, logits_b)
        torch.testing.assert_close(values_a, values_b)

    def test_load_rejects_mismatched_dimensions(self, tmp_path) -> None:
        agent = PPOAgent(obs_dim=16, n_actions=4)
        path = tmp_path / "ckpt.pt"
        agent.save(path)

        other = PPOAgent(obs_dim=16, n_actions=5)
        with pytest.raises(ValueError):
            other.load(path)


class TestSeedReproducibility:
    @staticmethod
    def _rollout(seed: int, n_steps: int = 64) -> tuple[list[int], list[float]]:
        torch.manual_seed(seed)
        env = ToyFsoRoutingEnv()
        agent = PPOAgent(
            obs_dim=int(np.prod(env.observation_space.shape)),
            n_actions=int(env.action_space.n),
        )
        obs, _ = env.reset(seed=seed)
        actions, rewards = [], []
        for _ in range(n_steps):
            action, _, _ = agent.select_action(obs)
            obs, reward, _, truncated, _ = env.step(action)
            actions.append(action)
            rewards.append(reward)
            if truncated:
                obs, _ = env.reset()
        return actions, rewards

    def test_same_seed_same_rollout(self) -> None:
        actions_a, rewards_a = self._rollout(SEED)
        actions_b, rewards_b = self._rollout(SEED)
        assert actions_a == actions_b
        np.testing.assert_array_equal(rewards_a, rewards_b)

    def test_different_seed_different_rollout(self) -> None:
        actions_a, _ = self._rollout(SEED)
        actions_b, _ = self._rollout(SEED + 1)
        assert actions_a != actions_b


# ---------------------------------------------------------------------------
# Training: numerical stability and learning
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_training_run() -> object:
    """One short training run shared by the learning/stability tests."""
    return train(_short_train_config())


class TestTrainingStability:
    def test_no_nan_or_inf_in_losses_and_grads(self, toy_training_run) -> None:
        for metrics in toy_training_run.update_metrics:
            for key, value in metrics.items():
                assert np.isfinite(value), f"{key} is not finite: {value}"

        for name, param in toy_training_run.agent.network.named_parameters():
            assert torch.isfinite(param).all(), f"non-finite parameter: {name}"

    def test_entropy_finite_positive_and_decreasing(self, toy_training_run) -> None:
        entropies = [m["entropy"] for m in toy_training_run.update_metrics]
        assert all(np.isfinite(e) and e > 0 for e in entropies)

        k = max(1, len(entropies) // 4)
        early, late = np.mean(entropies[:k]), np.mean(entropies[-k:])
        assert late < early, f"entropy did not decrease: {early:.3f} -> {late:.3f}"

    def test_reward_improves_significantly(self, toy_training_run) -> None:
        rewards = toy_training_run.episode_rewards
        assert len(rewards) >= 20

        k = max(1, len(rewards) // 5)
        early, late = np.mean(rewards[:k]), np.mean(rewards[-k:])
        early_std = np.std(rewards[:k])
        assert late > early + max(1.0, early_std), (
            f"no significant improvement: first-20% mean {early:.3f}, "
            f"last-20% mean {late:.3f}"
        )


class TestTrainLoop:
    def test_checkpoint_written_and_resumable(self, tmp_path) -> None:
        ckpt = tmp_path / "agent.pt"
        config = _short_train_config(
            total_steps=1024, checkpoint_path=str(ckpt)
        )
        result = train(config)
        assert ckpt.exists()

        restored = PPOAgent(obs_dim=16, n_actions=4)
        restored.load(ckpt)
        obs = torch.randn(4, 16)
        with torch.no_grad():
            logits_a, _ = result.agent.network(obs)
            logits_b, _ = restored.network(obs)
        torch.testing.assert_close(logits_a, logits_b)

    def test_tensorboard_events_written(self, tmp_path) -> None:
        config = _short_train_config(total_steps=1024, log_dir=str(tmp_path / "tb"))
        train(config)
        events = list((tmp_path / "tb").glob("events.out.tfevents.*"))
        assert events, "no TensorBoard event files written"

    def test_yaml_config_round_trip(self, tmp_path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "total_steps: 2048\nlearning_rate: 0.001\nhidden_sizes: [32, 32]\n"
        )
        config = load_config(cfg_file)
        assert config.total_steps == 2048
        assert config.learning_rate == 0.001
        assert config.hidden_sizes == (32, 32)

    def test_yaml_config_rejects_unknown_keys(self, tmp_path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("total_steps: 2048\nlerning_rate: 0.001\n")
        with pytest.raises(ValueError, match="lerning_rate"):
            load_config(cfg_file)

    def test_cli_overrides_yaml_config(self, tmp_path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("total_steps: 2048\nlearning_rate: 0.001\nseed: 7\n")
        config = parse_args(
            ["--config", str(cfg_file), "--total-steps", "4096", "--seed", "1"]
        )
        assert config.total_steps == 4096
        assert config.seed == 1
        assert config.learning_rate == 0.001  # YAML value survives

    def test_cli_defaults_without_config(self) -> None:
        config = parse_args([])
        assert config == TrainConfig()

    def test_custom_env_factory(self) -> None:
        config = _short_train_config(total_steps=1024)
        result = train(config, env_factory=lambda: ToyFsoRoutingEnv(n_routes=3))
        assert result.agent.network.n_actions == 3
