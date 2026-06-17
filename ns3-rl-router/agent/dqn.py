"""Double DQN for FSO mesh route selection (the Phase 9 experiment).

Phase 8 convicted the on-policy gradient: PPO destroys a working
switching policy it was handed. This module is the off-policy arm of
the {PPO, DQN} x {scratch, BC-init} 2x2 — Double DQN (van Hasselt et
al., 2016) estimates per-action values by averaging TD errors over a
replay buffer instead of the last on-policy batch, exactly the property
that should resist the ~25% per-episode return noise that collapses PPO.

Components: a Q-network with the ActorCritic actor's trunk shape (so
the Phase 8 BC checkpoints load key-for-key), a uniform replay buffer,
a periodically-synced target network, an epsilon-greedy linear
schedule, Huber loss, and gradient clipping.

BC -> Q initialisation: the BC policy's logits become the initial
Q-values (identical architecture, weights copied verbatim), so the
greedy policy at initialisation *is* the BC policy. Logits live at
O(1) while returns live at O(-10^2..-10^3), so a constant offset —
argmax-invariant, added to the final layer's bias — moves the whole
Q-surface to the return scale (the caller derives it from the BC
policy's measured mean step reward: offset = r_step / (1 - gamma)).
The residual per-action value differences are then learned by TD
refinement instead of a from-scratch rebuild that would have to pass
through a value surface with an arbitrary argmax. The alternative
(distilling Q-values from teacher rollouts) needs value targets the BC
dataset does not contain; the copy-plus-offset mapping preserves the
policy exactly and is testable in isolation.

The ``train`` CLI owns a gym env and must run in its own process
(ns3-ai allows a single Experiment per process — the Phase 8 stage
pattern). Every ``--trajectory-interval`` env steps it appends one row
of greedy-switch / Q-gap / epsilon / TD-loss diagnostics to a
trajectory CSV — that trajectory is the experiment's result.

Typical usage (agent venv active, modules linked and built):
    $ python dqn.py train --checkpoint dqn.pt --trajectory-csv traj.csv \\
          --total-steps 80000 --c2n 1e-13 --topology disjoint ...
    $ python dqn.py train --bc-checkpoint bc.pt --q-offset -309 \\
          --eps-start 0.05 --eps-end 0.01 --eps-decay-steps 20000 ...
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from imitation import count_route_switches
from network import _orthogonal_mlp

TRAJECTORY_FIELDS = ("update", "global_step", "epsilon", "td_loss",
                     "mean_q_gap", "max_q_gap", "sampled_switches_per_200",
                     "greedy_switches_per_200", "mean_episode_reward",
                     "episodes_done")


@dataclass
class DQNConfig:
    """Hyperparameters for Double DQN training.

    Attributes:
        learning_rate: Adam step size.
        gamma: Discount factor for future rewards.
        buffer_capacity: Replay buffer size [transitions].
        batch_size: Transitions per gradient step.
        learning_starts: Env steps collected before the first update.
        target_sync_interval: Env steps between hard target syncs.
        max_grad_norm: Global gradient-norm clipping threshold.
        hidden_sizes: Q-network trunk widths (must match the BC
            checkpoints' actor trunk for weight loading).
    """

    learning_rate: float = 3e-4
    gamma: float = 0.99
    buffer_capacity: int = 50_000
    batch_size: int = 64
    learning_starts: int = 1_000
    target_sync_interval: int = 1_000
    max_grad_norm: float = 10.0
    hidden_sizes: tuple[int, ...] = (64, 64)


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    """Linearly decayed exploration rate at an env step.

    Args:
        step: Current env step (0-based).
        start: Epsilon at step 0.
        end: Epsilon from ``decay_steps`` onward.
        decay_steps: Steps over which epsilon decays; <= 0 returns end.

    Returns:
        The epsilon value, clipped to [min(start, end), max(start, end)].
    """
    if decay_steps <= 0 or step >= decay_steps:
        return end
    return start + (end - start) * step / decay_steps


class ReplayBuffer:
    """Uniform-sampling circular replay buffer for off-policy learning."""

    def __init__(self, capacity: int, obs_dim: int, seed: int | None = None) -> None:
        """Allocate storage.

        Args:
            capacity: Maximum stored transitions (oldest evicted first).
            obs_dim: Flattened observation dimensionality.
            seed: RNG seed for uniform sampling.
        """
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Number of transitions currently stored."""
        return self.size

    def add(self, obs: np.ndarray, action: int, reward: float,
            next_obs: np.ndarray, done: bool) -> None:
        """Append one transition, evicting the oldest when full.

        Args:
            obs: Observation s_t.
            action: Discrete action a_t.
            reward: Reward r_t.
            next_obs: Observation s_{t+1} (pre-reset on episode end).
            done: Whether the episode terminated at this step (cuts the
                bootstrap in the TD target).
        """
        i = self.pos
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a uniform minibatch of stored transitions.

        Args:
            batch_size: Number of transitions to draw (with replacement
                across calls, without within one call when possible).

        Returns:
            Dict of tensors: obs, actions, rewards, next_obs, dones.

        Raises:
            ValueError: If the buffer is empty.
        """
        if self.size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        idx = self._rng.integers(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx]),
            "actions": torch.as_tensor(self.actions[idx]),
            "rewards": torch.as_tensor(self.rewards[idx]),
            "next_obs": torch.as_tensor(self.next_obs[idx]),
            "dones": torch.as_tensor(self.dones[idx]),
        }


class QNetwork(nn.Module):
    """MLP mapping observations to per-action Q-values.

    Same trunk shape and initialisation as the ActorCritic actor
    (network.py), so a Phase 8 BC checkpoint's actor weights load
    verbatim and the greedy policy at initialisation equals the BC
    policy's argmax.

    Attributes:
        layers: The Tanh MLP ending in an n_actions-wide linear layer.
    """

    def __init__(self, obs_dim: int, n_actions: int,
                 hidden_sizes: tuple[int, ...] = (64, 64)) -> None:
        """Initialise the network.

        Args:
            obs_dim: Flattened observation dimensionality. Must be > 0.
            n_actions: Number of discrete actions (routes). Must be > 1.
            hidden_sizes: Widths of the hidden layers.

        Raises:
            ValueError: If obs_dim or n_actions are out of range.
        """
        super().__init__()
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {obs_dim}")
        if n_actions <= 1:
            raise ValueError(f"n_actions must be > 1, got {n_actions}")
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.layers = _orthogonal_mlp((obs_dim, *hidden_sizes, n_actions),
                                      out_gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for a batch of observations.

        Args:
            obs: Float tensor of shape (B, obs_dim) or (obs_dim,).

        Returns:
            Tensor of shape (B, n_actions) with Q(s, a) estimates.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.layers(obs)


def double_dqn_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_q_online: torch.Tensor,
    next_q_target: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute Double DQN TD targets for one minibatch.

    The online network chooses the next action, the target network
    evaluates it (van Hasselt et al., 2016) — decoupling selection from
    evaluation to remove the max-operator overestimation bias:

        y = r + gamma * (1 - done) * Q_target(s', argmax_a Q_online(s', a))

    Args:
        rewards: Shape (B,) rewards r_t.
        dones: Shape (B,) termination flags (1.0 cuts the bootstrap).
        next_q_online: Shape (B, A) online-network Q-values at s'.
        next_q_target: Shape (B, A) target-network Q-values at s'.
        gamma: Discount factor.

    Returns:
        Shape (B,) TD target values.
    """
    best_actions = next_q_online.argmax(dim=1, keepdim=True)
    next_values = next_q_target.gather(1, best_actions).squeeze(1)
    return rewards + gamma * (1.0 - dones) * next_values


@torch.no_grad()
def q_gaps(network: QNetwork, obs: torch.Tensor) -> tuple[float, float]:
    """Mean and max gap between the best and second-best Q-value.

    The Q-gap is the value margin the greedy policy acts on: a policy
    whose gap collapses to ~0 is indifferent between routes, one whose
    gap explodes on a single action everywhere has collapsed to a
    constant route.

    Args:
        network: The Q-network.
        obs: Float tensor of shape (N, obs_dim).

    Returns:
        Tuple (mean gap, max gap) over the observations.
    """
    top2 = network(obs).topk(2, dim=1).values
    gaps = top2[:, 0] - top2[:, 1]
    return float(gaps.mean().item()), float(gaps.max().item())


class DQNAgent:
    """Double DQN agent: online/target QNetwork pair plus optimiser."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        config: DQNConfig | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialise networks, optimiser, and the action RNG.

        Args:
            obs_dim: Flattened observation dimensionality.
            n_actions: Number of discrete actions.
            config: Hyperparameters; defaults to DQNConfig().
            device: Torch device string.
        """
        self.config = config or DQNConfig()
        self.device = torch.device(device)
        self.network = QNetwork(obs_dim, n_actions,
                                self.config.hidden_sizes).to(self.device)
        self.target = QNetwork(obs_dim, n_actions,
                               self.config.hidden_sizes).to(self.device)
        self.sync_target()
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=self.config.learning_rate)
        self._rng = np.random.default_rng()

    def seed(self, seed: int) -> None:
        """Seed the epsilon-greedy action RNG.

        Args:
            seed: Seed value.
        """
        self._rng = np.random.default_rng(seed)

    def sync_target(self) -> None:
        """Hard-copy the online network's weights into the target network."""
        self.target.load_state_dict(self.network.state_dict())

    @torch.no_grad()
    def act_greedy(self, obs: np.ndarray) -> int:
        """Return the argmax-Q action (deterministic policy for evaluation).

        Args:
            obs: Observation of shape (obs_dim,).

        Returns:
            Index of the highest-valued action.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return int(self.network(obs_t).argmax(dim=-1).item())

    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy action selection.

        Args:
            obs: Observation of shape (obs_dim,).
            epsilon: Probability of a uniform random action.

        Returns:
            The chosen action.
        """
        if self._rng.random() < epsilon:
            return int(self._rng.integers(self.network.n_actions))
        return self.act_greedy(obs)

    def train_step(self, batch: dict[str, torch.Tensor]) -> float:
        """One Double DQN gradient step on a replay minibatch.

        Huber (smooth L1) loss between Q(s, a) and the Double DQN
        target keeps single large TD errors — routine early on, when a
        BC-initialised value surface is still on the wrong scale — from
        dominating the gradient.

        Args:
            batch: Minibatch from :meth:`ReplayBuffer.sample`.

        Returns:
            The scalar TD (Huber) loss of this step.
        """
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        with torch.no_grad():
            next_obs = batch["next_obs"].to(self.device)
            targets = double_dqn_targets(
                batch["rewards"].to(self.device),
                batch["dones"].to(self.device),
                self.network(next_obs), self.target(next_obs),
                self.config.gamma)
        q_taken = self.network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q_taken, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(),
                                 self.config.max_grad_norm)
        self.optimizer.step()
        return float(loss.item())

    def load_bc_policy(self, path: str | Path, q_offset: float = 0.0) -> None:
        """Initialise the Q-network from a Phase 8 BC policy checkpoint.

        Copies the checkpointed ActorCritic's actor weights into the
        Q-network (identical trunk shape), so the greedy policy equals
        the BC policy's argmax, then adds ``q_offset`` to the final
        layer's bias — a constant shift of every Q(s, a), argmax-
        invariant — to move the O(1) logits onto the return scale. The
        target network is re-synced so the first TD targets bootstrap
        from the same initialisation.

        Args:
            path: Checkpoint file written by PPOAgent.save (Phase 8 bc_*).
            q_offset: Constant added to all Q-values (e.g. the BC
                policy's mean step reward / (1 - gamma)).

        Raises:
            ValueError: If the checkpoint's dimensions do not match.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        if (ckpt["obs_dim"] != self.network.obs_dim
                or ckpt["n_actions"] != self.network.n_actions):
            raise ValueError(
                f"BC checkpoint dims (obs={ckpt['obs_dim']}, "
                f"act={ckpt['n_actions']}) do not match agent "
                f"(obs={self.network.obs_dim}, act={self.network.n_actions})")
        actor_state = {key[len("actor."):]: value
                       for key, value in ckpt["network"].items()
                       if key.startswith("actor.")}
        self.network.layers.load_state_dict(actor_state)
        with torch.no_grad():
            final_linear = self.network.layers[-1]
            final_linear.bias.add_(q_offset)
        self.sync_target()

    def save(self, path: str | Path) -> None:
        """Save networks and optimiser state to a checkpoint file.

        Args:
            path: Destination file path (created/overwritten).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "network": self.network.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "obs_dim": self.network.obs_dim,
                "n_actions": self.network.n_actions,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """Restore networks and optimiser state from a checkpoint file.

        Args:
            path: Checkpoint file written by save().

        Raises:
            ValueError: If the checkpoint's dimensions do not match.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        if (ckpt["obs_dim"] != self.network.obs_dim
                or ckpt["n_actions"] != self.network.n_actions):
            raise ValueError(
                f"checkpoint dims (obs={ckpt['obs_dim']}, "
                f"act={ckpt['n_actions']}) do not match agent "
                f"(obs={self.network.obs_dim}, act={self.network.n_actions})")
        self.network.load_state_dict(ckpt["network"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])


# ---------------------------------------------------------------------------
# CLI train stage (owns an env and must run in a fresh process)
# ---------------------------------------------------------------------------


def run_train(args: argparse.Namespace) -> None:
    """Train Double DQN on the ns-3 env, logging the trajectory CSV."""
    from imitation import _make_env
    from train import set_global_seed

    set_global_seed(args.seed)

    env = _make_env(args)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)
    agent = DQNAgent(obs_dim, n_actions)
    agent.seed(args.seed)
    if args.bc_checkpoint:
        agent.load_bc_policy(Path(args.bc_checkpoint).resolve(),
                             q_offset=args.q_offset)
        print(f"[dqn] initialised Q-network from {args.bc_checkpoint} "
              f"(q_offset={args.q_offset})", flush=True)

    buffer = ReplayBuffer(agent.config.buffer_capacity, obs_dim,
                          seed=args.seed)

    trajectory_path = Path(args.trajectory_csv).resolve()
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trajectory_path, "w", newline="", encoding="utf-8") as fp:
        csv.writer(fp).writerow(TRAJECTORY_FIELDS)
    rewards_path = Path(args.rewards_csv).resolve() if args.rewards_csv else None
    if rewards_path is not None:
        rewards_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rewards_path, "w", newline="", encoding="utf-8") as fp:
            csv.writer(fp).writerow(("episode", "global_step", "reward"))

    checkpoint = Path(args.checkpoint).resolve()
    interval = args.trajectory_interval
    n_updates = args.total_steps // interval

    obs, _ = env.reset(seed=args.seed)
    obs = np.asarray(obs, dtype=np.float32)
    episode_reward, episodes_done = 0.0, 0
    window_obs: list[np.ndarray] = []
    window_actions: list[int] = []
    window_dones: list[float] = []
    window_losses: list[float] = []
    window_rewards: list[float] = []

    for step in range(1, args.total_steps + 1):
        epsilon = linear_epsilon(step - 1, args.eps_start, args.eps_end,
                                 args.eps_decay_steps)
        action = agent.select_action(obs, epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        buffer.add(obs, action, float(reward), next_obs, terminated)
        window_obs.append(obs)
        window_actions.append(action)
        window_dones.append(float(terminated or truncated))
        episode_reward += float(reward)
        if terminated or truncated:
            episodes_done += 1
            window_rewards.append(episode_reward)
            if rewards_path is not None:
                with open(rewards_path, "a", newline="",
                          encoding="utf-8") as fp:
                    csv.writer(fp).writerow(
                        (episodes_done, step, f"{episode_reward:.6g}"))
            episode_reward = 0.0
            next_obs, _ = env.reset()
            next_obs = np.asarray(next_obs, dtype=np.float32)
        obs = next_obs

        if len(buffer) >= agent.config.learning_starts:
            window_losses.append(
                agent.train_step(buffer.sample(agent.config.batch_size)))
        if step % agent.config.target_sync_interval == 0:
            agent.sync_target()

        if step % interval == 0:
            obs_t = torch.as_tensor(np.asarray(window_obs))
            dones_arr = np.asarray(window_dones)
            with torch.no_grad():
                greedy = agent.network(obs_t).argmax(dim=-1).cpu().numpy()
            per_200 = 200.0 / len(window_actions)
            mean_gap, max_gap = q_gaps(agent.network, obs_t)
            row = {
                "update": step // interval,
                "global_step": step,
                "epsilon": f"{epsilon:.4f}",
                "td_loss": (f"{np.mean(window_losses):.6g}"
                            if window_losses else ""),
                "mean_q_gap": f"{mean_gap:.6g}",
                "max_q_gap": f"{max_gap:.6g}",
                "sampled_switches_per_200":
                    f"{count_route_switches(np.asarray(window_actions), dones_arr) * per_200:.3f}",
                "greedy_switches_per_200":
                    f"{count_route_switches(greedy, dones_arr) * per_200:.3f}",
                "mean_episode_reward": (f"{np.mean(window_rewards):.6g}"
                                        if window_rewards else ""),
                "episodes_done": episodes_done,
            }
            with open(trajectory_path, "a", newline="", encoding="utf-8") as fp:
                csv.writer(fp).writerow([row[f] for f in TRAJECTORY_FIELDS])
            print(f"[dqn] update {step // interval}/{n_updates} eps={epsilon:.3f} "
                  f"td={row['td_loss']} q_gap={row['mean_q_gap']} "
                  f"greedy_sw/200={row['greedy_switches_per_200']}", flush=True)
            agent.save(checkpoint)
            window_obs.clear()
            window_actions.clear()
            window_dones.clear()
            window_losses.clear()
            window_rewards.clear()

    env.close()
    agent.save(checkpoint)
    print(f"[dqn] done: {args.total_steps} env steps, {episodes_done} episodes, "
          f"checkpoint {checkpoint}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the DQN CLI.

    Args:
        argv: Argument list; None uses sys.argv.

    Returns:
        The parsed namespace with a ``command`` attribute.
    """
    from imitation import _add_env_args

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train Double DQN on the ns-3 env")
    p_train.add_argument("--checkpoint", type=str, required=True)
    p_train.add_argument("--trajectory-csv", type=str, required=True)
    p_train.add_argument("--rewards-csv", type=str, default=None)
    p_train.add_argument("--bc-checkpoint", type=str, default=None,
                         help="Phase 8 BC checkpoint to initialise from")
    p_train.add_argument("--q-offset", type=float, default=0.0,
                         help="constant added to all initial Q-values "
                              "(BC init only; argmax-invariant)")
    p_train.add_argument("--total-steps", type=int, default=80_000)
    p_train.add_argument("--trajectory-interval", type=int, default=500)
    p_train.add_argument("--eps-start", type=float, default=1.0)
    p_train.add_argument("--eps-end", type=float, default=0.05)
    p_train.add_argument("--eps-decay-steps", type=int, default=40_000)
    p_train.add_argument("--seed", type=int, default=42,
                         help="global seed and first episode's run number")
    _add_env_args(p_train)

    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point: dispatch to the selected stage."""
    args = parse_args()
    {"train": run_train}[args.command](args)


if __name__ == "__main__":
    main()
