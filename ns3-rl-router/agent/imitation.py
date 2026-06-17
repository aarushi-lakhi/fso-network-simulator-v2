"""Imitation-then-RL machinery for the Phase 8 experiment.

Behavior-clone the scripted greedy-PER teacher (agent/teacher.py), warm
up the critic on rollouts from the frozen BC policy, then fine-tune
with PPO — probing whether the on-policy gradient preserves a switching
policy it did not have to discover, or degrades it back to the
constant-route collapse Phase 7c measured.

Three stages, one CLI subcommand each. ``collect`` and ``finetune`` own
a gym env and must therefore run in their own process (ns3-ai allows a
single Experiment per process); ``bc`` is pure PyTorch.

* ``collect`` — roll the teacher on training seeds and save an
  (obs, action) dataset (.npz) plus per-episode teacher stats.
* ``bc`` — train the ActorCritic's actor head with cross-entropy on the
  dataset (episode-level validation split, so no episode straddles the
  split) and save a checkpoint in the standard PPOAgent format.
* ``finetune`` — stage 1 fits the value head only, on rollouts sampled
  from the frozen BC policy (a freshly-initialised critic produces
  garbage advantages that can destroy the BC policy in the first PPO
  updates — the Phase 8 decision-log hazard); stage 2 runs standard PPO
  from that warmed-up initialisation. Every update appends one row of
  entropy / KL-from-BC / switch-count diagnostics to a trajectory CSV —
  that trajectory is the experiment's result.

Typical usage (agent venv active, modules linked and built):
    $ python imitation.py collect --out ds.npz --episodes 25 --seed 42 \\
          --c2n 1e-13 --topology disjoint ...
    $ python imitation.py bc --dataset ds.npz --checkpoint bc.pt
    $ python imitation.py finetune --bc-checkpoint bc.pt --checkpoint out.pt \\
          --trajectory-csv traj.csv --warmup-steps 8000 --total-steps 80000 ...
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from network import ActorCritic
from ppo_agent import PPOAgent, RolloutBuffer

TRAJECTORY_FIELDS = ("phase", "update", "global_step", "entropy", "value_loss",
                     "policy_loss", "approx_kl", "kl_from_bc",
                     "sampled_switches_per_200", "greedy_switches_per_200",
                     "mean_episode_reward", "episodes_done")


# ---------------------------------------------------------------------------
# Behavior cloning (pure logic, hermetically testable)
# ---------------------------------------------------------------------------


@dataclass
class BCConfig:
    """Hyperparameters for behavior-cloning the teacher.

    Attributes:
        epochs: Passes over the training split.
        batch_size: Cross-entropy minibatch size.
        learning_rate: Adam step size (actor parameters only).
        val_fraction: Fraction of *episodes* held out for validation.
        seed: RNG seed for the split and minibatch shuffling.
    """

    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 1e-3
    val_fraction: float = 0.2
    seed: int = 0


@dataclass
class BCResult:
    """Learning curves of one behavior-cloning run.

    Attributes:
        train_losses: Mean training cross-entropy per epoch.
        val_losses: Validation cross-entropy per epoch.
        val_accuracies: Validation action-match accuracy per epoch.
    """

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)

    @property
    def val_accuracy(self) -> float:
        """Final validation accuracy (0.0 if never evaluated)."""
        return self.val_accuracies[-1] if self.val_accuracies else 0.0


def split_by_episode(
    episode_ids: np.ndarray, val_fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Split sample indices into train/validation by whole episodes.

    Consecutive steps of one episode are strongly correlated, so a
    step-level split would leak the validation set into training;
    holding out whole episodes keeps the accuracy estimate honest.

    Args:
        episode_ids: Shape (N,) episode index of every sample.
        val_fraction: Fraction of episodes held out (at least one).
        rng: Generator choosing which episodes are held out.

    Returns:
        Tuple (train_indices, validation_indices) into the samples.

    Raises:
        ValueError: If fewer than two distinct episodes are present.
    """
    episodes = np.unique(episode_ids)
    if len(episodes) < 2:
        raise ValueError("need at least two episodes to hold out a validation split")
    n_val = max(1, round(val_fraction * len(episodes)))
    val_episodes = rng.choice(episodes, size=n_val, replace=False)
    val_mask = np.isin(episode_ids, val_episodes)
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask)


@torch.no_grad()
def evaluate_actions(
    network: ActorCritic, obs: torch.Tensor, actions: torch.Tensor
) -> tuple[float, float]:
    """Compute cross-entropy loss and accuracy of the actor on a set.

    Args:
        network: The actor-critic network.
        obs: Float tensor of shape (N, obs_dim).
        actions: Long tensor of shape (N,) with teacher actions.

    Returns:
        Tuple (mean cross-entropy, action-match accuracy).
    """
    logits, _ = network(obs)
    loss = nn.functional.cross_entropy(logits, actions)
    accuracy = (logits.argmax(dim=-1) == actions).float().mean()
    return float(loss.item()), float(accuracy.item())


def train_bc(
    network: ActorCritic,
    obs: np.ndarray,
    actions: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    config: BCConfig | None = None,
) -> BCResult:
    """Behavior-clone teacher actions into the network's actor head.

    Only the actor parameters are optimised; the critic stays at its
    initialisation (it is fitted later, in the value-warmup stage of
    the fine-tune).

    Args:
        network: The actor-critic network to train in place.
        obs: Shape (N, obs_dim) observations.
        actions: Shape (N,) teacher actions.
        train_idx: Sample indices used for training.
        val_idx: Sample indices used for validation.
        config: Hyperparameters; defaults to BCConfig().

    Returns:
        BCResult with per-epoch losses and validation accuracy.
    """
    cfg = config or BCConfig()
    rng = np.random.default_rng(cfg.seed)
    optimizer = torch.optim.Adam(network.actor.parameters(), lr=cfg.learning_rate)

    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    act_t = torch.as_tensor(actions, dtype=torch.int64)
    val_obs, val_act = obs_t[val_idx], act_t[val_idx]

    result = BCResult()
    for _ in range(cfg.epochs):
        order = rng.permutation(len(train_idx))
        epoch_losses = []
        for start in range(0, len(order), cfg.batch_size):
            idx = train_idx[order[start:start + cfg.batch_size]]
            logits, _ = network(obs_t[idx])
            loss = nn.functional.cross_entropy(logits, act_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        val_loss, val_acc = evaluate_actions(network, val_obs, val_act)
        result.train_losses.append(float(np.mean(epoch_losses)))
        result.val_losses.append(val_loss)
        result.val_accuracies.append(val_acc)
    return result


# ---------------------------------------------------------------------------
# Fine-tune diagnostics and value warmup (pure logic)
# ---------------------------------------------------------------------------


def count_route_switches(
    actions: np.ndarray, dones: np.ndarray, initial_route: int = 0
) -> int:
    """Count within-episode route changes along an action sequence.

    Matches the benchmark's eval-time metric: each episode starts on
    ``initial_route`` (the env installs route 0 on reset), and every
    step whose action differs from the previous step's route counts as
    one switch. A ``done`` flag ends the episode; the next step starts
    a fresh one.

    Args:
        actions: Shape (T,) actions taken.
        dones: Shape (T,) episode-termination flags (0.0 or 1.0).
        initial_route: Route installed at every episode start.

    Returns:
        Total number of switches across all (possibly partial) episodes.
    """
    switches = 0
    route = initial_route
    for action, done in zip(actions, dones):
        if int(action) != route:
            switches += 1
            route = int(action)
        if done:
            route = initial_route
    return switches


@torch.no_grad()
def mean_kl_from_reference(
    reference: ActorCritic, network: ActorCritic, obs: torch.Tensor
) -> float:
    """Mean KL(reference || network) of the policies over observations.

    Args:
        reference: The frozen BC policy.
        network: The current policy.
        obs: Float tensor of shape (N, obs_dim).

    Returns:
        Mean KL divergence in nats.
    """
    kl = torch.distributions.kl_divergence(
        reference.action_distribution(obs), network.action_distribution(obs)
    )
    return float(kl.mean().item())


@torch.no_grad()
def mean_entropy(network: ActorCritic, obs: torch.Tensor) -> float:
    """Mean policy entropy over observations [nats].

    Args:
        network: The current policy.
        obs: Float tensor of shape (N, obs_dim).

    Returns:
        Mean categorical entropy.
    """
    return float(network.action_distribution(obs).entropy().mean().item())


@torch.no_grad()
def greedy_actions(network: ActorCritic, obs: torch.Tensor) -> np.ndarray:
    """Argmax actions of the policy on a batch of observations.

    Args:
        network: The current policy.
        obs: Float tensor of shape (N, obs_dim).

    Returns:
        Shape (N,) int64 array of greedy actions.
    """
    logits, _ = network(obs)
    return logits.argmax(dim=-1).cpu().numpy()


def fit_value(
    agent: PPOAgent,
    buffer: RolloutBuffer,
    last_value: float,
    optimizer: torch.optim.Optimizer,
    n_epochs: int = 4,
    minibatch_size: int = 64,
) -> float:
    """Fit the critic on one rollout, leaving the actor untouched.

    Targets are Monte-Carlo style returns (GAE with λ=1), so they do
    not depend on the critic's own — initially garbage — intermediate
    value estimates; only the bootstrap tail uses ``last_value``.
    Resets the buffer when done, like ``PPOAgent.update``.

    Args:
        agent: Agent whose critic is optimised (actor frozen by simply
            not optimising it — the optimizer only holds critic params).
        buffer: Filled rollout buffer.
        last_value: Bootstrap value for the state after the rollout.
        optimizer: Optimizer over the critic parameters only.
        n_epochs: Optimisation epochs over the rollout.
        minibatch_size: Samples per minibatch.

    Returns:
        Mean value loss over all minibatches.
    """
    buffer.finalize(last_value, agent.config.gamma, gae_lambda=1.0)
    losses = []
    for _ in range(n_epochs):
        for batch in buffer.minibatches(minibatch_size, agent._rng):
            _, values = agent.network(batch["obs"].to(agent.device))
            loss = nn.functional.mse_loss(values, batch["returns"].to(agent.device))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.network.critic.parameters(),
                                     agent.config.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))
    buffer.reset()
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# CLI stages (collect/finetune own an env and must run in a fresh process)
# ---------------------------------------------------------------------------


def _make_env(args: argparse.Namespace):
    """Build the ns-3 env from the shared CLI env-override arguments."""
    from ns3_env import DEFAULT_CONFIG_PATH, make_ns3_env

    sim_config = str(Path(args.sim_config or DEFAULT_CONFIG_PATH).resolve())
    return make_ns3_env(sim_config, c2n=args.c2n, seed=args.seed,
                        coherence_large=args.coherence_large,
                        coherence_small=args.coherence_small,
                        step_time_s=args.step_time,
                        episode_steps=args.episode_steps,
                        topology=args.topology,
                        traffic_protocol=args.traffic_protocol)


def run_collect(args: argparse.Namespace) -> None:
    """Roll the teacher on training seeds and save the BC dataset."""
    from teacher import DEFAULT_MARGIN, GreedyPerTeacher, route_links_for

    env = _make_env(args)
    route_links = route_links_for(args.topology or "pentagon")
    all_obs: list[np.ndarray] = []
    all_actions: list[int] = []
    episode_ids: list[int] = []
    episode_rewards: list[float] = []
    episode_switches: list[int] = []
    try:
        for ep in range(args.episodes):
            teacher = GreedyPerTeacher(route_links, margin=DEFAULT_MARGIN)
            obs, _ = env.reset(seed=args.seed + ep)
            obs = np.asarray(obs, dtype=np.float32)
            reward_total, switches, route, done = 0.0, 0, 0, False
            while not done:
                action = teacher.act(obs)
                all_obs.append(obs)
                all_actions.append(action)
                episode_ids.append(ep)
                if action != route:
                    switches += 1
                    route = action
                obs, reward, terminated, truncated, _ = env.step(action)
                obs = np.asarray(obs, dtype=np.float32)
                reward_total += float(reward)
                done = terminated or truncated
            episode_rewards.append(reward_total)
            episode_switches.append(switches)
            print(f"[collect] ep{ep} (simSeed={args.seed + ep}): "
                  f"reward={reward_total:.1f} switches={switches}", flush=True)
    finally:
        env.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        obs=np.asarray(all_obs, dtype=np.float32),
        actions=np.asarray(all_actions, dtype=np.int64),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        episode_rewards=np.asarray(episode_rewards, dtype=np.float64),
        episode_switches=np.asarray(episode_switches, dtype=np.int64),
        seeds=np.arange(args.seed, args.seed + args.episodes),
    )
    print(f"[collect] wrote {len(all_actions)} (obs, action) pairs from "
          f"{args.episodes} episodes to {out}")
    print(f"[collect] teacher reward {np.mean(episode_rewards):.1f} +/- "
          f"{np.std(episode_rewards):.1f}, switches/ep "
          f"{np.mean(episode_switches):.1f}")


def run_bc(args: argparse.Namespace) -> None:
    """Train the BC policy from a collected dataset and save a checkpoint."""
    data = np.load(args.dataset)
    obs, actions = data["obs"], data["actions"]
    episode_ids = data["episode_ids"]

    torch.manual_seed(args.bc_seed)
    rng = np.random.default_rng(args.bc_seed)
    train_idx, val_idx = split_by_episode(episode_ids, args.val_fraction, rng)

    agent = PPOAgent(obs_dim=obs.shape[1], n_actions=args.n_actions)
    config = BCConfig(epochs=args.epochs, learning_rate=args.learning_rate,
                      val_fraction=args.val_fraction, seed=args.bc_seed)
    result = train_bc(agent.network, obs, actions, train_idx, val_idx, config)
    agent.save(args.checkpoint)

    if args.metrics_csv:
        path = Path(args.metrics_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(("epoch", "train_loss", "val_loss", "val_accuracy"))
            for i, (tl, vl, va) in enumerate(zip(result.train_losses,
                                                 result.val_losses,
                                                 result.val_accuracies), 1):
                writer.writerow((i, f"{tl:.6g}", f"{vl:.6g}", f"{va:.6g}"))

    majority = float(np.bincount(actions[val_idx]).max() / len(val_idx))
    print(f"[bc] {len(train_idx)} train / {len(val_idx)} val samples "
          f"({len(np.unique(episode_ids))} episodes)")
    print(f"[bc] final train loss {result.train_losses[-1]:.4f}, "
          f"val loss {result.val_losses[-1]:.4f}")
    print(f"[bc] VAL-ACCURACY {result.val_accuracy:.4f} "
          f"(majority-class baseline {majority:.4f})")
    print(f"[bc] saved checkpoint to {args.checkpoint}")


def run_finetune(args: argparse.Namespace) -> None:
    """Warm up the critic, then PPO fine-tune from the BC initialisation."""
    from train import set_global_seed

    set_global_seed(args.seed)

    env = _make_env(args)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)
    agent = PPOAgent(obs_dim, n_actions)
    agent.load(Path(args.bc_checkpoint).resolve())
    agent.seed(args.seed)

    bc_reference = copy.deepcopy(agent.network)
    bc_reference.eval()

    buffer = RolloutBuffer(args.rollout_steps, obs_dim)
    critic_opt = torch.optim.Adam(agent.network.critic.parameters(),
                                  lr=agent.config.learning_rate)

    trajectory_path = Path(args.trajectory_csv).resolve()
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trajectory_path, "w", newline="", encoding="utf-8") as fp:
        csv.writer(fp).writerow(TRAJECTORY_FIELDS)
    rewards_path = Path(args.rewards_csv).resolve() if args.rewards_csv else None
    if rewards_path is not None:
        rewards_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rewards_path, "w", newline="", encoding="utf-8") as fp:
            csv.writer(fp).writerow(("episode", "global_step", "phase", "reward"))

    n_warmup = args.warmup_steps // args.rollout_steps
    n_ppo = args.total_steps // args.rollout_steps
    checkpoint = Path(args.checkpoint).resolve()

    obs, _ = env.reset(seed=args.seed)
    obs = np.asarray(obs, dtype=np.float32)
    episode_reward, episodes_done, global_step = 0.0, 0, 0

    def log_row(row: dict) -> None:
        with open(trajectory_path, "a", newline="", encoding="utf-8") as fp:
            csv.writer(fp).writerow(
                ["" if row.get(f) is None else row[f] for f in TRAJECTORY_FIELDS])

    for update in range(1, n_warmup + n_ppo + 1):
        phase = "warmup" if update <= n_warmup else "ppo"
        window_rewards: list[float] = []
        terminated = False
        for _ in range(args.rollout_steps):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            buffer.add(obs, action, log_prob, reward, terminated, value)
            episode_reward += float(reward)
            global_step += 1
            if terminated or truncated:
                episodes_done += 1
                window_rewards.append(episode_reward)
                if rewards_path is not None:
                    with open(rewards_path, "a", newline="",
                              encoding="utf-8") as fp:
                        csv.writer(fp).writerow(
                            (episodes_done, global_step, phase,
                             f"{episode_reward:.6g}"))
                episode_reward = 0.0
                next_obs, _ = env.reset()
                next_obs = np.asarray(next_obs, dtype=np.float32)
            obs = next_obs

        # Snapshot the rollout before the update consumes the buffer
        rollout_obs = torch.as_tensor(buffer.obs[:buffer.pos].copy())
        rollout_actions = buffer.actions[:buffer.pos].copy()
        rollout_dones = buffer.dones[:buffer.pos].copy()
        last_value = 0.0 if terminated else agent.predict_value(obs)

        if phase == "warmup":
            value_loss = fit_value(agent, buffer, last_value, critic_opt,
                                   agent.config.n_epochs,
                                   agent.config.minibatch_size)
            policy_loss = approx_kl = None
        else:
            metrics = agent.update(buffer, last_value)
            value_loss = metrics["value_loss"]
            policy_loss = f"{metrics['policy_loss']:.6g}"
            approx_kl = f"{metrics['approx_kl']:.6g}"

        per_200 = 200.0 / len(rollout_actions)
        sampled_switches = count_route_switches(rollout_actions, rollout_dones)
        greedy = greedy_actions(agent.network, rollout_obs)
        greedy_switches = count_route_switches(greedy, rollout_dones)
        kl_from_bc = mean_kl_from_reference(bc_reference, agent.network,
                                            rollout_obs)
        row = {
            "phase": phase,
            "update": update,
            "global_step": global_step,
            "entropy": f"{mean_entropy(agent.network, rollout_obs):.6g}",
            "value_loss": f"{value_loss:.6g}",
            "policy_loss": policy_loss,
            "approx_kl": approx_kl,
            "kl_from_bc": f"{kl_from_bc:.6g}",
            "sampled_switches_per_200": f"{sampled_switches * per_200:.3f}",
            "greedy_switches_per_200": f"{greedy_switches * per_200:.3f}",
            "mean_episode_reward": (f"{np.mean(window_rewards):.6g}"
                                    if window_rewards else None),
            "episodes_done": episodes_done,
        }
        log_row(row)
        print(f"[finetune:{phase}] update {update}/{n_warmup + n_ppo} "
              f"entropy={row['entropy']} kl_bc={row['kl_from_bc']} "
              f"greedy_sw/200={row['greedy_switches_per_200']} "
              f"vloss={row['value_loss']}", flush=True)
        agent.save(checkpoint)

    env.close()
    print(f"[finetune] done: {global_step} env steps "
          f"({n_warmup} warmup + {n_ppo} PPO updates), checkpoint {checkpoint}")


def _add_env_args(parser: argparse.ArgumentParser) -> None:
    """Register the shared ns-3 env-override arguments."""
    parser.add_argument("--sim-config", type=str, default=None)
    parser.add_argument("--c2n", type=str, default=None)
    parser.add_argument("--coherence-large", type=str, default=None)
    parser.add_argument("--coherence-small", type=str, default=None)
    parser.add_argument("--step-time", type=str, default=None)
    parser.add_argument("--episode-steps", type=str, default=None)
    parser.add_argument("--topology", type=str, default=None,
                        choices=("pentagon", "disjoint"))
    parser.add_argument("--traffic-protocol", type=str, default=None,
                        choices=("udp", "tcp"), dest="traffic_protocol")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the imitation CLI.

    Args:
        argv: Argument list; None uses sys.argv.

    Returns:
        The parsed namespace with a ``command`` attribute.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser("collect", help="roll the teacher, save a dataset")
    p_collect.add_argument("--out", type=str, required=True)
    p_collect.add_argument("--episodes", type=int, default=25)
    p_collect.add_argument("--seed", type=int, default=42,
                           help="first episode's ns-3 run number")
    _add_env_args(p_collect)

    p_bc = sub.add_parser("bc", help="behavior-clone the dataset")
    p_bc.add_argument("--dataset", type=str, required=True)
    p_bc.add_argument("--checkpoint", type=str, required=True)
    p_bc.add_argument("--metrics-csv", type=str, default=None)
    p_bc.add_argument("--epochs", type=int, default=40)
    p_bc.add_argument("--learning-rate", type=float, default=1e-3)
    p_bc.add_argument("--val-fraction", type=float, default=0.2)
    p_bc.add_argument("--bc-seed", type=int, default=0)
    p_bc.add_argument("--n-actions", type=int, default=4)

    p_ft = sub.add_parser("finetune", help="value warmup + PPO fine-tune")
    p_ft.add_argument("--bc-checkpoint", type=str, required=True)
    p_ft.add_argument("--checkpoint", type=str, required=True)
    p_ft.add_argument("--trajectory-csv", type=str, required=True)
    p_ft.add_argument("--rewards-csv", type=str, default=None)
    p_ft.add_argument("--warmup-steps", type=int, default=8_000)
    p_ft.add_argument("--total-steps", type=int, default=80_000)
    p_ft.add_argument("--rollout-steps", type=int, default=500)
    p_ft.add_argument("--seed", type=int, default=42,
                      help="global seed and first episode's run number")
    _add_env_args(p_ft)

    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point: dispatch to the selected stage."""
    args = parse_args()
    {"collect": run_collect, "bc": run_bc, "finetune": run_finetune}[args.command](args)


if __name__ == "__main__":
    main()
