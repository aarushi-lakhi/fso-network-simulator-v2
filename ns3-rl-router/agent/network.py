"""
Actor-Critic network for the FSO routing agent.

Maps a flattened link-state observation (per-link SNR, drop rate,
scintillation index, queue depth) to:
    - Policy logits over discrete next-hop / route choices (actor head)
    - A scalar state-value estimate (critic head)

Actor and critic use separate MLP trunks. For the small observation
vectors produced by the FSO mesh this costs little compute and avoids
the gradient interference a shared trunk can introduce between the
policy and value objectives.

Typical usage:
    >>> net = ActorCritic(obs_dim=20, n_actions=5)
    >>> logits, value = net(torch.randn(8, 20))
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _orthogonal_mlp(sizes: tuple[int, ...], out_gain: float) -> nn.Sequential:
    """Build a Tanh MLP with orthogonally initialised linear layers.

    Orthogonal initialisation with a small final-layer gain is the
    standard PPO recipe (Engstrom et al., 2020): it keeps early policy
    outputs near-uniform and value estimates near zero.

    Args:
        sizes: Layer widths including input and output, e.g. (20, 64, 64, 5).
        out_gain: Initialisation gain for the final linear layer.

    Returns:
        nn.Sequential of Linear/Tanh layers ending in a Linear layer.
    """
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        linear = nn.Linear(sizes[i], sizes[i + 1])
        is_last = i == len(sizes) - 2
        nn.init.orthogonal_(linear.weight, gain=out_gain if is_last else 2.0**0.5)
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if not is_last:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Separate-trunk actor-critic MLP for discrete route selection.

    Attributes:
        actor: MLP producing unnormalised policy logits, shape (B, n_actions).
        critic: MLP producing state values, shape (B, 1).
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
    ) -> None:
        """Initialise the network.

        Args:
            obs_dim: Flattened observation dimensionality. Must be > 0.
            n_actions: Number of discrete actions (routes). Must be > 1.
            hidden_sizes: Widths of the hidden layers in each trunk.

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
        self.actor = _orthogonal_mlp((obs_dim, *hidden_sizes, n_actions), out_gain=0.01)
        self.critic = _orthogonal_mlp((obs_dim, *hidden_sizes, 1), out_gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute policy logits and state value for a batch of observations.

        Args:
            obs: Float tensor of shape (B, obs_dim) or (obs_dim,).

        Returns:
            Tuple (logits, value):
                logits: shape (B, n_actions) — unnormalised action preferences.
                value: shape (B,) — state-value estimates.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.actor(obs), self.critic(obs).squeeze(-1)

    def action_distribution(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        """Return the categorical policy distribution π(·|obs).

        Args:
            obs: Float tensor of shape (B, obs_dim) or (obs_dim,).

        Returns:
            torch.distributions.Categorical over the n_actions routes.
        """
        logits, _ = self.forward(obs)
        return torch.distributions.Categorical(logits=logits)
