"""Scripted greedy-PER routing teacher (the Phase 7c winning baseline).

A memoryless-observation, hysteresis-state rule over the env's link
observation: hold the current route, switch to the route with the lowest
summed link PER when that beats the current route's cost by ``margin``.
Phase 7c measured it beating the best static route 8/10 shared-seed
episodes in both correlated cells of the adaptation study, which makes
it the natural imitation teacher for Phase 8.

The route tables of both topologies live here too (link indices into
the observation, install orders in sim/README.md) so the benchmark
orchestrator and the imitation machinery share one definition.

Typical usage:
    >>> teacher = GreedyPerTeacher(DISJOINT_ROUTE_LINKS)
    >>> action = teacher.act(obs)          # obs: flat Box(28) observation
"""

from __future__ import annotations

import numpy as np

# Candidate routes as link indices into the observation, per topology.
# Pentagon links (0,1) (1,2) (2,3) (3,4) (4,0) (0,2) (1,3) with routes
# 0: 0-2-3, 1: 0-1-3, 2: 0-4-3, 3: 0-1-2-3. Disjoint links (0,3) (0,1)
# (1,3) (0,2) (2,3) (0,4) (4,3) with routes 0: 0-3, 1: 0-1-3,
# 2: 0-2-3, 3: 0-4-3.
PENTAGON_ROUTE_LINKS: tuple[tuple[int, ...], ...] = ((5, 2), (0, 6), (4, 3),
                                                     (0, 1, 2))
DISJOINT_ROUTE_LINKS: tuple[tuple[int, ...], ...] = ((0,), (1, 2), (3, 4),
                                                     (5, 6))

# Hysteresis [summed PER]: at ~12 packets per 0.1 s step a sustained
# PER-sum improvement of 0.1 repays the flap penalty of 5 in about two
# steps (Phase 7c calibration).
DEFAULT_MARGIN = 0.1

# Observation layout: four features per link, linkPer at offset 1
# (sim/README.md).
LINK_FEATURES = 4
PER_FEATURE = 1


def held_route_from_obs(obs: np.ndarray, n_routes: int = 4) -> int:
    """Decode the current-route one-hot a route-aware observation ends with.

    The env's ``routeInObs`` flag (Phase 10) appends a one-hot of the
    currently held route to the observation; this reads it back.

    Args:
        obs: Flat route-aware observation (one-hot in the last
            ``n_routes`` entries).
        n_routes: Number of candidate routes.

    Returns:
        Index of the held route.

    Raises:
        ValueError: If the observation tail is not a valid one-hot.
    """
    tail = np.asarray(obs, dtype=np.float64)[-n_routes:]
    if len(tail) != n_routes or not (
        np.count_nonzero(tail == 1.0) == 1 and np.count_nonzero(tail) == 1
    ):
        raise ValueError(f"observation tail {tail} is not a route one-hot")
    return int(np.argmax(tail))


def route_links_for(topology: str) -> tuple[tuple[int, ...], ...]:
    """Return the route->links table of a topology.

    Args:
        topology: Mesh layout name ("pentagon" or "disjoint").

    Returns:
        Tuple of per-route link index tuples.

    Raises:
        ValueError: If the topology name is unknown.
    """
    tables = {"pentagon": PENTAGON_ROUTE_LINKS, "disjoint": DISJOINT_ROUTE_LINKS}
    if topology not in tables:
        raise ValueError(f"unknown topology: {topology!r}")
    return tables[topology]


class GreedyPerTeacher:
    """Hold-or-switch route selection on summed link PER with hysteresis.

    Deterministic given the observation sequence: the only state is the
    currently held route, updated whenever a strictly better route
    (by more than ``margin``) appears.

    Attributes:
        route_links: Per-route link indices into the observation.
        margin: Summed-PER improvement required to switch routes.
        current: Route currently held.
    """

    def __init__(
        self,
        route_links: tuple[tuple[int, ...], ...] = DISJOINT_ROUTE_LINKS,
        margin: float = DEFAULT_MARGIN,
        initial_route: int = 0,
    ) -> None:
        """Initialise the teacher.

        Args:
            route_links: Per-route link indices into the observation.
            margin: Summed-PER improvement required to switch routes.
            initial_route: Route held before the first observation (the
                env installs route 0 at episode start).
        """
        self.route_links = route_links
        self.margin = margin
        self._initial_route = initial_route
        self.current = initial_route

    def reset(self) -> None:
        """Forget the held route (call between independent episodes)."""
        self.current = self._initial_route

    def route_costs(self, obs: np.ndarray) -> np.ndarray:
        """Compute each route's summed link PER from one observation.

        Args:
            obs: Flat observation (LINK_FEATURES features per link; a
                trailing route one-hot from the env's route-aware mode
                is ignored — the teacher keeps its own held-route state).

        Returns:
            Array of shape (n_routes,) with each route's cost.
        """
        n_links = 1 + max(i for links in self.route_links for i in links)
        links_view = np.asarray(obs, dtype=np.float64)[
            : n_links * LINK_FEATURES
        ].reshape(-1, LINK_FEATURES)
        per = links_view[:, PER_FEATURE]
        return np.array([float(sum(per[i] for i in links))
                         for links in self.route_links])

    def act(self, obs: np.ndarray) -> int:
        """Choose the route for this step (updates the held route).

        Args:
            obs: Flat observation (LINK_FEATURES features per link).

        Returns:
            The route to install this step.
        """
        costs = self.route_costs(obs)
        best = int(np.argmin(costs))
        if costs[best] < costs[self.current] - self.margin:
            self.current = best
        return self.current
