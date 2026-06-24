"""Hermetic tests for the greedy-PER teacher (no ns-3 needed)."""

from __future__ import annotations

import numpy as np

from teacher import (
    DEFAULT_MARGIN,
    DISJOINT_ROUTE_LINKS,
    LINK_FEATURES,
    PENTAGON_ROUTE_LINKS,
    PER_FEATURE,
    GreedyPerTeacher,
    held_route_from_obs,
    route_links_for,
)

N_LINKS = 7


def obs_with_per(per: list[float]) -> np.ndarray:
    """Build a flat observation whose linkPer slots hold ``per``."""
    obs = np.zeros((N_LINKS, LINK_FEATURES), dtype=np.float64)
    obs[:, PER_FEATURE] = per
    return obs.reshape(-1)


def route_aware(obs: np.ndarray, route: int, n_routes: int = 4) -> np.ndarray:
    """Append a route one-hot to a flat observation (the env's Phase 10 tail)."""
    onehot = np.zeros(n_routes, dtype=np.float64)
    onehot[route] = 1.0
    return np.concatenate([obs, onehot])


class TestRouteTables:
    def test_route_links_for_selects_topology(self):
        assert route_links_for("pentagon") is PENTAGON_ROUTE_LINKS
        assert route_links_for("disjoint") is DISJOINT_ROUTE_LINKS

    def test_route_links_for_rejects_unknown(self):
        import pytest

        with pytest.raises(ValueError, match="unknown topology"):
            route_links_for("ring")

    def test_disjoint_routes_share_no_links(self):
        seen: set[int] = set()
        for links in DISJOINT_ROUTE_LINKS:
            assert seen.isdisjoint(links)
            seen.update(links)


class TestGreedyPerTeacher:
    def test_route_costs_sum_link_per(self):
        teacher = GreedyPerTeacher(DISJOINT_ROUTE_LINKS)
        obs = obs_with_per([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        costs = teacher.route_costs(obs)
        np.testing.assert_allclose(costs, [0.1, 0.5, 0.9, 1.3])

    def test_holds_route_within_margin(self):
        teacher = GreedyPerTeacher(DISJOINT_ROUTE_LINKS, margin=DEFAULT_MARGIN)
        # Route 1 (cost 0.05) beats route 0 (cost 0.1) by less than 0.1
        obs = obs_with_per([0.1, 0.05, 0.0, 0.5, 0.5, 0.5, 0.5])
        assert teacher.act(obs) == 0

    def test_switches_beyond_margin(self):
        teacher = GreedyPerTeacher(DISJOINT_ROUTE_LINKS, margin=DEFAULT_MARGIN)
        obs = obs_with_per([0.5, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5])
        assert teacher.act(obs) == 1
        assert teacher.current == 1

    def test_hysteresis_prevents_flapping(self):
        teacher = GreedyPerTeacher(DISJOINT_ROUTE_LINKS, margin=DEFAULT_MARGIN)
        low = obs_with_per([0.5, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5])
        assert teacher.act(low) == 1
        # Route 0 recovers to just under route 1's cost: hold route 1
        near = obs_with_per([0.15, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5])
        assert teacher.act(near) == 1
        # Route 0 clearly better: switch back
        clear = obs_with_per([0.05, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5])
        assert teacher.act(clear) == 0

    def test_deterministic_on_identical_sequences(self):
        rng = np.random.default_rng(7)
        seq = [obs_with_per(list(rng.uniform(0, 1, N_LINKS))) for _ in range(50)]
        a = GreedyPerTeacher(DISJOINT_ROUTE_LINKS)
        b = GreedyPerTeacher(DISJOINT_ROUTE_LINKS)
        assert [a.act(o) for o in seq] == [b.act(o) for o in seq]

    def test_reset_restores_initial_route(self):
        teacher = GreedyPerTeacher(DISJOINT_ROUTE_LINKS)
        teacher.act(obs_with_per([0.9, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]))
        assert teacher.current == 1
        teacher.reset()
        assert teacher.current == 0

    def test_zero_margin_tracks_argmin(self):
        teacher = GreedyPerTeacher(DISJOINT_ROUTE_LINKS, margin=0.0)
        rng = np.random.default_rng(11)
        for _ in range(20):
            obs = obs_with_per(list(rng.uniform(0.01, 1, N_LINKS)))
            action = teacher.act(obs)
            assert action == int(np.argmin(teacher.route_costs(obs)))

    def test_route_costs_ignore_appended_onehot(self):
        teacher = GreedyPerTeacher(DISJOINT_ROUTE_LINKS)
        obs = obs_with_per([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        for route in range(4):
            np.testing.assert_array_equal(
                teacher.route_costs(route_aware(obs, route)),
                teacher.route_costs(obs))

    def test_actions_identical_on_route_aware_obs(self):
        rng = np.random.default_rng(3)
        plain = GreedyPerTeacher(DISJOINT_ROUTE_LINKS)
        aware = GreedyPerTeacher(DISJOINT_ROUTE_LINKS)
        for _ in range(50):
            obs = obs_with_per(list(rng.uniform(0, 1, N_LINKS)))
            expected = plain.act(obs)
            assert aware.act(route_aware(obs, aware.current)) == expected


class TestHeldRouteFromObs:
    def test_decodes_each_route(self):
        obs = obs_with_per([0.5] * N_LINKS)
        for route in range(4):
            assert held_route_from_obs(route_aware(obs, route)) == route

    def test_rejects_all_zero_tail(self):
        import pytest

        obs = np.concatenate([obs_with_per([0.0] * N_LINKS), np.zeros(4)])
        with pytest.raises(ValueError, match="not a route one-hot"):
            held_route_from_obs(obs)

    def test_rejects_multi_hot_tail(self):
        import pytest

        obs = np.concatenate([obs_with_per([0.0] * N_LINKS),
                              np.array([1.0, 0.0, 1.0, 0.0])])
        with pytest.raises(ValueError, match="not a route one-hot"):
            held_route_from_obs(obs)

    def test_rejects_fractional_tail(self):
        import pytest

        obs = np.concatenate([obs_with_per([0.0] * N_LINKS),
                              np.array([0.0, 0.5, 0.5, 0.0])])
        with pytest.raises(ValueError, match="not a route one-hot"):
            held_route_from_obs(obs)
