"""
Tests for the temporally correlated Gamma-Gamma fading sampler (Phase 6a).

Test strategy:
    - Unit tests: AR(1) coefficient helper against the closed form exp(−Δt/τ)
    - Marginal preservation: the Gaussian copula AR(1) construction must leave
      the stationary marginal EXACTLY Gamma-Gamma, so E[I] = 1 and the SI
      identity SI = 1/α + 1/β + 1/(αβ) must hold on a long correlated series
    - Memoryless limit: τ = 0 must statistically match the i.i.d. sampler
    - Temporal structure: lag-k autocorrelation decays with k and increases
      with τ (qualitative only — the nonlinear copula transform means the
      autocorrelation of I is NOT exactly exp(−kΔt/τ))
    - Reproducibility and input validation

Serial correlation inflates the variance of moment estimators, so the
correlated-series tests use longer chains (10⁶ samples) than the i.i.d.
tests. Measured SI relative error at 10⁶ samples is < 1% across seeds for
the parameters below; tolerances are set with generous margin above that.
"""

from __future__ import annotations

import numpy as np
import pytest

from gamma_gamma import (
    C2N_STRONG,
    alpha_beta_from_rytov,
    ar1_coefficient,
    correlated_gamma_gamma_sample,
    empirical_autocorrelation,
    empirical_scintillation_index,
    gamma_gamma_sample,
    rytov_variance,
    scintillation_index,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

#: Sample interval used throughout: 1 ms (the Phase 5 decision-step scale).
DT = 1e-3

#: Coherence times spanning the physically interesting range.
TAU_LARGE = 100e-3   # large eddies: slow, ~100 ms
TAU_SMALL = 10e-3    # small eddies: faster, ~10 ms


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded RNG for reproducible statistical tests."""
    return np.random.default_rng(seed=0)


@pytest.fixture
def strong_alpha_beta() -> tuple[float, float]:
    """(α, β) for the strong-turbulence reference case (C²ₙ=10⁻¹³, 1 km, 1550 nm).

    Gives α ≈ 3.99, β ≈ 1.71, SI ≈ 0.98 — the case the ns-3 implementation
    validates against.
    """
    sigma2_R = rytov_variance(C2N_STRONG, 1550e-9, 1000.0)
    return alpha_beta_from_rytov(sigma2_R)


# ---------------------------------------------------------------------------
# AR(1) coefficient helper
# ---------------------------------------------------------------------------


class TestAr1Coefficient:
    """ρ = exp(−Δt/τ) with the τ = 0 memoryless convention."""

    def test_matches_closed_form(self) -> None:
        assert abs(ar1_coefficient(1e-3, 10e-3) - np.exp(-0.1)) < 1e-12
        assert abs(ar1_coefficient(1e-3, 100e-3) - np.exp(-0.01)) < 1e-12

    def test_tau_zero_gives_zero(self) -> None:
        """τ = 0 is the memoryless limit: ρ = 0 exactly."""
        assert ar1_coefficient(1e-3, 0.0) == 0.0

    def test_bounded_in_unit_interval(self) -> None:
        for tau in [0.0, 1e-6, 1e-3, 1.0, 1e3]:
            rho = ar1_coefficient(1e-3, tau)
            assert 0.0 <= rho < 1.0

    def test_increases_with_tau(self) -> None:
        """Longer coherence time → more memory → larger ρ."""
        rhos = [ar1_coefficient(DT, tau) for tau in [1e-3, 10e-3, 100e-3]]
        assert rhos[0] < rhos[1] < rhos[2]

    def test_invalid_dt_raises(self) -> None:
        with pytest.raises(ValueError, match="dt"):
            ar1_coefficient(0.0, 10e-3)
        with pytest.raises(ValueError, match="dt"):
            ar1_coefficient(-1e-3, 10e-3)

    def test_negative_tau_raises(self) -> None:
        with pytest.raises(ValueError, match="tau"):
            ar1_coefficient(1e-3, -10e-3)


# ---------------------------------------------------------------------------
# Marginal preservation
# ---------------------------------------------------------------------------


class TestMarginalPreserved:
    """The copula construction must not change the stationary marginal."""

    N_LONG = 1_000_000  # serial correlation inflates estimator variance

    def test_mean_is_one_strong_turbulence(
        self, rng: np.random.Generator, strong_alpha_beta: tuple[float, float]
    ) -> None:
        """E[I] = 1 must survive the correlation (normalised irradiance)."""
        alpha, beta = strong_alpha_beta
        series = correlated_gamma_gamma_sample(
            alpha, beta, self.N_LONG, DT, TAU_LARGE, TAU_SMALL, rng
        )
        assert abs(series.mean() - 1.0) < 0.02, (
            f"Expected E[I] ≈ 1, got {series.mean():.4f}"
        )

    def test_scintillation_index_identity(
        self, rng: np.random.Generator, strong_alpha_beta: tuple[float, float]
    ) -> None:
        """KEY TEST: empirical SI of a long correlated series must match
        SI = 1/α + 1/β + 1/(αβ) — the exact-marginal property of the copula.

        Measured relative error at 10⁶ samples is < 0.5% across seeds; the 5%
        tolerance leaves a wide margin for the correlation-inflated variance.
        """
        cases = [strong_alpha_beta, (4.0, 2.0)]
        for alpha, beta in cases:
            series = correlated_gamma_gamma_sample(
                alpha, beta, self.N_LONG, DT, TAU_LARGE, TAU_SMALL, rng
            )
            si_theoretical = scintillation_index(alpha, beta)
            si_empirical = empirical_scintillation_index(series)
            rel_error = abs(si_empirical - si_theoretical) / si_theoretical
            assert rel_error < 0.05, (
                f"α={alpha}, β={beta}: SI_theoretical={si_theoretical:.4f}, "
                f"SI_empirical={si_empirical:.4f}, relative_error={rel_error:.2%}"
            )

    def test_all_positive(self, rng: np.random.Generator) -> None:
        """Irradiance is always strictly positive (no negative power)."""
        series = correlated_gamma_gamma_sample(
            3.0, 3.0, 10_000, DT, TAU_LARGE, TAU_SMALL, rng
        )
        assert np.all(series > 0)


# ---------------------------------------------------------------------------
# Memoryless limit (τ → 0)
# ---------------------------------------------------------------------------


class TestMemorylessLimit:
    """τ_large = τ_small = 0 must recover i.i.d. Gamma-Gamma behaviour."""

    N = 500_000

    def test_si_matches_iid_sampler(
        self, rng: np.random.Generator, strong_alpha_beta: tuple[float, float]
    ) -> None:
        """With ρ = 0, SI of the correlated sampler matches the i.i.d. one."""
        alpha, beta = strong_alpha_beta
        correlated = correlated_gamma_gamma_sample(
            alpha, beta, self.N, DT, tau_large=0.0, tau_small=0.0, rng=rng
        )
        iid = gamma_gamma_sample(alpha, beta, self.N, rng)

        si_corr = empirical_scintillation_index(correlated)
        si_iid = empirical_scintillation_index(iid)
        assert abs(si_corr - si_iid) / si_iid < 0.05, (
            f"τ=0 SI ({si_corr:.4f}) should match i.i.d. SI ({si_iid:.4f})"
        )

    def test_lag1_autocorrelation_is_zero(
        self, rng: np.random.Generator, strong_alpha_beta: tuple[float, float]
    ) -> None:
        """With ρ = 0 successive samples are independent: r₁ ≈ 0."""
        alpha, beta = strong_alpha_beta
        series = correlated_gamma_gamma_sample(
            alpha, beta, self.N, DT, tau_large=0.0, tau_small=0.0, rng=rng
        )
        r1 = empirical_autocorrelation(series, lag=1)
        assert abs(r1) < 0.02, f"Expected r₁ ≈ 0 for τ = 0, got {r1:.4f}"


# ---------------------------------------------------------------------------
# Temporal structure
# ---------------------------------------------------------------------------


class TestTemporalStructure:
    """Autocorrelation must decay with lag and increase with coherence time.

    Qualitative assertions only: the nonlinear copula transform changes the
    correlation function, so r_k ≠ exp(−kΔt/τ) exactly.
    """

    N = 500_000
    LAGS = [1, 2, 5, 10, 20]

    @staticmethod
    def _autocorrs(tau: float, seed: int = 1) -> list[float]:
        series = correlated_gamma_gamma_sample(
            4.0, 2.0, TestTemporalStructure.N, DT,
            tau_large=tau, tau_small=tau, rng=np.random.default_rng(seed),
        )
        return [empirical_autocorrelation(series, k) for k in TestTemporalStructure.LAGS]

    def test_autocorrelation_decays_with_lag(self) -> None:
        """r_k strictly decreases over k for a series with memory."""
        for tau in [5e-3, 50e-3]:
            r = self._autocorrs(tau)
            assert all(r[i] > r[i + 1] for i in range(len(r) - 1)), (
                f"τ={tau}: autocorrelation should decay with lag, got {r}"
            )

    def test_autocorrelation_monotone_in_tau(self) -> None:
        """At every lag, larger τ → more memory → higher r_k."""
        r_short = self._autocorrs(5e-3)
        r_long = self._autocorrs(50e-3)
        for k, (rs, rl) in zip(self.LAGS, zip(r_short, r_long)):
            assert rl > rs, (
                f"lag {k}: r(τ=50ms)={rl:.4f} should exceed r(τ=5ms)={rs:.4f}"
            )

    def test_memory_is_substantial_for_long_tau(self) -> None:
        """τ = 50 ms at Δt = 1 ms should retain strong lag-1 correlation."""
        r1 = self._autocorrs(50e-3)[0]
        assert r1 > 0.5, f"Expected substantial lag-1 memory, got r₁={r1:.4f}"


# ---------------------------------------------------------------------------
# Reproducibility and input validation
# ---------------------------------------------------------------------------


class TestReproducibilityAndValidation:
    """Deterministic under a seed; loud on invalid input."""

    def test_reproducible_with_seed(self) -> None:
        """Same seed → identical series."""
        rng1 = np.random.default_rng(seed=99)
        rng2 = np.random.default_rng(seed=99)
        s1 = correlated_gamma_gamma_sample(3.0, 2.0, 1_000, DT, TAU_LARGE, TAU_SMALL, rng1)
        s2 = correlated_gamma_gamma_sample(3.0, 2.0, 1_000, DT, TAU_LARGE, TAU_SMALL, rng2)
        np.testing.assert_array_equal(s1, s2)

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            correlated_gamma_gamma_sample(-1.0, 2.0, 100, DT, TAU_LARGE, TAU_SMALL)

    def test_invalid_beta_raises(self) -> None:
        with pytest.raises(ValueError, match="beta"):
            correlated_gamma_gamma_sample(2.0, 0.0, 100, DT, TAU_LARGE, TAU_SMALL)

    def test_invalid_n_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="n_samples"):
            correlated_gamma_gamma_sample(2.0, 2.0, 0, DT, TAU_LARGE, TAU_SMALL)

    def test_invalid_dt_raises(self) -> None:
        with pytest.raises(ValueError, match="dt"):
            correlated_gamma_gamma_sample(2.0, 2.0, 100, 0.0, TAU_LARGE, TAU_SMALL)

    def test_negative_tau_raises(self) -> None:
        with pytest.raises(ValueError, match="tau"):
            correlated_gamma_gamma_sample(2.0, 2.0, 100, DT, -1e-3, TAU_SMALL)
        with pytest.raises(ValueError, match="tau"):
            correlated_gamma_gamma_sample(2.0, 2.0, 100, DT, TAU_LARGE, -1e-3)


# ---------------------------------------------------------------------------
# Autocorrelation helper
# ---------------------------------------------------------------------------


class TestEmpiricalAutocorrelation:
    """The estimator itself should behave on known inputs."""

    def test_lag_zero_is_one(self) -> None:
        samples = np.random.default_rng(0).standard_normal(1_000)
        assert empirical_autocorrelation(samples, 0) == 1.0

    def test_white_noise_near_zero(self) -> None:
        samples = np.random.default_rng(0).standard_normal(100_000)
        assert abs(empirical_autocorrelation(samples, 1)) < 0.02

    def test_constant_shift_invariant(self) -> None:
        """Adding a constant must not change the autocorrelation."""
        rng = np.random.default_rng(3)
        samples = rng.standard_normal(10_000)
        r_base = empirical_autocorrelation(samples, 5)
        r_shift = empirical_autocorrelation(samples + 100.0, 5)
        assert abs(r_base - r_shift) < 1e-9

    def test_negative_lag_raises(self) -> None:
        with pytest.raises(ValueError, match="lag"):
            empirical_autocorrelation(np.ones(10), -1)

    def test_lag_too_large_raises(self) -> None:
        with pytest.raises(ValueError, match="lag"):
            empirical_autocorrelation(np.ones(10), 10)
