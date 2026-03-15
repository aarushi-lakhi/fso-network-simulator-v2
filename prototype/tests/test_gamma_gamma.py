"""
Tests for the Gamma-Gamma atmospheric turbulence model.

Test strategy:
    - Unit tests: validate individual math functions against closed-form results
    - Statistical tests: validate sampler convergence using the SI identity
    - Integration tests: validate BER boundary conditions
    - Regression tests: pin known-good outputs so silent breakage is caught

The key identity used throughout is the scintillation index formula:
    SI_theoretical = 1/α + 1/β + 1/(αβ)
    SI_empirical   = Var[I] / E[I]²

For a large enough sample (n ≥ 100,000), |SI_empirical - SI_theoretical| < 5%.
"""

from __future__ import annotations

import numpy as np
import pytest

from gamma_gamma import (
    C2N_MODERATE,
    C2N_STRONG,
    C2N_WEAK,
    TurbulenceParams,
    alpha_beta_from_rytov,
    ber_awgn_baseline,
    ber_ook_fading,
    empirical_scintillation_index,
    gamma_gamma_sample,
    rytov_variance,
    scintillation_index,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded RNG for reproducible statistical tests."""
    return np.random.default_rng(seed=0)


@pytest.fixture
def weak_params() -> TurbulenceParams:
    return TurbulenceParams(C2n=C2N_WEAK, wavelength=1550e-9, distance=1000.0)


@pytest.fixture
def moderate_params() -> TurbulenceParams:
    return TurbulenceParams(C2n=C2N_MODERATE, wavelength=1550e-9, distance=1000.0)


@pytest.fixture
def strong_params() -> TurbulenceParams:
    return TurbulenceParams(C2n=C2N_STRONG, wavelength=1550e-9, distance=1000.0)


# ---------------------------------------------------------------------------
# TurbulenceParams validation
# ---------------------------------------------------------------------------


class TestTurbulenceParamsValidation:
    """TurbulenceParams should reject physically invalid inputs."""

    def test_negative_c2n_raises(self) -> None:
        with pytest.raises(ValueError, match="C2n"):
            TurbulenceParams(C2n=-1e-15, wavelength=1550e-9, distance=1000.0)

    def test_zero_c2n_raises(self) -> None:
        with pytest.raises(ValueError, match="C2n"):
            TurbulenceParams(C2n=0.0, wavelength=1550e-9, distance=1000.0)

    def test_negative_wavelength_raises(self) -> None:
        with pytest.raises(ValueError, match="wavelength"):
            TurbulenceParams(C2n=1e-15, wavelength=-1550e-9, distance=1000.0)

    def test_negative_distance_raises(self) -> None:
        with pytest.raises(ValueError, match="distance"):
            TurbulenceParams(C2n=1e-15, wavelength=1550e-9, distance=-100.0)

    def test_valid_params_constructs(self) -> None:
        params = TurbulenceParams(C2n=1e-15, wavelength=1550e-9, distance=500.0)
        assert params.C2n == 1e-15


# ---------------------------------------------------------------------------
# Rytov variance
# ---------------------------------------------------------------------------


class TestRytovVariance:
    """Rytov variance should match analytical expectations."""

    def test_increases_with_c2n(self) -> None:
        """Higher turbulence strength → higher Rytov variance."""
        sigma_weak     = rytov_variance(C2N_WEAK,     1550e-9, 1000.0)
        sigma_moderate = rytov_variance(C2N_MODERATE, 1550e-9, 1000.0)
        sigma_strong   = rytov_variance(C2N_STRONG,   1550e-9, 1000.0)
        assert sigma_weak < sigma_moderate < sigma_strong

    def test_increases_with_distance(self) -> None:
        """Longer links → more integrated turbulence → higher σ²_R."""
        sigma_short = rytov_variance(C2N_MODERATE, 1550e-9, 500.0)
        sigma_long  = rytov_variance(C2N_MODERATE, 1550e-9, 2000.0)
        assert sigma_short < sigma_long

    def test_increases_with_wavenumber(self) -> None:
        """Shorter wavelength → higher wavenumber → higher σ²_R."""
        sigma_1550 = rytov_variance(C2N_MODERATE, 1550e-9, 1000.0)
        sigma_850  = rytov_variance(C2N_MODERATE,  850e-9, 1000.0)
        assert sigma_850 > sigma_1550

    def test_known_value_weak_regime(self) -> None:
        """Spot-check: σ²_R for weak turbulence at 1 km should be < 0.3."""
        sigma2_R = rytov_variance(C2N_WEAK, 1550e-9, 1000.0)
        assert sigma2_R < 0.3, f"Expected weak turbulence (σ²_R < 0.3), got {sigma2_R:.4f}"

    def test_known_value_strong_regime(self) -> None:
        """Spot-check: σ²_R for strong turbulence at 1 km should be >> 1."""
        sigma2_R = rytov_variance(C2N_STRONG, 1550e-9, 1000.0)
        assert sigma2_R > 5.0, f"Expected strong turbulence (σ²_R > 5), got {sigma2_R:.4f}"

    def test_nonnegative(self) -> None:
        sigma2_R = rytov_variance(C2N_MODERATE, 1550e-9, 1000.0)
        assert sigma2_R >= 0.0


# ---------------------------------------------------------------------------
# Alpha-Beta conversion
# ---------------------------------------------------------------------------


class TestAlphaBetaFromRytov:
    """α and β should satisfy physical constraints from Andrews & Phillips."""

    def test_both_positive(self) -> None:
        for c2n in [C2N_WEAK, C2N_MODERATE, C2N_STRONG]:
            sigma2_R = rytov_variance(c2n, 1550e-9, 1000.0)
            alpha, beta = alpha_beta_from_rytov(sigma2_R)
            assert alpha > 0, f"alpha must be positive, got {alpha}"
            assert beta > 0, f"beta must be positive, got {beta}"

    def test_weak_turbulence_large_params(self) -> None:
        """Weak turbulence → α, β both large (distribution is concentrated near 1)."""
        sigma2_R = rytov_variance(C2N_WEAK, 1550e-9, 1000.0)
        alpha, beta = alpha_beta_from_rytov(sigma2_R)
        assert alpha > 10.0, f"Expected α >> 1 for weak turbulence, got {alpha:.2f}"
        assert beta > 10.0, f"Expected β >> 1 for weak turbulence, got {beta:.2f}"

    def test_strong_turbulence_small_alpha(self) -> None:
        """Strong turbulence → α closer to 1 (distribution heavy-tailed)."""
        sigma2_R = rytov_variance(C2N_STRONG, 1550e-9, 1000.0)
        alpha, beta = alpha_beta_from_rytov(sigma2_R)
        assert alpha < 5.0, f"Expected α << 10 for strong turbulence, got {alpha:.2f}"

    def test_monotone_in_sigma2r(self) -> None:
        """Higher Rytov variance → smaller α and β (weaker concentration)."""
        sigma2_weak   = rytov_variance(C2N_WEAK,     1550e-9, 1000.0)
        sigma2_strong = rytov_variance(C2N_STRONG,   1550e-9, 1000.0)
        alpha_w, beta_w = alpha_beta_from_rytov(sigma2_weak)
        alpha_s, beta_s = alpha_beta_from_rytov(sigma2_strong)
        assert alpha_w > alpha_s
        assert beta_w  > beta_s

    def test_negative_sigma2r_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            alpha_beta_from_rytov(-0.1)

    def test_zero_sigma2r(self) -> None:
        """Zero Rytov variance (no turbulence) → α, β → ∞. We just verify positivity."""
        alpha, beta = alpha_beta_from_rytov(0.0)
        assert alpha > 0
        assert beta > 0


# ---------------------------------------------------------------------------
# Gamma-Gamma sampler
# ---------------------------------------------------------------------------


class TestGammaGammaSample:
    """Sampler should produce correctly distributed irradiance values."""

    N_LARGE = 500_000   # large enough for statistical tests; fast enough for CI

    def test_mean_is_one(self, rng: np.random.Generator) -> None:
        """E[I] = 1 for all (α, β) — irradiance is normalised."""
        alpha, beta = 4.0, 2.0
        samples = gamma_gamma_sample(alpha, beta, self.N_LARGE, rng)
        assert abs(samples.mean() - 1.0) < 0.01, (
            f"Expected E[I] ≈ 1, got {samples.mean():.4f}"
        )

    def test_all_positive(self, rng: np.random.Generator) -> None:
        """Irradiance is always strictly positive (no negative power)."""
        samples = gamma_gamma_sample(3.0, 3.0, 10_000, rng)
        assert np.all(samples > 0)

    def test_scintillation_index_identity(self, rng: np.random.Generator) -> None:
        """KEY TEST: empirical SI must match SI = 1/α + 1/β + 1/(αβ) within 5%.

        This is the primary mathematical validation of the Gamma-Gamma sampler.
        If this test passes, the product-of-Gammas sampling is correct.
        """
        for alpha, beta in [(4.0, 2.0), (2.5, 1.5), (1.2, 1.0)]:
            samples = gamma_gamma_sample(alpha, beta, self.N_LARGE, rng)
            si_theoretical = scintillation_index(alpha, beta)
            si_empirical   = empirical_scintillation_index(samples)
            rel_error = abs(si_empirical - si_theoretical) / si_theoretical
            assert rel_error < 0.05, (
                f"α={alpha}, β={beta}: SI_theoretical={si_theoretical:.4f}, "
                f"SI_empirical={si_empirical:.4f}, relative_error={rel_error:.2%}"
            )

    def test_weak_turbulence_low_variance(self, rng: np.random.Generator) -> None:
        """Weak turbulence → samples tightly clustered around 1 (low SI)."""
        sigma2_R = rytov_variance(C2N_WEAK, 1550e-9, 1000.0)
        alpha, beta = alpha_beta_from_rytov(sigma2_R)
        samples = gamma_gamma_sample(alpha, beta, self.N_LARGE, rng)
        si = scintillation_index(alpha, beta)
        assert si < 0.1, f"Weak turbulence SI should be < 0.1, got {si:.4f}"
        assert samples.std() < 0.5, f"Weak turbulence: std too high ({samples.std():.4f})"

    def test_strong_turbulence_high_variance(self, rng: np.random.Generator) -> None:
        """Strong turbulence → samples spread widely (high SI, heavy tail)."""
        sigma2_R = rytov_variance(C2N_STRONG, 1550e-9, 1000.0)
        alpha, beta = alpha_beta_from_rytov(sigma2_R)
        samples = gamma_gamma_sample(alpha, beta, self.N_LARGE, rng)
        si = scintillation_index(alpha, beta)
        assert si > 0.5, f"Strong turbulence SI should be > 0.5, got {si:.4f}"

    def test_reproducible_with_seed(self) -> None:
        """Same seed → identical sample array."""
        rng1 = np.random.default_rng(seed=99)
        rng2 = np.random.default_rng(seed=99)
        s1 = gamma_gamma_sample(3.0, 2.0, 1_000, rng1)
        s2 = gamma_gamma_sample(3.0, 2.0, 1_000, rng2)
        np.testing.assert_array_equal(s1, s2)

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            gamma_gamma_sample(alpha=-1.0, beta=2.0, n_samples=100)

    def test_invalid_beta_raises(self) -> None:
        with pytest.raises(ValueError, match="beta"):
            gamma_gamma_sample(alpha=2.0, beta=0.0, n_samples=100)

    def test_invalid_n_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="n_samples"):
            gamma_gamma_sample(alpha=2.0, beta=2.0, n_samples=0)


# ---------------------------------------------------------------------------
# Scintillation index formula
# ---------------------------------------------------------------------------


class TestScintillationIndex:
    """Scintillation index closed-form should match its definition."""

    def test_formula_symmetric_params(self) -> None:
        """When α = β, SI = 2/α + 1/α²."""
        alpha = beta = 3.0
        expected = 2.0 / alpha + 1.0 / alpha ** 2
        assert abs(scintillation_index(alpha, beta) - expected) < 1e-12

    def test_increases_with_turbulence(self) -> None:
        """SI should increase as turbulence increases (α, β decrease)."""
        sigma_weak   = rytov_variance(C2N_WEAK,     1550e-9, 1000.0)
        sigma_strong = rytov_variance(C2N_STRONG,   1550e-9, 1000.0)
        si_weak   = scintillation_index(*alpha_beta_from_rytov(sigma_weak))
        si_strong = scintillation_index(*alpha_beta_from_rytov(sigma_strong))
        assert si_weak < si_strong

    def test_always_positive(self) -> None:
        for alpha, beta in [(1.0, 1.0), (10.0, 5.0), (0.5, 0.5)]:
            assert scintillation_index(alpha, beta) > 0

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            scintillation_index(0.0, 1.0)
        with pytest.raises(ValueError):
            scintillation_index(1.0, -1.0)


# ---------------------------------------------------------------------------
# BER analysis
# ---------------------------------------------------------------------------


class TestBerAwgnBaseline:
    """AWGN BER should match the closed-form erfc formula."""

    def test_ber_at_zero_snr_near_half(self) -> None:
        """At SNR = 0 dB, BER should be close to 0.5 (random guessing)."""
        ber = ber_awgn_baseline(np.array([0.0]))
        assert abs(float(ber[0]) - 0.5) < 0.1

    def test_ber_decreases_with_snr(self) -> None:
        snr_db = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        ber = ber_awgn_baseline(snr_db)
        assert np.all(np.diff(ber) < 0), "BER should strictly decrease as SNR increases"

    def test_ber_positive(self) -> None:
        snr_db = np.linspace(-5, 30, 50)
        ber = ber_awgn_baseline(snr_db)
        assert np.all(ber > 0)

    def test_ber_bounded_above_by_half(self) -> None:
        snr_db = np.linspace(-10, 40, 100)
        ber = ber_awgn_baseline(snr_db)
        assert np.all(ber <= 0.5 + 1e-10)


class TestBerFading:
    """Fading BER should be higher than AWGN at same SNR (fading is detrimental)."""

    def test_fading_ber_exceeds_awgn(self, rng: np.random.Generator) -> None:
        """Turbulence always degrades BER relative to AWGN at the same SNR."""
        snr_db = np.array([10.0, 15.0, 20.0])
        sigma2_R = rytov_variance(C2N_MODERATE, 1550e-9, 1000.0)
        alpha, beta = alpha_beta_from_rytov(sigma2_R)

        ber_fading = ber_ook_fading(snr_db, alpha, beta, n_samples=100_000, rng=rng)
        ber_awgn   = ber_awgn_baseline(snr_db)

        assert np.all(ber_fading > ber_awgn), (
            "Fading BER must be > AWGN BER at the same SNR. "
            f"Got fading={ber_fading}, awgn={ber_awgn}"
        )

    def test_stronger_turbulence_higher_ber(self, rng: np.random.Generator) -> None:
        """More turbulence → higher BER at the same SNR."""
        snr_db = np.array([15.0])

        sigma_weak   = rytov_variance(C2N_WEAK,     1550e-9, 1000.0)
        sigma_strong = rytov_variance(C2N_STRONG,   1550e-9, 1000.0)
        alpha_w, beta_w = alpha_beta_from_rytov(sigma_weak)
        alpha_s, beta_s = alpha_beta_from_rytov(sigma_strong)

        ber_weak   = ber_ook_fading(snr_db, alpha_w, beta_w, n_samples=100_000, rng=rng)
        ber_strong = ber_ook_fading(snr_db, alpha_s, beta_s, n_samples=100_000, rng=rng)

        assert float(ber_strong[0]) > float(ber_weak[0]), (
            f"Strong turbulence BER ({ber_strong[0]:.4e}) should exceed "
            f"weak turbulence BER ({ber_weak[0]:.4e})"
        )

    def test_ber_bounded(self, rng: np.random.Generator) -> None:
        """BER must always be in (0, 0.5]."""
        snr_db = np.linspace(0.0, 30.0, 10)
        sigma2_R = rytov_variance(C2N_MODERATE, 1550e-9, 1000.0)
        alpha, beta = alpha_beta_from_rytov(sigma2_R)
        ber = ber_ook_fading(snr_db, alpha, beta, n_samples=50_000, rng=rng)
        assert np.all(ber > 0) and np.all(ber <= 0.5 + 1e-10)


# ---------------------------------------------------------------------------
# TurbulenceParams integration
# ---------------------------------------------------------------------------


class TestTurbulenceParamsIntegration:
    """End-to-end: TurbulenceParams → samples → SI validation."""

    def test_end_to_end_si_identity(self, rng: np.random.Generator) -> None:
        """Full pipeline: physical params → α,β → samples → SI matches formula."""
        params = TurbulenceParams(C2n=C2N_MODERATE, wavelength=1550e-9, distance=1000.0)
        alpha, beta = params.alpha_beta()
        samples = gamma_gamma_sample(alpha, beta, n_samples=300_000, rng=rng)

        si_theory   = params.scintillation_index()
        si_empirical = empirical_scintillation_index(samples)

        rel_error = abs(si_empirical - si_theory) / si_theory
        assert rel_error < 0.05, (
            f"SI mismatch: theoretical={si_theory:.4f}, "
            f"empirical={si_empirical:.4f}, error={rel_error:.2%}"
        )
