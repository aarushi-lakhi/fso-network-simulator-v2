"""
Gamma-Gamma atmospheric turbulence model for Free-Space Optical (FSO) links.

Implements the Gamma-Gamma irradiance distribution following:
    Andrews, L.C. & Phillips, R.L. (2005).
    Laser Beam Propagation through Random Media (2nd ed.). SPIE Press.

The Gamma-Gamma model treats atmospheric turbulence as the product of two
independent Gamma-distributed random processes:
    - Large-scale (slow) irradiance fluctuations: X ~ Gamma(α, 1/α)
    - Small-scale (fast) irradiance fluctuations:  Y ~ Gamma(β, 1/β)
    - Received irradiance: I = X * Y   (E[I] = 1, normalized)

This two-parameter model accurately covers weak through strong turbulence
regimes and is the standard for FSO link performance analysis.

Temporal correlation (Phase 6a): correlated_gamma_gamma_sample() adds
tunable coherence times via a Gaussian copula AR(1) process per Gamma
component, preserving the exact Gamma-Gamma marginal.

Typical usage:
    >>> params = TurbulenceParams(C2n=1e-15, wavelength=1550e-9, distance=1000.0)
    >>> alpha, beta = params.alpha_beta()
    >>> samples = gamma_gamma_sample(alpha, beta, n_samples=10_000)
    >>> print(f"SI = {scintillation_index(alpha, beta):.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import lfilter
from scipy.special import erfc
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

#: Telecom-band optical wavelength [m]. FSO systems commonly use 1550 nm.
WAVELENGTH_1550NM: float = 1550e-9

#: Visible-band optical wavelength [m]. Sometimes used in shorter FSO links.
WAVELENGTH_850NM: float = 850e-9


# ---------------------------------------------------------------------------
# Turbulence regime presets (C²_n values)
# ---------------------------------------------------------------------------

#: Weak turbulence: C²_n ≈ 10⁻¹⁷ m^(-2/3)
C2N_WEAK: float = 1e-17

#: Moderate turbulence: C²_n ≈ 10⁻¹⁵ m^(-2/3)
C2N_MODERATE: float = 1e-15

#: Strong turbulence: C²_n ≈ 10⁻¹³ m^(-2/3)
C2N_STRONG: float = 1e-13


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TurbulenceParams:
    """Physical parameters defining an FSO link and its turbulence regime.

    Attributes:
        C2n: Refractive index structure parameter [m^(-2/3)].
            Controls turbulence strength. Typical range: 10⁻¹⁷ (weak)
            to 10⁻¹³ (strong).
        wavelength: Optical carrier wavelength [m]. Default 1550 nm.
        distance: Propagation path length (link distance) [m].

    Example:
        >>> params = TurbulenceParams(C2n=1e-15, wavelength=1550e-9, distance=1000.0)
        >>> sigma2_R = params.rytov_variance()
        >>> alpha, beta = params.alpha_beta()
    """

    C2n: float
    wavelength: float = field(default=WAVELENGTH_1550NM)
    distance: float = field(default=1000.0)

    def __post_init__(self) -> None:
        if self.C2n <= 0:
            raise ValueError(f"C2n must be positive, got {self.C2n}")
        if self.wavelength <= 0:
            raise ValueError(f"wavelength must be positive, got {self.wavelength}")
        if self.distance <= 0:
            raise ValueError(f"distance must be positive, got {self.distance}")

    def rytov_variance(self) -> float:
        """Compute the Rytov variance for this link.

        Returns:
            σ²_R — the Rytov variance (dimensionless). Regimes
            (Andrews & Phillips, 2005):
                σ²_R < 0.3 → weak fluctuations
                0.3 ≤ σ²_R ≤ 1 → moderate fluctuations
                σ²_R > 1 → strong fluctuations (saturation as σ²_R → ∞)
        """
        return rytov_variance(self.C2n, self.wavelength, self.distance)

    def alpha_beta(self) -> tuple[float, float]:
        """Compute Gamma-Gamma shape parameters α and β for this link.

        Returns:
            Tuple (alpha, beta) where both values are > 0.
            Larger values → weaker turbulence (distribution more concentrated).
        """
        return alpha_beta_from_rytov(self.rytov_variance())

    def scintillation_index(self) -> float:
        """Compute the theoretical scintillation index SI for this link.

        Returns:
            SI = 1/α + 1/β + 1/(αβ). Range: ~0 (no turbulence) to >1 (strong).
        """
        alpha, beta = self.alpha_beta()
        return scintillation_index(alpha, beta)


# ---------------------------------------------------------------------------
# Core mathematical functions
# ---------------------------------------------------------------------------


def rytov_variance(C2n: float, wavelength: float, distance: float) -> float:
    """Compute the Rytov variance (plane wave approximation).

    The Rytov variance σ²_R characterises the turbulence strength along a
    propagation path. It is the key parameter linking atmospheric conditions
    (C²_n) to the Gamma-Gamma shape parameters α and β.

    Formula:
        σ²_R = 1.23 × C²_n × k^(7/6) × L^(11/6)
        where k = 2π/λ (optical wavenumber)

    Args:
        C2n: Refractive index structure parameter [m^(-2/3)]. Must be > 0.
        wavelength: Optical wavelength [m]. Must be > 0.
        distance: Link distance [m]. Must be > 0.

    Returns:
        Rytov variance σ²_R (dimensionless, ≥ 0).

    References:
        Andrews & Phillips (2005), Eq. 8.7, p. 261.
    """
    k = 2.0 * np.pi / wavelength
    return 1.23 * C2n * k ** (7.0 / 6.0) * distance ** (11.0 / 6.0)


def alpha_beta_from_rytov(sigma2_R: float) -> tuple[float, float]:
    """Convert Rytov variance to Gamma-Gamma shape parameters α and β.

    Uses the plane-wave closed-form approximations from Andrews & Phillips.
    These expressions relate the physical turbulence strength (σ²_R) to the
    statistical parameters (α, β) of the Gamma-Gamma irradiance distribution.

    Formulas:
        α = 1 / [exp(A) − 1],
            where A = 0.49 σ²_R / (1 + 1.11 σ_R^(12/5))^(7/6)

        β = 1 / [exp(B) − 1],
            where B = 0.51 σ²_R / (1 + 0.69 σ_R^(12/5))^(5/6)

    Args:
        sigma2_R: Rytov variance (dimensionless). Must be ≥ 0.

    Returns:
        Tuple (alpha, beta), both > 0.
        As σ²_R → 0 (no turbulence): α, β → ∞.
        As σ²_R → ∞ (saturation): α → 1, β remains moderate.

    Raises:
        ValueError: If sigma2_R is negative.

    References:
        Andrews & Phillips (2005), Eqs. 8.16–8.17, p. 264.
    """
    if sigma2_R < 0:
        raise ValueError(f"Rytov variance must be non-negative, got {sigma2_R}")

    sigma_R = np.sqrt(sigma2_R)

    exp_arg_alpha = 0.49 * sigma2_R / (1.0 + 1.11 * sigma_R ** (12.0 / 5.0)) ** (7.0 / 6.0)
    exp_arg_beta  = 0.51 * sigma2_R / (1.0 + 0.69 * sigma_R ** (12.0 / 5.0)) ** (5.0 / 6.0)

    alpha = 1.0 / (np.exp(exp_arg_alpha) - 1.0)
    beta  = 1.0 / (np.exp(exp_arg_beta)  - 1.0)

    return alpha, beta


def gamma_gamma_sample(
    alpha: float,
    beta: float,
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate Gamma-Gamma distributed irradiance samples.

    Samples I = X * Y where:
        X ~ Gamma(α, scale=1/α)  — large-scale fluctuation, E[X] = 1
        Y ~ Gamma(β, scale=1/β)  — small-scale fluctuation, E[Y] = 1

    This product-of-Gammas method is numerically efficient and exact.
    The resulting I satisfies E[I] = 1 (normalised irradiance).

    Args:
        alpha: Large-scale scintillation parameter. Must be > 0.
        beta: Small-scale scintillation parameter. Must be > 0.
        n_samples: Number of irradiance samples to generate. Must be > 0.
        rng: NumPy random Generator for reproducibility. If None, a fresh
            default_rng() is used (not reproducible across calls).

    Returns:
        np.ndarray of shape (n_samples,) with Gamma-Gamma distributed values.
        All values are > 0 and E[I] ≈ 1.

    Raises:
        ValueError: If alpha, beta, or n_samples are not positive.

    Example:
        >>> rng = np.random.default_rng(seed=42)
        >>> samples = gamma_gamma_sample(alpha=4.0, beta=2.0, n_samples=1_000, rng=rng)
        >>> assert abs(samples.mean() - 1.0) < 0.05  # E[I] ≈ 1
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    if rng is None:
        rng = np.random.default_rng()

    # Large-scale irradiance fluctuation (atmospheric eddies larger than Fresnel zone)
    large_scale = rng.gamma(shape=alpha, scale=1.0 / alpha, size=n_samples)

    # Small-scale irradiance fluctuation (atmospheric eddies smaller than Fresnel zone)
    small_scale = rng.gamma(shape=beta, scale=1.0 / beta, size=n_samples)

    return large_scale * small_scale


def scintillation_index(alpha: float, beta: float) -> float:
    """Compute the theoretical scintillation index (SI) from α and β.

    The scintillation index measures the normalised variance of the received
    irradiance. SI = 0 means no fluctuation; SI > 1 indicates strong fading.

    Formula (closed-form for Gamma-Gamma):
        SI = E[I²] / E[I]² − 1 = 1/α + 1/β + 1/(αβ)

    This identity is used to validate the Gamma-Gamma sampler: the empirical
    variance of a large sample set should match this theoretical value.

    Args:
        alpha: Large-scale scintillation parameter. Must be > 0.
        beta: Small-scale scintillation parameter. Must be > 0.

    Returns:
        Scintillation index SI (dimensionless, ≥ 0).

    Raises:
        ValueError: If alpha or beta are not positive.

    References:
        Andrews & Phillips (2005), Eq. 8.13, p. 263.
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError(f"alpha and beta must be positive, got α={alpha}, β={beta}")
    return 1.0 / alpha + 1.0 / beta + 1.0 / (alpha * beta)


def empirical_scintillation_index(samples: np.ndarray) -> float:
    """Compute the scintillation index empirically from irradiance samples.

    SI_empirical = E[I²] / E[I]² − 1 = Var[I] / E[I]²

    Used to validate that gamma_gamma_sample() converges to the theoretical SI.

    Args:
        samples: Array of irradiance samples. All values should be > 0.

    Returns:
        Empirical scintillation index.
    """
    mean_I = np.mean(samples)
    mean_I2 = np.mean(samples ** 2)
    return mean_I2 / mean_I ** 2 - 1.0


# ---------------------------------------------------------------------------
# Temporally correlated sampling (Gaussian copula AR(1))
# ---------------------------------------------------------------------------


def ar1_coefficient(dt: float, tau: float) -> float:
    """Compute the AR(1) coefficient ρ for a given coherence time.

    The latent Gaussian process underlying each Gamma component decorrelates
    exponentially with the atmospheric coherence time τ:

        ρ = exp(−Δt / τ)

    so that Corr[g_t, g_{t+k}] = ρᵏ = exp(−kΔt/τ). τ = 0 is taken as the
    memoryless limit ρ = 0 (each step is an independent draw).

    Args:
        dt: Sample interval Δt [s]. Must be > 0.
        tau: Coherence time τ [s]. Must be ≥ 0; τ = 0 → ρ = 0 (i.i.d.).

    Returns:
        AR(1) coefficient ρ ∈ [0, 1).

    Raises:
        ValueError: If dt is not positive or tau is negative.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if tau < 0:
        raise ValueError(f"tau must be non-negative, got {tau}")
    if tau == 0.0:
        return 0.0
    return float(np.exp(-dt / tau))


def _correlated_gamma_component(
    shape: float,
    n_samples: int,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a stationary Gamma(shape, 1/shape) series with AR(1) copula memory.

    Latent standard-normal AR(1) process (stationary initialisation):
        g_0 ~ N(0, 1),   g_{t+1} = ρ·g_t + √(1−ρ²)·ε_t,   ε_t ~ N(0, 1)

    mapped through the Gaussian copula to the exact Gamma marginal:
        u_t = Φ(g_t),   component_t = F⁻¹_Gamma(u_t; shape, scale=1/shape)

    Since g_t is exactly N(0, 1) at every t, the marginal of component_t is
    exactly Gamma(shape, 1/shape) at every t — only the temporal dependence
    is introduced by ρ.
    """
    eps = rng.standard_normal(n_samples)

    # Innovation sequence: g_0 = ε_0 (stationary start), then scaled by √(1−ρ²)
    innovations = np.sqrt(1.0 - rho ** 2) * eps
    innovations[0] = eps[0]

    # g_t = ρ·g_{t−1} + innovation_t, computed as an IIR filter (vectorised)
    latent = lfilter([1.0], [1.0, -rho], innovations)

    # Copula transform; clip guards against Φ(g) rounding to exactly 0 or 1
    u = np.clip(norm.cdf(latent), 1e-16, 1.0 - 1e-16)
    return gamma_dist.ppf(u, a=shape, scale=1.0 / shape)


def correlated_gamma_gamma_sample(
    alpha: float,
    beta: float,
    n_samples: int,
    dt: float,
    tau_large: float,
    tau_small: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate a temporally correlated Gamma-Gamma irradiance time series.

    Extends gamma_gamma_sample() with tunable temporal coherence via a
    Gaussian copula AR(1) construction applied independently to each Gamma
    component:

        X_t = F⁻¹_Gamma(Φ(gX_t); α, 1/α)   with gX an AR(1), ρ_X = exp(−Δt/τ_large)
        Y_t = F⁻¹_Gamma(Φ(gY_t); β, 1/β)   with gY an AR(1), ρ_Y = exp(−Δt/τ_small)
        I_t = X_t · Y_t

    Because each copula transform is exact, the stationary marginal of I_t is
    EXACTLY the same Gamma-Gamma distribution as the i.i.d. sampler: E[I] = 1
    and SI = 1/α + 1/β + 1/(αβ) still hold at every time step. Only the
    temporal autocorrelation changes. τ_large = τ_small = 0 recovers i.i.d.
    behaviour. Note the autocorrelation of I_t is NOT exactly exp(−kΔt/τ):
    the nonlinear quantile transform reshapes the latent exponential decay
    (it stays monotone in lag and in τ).

    The two-scale structure follows the physical picture in Andrews &
    Phillips (2005): large eddies (X, refractive) evolve more slowly than
    small eddies (Y, diffractive), so typically τ_large > τ_small.

    Args:
        alpha: Large-scale scintillation parameter. Must be > 0.
        beta: Small-scale scintillation parameter. Must be > 0.
        n_samples: Number of time samples to generate. Must be > 0.
        dt: Sample interval Δt [s]. Must be > 0.
        tau_large: Coherence time [s] of the large-scale component X.
            Must be ≥ 0; 0 → memoryless (i.i.d.) X.
        tau_small: Coherence time [s] of the small-scale component Y.
            Must be ≥ 0; 0 → memoryless (i.i.d.) Y.
        rng: NumPy random Generator for reproducibility. If None, a fresh
            default_rng() is used (not reproducible across calls).

    Returns:
        np.ndarray of shape (n_samples,): irradiance time series I_t sampled
        every dt seconds. All values > 0, E[I] ≈ 1.

    Raises:
        ValueError: If alpha, beta, n_samples, or dt are not positive, or if
            either coherence time is negative.

    Example:
        >>> rng = np.random.default_rng(seed=42)
        >>> series = correlated_gamma_gamma_sample(
        ...     alpha=4.0, beta=1.7, n_samples=10_000,
        ...     dt=1e-3, tau_large=100e-3, tau_small=10e-3, rng=rng)
        >>> assert abs(series.mean() - 1.0) < 0.1
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    rho_large = ar1_coefficient(dt, tau_large)  # validates dt and tau_large
    rho_small = ar1_coefficient(dt, tau_small)

    if rng is None:
        rng = np.random.default_rng()

    large_scale = _correlated_gamma_component(alpha, n_samples, rho_large, rng)
    small_scale = _correlated_gamma_component(beta, n_samples, rho_small, rng)

    return large_scale * small_scale


def empirical_autocorrelation(samples: np.ndarray, lag: int) -> float:
    """Compute the lag-k autocorrelation of a time series empirically.

    r_k = Σ_t (I_t − Ī)(I_{t+k} − Ī) / Σ_t (I_t − Ī)²

    Used to validate the temporal structure of correlated_gamma_gamma_sample():
    r_k must decay with k and increase with the coherence times.

    Args:
        samples: Time-series array. Must have more than `lag` elements.
        lag: Lag k in samples. Must be ≥ 0.

    Returns:
        Empirical autocorrelation r_k ∈ [−1, 1] (r_0 = 1).

    Raises:
        ValueError: If lag is negative or not smaller than len(samples).
    """
    if lag < 0:
        raise ValueError(f"lag must be non-negative, got {lag}")
    if lag >= len(samples):
        raise ValueError(f"lag ({lag}) must be < len(samples) ({len(samples)})")

    centered = samples - samples.mean()
    denom = np.sum(centered ** 2)
    if lag == 0:
        return 1.0
    return float(np.sum(centered[:-lag] * centered[lag:]) / denom)


# ---------------------------------------------------------------------------
# BER analysis
# ---------------------------------------------------------------------------


def ber_ook_fading(
    snr_db_range: np.ndarray,
    alpha: float,
    beta: float,
    n_samples: int = 200_000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Compute average BER for OOK-IM/DD under Gamma-Gamma fading via Monte Carlo.

    Models intensity-modulated direct-detection (IM/DD) OOK, the dominant
    modulation scheme in commercial FSO systems. The conditional BER given
    instantaneous normalised irradiance I is:

        BER(I | SNR) = (1/2) * erfc(√(SNR * I / 2))

    The average BER is the expectation over the Gamma-Gamma distribution:

        BER_avg(SNR) = E_I[(1/2) * erfc(√(SNR * I / 2))]

    computed here via Monte Carlo integration over Gamma-Gamma samples.

    Args:
        snr_db_range: Array of SNR values [dB] at which to evaluate BER.
        alpha: Gamma-Gamma large-scale parameter. Must be > 0.
        beta: Gamma-Gamma small-scale parameter. Must be > 0.
        n_samples: Number of Monte Carlo samples. Higher → less variance
            in the BER estimate. Default 200,000.
        rng: Optional NumPy random Generator for reproducibility.

    Returns:
        np.ndarray of shape (len(snr_db_range),) with average BER values.
        Values are in (0, 0.5]; BER → 0.5 as SNR → −∞.

    References:
        Zhu & Kahn (2002), "Free-space optical communication through
        atmospheric turbulence channels," IEEE Trans. Commun., 50(8).
    """
    snr_linear = 10.0 ** (snr_db_range / 10.0)

    # Draw fading samples once; reuse across all SNR points (efficient)
    fading = gamma_gamma_sample(alpha, beta, n_samples, rng)

    ber_values = np.zeros(len(snr_db_range))
    for i, snr in enumerate(snr_linear):
        ber_values[i] = 0.5 * np.mean(erfc(np.sqrt(snr * fading / 2.0)))

    return ber_values


def ber_awgn_baseline(snr_db_range: np.ndarray) -> np.ndarray:
    """Compute BER for OOK-IM/DD in AWGN (no fading baseline).

    Formula:
        BER_AWGN(SNR) = (1/2) * erfc(√(SNR / 2))

    Args:
        snr_db_range: Array of SNR values [dB].

    Returns:
        np.ndarray of shape (len(snr_db_range),) with AWGN BER values.
    """
    snr_linear = 10.0 ** (snr_db_range / 10.0)
    return 0.5 * erfc(np.sqrt(snr_linear / 2.0))
