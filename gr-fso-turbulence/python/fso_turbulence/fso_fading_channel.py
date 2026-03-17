"""
Gamma-Gamma atmospheric fading channel block for GNU Radio.

Applies Gamma-Gamma distributed irradiance fading to a complex baseband
stream, following:
    Andrews, L.C. & Phillips, R.L. (2005).
    Laser Beam Propagation through Random Media (2nd ed.). SPIE Press.

The irradiance is modeled as the product of two independent Gamma-distributed
random processes (large-scale and small-scale scintillation), with shape
parameters alpha and beta derived from the Rytov variance via the plane-wave
closed-form approximations. The math mirrors prototype/gamma_gamma.py; only
numpy is required (no scipy).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from gnuradio import gr


def rytov_variance(C2n: float, wavelength: float, distance: float) -> float:
    """Compute the Rytov variance (plane wave approximation).

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

    Formulas (plane-wave approximation):
        α = 1 / [exp(A) − 1],
            where A = 0.49 σ²_R / (1 + 1.11 σ_R^(12/5))^(7/6)

        β = 1 / [exp(B) − 1],
            where B = 0.51 σ²_R / (1 + 0.69 σ_R^(12/5))^(5/6)

    Args:
        sigma2_R: Rytov variance (dimensionless). Must be ≥ 0.

    Returns:
        Tuple (alpha, beta), both > 0.

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


def scintillation_index(alpha: float, beta: float) -> float:
    """Compute the theoretical scintillation index (SI) from α and β.

    Formula (closed-form for Gamma-Gamma):
        SI = E[I²] / E[I]² − 1 = 1/α + 1/β + 1/(αβ)

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


class fso_fading_channel(gr.sync_block):
    """Gamma-Gamma atmospheric fading channel (complex in → complex out).

    Multiplies the complex field amplitude by √I, where I is Gamma-Gamma
    distributed irradiance with E[I] = 1. Scaling the *amplitude* by √I means
    the signal *power* |out|² = I·|in|² follows the irradiance statistics —
    the physically correct mapping, since photodetectors respond to optical
    intensity, and the Gamma-Gamma PDF describes intensity, not field.

    Each fading coefficient is held for ``coherence_samples`` consecutive
    samples to model the turbulence coherence time: atmospheric fading evolves
    on millisecond scales (~kHz) while baseband sample rates are typically
    MHz, so one coefficient spans many samples.

    Args:
        C2n: Refractive index structure parameter [m^(-2/3)]. Must be > 0.
            Typical: 1e-17 (weak) to 1e-13 (strong).
        wavelength: Optical carrier wavelength [m]. Default 1550 nm.
        distance: Link distance [m]. Default 1000 m.
        coherence_samples: Samples per fading coefficient. Must be ≥ 1.
            Set to sample_rate × coherence_time (e.g. 1 MHz × 1 ms = 1000).
        seed: Seed for the NumPy random generator. None → non-reproducible.
    """

    def __init__(
        self,
        C2n: float = 1e-15,
        wavelength: float = 1550e-9,
        distance: float = 1000.0,
        coherence_samples: int = 1000,
        seed: Optional[int] = None,
    ) -> None:
        gr.sync_block.__init__(
            self,
            name="fso_fading_channel",
            in_sig=[np.complex64],
            out_sig=[np.complex64],
        )

        if C2n <= 0:
            raise ValueError(f"C2n must be positive, got {C2n}")
        if wavelength <= 0:
            raise ValueError(f"wavelength must be positive, got {wavelength}")
        if distance <= 0:
            raise ValueError(f"distance must be positive, got {distance}")
        if coherence_samples < 1:
            raise ValueError(f"coherence_samples must be >= 1, got {coherence_samples}")

        self.alpha, self.beta = alpha_beta_from_rytov(
            rytov_variance(C2n, wavelength, distance)
        )
        self._coherence_samples = int(coherence_samples)
        self._rng = np.random.default_rng(seed)
        self._gain = np.float32(0.0)
        self._remaining = 0  # samples left in the current coherence window

    def _next_gain(self) -> np.float32:
        """Draw a fresh amplitude gain √I from the Gamma-Gamma distribution."""
        large_scale = self._rng.gamma(shape=self.alpha, scale=1.0 / self.alpha)
        small_scale = self._rng.gamma(shape=self.beta, scale=1.0 / self.beta)
        return np.float32(np.sqrt(large_scale * small_scale))

    def work(self, input_items: list, output_items: list) -> int:
        inp = input_items[0]
        out = output_items[0]
        n = len(out)

        idx = 0
        while idx < n:
            if self._remaining == 0:
                self._gain = self._next_gain()
                self._remaining = self._coherence_samples
            take = min(n - idx, self._remaining)
            out[idx:idx + take] = inp[idx:idx + take] * self._gain
            self._remaining -= take
            idx += take

        return n
