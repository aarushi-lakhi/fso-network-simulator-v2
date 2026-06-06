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

Temporal correlation (Phase 6 parity): non-zero coherence times tau_large /
tau_small correlate successive coherence windows via a Gaussian copula AR(1)
process per Gamma component, preserving the exact Gamma-Gamma marginal
(mirrors prototype/gamma_gamma.py::correlated_gamma_gamma_sample and
ns3-fso-channel/model/correlated-gamma-gamma-fading.cc).
"""

from __future__ import annotations

import math
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


def _gammp(a: float, x: float) -> float:
    """Evaluate the regularized lower incomplete gamma function P(a, x).

    Uses the series expansion for x < a + 1 and the Lentz continued fraction
    for the complement otherwise (Numerical Recipes 3rd ed., §6.2).

    Args:
        a: Gamma shape parameter. Must be > 0.
        x: Evaluation point. Must be ≥ 0.

    Returns:
        P(a, x) ∈ [0, 1].

    Raises:
        ValueError: If a is not positive or x is negative.
    """
    if a <= 0:
        raise ValueError(f"a must be positive, got {a}")
    if x < 0:
        raise ValueError(f"x must be non-negative, got {x}")
    if x == 0.0:
        return 0.0

    eps = 1e-15
    gln = math.lgamma(a)
    prefactor = math.exp(-x + a * math.log(x) - gln)

    if x < a + 1.0:
        # Series: P(a, x) = prefactor * Σ_n x^n / (a (a+1) ... (a+n))
        ap = a
        term = 1.0 / a
        total = term
        for _ in range(500):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * eps:
                break
        return min(prefactor * total, 1.0)

    # Continued fraction for Q(a, x), modified Lentz's method
    fpmin = 1e-300
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return max(1.0 - prefactor * h, 0.0)


def _gamma_ppf_scalar(shape: float, u: float) -> float:
    """Invert P(shape, x) = u for the unit-scale Gamma quantile.

    Port of the Numerical Recipes `invgammp` algorithm (3rd ed., §6.2.1):
    an asymptotic initial guess (Wilson-Hilferty for shape > 1) refined by
    Halley iterations on _gammp(), matching scipy.stats.gamma.ppf(u, shape)
    to near machine precision.
    """
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return math.inf

    eps = 1e-12
    a1 = shape - 1.0
    gln = math.lgamma(shape)

    if shape > 1.0:
        lna1 = math.log(a1) if a1 > 0 else 0.0
        afac = math.exp(a1 * (lna1 - 1.0) - gln) if a1 > 0 else math.exp(-gln)
        pp = u if u < 0.5 else 1.0 - u
        t = math.sqrt(-2.0 * math.log(pp))
        x = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t
        if u < 0.5:
            x = -x
        x = max(1e-3, shape * (1.0 - 1.0 / (9.0 * shape) - x / (3.0 * math.sqrt(shape))) ** 3)
    else:
        t = 1.0 - shape * (0.253 + shape * 0.12)
        if u < t:
            x = (u / t) ** (1.0 / shape)
        else:
            x = 1.0 - math.log(1.0 - (u - t) / (1.0 - t))

    for _ in range(20):
        if x <= 0.0:
            return 0.0
        err = _gammp(shape, x) - u
        if shape > 1.0:
            t = afac * math.exp(-(x - a1) + a1 * (math.log(x) - lna1))
        else:
            t = math.exp(-x + a1 * math.log(x) - gln)
        # Halley correction on top of the Newton step err / t
        step = err / t
        step = step / (1.0 - 0.5 * min(1.0, step * ((shape - 1.0) / x - 1.0)))
        x -= step
        if x <= 0.0:
            x = 0.5 * (x + step)
        if abs(step) < eps * x:
            break
    return x


def gamma_ppf(shape: float, u: float | np.ndarray) -> float | np.ndarray:
    """Compute the unit-scale Gamma quantile function (inverse CDF).

    Pure numpy/math replacement for scipy.stats.gamma.ppf(u, shape), required
    because the GNU Radio runtime python has no scipy. Accuracy is validated
    in QA against a hardcoded scipy reference table (~1e-10 relative).

    Args:
        shape: Gamma shape parameter. Must be > 0.
        u: Probability (scalar or array) in [0, 1].

    Returns:
        x such that P(shape, x) = u, with scale 1 (0 at u ≤ 0, inf at u ≥ 1).

    Raises:
        ValueError: If shape is not positive.
    """
    if shape <= 0:
        raise ValueError(f"shape must be positive, got {shape}")
    if np.isscalar(u):
        return _gamma_ppf_scalar(shape, float(u))
    return np.vectorize(_gamma_ppf_scalar, otypes=[np.float64])(shape, u)


class _CopulaAr1State:
    """Latent standard-normal AR(1) state for one Gamma component.

    Mirrors ComponentState in ns-3's correlated-gamma-gamma-fading:
    g_0 = ε_0 (stationary start), g_t = ρ·g_{t-1} + √(1-ρ²)·ε_t.
    """

    __slots__ = ("rho", "g", "initialized")

    def __init__(self, rho: float) -> None:
        self.rho = rho
        self.g = 0.0
        self.initialized = False

    def step(self, epsilon: float) -> float:
        """Advance the latent process by one window and return g_t."""
        if self.initialized:
            self.g = self.rho * self.g + math.sqrt(1.0 - self.rho**2) * epsilon
        else:
            self.g = epsilon
            self.initialized = True
        return self.g


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

    Successive coefficients are temporally correlated when tau_large /
    tau_small are non-zero: each Gamma component follows a Gaussian copula
    AR(1) process (latent normal g_t = ρ·g_{t-1} + √(1-ρ²)·ε_t mapped through
    Φ and the Gamma quantile), which preserves the exact Gamma-Gamma marginal
    while making the coherence tunable. The AR(1) step happens once per
    coherence window, so Δt = coherence_samples / sample_rate and
    ρ = exp(−Δt/τ) per component. τ = 0 (the default) draws that component
    i.i.d. per window — exactly the pre-Phase-6 behaviour.

    Args:
        C2n: Refractive index structure parameter [m^(-2/3)]. Must be > 0.
            Typical: 1e-17 (weak) to 1e-13 (strong).
        wavelength: Optical carrier wavelength [m]. Default 1550 nm.
        distance: Link distance [m]. Default 1000 m.
        coherence_samples: Samples per fading coefficient. Must be ≥ 1.
            Set to sample_rate × coherence_time (e.g. 1 MHz × 1 ms = 1000).
        seed: Seed for the NumPy random generator. None → non-reproducible.
        tau_large: Coherence time [s] of the large-scale (alpha) component.
            Must be ≥ 0; 0 → i.i.d. window draws (backward compatible).
        tau_small: Coherence time [s] of the small-scale (beta) component.
            Must be ≥ 0; 0 → i.i.d. window draws (backward compatible).
        sample_rate: Baseband sample rate [Hz]. Must be > 0. Only used to
            convert the window duration into Δt for ρ = exp(−Δt/τ).
    """

    def __init__(
        self,
        C2n: float = 1e-15,
        wavelength: float = 1550e-9,
        distance: float = 1000.0,
        coherence_samples: int = 1000,
        seed: Optional[int] = None,
        tau_large: float = 0.0,
        tau_small: float = 0.0,
        sample_rate: float = 1e6,
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
        if tau_large < 0:
            raise ValueError(f"tau_large must be non-negative, got {tau_large}")
        if tau_small < 0:
            raise ValueError(f"tau_small must be non-negative, got {tau_small}")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")

        self.alpha, self.beta = alpha_beta_from_rytov(
            rytov_variance(C2n, wavelength, distance)
        )
        self._coherence_samples = int(coherence_samples)
        self._rng = np.random.default_rng(seed)
        self._gain = np.float32(0.0)
        self._remaining = 0  # samples left in the current coherence window

        # One AR(1) step per coherence window: Δt = window duration
        dt = self._coherence_samples / sample_rate
        self._state_large = self._make_state(dt, tau_large)
        self._state_small = self._make_state(dt, tau_small)

    @staticmethod
    def _make_state(dt: float, tau: float) -> Optional[_CopulaAr1State]:
        """Build the latent AR(1) state for one component; None → i.i.d. path."""
        if tau == 0.0:
            return None
        return _CopulaAr1State(rho=math.exp(-dt / tau))

    def _next_component(
        self, shape: float, state: Optional[_CopulaAr1State]
    ) -> float:
        """Draw one Gamma(shape, 1/shape) component for a new coherence window.

        With no AR(1) state the draw is i.i.d. (identical to the pre-Phase-6
        block, RNG stream included). Otherwise the latent normal advances one
        AR(1) step and is mapped through the Gaussian copula, u = Φ(g_t) and
        F⁻¹_Gamma(u; shape, 1/shape), so the marginal stays exact.
        """
        if state is None:
            return float(self._rng.gamma(shape=shape, scale=1.0 / shape))
        g = state.step(float(self._rng.standard_normal()))
        # Clip guards against Φ(g) rounding to exactly 0 or 1
        u = min(max(0.5 * math.erfc(-g / math.sqrt(2.0)), 1e-16), 1.0 - 1e-16)
        return gamma_ppf(shape, u) / shape

    def _next_gain(self) -> np.float32:
        """Draw a fresh amplitude gain √I from the Gamma-Gamma distribution."""
        large_scale = self._next_component(self.alpha, self._state_large)
        small_scale = self._next_component(self.beta, self._state_small)
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
