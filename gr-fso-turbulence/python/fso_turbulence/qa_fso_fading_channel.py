"""QA tests for the fso_fading_channel block.

Runs the block inside a real GNU Radio flowgraph
(vector_source_c → fso_fading_channel → vector_sink_c) and validates
its statistics against the Gamma-Gamma theory used in the prototype.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
from gnuradio import blocks, gr, gr_unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fso_turbulence import fso_fading_channel, gamma_ppf, scintillation_index

# scipy.stats.gamma.ppf(u, shape) reference values, generated once with
# scipy 1.x (prototype/.venv) so the runtime stays scipy-free.
# Grid: shape 0.5–100 x u from 1e-6 to 1-1e-6. Format: (shape, u, x).
GAMMA_PPF_REFERENCE = [
    (0.5, 1e-06, 7.853981633978593e-13),
    (0.5, 0.01, 7.854392895485092e-05),
    (0.5, 0.1, 0.00789538704671561),
    (0.5, 0.5, 0.227468211559786),
    (0.5, 0.9, 1.352771727047709),
    (0.5, 0.99, 3.3174483005106077),
    (0.5, 0.999999, 11.964063488439734),
    (1.0, 1e-06, 1.0000005000003338e-06),
    (1.0, 0.01, 0.010050335853501437),
    (1.0, 0.1, 0.10536051565782636),
    (1.0, 0.5, 0.6931471805599455),
    (1.0, 0.9, 2.302585092994046),
    (1.0, 0.99, 4.60517018598809),
    (1.0, 0.999999, 13.815510557935518),
    (2.0, 1e-06, 0.0014148806614793436),
    (2.0, 0.01, 0.14855474025326595),
    (2.0, 0.1, 0.5318116083896121),
    (2.0, 0.5, 1.6783469900166612),
    (2.0, 0.9, 3.889720169867429),
    (2.0, 0.99, 6.638352067993811),
    (2.0, 0.999999, 16.68842079082944),
    (4.0, 1e-06, 0.07099239135862108),
    (4.0, 0.01, 0.8232486863453852),
    (4.0, 0.1, 1.7447695628249114),
    (4.0, 0.5, 3.672060748850897),
    (4.0, 0.9, 6.680783068255864),
    (4.0, 0.99, 10.045117514831617),
    (4.0, 0.999999, 21.350456963238944),
    (10.0, 1e-06, 1.276818787864408),
    (10.0, 0.01, 4.130199166273199),
    (10.0, 0.1, 6.221304605225031),
    (10.0, 0.5, 9.66871461471413),
    (10.0, 0.9, 14.205990292152817),
    (10.0, 0.99, 18.78311739331253),
    (10.0, 0.999999, 32.71034051748481),
    (50.0, 1e-06, 23.250665357946566),
    (50.0, 0.01, 35.03244746269989),
    (50.0, 0.1, 41.179067906178574),
    (50.0, 0.5, 49.66706461799423),
    (50.0, 0.9, 59.24900190553106),
    (50.0, 0.99, 67.90336158551338),
    (50.0, 0.999999, 91.06338855971312),
    (100.0, 1e-06, 59.43632069812289),
    (100.0, 0.01, 78.21598305379582),
    (100.0, 0.1, 87.41763649959366),
    (100.0, 0.5, 99.66686491931549),
    (100.0, 0.9, 113.01052385984448),
    (100.0, 0.99, 124.7225614907208),
    (100.0, 0.999999, 154.91904599496146),
]

# One AR(1) step per coherence window at dt = 1 ms in the correlated tests
WINDOW_DT = 1e-3
TAU_RHO_05 = WINDOW_DT / math.log(2.0)  # rho = exp(-dt/tau) = 0.5
TAU_RHO_09 = -WINDOW_DT / math.log(0.9)  # rho = 0.9


def run_flowgraph(
    data: np.ndarray,
    C2n: float = 1e-15,
    coherence_samples: int = 1,
    seed: int = 42,
    **kwargs,
) -> np.ndarray:
    """Push data through vector_source → fading block → vector_sink.

    Args:
        data: Complex input samples.
        C2n: Turbulence strength [m^(-2/3)].
        coherence_samples: Samples per fading coefficient.
        seed: RNG seed for reproducibility.
        **kwargs: Extra fso_fading_channel constructor arguments.

    Returns:
        Complex output samples from the sink.
    """
    tb = gr.top_block()
    src = blocks.vector_source_c(data.tolist(), repeat=False)
    channel = fso_fading_channel(
        C2n=C2n, coherence_samples=coherence_samples, seed=seed, **kwargs
    )
    sink = blocks.vector_sink_c()
    tb.connect(src, channel, sink)
    tb.run()
    return np.asarray(sink.data())


def window_gains(out: np.ndarray, coherence_samples: int) -> np.ndarray:
    """Extract the per-window power gains I from an all-ones-input run."""
    return np.abs(out[::coherence_samples]) ** 2


def lag1_autocorrelation(samples: np.ndarray) -> float:
    """Compute the lag-1 autocorrelation of a series (prototype convention)."""
    centered = samples - samples.mean()
    return float(np.sum(centered[:-1] * centered[1:]) / np.sum(centered**2))


class qa_fso_fading_channel(gr_unittest.TestCase):

    def test_mean_power_ratio_is_unity(self):
        """E[I] = 1, so mean output power ≈ mean input power."""
        n = 500_000
        data = np.ones(n, dtype=np.complex64)
        out = run_flowgraph(data, C2n=1e-15, coherence_samples=1, seed=7)
        power_ratio = np.mean(np.abs(out) ** 2) / np.mean(np.abs(data) ** 2)
        self.assertAlmostEqual(power_ratio, 1.0, delta=0.02)

    def test_scintillation_index_matches_theory(self):
        """Empirical SI of power gains matches SI = 1/α + 1/β + 1/(αβ)."""
        n = 500_000
        data = np.ones(n, dtype=np.complex64)
        channel = fso_fading_channel(C2n=1e-15, coherence_samples=1, seed=123)
        si_theory = scintillation_index(channel.alpha, channel.beta)

        out = run_flowgraph(data, C2n=1e-15, coherence_samples=1, seed=123)
        gains = np.abs(out) ** 2
        si_empirical = np.mean(gains ** 2) / np.mean(gains) ** 2 - 1.0

        self.assertLess(abs(si_empirical - si_theory) / si_theory, 0.05)

    def test_gain_held_within_coherence_window(self):
        """The fading coefficient is constant within each coherence window."""
        coherence = 64
        n_windows = 200
        data = np.ones(coherence * n_windows, dtype=np.complex64)
        out = run_flowgraph(data, coherence_samples=coherence, seed=5)

        windows = np.abs(out).reshape(n_windows, coherence)
        self.assertTrue(np.all(np.ptp(windows, axis=1) < 1e-6))
        # Sanity check: gains actually vary across windows.
        self.assertGreater(np.ptp(windows[:, 0]), 1e-3)

    def test_same_seed_gives_identical_output(self):
        """Reproducibility: identical seed → identical fading realization."""
        rng = np.random.default_rng(0)
        data = (rng.normal(size=10_000) + 1j * rng.normal(size=10_000)).astype(
            np.complex64
        )
        out_a = run_flowgraph(data, coherence_samples=100, seed=99)
        out_b = run_flowgraph(data, coherence_samples=100, seed=99)
        np.testing.assert_array_equal(out_a, out_b)

    def test_weak_turbulence_is_near_unity(self):
        """C2n = 1e-17 → near-unity gain (SI ~ 1e-4)."""
        n = 100_000
        data = np.ones(n, dtype=np.complex64)
        out = run_flowgraph(data, C2n=1e-17, coherence_samples=1, seed=11)
        gains = np.abs(out) ** 2
        self.assertTrue(np.all(np.abs(gains - 1.0) < 0.15))
        si_empirical = np.mean(gains ** 2) / np.mean(gains) ** 2 - 1.0
        self.assertLess(si_empirical, 1e-3)

    def test_rejects_invalid_parameters(self):
        """Constructor validates physical parameters."""
        for bad_kwargs in (
            {"C2n": 0.0},
            {"C2n": -1e-15},
            {"wavelength": 0.0},
            {"distance": -1.0},
            {"coherence_samples": 0},
        ):
            with self.assertRaises(ValueError):
                fso_fading_channel(**bad_kwargs)

    def test_gamma_ppf_matches_scipy_reference(self):
        """gamma_ppf agrees with hardcoded scipy.stats.gamma.ppf values."""
        max_rel_err = 0.0
        for shape, u, x_ref in GAMMA_PPF_REFERENCE:
            x = gamma_ppf(shape, u)
            max_rel_err = max(max_rel_err, abs(x - x_ref) / x_ref)
        self.assertLess(max_rel_err, 1e-6)
        # Vectorized path returns the same values as the scalar path.
        shapes_05 = [row for row in GAMMA_PPF_REFERENCE if row[0] == 0.5]
        us = np.array([u for _, u, _ in shapes_05])
        refs = np.array([x for _, _, x in shapes_05])
        np.testing.assert_allclose(gamma_ppf(0.5, us), refs, rtol=1e-6)

    def test_correlated_marginal_preserved(self):
        """The copula AR(1) keeps E[I] = 1 and the closed-form SI intact."""
        coherence = 4
        n_windows = 100_000
        data = np.ones(coherence * n_windows, dtype=np.complex64)
        channel = fso_fading_channel(C2n=1e-13, coherence_samples=coherence, seed=42)
        si_theory = scintillation_index(channel.alpha, channel.beta)

        out = run_flowgraph(
            data,
            C2n=1e-13,
            coherence_samples=coherence,
            seed=42,
            tau_large=TAU_RHO_05,
            tau_small=TAU_RHO_05,
            sample_rate=coherence / WINDOW_DT,
        )
        gains = window_gains(out, coherence)
        si_empirical = np.mean(gains**2) / np.mean(gains) ** 2 - 1.0

        self.assertAlmostEqual(gains.mean(), 1.0, delta=0.03)
        self.assertLess(abs(si_empirical - si_theory) / si_theory, 0.10)

    def test_tau_zero_matches_legacy_sampler(self):
        """tau = 0 reproduces the pre-Phase-6 i.i.d. draws bit-for-bit."""
        coherence = 50
        n_windows = 1_000
        seed = 21
        data = np.ones(coherence * n_windows, dtype=np.complex64)
        out = run_flowgraph(
            data, coherence_samples=coherence, seed=seed,
            tau_large=0.0, tau_small=0.0,
        )

        # Pre-Phase-6 blocks drew large then small per window, one pair at a time
        channel = fso_fading_channel(coherence_samples=coherence, seed=seed)
        rng = np.random.default_rng(seed)
        legacy_amplitude = np.empty(n_windows, dtype=np.float32)
        for i in range(n_windows):
            large = rng.gamma(shape=channel.alpha, scale=1.0 / channel.alpha)
            small = rng.gamma(shape=channel.beta, scale=1.0 / channel.beta)
            legacy_amplitude[i] = np.float32(np.sqrt(large * small))

        np.testing.assert_array_equal(
            np.abs(out[::coherence]).astype(np.float32), legacy_amplitude
        )

    def test_lag1_autocorrelation_scales_with_tau(self):
        """Window gains: r1 ≈ 0 at tau = 0, positive and monotone in tau."""
        n = 30_000
        data = np.ones(n, dtype=np.complex64)
        r1 = {}
        for label, tau in (("zero", 0.0), ("mid", TAU_RHO_05), ("big", TAU_RHO_09)):
            out = run_flowgraph(
                data,
                C2n=1e-13,
                coherence_samples=1,
                seed=7,
                tau_large=tau,
                tau_small=tau,
                sample_rate=1.0 / WINDOW_DT,
            )
            r1[label] = lag1_autocorrelation(window_gains(out, 1))

        self.assertLess(abs(r1["zero"]), 0.02)
        self.assertGreater(r1["mid"], 0.3)
        self.assertGreater(r1["big"], r1["mid"] + 0.2)

    def test_correlated_same_seed_gives_identical_output(self):
        """Reproducibility holds on the correlated path too."""
        rng = np.random.default_rng(3)
        data = (rng.normal(size=20_000) + 1j * rng.normal(size=20_000)).astype(
            np.complex64
        )
        kwargs = dict(
            coherence_samples=100,
            tau_large=TAU_RHO_09,
            tau_small=TAU_RHO_05,
            sample_rate=100 / WINDOW_DT,
        )
        out_a = run_flowgraph(data, seed=99, **kwargs)
        out_b = run_flowgraph(data, seed=99, **kwargs)
        out_c = run_flowgraph(data, seed=100, **kwargs)
        np.testing.assert_array_equal(out_a, out_b)
        self.assertFalse(np.array_equal(out_a, out_c))

    def test_rejects_invalid_correlation_parameters(self):
        """Constructor validates the coherence-time parameters."""
        for bad_kwargs in (
            {"tau_large": -1e-3},
            {"tau_small": -1e-3},
            {"sample_rate": 0.0},
            {"sample_rate": -1e6},
        ):
            with self.assertRaises(ValueError):
                fso_fading_channel(**bad_kwargs)


if __name__ == "__main__":
    gr_unittest.run(qa_fso_fading_channel)
