"""QA tests for the fso_fading_channel block.

Runs the block inside a real GNU Radio flowgraph
(vector_source_c → fso_fading_channel → vector_sink_c) and validates
its statistics against the Gamma-Gamma theory used in the prototype.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from gnuradio import blocks, gr, gr_unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fso_turbulence import fso_fading_channel, scintillation_index


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


if __name__ == "__main__":
    gr_unittest.run(qa_fso_fading_channel)
