"""GNU Radio OOT module for Free-Space Optical atmospheric turbulence.

Provides the Gamma-Gamma fading channel block and the supporting
Rytov-variance math (Andrews & Phillips, 2005).
"""

from .fso_fading_channel import (
    alpha_beta_from_rytov,
    fso_fading_channel,
    gamma_ppf,
    rytov_variance,
    scintillation_index,
)

__all__ = [
    "alpha_beta_from_rytov",
    "fso_fading_channel",
    "gamma_ppf",
    "rytov_variance",
    "scintillation_index",
]
