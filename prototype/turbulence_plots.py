"""
Visualization tools for Gamma-Gamma FSO turbulence analysis.

Generates publication-quality plots saved to prototype/plots/:
    1. fading_traces.png            — time-series irradiance under weak/moderate/strong turbulence
    2. ber_vs_snr.png               — BER vs SNR for fading vs AWGN baseline
    3. scintillation_map.png        — scintillation index across C²_n and link distance
    4. correlated_fading_traces.png — correlated fading for increasing coherence times

Run directly:
    python turbulence_plots.py

All plots are also available as importable functions for use in notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from gamma_gamma import (
    C2N_MODERATE,
    C2N_STRONG,
    C2N_WEAK,
    TurbulenceParams,
    alpha_beta_from_rytov,
    ber_awgn_baseline,
    ber_ook_fading,
    correlated_gamma_gamma_sample,
    empirical_autocorrelation,
    gamma_gamma_sample,
    rytov_variance,
    scintillation_index,
)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

PLOTS_DIR = Path(__file__).parent / "plots"


def _ensure_plots_dir() -> Path:
    """Create the plots/ directory if it doesn't exist."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return PLOTS_DIR


# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------

# Consistent color palette across all plots
REGIME_COLORS = {
    "Weak":     "#2196F3",   # blue
    "Moderate": "#FF9800",   # orange
    "Strong":   "#F44336",   # red
}

REGIME_PARAMS: dict[str, dict] = {
    "Weak":     {"C2n": C2N_WEAK,     "linestyle": "-"},
    "Moderate": {"C2n": C2N_MODERATE, "linestyle": "--"},
    "Strong":   {"C2n": C2N_STRONG,   "linestyle": ":"},
}


def _apply_base_style() -> None:
    """Apply consistent matplotlib style for all plots."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "lines.linewidth": 1.8,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ---------------------------------------------------------------------------
# Plot 1: Fading time-series
# ---------------------------------------------------------------------------


def plot_fading_traces(
    n_samples: int = 2_000,
    distance_m: float = 1_000.0,
    wavelength_m: float = 1550e-9,
    seed: int = 42,
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """Plot normalised irradiance time-series for three turbulence regimes.

    Each subplot shows a realisation of the Gamma-Gamma fading process —
    what the received optical power looks like as a function of time under
    weak, moderate, and strong atmospheric turbulence.

    Args:
        n_samples: Number of time samples per trace.
        distance_m: FSO link distance [m].
        wavelength_m: Optical wavelength [m].
        seed: Random seed for reproducibility.
        save: If True, saves to plots/fading_traces.png.
        show: If True, calls plt.show() (blocks; use in interactive sessions).

    Returns:
        The matplotlib Figure object.
    """
    _apply_base_style()
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        f"Gamma-Gamma Fading Traces  |  L = {distance_m/1e3:.1f} km  |  λ = {wavelength_m*1e9:.0f} nm",
        fontsize=14, y=1.01,
    )

    t = np.arange(n_samples)

    for ax, (label, cfg) in zip(axes, REGIME_PARAMS.items()):
        params = TurbulenceParams(
            C2n=cfg["C2n"],
            wavelength=wavelength_m,
            distance=distance_m,
        )
        alpha, beta = params.alpha_beta()
        si = params.scintillation_index()
        sigma2_R = params.rytov_variance()

        samples = gamma_gamma_sample(alpha, beta, n_samples, rng)

        ax.plot(t, samples, color=REGIME_COLORS[label], lw=0.9, alpha=0.85)
        ax.axhline(1.0, color="black", lw=0.8, linestyle="--", alpha=0.5, label="E[I] = 1")

        ax.set_ylabel("Irradiance  I(t)")
        ax.set_title(
            f"{label} Turbulence  |  C²ₙ = {cfg['C2n']:.0e}  |  "
            f"α={alpha:.2f}, β={beta:.2f}  |  σ²_R={sigma2_R:.3f}  |  SI={si:.4f}",
            fontsize=10,
        )
        ax.legend(loc="upper right", framealpha=0.7)

    axes[-1].set_xlabel("Sample index (arbitrary time units)")
    fig.tight_layout()

    if save:
        out = _ensure_plots_dir() / "fading_traces.png"
        fig.savefig(out, bbox_inches="tight")
        print(f"[saved] {out}")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Plot 2: BER vs SNR
# ---------------------------------------------------------------------------


def plot_ber_vs_snr(
    snr_db_min: float = -5.0,
    snr_db_max: float = 40.0,
    n_snr_points: int = 60,
    distance_m: float = 1_000.0,
    wavelength_m: float = 1550e-9,
    n_mc_samples: int = 200_000,
    seed: int = 42,
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """Plot average BER vs SNR for AWGN baseline and three fading regimes.

    Demonstrates the SNR penalty imposed by atmospheric turbulence relative
    to the ideal AWGN case. For a target BER of 10⁻³, the gap between the
    AWGN curve and the strong-turbulence curve shows how many dB of extra
    transmitted power would be needed to compensate for the channel.

    Args:
        snr_db_min: Minimum SNR [dB] for x-axis.
        snr_db_max: Maximum SNR [dB] for x-axis.
        n_snr_points: Number of SNR evaluation points.
        distance_m: FSO link distance [m].
        wavelength_m: Optical wavelength [m].
        n_mc_samples: Monte Carlo samples for BER averaging.
        seed: Random seed for reproducibility.
        save: If True, saves to plots/ber_vs_snr.png.
        show: If True, calls plt.show().

    Returns:
        The matplotlib Figure object.
    """
    _apply_base_style()
    rng = np.random.default_rng(seed)

    snr_db = np.linspace(snr_db_min, snr_db_max, n_snr_points)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(
        f"BER vs SNR — OOK IM/DD  |  L = {distance_m/1e3:.1f} km  |  λ = {wavelength_m*1e9:.0f} nm",
        fontsize=13,
    )

    # AWGN baseline (no fading)
    ber_awgn = ber_awgn_baseline(snr_db)
    ax.semilogy(snr_db, ber_awgn, "k-", lw=2.2, label="AWGN (no fading)", zorder=5)

    # Fading regimes
    for label, cfg in REGIME_PARAMS.items():
        params = TurbulenceParams(
            C2n=cfg["C2n"],
            wavelength=wavelength_m,
            distance=distance_m,
        )
        alpha, beta = params.alpha_beta()
        si = params.scintillation_index()

        print(f"  Computing BER: {label} turbulence (α={alpha:.2f}, β={beta:.2f}, SI={si:.4f})...")
        ber = ber_ook_fading(snr_db, alpha, beta, n_mc_samples, rng)

        ax.semilogy(
            snr_db, ber,
            color=REGIME_COLORS[label],
            linestyle=cfg["linestyle"],
            lw=2.0,
            label=f"{label}  (SI={si:.3f})",
        )

    # Reference BER level lines
    for ber_ref, label_ref in [(1e-3, "10⁻³"), (1e-6, "10⁻⁶")]:
        ax.axhline(ber_ref, color="grey", lw=0.8, linestyle=":", alpha=0.6)
        ax.text(snr_db_max + 0.3, ber_ref, f"BER={label_ref}", va="center", fontsize=8, color="grey")

    ax.set_xlabel("Average SNR [dB]")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_ylim(1e-7, 0.6)
    ax.set_xlim(snr_db_min, snr_db_max)
    ax.legend(loc="upper right", framealpha=0.85)
    fig.tight_layout()

    if save:
        out = _ensure_plots_dir() / "ber_vs_snr.png"
        fig.savefig(out, bbox_inches="tight")
        print(f"[saved] {out}")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Plot 3: Scintillation index map
# ---------------------------------------------------------------------------


def plot_scintillation_map(
    distances_km: Optional[list[float]] = None,
    c2n_range: Optional[tuple[float, float]] = None,
    wavelength_m: float = 1550e-9,
    n_c2n_points: int = 200,
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """Plot scintillation index (SI) vs C²_n for multiple link distances.

    Provides a quick design reference: for a given deployment scenario
    (link distance + expected turbulence strength), read off the expected SI
    and the corresponding Gamma-Gamma parameters.

    Args:
        distances_km: Link distances [km] to plot. Defaults to [0.5, 1, 2, 5].
        c2n_range: (min, max) C²_n [m^(-2/3)]. Defaults to (1e-18, 1e-12).
        wavelength_m: Optical wavelength [m].
        n_c2n_points: Number of C²_n points on the x-axis.
        save: If True, saves to plots/scintillation_map.png.
        show: If True, calls plt.show().

    Returns:
        The matplotlib Figure object.
    """
    _apply_base_style()

    if distances_km is None:
        distances_km = [0.5, 1.0, 2.0, 5.0]
    if c2n_range is None:
        c2n_range = (1e-18, 1e-12)

    c2n_vals = np.logspace(np.log10(c2n_range[0]), np.log10(c2n_range[1]), n_c2n_points)
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(distances_km)))  # type: ignore[attr-defined]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(
        f"Scintillation Index vs Turbulence Strength  |  λ = {wavelength_m*1e9:.0f} nm",
        fontsize=13,
    )

    for dist_km, color in zip(distances_km, colors):
        si_vals = np.array([
            scintillation_index(*alpha_beta_from_rytov(
                rytov_variance(c2n, wavelength_m, dist_km * 1e3)
            ))
            for c2n in c2n_vals
        ])
        ax.loglog(c2n_vals, si_vals, color=color, label=f"L = {dist_km} km")

    # Turbulence regime annotation bands
    ax.axvspan(c2n_range[0], 5e-17, alpha=0.06, color="blue",   label="Weak regime")
    ax.axvspan(5e-16, 5e-14,        alpha=0.06, color="orange", label="Moderate regime")
    ax.axvspan(5e-14, c2n_range[1], alpha=0.06, color="red",    label="Strong regime")

    ax.set_xlabel("C²ₙ  [m^(-2/3)]")
    ax.set_ylabel("Scintillation Index  SI")
    ax.set_xlim(*c2n_range)
    ax.legend(loc="upper left", framealpha=0.85)
    fig.tight_layout()

    if save:
        out = _ensure_plots_dir() / "scintillation_map.png"
        fig.savefig(out, bbox_inches="tight")
        print(f"[saved] {out}")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Plot 4: Correlated fading time-series (Phase 6a)
# ---------------------------------------------------------------------------


def plot_correlated_fading_traces(
    n_samples: int = 2_000,
    dt_s: float = 1e-3,
    tau_large_values_s: Optional[list[float]] = None,
    distance_m: float = 1_000.0,
    wavelength_m: float = 1550e-9,
    seed: int = 42,
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """Plot correlated Gamma-Gamma sample paths for increasing coherence times.

    Same strong-turbulence link and same seed in every panel — only the
    coherence time changes, so the growing temporal memory is directly
    visible: τ = 0 is white Gamma-Gamma noise, larger τ produces fades that
    persist across many samples (what the RL router can learn to exploit).
    The small-scale coherence time is set to τ_large / 5, reflecting the
    faster evolution of small eddies.

    Args:
        n_samples: Number of time samples per trace.
        dt_s: Sample interval [s]. Default 1 ms.
        tau_large_values_s: Large-scale coherence times [s] to compare.
            Defaults to [0 (i.i.d.), 10 ms, 100 ms].
        distance_m: FSO link distance [m].
        wavelength_m: Optical wavelength [m].
        seed: Random seed, reused for every panel (same underlying draws).
        save: If True, saves to plots/correlated_fading_traces.png.
        show: If True, calls plt.show().

    Returns:
        The matplotlib Figure object.
    """
    _apply_base_style()

    if tau_large_values_s is None:
        tau_large_values_s = [0.0, 10e-3, 100e-3]

    params = TurbulenceParams(
        C2n=C2N_STRONG, wavelength=wavelength_m, distance=distance_m
    )
    alpha, beta = params.alpha_beta()
    si = params.scintillation_index()

    fig, axes = plt.subplots(
        len(tau_large_values_s), 1, figsize=(10, 8), sharex=True, sharey=True
    )
    fig.suptitle(
        f"Correlated Gamma-Gamma Fading — Strong Turbulence  |  "
        f"α={alpha:.2f}, β={beta:.2f}, SI={si:.3f}  |  Δt = {dt_s*1e3:.0f} ms",
        fontsize=14, y=1.01,
    )

    t_ms = np.arange(n_samples) * dt_s * 1e3
    colors = plt.cm.viridis(np.linspace(0.15, 0.75, len(tau_large_values_s)))  # type: ignore[attr-defined]

    for ax, tau_large, color in zip(np.atleast_1d(axes), tau_large_values_s, colors):
        tau_small = tau_large / 5.0  # small eddies decorrelate faster
        rng = np.random.default_rng(seed)  # same seed → same underlying draws
        series = correlated_gamma_gamma_sample(
            alpha, beta, n_samples, dt_s, tau_large, tau_small, rng
        )
        r1 = empirical_autocorrelation(series, lag=1)

        ax.plot(t_ms, series, color=color, lw=0.9, alpha=0.9)
        ax.axhline(1.0, color="black", lw=0.8, linestyle="--", alpha=0.5, label="E[I] = 1")

        if tau_large == 0.0:
            tau_label = "τ_L = 0 (i.i.d.)"
        else:
            tau_label = (
                f"τ_L = {tau_large*1e3:.0f} ms, τ_S = {tau_small*1e3:.0f} ms"
            )
        ax.set_ylabel("Irradiance  I(t)")
        ax.set_title(f"{tau_label}  |  lag-1 autocorr r₁ = {r1:.3f}", fontsize=10)
        ax.legend(loc="upper right", framealpha=0.7)

    np.atleast_1d(axes)[-1].set_xlabel("Time  [ms]")
    fig.tight_layout()

    if save:
        out = _ensure_plots_dir() / "correlated_fading_traces.png"
        fig.savefig(out, bbox_inches="tight")
        print(f"[saved] {out}")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("Generating FSO turbulence plots...")
    print("\n[1/4] Fading traces")
    plot_fading_traces()
    print("\n[2/4] BER vs SNR")
    plot_ber_vs_snr()
    print("\n[3/4] Scintillation map")
    plot_scintillation_map()
    print("\n[4/4] Correlated fading traces")
    plot_correlated_fading_traces()
    print("\nDone. Plots written to prototype/plots/")
