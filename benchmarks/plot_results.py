"""Publication-quality plots of the Phase 5 benchmark summary.

Reads results/summary.csv (written by parse_traces.py) and renders four
grouped bar charts — PDR, mean delay, PHY drops, and episode reward per
turbulence regime per policy, with +/- 1 std error bars — into
results/plots/. Styling follows prototype/turbulence_plots.py; the
categorical palette is Okabe-Ito-derived and colorblind-validated
(adjacent-pair CVD deltaE >= 12).

Typical usage:
    $ python plot_results.py
    $ python plot_results.py --summary path/to/summary.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

REGIME_ORDER = ("weak", "moderate", "strong")
REGIME_C2N = {"weak": "1e-17", "moderate": "1e-15", "strong": "1e-13"}

# Fixed policy -> color assignment (identity encoding, never re-ranked)
POLICY_COLORS = {
    "ppo": "#0072B2",
    "ppo-transfer": "#56B4E9",
    "best-static": "#009E73",
    "random": "#E69F00",
    "aodv": "#CC79A7",
}

POLICY_LABELS = {
    "ppo": "PPO (per-regime)",
    "ppo-transfer": "PPO (1e-13 ckpt)",
    "best-static": "Best static route",
    "random": "Random",
    "aodv": "AODV",
}

METRIC_SPECS = (
    ("pdr", "Flow packet delivery ratio", "PDR (higher is better)", "pdr.png"),
    ("mean_delay_ms", "Mean end-to-end delay",
     "Mean delay [ms] (lower is better)", "mean_delay.png"),
    ("drops", "PHY packets lost to fading",
     "PHY drops per episode (lower is better)", "phy_drops.png"),
    ("reward", "Episode reward",
     "Episode reward (higher is better)", "reward.png"),
)


def _apply_base_style() -> None:
    """Apply the shared matplotlib style (see prototype/turbulence_plots.py)."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def load_summary(path: str | Path) -> dict[tuple[str, str], dict]:
    """Load summary.csv keyed by (regime, policy).

    Args:
        path: Path to summary.csv.

    Returns:
        Mapping of (regime, policy) to the row dict with float stats.
    """
    table: dict[tuple[str, str], dict] = {}
    with open(path, newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            entry = dict(row)
            for key, value in row.items():
                if key.endswith(("_mean", "_std")):
                    entry[key] = float(value)
            table[(row["regime"], row["policy"])] = entry
    return table


def plot_metric(
    table: dict[tuple[str, str], dict],
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
    save: bool = True,
) -> plt.Figure:
    """Render one grouped bar chart (regimes on x, one bar per policy).

    Args:
        table: Summary rows from :func:`load_summary`.
        metric: Metric prefix in the summary columns (e.g. "pdr").
        title: Figure title.
        ylabel: y-axis label (states the better direction).
        filename: Output file name under results/plots/.
        save: If True, save the PNG.

    Returns:
        The matplotlib Figure.
    """
    _apply_base_style()
    policies = [p for p in POLICY_COLORS
                if any((r, p) in table for r in REGIME_ORDER)]
    regimes = [r for r in REGIME_ORDER
               if any((r, p) in table for p in policies)]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    n = len(policies)
    group_width = 0.82
    bar_width = group_width / n
    x = np.arange(len(regimes))

    for i, policy in enumerate(policies):
        offsets = x - group_width / 2 + (i + 0.5) * bar_width
        means = [table[(r, policy)][f"{metric}_mean"]
                 if (r, policy) in table else np.nan for r in regimes]
        stds = [table[(r, policy)][f"{metric}_std"]
                if (r, policy) in table else 0.0 for r in regimes]
        ax.bar(offsets, means, width=bar_width * 0.9,
               color=POLICY_COLORS[policy], label=POLICY_LABELS[policy],
               yerr=stds, error_kw={"ecolor": "#333333", "capsize": 3,
                                    "capthick": 1.0, "elinewidth": 1.0},
               zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\nC²ₙ = {REGIME_C2N.get(r, '?')} m⁻²ᐟ³"
                        for r in regimes])
    ax.set_xlabel("Turbulence regime")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} — 0→3 flow, 5-node FSO mesh, "
                 "10 episodes, shared seeds")
    ax.axhline(0.0, color="black", lw=0.8, zorder=4)
    ax.legend(loc="best", framealpha=0.85, ncols=2)
    fig.tight_layout()

    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        out = PLOTS_DIR / filename
        fig.savefig(out, bbox_inches="tight")
        print(f"[saved] {out}")
    return fig


def main() -> None:
    """CLI entry point: render all four benchmark charts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=str,
                        default=str(RESULTS_DIR / "summary.csv"),
                        help="input summary CSV")
    args = parser.parse_args()

    table = load_summary(args.summary)
    for metric, title, ylabel, filename in METRIC_SPECS:
        plot_metric(table, metric, title, ylabel, filename)
    plt.close("all")


if __name__ == "__main__":
    main()
