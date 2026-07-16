"""Publication-quality plots of the benchmark summaries.

Reads a summary CSV (written by parse_traces.py) and renders grouped
bar charts — PDR, mean delay, PHY drops, and episode reward per sweep
point per policy, with +/- 1 std error bars — into results/plots/.
``--study turbulence`` (default) plots the Phase 5 C2n sweep from
results/summary.csv; ``--study correlated`` plots the Phase 6 fading
coherence-time sweep from results/correlated_summary.csv (files prefixed
``correlated_``); ``--study adaptation`` plots the Phase 7
disjoint-topology study from results/adaptation_summary.csv (files
prefixed ``adaptation_``, plus a route-switches chart). Styling follows
prototype/turbulence_plots.py; the categorical palette is
Okabe-Ito-derived and colorblind-validated (adjacent-pair CVD
deltaE >= 12).

Typical usage:
    $ python plot_results.py
    $ python plot_results.py --study correlated
    $ python plot_results.py --study adaptation
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
REGIME_LABELS = {
    "weak": "weak\nC²ₙ = 1e-17 m⁻²ᐟ³",
    "moderate": "moderate\nC²ₙ = 1e-15 m⁻²ᐟ³",
    "strong": "strong\nC²ₙ = 1e-13 m⁻²ᐟ³",
}

# Phase 6 correlated-fading study (strong turbulence throughout)
COHERENCE_ORDER = ("iid", "tau100-20", "tau500-100", "tau500-100-step50")
COHERENCE_LABELS = {
    "iid": "i.i.d.\nτ = 0 (control)",
    "tau100-20": "τ_L/τ_S = 100/20 ms\n50 ms steps",
    "tau500-100": "τ_L/τ_S = 500/100 ms\n100 ms steps",
    "tau500-100-step50": "τ_L/τ_S = 500/100 ms\n50 ms steps",
}

# Phase 7 adaptation study (strong turbulence, disjoint topology)
ADAPTATION_ORDER = ("disjoint-iid-udp", "disjoint-tau500-udp",
                    "disjoint-tau500-tcp")
ADAPTATION_LABELS = {
    "disjoint-iid-udp": "i.i.d. + UDP\n(control)",
    "disjoint-tau500-udp": "τ_L/τ_S = 500/100 ms + UDP\n50 ms steps",
    "disjoint-tau500-tcp": "τ_L/τ_S = 500/100 ms + TCP\n50 ms steps",
}

# Fixed policy -> color assignment (identity encoding, never re-ranked)
POLICY_COLORS = {
    "ppo": "#0072B2",
    "ppo-per": "#D55E00",
    "ppo-per-ent": "#F0E442",
    "ppo-stack": "#000000",
    "ppo-transfer": "#56B4E9",
    "best-static": "#009E73",
    "greedy-per": "#999999",
    "random": "#E69F00",
    "aodv": "#CC79A7",
}

POLICY_LABELS = {
    "ppo": "PPO (per-regime)",
    "ppo-per": "PPO (PER observation)",
    "ppo-per-ent": "PPO (PER obs, 160k, ent 0.03)",
    "ppo-stack": "PPO (8-frame stack, 160k)",
    "ppo-transfer": "PPO (1e-13 ckpt)",
    "best-static": "Best static route",
    "greedy-per": "Greedy PER (scripted)",
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

# Extra charts for --study adaptation (columns absent from older summaries)
ADAPTATION_EXTRA_SPECS = (
    ("switches", "Within-episode route switches",
     "Route switches per episode", "switches.png"),
    ("goodput_mbps", "Application goodput (TCP cells)",
     "Mean goodput [Mbps] (higher is better)", "goodput.png"),
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
    x_order: tuple[str, ...] = REGIME_ORDER,
    x_labels: dict[str, str] | None = None,
    x_axis_label: str = "Turbulence regime",
    subtitle: str = "10 episodes, shared seeds",
) -> plt.Figure:
    """Render one grouped bar chart (sweep points on x, one bar per policy).

    Args:
        table: Summary rows from :func:`load_summary`.
        metric: Metric prefix in the summary columns (e.g. "pdr").
        title: Figure title.
        ylabel: y-axis label (states the better direction).
        filename: Output file name under results/plots/.
        save: If True, save the PNG.
        x_order: Sweep point (``regime`` column) display order.
        x_labels: Sweep point tick labels; defaults to REGIME_LABELS.
        x_axis_label: x-axis title.
        subtitle: Trailing fragment of the figure title.

    Returns:
        The matplotlib Figure.
    """
    _apply_base_style()
    x_labels = REGIME_LABELS if x_labels is None else x_labels
    policies = [p for p in POLICY_COLORS
                if any((r, p) in table for r in x_order)]
    regimes = [r for r in x_order
               if any((r, p) in table for p in policies)]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    n = len(policies)
    group_width = 0.82
    bar_width = group_width / n
    x = np.arange(len(regimes))

    nonnegative = metric != "reward"
    for i, policy in enumerate(policies):
        offsets = x - group_width / 2 + (i + 0.5) * bar_width
        means = np.array([table[(r, policy)][f"{metric}_mean"]
                          if (r, policy) in table else np.nan for r in regimes])
        stds = np.array([table[(r, policy)][f"{metric}_std"]
                         if (r, policy) in table else 0.0 for r in regimes])
        # Whiskers on nonnegative metrics must not cross zero
        lower = np.minimum(stds, means) if nonnegative else stds
        ax.bar(offsets, means, width=bar_width * 0.9,
               color=POLICY_COLORS[policy], label=POLICY_LABELS[policy],
               yerr=np.vstack([lower, stds]),
               error_kw={"ecolor": "#333333", "capsize": 3,
                         "capthick": 1.0, "elinewidth": 1.0},
               zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([x_labels.get(r, r) for r in regimes])
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} — 0→3 flow, 5-node FSO mesh, {subtitle}")
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
    parser.add_argument("--study", choices=("turbulence", "correlated", "adaptation"),
                        default="turbulence",
                        help="picks the default summary file, x-axis, and "
                             "output file prefix")
    parser.add_argument("--summary", type=str, default=None,
                        help="input summary CSV")
    args = parser.parse_args()

    defaults = {"turbulence": "summary.csv",
                "correlated": "correlated_summary.csv",
                "adaptation": "adaptation_summary.csv"}
    table = load_summary(args.summary or str(RESULTS_DIR / defaults[args.study]))
    if args.study == "correlated":
        for metric, title, ylabel, filename in METRIC_SPECS:
            plot_metric(table, metric, title, ylabel, f"correlated_{filename}",
                        x_order=COHERENCE_ORDER, x_labels=COHERENCE_LABELS,
                        x_axis_label="Fading coherence time "
                                     "(strong turbulence, C²ₙ = 1e-13 m⁻²ᐟ³)",
                        subtitle="10 episodes, shared seeds")
    elif args.study == "adaptation":
        for metric, title, ylabel, filename in (*METRIC_SPECS,
                                                *ADAPTATION_EXTRA_SPECS):
            plot_metric(table, metric, title, ylabel, f"adaptation_{filename}",
                        x_order=ADAPTATION_ORDER, x_labels=ADAPTATION_LABELS,
                        x_axis_label="Environment config (disjoint topology, "
                                     "C²ₙ = 1e-13 m⁻²ᐟ³)",
                        subtitle="10 episodes, shared seeds")
    else:
        for metric, title, ylabel, filename in METRIC_SPECS:
            plot_metric(table, metric, title, ylabel, filename)
    plt.close("all")


if __name__ == "__main__":
    main()
