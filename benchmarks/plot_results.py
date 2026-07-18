"""Publication-quality plots of the benchmark summaries.

Reads a summary CSV (written by parse_traces.py) and renders grouped
bar charts — PDR, mean delay, PHY drops, and episode reward per sweep
point per policy, with +/- 1 std error bars — into results/plots/.
``--study turbulence`` (default) plots the Phase 5 C2n sweep from
results/summary.csv; ``--study correlated`` plots the Phase 6 fading
coherence-time sweep from results/correlated_summary.csv (files prefixed
``correlated_``); ``--study adaptation`` plots the Phase 7
disjoint-topology study from results/adaptation_summary.csv (files
prefixed ``adaptation_``, plus a route-switches chart);
``--study imitation`` plots the Phase 8 study from
results/imitation_summary.csv (prefix ``imitation_``, headline policies
only) plus the fine-tuning trajectory chart — entropy, KL from the BC
policy, and switch rate per update — from
results/imitation_trajectory_<regime>.csv;
``--study offpolicy`` plots the Phase 9 study from
results/offpolicy_summary.csv (prefix ``offpolicy_``) plus its own
trajectory chart — greedy switch rate, Q-gap, and TD loss per update
for all four DQN runs — from
results/offpolicy_trajectory_<arm>_<regime>.csv;
``--study routeaware`` plots the Phase 10 study from
results/routeaware_summary.csv (prefix ``routeaware_``) plus a
trajectory chart — greedy switch rate for all six route-aware training
runs, policy entropy for the PPO fine-tunes, and Q-gap for the DQN
arms — from results/routeaware_trajectory_<arm>_<regime>.csv. Styling
follows prototype/turbulence_plots.py; the categorical palette is
Okabe-Ito-derived and colorblind-validated (adjacent-pair CVD
deltaE >= 12; the Phase 8 additions bc/bc-ppo were validated all-pairs
against every co-plotted policy color; the Phase 9 additions
dqn-scratch/dqn-bc use Paul Tol's colorblind-safe indigo and sand; the
Phase 10 route-aware arms reuse their 28-dim counterpart's hue family
at a clearly darker/lighter value, so arm identity is the hue and
observability is the lightness).

Typical usage:
    $ python plot_results.py
    $ python plot_results.py --study correlated
    $ python plot_results.py --study adaptation
    $ python plot_results.py --study imitation
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
    "bc": "#AA4499",
    "bc-ppo": "#E07126",
    "dqn-scratch": "#332288",
    "dqn-bc": "#DDCC77",
    "bc-route": "#5C1237",
    "bc-ppo-route": "#8C3A0F",
    "dqn-scratch-route": "#9C93E8",
    "dqn-bc-route": "#8A6D00",
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
    "bc": "BC of greedy-PER teacher",
    "bc-ppo": "BC + PPO fine-tune",
    "dqn-scratch": "Double DQN (scratch)",
    "dqn-bc": "Double DQN (BC-init)",
    "bc-route": "BC (route-aware obs)",
    "bc-ppo-route": "BC + PPO fine-tune (route-aware)",
    "dqn-scratch-route": "Double DQN scratch (route-aware)",
    "dqn-bc-route": "Double DQN BC-init (route-aware)",
    "ppo-transfer": "PPO (1e-13 ckpt)",
    "best-static": "Best static route",
    "greedy-per": "Greedy PER (scripted)",
    "random": "Random",
    "aodv": "AODV",
}

# Phase 8 imitation study: the two correlated adaptation cells; bars are
# limited to the policies the phase compares so the groups stay legible
# (the pulled Phase 7 baselines random/aodv/ppo-stack stay in the
# summary CSV and tables).
IMITATION_ORDER = ("disjoint-tau500-udp", "disjoint-tau500-tcp")
IMITATION_POLICIES = ("ppo", "bc", "bc-ppo", "best-static", "greedy-per")

# Fine-tuning trajectory encoding: the entity is the environment config
# (one line per cell), Okabe-Ito blue/vermillion with distinct line
# styles as secondary encoding.
TRAJECTORY_STYLES = {
    "disjoint-tau500-udp": ("#0072B2", "-", "τ 500/100 ms + UDP"),
    "disjoint-tau500-tcp": ("#D55E00", "--", "τ 500/100 ms + TCP"),
}

# Phase 9 off-policy study: same two cells; the four DQN training runs
# are encoded as color = cell (matching TRAJECTORY_STYLES), line style
# = arm (solid scratch, dashed BC-init).
OFFPOLICY_ORDER = IMITATION_ORDER
OFFPOLICY_POLICIES = ("ppo", "bc", "bc-ppo", "dqn-scratch", "dqn-bc",
                      "best-static", "greedy-per")
OFFPOLICY_ARMS = ("dqn-scratch", "dqn-bc")
OFFPOLICY_TRAJECTORY_STYLES = {
    ("dqn-scratch", "disjoint-tau500-udp"):
        ("#0072B2", "-", "UDP, scratch"),
    ("dqn-bc", "disjoint-tau500-udp"):
        ("#0072B2", "--", "UDP, BC-init"),
    ("dqn-scratch", "disjoint-tau500-tcp"):
        ("#D55E00", "-", "TCP, scratch"),
    ("dqn-bc", "disjoint-tau500-tcp"):
        ("#D55E00", "--", "TCP, BC-init"),
}

# Phase 10 route-aware study: same two cells, six training runs. Bars
# pair each route-aware arm with its 28-dim counterpart; the trajectory
# figure encodes color = cell, line style = arm.
ROUTEAWARE_ORDER = IMITATION_ORDER
ROUTEAWARE_PLOT_POLICIES = ("bc", "bc-route", "bc-ppo", "bc-ppo-route",
                            "dqn-scratch", "dqn-scratch-route", "dqn-bc",
                            "dqn-bc-route", "best-static", "greedy-per")
ROUTEAWARE_ARMS = ("bc-ppo-route", "dqn-scratch-route", "dqn-bc-route")
ROUTEAWARE_TRAJECTORY_STYLES = {
    ("bc-ppo-route", "disjoint-tau500-udp"):
        ("#0072B2", "-", "UDP, BC + PPO"),
    ("dqn-bc-route", "disjoint-tau500-udp"):
        ("#0072B2", "--", "UDP, DQN BC-init"),
    ("dqn-scratch-route", "disjoint-tau500-udp"):
        ("#0072B2", ":", "UDP, DQN scratch"),
    ("bc-ppo-route", "disjoint-tau500-tcp"):
        ("#D55E00", "-", "TCP, BC + PPO"),
    ("dqn-bc-route", "disjoint-tau500-tcp"):
        ("#D55E00", "--", "TCP, DQN BC-init"),
    ("dqn-scratch-route", "disjoint-tau500-tcp"):
        ("#D55E00", ":", "TCP, DQN scratch"),
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
    policies: tuple[str, ...] | None = None,
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
        policies: Policies to draw (POLICY_COLORS order); None draws
            every policy present in the table.

    Returns:
        The matplotlib Figure.
    """
    _apply_base_style()
    x_labels = REGIME_LABELS if x_labels is None else x_labels
    policies = [p for p in POLICY_COLORS
                if (policies is None or p in policies)
                and any((r, p) in table for r in x_order)]
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


def load_trajectory(path: str | Path) -> dict[str, list]:
    """Load one fine-tuning trajectory CSV as columns.

    Args:
        path: Path to imitation_trajectory_<regime>.csv (written by
            agent/imitation.py finetune).

    Returns:
        Mapping of column name to list; numeric columns as floats with
        NaN for blanks.
    """
    numeric = ("update", "global_step", "entropy", "value_loss", "policy_loss",
               "approx_kl", "kl_from_bc", "sampled_switches_per_200",
               "greedy_switches_per_200", "mean_episode_reward",
               "epsilon", "td_loss", "mean_q_gap", "max_q_gap")
    columns: dict[str, list] = {}
    with open(path, newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            for key, value in row.items():
                if key in numeric:
                    value = float(value) if value not in ("", None) else np.nan
                columns.setdefault(key, []).append(value)
    return columns


def plot_imitation_trajectory(
    trajectories: dict[str, dict[str, list]],
    filename: str = "imitation_trajectory.png",
    save: bool = True,
) -> plt.Figure:
    """Render the Phase 8 fine-tuning trajectory (the study's result).

    Three stacked panels over PPO updates: policy entropy, KL from the
    frozen BC policy, and the greedy-policy switch rate (per 200-step
    episode). The value-warmup updates are shaded; reference lines mark
    the uniform-policy entropy, the Phase 7c collapse entropy, and the
    teacher's eval switch rate.

    Args:
        trajectories: Mapping of regime name to trajectory columns
            (from :func:`load_trajectory`).
        filename: Output file name under results/plots/.
        save: If True, save the PNG.

    Returns:
        The matplotlib Figure.
    """
    _apply_base_style()
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    ax_entropy, ax_kl, ax_switch = axes

    warmup_end = 0.0
    for regime, cols in trajectories.items():
        color, linestyle, label = TRAJECTORY_STYLES.get(
            regime, ("#0072B2", "-", regime))
        updates = np.asarray(cols["update"], dtype=float)
        phases = cols["phase"]
        warmup_end = max(warmup_end,
                         max((u for u, ph in zip(updates, phases)
                              if ph == "warmup"), default=0.0))
        ax_entropy.plot(updates, cols["entropy"], color=color,
                        linestyle=linestyle, lw=2, label=label)
        ax_kl.plot(updates, cols["kl_from_bc"], color=color,
                   linestyle=linestyle, lw=2, label=label)
        ax_switch.plot(updates, cols["greedy_switches_per_200"], color=color,
                       linestyle=linestyle, lw=2, label=label)

    ax_entropy.axhline(np.log(4), color="#666666", lw=1, linestyle=":")
    ax_entropy.annotate("uniform (ln 4)", xy=(0.99, np.log(4)),
                        xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=9, color="#666666")
    ax_entropy.axhline(0.005, color="#666666", lw=1, linestyle=":")
    ax_entropy.annotate("Phase 7c collapse (0.005)", xy=(0.99, 0.005),
                        xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=9, color="#666666")
    ax_switch.axhline(46.0, color="#666666", lw=1, linestyle=":")
    ax_switch.annotate("teacher (46/ep)", xy=(0.99, 46.0),
                       xycoords=("axes fraction", "data"),
                       ha="right", va="bottom", fontsize=9, color="#666666")

    ax_entropy.set_ylabel("Policy entropy [nats]")
    ax_kl.set_ylabel("KL from BC policy [nats]")
    ax_switch.set_ylabel("Greedy switches / 200 steps")
    ax_switch.set_xlabel("Update (500 env steps each)")
    for ax in axes:
        if warmup_end > 0:
            ax.axvspan(0, warmup_end + 0.5, color="#dddddd", alpha=0.5,
                       zorder=0)
        ax.legend(loc="best", framealpha=0.85)
    ax_entropy.annotate("value warmup\n(policy frozen)",
                        xy=(warmup_end / 2, 0.95),
                        xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=9, color="#444444")
    ax_entropy.set_title("PPO fine-tuning from the BC initialisation — "
                         "does the switching policy survive?")
    fig.tight_layout()

    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        out = PLOTS_DIR / filename
        fig.savefig(out, bbox_inches="tight")
        print(f"[saved] {out}")
    return fig


def plot_offpolicy_trajectory(
    trajectories: dict[tuple[str, str], dict[str, list]],
    filename: str = "offpolicy_trajectory.png",
    save: bool = True,
) -> plt.Figure:
    """Render the Phase 9 DQN training trajectory (the study's result).

    Three stacked panels over training updates (500 env steps each):
    the greedy policy's switch rate (per 200-step episode), the mean
    Q-gap between the best and second-best action, and the TD (Huber)
    loss on a log scale. One line per (arm, cell) training run; a
    reference line marks the teacher's eval switch rate.

    Args:
        trajectories: Mapping of (arm, regime) to trajectory columns
            (from :func:`load_trajectory`).
        filename: Output file name under results/plots/.
        save: If True, save the PNG.

    Returns:
        The matplotlib Figure.
    """
    _apply_base_style()
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    ax_switch, ax_gap, ax_td = axes

    for key, cols in trajectories.items():
        color, linestyle, label = OFFPOLICY_TRAJECTORY_STYLES.get(
            key, ("#0072B2", "-", f"{key[0]} {key[1]}"))
        updates = np.asarray(cols["update"], dtype=float)
        ax_switch.plot(updates, cols["greedy_switches_per_200"], color=color,
                       linestyle=linestyle, lw=2, label=label)
        ax_gap.plot(updates, cols["mean_q_gap"], color=color,
                    linestyle=linestyle, lw=2, label=label)
        ax_td.plot(updates, cols["td_loss"], color=color,
                   linestyle=linestyle, lw=2, label=label)

    ax_switch.axhline(46.0, color="#666666", lw=1, linestyle=":")
    ax_switch.annotate("teacher (46/ep)", xy=(0.99, 46.0),
                       xycoords=("axes fraction", "data"),
                       ha="right", va="bottom", fontsize=9, color="#666666")

    ax_switch.set_ylabel("Greedy switches / 200 steps")
    ax_gap.set_ylabel("Mean Q-gap (best − 2nd) ")
    ax_td.set_ylabel("TD loss (Huber)")
    ax_td.set_yscale("log")
    ax_td.set_xlabel("Update (500 env steps each)")
    for ax in axes:
        ax.legend(loc="best", framealpha=0.85, ncols=2)
    ax_switch.set_title("Double DQN training on the correlated cells — "
                        "does off-policy learning find/keep switching?")
    fig.tight_layout()

    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        out = PLOTS_DIR / filename
        fig.savefig(out, bbox_inches="tight")
        print(f"[saved] {out}")
    return fig


def plot_routeaware_trajectory(
    trajectories: dict[tuple[str, str], dict[str, list]],
    filename: str = "routeaware_trajectory.png",
    save: bool = True,
) -> plt.Figure:
    """Render the Phase 10 route-aware training trajectory.

    Three stacked panels over training updates (500 env steps each):
    the greedy policy's switch rate for all six route-aware runs, the
    policy entropy of the PPO fine-tunes (does the route one-hot stop
    the Phase 8 entropy drain?), and the mean Q-gap of the DQN arms
    (does it stop the Phase 9 TCP gap erasure?). Reference lines mark
    the teacher's eval switch rate and the Phase 7c collapse entropy.

    Args:
        trajectories: Mapping of (arm, regime) to trajectory columns
            (from :func:`load_trajectory`).
        filename: Output file name under results/plots/.
        save: If True, save the PNG.

    Returns:
        The matplotlib Figure.
    """
    _apply_base_style()
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    ax_switch, ax_entropy, ax_gap = axes

    for (arm, regime), cols in trajectories.items():
        color, linestyle, label = ROUTEAWARE_TRAJECTORY_STYLES.get(
            (arm, regime), ("#0072B2", "-", f"{arm} {regime}"))
        updates = np.asarray(cols["update"], dtype=float)
        ax_switch.plot(updates, cols["greedy_switches_per_200"], color=color,
                       linestyle=linestyle, lw=2, label=label)
        if "entropy" in cols:
            ax_entropy.plot(updates, cols["entropy"], color=color,
                            linestyle=linestyle, lw=2, label=label)
        if "mean_q_gap" in cols:
            ax_gap.plot(updates, cols["mean_q_gap"], color=color,
                        linestyle=linestyle, lw=2, label=label)

    ax_switch.axhline(46.0, color="#666666", lw=1, linestyle=":")
    ax_switch.annotate("teacher (46/ep)", xy=(0.99, 46.0),
                       xycoords=("axes fraction", "data"),
                       ha="right", va="bottom", fontsize=9, color="#666666")
    ax_entropy.axhline(0.005, color="#666666", lw=1, linestyle=":")
    ax_entropy.annotate("Phase 7c collapse (0.005)", xy=(0.99, 0.005),
                        xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=9, color="#666666")

    ax_switch.set_ylabel("Greedy switches / 200 steps")
    ax_entropy.set_ylabel("Policy entropy [nats]\n(PPO fine-tunes)")
    ax_gap.set_ylabel("Mean Q-gap (best − 2nd)\n(DQN arms)")
    ax_gap.set_xlabel("Update (500 env steps each)")
    for ax in axes:
        ax.legend(loc="best", framealpha=0.85, ncols=2)
    ax_switch.set_title("Route-aware training on the correlated cells — "
                        "does observability preserve switching?")
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
    parser.add_argument("--study", choices=("turbulence", "correlated",
                                            "adaptation", "imitation",
                                            "offpolicy", "routeaware"),
                        default="turbulence",
                        help="picks the default summary file, x-axis, and "
                             "output file prefix")
    parser.add_argument("--summary", type=str, default=None,
                        help="input summary CSV")
    args = parser.parse_args()

    defaults = {"turbulence": "summary.csv",
                "correlated": "correlated_summary.csv",
                "adaptation": "adaptation_summary.csv",
                "imitation": "imitation_summary.csv",
                "offpolicy": "offpolicy_summary.csv",
                "routeaware": "routeaware_summary.csv"}
    table = load_summary(args.summary or str(RESULTS_DIR / defaults[args.study]))
    if args.study == "routeaware":
        for metric, title, ylabel, filename in (*METRIC_SPECS,
                                                *ADAPTATION_EXTRA_SPECS):
            plot_metric(table, metric, title, ylabel, f"routeaware_{filename}",
                        x_order=ROUTEAWARE_ORDER, x_labels=ADAPTATION_LABELS,
                        x_axis_label="Environment config (disjoint topology, "
                                     "C²ₙ = 1e-13 m⁻²ᐟ³)",
                        subtitle="10 episodes, shared seeds",
                        policies=ROUTEAWARE_PLOT_POLICIES)
        trajectories = {
            (arm, regime): load_trajectory(path)
            for arm in ROUTEAWARE_ARMS
            for regime in ROUTEAWARE_ORDER
            if (path := RESULTS_DIR /
                f"routeaware_trajectory_{arm}_{regime}.csv").exists()
        }
        if trajectories:
            plot_routeaware_trajectory(trajectories)
    elif args.study == "offpolicy":
        for metric, title, ylabel, filename in (*METRIC_SPECS,
                                                *ADAPTATION_EXTRA_SPECS):
            plot_metric(table, metric, title, ylabel, f"offpolicy_{filename}",
                        x_order=OFFPOLICY_ORDER, x_labels=ADAPTATION_LABELS,
                        x_axis_label="Environment config (disjoint topology, "
                                     "C²ₙ = 1e-13 m⁻²ᐟ³)",
                        subtitle="10 episodes, shared seeds",
                        policies=OFFPOLICY_POLICIES)
        trajectories = {
            (arm, regime): load_trajectory(path)
            for arm in OFFPOLICY_ARMS
            for regime in OFFPOLICY_ORDER
            if (path := RESULTS_DIR /
                f"offpolicy_trajectory_{arm}_{regime}.csv").exists()
        }
        if trajectories:
            plot_offpolicy_trajectory(trajectories)
    elif args.study == "imitation":
        for metric, title, ylabel, filename in (*METRIC_SPECS,
                                                *ADAPTATION_EXTRA_SPECS):
            plot_metric(table, metric, title, ylabel, f"imitation_{filename}",
                        x_order=IMITATION_ORDER, x_labels=ADAPTATION_LABELS,
                        x_axis_label="Environment config (disjoint topology, "
                                     "C²ₙ = 1e-13 m⁻²ᐟ³)",
                        subtitle="10 episodes, shared seeds",
                        policies=IMITATION_POLICIES)
        trajectories = {
            regime: load_trajectory(path)
            for regime in IMITATION_ORDER
            if (path := RESULTS_DIR / f"imitation_trajectory_{regime}.csv").exists()
        }
        if trajectories:
            plot_imitation_trajectory(trajectories)
    elif args.study == "correlated":
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
