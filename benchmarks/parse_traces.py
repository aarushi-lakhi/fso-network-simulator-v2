"""Aggregate raw benchmark episodes into a policy x regime summary table.

Reads results/raw_results.csv (written by run_benchmark.py), computes
mean +/- std of reward, PHY drops, PDR, and mean delay per (regime,
policy), collapses the four fixed routes into a single ``best-static``
policy (the route with the highest mean episode reward per regime),
prints the table, and writes results/summary.csv.

Typical usage:
    $ python parse_traces.py
    $ python parse_traces.py --raw path/to/raw.csv --out path/to/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

REGIME_ORDER = ("weak", "moderate", "strong")
POLICY_ORDER = ("ppo", "ppo-transfer", "best-static", "static-0", "static-1",
                "static-2", "static-3", "random", "aodv")

METRICS = ("reward", "drops", "pdr", "mean_delay_ms")

SUMMARY_FIELDS = ("regime", "policy", "detail", "n_episodes",
                  *(f"{m}_{s}" for m in METRICS for s in ("mean", "std")))

# Policies shown in the printed table and the plots; individual static
# routes stay in summary.csv for reference.
HEADLINE_POLICIES = ("ppo", "ppo-transfer", "best-static", "random", "aodv")


def load_raw(path: str | Path) -> list[dict]:
    """Load raw episode rows with numeric fields converted.

    Args:
        path: Path to raw_results.csv.

    Returns:
        List of row dicts; reward/pdr/mean_delay_ms as float, counts as int.

    Raises:
        FileNotFoundError: If the CSV does not exist.
    """
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as fp:
        for raw in csv.DictReader(fp):
            row = dict(raw)
            for key in ("reward", "pdr", "mean_delay_ms"):
                row[key] = float(raw[key])
            for key in ("episode", "sim_seed", "drops", "tx_pkts", "rx_pkts"):
                row[key] = int(raw[key])
            rows.append(row)
    return rows


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Compute mean and population standard deviation.

    Args:
        values: Non-empty list of samples.

    Returns:
        Tuple (mean, std); std is 0 for a single sample.
    """
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def summarize(rows: list[dict]) -> list[dict]:
    """Aggregate episode rows into per-(regime, policy) summary rows.

    Fixed routes additionally produce a ``best-static`` row per regime:
    the static route with the highest mean episode reward, with the
    winning route recorded in the ``detail`` column.

    Args:
        rows: Typed raw rows from :func:`load_raw`.

    Returns:
        Summary row dicts (SUMMARY_FIELDS keys) in canonical order.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["regime"], row["policy"]), []).append(row)

    summary: dict[tuple[str, str], dict] = {}
    for (regime, policy), episodes in groups.items():
        entry: dict = {"regime": regime, "policy": policy, "detail": "",
                       "n_episodes": len(episodes)}
        for metric in METRICS:
            mean, std = _mean_std([float(e[metric]) for e in episodes])
            entry[f"{metric}_mean"] = mean
            entry[f"{metric}_std"] = std
        summary[(regime, policy)] = entry

    for regime in {r["regime"] for r in rows}:
        statics = [e for (reg, pol), e in summary.items()
                   if reg == regime and pol.startswith("static-")]
        if not statics:
            continue
        best = max(statics, key=lambda e: e["reward_mean"])
        best_row = dict(best)
        best_row["policy"] = "best-static"
        best_row["detail"] = f"route={best['policy'].split('-', 1)[1]}"
        summary[(regime, "best-static")] = best_row

    def order(entry: dict) -> tuple[int, int]:
        regime_idx = (REGIME_ORDER.index(entry["regime"])
                      if entry["regime"] in REGIME_ORDER else 99)
        policy_idx = (POLICY_ORDER.index(entry["policy"])
                      if entry["policy"] in POLICY_ORDER else 99)
        return regime_idx, policy_idx

    return sorted(summary.values(), key=order)


def format_table(summary: list[dict], policies: tuple[str, ...] = HEADLINE_POLICIES) -> str:
    """Render summary rows as an aligned text table.

    Args:
        summary: Rows from :func:`summarize`.
        policies: Policies to include, in POLICY_ORDER precedence.

    Returns:
        Multi-line table string.
    """
    header = (f"{'regime':10s} {'policy':14s} {'n':>3s} "
              f"{'reward':>19s} {'PDR':>15s} {'delay [ms]':>15s} {'PHY drops':>17s}")
    lines = [header, "-" * len(header)]
    for entry in summary:
        if entry["policy"] not in policies:
            continue
        name = entry["policy"]
        if entry["detail"]:
            name = f"{name} ({entry['detail']})"
        lines.append(
            f"{entry['regime']:10s} {name:14s} {entry['n_episodes']:3d} "
            f"{entry['reward_mean']:10.1f} +/- {entry['reward_std']:5.1f} "
            f"{entry['pdr_mean']:7.3f} +/- {entry['pdr_std']:5.3f} "
            f"{entry['mean_delay_ms_mean']:7.3f} +/- {entry['mean_delay_ms_std']:5.3f} "
            f"{entry['drops_mean']:9.1f} +/- {entry['drops_std']:5.1f}"
        )
    return "\n".join(lines)


def write_summary(path: str | Path, summary: list[dict]) -> None:
    """Write summary rows to CSV.

    Args:
        path: Output CSV path.
        summary: Rows from :func:`summarize`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for entry in summary:
            row = dict(entry)
            for metric in METRICS:
                for stat in ("mean", "std"):
                    row[f"{metric}_{stat}"] = f"{entry[f'{metric}_{stat}']:.6g}"
            writer.writerow(row)


def main() -> None:
    """CLI entry point: aggregate the raw CSV and print the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=str, default=str(RESULTS_DIR / "raw_results.csv"),
                        help="input raw episode CSV")
    parser.add_argument("--out", type=str, default=str(RESULTS_DIR / "summary.csv"),
                        help="output summary CSV")
    parser.add_argument("--all-policies", action="store_true",
                        help="print every policy, including each static route")
    args = parser.parse_args()

    rows = load_raw(args.raw)
    summary = summarize(rows)
    policies = POLICY_ORDER if args.all_policies else HEADLINE_POLICIES
    print(format_table(summary, policies))
    write_summary(args.out, summary)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
