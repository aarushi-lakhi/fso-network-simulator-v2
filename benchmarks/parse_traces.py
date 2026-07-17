"""Aggregate raw benchmark episodes into a policy x regime summary table.

Reads a raw episode CSV written by run_benchmark.py, computes mean +/-
std of reward, PHY drops, PDR, mean delay (and, where the CSV carries
them, goodput, TCP retransmissions, and route switches) per (regime,
policy), collapses the four fixed routes into a single ``best-static``
policy (the route with the highest mean episode reward per regime),
prints the table, and writes the summary CSV. The ``regime`` column
holds the turbulence regime (Phase 5 study), the fading coherence
config name (Phase 6 correlated study), or the environment config name
(Phase 7 adaptation study); the aggregation is identical.

``--paired`` prints the Phase 7 headline analysis instead: a
per-episode (shared-seed) comparison of PPO against the best static
route per regime, with win/tie/loss counts and switching statistics.

``--study imitation`` (Phase 8) reads imitation_raw.csv and merges in
the committed adaptation_raw.csv rows of the same regimes (ppo,
statics, greedy-per, ... — measured in Phase 7 on the same seeds and
settings, so they are directly comparable and are not re-run). Its
``--paired`` mode compares bc, bc-ppo, and greedy-per against
best-static, plus bc-ppo against bc (the fine-tuning delta itself).

Typical usage:
    $ python parse_traces.py                      # Phase 5 raw_results.csv
    $ python parse_traces.py --study correlated   # Phase 6 correlated_raw.csv
    $ python parse_traces.py --study adaptation   # Phase 7 adaptation_raw.csv
    $ python parse_traces.py --study adaptation --paired
    $ python parse_traces.py --study imitation [--paired]
    $ python parse_traces.py --raw path/to/raw.csv --out path/to/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Canonical row order: Phase 5 turbulence regimes, then Phase 6 coherence
# configs, then Phase 7 adaptation configs (the studies live in separate
# CSVs and never mix labels).
REGIME_ORDER = ("weak", "moderate", "strong", "iid", "tau100-20", "tau500-100",
                "tau500-100-step50", "disjoint-iid-udp", "disjoint-tau500-udp",
                "disjoint-tau500-tcp")
POLICY_ORDER = ("ppo", "ppo-per", "ppo-per-ent", "ppo-stack", "bc", "bc-ppo",
                "ppo-transfer",
                "best-static", "static-0", "static-1", "static-2", "static-3",
                "greedy-per", "random", "aodv")

# goodput_mbps/retx/switches only exist in Phase 7 CSVs; older raw CSVs
# aggregate them as 0.
METRICS = ("reward", "drops", "pdr", "mean_delay_ms", "goodput_mbps",
           "retx", "switches")

SUMMARY_FIELDS = ("regime", "policy", "detail", "n_episodes",
                  *(f"{m}_{s}" for m in METRICS for s in ("mean", "std")))

# Policies shown in the printed table and the plots; individual static
# routes stay in summary.csv for reference.
HEADLINE_POLICIES = ("ppo", "ppo-per", "ppo-per-ent", "ppo-stack", "bc",
                     "bc-ppo", "ppo-transfer", "best-static", "greedy-per",
                     "random", "aodv")


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
            # Phase 7 columns; absent from older raw CSVs
            row["goodput_mbps"] = float(raw.get("goodput_mbps") or 0.0)
            for key in ("retx", "switches"):
                row[key] = int(raw.get(key) or 0)
            rows.append(row)
    return rows


def merge_reference_rows(rows: list[dict], reference: list[dict]) -> list[dict]:
    """Merge committed baseline rows of another study into these rows.

    Used by the imitation study to place the Phase 7 baselines (ppo,
    static routes, greedy-per, ...) next to the bc/bc-ppo rows without
    re-measuring them: reference rows are added when their regime
    appears in ``rows`` and that (regime, policy) group is not already
    present.

    Args:
        rows: Typed raw rows of the primary study.
        reference: Typed raw rows of the reference study.

    Returns:
        Combined row list (primary rows first).
    """
    regimes = {r["regime"] for r in rows}
    present = {(r["regime"], r["policy"]) for r in rows}
    merged = list(rows)
    merged.extend(r for r in reference
                  if r["regime"] in regimes
                  and (r["regime"], r["policy"]) not in present)
    return merged


def paired_policies(rows: list[dict], policy_a: str,
                    policy_b: str) -> list[dict]:
    """Compare two policies episode-by-episode on their shared seeds.

    Args:
        rows: Typed raw rows from :func:`load_raw`.
        policy_a: Policy whose deltas are reported (A minus B).
        policy_b: Baseline policy.

    Returns:
        One dict per regime where both policies have rows, with regime,
        n, reward_delta_mean/std, pdr_delta_mean, wins/ties/losses, and
        both policies' mean switch counts.
    """
    regimes = sorted({r["regime"] for r in rows},
                     key=lambda x: (REGIME_ORDER.index(x)
                                    if x in REGIME_ORDER else 99))
    results: list[dict] = []
    for regime in regimes:
        by_policy: dict[str, dict[int, dict]] = {}
        for row in rows:
            if row["regime"] == regime and row["policy"] in (policy_a, policy_b):
                by_policy.setdefault(row["policy"], {})[row["sim_seed"]] = row
        if policy_a not in by_policy or policy_b not in by_policy:
            continue
        a, b = by_policy[policy_a], by_policy[policy_b]
        seeds = sorted(set(a) & set(b))
        if not seeds:
            continue
        deltas = [a[s]["reward"] - b[s]["reward"] for s in seeds]
        pdr_deltas = [a[s]["pdr"] - b[s]["pdr"] for s in seeds]
        delta_mean, delta_std = _mean_std(deltas)
        pdr_delta_mean, _ = _mean_std(pdr_deltas)
        results.append({
            "regime": regime,
            "n": len(seeds),
            "reward_delta_mean": delta_mean,
            "reward_delta_std": delta_std,
            "pdr_delta_mean": pdr_delta_mean,
            "wins": sum(d > 0 for d in deltas),
            "ties": sum(d == 0 for d in deltas),
            "losses": sum(d < 0 for d in deltas),
            "switches_a_mean": sum(a[s]["switches"] for s in seeds) / len(seeds),
            "switches_b_mean": sum(b[s]["switches"] for s in seeds) / len(seeds),
        })
    return results


def format_paired_policies(results: list[dict], policy_a: str,
                           policy_b: str) -> str:
    """Render a two-policy paired comparison as a text table.

    Args:
        results: Rows from :func:`paired_policies`.
        policy_a: Policy whose deltas are reported.
        policy_b: Baseline policy.

    Returns:
        Multi-line table string.
    """
    header = (f"{'regime':20s} {'n':>3s} {'reward delta':>19s} "
              f"{'PDR delta':>10s} {'W/T/L':>8s} "
              f"{'sw(' + policy_a + ')':>14s} {'sw(' + policy_b + ')':>14s}")
    lines = [f"paired per-episode comparison: {policy_a} minus {policy_b} "
             f"(shared seeds)", header, "-" * len(header)]
    for row in results:
        wtl = f"{row['wins']}/{row['ties']}/{row['losses']}"
        lines.append(
            f"{row['regime']:20s} {row['n']:3d} "
            f"{row['reward_delta_mean']:10.1f} +/- {row['reward_delta_std']:5.1f} "
            f"{row['pdr_delta_mean']:+10.3f} {wtl:>8s} "
            f"{row['switches_a_mean']:14.1f} {row['switches_b_mean']:14.1f}")
    return "\n".join(lines)


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


def format_table(summary: list[dict], policies: tuple[str, ...] = HEADLINE_POLICIES,
                 extended: bool = False) -> str:
    """Render summary rows as an aligned text table.

    Args:
        summary: Rows from :func:`summarize`.
        policies: Policies to include, in POLICY_ORDER precedence.
        extended: Add the Phase 7 goodput/retx/switches columns.

    Returns:
        Multi-line table string.
    """
    header = (f"{'regime':20s} {'policy':22s} {'n':>3s} "
              f"{'reward':>19s} {'PDR':>15s} {'delay [ms]':>15s} {'PHY drops':>17s}")
    if extended:
        header += f" {'goodput [Mbps]':>15s} {'retx':>10s} {'switches':>11s}"
    lines = [header, "-" * len(header)]
    for entry in summary:
        if entry["policy"] not in policies:
            continue
        name = entry["policy"]
        if entry["detail"]:
            name = f"{name} ({entry['detail']})"
        line = (
            f"{entry['regime']:20s} {name:22s} {entry['n_episodes']:3d} "
            f"{entry['reward_mean']:10.1f} +/- {entry['reward_std']:5.1f} "
            f"{entry['pdr_mean']:7.3f} +/- {entry['pdr_std']:5.3f} "
            f"{entry['mean_delay_ms_mean']:7.3f} +/- {entry['mean_delay_ms_std']:5.3f} "
            f"{entry['drops_mean']:9.1f} +/- {entry['drops_std']:5.1f}"
        )
        if extended:
            line += (f" {entry['goodput_mbps_mean']:7.3f} "
                     f"{entry['retx_mean']:10.1f} "
                     f"{entry['switches_mean']:11.1f}")
        lines.append(line)
    return "\n".join(lines)


def paired_ppo_vs_best_static(rows: list[dict],
                              ppo_policy: str = "ppo") -> list[dict]:
    """Compare PPO against the best static route episode-by-episode.

    All policies of a regime share the same ns-3 run numbers, so the
    comparison pairs episodes by ``sim_seed``: same seed, same fading
    realisation, the only difference is the routing policy. The best
    static route is the one with the highest mean episode reward in the
    regime (matching the ``best-static`` summary row).

    Args:
        rows: Typed raw rows from :func:`load_raw`.
        ppo_policy: Policy name to compare against best-static.

    Returns:
        One dict per regime that has both PPO and static rows, with:
        regime, best_route, n (paired episodes), reward_delta_mean/std
        and pdr_delta_mean (PPO minus best-static), wins/ties/losses
        (per-episode reward sign; ties are exact float equality),
        switches_mean/switches_max (PPO route changes per episode), and
        identical_route — the static route whose per-episode rewards
        PPO reproduces exactly on every paired seed ("" if none), the
        constant-route-policy fingerprint.
    """
    regimes = sorted({r["regime"] for r in rows},
                     key=lambda x: (REGIME_ORDER.index(x)
                                    if x in REGIME_ORDER else 99))
    results: list[dict] = []
    for regime in regimes:
        regime_rows = [r for r in rows if r["regime"] == regime]
        ppo = {r["sim_seed"]: r for r in regime_rows
               if r["policy"] == ppo_policy}
        statics: dict[str, dict[int, dict]] = {}
        for row in regime_rows:
            if row["policy"].startswith("static-"):
                statics.setdefault(row["policy"], {})[row["sim_seed"]] = row
        if not ppo or not statics:
            continue

        def mean_reward(by_seed: dict[int, dict]) -> float:
            return sum(r["reward"] for r in by_seed.values()) / len(by_seed)

        best_policy = max(statics, key=lambda p: mean_reward(statics[p]))
        best = statics[best_policy]
        seeds = sorted(set(ppo) & set(best))
        if not seeds:
            continue
        deltas = [ppo[s]["reward"] - best[s]["reward"] for s in seeds]
        pdr_deltas = [ppo[s]["pdr"] - best[s]["pdr"] for s in seeds]
        switches = [ppo[s]["switches"] for s in seeds]
        identical_route = ""
        for policy, by_seed in sorted(statics.items()):
            shared = sorted(set(ppo) & set(by_seed))
            if shared and all(ppo[s]["reward"] == by_seed[s]["reward"]
                              for s in shared):
                identical_route = policy.split("-", 1)[1]
                break
        delta_mean, delta_std = _mean_std(deltas)
        pdr_delta_mean, _ = _mean_std(pdr_deltas)
        results.append({
            "regime": regime,
            "best_route": best_policy.split("-", 1)[1],
            "n": len(seeds),
            "reward_delta_mean": delta_mean,
            "reward_delta_std": delta_std,
            "pdr_delta_mean": pdr_delta_mean,
            "wins": sum(d > 0 for d in deltas),
            "ties": sum(d == 0 for d in deltas),
            "losses": sum(d < 0 for d in deltas),
            "switches_mean": sum(switches) / len(switches),
            "switches_max": max(switches),
            "identical_route": identical_route,
        })
    return results


def format_paired(results: list[dict], ppo_policy: str = "ppo") -> str:
    """Render the paired PPO-vs-best-static comparison as a text table.

    Args:
        results: Rows from :func:`paired_ppo_vs_best_static`.
        ppo_policy: Policy name used in the header.

    Returns:
        Multi-line table string.
    """
    header = (f"{'regime':20s} {'vs':>8s} {'n':>3s} {'reward delta':>19s} "
              f"{'PDR delta':>10s} {'W/T/L':>8s} {'switches':>14s} "
              f"{'identical-to':>13s}")
    lines = [f"paired per-episode comparison: {ppo_policy} minus best-static "
             f"(shared seeds)", header, "-" * len(header)]
    for row in results:
        wtl = f"{row['wins']}/{row['ties']}/{row['losses']}"
        switches = f"{row['switches_mean']:.1f} (max {row['switches_max']})"
        identical = (f"route {row['identical_route']}"
                     if row["identical_route"] else "-")
        lines.append(
            f"{row['regime']:20s} {'route ' + row['best_route']:>8s} "
            f"{row['n']:3d} "
            f"{row['reward_delta_mean']:10.1f} +/- {row['reward_delta_std']:5.1f} "
            f"{row['pdr_delta_mean']:+10.3f} {wtl:>8s} {switches:>14s} "
            f"{identical:>13s}")
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
    parser.add_argument("--study", choices=("turbulence", "correlated",
                                            "adaptation", "imitation"),
                        default="turbulence",
                        help="picks the default --raw/--out file pair")
    parser.add_argument("--raw", type=str, default=None,
                        help="input raw episode CSV")
    parser.add_argument("--out", type=str, default=None,
                        help="output summary CSV")
    parser.add_argument("--all-policies", action="store_true",
                        help="print every policy, including each static route")
    parser.add_argument("--paired", action="store_true",
                        help="print the paired per-episode PPO vs best-static "
                             "comparison instead of writing the summary")
    args = parser.parse_args()
    stems = {"turbulence": ("raw_results", "summary"),
             "correlated": ("correlated_raw", "correlated_summary"),
             "adaptation": ("adaptation_raw", "adaptation_summary"),
             "imitation": ("imitation_raw", "imitation_summary")}
    stem, out_stem = stems[args.study]
    args.raw = args.raw or str(RESULTS_DIR / f"{stem}.csv")
    args.out = args.out or str(RESULTS_DIR / f"{out_stem}.csv")

    rows = load_raw(args.raw)
    if args.study == "imitation":
        reference_csv = RESULTS_DIR / "adaptation_raw.csv"
        if reference_csv.exists():
            rows = merge_reference_rows(rows, load_raw(reference_csv))
    if args.paired:
        if args.study == "imitation":
            for policy in ("bc", "bc-ppo", "greedy-per"):
                print(format_paired(paired_ppo_vs_best_static(rows, policy),
                                    policy))
                print()
            print(format_paired_policies(paired_policies(rows, "bc-ppo", "bc"),
                                         "bc-ppo", "bc"))
        else:
            print(format_paired(paired_ppo_vs_best_static(rows)))
        return
    summary = summarize(rows)
    policies = POLICY_ORDER if args.all_policies else HEADLINE_POLICIES
    extended = args.study in ("adaptation", "imitation")
    print(format_table(summary, policies, extended=extended))
    write_summary(args.out, summary)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
