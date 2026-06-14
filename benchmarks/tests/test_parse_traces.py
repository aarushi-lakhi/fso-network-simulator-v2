"""Hermetic tests for the benchmark aggregation logic (no ns-3 needed)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parse_traces import (  # noqa: E402
    format_paired,
    format_paired_policies,
    format_table,
    load_raw,
    merge_reference_rows,
    paired_policies,
    paired_ppo_vs_best_static,
    summarize,
    write_summary,
)

CSV_HEADER = ("regime,c2n,policy,episode,sim_seed,reward,drops,tx_pkts,"
              "rx_pkts,pdr,mean_delay_ms")

SYNTHETIC_ROWS = [
    # strong regime: static-1 clearly better than static-0; ppo best of all
    "strong,1e-13,ppo,0,100,-700.0,650,2440,1800,0.738,0.20",
    "strong,1e-13,ppo,1,101,-720.0,670,2440,1780,0.730,0.22",
    "strong,1e-13,static-0,0,100,-1000.0,900,2440,1500,0.615,0.30",
    "strong,1e-13,static-0,1,101,-1040.0,920,2440,1480,0.607,0.32",
    "strong,1e-13,static-1,0,100,-800.0,700,2440,1700,0.697,0.25",
    "strong,1e-13,static-1,1,101,-820.0,720,2440,1690,0.693,0.27",
    "strong,1e-13,random,0,100,-1200.0,1000,2440,1400,0.574,0.35",
    "strong,1e-13,random,1,101,-1260.0,1040,2440,1380,0.566,0.37",
    # weak regime: everyone delivers everything
    "weak,1e-17,ppo,0,100,-30.0,0,2440,2440,1.0,0.18",
    "weak,1e-17,ppo,1,101,-30.0,0,2440,2440,1.0,0.18",
    "weak,1e-17,static-0,0,100,-25.0,0,2440,2440,1.0,0.18",
    "weak,1e-17,static-0,1,101,-25.0,0,2440,2440,1.0,0.18",
    "weak,1e-17,static-1,0,100,-27.0,0,2440,2440,1.0,0.18",
    "weak,1e-17,static-1,1,101,-27.0,0,2440,2440,1.0,0.18",
]


@pytest.fixture()
def raw_csv(tmp_path: Path) -> Path:
    """Write the synthetic raw CSV and return its path."""
    path = tmp_path / "raw_results.csv"
    path.write_text("\n".join([CSV_HEADER, *SYNTHETIC_ROWS]) + "\n")
    return path


def test_load_raw_types(raw_csv: Path) -> None:
    rows = load_raw(raw_csv)
    assert len(rows) == len(SYNTHETIC_ROWS)
    first = rows[0]
    assert isinstance(first["reward"], float)
    assert isinstance(first["drops"], int)
    assert isinstance(first["pdr"], float)
    assert first["policy"] == "ppo"


def test_summarize_means_and_stds(raw_csv: Path) -> None:
    summary = summarize(load_raw(raw_csv))
    ppo_strong = next(e for e in summary
                      if e["regime"] == "strong" and e["policy"] == "ppo")
    assert ppo_strong["n_episodes"] == 2
    assert ppo_strong["reward_mean"] == pytest.approx(-710.0)
    assert ppo_strong["reward_std"] == pytest.approx(10.0)
    assert ppo_strong["pdr_mean"] == pytest.approx(0.734)
    assert ppo_strong["drops_mean"] == pytest.approx(660.0)


def test_best_static_picks_highest_reward_route_per_regime(raw_csv: Path) -> None:
    summary = summarize(load_raw(raw_csv))
    best_strong = next(e for e in summary
                       if e["regime"] == "strong" and e["policy"] == "best-static")
    assert best_strong["detail"] == "route=1"
    assert best_strong["reward_mean"] == pytest.approx(-810.0)
    # weak regime: static-0 (-25) beats static-1 (-27)
    best_weak = next(e for e in summary
                     if e["regime"] == "weak" and e["policy"] == "best-static")
    assert best_weak["detail"] == "route=0"


def test_summary_order_is_regime_then_policy(raw_csv: Path) -> None:
    summary = summarize(load_raw(raw_csv))
    keys = [(e["regime"], e["policy"]) for e in summary]
    assert keys.index(("weak", "ppo")) < keys.index(("strong", "ppo"))
    strong_keys = [p for r, p in keys if r == "strong"]
    assert strong_keys.index("ppo") < strong_keys.index("best-static")
    assert strong_keys.index("best-static") < strong_keys.index("random")


def test_format_table_headline_hides_individual_static_routes(raw_csv: Path) -> None:
    table = format_table(summarize(load_raw(raw_csv)))
    assert "best-static (route=1)" in table
    assert "static-0" not in table
    assert "ppo" in table
    assert "random" in table


def test_write_summary_roundtrip(raw_csv: Path, tmp_path: Path) -> None:
    summary = summarize(load_raw(raw_csv))
    out = tmp_path / "summary.csv"
    write_summary(out, summary)
    with open(out, newline="") as fp:
        rows = list(csv.DictReader(fp))
    assert len(rows) == len(summary)
    ppo_strong = next(r for r in rows
                      if r["regime"] == "strong" and r["policy"] == "ppo")
    assert float(ppo_strong["reward_mean"]) == pytest.approx(-710.0)
    assert int(ppo_strong["n_episodes"]) == 2


def test_correlated_study_labels_sort_after_regimes(tmp_path: Path) -> None:
    """Coherence config names (Phase 6) order iid < tau100-20 < tau500-100."""
    path = tmp_path / "correlated_raw.csv"
    rows = [
        "tau500-100,1e-13,ppo,0,100,-500.0,450,2440,2000,0.820,0.20",
        "tau500-100,1e-13,static-2,0,100,-700.0,650,2440,1800,0.738,0.20",
        "iid,1e-13,ppo,0,100,-730.0,675,2440,1770,0.725,0.18",
        "tau100-20,1e-13,ppo,0,100,-650.0,600,2440,1850,0.758,0.19",
    ]
    path.write_text("\n".join([CSV_HEADER, *rows]) + "\n")
    summary = summarize(load_raw(path))
    labels = [e["regime"] for e in summary]
    assert labels.index("iid") < labels.index("tau100-20")
    assert labels.index("tau100-20") < labels.index("tau500-100")
    best = next(e for e in summary
                if e["regime"] == "tau500-100" and e["policy"] == "best-static")
    assert best["detail"] == "route=2"


EXTENDED_HEADER = CSV_HEADER + ",goodput_mbps,retx,switches"

# Phase 7 adaptation rows: PPO switches routes and beats static-0 on
# seed 100, loses on 101, matches nothing byte-identically.
ADAPTATION_ROWS = [
    "disjoint-tau500-tcp,1e-13,ppo,0,100,-500.0,400,2440,2000,0.820,0.20,1.64,30,4",
    "disjoint-tau500-tcp,1e-13,ppo,1,101,-720.0,600,2440,1800,0.738,0.22,1.48,55,2",
    "disjoint-tau500-tcp,1e-13,static-0,0,100,-600.0,500,2440,1900,0.779,0.21,1.56,40,0",
    "disjoint-tau500-tcp,1e-13,static-0,1,101,-700.0,580,2440,1820,0.746,0.21,1.49,50,0",
    "disjoint-tau500-tcp,1e-13,static-1,0,100,-900.0,800,2440,1600,0.656,0.25,1.31,80,0",
    "disjoint-tau500-tcp,1e-13,static-1,1,101,-950.0,840,2440,1580,0.648,0.26,1.30,85,0",
]


def test_load_raw_defaults_missing_phase7_columns(raw_csv: Path) -> None:
    """Older raw CSVs (no goodput/retx/switches) load with zeros."""
    rows = load_raw(raw_csv)
    assert rows[0]["goodput_mbps"] == 0.0
    assert rows[0]["retx"] == 0
    assert rows[0]["switches"] == 0


def test_load_raw_parses_phase7_columns(tmp_path: Path) -> None:
    path = tmp_path / "adaptation_raw.csv"
    path.write_text("\n".join([EXTENDED_HEADER, *ADAPTATION_ROWS]) + "\n")
    rows = load_raw(path)
    ppo0 = rows[0]
    assert ppo0["goodput_mbps"] == pytest.approx(1.64)
    assert ppo0["retx"] == 30
    assert ppo0["switches"] == 4


def test_summarize_includes_phase7_metrics(tmp_path: Path) -> None:
    path = tmp_path / "adaptation_raw.csv"
    path.write_text("\n".join([EXTENDED_HEADER, *ADAPTATION_ROWS]) + "\n")
    summary = summarize(load_raw(path))
    ppo = next(e for e in summary if e["policy"] == "ppo")
    assert ppo["goodput_mbps_mean"] == pytest.approx(1.56)
    assert ppo["retx_mean"] == pytest.approx(42.5)
    assert ppo["switches_mean"] == pytest.approx(3.0)
    table = format_table(summary, extended=True)
    assert "goodput" in table and "switches" in table


def test_adaptation_labels_sort_after_coherence_configs(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    rows = [
        "disjoint-tau500-udp,1e-13,ppo,0,100,-500.0,450,2440,2000,0.820,0.20",
        "disjoint-iid-udp,1e-13,ppo,0,100,-730.0,675,2440,1770,0.725,0.18",
    ]
    path.write_text("\n".join([CSV_HEADER, *rows]) + "\n")
    labels = [e["regime"] for e in summarize(load_raw(path))]
    assert labels.index("disjoint-iid-udp") < labels.index("disjoint-tau500-udp")


def test_paired_comparison_pairs_by_seed(tmp_path: Path) -> None:
    path = tmp_path / "adaptation_raw.csv"
    path.write_text("\n".join([EXTENDED_HEADER, *ADAPTATION_ROWS]) + "\n")
    results = paired_ppo_vs_best_static(load_raw(path))
    assert len(results) == 1
    row = results[0]
    assert row["regime"] == "disjoint-tau500-tcp"
    assert row["best_route"] == "0"  # static-0 mean -650 beats static-1 -925
    assert row["n"] == 2
    # deltas: seed 100: -500 - (-600) = +100; seed 101: -720 - (-700) = -20
    assert row["reward_delta_mean"] == pytest.approx(40.0)
    assert row["wins"] == 1
    assert row["ties"] == 0
    assert row["losses"] == 1
    assert row["switches_mean"] == pytest.approx(3.0)
    assert row["switches_max"] == 4
    assert row["identical_route"] == ""


def test_paired_comparison_flags_constant_route_policy(tmp_path: Path) -> None:
    """A PPO that reproduces a static route exactly is fingerprinted."""
    path = tmp_path / "raw.csv"
    rows = [
        "strong,1e-13,ppo,0,100,-800.0,700,2440,1700,0.697,0.25",
        "strong,1e-13,ppo,1,101,-820.0,720,2440,1690,0.693,0.27",
        "strong,1e-13,static-0,0,100,-1000.0,900,2440,1500,0.615,0.30",
        "strong,1e-13,static-0,1,101,-1040.0,920,2440,1480,0.607,0.32",
        "strong,1e-13,static-1,0,100,-800.0,700,2440,1700,0.697,0.25",
        "strong,1e-13,static-1,1,101,-820.0,720,2440,1690,0.693,0.27",
    ]
    path.write_text("\n".join([CSV_HEADER, *rows]) + "\n")
    results = paired_ppo_vs_best_static(load_raw(path))
    row = results[0]
    assert row["identical_route"] == "1"
    assert row["ties"] == 2
    assert row["reward_delta_mean"] == 0.0
    text = format_paired(results)
    assert "route 1" in text
    assert "2/2" not in text  # W/T/L renders as 0/2/0
    assert "0/2/0" in text


# Phase 8 imitation rows sharing the adaptation regime and seeds
IMITATION_ROWS = [
    "disjoint-tau500-tcp,1e-13,bc,0,100,-550.0,450,2440,1950,0.799,0.20,1.60,35,40",
    "disjoint-tau500-tcp,1e-13,bc,1,101,-710.0,590,2440,1810,0.742,0.22,1.47,52,44",
    "disjoint-tau500-tcp,1e-13,bc-ppo,0,100,-590.0,480,2440,1920,0.787,0.21,1.58,38,1",
    "disjoint-tau500-tcp,1e-13,bc-ppo,1,101,-700.0,585,2440,1815,0.744,0.22,1.48,51,1",
]


def test_merge_reference_rows_fills_missing_baselines(tmp_path: Path) -> None:
    imitation_path = tmp_path / "imitation_raw.csv"
    imitation_path.write_text("\n".join([EXTENDED_HEADER, *IMITATION_ROWS]) + "\n")
    reference_path = tmp_path / "adaptation_raw.csv"
    reference_path.write_text(
        "\n".join([EXTENDED_HEADER, *ADAPTATION_ROWS,
                   # a regime absent from the imitation rows: must not leak in
                   "disjoint-iid-udp,1e-13,ppo,0,100,-500.0,450,2440,2000,"
                   "0.820,0.20,0.0,0,0"]) + "\n")
    merged = merge_reference_rows(load_raw(imitation_path),
                                  load_raw(reference_path))
    policies = {r["policy"] for r in merged}
    assert policies == {"bc", "bc-ppo", "ppo", "static-0", "static-1"}
    assert {r["regime"] for r in merged} == {"disjoint-tau500-tcp"}


def test_merge_reference_rows_keeps_primary_measurements(tmp_path: Path) -> None:
    imitation_path = tmp_path / "imitation_raw.csv"
    imitation_path.write_text(
        "\n".join([EXTENDED_HEADER, *IMITATION_ROWS,
                   # a bc row also present in the reference must win
                   ]) + "\n")
    reference_path = tmp_path / "adaptation_raw.csv"
    reference_path.write_text(
        "\n".join([EXTENDED_HEADER,
                   "disjoint-tau500-tcp,1e-13,bc,0,100,-9999.0,450,2440,1950,"
                   "0.799,0.20,1.60,35,40"]) + "\n")
    merged = merge_reference_rows(load_raw(imitation_path),
                                  load_raw(reference_path))
    bc_rewards = [r["reward"] for r in merged if r["policy"] == "bc"]
    assert -9999.0 not in bc_rewards


def test_paired_policies_deltas_and_switches(tmp_path: Path) -> None:
    path = tmp_path / "imitation_raw.csv"
    path.write_text("\n".join([EXTENDED_HEADER, *IMITATION_ROWS]) + "\n")
    results = paired_policies(load_raw(path), "bc-ppo", "bc")
    assert len(results) == 1
    row = results[0]
    assert row["n"] == 2
    # seed 100: -590 - (-550) = -40; seed 101: -700 - (-710) = +10
    assert row["reward_delta_mean"] == pytest.approx(-15.0)
    assert row["wins"] == 1 and row["losses"] == 1
    assert row["switches_a_mean"] == pytest.approx(1.0)
    assert row["switches_b_mean"] == pytest.approx(42.0)
    text = format_paired_policies(results, "bc-ppo", "bc")
    assert "bc-ppo minus bc" in text
    assert "1/0/1" in text


def test_single_episode_std_is_zero(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    path.write_text(CSV_HEADER + "\n" +
                    "strong,1e-13,aodv,0,100,-900.0,800,2440,1600,0.656,0.19\n")
    summary = summarize(load_raw(path))
    assert summary[0]["reward_std"] == 0.0
    assert summary[0]["n_episodes"] == 1
