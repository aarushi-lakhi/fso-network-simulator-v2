"""Hermetic tests for the benchmark aggregation logic (no ns-3 needed)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parse_traces import (  # noqa: E402
    format_table,
    load_raw,
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


def test_single_episode_std_is_zero(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    path.write_text(CSV_HEADER + "\n" +
                    "strong,1e-13,aodv,0,100,-900.0,800,2440,1600,0.656,0.19\n")
    summary = summarize(load_raw(path))
    assert summary[0]["reward_std"] == 0.0
    assert summary[0]["n_episodes"] == 1
