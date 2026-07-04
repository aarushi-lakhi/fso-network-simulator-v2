"""Plot the PPO training curve (episode reward vs. episode) as a PNG.

Reads the per-episode rewards CSV written by ``train.py --rewards-csv``
and renders the raw rewards plus a rolling mean.

Typical usage:
    $ python plot_training.py --csv plots/training_rewards.csv \\
          --out plots/training_curve.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

AGENT_DIR = Path(__file__).resolve().parent

SURFACE = "#fcfcfb"
TEXT = "#52514e"
GRID = "#e6e5e1"
RAW_COLOR = "#9ec5f4"
MEAN_COLOR = "#2a78d6"


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Compute a trailing rolling mean with a growing initial window.

    Args:
        values: 1-D array of episode rewards.
        window: Maximum window length in episodes.

    Returns:
        Array of the same length; entry i averages values[max(0, i-window+1):i+1].
    """
    means = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        means[i] = values[max(0, i - window + 1) : i + 1].mean()
    return means


def plot_curve(rewards: np.ndarray, out_path: Path, window: int) -> None:
    """Render the training curve to a PNG file.

    Args:
        rewards: Per-episode total rewards, in training order.
        out_path: Destination PNG path (parent directories created).
        window: Rolling-mean window in episodes.
    """
    episodes = np.arange(1, len(rewards) + 1)
    smoothed = rolling_mean(rewards, window)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(episodes, rewards, color=RAW_COLOR, linewidth=1.0,
            label="episode reward")
    ax.plot(episodes, smoothed, color=MEAN_COLOR, linewidth=2.0,
            label=f"rolling mean ({window} ep)")

    ax.set_xlabel("Episode", color=TEXT)
    ax.set_ylabel("Episode reward", color=TEXT)
    ax.set_title("PPO training on fso-rl-env", color=TEXT, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend = ax.legend(loc="lower right", frameon=False)
    for text in legend.get_texts():
        text.set_color(TEXT)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    """CLI entry point: read the rewards CSV and write the PNG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path,
                        default=AGENT_DIR / "plots" / "training_rewards.csv",
                        help="rewards CSV from train.py --rewards-csv")
    parser.add_argument("--out", type=Path,
                        default=AGENT_DIR / "plots" / "training_curve.png",
                        help="output PNG path")
    parser.add_argument("--window", type=int, default=20,
                        help="rolling-mean window in episodes")
    args = parser.parse_args()

    rewards = np.loadtxt(args.csv)
    plot_curve(np.atleast_1d(rewards), args.out, args.window)
    print(f"wrote {args.out} ({len(np.atleast_1d(rewards))} episodes)")


if __name__ == "__main__":
    main()
