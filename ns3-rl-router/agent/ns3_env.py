"""Gymnasium factory for the real ns3-ai FSO routing environment.

Importing this module applies the ns3-ai compatibility shims (NumPy
scalar aliases and the gym-interface ``sys.path`` entry) via
``sim/ns3ai_shim.py``, then exposes :func:`make_ns3_env`, which spawns
the ``fso-rl-env`` ns-3 program (see sim/README.md) as a subprocess
communicating over shared memory.

Requirements: the fso-channel/fso-rl-env symlinks must be built into the
ns-3 tree (``setup/link_fso_modules.sh``) and the process must run inside
the agent venv so that ``env python3`` resolves to Python 3.11.

Side effect to be aware of: constructing the environment chdirs the
process into the ns-3 root (an ns3ai_utils.Experiment behaviour), so
resolve any relative output paths before calling :func:`make_ns3_env`.

Typical usage:
    >>> from ns3_env import make_ns3_env
    >>> env = make_ns3_env(c2n="1e-13", seed=42)
    >>> obs, _ = env.reset(seed=42)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import gymnasium as gym

_SIM_DIR = str(Path(__file__).resolve().parent.parent / "sim")
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)

from ns3ai_shim import DEFAULT_NS3_PATH, load_flat_yaml, ns3_settings  # noqa: E402
from ns3ai_gym_env.envs.ns3_environment import Ns3Env  # noqa: E402

DEFAULT_CONFIG_PATH = str(
    Path(__file__).resolve().parent.parent / "config" / "sim_config.yaml"
)


class FsoNs3Env(Ns3Env):
    """Ns3Env with settings-preserving, seed-aware episode restarts.

    Upstream ``Ns3Env.reset`` re-runs the ns-3 target *without* the
    original command-line settings, silently reverting C2n, episode
    length, reward weights, and seed to their C++ defaults from the
    second episode on. This subclass re-applies ``ns3Settings`` on every
    restart and advances ``simSeed`` between episodes so consecutive
    episodes sample different fading realisations; passing ``seed``
    pins the run number explicitly (for reproducible evaluation).
    """

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Start a new episode, restarting ns-3 with the stored settings.

        Args:
            seed: Optional ns-3 run number (``simSeed``) for this episode.
                When omitted, the previous episode's run number + 1 is used.
            options: Unused; present for Gymnasium API compatibility.

        Returns:
            Tuple (observation, info) per the Gymnasium API.
        """
        current_seed = int(self.ns3Settings["simSeed"])
        if not self.envDirty and (seed is None or int(seed) == current_seed):
            return self.get_obs(), {}

        self.ns3Settings["simSeed"] = str(int(seed) if seed is not None else current_seed + 1)

        if not self.gameOver:
            self.rx_env_state()
            self.send_close_command()

        self.newStateRx = False
        self.obsData = None
        self.reward = 0
        self.gameOver = False
        self.gameOverReason = None
        self.extraInfo = None

        self.msgInterface = self.exp.run(setting=self.ns3Settings, show_output=True)
        self.initialize_env()
        self.rx_env_state()
        self.envDirty = False
        return self.get_obs(), {}


def make_ns3_env(
    config_path: str = DEFAULT_CONFIG_PATH,
    c2n: str | float | None = None,
    seed: int | None = None,
    ns3_path: str = DEFAULT_NS3_PATH,
    coherence_large: str | None = None,
    coherence_small: str | None = None,
    step_time_s: str | float | None = None,
    episode_steps: int | None = None,
) -> gym.Env:
    """Create the real FSO routing environment backed by ns-3.

    Args:
        config_path: Path to sim_config.yaml (flat key: value file).
        c2n: Optional override of the refractive index structure
            parameter [m^-2/3], e.g. ``"1e-13"`` for strong turbulence.
        seed: ns-3 run number of the first episode; later episodes
            advance it by one per reset. Defaults to ``sim_seed`` from
            the config file.
        ns3_path: ns-3 root directory containing the built fso-rl-env.
        coherence_large: Optional override of the large-scale fading
            coherence time (ns-3 Time string, e.g. ``"500ms"``;
            ``"0ms"`` means i.i.d. block fading).
        coherence_small: Optional override of the small-scale fading
            coherence time (same format as ``coherence_large``).
        step_time_s: Optional override of the agent decision interval
            [s], e.g. ``0.05``.
        episode_steps: Optional override of the number of decision
            steps per episode.

    Returns:
        A Gymnasium env with Box(28) observations and Discrete(4) actions.
    """
    config = load_flat_yaml(config_path)
    sim_seed = seed if seed is not None else int(config.get("sim_seed", "1"))
    settings = ns3_settings(config, sim_seed)
    overrides = {
        "c2n": c2n,
        "coherenceLarge": coherence_large,
        "coherenceSmall": coherence_small,
        "stepTime": step_time_s,
        "episodeSteps": episode_steps,
    }
    for key, value in overrides.items():
        if value is not None:
            settings[key] = str(value)
    return FsoNs3Env(targetName="fso-rl-env", ns3Path=ns3_path, ns3Settings=settings)
