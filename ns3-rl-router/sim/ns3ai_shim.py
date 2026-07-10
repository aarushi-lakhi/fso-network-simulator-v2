"""Compatibility shims and config helpers for the ns3-ai Gym bindings.

Importing this module applies the two fixes required before
``ns3ai_gym_env`` can be imported (documented in sim/README.md):

* ``ns3ai_gym_env`` still uses the ``np.float``/``np.int``/``np.uint``
  aliases removed in NumPy >= 1.24 — they are restored here.
* The editable install of ``ns3ai_gym_env`` does not expose its top-level
  ``messages_pb2``/``ns3ai_gym_msg_py`` modules; the gym-interface source
  directory inside the ns-3 tree is appended to ``sys.path``.

Also hosts the sim_config.yaml helpers shared by ``check_env.py`` and
``agent/ns3_env.py``.
"""

from __future__ import annotations

import os
import sys

import numpy as np

DEFAULT_NS3_PATH = os.path.expanduser(
    os.path.join(os.environ.get("FSO_TOOLS_DIR", "~/fso-tools"), "ns-3-dev")
)


def apply_numpy_aliases() -> None:
    """Restore the deprecated NumPy scalar aliases ns3ai_gym_env relies on."""
    np.float = float  # type: ignore[attr-defined]
    np.int = int  # type: ignore[attr-defined]
    np.uint = np.uint64  # type: ignore[attr-defined]


def add_gym_msg_path(ns3_path: str | None = None) -> str:
    """Put ns3-ai's gym-interface Python sources on ``sys.path``.

    Args:
        ns3_path: ns-3 root directory; defaults to ``DEFAULT_NS3_PATH``
            (``$FSO_TOOLS_DIR/ns-3-dev``).

    Returns:
        The directory that was appended (or already present).
    """
    path = os.path.join(
        ns3_path or DEFAULT_NS3_PATH, "contrib", "ai", "model", "gym-interface", "py"
    )
    if path not in sys.path:
        sys.path.append(path)
    return path


def load_flat_yaml(path: str) -> dict[str, str]:
    """Parse a flat ``key: value`` YAML file without a yaml dependency.

    Args:
        path: Path to the YAML file. Only top-level scalar keys are
            supported (which is all sim_config.yaml contains).

    Returns:
        Mapping of key to raw string value, comments stripped.
    """
    config: dict[str, str] = {}
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            config[key.strip()] = value.strip()
    return config


def ns3_settings(config: dict[str, str], seed: int) -> dict[str, str]:
    """Map sim_config.yaml keys onto fso-rl-env command-line arguments.

    Args:
        config: Flat config mapping from :func:`load_flat_yaml`.
        seed: Simulation run number.

    Returns:
        Settings dict passed to the ns-3 process as ``--key=value`` pairs.
    """
    return {
        "c2n": config["c2n"],
        "episodeSteps": config["episode_steps"],
        "stepTime": config["step_time_s"],
        "updateIntervalMs": config["update_interval_ms"],
        "coherenceLarge": config.get("coherence_large", "0ms"),
        "coherenceSmall": config.get("coherence_small", "0ms"),
        "txPowerDbm": config["tx_power_dbm"],
        "noiseDbm": config["noise_dbm"],
        "wavelength": config["wavelength_m"],
        "extinction": config["extinction_coeff_per_m"],
        "meshRadius": config["mesh_radius_m"],
        "dataRate": config["data_rate"],
        "trafficRate": config["traffic_rate"],
        "packetSize": config["packet_size_bytes"],
        "dropWeight": config["reward_drop_weight"],
        "delayWeight": config["reward_delay_weight"],
        "flapPenalty": config["reward_flap_penalty"],
        "energyWeight": config["reward_energy_weight"],
        "simSeed": str(seed),
    }


apply_numpy_aliases()
add_gym_msg_path()
