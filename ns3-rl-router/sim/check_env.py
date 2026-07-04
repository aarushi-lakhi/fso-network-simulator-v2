"""Smoke test for the fso-rl-env Gym environment.

Creates the environment through ns3ai_gym (which spawns the ns-3 simulation
as a subprocess over shared memory), runs a handful of random-action steps,
prints the observation/reward shapes and a few transitions, and exits
cleanly.

Run with the ns3-ai virtualenv ACTIVATED so that ``env python3`` inside the
ns3 wrapper resolves to Python 3.11::

    source ~/fso-tools/ns3ai-venv/bin/activate
    python check_env.py --steps 10

The fso-channel and fso-rl-env symlinks must be installed in the ns-3 tree
first (see setup/link_fso_modules.sh).
"""

import argparse
import os
import sys
import traceback

import numpy as np

# ns3ai_gym_env still uses the np.float/np.int aliases removed in NumPy 1.24;
# restore them before the import.
np.float = float  # type: ignore[attr-defined]
np.int = int  # type: ignore[attr-defined]
np.uint = np.uint64  # type: ignore[attr-defined]

import gymnasium as gym

DEFAULT_NS3_PATH = os.path.expanduser(
    os.path.join(os.environ.get("FSO_TOOLS_DIR", "~/fso-tools"), "ns-3-dev")
)
DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "config", "sim_config.yaml"
)


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


def main() -> int:
    """Run random actions against the environment and report transitions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10,
                        help="number of random-action steps to run")
    parser.add_argument("--ns3-path", type=str, default=DEFAULT_NS3_PATH,
                        help="ns-3 root directory")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help="path to sim_config.yaml")
    parser.add_argument("--seed", type=int, default=1,
                        help="simulation run number")
    parser.add_argument("--c2n", type=str, default=None,
                        help="override C2n [m^-2/3], e.g. 1e-13 for strong "
                             "turbulence")
    args = parser.parse_args()

    # ns3ai_gym_env's editable install does not expose its top-level
    # messages_pb2 / ns3ai_gym_msg_py modules; import them from the source dir.
    sys.path.append(os.path.join(args.ns3_path, "contrib", "ai",
                                 "model", "gym-interface", "py"))
    import ns3ai_gym_env  # noqa: F401  (registers ns3ai_gym_env/Ns3-v0)

    config = load_flat_yaml(os.path.abspath(args.config))
    settings = ns3_settings(config, args.seed)
    if args.c2n is not None:
        settings["c2n"] = args.c2n
    # Size the episode so the last step consumes the simulation-end state,
    # exercising the clean done=True termination path
    settings["episodeSteps"] = str(args.steps)

    rng = np.random.default_rng(args.seed)
    env = gym.make(
        "ns3ai_gym_env/Ns3-v0",
        targetName="fso-rl-env",
        ns3Path=args.ns3_path,
        ns3Settings=settings,
    )
    exit_code = 0
    try:
        print(f"observation space: {env.observation_space}")
        print(f"action space:      {env.action_space}")

        obs, _ = env.reset()
        obs = np.asarray(obs)
        print(f"reset: obs shape={obs.shape} dtype={obs.dtype}")
        print(f"  obs[link0] (snrMarginDb, dropRate, scintIndex, queuePkts) "
              f"= {np.round(obs[:4], 4)}")

        for step in range(args.steps):
            action = int(rng.integers(env.action_space.n))
            obs, reward, done, truncated, info = env.step(action)
            obs = np.asarray(obs)
            print(f"step {step:2d}: action={action} reward={reward:8.3f} "
                  f"done={done} obs shape={obs.shape} "
                  f"info={info.get('info', '')}")
            if done:
                print("episode finished (simulation end)")
                break
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"check_env FAILED: {exc}")
        traceback.print_exc()
        exit_code = 1
    finally:
        env.close()

    print("check_env: " + ("OK" if exit_code == 0 else "FAILED"))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
