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

import gymnasium as gym
import numpy as np

# Applies the numpy-alias and sys.path shims ns3ai_gym_env needs (on import)
from ns3ai_shim import DEFAULT_NS3_PATH, add_gym_msg_path, load_flat_yaml, ns3_settings

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "config", "sim_config.yaml"
)


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

    add_gym_msg_path(args.ns3_path)
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
