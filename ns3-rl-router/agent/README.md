# PPO Routing Agent

PyTorch PPO agent for FSO mesh route selection. Environment-agnostic: any
Gymnasium env with a flattened `Box` observation and `Discrete` action space
plugs in via the `env_factory` argument of `train.train()`. The built-in
`toy_env.ToyFsoRoutingEnv` (K routes with drifting link qualities) is the
default, so the agent trains and tests without ns-3.

## Setup

Requires Python 3.11 (ns3-ai bindings target 3.11).

```bash
cd ns3-rl-router
"$(brew --prefix python@3.11)/bin/python3.11" -m venv agent/.venv
agent/.venv/bin/pip install -r requirements.txt
```

For the real ns-3 environment (`--env ns3`), additionally install the
ns3-ai Python packages into the *same* venv:

```bash
agent/.venv/bin/pip install -e ~/fso-tools/ns-3-dev/contrib/ai/python_utils
agent/.venv/bin/pip install -e ~/fso-tools/ns-3-dev/contrib/ai/model/gym-interface/py
```

## Train

```bash
cd agent
.venv/bin/python train.py --total-steps 50000 --seed 42 \
    --log-dir runs/toy --checkpoint-path checkpoints/agent.pt
tensorboard --logdir runs/
```

All hyperparameters can also come from YAML (CLI flags win):

```bash
.venv/bin/python train.py --config my_config.yaml
```

## Train on the real FSO env (ns-3)

One-time: link and build the ns-3 pieces, then run every Python command
below with the venv **activated** (`ns3ai_utils` spawns `./ns3` via
`env python3`, which must resolve to the venv's 3.11):

```bash
../../setup/link_fso_modules.sh
source .venv/bin/activate
python train.py --env ns3 --c2n 1e-13 --total-steps 80000 \
    --rollout-steps 500 --seed 42 \
    --checkpoint-path checkpoints/ns3_ppo.pt \
    --rewards-csv plots/training_rewards.csv
python plot_training.py       # writes plots/training_curve.png
```

`ns3_env.make_ns3_env()` wraps ns3-ai's `Ns3Env` so every episode restart
re-applies the sim settings and advances the ns-3 run number (upstream
`reset()` silently drops both). Reads `../config/sim_config.yaml`;
`--c2n` selects the turbulence regime. Note that creating the env chdirs
the process into the ns-3 root, so relative output paths are resolved
before the env starts.

## Evaluate: trained checkpoint vs random

```bash
python eval_policy.py --episodes 10 --seed 100 --c2n 1e-13
python eval_policy.py --episodes 10 --seed 100 --c2n 1e-13 \
    --checkpoint checkpoints/ns3_ppo.pt
```

Both runs use identical per-episode ns-3 run numbers (seed, seed+1, ...),
so the policies face the same fading realisations.

### Results (80k steps, C2n = 1e-13, strong turbulence)

Training: 800 episodes in 7 m 44 s wall; mean episode reward rose from
-1183 (first 10% of episodes) to -732 (last 10%), plateauing around
episode 200 (`plots/training_curve.png`). Evaluation on 10 held-out
episodes (seeds 100-109, identical for both policies):

| policy | episode reward | PHY drops/episode | flow PDR |
|---|---|---|---|
| random | -1233.0 +/- 26.6 | 805.3 +/- 19.8 | 0.670 +/- 0.008 |
| PPO (greedy, `checkpoints/ns3_ppo.pt`) | **-730.6 +/- 16.1** | **675.0 +/- 16.1** | **0.724 +/- 0.007** |

The trained policy beats random by ~41% episode reward on every seed —
partly by dropping ~16% fewer packets (it holds the best 2-hop route
instead of wandering onto worse ones) and partly by not paying the
route-flap penalty that random incurs on ~3/4 of steps. The committed
checkpoint (158 KB) reproduces the table via the eval commands above.

Programmatic use of the factory:

```python
from ns3_env import make_ns3_env
from train import TrainConfig, train
train(TrainConfig(seed=42), env_factory=lambda: make_ns3_env(c2n="1e-13", seed=42))
```

## Test

```bash
cd agent
.venv/bin/python -m pytest tests/ --cov=. --cov-report=term
```

The suite validates GAE against a hand-computed example, checkpoint
round-trips, seeded rollout reproducibility, numerical stability
(no NaN/inf in losses or gradients), and that reward improves and policy
entropy falls on the toy env — certifying the PPO pipeline before
ns-3 integration.
