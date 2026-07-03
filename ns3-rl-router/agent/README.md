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

Plugging in the ns3-ai env later:

```python
from train import TrainConfig, train
train(TrainConfig(seed=42), env_factory=make_ns3_env)
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
