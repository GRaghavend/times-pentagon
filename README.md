# RL

Reinforcement learning implementations built from scratch, for learning purposes.

## Contents

- [`DQN/`](DQN/README.md) — Deep Q-Network (CartPole-v1, FlappyBird-v0), with a `main.py` CLI (train/eval modes), replay buffer, and target-network sync.
- [`A2C/`](A2C/README.md) — Advantage Actor-Critic on CartPole-v1.
- [`Planning Algos/`](Planning%20Algos/README.md) — Value Iteration and Policy Iteration on a toy grid-world (known-transition dynamic programming, no learning involved).

Each subfolder has its own README with setup/usage details specific to that algorithm.

## Setup

```bash
python3.12 -m venv RL
source RL/bin/activate
pip install -r requirements.txt
```

`DQN/` additionally needs `flappy-bird-gymnasium` (not in `requirements.txt`):

```bash
pip install flappy-bird-gymnasium
```

Note: this venv is deliberately named `RL`, same as the repo folder itself — it's gitignored, so it won't collide with anything tracked.
