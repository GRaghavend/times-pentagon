# RL

Reinforcement learning is pretty much an ocean. I see that when we ultimately reach training world models, reinforcement learning would definitely push our bums back! So, I feel that a deeper understanding of RL algorithms is possible when we go incrementally, step by step, and understand them by acknowledging their use in toy environments like a 2 cross 2 or a 4 cross 4 grids with minimal actions! 

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
