# DQN

A from-scratch Deep Q-Network implementation trained on Gymnasium environments (CartPole-v1, FlappyBird-v0).

## Files

- `main.py` — training/eval entry point, `Agent` class (env loop, optimization, logging/plotting).
- `dqn.py` — the network: a 2-layer MLP (`state_dim -> fc1_nodes -> action_dim`) producing Q-values per action.
- `experience_replay.py` — fixed-size `deque`-backed replay buffer with random sampling.
- `hyperparameters.yml` — named hyperparameter sets (`cartpole1`, `flappybird`) read by `Agent.__init__`.
- `runs/` — created on first run; holds per-set `.log`, `.pt` (best model weights), and `.png` (reward/epsilon plot) output.

## Setup

Run from the **repo root** (`RL/`), not from inside `DQN/` — `main.py` opens hyperparameters and writes runs using paths relative to the repo root (`DQN/hyperparameters.yml`, `DQN/runs/`).

```bash
cd /Users/raghavender/Projects/RL
source RL/bin/activate
```

## Usage

```bash
# Train (runs indefinitely, Ctrl+C to stop; saves best model + a final graph on exit)
python DQN/main.py cartpole1 --train
python DQN/main.py flappybird --train

# Evaluate a trained model (loads runs/{name}.pt, renders a window)
python DQN/main.py cartpole1
python DQN/main.py flappybird
```

The positional argument must match a top-level key in `hyperparameters.yml`.

## How training works

- Epsilon-greedy policy over a `DQN` network, with a separate target network synced every `network_sync_rate` steps.
- Best-reward checkpoints are written to `runs/{name}.pt` whenever an episode beats the previous best.
- The reward/epsilon plot (`runs/{name}.png`) updates periodically during training (throttled to every 20s after a new best) and is always saved once more when the run stops (Ctrl+C or otherwise).
- `stop_on_reward` in the yaml is effectively unreachable (set very high) — episodes end on environment termination, and training itself only ends when you interrupt it.
- Epsilon decays every episode by `epsilon_decay` down to `epsilon_min`; how many episodes that takes differs a lot between the two hyperparameter sets (`cartpole1` ~6k episodes, `flappybird` ~60k episodes) — expect noisy rewards until epsilon bottoms out.

## Known gaps

- The environment is hardcoded to `FlappyBird-v0` in `main.py` regardless of which hyperparameter set is passed — `cartpole1` won't actually run CartPole unless you swap the `gymnasium.make(...)` line.
- `device` selection only checks CUDA; Apple GPU (`mps`) support is present but currently commented out, since for these small networks (`fc1_nodes` of 10–512, batch size 32) CPU is often as fast or faster than MPS due to per-op dispatch overhead.
