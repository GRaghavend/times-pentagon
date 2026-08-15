# A2C (Advantage Actor-Critic)

A single-file Advantage Actor-Critic implementation on CartPole-v1.

## Files

- `actor_critic.py` — network definition (`ActorCritic`: shared trunk, separate actor/critic heads) plus the training loop and a `test_A2C` evaluation helper. Runs top-to-bottom on import (no `if __name__ == '__main__'` guard) — training starts as soon as the script is executed.
- `hyperparameters.yml` — single `cartpole1` config (gamma, learning rate, hidden layer size, episode/step limits).

## Usage

Run from **inside this folder**, since `hyperparameters.yml` is opened via a relative path:

```bash
cd A2C
python actor_critic.py
```

This trains for `max_episodes` (100), printing average reward every 50 episodes, then automatically runs `test_episodes` (10) rendered evaluation episodes.

## How it works

- Full-episode (Monte Carlo) returns are computed per episode, then normalized before computing advantage (`return - value`).
- Actor loss is the policy-gradient term (`-log_prob * advantage`); critic loss is MSE between predicted value and the actual return.
- Both losses are summed and optimized jointly in a single backward pass — there's no separate optimizer per head.
