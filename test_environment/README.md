# test_environment

The one shared grid-world used by every tabular algorithm in this repo (`Planning_algos/`, `Monte-Carlo/`, and the TD methods to come). `DQN/` and `A2C/` are unrelated — they use Gymnasium environments.

## `GridWorldEnv`

A 4x4 grid, 4 actions (`0`=right, `1`=down, `2`=left, `3`=up), reward `10` at the goal `(3, 3)` and `-1` per step otherwise, walls simply block movement.

The same class exposes two interfaces so both families of algorithm can share it:

- **Model-based** (`Planning_algos/`: value/policy iteration) — the transition function is known, queried directly via `simulate(state, action) -> (next_state, reward)` and `get_all_states()`.
- **Model-free, Gym-style** (`Monte-Carlo/`, and TD/SARSA/Q-learning once added) — only experience is available via `reset()` / `step(action)` / `reset_random()`.

```python
from test_environment.gridworld_env import GridWorldEnv

env = GridWorldEnv()
```

Import it with the repo root on `sys.path` (each script does this itself, see any file under `Planning_algos/` or `Monte-Carlo/` for the pattern).

## Stochasticity (`is_slippery`)

`GridWorldEnv(is_slippery=True)` makes `step()` "icy", FrozenLake-style: the intended action happens with probability `slip_prob` (default `0.8`), and the remaining probability is split evenly between the two actions perpendicular to it (e.g. intending RIGHT, you might slide UP or DOWN instead — never backward). Walls still clamp movement at the grid edges regardless of which direction you actually slid.

This only affects `step()`/`reset()`/`reset_random()` — `simulate(state, action)` always stays deterministic, on purpose: `Planning_algos/` needs a known transition function to compute the Bellman update, so it keeps solving the exact deterministic MDP it was written for, unaffected by `is_slippery`. `is_slippery=True` is meant for the sampling-based methods (`Monte-Carlo/`, and TD/SARSA/Q-learning once added), where hundreds/thousands of sampled episodes can actually surface the stochastic outcomes — with `NUM_EPISODES` in the single digits (like `MC-pred.py`/`MC-control.py`'s current demo runs), you likely won't see a slip at all; it needs a longer run to show up.

Default is `is_slippery=False`, so existing scripts are unaffected unless they opt in.

## Why one env file

Every algorithm training/evaluating on the exact same states, actions, and rewards is what makes their results comparable — e.g. the value estimates MC-Prediction produces for a policy can be checked against the ground-truth values Value Iteration computed for that same policy, and later, MC-Prediction vs. TD-Prediction on the same policy.

## Policy hand-off convention

Planning algorithms (`value_iteration.py`, `policy_iteration.py`) each expose their final result as a module-level `policy_import` dict (`state -> action`). Prediction algorithms import that dict to evaluate a fixed policy instead of hard-coding one, e.g.:

```python
from Planning_algos.value_iteration import policy_import
```
