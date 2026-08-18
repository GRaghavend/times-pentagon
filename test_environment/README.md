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

## Why one env file

Every algorithm training/evaluating on the exact same states, actions, and rewards is what makes their results comparable — e.g. the value estimates MC-Prediction produces for a policy can be checked against the ground-truth values Value Iteration computed for that same policy, and later, MC-Prediction vs. TD-Prediction on the same policy.

## Policy hand-off convention

Planning algorithms (`value_iteration.py`, `policy_iteration.py`) each expose their final result as a module-level `policy_import` dict (`state -> action`). Prediction algorithms import that dict to evaluate a fixed policy instead of hard-coding one, e.g.:

```python
from Planning_algos.value_iteration import policy_import
```
