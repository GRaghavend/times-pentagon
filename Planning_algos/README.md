# Planning Algorithms

Dynamic-programming solutions to the shared 4x4 grid-world in [`test_environment/`](../test_environment/README.md), where the transition function is known and given explicitly (unlike the model-free methods in `Monte-Carlo/`, `DQN/`, and `A2C/`).

## Files

- `value_iteration.py` — computes the optimal value function directly via the Bellman optimality update, then derives the greedy policy from it.
- `policy_iteration.py` — alternates between policy evaluation (computing V for the current policy) and policy improvement (greedily updating the policy from V), until the policy stops changing.

Both scripts import `GridWorldEnv` from `test_environment`: a 4x4 grid, actions are `0`/`1`/`2`/`3` (right/down/left/up), reward is `10` at the goal `(3,3)` and `-1` otherwise, discount factor `gamma = 0.9`.

Each script also exposes its final result as a module-level `policy_import` dict (`state -> action`), so other algorithms (e.g. `Monte-Carlo/MC-pred.py`) can import and evaluate the policy it produced.

## Usage

```bash
python "Planning_algos/value_iteration.py"
python "Planning_algos/policy_iteration.py"
```

Each prints its convergence progress and the resulting optimal policy per grid cell.
