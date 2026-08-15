# Planning Algorithms

Dynamic-programming solutions to a toy 4x4 grid-world, where the transition function is known and given explicitly (unlike the model-free methods in `DQN/` and `A2C/`).

## Files

- `value_iteration.py` — computes the optimal value function directly via the Bellman optimality update, then derives the greedy policy from it.
- `policy_iteration.py` — alternates between policy evaluation (computing V for the current policy) and policy improvement (greedily updating the policy from V), until the policy stops changing.

Both scripts define the same `SimpleGridWorld`: a 4x4 grid, actions are `0` (move right) / `1` (move down), reward is `10` at the goal `(3,3)` and `-1` otherwise, discount factor `gamma = 0.9`.

## Usage

```bash
python "Planning Algos/value_iteration.py"
python "Planning Algos/policy_iteration.py"
```

Each prints its convergence progress and the resulting optimal policy per grid cell.
