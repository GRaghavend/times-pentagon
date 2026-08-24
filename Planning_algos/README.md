# Planning Algorithms
There are two algorithms called Value Iteration and Policy Iteration. And for heaven's sake they look so same. So I was just wondering what's that factor that stands out distinguishing both race of planning algos. 

Value Iteration can be drawn parallel to ML's training mechanism right? Say we train the model and keep it and then use the model to test. That's the very crux of VI too. Where the V table is the trained model and we dictate a policy from that by choosing the max policy 

Policy iteration is pretty simple. We first initialize a random policy. Then take actions based ONLY on it and simulate in the env. Once that is done, we check for scope of improvement. 
We go back to the env, try all other actions in that state, and check if the action has changed from the initialized action, if it's changed, then policy hasn't converged. 


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
