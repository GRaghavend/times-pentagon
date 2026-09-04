# TD

Temporal-Difference methods on the shared 4x4 grid-world in [`test_environment/`](../test_environment/README.md) (`GridWorldEnv`, `is_slippery=True`). Unlike Monte-Carlo, TD methods update their value estimate after every single step instead of waiting for the episode to end — they bootstrap off their own current estimate of the next state/action rather than summing up a full observed return.

## Files

- `TD-Pred.py` — TD(0) prediction. Evaluates a *fixed* policy (imported from `Planning_algos/value_iteration.py`'s `policy_import`) by updating `V(state)` one step at a time. Does not change the policy — same job as `Monte-Carlo/MC-pred.py`, so the two `V` tables can be compared.
- `SARSA.py` — on-policy TD control. Learns `Q(s, a)` using the action the epsilon-greedy policy *actually takes next* — the policy being learned and the policy generating behavior are the same one. Named after the quintuple it consumes each step: State, Action, Reward, State', Action'.
- `Q-learn.py` — off-policy TD control. Learns `Q(s, a)` using the *greedy* action at the next state (`max` over `Q[next_state]`), regardless of what the epsilon-greedy behavior policy actually does next. That decoupling of "policy that acts" from "policy being learned" is what makes it off-policy.

## The updates

All three use the same step size `ALPHA = 0.1` and discount `GAMMA = 0.9`, and share the same TD-error pattern: `new_estimate = old_estimate + ALPHA * (target - old_estimate)`. They differ only in what `target` bootstraps off:

- **TD(0) prediction**: `target = reward + GAMMA * V[next_state]`
- **SARSA**: `target = reward + GAMMA * Q[next_state][next_action]`, where `next_action` comes from the same epsilon-greedy policy that picked the current action.
- **Q-learning**: `target = reward + GAMMA * max(Q[next_state])`, the best possible action at the next state, not the one the behavior policy will actually take.

Both `SARSA.py` and `Q-learn.py` use epsilon-greedy action selection with `epsilon = 1 / (episode_number + 1)` (GLIE — decays toward pure exploitation as episodes go on), and both derive a final greedy policy from the learned `Q` table, exposed as `policy_import` so other algorithms can reuse it.

## Usage

```bash
python TD/TD-Pred.py
python TD/SARSA.py
python TD/Q-learn.py
```

Each prints the per-step state/action trace as it learns, then a final table (`V` or `Q`) and, for the control algorithms, the greedy policy grid derived from it.
