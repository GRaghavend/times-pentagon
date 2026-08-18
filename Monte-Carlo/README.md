# Monte-Carlo

Model-free methods on the shared 4x4 grid-world in [`test_environment/`](../test_environment/README.md), learning from sampled episodes (`reset()`/`step()`) rather than a known transition function.

## Files

- `MC-pred.py` — first-visit Monte-Carlo prediction. Evaluates a *fixed* policy (does not learn/improve it) by averaging observed returns per state. The policy it evaluates is imported from `Planning_algos/value_iteration.py`'s `policy_import`, so the resulting value estimates can be compared against the ground-truth values Value Iteration computed for that same policy.
- Monte-Carlo control (learning the policy, not just evaluating one) is not implemented yet.

## Usage

```bash
python "Monte-Carlo/MC-pred.py"
```
