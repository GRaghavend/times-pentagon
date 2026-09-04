# REINFORCE

Vanilla policy-gradient (REINFORCE), trained on the [`TwoArmedBanditEnv`](../test_environment/README.md) in [`test_environment/`](../test_environment/README.md). No baseline, no critic, no bootstrapping, no discounting — every episode is a single action, so the return `G_t` is just the raw reward received.

## Files

- `reinforce.py` — a fully-traced, 10-iteration run. Every quantity in the update is printed out, iteration by iteration: `theta -> probabilities -> sampled action -> reward -> G_t -> log pi(a|s) -> grad_theta log pi(a|s) -> loss -> new theta`. This script is not trying to train efficiently — it exists so the update can be watched happening, not just trusted.

## The policy

Since the bandit has only one state, the "state" fed into the network is a fixed constant input (`1.0`). The network is one `Linear(1 -> 2)` layer producing 2 logits, one per action, turned into action probabilities via a `Categorical` distribution.

## The update, in words

For each pull of the bandit:

1. Read the current probabilities from the policy (`softmax` of the logits).
2. Sample an action from those probabilities — not the highest-probability one, an actual random sample, so the policy keeps exploring.
3. Step the environment and get back a reward. Since each episode is one step, `G_t` (the return) is just that reward.
4. Compute `log pi(a|s)`, the log-probability the policy assigned to the action that was actually taken.
5. The REINFORCE update pushes `theta` in the direction that makes a good action more likely and a bad one less likely:

   ```
   theta <- theta + alpha * G_t * grad_theta[ log pi_theta(a_t | s_t) ]
   ```

   In words: if the reward was high, increase the probability of the action taken; if it was low (or negative), decrease it. The size of the push scales with how good or bad the reward was.

6. PyTorch optimizers only do gradient *descent*, so instead of applying that update by hand, the script minimizes:

   ```
   loss = -G_t * log pi_theta(a_t | s_t)
   ```

   Minimizing this loss produces exactly the REINFORCE update above — the minus sign is what flips "descend the loss" into "ascend the expected reward".

## Usage

```bash
python REINFORCE/reinforce.py
```

Prints, for each of the 10 iterations: theta before the update, the sampled action and reward, `G_t`, the log-probability and its gradient, the loss, the gradient actually applied, theta after the update, and how the probability of the chosen action moved. Ends with a summary comparing the final policy against the known-better action (Machine A).
