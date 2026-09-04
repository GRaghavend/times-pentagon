# PPO

Proximal Policy Optimization, trained on [`CliffGridWorld`](../test_environment/README.md) in [`test_environment/`](../test_environment/README.md). Like `REINFORCE/`, this is a fully-traced educational run — it prints out state values and action probabilities before and after every update, not just a training-loop progress bar.

## Files

- `ppo_demo.py` — actor and critic networks, rollout collection, return/advantage computation, and the clipped PPO update, all in one file with heavy print statements.

## The two networks

PPO keeps two separate neural networks:

- **Actor** — takes a state and outputs a probability for each action. This is the policy: it decides what to do.
- **Critic** — takes a state and outputs a single number, its estimate of how good that state is (the expected future return from it). It does not choose actions; it only judges states.

Both are small: one hidden layer of size 32, `ReLU` in between. The actor's last layer is a `softmax` so its outputs are valid probabilities; the critic's last layer outputs one raw number.

States are fed in as one-hot vectors (a vector of all zeros except a `1` at the index of the current state), since the grid has a fixed, small number of states.

## Step 1: Rolling out trajectories

A trajectory is one full episode played using the actor's *current* probabilities: at each state, an action is sampled (not the most likely one — an actual random draw, so the actor keeps exploring), the environment is stepped, and the state, action, reward, and the probability the actor assigned to that action at the time (`old_prob`) are all recorded. `old_prob` matters later — it is the actor as it was *before* any update, which the PPO loss compares against.

## Step 2: Batching

Instead of updating after a single episode, several trajectories (3 by default) are rolled out first and collected into one batch. All their states, actions, old probabilities, returns, and advantages are then flattened into single lists before the update — the update step below treats the whole batch as one pool of data, not trajectory by trajectory.

## Step 3: Calculating returns

For each trajectory, the return at a given step is that step's reward plus the discounted return of everything that came after it:

```
return_t = reward_t + gamma * return_{t+1}
```

This is computed backward through the trajectory (from the last step to the first), so the return at every step accounts for every reward still to come, discounted by how far away it is. `gamma = 0.99` here.

## Step 4: Calculating advantages

The return tells you how much reward actually followed a state. The critic's value estimate tells you how much reward was *expected* from that state. The advantage is the difference:

```
advantage = return - V(state)
```

A positive advantage means the outcome was better than the critic expected — the action taken there deserves to become more likely. A negative advantage means it did worse than expected — the action should become less likely. Advantages are computed using the critic *before* it gets updated this round, then normalized (mean subtracted, divided by standard deviation) across the whole batch, which keeps the update stable regardless of the raw reward scale.

## Step 5: The critic loss

The critic is trained to predict the return more accurately. Its loss is plain mean-squared error between its predicted value and the actual return observed:

```
critic_loss = mean( (V(state) - return)^2 )
```

Smaller loss means the critic's value estimates are closer to what states actually turned out to be worth.

## Step 6: The actor loss

The actor is trained to make good actions more likely and bad actions less likely, but PPO adds a safeguard: it does not let the actor change too much in a single update, since a large change based on limited data can make the policy worse rather than better.

First, a probability ratio is computed for each action taken: how likely the *current* (being-updated) actor is to take that action now, divided by how likely it was when the action was originally sampled:

```
ratio = new_prob(action) / old_prob(action)
```

A ratio above 1 means the action has become more likely since the data was collected; below 1 means less likely. This ratio is then multiplied by the advantage, and clipped to a fixed range (`1 - 0.2` to `1 + 0.2` by default) to cap how much a single update can push it:

```
surrogate_1 = ratio * advantage
surrogate_2 = clip(ratio, 1 - eps, 1 + eps) * advantage
actor_loss = -mean( min(surrogate_1, surrogate_2) )
```

Taking the minimum of the clipped and unclipped versions means: if the update is already pushing the ratio past the allowed range in the direction the advantage favors, the clipped value is used, and the loss no longer rewards pushing further. This is what keeps each update small and stable. The leading minus sign is there because PyTorch optimizers minimize, and this is written as a maximization of expected advantage.

## Step 7: Applying the update

The critic is updated first (minimizing `critic_loss`), then the actor is updated (minimizing `actor_loss`) — each with its own optimizer and its own learning rate. Both updates use the same batch of rolled-out data; no new environment interaction happens until the next iteration's rollout.

## Usage

```bash
python PPO/ppo_demo.py
```

Runs 5 iterations of: roll out 3 trajectories, then one PPO update. Each iteration prints the path each trajectory took, every state's value before and after the update, every chosen action's probability before and after the update, and the final actor and critic loss values.
