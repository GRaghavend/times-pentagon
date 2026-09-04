import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_environment.cliff_env import CliffGridWorld

torch.manual_seed(0)


# =========================================================
# 1. Actor and Critic networks
# =========================================================
class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, state):
        logits = self.net(state)
        return torch.softmax(logits, dim=-1)  # action probabilities


class Critic(nn.Module):
    def __init__(self, n_states, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)  # scalar state value


# =========================================================
# 2. Rollout: one trajectory, and a batch of 3
# =========================================================
def rollout_trajectory(env, actor, traj_id):
    """Runs ONE episode with the CURRENT actor (stochastic sampling).
    Stores states, actions, rewards, and the action-probability at the
    moment the action was taken (this becomes 'old_prob' for the PPO ratio)."""
    states, actions, rewards, old_probs = [], [], [], []
    path_log = []

    state = env.reset()
    done = False

    while not done:
        state_vec = torch.tensor(env.state_to_onehot(state))
        probs = actor(state_vec)
        dist = Categorical(probs)
        action = dist.sample()

        next_state, reward, done, _ = env.step(action.item())

        states.append(state_vec)
        actions.append(action)
        rewards.append(reward)
        old_probs.append(probs[action].item())

        path_log.append(f"{env.action_name(action.item())}(s{state}->s{next_state}, r={reward})")
        state = next_state

    print(f"  Trajectory {traj_id}: " + " -> ".join(path_log))
    return {"states": states, "actions": actions, "rewards": rewards, "old_probs": old_probs}


def rollout_batch(env, actor, n_trajectories=3):
    print("Rolling out trajectories:")
    return [rollout_trajectory(env, actor, i + 1) for i in range(n_trajectories)]


# =========================================================
# 3. Returns (discounted rewards-to-go)
# =========================================================
def compute_returns(rewards, gamma=0.99):
    returns = []
    running_return = 0.0
    for r in reversed(rewards):
        running_return = r + gamma * running_return
        returns.insert(0, running_return)
    return returns


# =========================================================
# 4. PPO update (critic = MSE, actor = clipped surrogate)
# =========================================================
def ppo_update(actor, critic, actor_opt, critic_opt, batch, clip_eps=0.2):
    all_states, all_actions, all_old_probs, all_returns, all_advantages = [], [], [], [], []

    # --- returns + advantages, per trajectory, using the critic BEFORE any update ---
    for traj in batch:
        returns = compute_returns(traj["rewards"])
        states_tensor = torch.stack(traj["states"])
        with torch.no_grad():
            values = critic(states_tensor)
        advantages = torch.tensor(returns, dtype=torch.float32) - values

        all_states.extend(traj["states"])
        all_actions.extend(traj["actions"])
        all_old_probs.extend(traj["old_probs"])
        all_returns.extend(returns)
        all_advantages.extend(advantages.tolist())

    states_tensor = torch.stack(all_states)
    actions_tensor = torch.stack(all_actions)
    old_probs_tensor = torch.tensor(all_old_probs, dtype=torch.float32)
    returns_tensor = torch.tensor(all_returns, dtype=torch.float32)
    advantages_tensor = torch.tensor(all_advantages, dtype=torch.float32)
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

    # --- snapshot BEFORE update ---
    with torch.no_grad():
        values_before = critic(states_tensor)
        probs_before = actor(states_tensor)
        chosen_probs_before = probs_before.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

    # --- critic loss: MSE(V(s), return) ---
    values_pred = critic(states_tensor)
    critic_loss = nn.functional.mse_loss(values_pred, returns_tensor)
    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    # --- actor loss: clipped surrogate (L_CLIP) ---
    new_probs = actor(states_tensor)
    chosen_new_probs = new_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

    ratio = chosen_new_probs / old_probs_tensor
    surr1 = ratio * advantages_tensor
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_tensor
    actor_loss = -torch.min(surr1, surr2).mean()

    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    # --- snapshot AFTER update ---
    with torch.no_grad():
        values_after = critic(states_tensor)
        probs_after = actor(states_tensor)
        chosen_probs_after = probs_after.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

    # --- print: critic values before vs after ---
    print("\n  --- Critic values (before -> after update) ---")
    for s, ret, vb, va in zip(all_states, returns_tensor, values_before, values_after):
        state_idx = torch.argmax(s).item()
        print(f"    state {state_idx:2d} | return={ret.item():7.2f} | "
              f"V_before={vb.item():7.3f} -> V_after={va.item():7.3f}")

    # --- print: chosen-action probabilities before vs after ---
    print("\n  --- Action probabilities of chosen action (before -> after update) ---")
    for s, a, pb, pa in zip(all_states, all_actions, chosen_probs_before, chosen_probs_after):
        state_idx = torch.argmax(s).item()
        action_name = CliffGridWorld.action_name(a.item())
        print(f"    state {state_idx:2d}, action {action_name:5s} | "
              f"P_before={pb.item():.3f} -> P_after={pa.item():.3f}")

    print(f"\n  actor_loss = {actor_loss.item():.4f}   critic_loss = {critic_loss.item():.4f}")


# =========================================================
# 5. Main training loop
# =========================================================
def main():
    env = CliffGridWorld()
    actor = Actor(env.n_states, env.n_actions)
    critic = Critic(env.n_states)
    actor_opt = optim.Adam(actor.parameters(), lr=0.01)
    critic_opt = optim.Adam(critic.parameters(), lr=0.05)

    n_iterations = 5
    for it in range(1, n_iterations + 1):
        print(f"\n================ Iteration {it} ================")
        batch = rollout_batch(env, actor, n_trajectories=3)
        ppo_update(actor, critic, actor_opt, critic_opt, batch)


if __name__ == "__main__":
    main()
