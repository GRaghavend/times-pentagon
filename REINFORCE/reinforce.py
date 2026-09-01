"""Plain REINFORCE on the two-armed bandit — a fully-traced educational run.

This script is NOT trying to train efficiently. It exists so every quantity
in the REINFORCE update can be watched, iteration by iteration:

    theta -> probabilities -> sampled action -> reward -> G_t
          -> log pi(a|s) -> grad_theta log pi(a|s) -> loss -> new theta

No baseline, no critic, no bootstrapping, no discounting (each episode is a
single action, so G_t is just the raw reward). This is vanilla REINFORCE and
nothing else.

REINFORCE update (the actual math we are implementing):

    theta <- theta + alpha * G_t * grad_theta[ log pi_theta(a_t | s_t) ]

PyTorch optimizers do gradient DESCENT, so instead of applying that update
by hand we minimize:

    Loss = -G_t * log pi_theta(a_t | s_t)

because grad_theta[Loss] = -G_t * grad_theta[log pi_theta(a_t|s_t)], and
"theta <- theta - alpha * grad_theta[Loss]" then becomes exactly the
REINFORCE ascent step above. That sign flip is the whole trick, and this
script prints every intermediate value so it isn't a leap of faith.
"""

import os
import sys

import torch
import torch.nn as nn
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_environment.bandit_env import TwoArmedBanditEnv


# ---------------------------------------------------------------------------
# 1. Policy network
# ---------------------------------------------------------------------------
# There is only one state, so the "state" fed to the network is just a fixed
# constant input (1.0). This keeps the network as small as it can possibly
# be: one Linear layer mapping 1 input -> 2 logits, one per action.
#
#   state (1.0)
#       |
#   Linear(1 -> 2)
#       |
#   2 logits -> Categorical -> P(A), P(B)
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Wrapped in Sequential purely so parameters are named "0.weight" /
        # "0.bias" — matches the naming convention used in bigger networks.
        self.net = nn.Sequential(
            nn.Linear(in_features=1, out_features=2),
        )

    def forward(self, state):
        return self.net(state)  # raw logits, shape (1, 2)


STATE = torch.tensor([[1.0]])  # the bandit's one and only "observation"


# ---------------------------------------------------------------------------
# Small printing helpers, so tensors show up as readable numbers, not
# multi-line tensor dumps.
# ---------------------------------------------------------------------------
def print_theta(policy, title):
    print(title)
    print("-" * len(title))
    for name, param in policy.named_parameters():
        values = [round(v, 4) for v in param.detach().flatten().tolist()]
        print(f"  {name:10s} = {values}")
    print()


def print_grads(named_grads, title):
    print(title)
    print("-" * len(title))
    for name, grad in named_grads:
        values = [round(v, 4) for v in grad.detach().flatten().tolist()]
        print(f"  {name:10s} = {values}")
    print()


def get_probs(policy):
    with torch.no_grad():
        logits = policy(STATE)
        probs = Categorical(logits=logits).probs.squeeze(0)
    return probs[0].item(), probs[1].item()


# ---------------------------------------------------------------------------
# 2. Training loop — fully traced, one iteration = one episode = one pull.
# ---------------------------------------------------------------------------
def run_traced_training(num_iterations=10, learning_rate=0.1, seed=0):
    torch.manual_seed(seed)

    env = TwoArmedBanditEnv()
    policy = PolicyNetwork()
    optimizer = torch.optim.SGD(policy.parameters(), lr=learning_rate)

    param_names = [name for name, _ in policy.named_parameters()]
    rewards_history = []

    print("=" * 70)
    print_theta(policy, "INITIAL THETA")
    p_a, p_b = get_probs(policy)
    print(f"Initial policy: P(A) = {p_a:.4f}, P(B) = {p_b:.4f}")
    print("=" * 70)

    for iteration in range(1, num_iterations + 1):
        print(f"\n########## ITERATION {iteration} ##########\n")

        # ---- 1 & 2: current theta -> current probabilities -----------------
        print_theta(policy, "OLD THETA")
        p_a_before, p_b_before = get_probs(policy)
        print(f"P(A) = {p_a_before:.4f}")
        print(f"P(B) = {p_b_before:.4f}\n")

        # ---- 3: sample an action from pi_theta(.|s) -----------------------
        # Sampling from the Categorical distribution, NOT argmax — REINFORCE
        # needs to explore, and always picking the current best action would
        # never let the policy discover it might be wrong.
        env.reset()
        logits = policy(STATE)
        dist = Categorical(logits=logits)
        action = dist.sample()
        action_name = env.ACTION_NAMES[action.item()]
        print(f"Sampled action: {action.item()} ({action_name})")

        # ---- 4: environment step -> stochastic reward ----------------------
        _, reward, terminated, truncated, info = env.step(action.item())
        print(f"Reward received: {reward:+.4f}  (terminated={terminated})")

        # ---- 5: return G_t --------------------------------------------------
        # Single-step episode, no discounting -> G_t is just the raw reward.
        G_t = reward
        print(f"G_t = {G_t:+.4f}")

        # ---- 6: log pi_theta(a_t | s_t) -------------------------------------
        log_prob = dist.log_prob(action)
        print(f"log pi_theta(a_t | s_t) = {log_prob.item():.4f}")

        # ---- 7: grad_theta log pi_theta(a_t | s_t) --------------------------
        # This is the quantity the user's spec calls "delta log" — that name
        # is NOT standard terminology. The correct name is the gradient of
        # the log-probability of the taken action w.r.t. the policy
        # parameters: grad_theta log pi_theta(a_t | s_t). Computed here
        # directly via autograd, BEFORE it gets multiplied by G_t, so it can
        # be inspected in isolation.
        grad_log_prob = torch.autograd.grad(
            log_prob, policy.parameters(), retain_graph=True
        )
        print_grads(
            zip(param_names, grad_log_prob),
            "GRADIENT OF LOG PROBABILITY  [ grad_theta log pi_theta(a_t | s_t) ]",
        )

        # ---- 8: loss = -G_t * log pi_theta(a_t | s_t) -----------------------
        loss = -G_t * log_prob
        print(f"LOSS = -G_t * log_prob = -({G_t:+.4f}) * ({log_prob.item():.4f}) "
              f"= {loss.item():.4f}")

        # ---- 9: backward + optimizer step -----------------------------------
        # loss.backward() fills param.grad with grad_theta[Loss], which for
        # this scalar loss is exactly -G_t * grad_theta log pi_theta(a_t|s_t)
        # -- i.e. the SAME tensors printed above, just scaled by -G_t.
        optimizer.zero_grad()
        loss.backward()

        applied_grads = [(name, p.grad.clone()) for name, p in policy.named_parameters()]
        print_grads(
            applied_grads,
            "GRADIENT APPLIED TO THETA  [ grad_theta Loss = -G_t * grad_theta log pi ]",
        )

        optimizer.step()  # theta <- theta - lr * grad_theta[Loss]
                           #        = theta + lr * G_t * grad_theta log pi_theta(a_t|s_t)

        # ---- 10: new theta, new probabilities --------------------------------
        print_theta(policy, "NEW THETA")
        p_a_after, p_b_after = get_probs(policy)
        print(f"P(A) = {p_a_after:.4f}")
        print(f"P(B) = {p_b_after:.4f}\n")

        # ---- 11: explicit probability-movement summary ------------------------
        selected_idx = action.item()
        p_selected_before = [p_a_before, p_b_before][selected_idx]
        p_selected_after = [p_a_after, p_b_after][selected_idx]
        change = p_selected_after - p_selected_before

        print("Policy BEFORE update:")
        print(f"  P(A) = {p_a_before:.4f}")
        print(f"  P(B) = {p_b_before:.4f}")
        print("Policy AFTER update:")
        print(f"  P(A) = {p_a_after:.4f}")
        print(f"  P(B) = {p_b_after:.4f}")
        print(f"\nSelected action: {action_name}")
        print(f"G_t: {G_t:+.4f}")
        print(f"\nProbability of selected action ({action_name}):")
        print(f"  {p_selected_before:.4f} -> {p_selected_after:.4f}")
        print(f"Change: {change:+.4f}")

        if G_t > 0 and change > 0:
            print(f"-> Positive return: probability of {action_name} INCREASED, as expected.")
        elif G_t > 0 and change <= 0:
            print(f"-> Positive return but probability of {action_name} did not increase "
                  f"(can happen — the update also shifts the other action's mass).")
        elif G_t < 0 and change < 0:
            print(f"-> Negative return: probability of {action_name} DECREASED, as expected.")
        else:
            print(f"-> Negative return but probability of {action_name} did not decrease.")

        rewards_history.append(reward)

    return policy, rewards_history


# ---------------------------------------------------------------------------
# 3. Optional longer, untraced run — just to show convergence toward
#    Machine A (the higher expected-reward action) over many more episodes.
#    NOT run by default (see bottom of file) — left here for later use.
# ---------------------------------------------------------------------------
def run_quiet_training(policy, num_iterations=300, learning_rate=0.1, log_every=50):
    env = TwoArmedBanditEnv()
    optimizer = torch.optim.SGD(policy.parameters(), lr=learning_rate)
    rewards_history = []

    for iteration in range(1, num_iterations + 1):
        env.reset()
        logits = policy(STATE)
        dist = Categorical(logits=logits)
        action = dist.sample()
        _, reward, _, _, _ = env.step(action.item())

        G_t = reward
        log_prob = dist.log_prob(action)
        loss = -G_t * log_prob

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rewards_history.append(reward)

        if iteration % log_every == 0:
            p_a, p_b = get_probs(policy)
            print(f"  iter {iteration:4d}: P(A) = {p_a:.4f}, P(B) = {p_b:.4f}, "
                  f"last reward = {reward:+.4f}")

    return rewards_history


if __name__ == "__main__":
    # ---- Part 1: fully traced 10-iteration demonstration ----
    policy, traced_rewards = run_traced_training(num_iterations=10, learning_rate=0.1, seed=0)

    p_a, p_b = get_probs(policy)
    print("\n" + "=" * 70)
    print("FINAL POLICY (after 10 traced iterations)")
    print("-" * 70)
    print(f"P(A) = {p_a:.4f}")
    print(f"P(B) = {p_b:.4f}")
    print("\nExpected optimal action: Machine A  (mean reward 1.0 > 0.0)")
    print(f"Average reward over these {len(traced_rewards)} iterations: "
          f"{sum(traced_rewards) / len(traced_rewards):+.4f}")
    print("=" * 70)

    # ---- Part 2 (optional longer run) is implemented above in
    # run_quiet_training() but left commented out for now — uncomment to
    # see the policy continue converging toward Machine A over many more
    # episodes, without the full per-iteration trace.
    #
    # print("\n\n" + "#" * 70)
    # print("# LONGER RUN (untraced) - showing convergence toward Machine A")
    # print("#" * 70 + "\n")
    #
    # long_rewards = run_quiet_training(policy, num_iterations=300, learning_rate=0.1, log_every=50)
    #
    # p_a, p_b = get_probs(policy)
    # print("\n" + "=" * 70)
    # print("FINAL POLICY (after 10 + 300 iterations)")
    # print("-" * 70)
    # print(f"P(A) = {p_a:.4f}")
    # print(f"P(B) = {p_b:.4f}")
    # print("\nExpected optimal action: Machine A")
    # all_rewards = traced_rewards + long_rewards
    # print(f"Average reward over all {len(all_rewards)} iterations: "
    #       f"{sum(all_rewards) / len(all_rewards):+.4f}")
    # print("=" * 70)
