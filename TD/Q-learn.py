# ---- Environment must be Gym-style here, NOT simulate() ----
# Q-Learning is OFF-POLICY TD CONTROL: it learns Q(s,a) using the GREEDY
# action at the next state (max over Q[next_state]), regardless of which
# action the epsilon-greedy behavior policy actually takes next. That
# decoupling of "policy that acts" from "policy being learned" is what
# makes it off-policy - contrast with SARSA.py, which is on-policy.

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_environment.gridworld_env import GridWorldEnv

env = GridWorldEnv(is_slippery=True)
actions = env.action_space

# ---- Q table -> nested dict, state -> action -> value ----
Q = {state: {action: 0.0 for action in actions} for state in env.observation_space}

GAMMA = 0.9
ALPHA = 0.1     # step size / learning rate for the TD(0) update
NUM_EPISODES = 10


# ---- epsilon-greedy action selection over the current Q table ----
# This is only the BEHAVIOR policy here - it picks what action gets taken,
# but has no say in what the update target bootstraps off of (see below).
def epsilon_greedy_action(state, epsilon):
    if random.random() < epsilon:
        return random.choice(actions)
    best_value = max(Q[state].values())
    best_actions = [a for a, v in Q[state].items() if v == best_value]
    return random.choice(best_actions)


for episode_number in range(NUM_EPISODES):
    epsilon = 1.0 / (episode_number + 1)   # GLIE: decays toward 0 as episodes -> infinity
    print(f"\n----- Episode {episode_number + 1}/{NUM_EPISODES} -----")

    state = env.reset_random()
    while True:
        action = epsilon_greedy_action(state, epsilon)   # behavior policy picks A
        print(f"State: {state}")
        print(f"Action taken: {env.ACTION_NAMES[action]}")

        next_state, reward, terminated, truncated, info = env.step(action)

        # ---- Q-learning update ----
        # Target bootstraps off the GREEDY action at s' (max Q), not off
        # whatever the epsilon-greedy behavior policy would pick next.
        best_next_value = max(Q[next_state].values())
        Q[state][action] = Q[state][action] + ALPHA * (
            reward + GAMMA * best_next_value - Q[state][action]
        )

        state = next_state
        if terminated or truncated:
            break

print("\n===== Learned Q table (after all episodes) =====")
header = "State".ljust(10) + "".join(f"Action {a}".rjust(12) for a in actions)
print(header)
for state in env.observation_space:
    row = str(state).ljust(10) + "".join(f"{Q[state][action]:>12.3f}" for action in actions)
    print(row)

# Final greedy policy derived from Q: state -> best action.
non_terminal_states = [s for s in env.observation_space if s != env.goal_state]
policy_import = {state: max(Q[state], key=Q[state].get) for state in non_terminal_states}

print("\n===== Policy grid (row = i, column = j) =====")
header = "".ljust(6) + "".join(f"col {j}".rjust(10) for j in range(env.cols))
print(header)
for i in range(env.rows):
    row = f"row {i}".ljust(6)
    for j in range(env.cols):
        state = (i, j)
        cell = "GOAL" if state == env.goal_state else env.ACTION_NAMES[policy_import[state]]
        row += cell.rjust(10)
    print(row)
