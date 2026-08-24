# ---- Environment must be Gym-style here, NOT simulate() ----
# On-policy MC control LEARNS from sampled episodes, same as MC-pred, but it
# also improves the policy it's sampling from as it goes.

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_environment.gridworld_env import GridWorldEnv

env = GridWorldEnv()
actions = env.action_space

# ---- Q table -> nested dict, state -> action -> value ----
Q = {state: {action: 0.0 for action in actions} for state in env.observation_space}
# Returns(s,a) -> list of first-visit G's seen for that state-action pair
returns = {(state, action): [] for state in env.observation_space for action in actions}

GAMMA = 0.9
NUM_EPISODES = 3


# ---- epsilon-greedy action selection over the current Q table ----
# With prob epsilon pick a random action (explore), else pick the best
# known action for this state (exploit). Ties broken randomly so no
# single action is favored just because it happens to come first.
def epsilon_greedy_action(state, epsilon):
    if random.random() < epsilon:
        return random.choice(actions)
    best_value = max(Q[state].values())
    best_actions = [a for a, v in Q[state].items() if v == best_value]
    return random.choice(best_actions)


# ---- episode generation, same shape as MC-pred's generate_episode ----
def generate_episode(epsilon):
    episode = []                    # list of (state, action, reward)
    state = env.reset_random()
    while True:
        action = epsilon_greedy_action(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if terminated or truncated:
            break
    return episode


# Checks the first-visit condition on the (state, action) pair, not just state.
def state_action_appeared_before(episode, t):
    for earlier_t in range(t):
        if episode[earlier_t][:2] == episode[t][:2]:
            return True
    return False


for episode_number in range(NUM_EPISODES):
    epsilon = 1.0 / (episode_number + 1)   # GLIE: decays toward 0 as episodes -> infinity

    episode = generate_episode(epsilon)

    G = 0   # running return, reset at the start of every episode

    # Walk backward through the episode
    for t in reversed(range(len(episode))):
        state, action, reward = episode[t]
        G = GAMMA * G + reward              # update Gt

        # First-visit check: only update if this (state, action) hasn't shown up earlier in this episode
        if not state_action_appeared_before(episode, t):
            returns[(state, action)].append(G)
            Q[state][action] = sum(returns[(state, action)]) / len(returns[(state, action)])

print("===== Averaged Q table (after all episodes) =====")
header = "State".ljust(10) + "".join(f"Action {a}".rjust(12) for a in actions)
print(header)
for state in env.observation_space:
    row = str(state).ljust(10) + "".join(f"{Q[state][action]:>12.3f}" for action in actions)
    print(row)

# Final greedy policy derived from Q: state -> best action. Exposed as
# policy_import, same convention Planning_algos uses, so it can be imported
# and compared elsewhere (e.g. against Planning_algos.value_iteration.policy_import).
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
