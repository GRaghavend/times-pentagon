# ---- Environment must be Gym-style here, NOT simulate() ----
# MC-Prediction learns from SAMPLED episodes (experience), it has no
# access to a transition function - so it needs reset()/step(), not simulate().

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_environment.gridworld_env import GridWorldEnv
from Planning_algos.value_iteration import policy_import

env = GridWorldEnv(is_slippery=True)

# ---- Q3: policy pi -> plain dict, state -> action ----
# MC-Prediction EVALUATES this, it does not learn/change it.
# The policy being evaluated comes straight from Value Iteration, so MC-Prediction's
# value estimates can later be compared against the ground-truth values that produced it.
policy = policy_import

# ---- Q2: V and Returns -> dicts keyed by state ----
V = {state: 0.0 for state in env.observation_space}
returns = {state: [] for state in env.observation_space}   # Returns(s): list of G's seen at s

# ---- Q4: episode generation ----
def generate_episode(env, policy):
    episode = []                    # list of (state, action, reward)
    state = env.reset_random()
    while True:
        action = policy[state]
        next_state, reward, terminated, truncated, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if terminated or truncated:
            break
    return episode

# ---- Q1: G(t) -> a single running scalar, NOT a list ----
# It gets recomputed (overwritten) once per backward step of one episode:
GAMMA = 0.9
# is_slippery needs far more episodes than the deterministic case to actually
# surface stochastic outcomes often enough for the returns to average out
# (see test_environment/README.md's note on is_slippery).
NUM_EPISODES = 1000

# Checks the first-visit condition: does this state already appear
# earlier in this same episode (any time step before t)?
def state_appeared_before(episode, t):
    for earlier_t in range(t):
        earlier_state = episode[earlier_t][0]
        if earlier_state == episode[t][0]:
            return True
    return False


print("===== Policy grid being evaluated (row = i, column = j) =====")
header = "".ljust(6) + "".join(f"col {j}".rjust(10) for j in range(env.cols))
print(header)
for i in range(env.rows):
    row = f"row {i}".ljust(6) + "".join(f"{env.ACTION_NAMES[policy[(i, j)]]:>10}" for j in range(env.cols))
    print(row)

for episode_number in range(NUM_EPISODES):
    episode = generate_episode(env, policy)   # full (state, action, reward) list

    G = 0   # running return, reset at the start of every episode

    # Walk backward through the episode
    for t in reversed(range(len(episode))):
        state, action, reward = episode[t]
        G = GAMMA * G + reward              # update Gt

        # First-visit check: only update if this state hasn't shown up earlier in this episode
        if not state_appeared_before(episode, t):
            returns[state].append(G)                          # store this return
            V[state] = sum(returns[state]) / len(returns[state])   # average all returns seen so far

    if (episode_number + 1) % 100 == 0:
        print(f"Episode {episode_number + 1}/{NUM_EPISODES} done")

#This is not just V, but V^π(state)
#The value of state using the policy π
print("\n===== V^pi(state) table (row = i, column = j) =====")
header = "".ljust(6) + "".join(f"col {j}".rjust(10) for j in range(env.cols))
print(header)
for i in range(env.rows):
    row = f"row {i}".ljust(6) + "".join(f"{V[(i, j)]:>10.3f}" for j in range(env.cols))
    print(row)
