# ---- Environment must be Gym-style here, NOT simulate() ----
# TD-Prediction is JUST POLICY EVALUATION, same job as MC-pred.py, but it
# updates V(state) after every SINGLE STEP instead of waiting for the whole
# episode to end. It bootstraps off its own current estimate of the next
# state (V[next_state]) rather than summing up a full observed return.
# There is no fixed policy IMPROVEMENT here - the policy below never changes.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_environment.gridworld_env import GridWorldEnv
from Planning_algos.value_iteration import policy_import

env = GridWorldEnv(is_slippery=True)

# ---- policy pi -> plain dict, state -> action ----
# TD-Prediction EVALUATES this fixed policy, it does not learn/change it.
# Same policy MC-pred.py evaluates, so the two V tables can be compared.
policy = policy_import

# ---- V table -> dict keyed by state ----
V = {state: 0.0 for state in env.observation_space}

GAMMA = 0.9
ALPHA = 0.1     # step size / learning rate for the TD(0) update
NUM_EPISODES = 10


print("===== Policy grid being evaluated (row = i, column = j) =====")
header = "".ljust(6) + "".join(f"col {j}".rjust(10) for j in range(env.cols))
print(header)
for i in range(env.rows):
    row = f"row {i}".ljust(6) + "".join(f"{env.ACTION_NAMES[policy[(i, j)]]:>10}" for j in range(env.cols))
    print(row)

for episode_number in range(NUM_EPISODES):
    print(f"\n----- Episode {episode_number + 1}/{NUM_EPISODES} -----")

    state = env.reset_random()
    while True:
        action = policy[state]
        print(f"State: {state}")
        print(f"Action taken: {env.ACTION_NAMES[action]}")

        next_state, reward, terminated, truncated, info = env.step(action)

        # ---- TD(0) update, applied immediately, one step at a time ----
        # TD target = reward + GAMMA * V[next_state]   (bootstraps off the CURRENT V estimate)
        # TD error  = TD target - V[state]
        V[state] = V[state] + ALPHA * (reward + GAMMA * V[next_state] - V[state])

        state = next_state
        if terminated or truncated:
            break

# This is not just V, but V^pi(state) - the value of state using policy pi
print("\n===== V^pi(state) table (row = i, column = j) =====")
header = "".ljust(6) + "".join(f"col {j}".rjust(10) for j in range(env.cols))
print(header)
for i in range(env.rows):
    row = f"row {i}".ljust(6) + "".join(f"{V[(i, j)]:>10.3f}" for j in range(env.cols))
    print(row)
