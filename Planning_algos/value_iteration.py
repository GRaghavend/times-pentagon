import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_environment.gridworld_env import GridWorldEnv

# ---- Setting up state, action, and value storage ----

env = GridWorldEnv()
states = env.get_all_states()          # every (row, col) position in the grid
actions = env.action_space             # [0, 1, 2, 3] -> right, down, left, up

# V[i][j] = expected total future reward from that cell
V = np.zeros((env.rows, env.cols))     # start all at 0, same as our manual trace

gamma = 0.9   # discount factor, controls how much future reward matters

np.set_printoptions(precision=2, suppress=True, floatmode="fixed")

#VALUE ITERATION
iterations = 10
if __name__ == "__main__":
    print("=" * 40)
    print(f"TRAINING STARTS WITH {iterations} ITERATIONS")
    print("=" * 40)

for iters in range(iterations):
    for state in states:
        i,j = state
        action_values = []
        for action in actions: # 0, 1, 2, 3

            new_state, reward = env.simulate(state,action)
            ni,nj = new_state

            value_for_that_action = reward + gamma * V[ni][nj]
            action_values.append(value_for_that_action)

        V[i][j] = max(action_values)

    if __name__ == "__main__":
        print(f"\n--- Iteration {iters + 1:>2} ---")
        print(V)

if __name__ == "__main__":
    print("\n" + "=" * 40)
    print("VALUE TABLE CONVERGED")
    print("=" * 40)

policy_dict = {}

for state in states:
    i,j = state

    action_values = []
    for action in actions:
        next_state, reward = env.simulate(state,action)
        ni,nj = next_state

        action_value = reward + gamma * V[ni][nj]
        action_values.append(action_value)

    # argmax gives the *index* of the best value,
    # not the action itself — index into actions to get the real label
    best_action = actions[np.argmax(action_values)]

    policy_dict[state] = best_action

# Final policy, exposed so other algorithms (e.g. Monte-Carlo prediction) can
# import it and evaluate it against their own value estimates.
policy_import = policy_dict

if __name__ == "__main__":
    print("\n" + "=" * 40)
    print("OPTIMAL POLICY")
    print("=" * 40)
    print("\n===== Policy grid (row = i, column = j) =====")
    header = "".ljust(6) + "".join(f"col {j}".rjust(10) for j in range(env.cols))
    print(header)
    for i in range(env.rows):
        row = f"row {i}".ljust(6) + "".join(f"{env.ACTION_NAMES[policy_dict[(i, j)]]:>10}" for j in range(env.cols))
        print(row)
