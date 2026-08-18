import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_environment.gridworld_env import GridWorldEnv

env = GridWorldEnv()

states = env.get_all_states()
actions = env.action_space

policy_dict = {state: random.choice(actions) for state in states}

V = np.zeros((env.rows, env.cols))

gamma = 0.9
is_stable = False

ITERATIONS = 10

while not is_stable:
    #POLICY EVALUATION
    for iters in range(ITERATIONS):
        for state in states:
            i,j = state
            action = policy_dict[state]

            new_state, reward = env.simulate(state,action)
            ni,nj = new_state

            V[i][j] = reward + gamma * V[ni][nj]

    #POLICY IMPROVEMENT
    # We initialize it here becoz, we are assuming that this policy is stable but
    # even a small change would prove that this stability is false
    is_stable = True
    for state in states:
        i,j = state
        old_action = policy_dict[state]

        best_action, best_value = None, -float('inf')

        for action in actions:
            new_state, reward = env.simulate(state,action)
            ni,nj = new_state
            curr_action_value = reward + gamma * V[ni][nj]

            if curr_action_value > best_value:
                best_action = action
                best_value = curr_action_value

        policy_dict[state] = best_action
        if best_action != old_action:
            is_stable = False

print(policy_dict)

# Final policy, exposed so other algorithms (e.g. Monte-Carlo prediction) can
# import it and evaluate it against their own value estimates.
policy_import = policy_dict
