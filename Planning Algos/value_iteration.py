import numpy as np

class SimpleGridWorld:
    def __init__(self):
        # Grid is 2x2. States are just (row, col) positions.
        self.rows = 4
        self.cols = 4
        self.goal_state = (3, 3)

        # Actions: 0 = move right, 1 = move down
        self.actions = [0, 1]

    def get_all_states(self):
        # Returns every (row, col) position in the grid
        return [(i, j) for i in range(self.rows) for j in range(self.cols)]

    def simulate(self, state, action):
        """
        Given a state and action, returns (next_state, reward).
        This is the transition function — normally hidden in real envs,
        but for Value/Policy Iteration we need it explicitly.
        """
        i, j = state

        if state == self.goal_state:
            # Already at goal, no more movement/reward
            return state, 0

        if action == 0:      # move right
            next_state = (i, j + 1) if j + 1 < self.cols else (i, j)
        elif action == 1:    # move down
            next_state = (i + 1, j) if i + 1 < self.rows else (i, j)

        reward = 10 if next_state == self.goal_state else -1
        return next_state, reward


# ---- Setting up state, action, and value storage ----

env = SimpleGridWorld()
states = env.get_all_states()          # [(0,0), (0,1), (1,0), (1,1)]
actions = env.actions                  # [0, 1]

# V[i][j] = expected total future reward from that cell
V = np.zeros((env.rows, env.cols))     # start all at 0, same as our manual trace

gamma = 0.9   # discount factor, controls how much future reward matters

np.set_printoptions(precision=2, suppress=True, floatmode="fixed")

#VALUE ITERATION
iterations = 10
print("=" * 40)
print(f"TRAINING STARTS WITH {iterations} ITERATIONS")
print("=" * 40)
for iters in range(iterations):
    for state in states:
        i,j = state
        action_values = []
        for action in actions: # 0 and 1

            new_state, reward = env.simulate(state,action)
            ni,nj = new_state

            value_for_that_action = reward + gamma * V[ni][nj]
            action_values.append(value_for_that_action)

        V[i][j] = max(action_values)

    print(f"\n--- Iteration {iters + 1:>2} ---")
    print(V)

print("\n" + "=" * 40)
print("VALUE TABLE CONVERGED")
print("=" * 40)

policy_dict = {}
ACTION_NAMES = {0: "RIGHT", 1: "DOWN "}

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

print("\n" + "=" * 40)
print("OPTIMAL POLICY")
print("=" * 40)
for state, action in policy_dict.items():
    print(f"STATE {state!s:<8} ->  ACTION {ACTION_NAMES[action]}")

