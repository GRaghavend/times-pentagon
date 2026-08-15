import numpy as np
import random 

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
    
env = SimpleGridWorld()

states = env.get_all_states()
actions = env.actions

policy_dict = {state: random.choice(actions) for state in states}

V = np.zeros((4,4))

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
            
        
    
        

