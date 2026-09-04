import numpy as np


class CliffGridWorld:
    """
    Simple 4x4 grid world.
    States are numbered 0-15, row-major (row * 4 + col):

        0  1  2  3
        4  5  6  7
        8  9 10 11
       12 13 14 15

    Cell 8  -> CLIFF: episode ends, reward -100
    Cell 15 -> GOAL:  episode ends, reward +10
    Every other step: reward -1 (pushes the agent toward shorter paths)

    Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    """

    def __init__(self, start_state=0, cliff_state=8, goal_state=15, max_steps=20):
        self.n_states = 16
        self.n_actions = 4
        self.grid_size = 4
        self.start_state = start_state
        self.cliff_state = cliff_state
        self.goal_state = goal_state
        self.max_steps = max_steps
        self.state = None
        self.steps_taken = 0

    def reset(self):
        self.state = self.start_state
        self.steps_taken = 0
        return self.state

    def step(self, action):
        row, col = divmod(self.state, self.grid_size)

        if action == 0:      # UP
            row = max(row - 1, 0)
        elif action == 1:    # DOWN
            row = min(row + 1, self.grid_size - 1)
        elif action == 2:    # LEFT
            col = max(col - 1, 0)
        elif action == 3:    # RIGHT
            col = min(col + 1, self.grid_size - 1)
        else:
            raise ValueError(f"Invalid action: {action}")

        self.state = row * self.grid_size + col
        self.steps_taken += 1

        reward = -1.0
        done = False

        if self.state == self.cliff_state:
            reward = -100.0
            done = True
        elif self.state == self.goal_state:
            reward = 10.0
            done = True
        elif self.steps_taken >= self.max_steps:
            done = True

        return self.state, reward, done, {}

    def state_to_onehot(self, state):
        vec = np.zeros(self.n_states, dtype=np.float32)
        vec[state] = 1.0
        return vec

    @staticmethod
    def action_name(action):
        return {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}[action]
