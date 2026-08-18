"""Shared 4x4 grid-world used by every algorithm in this repo (except DQN/A2C).

Two interfaces are exposed on the same class so both families of algorithm can
share one environment:

- Model-based (Planning Algos: value/policy iteration): the transition
  function is known and queried directly via `simulate(state, action)`.
- Model-free, Gym-style (Monte-Carlo, TD, SARSA, Q-learning): only experience
  is available via `reset()` / `step(action)`.

Keeping both on one class means every algorithm trains/evaluates on the exact
same states, actions, and rewards, so their results (e.g. a policy learned by
MC-Prediction vs. TD-Prediction) are directly comparable.
"""

import random


class GridWorldEnv:
    # Actions: 0 = right, 1 = down, 2 = left, 3 = up.
    ACTION_NAMES = {0: "RIGHT", 1: "DOWN", 2: "LEFT", 3: "UP"}

    def __init__(self, rows=4, cols=4, start_state=(0, 0), goal_state=(3, 3),
                 step_reward=-1, goal_reward=10):
        self.rows = rows
        self.cols = cols
        self.start_state = start_state
        self.goal_state = goal_state
        self.step_reward = step_reward
        self.goal_reward = goal_reward

        self.action_space = [0, 1, 2, 3]
        self.observation_space = self.get_all_states()

        self.state = None

    def get_all_states(self):
        return [(i, j) for i in range(self.rows) for j in range(self.cols)]

    def _move(self, state, action):
        i, j = state
        if action == 0:      # right
            j = min(j + 1, self.cols - 1)
        elif action == 1:    # down
            i = min(i + 1, self.rows - 1)
        elif action == 2:    # left
            j = max(j - 1, 0)
        elif action == 3:    # up
            i = max(i - 1, 0)
        return (i, j)

    # ---- model-based interface, used by Planning Algos ----
    def simulate(self, state, action):
        """Given a state and action, returns (next_state, reward) without mutating env state."""
        if state == self.goal_state:
            return state, 0

        next_state = self._move(state, action)
        reward = self.goal_reward if next_state == self.goal_state else self.step_reward
        return next_state, reward

    # ---- model-free, Gym-style interface, used by MC / TD / SARSA / Q-learning ----
    def reset(self, state=None):
        self.state = state if state is not None else self.start_state
        return self.state

    def reset_random(self, rng=random):
        """Start an episode from a random non-terminal state (useful for exploring starts)."""
        non_terminal_states = [s for s in self.observation_space if s != self.goal_state]
        self.state = rng.choice(non_terminal_states)
        return self.state

    def step(self, action):
        next_state, reward = self.simulate(self.state, action)
        terminated = next_state == self.goal_state
        truncated = False
        self.state = next_state
        return next_state, reward, terminated, truncated, {}
