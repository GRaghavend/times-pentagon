"""Minimal two-armed bandit used as a REINFORCE sanity check.

This is deliberately the simplest possible environment: one state, two
actions, one step per episode. It exists to answer a single question — does
a REINFORCE-trained policy learn to prefer the action with the higher
EXPECTED reward — not to be a realistic or interesting environment.

The environment does NOT tell you which machine is best from a single pull:
each pull is one noisy sample from a fixed Normal distribution, and either
machine can occasionally produce a high (or low) reward. Only by pulling
many times and averaging (which is exactly what a policy-gradient algorithm
like REINFORCE does over many episodes) does the higher-mean machine become
distinguishable from the lower-mean one.
"""

import random


class TwoArmedBanditEnv:
    # Actions: 0 = Machine A, 1 = Machine B.
    ACTION_NAMES = {0: "Machine A", 1: "Machine B"}

    # Reward distributions, fixed for the lifetime of the environment.
    # Machine A has the higher expected reward (1.0 vs 0.0) — a correctly
    # trained REINFORCE policy should converge to preferring action 0.
    REWARD_MEANS = {0: 1.0, 1: 0.0}
    REWARD_STD = 1.0

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else random
        # Single state: there is nothing to observe, so 0 is just a placeholder.
        self.state = 0

    def reset(self):
        """Start a new episode. Always returns the (only) state, 0."""
        self.state = 0
        return self.state

    def step(self, action):
        """Pull one arm and end the episode.

        Every call samples a NEW reward from the chosen machine's Normal
        distribution — the same action can yield very different rewards
        from one call to the next.

        Returns:
            observation: always 0 (there is only one state).
            reward: a fresh sample from the chosen machine's distribution.
            terminated: always True — the episode is exactly one action long.
            truncated: always False — episodes never end early for other reasons.
            info: dict with the machine name that was pulled.
        """
        mean = self.REWARD_MEANS[action]
        reward = self.rng.gauss(mean, self.REWARD_STD)

        observation = self.state
        terminated = True
        truncated = False
        info = {"machine": self.ACTION_NAMES[action]}
        return observation, reward, terminated, truncated, info


if __name__ == "__main__":
    env = TwoArmedBanditEnv()

    # A single episode: reset, pick a machine, step, observe the result.
    obs = env.reset()

    action = 0  # 0 = Machine A, 1 = Machine B — pick manually here.
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"Pulled {info['machine']}: reward={reward:.3f}, terminated={terminated}")

    # A single reward tells you almost nothing: Machine B can beat Machine A
    # on any given pull despite having a lower mean. Only many pulls, averaged,
    # reveal which machine is actually better — which is what a policy trained
    # with REINFORCE over many episodes is implicitly estimating.
    for _ in range(5):
        obs = env.reset()
        obs, reward, terminated, truncated, info = env.step(1)  # try Machine B
        print(f"Pulled {info['machine']}: reward={reward:.3f}, terminated={terminated}")
