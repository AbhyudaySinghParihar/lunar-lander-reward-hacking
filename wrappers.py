"""Reward wrappers used in the Lunar Lander reward-shaping experiments.

Each wrapper sits around the LunarLander-v3 environment and changes only the
reward. Observations, actions and episode endings pass through unchanged.

Observation indices (LunarLander-v3):
    0 x position, 1 height, 2 sideways speed, 3 up/down speed,
    4 tilt, 5 spin, 6 left leg touching, 7 right leg touching
"""

import gymnasium as gym


class TimePenalty(gym.Wrapper):
    """Charges a small fee on every frame, so dawdling costs points."""

    def __init__(self, env, fee=0.05):
        super().__init__(env)
        self.fee = fee

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward - self.fee, terminated, truncated, info


def in_hover_zone(obs, threshold=0.1):
    """True when the lander is upright and slow (the hover-bonus condition)."""
    return (abs(obs[4]) < threshold
            and abs(obs[2]) < threshold
            and abs(obs[3]) < threshold)


class HoverBonus(gym.Wrapper):
    """State-based bonus: pays +bonus on every frame the lander is upright and slow.

    Because it pays again on every frame the lander stays in the zone, it can
    be farmed. In our pilot the agent landed, then pulsed both side engines so
    the episode never ended, collecting the bonus until the time limit.
    """

    def __init__(self, env, bonus=1.0, threshold=0.1):
        super().__init__(env)
        self.bonus = bonus
        self.threshold = threshold

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if in_hover_zone(obs, self.threshold):
            reward = reward + self.bonus
        return obs, reward, terminated, truncated, info


class ImprovementBonus(gym.Wrapper):
    """Potential-based bonus: pays for getting better, not for being good.

    reward += scale * (potential(now) - potential(previous frame))

    Summed over an episode this equals scale * (potential(end) - potential(start)),
    so staying still earns nothing and the bonus cannot be farmed.

    Note: the textbook form (Ng, Harada and Russell, 1999) uses
    gamma * potential(now) - potential(previous). We use gamma = 1 here.
    """

    def __init__(self, env, scale=10.0):
        super().__init__(env)
        self.scale = scale
        self.last = 0.0

    @staticmethod
    def potential(obs):
        return -(abs(obs[4]) + abs(obs[2]) + abs(obs[3]))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last = self.potential(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        now = self.potential(obs)
        reward = reward + self.scale * (now - self.last)
        self.last = now
        return obs, reward, terminated, truncated, info


def make_wrapper(reward):
    """Returns the wrapper function for one of the named reward conditions."""
    if reward == "default":
        return None
    if reward == "fee":
        return lambda e: TimePenalty(e)
    if reward == "hover":
        return lambda e: HoverBonus(TimePenalty(e))
    if reward == "improve":
        return lambda e: ImprovementBonus(TimePenalty(e))
    raise ValueError(f"Unknown reward condition: {reward}")
