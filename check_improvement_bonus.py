"""Checks that the improvement bonus depends only on the start and end states.

Runs the plain game and the ImprovementBonus game side by side with the same
seed and the same actions, adds up the extra reward, and compares it with
scale * (potential(end) - potential(start)). The two numbers should match.

Uses random actions, so no trained model is needed.
"""

import gymnasium as gym

from wrappers import ImprovementBonus

plain = gym.make("LunarLander-v3")
wrapped = ImprovementBonus(gym.make("LunarLander-v3"), scale=10.0)

obs, _ = plain.reset(seed=0)
obs_w, _ = wrapped.reset(seed=0)
plain.action_space.seed(0)
start = wrapped.potential(obs_w)

total_extra, done = 0.0, False
while not done:
    action = plain.action_space.sample()
    obs, r1, terminated, truncated, _ = plain.step(action)
    obs_w, r2, _, _, _ = wrapped.step(action)
    total_extra += r2 - r1
    done = terminated or truncated

end = wrapped.potential(obs_w)
print("Total bonus added:  ", round(total_extra, 4))
print("10 x (end - start): ", round(10 * (end - start), 4))
