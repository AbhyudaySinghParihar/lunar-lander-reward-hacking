"""Evaluate trained agents on the plain LunarLander-v3 game.

Every agent is judged on the real game score, with no fee or bonus, on the
same held-out starts (seeds 100 onward). Each test runs twice:

    dice off  deterministic=True   always press the most likely action
    dice on   deterministic=False  sample from the action probabilities

Examples:
    python evaluate.py models/ppo_hover_seed0.zip
    python evaluate.py models/*.zip --episodes 100
"""

import argparse
import glob

import gymnasium as gym
from stable_baselines3 import PPO

from wrappers import in_hover_zone


def evaluate(model, n=20, det=True, first_seed=100, verbose=True):
    """Plays n test episodes and reports score, length and time-limit count."""
    env = gym.make("LunarLander-v3")
    scores, lengths, time_limits, ground_stalls = [], [], 0, 0
    for episode in range(n):
        obs, info = env.reset(seed=first_seed + episode)
        total, frames, done = 0.0, 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=det)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            frames += 1
            done = terminated or truncated
        scores.append(round(float(total), 1))
        lengths.append(frames)
        if truncated:
            time_limits += 1
            if obs[6] == 1 and obs[7] == 1:
                ground_stalls += 1
    env.close()
    result = {
        "mean_score": round(sum(scores) / n, 1),
        "mean_length": round(sum(lengths) / n),
        "time_limits": time_limits,
        "ground_stalls": ground_stalls,
        "episodes": n,
        "scores": scores,
        "lengths": lengths,
    }
    if verbose:
        print(f"  Average score:  {result['mean_score']}")
        print(f"  Average length: {result['mean_length']}")
        print(f"  Hit time limit: {time_limits} out of {n}"
              f" ({ground_stalls} with both legs down)")
    return result


def inspect_episode(model, seed=100, det=False):
    """Plays one episode and counts frames in the hover zone and on the ground."""
    env = gym.make("LunarLander-v3")
    obs, info = env.reset(seed=seed)
    frames, in_zone, on_ground, done = 0, 0, 0, False
    while not done:
        action, _ = model.predict(obs, deterministic=det)
        obs, reward, terminated, truncated, info = env.step(action)
        frames += 1
        in_zone += in_hover_zone(obs)
        on_ground += int(obs[6] == 1 and obs[7] == 1)
        done = terminated or truncated
    env.close()
    print(f"Frames: {frames} | In bonus zone: {in_zone} | Both legs down: {on_ground}")
    return frames, in_zone, on_ground


def record_video(model, folder, seed=0, det=True):
    """Records one episode as an .mp4 in the given folder (needs moviepy)."""
    from gymnasium.wrappers import RecordVideo
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = RecordVideo(env, folder, episode_trigger=lambda e: True)
    obs, info = env.reset(seed=seed)
    total, frames, done = 0.0, 0, False
    while not done:
        action, _ = model.predict(obs, deterministic=det)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        frames += 1
        done = terminated or truncated
    env.close()
    print(f"Score: {total:.1f} | Frames: {frames} | "
          f"Ended by: {'landing/crash' if terminated else 'time limit'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--episodes", type=int, default=20)
    a = p.parse_args()
    paths = [f for pattern in a.models for f in glob.glob(pattern)]
    for path in paths:
        model = PPO.load(path)
        print(f"\n=== {path} ===")
        print("Dice OFF:")
        evaluate(model, a.episodes, det=True)
        print("Dice ON:")
        evaluate(model, a.episodes, det=False)
