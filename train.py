"""Train a PPO agent on LunarLander-v3 under one reward condition.

Examples:
    python train.py --reward default
    python train.py --reward fee
    python train.py --reward hover --seed 1
    python train.py --reward improve --steps 1000000

Reward conditions:
    default  the game's built-in reward
    fee      built-in reward minus 0.05 per frame
    hover    fee + 1.0 per frame when upright and slow (state-based bonus)
    improve  fee + 10 x change in potential (potential-based bonus)

The training log's ep_rew_mean is recorded inside the wrappers, so it always
shows the real game score, without the added fee or bonus.
"""

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from wrappers import make_wrapper


def train(reward="default", seed=0, steps=1_000_000, out=None):
    env = make_vec_env("LunarLander-v3", n_envs=16, seed=seed,
                       wrapper_class=make_wrapper(reward))
    model = PPO("MlpPolicy", env,
                n_steps=1024, batch_size=64, n_epochs=4,
                gamma=0.999, gae_lambda=0.98, ent_coef=0.01,
                seed=seed, verbose=1)
    model.learn(total_timesteps=steps)
    out = out or f"models/ppo_{reward}_seed{seed}"
    model.save(out)
    print(f"Saved model to {out}.zip")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reward", default="default",
                   choices=["default", "fee", "hover", "improve"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    train(a.reward, a.seed, a.steps, a.out)
