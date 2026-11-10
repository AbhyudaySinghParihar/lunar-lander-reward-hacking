# Lunar Lander: Reward Shaping and Hidden Reward Hacking

I trained a PPO agent to land a rocket in Gymnasium's **LunarLander-v3**, then changed the reward to see how the agent's behaviour changed. One "sensible" bonus got hacked: the agent landed, then kept pulsing its side engines so the episode never ended, farming points until the time limit. The standard way of testing agents (deterministic evaluation) **completely hid this hack**. It only showed up when the agent was tested the way it was trained, with sampled actions.

This repository has the code and the results of that pilot study.

## Results

Every agent was trained once for 1,000,000 timesteps, then tested on the **real game score** (no added bonuses or fees) on 20 held-out starts (seeds 100 to 119).

- **Dice off:** `deterministic=True`, so the agent always presses its most likely action. This is the usual way to test.
- **Dice on:** `deterministic=False`, so the agent samples from its action probabilities, the same as during training.

| Run | Reward | Dice off: score / length / hit time limit | Dice on: score / length / hit time limit |
|---|---|---|---|
| Random agent | none | −206.2 average (10 episodes) | – |
| 1 | Default game reward | 235.2 / 478 / 1 of 20 | not measured |
| 2 | Fee of 0.05 per frame | 264.3 / 311 / 0 of 20 | 256.5 / 460 / 1 of 20 |
| 3 | Fee + hover bonus (+1 per upright, slow frame) | 265.0 / 312 / 0 of 20 | **134.5 / 838 / 16 of 20** |
| 4 | Fee + improvement bonus (potential-based) | **270.6 / 293 / 0 of 20** | **269.6 / 273 / 0 of 20** |

A score of 200 or more counts as solving Lunar Lander.

## What happened

**Run 1 (default reward).** It solved the game, but on one start (seed 105) it touched down just outside the pad, then kept pulsing engine 3 to slide inside. The game only counts a landing (+100) once the lander is completely at rest, so the pulsing kept it "awake" until the time limit. After touchdown at frame 361 it chose "do nothing" 346 times and engine 3 293 times, ending with 120.6 instead of about 220.

**Run 2 (time fee).** A small cost per frame made the agent land about 35% faster (311 frames against 478) and removed the stall in testing.

**Run 3 (hover bonus): reward hacking.** I added +1 for every frame the lander was upright and slow (tilt, sideways speed and vertical speed all under 0.1). The aim was to encourage gentle landings. The agent found that sitting on the ground is the easiest way to be upright and slow. In one dice-on episode (seed 100) it spent 795 of 1,000 frames in the bonus zone and 799 with both legs down, alternating the left and right engines so their spins cancelled and it stayed upright.

With dice off it looked perfect (0 of 20 time limits). With dice on it stalled in 16 of 20 episodes. The habit was in the agent all along, in the lower-probability "tap an engine" actions. Deterministic testing never chose them.

**Run 4 (improvement bonus): the fix.** Instead of paying for *being* upright and slow, this pays for *getting* more upright and slower:

```
bonus = 10 × (potential(now) − potential(previous frame))
potential = −(|tilt| + |sideways speed| + |vertical speed|)
```

Over an episode, these bonuses add up to just `10 × (end − start)`, so sitting still earns nothing. `check_improvement_bonus.py` checks this. On our test episode, both numbers came out at 11.1195. This is potential-based reward shaping (Ng, Harada and Russell, 1999). Run 4 never stalled in either mode and scored highest.

## Videos

| Video | What it shows |
|---|---|
| [0_random_agent_crash.mp4](videos/0_random_agent_crash.mp4) | Random actions. Crashes in about 1 second (score −131.0) |
| [1_run1_landing_seed0.mp4](videos/1_run1_landing_seed0.mp4) | Run 1, dice off. Clean landing between the flags (267.8, 426 frames) |
| [2_run1_ground_stall_seed105.mp4](videos/2_run1_ground_stall_seed105.mp4) | Run 1, dice off, seed 105. Lands at the left flag, then pulses engine 3 until the time limit (120.6, 1,000 frames) |
| [3_run3_dice_off_clean_landing_seed100.mp4](videos/3_run3_dice_off_clean_landing_seed100.mp4) | Run 3, dice off. Looks perfect: lands straight in the centre (233.6, 327 frames) |
| [4_run3_dice_on_reward_hacking_seed100.mp4](videos/4_run3_dice_on_reward_hacking_seed100.mp4) | The same Run 3 agent and start, dice on. Lands, then farms the hover bonus until the time limit (1,000 frames) |

Videos 3 and 4 are the core result: the same trained agent on the same start, tested two ways.

## Trained models

The four pilot agents are in `models/`. Load one with Stable-Baselines3 and test it:

```python
from stable_baselines3 import PPO
from evaluate import evaluate

model = PPO.load("models/run3_hover_ppo_hover.zip")
evaluate(model, det=True)    # dice off
evaluate(model, det=False)   # dice on
```

| File | Run |
|---|---|
| `run1_default_ppo_lander.zip` | Run 1, default reward |
| `run2_fee_ppo_timefee.zip` | Run 2, time fee |
| `run3_hover_ppo_hover.zip` | Run 3, fee + hover bonus |
| `run4_improve_ppo_fixed.zip` | Run 4, fee + improvement bonus |

## Limitations

- Each version was trained **once**. Reinforcement learning varies a lot from run to run, so the differences between Runs 2, 3 and 4 need repeated training runs before they can be called solid.
- Only 20 test episodes per mode. The dice-on tests in the pilot were not seeded, so repeating them gives slightly different numbers (Run 3 gave 17 of 20 the first time, 16 the second).
- One environment, one algorithm (PPO), discrete actions.
- The improvement bonus uses `new − old` rather than the textbook `gamma × new − old`. With gamma = 0.999 the difference is very small.

## Files

| File | What it does |
|---|---|
| `lunar_lander_walkthrough.ipynb` | Step-by-step Colab notebook: exploring the game, the random baseline, training, testing and the reward experiments |
| `wrappers.py` | The three reward wrappers: `TimePenalty`, `HoverBonus`, `ImprovementBonus` |
| `train.py` | Trains one agent under a chosen reward condition |
| `evaluate.py` | Tests agents with dice off and dice on, inspects one episode, records videos |
| `check_improvement_bonus.py` | Checks that the improvement bonus equals end minus start |
| `models/` | The four trained pilot agents |
| `videos/` | Recorded episodes from the pilot |
| `docs/Lunar_Lander_Research_Glossary.pdf` | Plain-language glossary of the terms used, with the pilot results |

## How to run

The easiest way is Google Colab, so nothing heavy runs on your own computer. Open the notebook in Colab and run the cells in order.

To run the scripts instead:

```bash
pip install swig
pip install -r requirements.txt

python train.py --reward hover --seed 0         # about 20–25 minutes on Colab CPU
python evaluate.py models/ppo_hover_seed0.zip   # dice off and dice on
python check_improvement_bonus.py
```

Reward conditions for `--reward`: `default`, `fee`, `hover`, `improve`.

Training settings (Stable-Baselines3 PPO): 16 parallel environments, `n_steps=1024`, `batch_size=64`, `n_epochs=4`, `gamma=0.999`, `gae_lambda=0.98`, `ent_coef=0.01`, 1,000,000 timesteps.

Note: the pilot runs in the table were trained without a fixed training seed. The scripts here set one (`--seed`), so new runs are repeatable but will not match the table exactly.

## Credits

- The project started from Hari's guide *Training an AI to land a rocket with reward shaping: how it works and how to build it* (@harixganesh).
- Built with [Gymnasium](https://gymnasium.farama.org/) and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).
- Potential-based shaping: A. Y. Ng, D. Harada and S. Russell, "Policy invariance under reward transformations: Theory and application to reward shaping," ICML 1999.

## Author

Abhyuday Singh Parihar, The Daly College, Indore

## License

MIT
