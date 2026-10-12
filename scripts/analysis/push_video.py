"""Videos of the robot being shoved around (randomized physics + random pushes), including falls.

    python scripts/analysis/push_video.py runs/juan_v6                 # up to 4 falls + 1 survived episode
    python scripts/analysis/push_video.py runs/juan_v6 --falls 8 --episodes 60

Clips go to <run>/push_videos/. A red arrow shows each shove (direction and strength); the
red line / white marks on the floor are the start and every 0.5 m along the target line.
"""
import argparse
import json
from pathlib import Path

import imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from biped_rl.tasks.locomotion import BipedEnv, EnvConfig
from biped_rl.training.callbacks import end_of_clip

p = argparse.ArgumentParser()
p.add_argument("run", type=Path)
p.add_argument("--falls", type=int, default=4, help="stop after this many fall clips")
p.add_argument("--episodes", type=int, default=40, help="at most this many episodes")
p.add_argument("--speed", type=float, default=0.4)
p.add_argument("--no-break", action="store_true",
               help="ignore the hardware 'break' limits so episodes run until a real fall (videos only)")
args = p.parse_args()

cfg = EnvConfig.from_dict(json.loads((args.run / "config.json").read_text()))
cfg.randomization.enabled = True
if args.no_break:
    cfg.rewards.break_impact = cfg.rewards.break_joint_speed = 1e9
pol = PPO.load(args.run / "final_model", device="cpu")
norm = VecNormalize.load(str(args.run / "vecnormalize.pkl"), DummyVecEnv([lambda: BipedEnv(cfg.to_dict())]))
out = args.run / ("push_videos_real_falls" if args.no_break else "push_videos")
out.mkdir(exist_ok=True)
fps = round(cfg.control_hz)

falls, saved_ok = 0, False
for ep in range(args.episodes):
    env = BipedEnv(cfg, render_mode="rgb_array", width=480, height=360)
    obs, _ = env.reset(seed=1000 + ep, options={"speed": args.speed})
    frames, pushes, info = [env.render()], 0, {}
    last_push = env.episode.last_push
    for i in range(int(cfg.episode_seconds * cfg.control_hz)):
        a, _ = pol.predict(norm.normalize_obs(obs), deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        frames.append(env.render())
        if env.episode.last_push != last_push:
            pushes, last_push = pushes + 1, env.episode.last_push
        if term or trunc:
            break
    t = len(frames) / fps
    if term:
        cause = "broke" if info.get("broke") else "fell"
        falls += 1
        name = out / f"ep{ep:02d}_{cause}_after_{t:.1f}s_{pushes}pushes.mp4"
        clip = frames[-8 * fps:]  # last 8 s before the fall
    elif not saved_ok:
        saved_ok = True
        name = out / f"ep{ep:02d}_survived_20s_{pushes}pushes.mp4"
        clip = frames
    else:
        print(f"episode {ep:2d}: survived 20 s ({pushes} shoves)")
        env.close()
        continue
    imageio.mimsave(name, end_of_clip(clip, fps), fps=fps)
    print(f"episode {ep:2d}: {'FELL' if term else 'survived'} ({pushes} shoves) -> {name}")
    env.close()
    if falls >= args.falls and saved_ok:
        break
print(f"\n{falls} fall clip(s) in {out}")
