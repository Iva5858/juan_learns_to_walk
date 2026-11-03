"""Train a policy to get up from lying face down (biped_rl/tasks/getup.py), from scratch.

    python scripts/train_getup.py --run-name getup/juan_getup_1                  # 30M steps, ~1 h
    python scripts/train_getup.py --run-name getup/juan_getup_2 --weights standing=5.0
    python scripts/train_getup.py --run-name getup/juan_getup_2 --resume runs/getup/juan_getup_1 --weights feet_only=1.0

Every --video-every steps it films the current policy getting up (one episode: the start never varies) to
runs/getup/<name>/videos/: step_<k>_stood<seconds>.mp4 and latest.mp4, the seconds it stood on its feet
in a row. TensorBoard (tensorboard --logdir runs/getup): getup/ upright, height, standing share, episodes
that stood 1 s, and the filmed episode's result.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import imageio
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import biped_rl  # noqa: E402,F401  (registers BipedGetUp-v0)
from biped_rl.runs import RUNS, check_name, start_record, write_record  # noqa: E402
from biped_rl.tasks.getup import GetUpEnv, GetUpWeights, config  # noqa: E402
from biped_rl.training.ppo import n_steps_for, new_model  # noqa: E402


class GetUpLog(BaseCallback):
    """Averages over the steps and episodes since the last update, to TensorBoard."""
    def _on_rollout_start(self):
        self.sums, self.n, self.stood, self.episodes = {"upright": 0.0, "height": 0.0, "standing": 0.0, "settled": 0.0}, 0, 0, 0

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            for k in self.sums:
                self.sums[k] += info[k]
            self.n += 1
            if done:
                self.episodes += 1
                self.stood += info["stood_1s"]
        return True

    def _on_rollout_end(self):
        for k, v in self.sums.items():
            self.logger.record(f"getup/{k}", v / max(self.n, 1))
        if self.episodes:
            self.logger.record("getup/episodes_stood_1s_pct", 100 * self.stood / self.episodes)


class GetUpVideo(BaseCallback):
    """Film one deterministic episode every `freq` steps."""
    def __init__(self, cfg, weights, video_dir, freq):
        super().__init__()
        self.env = GetUpEnv(cfg.to_dict(), weights, render_mode="rgb_array", width=320, height=240)
        self.dir, self.freq, self.last = Path(video_dir), freq, 0
        self.dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps - self.last < self.freq:
            return True
        self.last = self.num_timesteps
        norm = self.model.get_vec_normalize_env()
        obs, _ = self.env.reset(seed=0)
        frames, peak_h = [self.env.render()], 0.0
        for i in range(self.env._max_steps):
            action, _ = self.model.predict(norm.normalize_obs(obs), deterministic=True)
            obs, _, _, truncated, info = self.env.step(action)
            peak_h = max(peak_h, info["height"])
            if i % 2:
                frames.append(self.env.render())
            if truncated:
                break
        stood = self.env.episode.best_stood
        self.logger.record("getup/filmed_stood_s", stood)
        self.logger.record("getup/filmed_peak_height", peak_h)
        name = f"step_{self.num_timesteps // 1000:06d}k_stood{stood:.1f}s.mp4"
        imageio.mimsave(self.dir / name, frames, fps=25)
        imageio.mimsave(self.dir / "latest.mp4", frames, fps=25)
        print(f"[video] {name}: peak height {peak_h:.2f} of standing, stood {stood:.1f} s", flush=True)
        return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True, help="getup/<name>")
    p.add_argument("--timesteps", type=int, default=30_000_000)
    p.add_argument("--n-envs", type=int, default=2 * os.cpu_count())
    p.add_argument("--video-every", type=int, default=1_000_000)
    p.add_argument("--seconds", type=float, default=6.0, help="episode length")
    p.add_argument("--weights", nargs="*", default=[], metavar="NAME=VALUE", help="reward weights (GetUpWeights)")
    p.add_argument("--resume", help="a getup run to continue from (its policy and normalization)")
    p.add_argument("--min-std", type=float, default=0.2, help="with --resume: raise the exploration noise to at least this")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    check_name(args.run_name)
    run_dir = RUNS / args.run_name
    if (run_dir / "final_model.zip").exists() and not args.overwrite:
        raise SystemExit(f"{run_dir} already has a policy (--overwrite to reuse the name)")
    run_dir.mkdir(parents=True, exist_ok=True)
    weights = {}
    for item in args.weights:
        k, v = item.split("=")
        if k not in GetUpWeights.__dataclass_fields__:
            raise SystemExit(f"unknown weight {k}: {', '.join(GetUpWeights.__dataclass_fields__)}")
        weights[k] = float(v)
    cfg = config(args.seconds)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    (run_dir / "getup.json").write_text(json.dumps({"weights": {**vars(GetUpWeights()), **weights}}, indent=2))
    args.stage = "getup"
    start_record(run_dir, args)

    env = make_vec_env("biped_rl:BipedGetUp-v0", n_envs=args.n_envs, seed=0, vec_env_cls=SubprocVecEnv,
                       env_kwargs={"config": cfg.to_dict(), "weights": weights})
    if args.resume:
        env = VecNormalize.load(str(Path(args.resume) / "vecnormalize.pkl"), env)
        model = PPO.load(Path(args.resume) / "final_model", env=env, device="cpu", tensorboard_log=str(run_dir),
                         custom_objects={"n_steps": n_steps_for(args.n_envs)})
        with torch.no_grad():  # room to try something new
            floor = torch.full_like(model.policy.log_std, float(np.log(args.min_std)))
            model.policy.log_std.copy_(torch.maximum(model.policy.log_std, floor))
        print(f"resumed from {args.resume}", flush=True)
    else:
        model, env = new_model(cfg, env, tensorboard_log=str(run_dir))
    callbacks = [GetUpLog(), GetUpVideo(cfg, weights, run_dir / "videos", args.video_every),
                 CheckpointCallback(save_freq=max(2_000_000 // args.n_envs, 1), save_path=str(run_dir / "checkpoints"),
                                    save_vecnormalize=True)]
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks, reset_num_timesteps=not args.resume)
    finally:
        model.save(run_dir / "final_model")
        env.save(str(run_dir / "vecnormalize.pkl"))
        write_record(run_dir, finished=datetime.now().isoformat(timespec="seconds"), total_steps=int(model.num_timesteps))
        print(f"Saved policy and normalization stats to {run_dir} (safe to Ctrl+C again)", flush=True)
        env.close()


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
