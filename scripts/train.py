"""Train a walking policy with PPO (Stable-Baselines3). Run from the project folder.

    python scripts/train.py --run-name test --stage walk       # one stage, from scratch
    python scripts/train.py --run-name next --stage pushes --resume checkpoints/juan_v7_12b_pushes_best \
        --timesteps 30000000 --lr 1e-4 --reward heading=2.0        # fine-tune, with a changed setting
    tensorboard --logdir runs                                  # charts: see TrainingDashboard

Every setting is in biped_rl/tasks/locomotion/env_cfg.py (task) and biped_rl/training/ppo.py
(PPO). The configuration a run used is saved as runs/<name>/config.json.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from biped_rl.tasks.locomotion import BipedEnv, EnvConfig
from biped_rl.training.callbacks import CurriculumSnapshot, SaveVecNormalize, TrainingDashboard, VideoRecorder
from biped_rl.training.ppo import n_steps_for, new_model
from biped_rl.training.stages import STAGES, apply_stage, load_curriculum, save_curriculum


def parse_value(text):
    """Command-line value: "3" -> 3, "0.5" -> 0.5, "1,2" -> (1, 2)."""
    if "," in text:
        return tuple(parse_value(v) for v in text.split(","))
    return int(text) if text.lstrip("-").isdigit() else float(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--overwrite", action="store_true", help="allow reusing the name of a run that already has a policy")
    p.add_argument("--timesteps", type=int, default=30_000_000)
    p.add_argument("--n-envs", type=int, default=2 * os.cpu_count(),
                   help="simulation workers (default 2 x cores: 36 on this Mac was 43%% faster than 18)")
    p.add_argument("--resume", help="run directory to continue from (fine-tuning / curriculum)")
    p.add_argument("--min-std", type=float, default=0.1,
                   help="with --resume and --stage: raise the policy's exploration noise to at least this "
                        "(it shrinks during training; a new stage needs room to try new movements)")
    p.add_argument("--stage", choices=STAGES, help="training stage (see apply_stage); with --resume for "
                   "all but the first")
    p.add_argument("--video-every", type=int, default=1_000_000, help="steps between walking clips")
    p.add_argument("--log-details", action="store_true",
                   help="also chart every single penalty term (for reward tuning)")
    p.add_argument("--lr", type=float, help="learning rate (with --resume: overrides the saved one; "
                   "use smaller values for late fine-tuning, where the policy's noise is tiny)")
    p.add_argument("--reward", nargs="*", default=[], action="extend", metavar="NAME=VALUE",
                   help="override reward weights for this run (after --stage), e.g. --reward symmetry=3 gait_mismatch=0.3")
    p.add_argument("--command", nargs="*", default=[], action="extend", metavar="NAME=VALUE",
                   help="override command settings for this run (after --stage), e.g. --command backward_prob=0.3")
    p.add_argument("--robot", nargs="*", default=[], action="extend", metavar="NAME=VALUE",
                   help="override robot / servo settings, e.g. --robot no_load_speed=3.0 kp=6")
    p.add_argument("--randomization", nargs="*", default=[], action="extend", metavar="NAME=VALUE",
                   help="override randomization settings, e.g. --randomization action_delay_steps=1,2")
    args = p.parse_args()

    run_dir = Path("runs") / args.run_name
    if any((run_dir / f).exists() for f in ("final_model.zip", "best_model.zip")) and not args.overwrite:
        raise SystemExit(f"{run_dir} already holds a trained policy: choose another --run-name (or pass --overwrite)")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = EnvConfig.load(args.resume) if args.resume else EnvConfig()
    if args.stage:
        apply_stage(cfg, args.stage)
    # Overrides last, so the stage's settings don't undo them.
    for group, items in (("rewards", args.reward), ("command", args.command), ("robot", args.robot),
                         ("randomization", args.randomization)):
        for item in items:
            name, value = item.split("=")
            if not hasattr(getattr(cfg, group), name):
                raise SystemExit(f"unknown {group} setting: {name}")
            setattr(getattr(cfg, group), name, parse_value(value))
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    env_kwargs = {"config": cfg.to_dict()}
    env = make_vec_env("biped_rl:Biped-v0", n_envs=args.n_envs, seed=0,
                       vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
    # Eval on the same randomized env (including pushes): measures robustness, not a best case.
    # With pushes on, evaluation uses the fixed standard test instead of the curriculum (random shoves
    # up to 0.3 m/s per axis), so stages and runs are comparable.
    eval_cfg = EnvConfig.from_dict(cfg.to_dict())
    if eval_cfg.randomization.push_velocity > 0:
        eval_cfg.randomization.push_curriculum = False
        eval_cfg.randomization.push_velocity = 0.3
    eval_cfg.randomization.physics_curriculum = False  # evaluation: the full-difficulty robot
    eval_env = DummyVecEnv([lambda: Monitor(BipedEnv(eval_cfg.to_dict()))])

    if args.resume:
        env = VecNormalize.load(str(Path(args.resume) / "vecnormalize.pkl"), env)
        custom = {"n_steps": n_steps_for(args.n_envs)}  # same samples per update with any worker count
        if args.lr:
            custom["learning_rate"] = args.lr
        model = PPO.load(Path(args.resume) / "final_model", env=env, device="cpu",
                         tensorboard_log=str(run_dir), custom_objects=custom)
        print(f"resumed from {args.resume}, learning rate {model.learning_rate}, "
              f"{args.n_envs} workers x {model.n_steps} steps per update", flush=True)
        if args.stage:
            with torch.no_grad():
                model.policy.log_std.clamp_(min=float(np.log(args.min_std)))
        load_curriculum(env, Path(args.resume))
    else:
        model, env = new_model(cfg, env, tensorboard_log=str(run_dir))
    eval_env = VecNormalize(eval_env, training=False, norm_reward=False)

    curriculum = CurriculumSnapshot()
    callbacks = [
        curriculum,
        TrainingDashboard(cfg, log_details=args.log_details),
        VideoRecorder(cfg, run_dir / "videos", freq=args.video_every),
        CheckpointCallback(save_freq=max(2_000_000 // args.n_envs, 1),
                           save_path=str(run_dir / "checkpoints"), save_vecnormalize=True),
        EvalCallback(eval_env, n_eval_episodes=10, eval_freq=max(1_000_000 // args.n_envs, 1),
                     best_model_save_path=str(run_dir), log_path=str(run_dir), deterministic=True,
                     callback_on_new_best=SaveVecNormalize(str(run_dir / "best_vecnormalize.pkl"))),
    ]
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks,
                    reset_num_timesteps=not args.resume)
    finally:
        # Saved even on Ctrl+C, so you can stop early and evaluate.
        model.save(run_dir / "final_model")
        env.save(str(run_dir / "vecnormalize.pkl"))
        save_curriculum(env, run_dir, states=curriculum.states)
        # flush: stdout is often redirected to a log; after Ctrl+C the workers may hang on close
        print(f"Saved policy and normalization stats to {run_dir} (safe to Ctrl+C again)", flush=True)
        env.close()


if __name__ == "__main__":
    main()
