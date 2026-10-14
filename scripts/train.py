"""Train a walking policy with PPO (Stable-Baselines3). Run from the project folder.

    ./scripts/train_v7.sh                                      # the current recipe (see README)
    python scripts/train.py --run-name test --stage walk       # one stage, from scratch
    python scripts/train.py --run-name next --stage pushes --resume runs/juan_v7_4_pushes
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

from biped_rl.robots import portable_model_path
from biped_rl.tasks.locomotion import BipedEnv, EnvConfig
from biped_rl.tasks.locomotion.env_cfg import RewardWeights, RobotConfig
from biped_rl.training.callbacks import SaveVecNormalize, TrainingDashboard, VideoRecorder
from biped_rl.training.ppo import new_model
from biped_rl.training.stages import STAGES, apply_stage, load_curriculum, save_curriculum


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="walk_v1")
    p.add_argument("--model", help="MJCF of the robot (default: Juan)")
    p.add_argument("--feet", nargs=2, help="left and right foot body names")
    p.add_argument("--timesteps", type=int, default=30_000_000)
    p.add_argument("--n-envs", type=int, default=os.cpu_count())
    p.add_argument("--max-speed", type=float, help="upper end of the speed command [m/s]")
    p.add_argument("--swing-ratio", type=float, help=">0.5 gives a flight phase (running)")
    p.add_argument("--no-joint-feedback", action="store_true", help="servos cannot report their angle")
    p.add_argument("--resume", help="run directory to continue from (fine-tuning / curriculum)")
    p.add_argument("--min-std", type=float, default=0.1,
                   help="with --resume and --stage: raise the policy's exploration noise to at least this "
                        "(it shrinks during training; a new stage needs room to try new movements)")
    p.add_argument("--stage", choices=STAGES, help="training stage (see apply_stage); with --resume for "
                   "all but the first")
    p.add_argument("--new-rewards", action="store_true",
                   help="with --resume: use the reward weights and robot settings in the current code")
    p.add_argument("--video-every", type=int, default=1_000_000, help="steps between walking clips")
    p.add_argument("--log-details", action="store_true",
                   help="also chart every single penalty term (for reward tuning)")
    p.add_argument("--lr", type=float, help="learning rate (with --resume: overrides the saved one; "
                   "use smaller values for late fine-tuning, where the policy's noise is tiny)")
    p.add_argument("--reward", nargs="*", default=[], metavar="NAME=VALUE",
                   help="override reward weights for this run (after --stage), e.g. --reward symmetry=3 gait_mismatch=0.3")
    p.add_argument("--command", nargs="*", default=[], metavar="NAME=VALUE",
                   help="override command settings for this run (after --stage), e.g. --command backward_prob=0.3")
    args = p.parse_args()

    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        cfg = EnvConfig.from_dict(json.loads((Path(args.resume) / "config.json").read_text()))
    else:
        cfg = EnvConfig()
    if args.resume and args.new_rewards:
        cfg.rewards = RewardWeights()
        cfg.robot = RobotConfig(model_path=cfg.robot.model_path, foot_bodies=cfg.robot.foot_bodies)
    if args.model:
        cfg.robot.model_path = portable_model_path(args.model)
    if args.feet:
        cfg.robot.foot_bodies = tuple(args.feet)
    if args.max_speed is not None:
        cfg.command.speed_range = (cfg.command.speed_range[0], args.max_speed)
    if args.swing_ratio is not None:
        cfg.command.swing_ratio = args.swing_ratio
    if args.no_joint_feedback:
        cfg.robot.joint_feedback = False
    if args.stage:
        apply_stage(cfg, args.stage)
    # Overrides last, so the stage's settings don't undo them.
    for group, items in (("rewards", args.reward), ("command", args.command)):
        for item in items:
            name, value = item.split("=")
            if not hasattr(getattr(cfg, group), name):
                raise SystemExit(f"unknown {group} setting: {name}")
            setattr(getattr(cfg, group), name, float(value))
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    env_kwargs = {"config": cfg.to_dict()}
    env = make_vec_env("biped_rl:Biped-v0", n_envs=args.n_envs, seed=0,
                       vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
    # Eval on the same randomized env (including pushes): measures robustness, not a best case.
    # With pushes on, evaluation uses the fixed standard test instead of the curriculum (random shoves
    # up to 0.3 m/s per axis, as for juan_v3/v6), so stages and runs are comparable.
    eval_cfg = EnvConfig.from_dict(cfg.to_dict())
    if eval_cfg.randomization.push_velocity > 0:
        eval_cfg.randomization.push_curriculum = False
        eval_cfg.randomization.push_velocity = 0.3
    eval_cfg.randomization.physics_curriculum = False  # evaluation: the full-difficulty robot
    eval_env = DummyVecEnv([lambda: Monitor(BipedEnv(eval_cfg.to_dict()))])

    if args.resume:
        env = VecNormalize.load(str(Path(args.resume) / "vecnormalize.pkl"), env)
        custom = {"learning_rate": args.lr} if args.lr else None
        model = PPO.load(Path(args.resume) / "final_model", env=env, device="cpu",
                         tensorboard_log=str(run_dir), custom_objects=custom)
        print(f"resumed from {args.resume}, learning rate {model.learning_rate}", flush=True)
        if args.stage:
            with torch.no_grad():
                model.policy.log_std.clamp_(min=float(np.log(args.min_std)))
        load_curriculum(env, Path(args.resume))
    else:
        model, env = new_model(cfg, env, tensorboard_log=str(run_dir))
    eval_env = VecNormalize(eval_env, training=False, norm_reward=False)

    callbacks = [
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
        save_curriculum(env, run_dir)
        # flush: stdout is often redirected to a log; after Ctrl+C the workers may hang on close
        print(f"Saved policy and normalization stats to {run_dir} (safe to Ctrl+C again)", flush=True)
        env.close()


if __name__ == "__main__":
    main()
