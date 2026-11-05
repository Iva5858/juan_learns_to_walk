"""Train a walking policy with PPO (Stable-Baselines3). Run from the project folder.

    python scripts/train.py --run-name test/walk --stage walk   # one stage, from scratch
    python scripts/train.py --run-name v7/juan_v7_21_speed --stage pushes --resume runs/v7/juan_v7_13_straight \
        --timesteps 30000000 --lr 1e-4 --reward heading=2.0        # fine-tune, with a changed setting
    tensorboard --logdir runs                                  # charts: see TrainingDashboard

Every setting is in biped_rl/tasks/locomotion/env_cfg.py (task) and biped_rl/training/ppo.py
(PPO). A run is saved in runs/<generation>/<name>/ (docs/runs.md): config.json is the configuration it
used, run.json where it came from (the run it resumed, the command line, the commit).
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan, VecNormalize

from biped_rl.runs import RUNS, check_name, start_record, write_record
from biped_rl.tasks.locomotion import BipedEnv, EnvConfig
from biped_rl.training.callbacks import (
    CurriculumSnapshot,
    SaveVecNormalize,
    SpeedProbe,
    SpeedRamp,
    TrainingDashboard,
    VideoRecorder,
)
from biped_rl.training.guards import RuntimeGuards
from biped_rl.training.ppo import n_steps_for, new_model
from biped_rl.training.stages import STAGES, apply_stage, load_curriculum, save_curriculum


def parse_value(text):
    """Command-line value: "3" -> 3, "0.5" -> 0.5, "1,2" -> (1, 2), "abc" -> "abc"."""
    if "," in text:
        return tuple(parse_value(v) for v in text.split(","))
    if text.lstrip("-").isdigit():
        return int(text)
    try:
        return float(text)
    except ValueError:
        return text


def evaluation_env(cfg, runtime_guards=False):
    """Match the training wrapper stack: SB3 walks both stacks when copying normalization."""
    env = DummyVecEnv([lambda: Monitor(BipedEnv(cfg.to_dict()))])
    return VecCheckNan(env, raise_exception=True, check_inf=True) if runtime_guards else env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True, help="<generation>/<name>, e.g. v7/juan_v7_21_speed (docs/runs.md)")
    p.add_argument("--overwrite", action="store_true", help="allow reusing the name of a run that already has a policy")
    p.add_argument("--timesteps", type=int, default=30_000_000)
    p.add_argument("--n-envs", type=int, default=2 * os.cpu_count(),
                   help="simulation workers (default 2 x cores: 36 on this Mac was 43%% faster than 18)")
    p.add_argument("--resume", help="run directory to continue from (fine-tuning / curriculum)")
    p.add_argument("--seed", type=int, help="explicit training seed (use the same seed for matched pilot arms)")
    p.add_argument("--deadline-monotonic", type=float, help="stop learning gracefully at this local monotonic-clock deadline")
    p.add_argument("--min-std-rad", type=float, help="keep raw exploration std above this physical radian floor")
    p.add_argument("--runtime-guards", action="store_true", help="check finite data/parameters and write a 30-second heartbeat")
    p.add_argument("--min-std", type=float, default=0.1,
                   help="with --resume and --stage: raise the policy's exploration noise to at least this "
                        "(it shrinks during training; a new stage needs room to try new movements). In action units: "
                        "0.1 = ~3 deg at the 0.5 rad action scale")
    p.add_argument("--stage", choices=STAGES, help="training stage (see apply_stage); with --resume for "
                   "all but the first")
    p.add_argument("--video-every", type=int, default=1_000_000, help="steps between walking clips; 0 disables rendering")
    p.add_argument("--eval-every", type=int, default=1_000_000, help="steps between automatic training evaluations")
    p.add_argument("--eval-episodes", type=int, default=10, help="episodes per automatic training evaluation")
    p.add_argument("--video", choices=("walk", "stand"), default="walk",
                   help="what the clips show: walking, or stopping and being shoved while standing")
    p.add_argument("--tb-video", action="store_true", help="also put the clips in TensorBoard (~100 MB per run)")
    p.add_argument("--video-shove", type=float, default=0.4, help="with --video stand: shove strength [m/s]")
    p.add_argument("--speed-ramp", help="START,END: raise the top speed command linearly over the run (m/s), "
                   "whatever the speed curriculum would do (with --speed-probe-every: where does it top out?)")
    p.add_argument("--speed-probe-every", type=int, default=0,
                   help="steps between measurements of the speed reached at 0.2-1.2 m/s commands (speed_probe.csv)")
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
    p.add_argument("--actor", help="hidden layer widths of a new run's policy network, e.g. 512,256,128 "
                   "(default 256,128,64; a resumed run keeps its own)")
    p.add_argument("--log-std-init", type=float, help="a new run's starting exploration noise, log of its std in "
                   "action units (default -1.0: std 0.37; scale it with --env action_scale to keep it in radians)")
    p.add_argument("--init-std-rad", type=float, help="a new run's starting exploration noise in radians, the same for "
                   "every joint whatever its action range (instead of --log-std-init)")
    p.add_argument("--env", nargs="*", default=[], action="extend", metavar="NAME=VALUE",
                   help="override top-level settings for a new run, e.g. --env action_scale=1.0 (not with --resume: "
                        "a trained policy's action range is changed with scripts/rescale_actions.py)")
    args = p.parse_args()
    if args.timesteps <= 0 or args.n_envs <= 0 or args.video_every < 0 or args.eval_every <= 0 or args.eval_episodes <= 0:
        p.error("timesteps, n-envs and evaluation settings must be positive; video-every must be nonnegative")
    if args.seed is not None:
        set_random_seed(args.seed)
    if any(v is not None and (not np.isfinite(v) or v <= 0) for v in (args.deadline_monotonic, args.min_std_rad)):
        p.error("deadline-monotonic and min-std-rad must be positive and finite")

    check_name(args.run_name)
    run_dir = RUNS / args.run_name
    if any((run_dir / f).exists() for f in ("final_model.zip", "best_model.zip")) and not args.overwrite:
        raise SystemExit(f"{run_dir} already holds a trained policy: choose another --run-name (or pass --overwrite)")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = EnvConfig.load(args.resume) if args.resume else EnvConfig()
    if args.stage:
        meaning = (cfg.neutral_pose, np.asarray(cfg.action_scale).tolist())
        apply_stage(cfg, args.stage)
        if args.resume and (cfg.neutral_pose, np.asarray(cfg.action_scale).tolist()) != meaning:
            raise SystemExit(f"--stage {args.stage} changes what {args.resume}'s actions mean (neutral pose or action "
                             "range): start it from scratch, or resume a run of the same kind")
    # Overrides last, so the stage's settings don't undo them.
    for group, items in (("rewards", args.reward), ("command", args.command), ("robot", args.robot),
                         ("randomization", args.randomization)):
        for item in items:
            name, value = item.split("=")
            if not hasattr(getattr(cfg, group), name):
                raise SystemExit(f"unknown {group} setting: {name}")
            setattr(getattr(cfg, group), name, parse_value(value))
    if args.init_std_rad is not None and (args.resume or args.log_std_init is not None):
        raise SystemExit("--init-std-rad is for a new run, instead of --log-std-init")
    if args.log_std_init is not None and args.resume:
        raise SystemExit("--log-std-init is for a new run (a resumed run keeps its noise; see --min-std)")
    if args.actor and args.resume:
        raise SystemExit("--actor sets a new network: only for a new run, not with --resume")
    if args.env and args.resume:
        raise SystemExit("--env changes what the policy's outputs mean: only for a new run, not with --resume")
    for item in args.env:
        name, value = item.split("=")
        if name not in ("action_scale", "action_filter", "history", "episode_seconds"):
            raise SystemExit(f"unknown or unsupported --env setting: {name}")
        setattr(cfg, name, parse_value(value))
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    start_record(run_dir, args)

    env_kwargs = {"config": cfg.to_dict()}
    env = make_vec_env("biped_rl:Biped-v0", n_envs=args.n_envs, seed=args.seed if args.seed is not None else 0,
                       vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
    if args.runtime_guards:
        env = VecCheckNan(env, raise_exception=True, check_inf=True)
    # Eval on the same randomized env (including pushes): measures robustness, not a best case.
    # With pushes on, evaluation uses the fixed standard test instead of the curriculum (random shoves
    # up to 0.3 m/s per axis), so stages and runs are comparable.
    eval_cfg = EnvConfig.from_dict(cfg.to_dict())
    if eval_cfg.randomization.push_velocity > 0:
        eval_cfg.randomization.push_curriculum = False
        eval_cfg.randomization.push_velocity = 0.3
    eval_cfg.randomization.physics_curriculum = False  # evaluation: the full-difficulty robot
    eval_env = evaluation_env(eval_cfg, args.runtime_guards)

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
                floor = torch.full_like(model.policy.log_std, float(np.log(args.min_std)))
                model.policy.log_std.copy_(torch.maximum(model.policy.log_std, floor))
        load_curriculum(env, Path(args.resume))
    else:
        actor = [int(n) for n in args.actor.split(",")] if args.actor else None
        model, env = new_model(cfg, env, tensorboard_log=str(run_dir), actor=actor, log_std_init=args.log_std_init)
        if args.init_std_rad is not None:
            scale = np.broadcast_to(np.asarray(cfg.action_scale, dtype=np.float64), model.policy.log_std.shape)
            with torch.no_grad():
                model.policy.log_std.copy_(torch.as_tensor(np.log(args.init_std_rad / scale), dtype=model.policy.log_std.dtype))
    if args.seed is not None:
        model.set_random_seed(args.seed)
    eval_env = VecNormalize(eval_env, training=False, norm_reward=False)

    curriculum = CurriculumSnapshot()
    callbacks = [
        curriculum,
        TrainingDashboard(cfg, log_details=args.log_details),
        CheckpointCallback(save_freq=max(2_000_000 // args.n_envs, 1),
                           save_path=str(run_dir / "checkpoints"), save_vecnormalize=True),
        EvalCallback(eval_env, n_eval_episodes=args.eval_episodes, eval_freq=max(args.eval_every // args.n_envs, 1),
                     best_model_save_path=str(run_dir), log_path=str(run_dir), deterministic=True,
                     callback_on_new_best=SaveVecNormalize(str(run_dir / "best_vecnormalize.pkl"))),
    ]
    if args.runtime_guards or args.deadline_monotonic is not None or args.min_std_rad is not None:
        callbacks.insert(0, RuntimeGuards(cfg.action_scale, args.deadline_monotonic, args.min_std_rad,
                                         run_dir / "progress.json" if args.runtime_guards else None))
    if args.video_every:
        callbacks.append(VideoRecorder(cfg, run_dir / "videos", freq=args.video_every, mode=args.video, shove=args.video_shove,
                                       tensorboard=args.tb_video))
    if args.speed_ramp:
        start, end = (float(v) for v in args.speed_ramp.split(","))
        callbacks.append(SpeedRamp(start, end, args.timesteps))
    if args.speed_probe_every:
        callbacks.append(SpeedProbe(cfg, run_dir / "speed_probe.csv", args.speed_probe_every))
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks,
                    reset_num_timesteps=not args.resume)
    finally:
        # Saved even on Ctrl+C, so you can stop early and evaluate.
        model.save(run_dir / "final_model")
        env.save(str(run_dir / "vecnormalize.pkl"))
        save_curriculum(env, run_dir, states=curriculum.states)
        write_record(run_dir, finished=datetime.now().isoformat(timespec="seconds"), total_steps=int(model.num_timesteps))
        # flush: stdout is often redirected to a log; after Ctrl+C the workers may hang on close
        print(f"Saved policy and normalization stats to {run_dir} (safe to Ctrl+C again)", flush=True)
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
