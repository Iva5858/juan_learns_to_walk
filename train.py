"""Train a walking policy with PPO (Stable-Baselines3).

    python train.py --run-name walk_v1                          # placeholder robot
    python train.py --run-name walk_v1 --model models/my_robot.xml --feet left_foot right_foot
    python train.py --run-name run_v1 --resume runs/walk_v1 --swing-ratio 0.6   # walk -> run curriculum
    tensorboard --logdir runs                                   # rewards per term under rew/

All other knobs (reward weights, servo model, randomization) are in biped_rl/config.py.
The config used for a run is saved to runs/<name>/config.json.
"""
import argparse
import json
import os
from pathlib import Path

import imageio
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import Video
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from biped_rl.config import EnvConfig, RewardWeights, RobotConfig
from biped_rl.env import BipedEnv
from biped_rl.policy import AsymmetricActorCritic


class RewardTermLogger(BaseCallback):
    """Logs the mean of every reward term and gait metric to TensorBoard."""

    KEYS = ("speed_error", "tracking", "penalty_scale", "speed_max", "base_height",
            "peak_torque_stall", "max_joint_speed_ratio", "broke", "fell", "recovering",
            "heading_error_deg", "lateral_offset", "forward_velocity", "flight", "double_support", "peak_force_bw", "saturation", "action_delta")

    def _on_rollout_start(self):
        self._sums, self._count = {}, 0

    def _on_step(self):
        for info in self.locals["infos"]:
            for k, v in info.items():
                if k.startswith(("rew/", "pen/")) or k in self.KEYS:
                    self._sums[k] = self._sums.get(k, 0.0) + float(v)
            self._count += 1
        return True

    def _on_rollout_end(self):
        for k, v in self._sums.items():
            self.logger.record(k if k.startswith(("rew/", "pen/")) else f"gait/{k}", v / max(self._count, 1))


def end_of_clip(frames, fps):
    """Hold the last frame 0.5 s, then fade to black: when a player loops the clip, the jump
    back to the start position is clearly a cut, not the robot moving."""
    last = frames[-1].astype(np.float32)
    fade = [(last * (1 - i / 10)).astype(np.uint8) for i in range(1, 11)]
    return frames + [frames[-1]] * (fps // 2) + fade


class VideoRecorder(BaseCallback):
    """Every `freq` steps, film the current policy (nominal physics, at the speed curriculum's top).

    The clip goes to TensorBoard (IMAGES tab, as an animation) and to <run>/videos/.
    """

    def __init__(self, cfg, video_dir, freq, seconds=8.0, size=(240, 320)):
        super().__init__()
        cfg = EnvConfig.from_dict(cfg.to_dict())
        cfg.randomization.enabled = False
        self.env = BipedEnv(cfg, render_mode="rgb_array", height=size[0], width=size[1])
        self.video_dir, self.freq, self.steps = Path(video_dir), freq, int(seconds / self.env.dt)
        self._last = 0

    def _on_step(self):
        if self.num_timesteps - self._last < self.freq:
            return True
        self._last = self.num_timesteps
        norm = self.model.get_vec_normalize_env()
        speed = max(self.training_env.get_attr("_speed_max"))
        obs, _ = self.env.reset(seed=0, options={"speed": speed})
        start_x = self.env.data.qpos[0]  # nominal physics: target direction is +x
        frames = [self.env.render()]
        for _ in range(self.steps):
            action, _ = self.model.predict(norm.normalize_obs(obs), deterministic=True)
            obs, _, terminated, truncated, _ = self.env.step(action)
            frames.append(self.env.render())
            if terminated or truncated:
                break
        fps = round(1 / self.env.dt)
        distance = self.env.data.qpos[0] - start_x
        frames = end_of_clip(frames, fps)
        self.logger.record("video/distance_m", distance)
        self.logger.record("video/mean_speed", distance / (len(frames) * self.env.dt))
        self.video_dir.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(self.video_dir / f"step_{self.num_timesteps // 1000:06d}k_cmd{speed:.2f}_walked{distance:.2f}m.mp4", frames, fps=fps)
        video = torch.from_numpy(np.stack(frames)[::2]).permute(0, 3, 1, 2).unsqueeze(0)  # (1,T,C,H,W)
        self.logger.record("video/walk", Video(video, fps=fps // 2), exclude=("stdout", "log", "json", "csv"))
        imageio.mimsave(self.video_dir / "latest.mp4", frames, fps=fps)
        return True


class SaveVecNormalize(BaseCallback):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def _on_step(self):
        self.model.get_vec_normalize_env().save(self.path)
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="walk_v1")
    p.add_argument("--model", help="MJCF of the robot (default: placeholder)")
    p.add_argument("--feet", nargs=2, help="left and right foot body names")
    p.add_argument("--timesteps", type=int, default=30_000_000)
    p.add_argument("--n-envs", type=int, default=os.cpu_count())
    p.add_argument("--max-speed", type=float, help="upper end of the speed command [m/s]")
    p.add_argument("--swing-ratio", type=float, help=">0.5 gives a flight phase (running)")
    p.add_argument("--no-joint-feedback", action="store_true", help="servos cannot report their angle")
    p.add_argument("--resume", help="run directory to continue from (fine-tuning / curriculum)")
    p.add_argument("--new-rewards", action="store_true",
                   help="with --resume: use the reward weights and robot settings in the current code")
    p.add_argument("--video-every", type=int, default=1_000_000, help="steps between walking clips")
    p.add_argument("--lr", type=float, help="learning rate (with --resume: overrides the saved one; "
                   "use smaller values for late fine-tuning, where the policy's noise is tiny)")
    p.add_argument("--reward", nargs="*", default=[], metavar="NAME=VALUE",
                   help="override reward weights for this run, e.g. --reward symmetry=3 gait_mismatch=0.3")
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
        cfg.robot.model_path = str(Path(args.model).resolve())
    if args.feet:
        cfg.robot.foot_bodies = tuple(args.feet)
    if args.max_speed is not None:
        cfg.command.speed_range = (cfg.command.speed_range[0], args.max_speed)
    if args.swing_ratio is not None:
        cfg.command.swing_ratio = args.swing_ratio
    for item in args.reward:
        name, value = item.split("=")
        if not hasattr(cfg.rewards, name):
            raise SystemExit(f"unknown reward weight: {name}")
        setattr(cfg.rewards, name, float(value))
    if args.no_joint_feedback:
        cfg.robot.joint_feedback = False
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    env_kwargs = {"config": cfg.to_dict()}
    env = make_vec_env("biped_rl:Biped-v0", n_envs=args.n_envs, seed=0,
                       vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
    # Eval on the same randomized env (including pushes): measures robustness, not a best case.
    eval_env = DummyVecEnv([lambda: Monitor(BipedEnv(cfg.to_dict()))])

    if args.resume:
        env = VecNormalize.load(str(Path(args.resume) / "vecnormalize.pkl"), env)
        custom = {"learning_rate": args.lr} if args.lr else None
        model = PPO.load(Path(args.resume) / "final_model", env=env, device="cpu",
                         tensorboard_log=str(run_dir), custom_objects=custom)
        print(f"resumed from {args.resume}, learning rate {model.learning_rate}", flush=True)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        actor_obs_dim = BipedEnv(cfg.to_dict()).actor_obs_dim
        model = PPO(
            AsymmetricActorCritic,
            env,
            n_steps=2048,          # x n_envs samples per update
            batch_size=4096,
            n_epochs=5,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            max_grad_norm=1.0,
            target_kl=0.02,
            policy_kwargs={
                "actor_obs_dim": actor_obs_dim,
                "net_arch": {"pi": [256, 128, 64], "vf": [512, 256, 128]},
                "log_std_init": -1.0,
                "activation_fn": torch.nn.ELU,
            },
            tensorboard_log=str(run_dir),
            device="cpu",
            verbose=1,
        )
    eval_env = VecNormalize(eval_env, training=False, norm_reward=False)

    callbacks = [
        RewardTermLogger(),
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
        # flush: stdout is often redirected to a log; after Ctrl+C the workers may hang on close
        print(f"Saved policy and normalization stats to {run_dir} (safe to Ctrl+C again)", flush=True)
        env.close()


if __name__ == "__main__":
    main()
