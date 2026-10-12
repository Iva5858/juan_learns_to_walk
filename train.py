"""Train a walking policy with PPO (Stable-Baselines3).

    python train.py --run-name walk_v1                          # placeholder robot
    python train.py --run-name walk_v1 --model models/my_robot.xml --feet left_foot right_foot
    python train.py --run-name run_v1 --resume runs/walk_v1 --swing-ratio 0.6   # walk -> run curriculum
    tensorboard --logdir runs                                   # charts: see TrainingDashboard

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

from biped_rl.config import (CommandConfig, EnvConfig, RandomizationConfig, RewardWeights, RobotConfig,
                             portable_model_path)
from biped_rl.env import BipedEnv
from biped_rl.policy import AsymmetricActorCritic
from biped_rl.symmetry import SymmetricRunningMeanStd, check_robot_mirror, mirror_maps


class TrainingDashboard(BaseCallback):
    """TensorBoard charts, grouped and ordered (TensorBoard sorts groups by name):

    1_overview    is it learning: falls, hardware-limit stops, episode length, speed error, top speed
    2_commands    does it do what it's told: error per command axis, heading, path drift
    3_hardware    is it gentle on the servos: torque, joint speed, landing impact, saturation
    4_gait        what the walk looks like: double support, flight, action change, body height
    5_rewards     task terms with a non-zero weight, and the share of reward kept after penalties
    6_penalties   penalties summed into a few groups (log_details=True: also every single term)
    """

    PENALTY_GROUPS = {
        "path": ("path_offset", "stand_still"),
        "posture": ("orientation", "ang_vel_xy", "lin_vel_z", "base_height", "knee_hyperextension",
                    "hip_roll", "hip_roll_inward"),
        "smoothness": ("action_rate", "action_acc", "joint_acc", "reversals"),
        "servo_protection": ("servo_load", "servo_saturation", "servo_overload", "backdrive", "energy",
                             "joint_limits"),
        "feet_and_impacts": ("impact", "foot_slip", "feet_distance", "footstep"),
        "gait_shape": ("symmetry", "posture_symmetry", "step_timing", "gait_mismatch"),
    }

    def __init__(self, cfg, log_details=False):
        super().__init__()
        self.dt = 1.0 / cfg.control_hz
        self.weights = cfg.rewards
        self.log_details = log_details
        self.group_of = {term: g for g, terms in self.PENALTY_GROUPS.items() for term in terms}

    def _on_rollout_start(self):
        self._sums, self._count, self._ends = {}, 0, {"fell": 0, "broke": 0, "survived": 0}
        self._ep_seconds = []

    def _add(self, key, value):
        self._sums[key] = self._sums.get(key, 0.0) + float(value)

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            self._count += 1
            self._add("2_commands/forward_error_mps", info["speed_error"])
            self._add("2_commands/sideways_error_mps", abs(info["lateral_velocity"] - info["command_lateral"]))
            self._add("2_commands/turn_rate_error_radps", abs(info["yaw_rate"] - info["command_yaw_rate"]))
            self._add("2_commands/heading_error_deg", info["heading_error_deg"])
            self._add("2_commands/path_drift_cm", 100 * info["lateral_offset"])
            self._add("1_overview/speed_error_mps", info["speed_error"])
            self._add("1_overview/top_speed_command_mps", info["speed_max"])
            self._add("1_overview/push_strength_mps_avg_direction", info["push_level_mean"])
            self._add("1_overview/push_strength_mps_weakest_direction", info["push_level_min"])
            self._add("1_overview/physics_difficulty", info["physics_difficulty"])
            self._add("3_hardware/peak_torque_x_stall", info["peak_torque_stall"])
            self._add("3_hardware/peak_joint_speed_x_servo_max", info["max_joint_speed_ratio"])
            self._add("3_hardware/landing_force_x_body_weight", info["peak_force_bw"])
            self._add("3_hardware/servo_saturation_pct", 100 * info["saturation"])
            self._add("4_gait/double_support_pct", 100 * info["double_support"])
            self._add("4_gait/flight_pct", 100 * info["flight"])
            self._add("4_gait/action_change", info["action_delta"])
            self._add("4_gait/body_height_m", info["base_height"])
            self._add("5_rewards/kept_after_penalties", info["penalty_scale"])
            for k, v in info.items():
                if k.startswith("rew/") and getattr(self.weights, k[4:], 1.0) != 0:
                    self._add(f"5_rewards/{k[4:]}", v)
                elif k.startswith("pen/"):
                    self._add(f"6_penalties/{self.group_of.get(k[4:], 'other')}", v)
                    if self.log_details:
                        self._add(f"7_penalty_details/{k[4:]}", v)
            if done:  # VecEnv: info of the final step; truncated = ran out of time = survived
                end = ("survived" if info.get("TimeLimit.truncated") else
                       "broke" if info.get("broke", 0) > 0.5 else "fell")
                self._ends[end] += 1
                if "episode" in info:  # added by the Monitor wrapper
                    self._ep_seconds.append(info["episode"]["l"] * self.dt)
        return True

    def _on_rollout_end(self):
        for k, v in self._sums.items():
            self.logger.record(k, v / max(self._count, 1))
        episodes = sum(self._ends.values())
        if episodes:
            self.logger.record("1_overview/falls_pct_of_episodes", 100 * self._ends["fell"] / episodes)
            self.logger.record("1_overview/hardware_limit_stops_pct_of_episodes", 100 * self._ends["broke"] / episodes)
        if self._ep_seconds:
            self.logger.record("1_overview/episode_seconds", float(np.mean(self._ep_seconds)))


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


def new_model(cfg, env, tensorboard_log=None):
    """A fresh PPO model and the observation/reward normalization around `env` (a VecEnv).
    Mirror-symmetric (policy and normalization) when cfg.mirror_symmetry. Returns (model, env)."""
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    probe = BipedEnv(cfg.to_dict())
    actor_obs_dim = probe.actor_obs_dim
    mirror = None
    if cfg.mirror_symmetry:
        mismatch = check_robot_mirror(probe)
        print(f"mirror check: left/right part positions agree to {1000 * mismatch:.1f} mm", flush=True)
        if mismatch > 0.005:
            raise SystemExit("RobotConfig.mirror_pairs does not mirror this robot (a joint sign is "
                             "wrong, or the robot is not symmetric): fix it or set mirror_symmetry=False")
        mirror = mirror_maps(probe)
        env.obs_rms = SymmetricRunningMeanStd(mirror[0], mirror[1], shape=env.observation_space.shape)
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
            "mirror": mirror,
            "net_arch": {"pi": [256, 128, 64], "vf": [512, 256, 128]},
            "log_std_init": -1.0,
            "activation_fn": torch.nn.ELU,
        },
        tensorboard_log=tensorboard_log,
        device="cpu",
        verbose=1,
    )
    return model, env


STAGES = ("walk", "commands", "physics", "pushes")


def apply_stage(cfg, stage):
    """Staged training (Li et al. 2024, Cassie): learn the skill first, then the variety of tasks,
    then robustness. Pushes come last: frequent pushes early made Cassie's normal walking worse.
      walk      nominal physics, no pushes; one forward speed (0 to top) per episode; forgiving
                speed tracking
      commands  nominal physics, no pushes; all commands (backward, stand, sideways, turning, changed
                every few seconds); forgiving speed tracking
      physics   randomized physics, with the v7 realism (delays, heavy torso, slack, IMU bias...)
                phased in by the physics curriculum; no pushes; forgiving speed tracking
      pushes    randomized physics at full difficulty + the per-direction push curriculum
    Each stage is a separate run that --resume's the previous one (see train_v7.sh)."""
    rnd, cmd, w = cfg.randomization, cfg.command, cfg.rewards
    rnd.enabled = stage in ("physics", "pushes")
    # Speed-tracking kernel: forgiving while learning to walk (creeping must beat standing, or the
    # policy settles for standing still), then sharp (juan_v3's) once it walks, so it can't trade
    # speed away for small penalty savings.
    learning = stage in ("walk", "commands", "physics")
    w.tracking_vel = 2.0 if learning else RewardWeights().tracking_vel
    w.tracking_sigma2 = 0.02 if learning else RewardWeights().tracking_sigma2
    w.tracking_broad_share = 0.5 if learning else RewardWeights().tracking_broad_share
    # Stage 1 is the task the forward-only runs learned to walk on: one forward speed per episode,
    # no stand, no backward, no sideways/turning. With those mixed in from the start (~30% of
    # commands near zero, a new command every few seconds), standing still won.
    default = CommandConfig()
    walk = stage == "walk"
    cmd.straight_prob = 1.0 if walk else default.straight_prob
    cmd.stand_prob = 0.0 if walk else default.stand_prob
    cmd.backward_speed = 0.0 if walk else default.backward_speed
    cmd.resample_interval = (1e9, 1e9) if walk else default.resample_interval
    rnd.push_velocity = RandomizationConfig().push_velocity if stage == "pushes" else 0.0
    rnd.push_curriculum = stage == "pushes"
    rnd.physics_curriculum = stage == "physics"  # stage 4 trains on the full-difficulty robot


# Curriculum levels live in the env processes and would start over with every new run; they are
# saved at the end of a run and restored when another run resumes from it. (Through env methods:
# VecEnv.set_attr only reaches the Monitor wrapper, not the environment inside it.)
def save_curriculum(env, run_dir):
    states = env.env_method("curriculum_state")
    state = {k: np.mean([st[k] for st in states], axis=0).tolist() for k in states[0]}
    (run_dir / "curriculum.json").write_text(json.dumps(state, indent=2))


def load_curriculum(env, run_dir):
    path = run_dir / "curriculum.json"
    if path.exists():
        state = json.loads(path.read_text())
        env.env_method("set_curriculum_state", state)
        print(f"curriculum from {path}: {state}", flush=True)


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
        cfg.robot.model_path = portable_model_path(args.model)
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
    if args.stage:
        apply_stage(cfg, args.stage)
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
