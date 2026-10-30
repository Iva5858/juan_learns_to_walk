"""Training callbacks: the TensorBoard dashboard, walking clips, saving the best model's
normalization statistics."""
from pathlib import Path

import imageio
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Video

from biped_rl.tasks.locomotion import BipedEnv, EnvConfig


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
                    "hip_roll", "hip_roll_inward", "stand_pose"),
        "smoothness": ("action_rate", "action_acc", "joint_acc", "reversals"),
        "servo_protection": ("servo_load", "servo_saturation", "servo_overload", "backdrive", "energy",
                             "joint_limits"),
        "feet_and_impacts": ("impact", "foot_slip", "feet_distance", "footstep"),
        "gait_shape": ("symmetry", "gait_mismatch", "double_support"),
    }

    SAMPLE_EVERY = 4

    def __init__(self, cfg, log_details=False):
        super().__init__()
        self.dt = 1.0 / cfg.control_hz
        self.weights = cfg.rewards
        self.log_details = log_details
        self.group_of = {term: g for g, terms in self.PENALTY_GROUPS.items() for term in terms}

    def _on_rollout_start(self):
        self._sums, self._count, self._ends = {}, 0, {"fell": 0, "broke": 0, "survived": 0}
        self._ep_seconds = []
        self._backward = np.zeros(2)  # speed reached, speed commanded, over backward commands

    def _add(self, key, value):
        self._sums[key] = self._sums.get(key, 0.0) + float(value)

    def _on_step(self):
        # Statistics from every SAMPLE_EVERY-th step (episode ends from every step): the per-robot Python
        # bookkeeping was 16% of the training time; a quarter of the steps gives the same averages.
        sample = self.n_calls % self.SAMPLE_EVERY == 0
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if sample:
                self._count += 1
                self._add("2_commands/forward_error_mps", info["speed_error"])
                self._add("2_commands/sideways_error_mps", abs(info["lateral_velocity"] - info["command_lateral"]))
                self._add("2_commands/turn_rate_error_radps", abs(info["yaw_rate"] - info["command_yaw_rate"]))
                self._add("2_commands/heading_error_deg", info["heading_error_deg"])
                self._add("2_commands/path_drift_cm", 100 * info["lateral_offset"])
                if info["command"] < -0.03:
                    self._backward += (info["forward_velocity"], info["command"])
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
        if self._backward[1] < 0:  # 0 = stands still when told to back up, 100 = full speed
            self.logger.record("2_commands/backward_speed_pct_of_command", 100 * self._backward[0] / self._backward[1])


def end_of_clip(frames, fps):
    """Hold the last frame 0.5 s, then fade to black: when a player loops the clip, the jump
    back to the start position is clearly a cut, not the robot moving."""
    last = frames[-1].astype(np.float32)
    fade = [(last * (1 - i / 10)).astype(np.uint8) for i in range(1, 11)]
    return frames + [frames[-1]] * (fps // 2) + fade


class VideoRecorder(BaseCallback):
    """Every `freq` steps, film the current policy on the nominal robot (with the shortest servo delay
    it trains with: policies trained only with delay fall without any): walking forward at the speed
    curriculum's top, and, if it trains on backward commands, walking backward at its top backward speed.

    The forward clip goes to TensorBoard (IMAGES tab, as an animation); both go to <run>/videos/.
    """

    def __init__(self, cfg, video_dir, freq, seconds=8.0, size=(240, 320)):
        super().__init__()
        cfg = EnvConfig.from_dict(cfg.to_dict())
        cfg.randomization.enabled = False
        self.delay = int(cfg.randomization.action_delay_steps[0])
        self.backward = cfg.command.backward_speed if cfg.command.backward_prob > 0 else 0.0
        self.env = BipedEnv(cfg, render_mode="rgb_array", height=size[0], width=size[1])
        self.video_dir, self.freq, self.steps = Path(video_dir), freq, int(seconds / self.env.dt)
        self._last = 0

    def _clip(self, forward):
        """Film one clip at a fixed forward command. Returns (frames, distance walked along +x, seconds)."""
        norm = self.model.get_vec_normalize_env()
        obs, _ = self.env.reset(seed=0, options={"command": (forward, 0.0, 0.0)})
        self.env.servo.action_delay = self.delay
        start_x = self.env.data.qpos[0]  # nominal physics: the target direction is +x
        frames = [self.env.render()]
        for i in range(self.steps):
            action, _ = self.model.predict(norm.normalize_obs(obs), deterministic=True)
            obs, _, terminated, truncated, _ = self.env.step(action)
            if i % 2:  # 25 fps: drawing Juan's CAD meshes takes ~40 ms a frame on the CPU
                frames.append(self.env.render())
            if terminated or truncated:
                break
        return frames, self.env.data.qpos[0] - start_x, (i + 1) * self.env.dt

    def _on_step(self):
        if self.num_timesteps - self._last < self.freq:
            return True
        self._last = self.num_timesteps
        fps = round(1 / self.env.dt) // 2
        self.video_dir.mkdir(parents=True, exist_ok=True)
        speed = max(st["speed_max"] for st in self.training_env.env_method("curriculum_state"))
        frames, distance, seconds = self._clip(speed)
        frames = end_of_clip(frames, fps)
        self.logger.record("video/distance_m", distance)
        self.logger.record("video/mean_speed", distance / seconds)
        imageio.mimsave(self.video_dir / f"step_{self.num_timesteps // 1000:06d}k_cmd{speed:.2f}_walked{distance:.2f}m.mp4", frames, fps=fps)
        video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).unsqueeze(0)  # (1,T,C,H,W)
        self.logger.record("video/walk", Video(video, fps=fps), exclude=("stdout", "log", "json", "csv"))
        imageio.mimsave(self.video_dir / "latest.mp4", frames, fps=fps)
        if self.backward > 0:
            frames, distance, seconds = self._clip(-self.backward)
            self.logger.record("video/backward_distance_m", distance)
            self.logger.record("video/backward_mean_speed", distance / seconds)
            imageio.mimsave(self.video_dir / f"step_{self.num_timesteps // 1000:06d}k_back{self.backward:.2f}_walked{distance:.2f}m.mp4",
                            end_of_clip(frames, fps), fps=fps)
        return True


class CurriculumSnapshot(BaseCallback):
    """Cache worker curriculum state at safe points, for saving after an interrupt."""

    def __init__(self):
        super().__init__()
        self.states = None

    def _on_training_start(self):
        self.states = self.training_env.env_method("curriculum_state")

    def _on_rollout_end(self):
        self.states = self.training_env.env_method("curriculum_state")

    def _on_step(self):
        return True


class SaveVecNormalize(BaseCallback):
    """Saves the normalization statistics next to best_model.zip whenever EvalCallback saves a new best."""

    def __init__(self, path):
        super().__init__()
        self.path = path

    def _on_step(self):
        self.model.get_vec_normalize_env().save(self.path)
        return True
