"""The walking policy on the robot: a numpy-only copy of the trained actor and the exact observation
and action pipeline of the simulator, so the Pi needs neither torch nor the training code.

    policy = Policy("robot/policies/juan_v7_12b_pushes_best.npz")   # made by scripts/export_policy.py
    ctrl = Controller(policy)
    ctrl.start(sensors)            # at the home pose, robot standing still
    targets = ctrl.tick(sensors)   # every 20 ms: joint targets in radians (simulator convention)

`sensors` = dict(q=joint angles (rad, 8, in JOINTS order), qd=joint speeds (rad/s),
gravity=down unit vector in the torso frame, gyro=rad/s torso frame, yaw=rad).
The pipeline mirrors biped_rl/tasks/locomotion (observations.py, commands.py, env.py) and is checked
against it step by step by scripts/export_policy.py.
"""
import math
from collections import deque

import numpy as np

MAX_HEADING_ERROR = math.radians(110)  # training never goes past ~111 deg; the wrap is at 180


class Policy:
    def __init__(self, path):
        f = np.load(path, allow_pickle=False)
        self.meta = {k[5:]: f[k] for k in f.files if k.startswith("meta_")}
        self.layers = [(f[f"w{i}"], f[f"b{i}"]) for i in range(int(f["n_layers"]))]
        self.mean, self.std = f["obs_mean"], np.sqrt(f["obs_var"] + float(f["obs_eps"]))
        self.clip_obs = float(f["clip_obs"])
        self.mirror = bool(f["mirror"])
        if self.mirror:
            self.obs_perm, self.obs_sign = f["obs_perm"], f["obs_sign"]
            self.act_perm, self.act_sign = f["act_perm"], f["act_sign"]
        self.obs_dim = self.mean.shape[0]
        self.actor_dim = int(f["actor_dim"])

    def _net(self, x):
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))  # ELU
        return x

    def __call__(self, obs):
        """Actor observation (raw, unnormalized) -> action in [-1, 1]."""
        x = np.clip((np.asarray(obs, dtype=np.float64) - self.mean) / self.std, -self.clip_obs, self.clip_obs)
        x[self.actor_dim:] = 0.0  # the actor never sees the privileged part (masked after normalizing)
        mean = self._net(x)
        if self.mirror:
            mirrored = self._net(x[self.obs_perm] * self.obs_sign)
            mean = 0.5 * (mean + mirrored[self.act_perm] * self.act_sign)
        return np.clip(mean, -1.0, 1.0)


class Controller:
    def __init__(self, policy):
        self.policy, m = policy, policy.meta
        self.dt = 1.0 / float(m["control_hz"])
        self.history = int(m["history"])
        self.home = m["home"].astype(np.float64)
        self.lo, self.hi = m["joint_lo"], m["joint_hi"]
        self.scale, self.filter_a = float(m["action_scale"]), float(m["action_filter"])
        self.period, self.period_fast = float(m["gait_period"]), float(m["gait_period_fast"])
        self.fast_speed = float(m["fast_speed"])
        self.command = np.zeros(3)

    # ---- commands (the joystick) ----
    def set_command(self, forward, sideways=0.0, turn_rate=0.0):
        self.command = np.array([forward, sideways, turn_rate], dtype=np.float64)

    @property
    def standing(self):
        return not np.any(self.command)

    def gait_period(self):
        frac = min(math.hypot(self.command[0], self.command[1]) / self.fast_speed, 1.0)
        return self.period - (self.period - self.period_fast) * frac

    # ---- the loop ----
    def _reading(self, s):
        return np.concatenate([s["gravity"], s["gyro"], np.asarray(s["q"]) - self.home, 0.1 * np.asarray(s["qd"])])

    def start(self, sensors, phase=0.0):
        """Hand over to the policy from the current pose (as env.start_policy): heading = now,
        histories filled with the current reading, previous action and filter at zero."""
        n = len(self.home)
        self.heading_target = sensors["yaw"]
        self.phase = phase
        self.filtered = np.zeros(n)
        self.sensor_buf = deque([self._reading(sensors)] * self.history, maxlen=self.history)
        self.action_buf = deque([np.zeros(n)] * self.history, maxlen=self.history)
        self.target = np.clip(self.home.copy(), self.lo, self.hi)

    def observation(self, sensors):
        self.sensor_buf.append(self._reading(sensors))
        frames = [np.concatenate([self.sensor_buf[-1 - j], self.action_buf[-1 - j]])
                  for j in reversed(range(self.history))]
        ph = 2 * math.pi * (self.phase % 1.0)
        clock = [0.0, 0.0] if self.standing else [math.sin(ph), math.cos(ph)]
        yaw_err = (sensors["yaw"] - self.heading_target + math.pi) % (2 * math.pi) - math.pi
        # If the robot falls far behind the commanded turn (slipping feet), the target waits for it:
        # past +-180 deg the error would wrap and the policy would suddenly turn the other way. In
        # training (pushes and turns) 99.9% of errors are under 108 deg, so this cap changes almost
        # nothing the policy has seen.
        if abs(yaw_err) > MAX_HEADING_ERROR:
            yaw_err = math.copysign(MAX_HEADING_ERROR, yaw_err)
            self.heading_target = sensors["yaw"] - yaw_err
        return np.concatenate([*frames, self.command, clock, [math.sin(yaw_err), math.cos(yaw_err)]])

    def tick(self, sensors):
        """One 20 ms step: observation -> action -> joint targets (rad, clipped to the sim's limits)."""
        obs = self.observation(sensors)
        full = np.zeros(self.policy.obs_dim)
        full[:len(obs)] = obs  # the privileged part (critic only) stays zero
        action = self.policy(full)
        self.filtered = self.filter_a * action + (1 - self.filter_a) * self.filtered
        self.target = np.clip(self.home + self.scale * self.filtered, self.lo, self.hi)
        self.action_buf.append(action)
        self.last_obs, self.last_action = obs, action
        # The clocks advance over the step this action runs for, at the command in force now (as
        # in the simulator: a command that changes before the next tick doesn't change this step).
        self.phase += self.dt / self.gait_period()
        self.heading_target += self.command[2] * self.dt
        return self.target
