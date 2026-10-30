"""Reading the real robot's run logs (robot/logs/*.npz, written by robot/walker.py) with the
simulator's kinematics: where the body went and what the feet did.

The robot measures joint angles (servos) and torso orientation (IMU), not its position on the floor.
Position is estimated by keeping the standing foot planted: the lower foot stays where it touched down
and the body moves around it. Foot slip is not captured, so distances are estimates.

An IMU tilt error (mounting, level) poses the robot slightly tilted, so one sole always looks a few
millimetres lower and would never stop being "the standing foot". That constant left/right height
difference is measured while the robot stands (both feet down) and removed (sole_bias).
"""
import math

import mujoco
import numpy as np

from biped_rl.tasks.locomotion import BipedEnv, EnvConfig

SWITCH_MM = 3.0  # the other foot becomes the standing foot once it is this much lower


def torso_rotation(gravity, yaw):
    """Rotation matrix (world <- torso) from the IMU's down vector (torso frame) and heading, plus roll
    and pitch in degrees (+ = right side down, + = nose down)."""
    g = np.asarray(gravity, float) / np.linalg.norm(gravity)
    roll = math.atan2(-g[1], -g[2])
    pitch = math.atan2(g[0], math.hypot(g[1], g[2]))
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx, math.degrees(roll), math.degrees(pitch)


def kinematic_env(**kwargs):
    """A nominal simulator, for posing the robot (nothing is simulated)."""
    cfg = EnvConfig()
    cfg.randomization.enabled = False
    env = BipedEnv(cfg.to_dict(), **kwargs)
    env.reset(seed=0)
    return env


class Replay:
    """Steps through a log, posing the simulated robot as the real one was at each sample: joint
    angles from the servos, tilt and heading from the IMU, floor position from the planted foot.
    After pose(k): env.data holds the pose; base_xy, soles (2x3, lowest on the floor), roll, pitch."""

    def __init__(self, log, env=None, yaw0=None, sole_bias=None):
        self.log, self.env = log, env or kinematic_env()
        self.yaw0 = float(log["yaw"][0]) if yaw0 is None else yaw0
        self.sole_bias = estimate_sole_bias(log, self.env) if sole_bias is None else sole_bias
        self.base_xy, self.stance, self.anchor = np.zeros(2), None, None
        self.soles, self.roll, self.pitch = None, 0.0, 0.0

    def pose(self, k):
        env, log = self.env, self.log
        m, d, robot = env.model, env.data, env.robot
        R, self.roll, self.pitch = torso_rotation(log["gravity"][k], float(log["yaw"][k]) - self.yaw0)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R.ravel())
        d.qpos[3:7] = quat
        d.qpos[robot.qpos_idx] = log["q"][k]
        d.qpos[0:2], d.qpos[2] = self.base_xy, 0.3
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        soles = env.soles()[0]
        z = self.corrected(soles[:, 2])
        lower = int(np.argmin(z))
        if self.stance is None:
            self.stance, self.anchor = lower, soles[lower, :2].copy()
        elif lower != self.stance and z[self.stance] - z[lower] > SWITCH_MM / 1000:
            self.stance, self.anchor = lower, soles[lower, :2].copy()  # touchdown of the other foot
        else:
            self.base_xy = self.base_xy + self.anchor - soles[self.stance, :2]  # the standing foot stays put
        d.qpos[0:2] = self.base_xy
        d.qpos[2] -= soles[:, 2].min()  # lowest sole on the floor
        mujoco.mj_forward(m, d)
        self.soles = env.soles()[0]
        return self

    def corrected(self, z):
        """Sole heights (left, right) with the constant IMU-tilt difference removed."""
        return np.asarray(z) - np.array([self.sole_bias / 2, -self.sole_bias / 2])


def estimate_sole_bias(log, env):
    """Median left-minus-right sole height while commanded to stand still (both feet down), or over
    the whole run if it never stood. [m]"""
    standing = np.flatnonzero(~np.any(log["command"], axis=1))
    idx = standing if len(standing) >= 25 else np.arange(len(log["t"]))
    idx = idx[:: max(1, len(idx) // 200)]  # ~200 samples are plenty
    probe = Replay(log, env, sole_bias=0.0)
    diffs = [np.subtract(*probe.pose(k).soles[:, 2]) for k in idx]
    return float(np.median(diffs))
