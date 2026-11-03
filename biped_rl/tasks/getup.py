"""Getting up: the robot starts lying face down, legs straight, always the same way, and is rewarded for
getting up onto its feet, in any pose. A first look at whether Juan can get up at all.

Same robot, servo model (speed and torque limits, the measured stiffness and delay) and observations
as walking (BipedEnv), so a policy that gets up could later hand over to the walking one. What differs:
  * a much wider action range (GETUP_ACTION_SCALE): getting up needs most of each joint's travel, not
    +-29 deg around the home crouch;
  * no randomization: the same start every time (nominal physics), on the measured servos;
  * the reward (GetUpWeights, added each step): upright torso, height, and a bonus for standing on the
    feet only (held still), minus a small cost on jerky actions; nothing about protecting the servos;
  * an episode ends only at its time limit: lying on the ground is where it starts, not a fall.
"""
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from biped_rl.robots import juan
from biped_rl.tasks.locomotion import BipedEnv, EnvConfig
from biped_rl.tasks.locomotion.mdp import events, observations

# rad per unit action, per joint (robot/joints.py order: hip roll, hip pitch, knee, ankle; left, right):
# hip pitch +-92 deg, knee +-69 deg, ankle +-57 deg around home (clipped to the joint ranges).
GETUP_ACTION_SCALE = [0.5, 1.6, 1.2, 1.0] * 2
SETTLE_S = 1.0       # lying still, servos holding straight legs, before the policy takes over


@dataclass
class GetUpWeights:
    upright: float = 1.0       # torso's up axis vs vertical: 0 lying, 1 upright
    upright_on_feet: float = 0.0   # 1: upright counts only while just the feet touch the ground. Without it
                                   # sitting up on folded legs pays and nothing risky does (juan_getup_1/2
                                   # knelt); with it the way up goes hips first, feet under, torso last
    height: float = 1.0        # centre of mass from lying (0) to standing at home height (1), capped at 1.2
    height_right_side_up: float = 0.0   # 1: height counts in proportion to how right side up the torso is
                                        # (upright 1, lying 0.5, upside down 0): juan_getup_4 stood on its head
                                        # with its legs in the air, its centre of mass higher than kneeling
    standing: float = 3.0      # upright (> 0.85), high (> 0.75) and only the feet on the ground
    feet_only: float = 0.0     # only the feet on the ground, at any height (a squat, not a kneel): the step
                               # between sitting up on folded legs and standing (juan_getup_1 stopped there)
    still: float = 1.0         # while standing: not wobbling (torso roll / pitch rate)
    settled: float = 0.0       # while standing: both feet down, not moving, joints still: a stance, not a
                               # shuffle (juan_getup_5 got up fast, then kept stepping to stay up). Graded,
                               # so every bit of calming pays: juan_getup_6 had it all-or-nothing (feet under
                               # 0.1 m/s, joints under 0.5 rad/s) and, shuffling at 4 rad/s, never earned any
    action_rate: float = 0.05  # jerky actions (mean squared change per step, in action units)


def config(episode_seconds=6.0):
    """The getup task's EnvConfig: walking's, with the wide action range, no randomization, the measured
    servos (as the nominal robot), and no commands."""
    cfg = EnvConfig()
    cfg.action_scale = list(GETUP_ACTION_SCALE)
    cfg.episode_seconds = episode_seconds
    cfg.randomization.enabled = False
    cfg.robot.kp = juan.MEASURED_KP
    cfg.randomization.nominal_delay_steps = juan.MEASURED_DELAY_STEPS[0]
    return cfg


class GetUpEnv(BipedEnv):
    def __init__(self, config=None, weights=None, render_mode=None, width=640, height=480):
        super().__init__(config, render_mode=render_mode, width=width, height=height)
        self.w = GetUpWeights(**(weights or {}))
        self.lying_height = None

    def lie_face_down(self):
        """Legs straight, torso pitched 90 deg nose down, just above the floor; then SETTLE_S lying with
        the servos holding straight legs."""
        d, robot = self.data, self.robot
        self._reset_to_home()
        d.qpos[robot.qpos_idx] = 0.0                           # legs straight
        d.qpos[3:7] = [np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]   # +90 deg about y: x (forward) points down
        d.qpos[2] = 0.0
        d.qvel[:] = 0.0
        mujoco.mj_forward(self.model, d)
        lowest = min(d.geom_xpos[g][2] - self.model.geom_rbound[g] for g in range(self.model.ngeom)
                     if self.model.geom_bodyid[g] != 0 and (self.model.geom_contype[g] or self.model.geom_conaffinity[g]))
        d.qpos[2] = -lowest + 0.01
        mujoco.mj_forward(self.model, d)
        self.episode.target = np.zeros(self.nu)
        for _ in range(int(SETTLE_S / self.dt)):
            self.hold_pose(np.zeros(self.nu))

    def reset(self, *, seed=None, options=None):
        super(BipedEnv, self).reset(seed=seed)
        events.set_nominal_physics(self)
        self._reset_to_home()
        self._face_here()
        self._start_episode({"stand": True})   # (episode state for hold_pose; restarted below)
        self.lie_face_down()
        self.lying_height = self.com_height()
        self._face_here()
        obs, info = self._start_episode({"stand": True})
        self.episode.target = self.data.qpos[self.robot.qpos_idx].copy()
        return obs, info

    def _face_here(self):
        """The heading the observations hold is the current one (there's no walking command)."""
        cmd = self.command
        cmd.heading0 = cmd.heading_target = self.yaw()
        cmd.start_xy = self.data.qpos[0:2].copy()
        cmd.ref_xy = cmd.start_xy.copy()

    def step(self, action):
        ep, w, robot = self.episode, self.w, self.robot
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        s = self._measure(action, self._actuate(action))
        upright = float(np.clip(-s.grav[2], 0.0, 1.0))
        height = float(np.clip((self.com_height() - self.lying_height) / (robot.home_height - self.lying_height), 0.0, 1.2))
        on_feet = not s.bad_contact and ep.contact.sum() >= 1
        standing = upright > 0.85 and height > 0.75 and on_feet
        wobble = float(np.sum(s.ang_world[:2] ** 2))
        feet_speed = float(np.linalg.norm(s.sole_vel, axis=1).max())   # m/s, the faster sole
        joint_speed = float(np.sqrt(np.mean(s.qd ** 2)))                # rad/s, rms over the joints
        settled = (standing * (0.5 + 0.5 * (ep.contact.sum() == 2))
                   * np.exp(-joint_speed / 2.0) * np.exp(-feet_speed / 0.5))
        jerk = float(np.mean((action - ep.last_action) ** 2))
        side_up = (1.0 - s.grav[2]) / 2 if w.height_right_side_up else 1.0   # grav[2]: -1 upright, +1 upside down
        terms = {"upright": w.upright * upright * (on_feet if w.upright_on_feet else 1.0), "height": w.height * height * side_up,
                 "standing": w.standing * standing, "still": w.still * standing * np.exp(-wobble),
                 "feet_only": w.feet_only * on_feet * upright, "settled": w.settled * settled,
                 "action_rate": -w.action_rate * jerk}
        reward = sum(terms.values()) * self.dt
        ep.stood_for = getattr(ep, "stood_for", 0.0) + self.dt if standing else 0.0
        ep.best_stood = max(getattr(ep, "best_stood", 0.0), ep.stood_for)

        ep.last_action2, ep.last_action, ep.last_qvel = ep.last_action, action, s.qd.copy()
        ep.sensor_buf.append(observations.read_sensors(self))
        ep.action_buf.append(action)
        truncated = ep.steps >= self._max_steps
        info = {f"rew/{k}": v for k, v in terms.items()}
        info.update({"upright": upright, "height": height, "standing": float(standing), "settled": float(settled),
                     "stood_1s": float(ep.best_stood >= 1.0), "base_height": self.com_height()})
        return observations.compute(self), reward, False, truncated, info

    def weights(self):
        return asdict(self.w)
