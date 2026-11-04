"""Getting up: the robot starts lying face down, legs straight, always the same way, and is rewarded for
getting up onto its feet, in any pose. A first look at whether Juan can get up at all.

Same robot, servo model (speed and torque limits, the measured stiffness and delay) and observations
as walking (BipedEnv), so a policy that gets up could later hand over to the walking one. What differs:
  * a much wider action range (GETUP_ACTION_SCALE): getting up needs most of each joint's travel, not
    +-29 deg around the home crouch;
  * the same start every time on the nominal robot (juan_getup_1-8), or with `start` (GetUpStart) varied
    lying poses (face down to either side, any heading, bent joints) and `physics_from` the walking runs'
    physics randomization (masses, friction, servo strength and delay, sensor noise; no shoves);
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
IDLE_JOINT_SPEED, IDLE_TORSO_RATE, IDLE_JERK, IDLE_HOLD_S = 0.3, 0.3, 0.01, 0.5   # rad/s, rad/s, action^2, s


@dataclass
class GetUpStart:
    side_deg: float = 0.0      # rolled onto a side by up to this (0 = face down, 90 = lying on a side)
    joint_rad: float = 0.0     # each joint bent by up to this from straight (clipped to its range)
    any_heading: bool = False
    knee_min_deg: float = None     # (a limit, not a start setting) the knees' lower limit (deg): juan_getup_5..19 stood
                                   # on a knee bent backwards, a stance the walker (knee 17..75 deg) can't continue from
                                   # and fine-tuning never left; with +5 that stance is impossible


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
    strict_standing: float = 0.0   # 1: standing also needs both feet down and under 0.2 m/s across the floor
                                   # (juan_getup_9 lunged and bounded on one foot at up to 1.7 m/s: "standing",
                                   # and no walker could take over). A knee bent backwards is allowed: every
                                   # getup since juan_getup_5 stands on one, and the walker takes over from it
    feet_only: float = 0.0     # only the feet on the ground, at any height (a squat, not a kneel): the step
                               # between sitting up on folded legs and standing (juan_getup_1 stopped there)
    still: float = 1.0         # while standing: not wobbling (torso roll / pitch rate)
    settled: float = 0.0       # while standing: both feet down, not moving, joints still: a stance, not a
                               # shuffle (juan_getup_5 got up fast, then kept stepping to stay up). Graded,
                               # so every bit of calming pays: juan_getup_6 had it all-or-nothing (feet under
                               # 0.1 m/s, joints under 0.5 rad/s) and, shuffling at 4 rad/s, never earned any
    action_rate: float = 0.05  # jerky actions (mean squared change per step, in action units)
    flat_feet: float = 0.0     # 1: standing also needs both soles within 20 deg of flat (juan_getup_12 stood on the
                               # edge of its left foot, hip + knee + ankle = -90 deg: touching the floor isn't standing)
    sole_flat: float = 0.0     # while upright on its feet: how flat the soles are (mean of cos^4 of their tilt: 1
                               # flat, 0.6 at 25 deg, 0 on edge), graded, so turning a foot down pays all the way
    idle_finish: float = 0.0   # >0: reaching idle ends the episode with this bonus, plus what standing perfectly until
                               # the time limit would have earned (so finishing early never loses): get to idle fast.
                               # Idle = standing (with the strict / flat rules in force) with the motors at rest:
                               # joints under IDLE_JOINT_SPEED, torso turning under IDLE_TORSO_RATE, actions steady,
                               # held IDLE_HOLD_S. There the walking policy takes over (getup_handover.py)
    joint_speed: float = 0.0   # while on its feet: joint motion (mean of joint speed^2, rad^2/s^2). juan_getup_13/14
                               # stood with a steady torso but legs buzzing at 3.5 rad/s rms (right knee 6): jitter
                               # in sim, wear and heat on the servos, and never idle
    two_legs: float = 0.0      # while upright on its feet: the lighter-loaded foot's share of the weight (full at 30% or
                               # more each): juan_getup_15 balanced on its right leg, ankle and knee buzzing, while the
                               # left foot tapped the floor 17 times a second
    home_pose: float = 0.0     # while standing: closeness to the home crouch, exp(-rms^2 / 2 (25 deg)^2): 0.23 at 43 deg
                               # rms (juan_getup_16's stance, a knee reversed), 0.92 at 10 deg; the walker starts from home
    idle_home_deg: float = 0.0     # >0: idle also needs every joint within this of home (deg)
    speed: float = 0.0         # moving across the floor (horizontal speed^2, m^2/s^2): getting up happens on the
                               # spot; juan_getup_9 and 10 bounded about at 0.8 m/s instead of standing


def config(episode_seconds=6.0, physics_from=None):
    """The getup task's EnvConfig: walking's, with the wide action range, the measured servos and no
    commands; the nominal robot, or `physics_from` a walking run's physics randomization (no shoves, no
    curricula: the full ranges from the start)."""
    cfg = EnvConfig()
    cfg.action_scale = list(GETUP_ACTION_SCALE)
    cfg.episode_seconds = episode_seconds
    cfg.robot.kp = juan.MEASURED_KP
    if physics_from:
        cfg.randomization = EnvConfig.load(physics_from).randomization
        r = cfg.randomization
        r.enabled, r.push_velocity, r.push_curriculum, r.physics_curriculum = True, 0.0, False, False
    else:
        cfg.randomization.enabled = False
        cfg.randomization.nominal_delay_steps = juan.MEASURED_DELAY_STEPS[0]
    return cfg


class GetUpEnv(BipedEnv):
    def __init__(self, config=None, weights=None, start=None, render_mode=None, width=640, height=480):
        super().__init__(config, render_mode=render_mode, width=width, height=height)
        self.w = GetUpWeights(**(weights or {}))
        self.start = GetUpStart(**(start or {}))
        if self.start.knee_min_deg is not None:   # servo targets (and lying poses) clipped to it
            self.robot.joint_lo[self.robot.knee_idx] = np.radians(self.start.knee_min_deg)
        self._sole_axis = self._sole_axes()
        self.lying_height = None

    def _sole_axes(self):
        """Per foot, the axis of its body frame that points up when it stands flat (measured at home)."""
        self._reset_to_home()
        mujoco.mj_forward(self.model, self.data)
        return [int(np.argmax(np.abs(self.data.xmat[b].reshape(3, 3)[2]))) for b in self.robot.foot_bodies]

    def sole_cos(self):
        """Per foot, the cosine of its sole's tilt from flat (1 = flat)."""
        return np.array([abs(self.data.xmat[b].reshape(3, 3)[2, a]) for b, a in zip(self.robot.foot_bodies, self._sole_axis)])

    def soles_flat(self, max_deg=20.0):
        """Both soles within max_deg of parallel to the floor."""
        up = [abs(self.data.xmat[b].reshape(3, 3)[2, a]) for b, a in zip(self.robot.foot_bodies, self._sole_axis)]
        return min(up) > np.cos(np.radians(max_deg))

    def lie_face_down(self):
        """Legs straight, torso pitched 90 deg nose down, just above the floor; then SETTLE_S lying with
        the servos holding the legs. With `start`: rolled toward a side, turned, joints bent, at random."""
        d, robot, st, rng = self.data, self.robot, self.start, self.np_random
        self._reset_to_home()
        joints = np.clip(rng.uniform(-st.joint_rad, st.joint_rad, self.nu), robot.joint_lo, robot.joint_hi)
        d.qpos[robot.qpos_idx] = joints
        quat = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])   # +90 deg about y: x (forward) points down
        side = np.radians(rng.uniform(-st.side_deg, st.side_deg))            # then rolled about the body's long axis
        heading = rng.uniform(-np.pi, np.pi) if st.any_heading else 0.0      # (world x, after the pitch), then turned
        for q in ([np.cos(side / 2), np.sin(side / 2), 0.0, 0.0], [np.cos(heading / 2), 0.0, 0.0, np.sin(heading / 2)]):
            out = np.empty(4)
            mujoco.mju_mulQuat(out, np.array(q), quat)
            quat = out
        d.qpos[3:7] = quat
        d.qpos[2] = 0.0
        d.qvel[:] = 0.0
        mujoco.mj_forward(self.model, d)
        lowest = min(d.geom_xpos[g][2] - self.model.geom_rbound[g] for g in range(self.model.ngeom)
                     if self.model.geom_bodyid[g] != 0 and (self.model.geom_contype[g] or self.model.geom_conaffinity[g]))
        d.qpos[2] = -lowest + 0.01
        mujoco.mj_forward(self.model, d)
        self.episode.target = joints.copy()
        for _ in range(int(SETTLE_S / self.dt)):
            self.hold_pose(joints)

    def reset(self, *, seed=None, options=None):
        super(BipedEnv, self).reset(seed=seed)
        if self.cfg.randomization.enabled:
            events.randomize_physics(self)
        else:
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
        if w.strict_standing:
            standing = standing and ep.contact.sum() == 2 and np.linalg.norm(self.data.qvel[:2]) < 0.2
        if w.flat_feet:
            standing = standing and self.soles_flat()
        wobble = float(np.sum(s.ang_world[:2] ** 2))
        feet_speed = float(np.linalg.norm(s.sole_vel, axis=1).max())   # m/s, the faster sole
        joint_speed = float(np.sqrt(np.mean(s.qd ** 2)))                # rad/s, rms over the joints
        settled = (standing * (0.5 + 0.5 * (ep.contact.sum() == 2))
                   * np.exp(-joint_speed / 2.0) * np.exp(-feet_speed / 0.5))
        jerk = float(np.mean((action - ep.last_action) ** 2))
        home_dev = np.degrees(np.abs(s.q - robot.home_joint))
        near_home = float(np.exp(-np.mean(home_dev ** 2) / (2 * 25.0 ** 2)))
        side_up = (1.0 - s.grav[2]) / 2 if w.height_right_side_up else 1.0   # grav[2]: -1 upright, +1 upside down
        terms = {"upright": w.upright * upright * (on_feet if w.upright_on_feet else 1.0), "height": w.height * height * side_up,
                 "standing": w.standing * standing, "still": w.still * standing * np.exp(-wobble),
                 "feet_only": w.feet_only * on_feet * upright, "settled": w.settled * settled,
                 "sole_flat": w.sole_flat * on_feet * upright * float(np.mean(self.sole_cos() ** 4)),
                 "action_rate": -w.action_rate * jerk, "speed": -w.speed * float(np.sum(self.data.qvel[:2] ** 2)),
                 "joint_speed": -w.joint_speed * on_feet * float(np.mean(s.qd ** 2)),
                 "home_pose": w.home_pose * standing * near_home,
                 "two_legs": w.two_legs * on_feet * upright * min(float(s.peak_force.min()) / max(float(s.peak_force.sum()), 1e-6) / 0.3, 1.0)}
        reward = sum(terms.values()) * self.dt
        ep.stood_for = getattr(ep, "stood_for", 0.0) + self.dt if standing else 0.0
        ep.best_stood = max(getattr(ep, "best_stood", 0.0), ep.stood_for)
        idle = (standing and joint_speed < IDLE_JOINT_SPEED and np.linalg.norm(s.ang_world[:2]) < IDLE_TORSO_RATE
                and jerk < IDLE_JERK)
        if w.idle_home_deg:
            idle = idle and home_dev.max() < w.idle_home_deg
        ep.idle_for = getattr(ep, "idle_for", 0.0) + self.dt if idle else 0.0
        finished = bool(w.idle_finish) and ep.idle_for >= IDLE_HOLD_S
        if finished:   # the bonus, and everything standing perfectly to the time limit would have earned
            best_rate = (w.upright + 1.2 * w.height + w.standing + w.still + w.feet_only + w.settled + w.sole_flat
                         + w.home_pose + w.two_legs)
            reward += w.idle_finish + best_rate * self.dt * max(self._max_steps - ep.steps, 0)

        ep.last_action2, ep.last_action, ep.last_qvel = ep.last_action, action, s.qd.copy()
        ep.sensor_buf.append(observations.read_sensors(self))
        ep.action_buf.append(action)
        truncated = ep.steps >= self._max_steps and not finished
        info = {f"rew/{k}": v for k, v in terms.items()}
        info.update({"upright": upright, "height": height, "standing": float(standing), "settled": float(settled),
                     "stood_1s": float(ep.best_stood >= 1.0), "base_height": self.com_height(),
                     "idle": float(idle), "home_rms_deg": float(np.sqrt(np.mean(home_dev ** 2))), "idle_done": float(ep.idle_for >= IDLE_HOLD_S),
                     "idle_at": ep.t if ep.idle_for >= IDLE_HOLD_S else float("nan")})
        return observations.compute(self), reward, finished, truncated, info

    def weights(self):
        return asdict(self.w)
