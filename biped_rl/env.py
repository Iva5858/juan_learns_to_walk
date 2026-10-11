"""Gymnasium environment for a small servo-driven biped in MuJoCo.

Works with any MJCF that has a <freejoint/> root, one actuator per driven joint,
foot bodies named in RobotConfig.foot_bodies and (optionally) a "home" keyframe.
Whatever actuator type the MJCF declares, the joints are driven by a hobby-servo
model computed here: PD control + torque-speed curve + command latency.

Observation = [actor part | privileged part]. The actor part contains only what
the real robot can sense (IMU, servo angles/speeds, past actions, the command, the
gait clock and the heading error). The privileged part (true velocities, contacts,
friction...) is only seen by the critic during training (see policy.py).

Reward structure (see RewardWeights):
    reward = (alive + tracking * sum(task terms)) * exp(-sum(penalties)) [+ termination]
"""
from collections import deque

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from biped_rl.config import EnvConfig, resolve_model_path


class BipedEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, config=None, render_mode=None, width=640, height=480):
        if isinstance(config, dict):
            config = EnvConfig.from_dict(config)
        self.cfg = cfg = config or EnvConfig()
        self.model = m = mujoco.MjModel.from_xml_path(resolve_model_path(cfg.robot.model_path))
        self.data = mujoco.MjData(m)
        self.frame_skip = round(1.0 / (cfg.control_hz * m.opt.timestep))
        self.dt = self.frame_skip * m.opt.timestep
        self.metadata = {**self.metadata, "render_fps": round(1 / self.dt)}
        self.render_mode = render_mode
        self._render_size = (height, width)
        self._renderer = None

        if m.njnt == 0 or m.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("The first joint must be the root <freejoint/> of the robot.")
        self._root = m.jnt_bodyid[0]
        joint_ids = m.actuator_trnid[:, 0]
        self.nu = m.nu
        self._qpos_idx = m.jnt_qposadr[joint_ids]
        self._dof_idx = m.jnt_dofadr[joint_ids]
        self._jnt_lo, self._jnt_hi = m.jnt_range[joint_ids].T.copy()
        self._convert_actuators_to_torque(joint_ids)
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joint_ids]
        self._knee_idx = [names.index(n) for n in cfg.robot.knee_joints if n in names]
        self._hip_roll_idx = [names.index(n) for n in cfg.robot.hip_roll_joints if n in names]
        pairs = [(names.index(l), names.index(r), sgn) for l, r, sgn in cfg.robot.mirror_pairs
                 if l in names and r in names]
        self._mirror_l = np.array([p[0] for p in pairs], dtype=int)
        self._mirror_r = np.array([p[1] for p in pairs], dtype=int)
        self._mirror_sign = np.array([p[2] for p in pairs], dtype=float)
        self._q_hist = deque(maxlen=int(cfg.command.gait_period / self.dt) + 2)  # joint angles, for symmetry
        self._td_hist = deque(maxlen=int(cfg.command.gait_period / self.dt) + 2)  # touchdowns per step

        # Feet: map every geom to its leg (0 left, 1 right, -1 not a foot), and keep each
        # foot's collision geoms to find the sole's lowest point (toe/heel aware).
        self._foot_body = []
        for name in cfg.robot.foot_bodies:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"Foot body '{name}' not found; set RobotConfig.foot_bodies.")
            self._foot_body.append(bid)
        self._foot_body = np.array(self._foot_body)
        self._geom_leg = np.full(m.ngeom, -1)
        self._foot_geoms = []
        for leg, bid in enumerate(self._foot_body):
            self._geom_leg[m.geom_bodyid == bid] = leg
            own = np.flatnonzero(m.geom_bodyid == bid)
            colliding = [g for g in own if m.geom_contype[g] or m.geom_conaffinity[g]]
            self._foot_geoms.append(colliding or list(own))
        self._geom_is_world = m.geom_bodyid == 0
        self._box_corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
        self._vel6 = np.zeros(6)

        # Nominal state and physics.
        self._key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        self._reset_to_home()
        self._home_joint = self.data.qpos[self._qpos_idx].copy()
        # Height = centre of mass, not the root body's origin: CAD exports often put the base
        # frame origin somewhere arbitrary (e.g. at floor level).
        self._home_height = self._com_height()
        self._home_sole_z = self._soles()[0][:, 2].copy()
        self._roll_out_sign = self._hip_roll_outward_signs()
        self._weight = m.body_subtreemass[self._root] * -m.opt.gravity[2]
        self._nominal = {
            "mass": m.body_mass.copy(), "inertia": m.body_inertia.copy(), "ipos": m.body_ipos.copy(),
            "friction": m.geom_friction.copy(), "armature": m.dof_armature.copy(),
            "frictionloss": m.dof_frictionloss.copy(), "gravity": m.opt.gravity.copy(),
        }
        self._set_nominal_physics()
        self._speed_max = cfg.command.speed_range[1]  # speed curriculum, grows during training

        frame_dim = 6 + (2 * self.nu if cfg.robot.joint_feedback else 0) + self.nu
        self.actor_obs_dim = cfg.history * frame_dim + 3 + 2  # + command, clock (2), heading error (2)
        privileged_dim = 1 + 3 + 1 + 2 + 2 + 2 * self.nu + 1
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.nu,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, dtype=np.float64,
                                            shape=(self.actor_obs_dim + privileged_dim,))
        self._history = deque(maxlen=cfg.history)
        self._max_steps = int(cfg.episode_seconds / self.dt)

    # ------------------------------------------------------------------ setup
    def _convert_actuators_to_torque(self, joint_ids):
        """Turn every actuator into a plain torque motor; the servo model lives in step()."""
        m = self.model
        m.actuator_dyntype[:] = mujoco.mjtDyn.mjDYN_NONE
        m.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_gainprm[:] = 0.0
        m.actuator_gainprm[:, 0] = 1.0
        m.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_NONE
        m.actuator_biasprm[:] = 0.0
        m.actuator_gear[:] = 0.0
        m.actuator_gear[:, 0] = 1.0
        m.actuator_ctrllimited[:] = 0
        m.actuator_forcelimited[:] = 0
        m.jnt_actfrclimited[joint_ids] = 0

    def _reset_to_home(self):
        if self._key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._key)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def _hip_roll_outward_signs(self):
        """+1 if positive hip roll swings that leg outward, -1 if inward. Measured on the model
        (nudge each roll joint, see whether the feet move away from the centre line), so it
        works whichever way the CAD's roll axes point."""
        m = self.model
        d = mujoco.MjData(m)
        d.qpos[:] = self.data.qpos
        signs = []
        for idx in self._hip_roll_idx:
            spread = []
            for delta in (0.0, 0.05):
                d.qpos[:] = self.data.qpos
                d.qpos[self._qpos_idx[idx]] += delta
                mujoco.mj_kinematics(m, d)
                R = d.xmat[self._root].reshape(3, 3)
                feet_y = (d.xpos[self._foot_body] - d.xpos[self._root]) @ R[:, 1]
                spread.append(np.abs(feet_y).sum())
            signs.append(1.0 if spread[1] > spread[0] else -1.0)
        return np.array(signs)

    def _set_nominal_physics(self):
        m, n, r = self.model, self._nominal, self.cfg.robot
        m.body_mass[:], m.body_inertia[:], m.body_ipos[:] = n["mass"], n["inertia"], n["ipos"]
        m.geom_friction[:] = n["friction"]
        m.dof_armature[:], m.dof_frictionloss[:] = n["armature"], n["frictionloss"]
        m.opt.gravity[:] = n["gravity"]
        self._stall = np.full(self.nu, r.stall_torque)
        self._speed = np.full(self.nu, r.no_load_speed)
        self._kp = np.full(self.nu, r.kp)
        self._kd = np.full(self.nu, r.kd)
        self._latency_steps = 0
        self._friction_scale = 1.0

    def _randomize_physics(self):
        m, n, c, rng = self.model, self._nominal, self.cfg.randomization, self.np_random
        nu, di = self.nu, self._dof_idx
        scale = rng.uniform(*c.mass_scale, size=m.nbody)
        m.body_mass[:] = n["mass"] * scale
        m.body_inertia[:] = n["inertia"] * scale[:, None]
        m.body_ipos[self._root] = n["ipos"][self._root] + rng.uniform(-c.com_offset, c.com_offset, 3)
        self._friction_scale = rng.uniform(*c.friction) / n["friction"][:, 0].max()
        m.geom_friction[:, 0] = n["friction"][:, 0] * self._friction_scale
        m.dof_armature[di] = n["armature"][di] * rng.uniform(*c.armature_scale, size=nu)
        m.dof_frictionloss[di] = n["frictionloss"][di] * rng.uniform(*c.frictionloss_scale, size=nu)
        tilt, heading = np.deg2rad(rng.uniform(0, c.floor_tilt_deg)), rng.uniform(0, 2 * np.pi)
        g = np.linalg.norm(n["gravity"])
        m.opt.gravity[:] = g * np.array([np.sin(tilt) * np.cos(heading),
                                         np.sin(tilt) * np.sin(heading), -np.cos(tilt)])
        r = self.cfg.robot
        self._stall = r.stall_torque * rng.uniform(*c.servo_strength) * rng.uniform(0.95, 1.05, nu)
        self._speed = r.no_load_speed * rng.uniform(*c.servo_speed) * rng.uniform(0.95, 1.05, nu)
        self._kp = r.kp * rng.uniform(*c.kp_scale, size=nu)
        self._kd = r.kd * rng.uniform(*c.kd_scale, size=nu)
        lat = int(rng.uniform(*c.latency) / m.opt.timestep)
        self._latency_steps = min(lat, self.frame_skip - 1)

    # ------------------------------------------------------------------ sensing
    def _root_rot(self):
        return self.data.xmat[self._root].reshape(3, 3)

    def _com_height(self):
        return float(self.data.subtree_com[self._root][2])

    def _gravity_body(self):
        g = self.model.opt.gravity
        return self._root_rot().T @ (g / np.linalg.norm(g))

    def _geom_lowest_point(self, g):
        m, d = self.model, self.data
        pos, R, size, G = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g], mujoco.mjtGeom
        t = m.geom_type[g]
        if t == G.mjGEOM_BOX:
            pts = pos + (self._box_corners * size) @ R.T
        elif t == G.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            a = m.mesh_vertadr[mid]
            pts = pos + m.mesh_vert[a:a + m.mesh_vertnum[mid]] @ R.T
        elif t in (G.mjGEOM_CAPSULE, G.mjGEOM_CYLINDER):
            pts = pos + np.outer([-size[1], size[1]], R[:, 2]) - np.array([0.0, 0.0, size[0]])
        else:  # sphere, ellipsoid, ...: bounding sphere
            pts = (pos - np.array([0.0, 0.0, m.geom_rbound[g]]))[None]
        return pts[np.argmin(pts[:, 2])]

    def _soles(self):
        """Lowest point of each foot's collision shape (world) and that point's velocity.

        Using the sole rather than the ankle closes the toe-drag loophole: a foot
        pitched toe-down lifts the ankle while the toe still scrapes the floor."""
        m, d = self.model, self.data
        pts, vel = np.empty((2, 3)), np.empty((2, 3))
        for leg, bid in enumerate(self._foot_body):
            p = min((self._geom_lowest_point(g) for g in self._foot_geoms[leg]), key=lambda c: c[2])
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, bid, self._vel6, 0)
            vel[leg] = self._vel6[3:] + np.cross(self._vel6[:3], p - d.xpos[bid])
            pts[leg] = p
        return pts, vel

    def _sensor_frame(self):
        """What the real robot measures this control step (plus noise)."""
        d, c, rng = self.data, self.cfg.randomization, self.np_random
        noisy = c.enabled
        noise = lambda scale, n: rng.normal(0, scale, n) if noisy else 0.0
        parts = [self._gravity_body() + noise(c.noise_gravity, 3),
                 d.qvel[3:6] + noise(c.noise_gyro, 3)]  # free-joint angular velocity is in body frame
        if self.cfg.robot.joint_feedback:
            parts += [d.qpos[self._qpos_idx] - self._home_joint + noise(c.noise_joint_pos, self.nu),
                      0.1 * (d.qvel[self._dof_idx] + noise(c.noise_joint_vel, self.nu))]
        parts.append(self._last_action)
        return np.concatenate(parts)

    def _gait_period(self):
        """Faster rhythm at higher commands: a fixed rhythm caps top speed."""
        c = self.cfg.command
        frac = min(self._cmd / c.fast_speed, 1.0)
        return c.gait_period - (c.gait_period - c.gait_period_fast) * frac

    def _phase(self):
        return self._phase_acc % 1.0

    def _heading_error(self):
        """Yaw relative to the episode's target direction, wrapped to [-pi, pi].
        On the robot: IMU yaw minus the yaw recorded when the walk started."""
        R = self._root_rot()
        yaw = np.arctan2(R[1, 0], R[0, 0])
        return (yaw - self._heading0 + np.pi) % (2 * np.pi) - np.pi

    def _lateral_offset(self):
        """Signed distance from the straight line through the start point (privileged)."""
        dx, dy = self.data.qpos[0:2] - self._start_xy
        return -np.sin(self._heading0) * dx + np.cos(self._heading0) * dy

    def _get_obs(self):
        d = self.data
        ph = 2 * np.pi * self._phase()
        yaw_err = self._heading_error() + (self.np_random.normal(0, self.cfg.randomization.noise_yaw)
                                           if self.cfg.randomization.enabled else 0.0)
        actor = np.concatenate([*self._history, [self._cmd, np.sin(ph), np.cos(ph),
                                                 np.sin(yaw_err), np.cos(yaw_err)]])
        privileged = np.concatenate([
            [self._lateral_offset()],
            self._root_rot().T @ d.qvel[0:3],
            [self._com_height() - self._home_height],
            self._contact,
            self._clearance,
            d.qpos[self._qpos_idx] - self._home_joint,
            0.1 * d.qvel[self._dof_idx],
            [self._friction_scale],
        ])
        return np.concatenate([actor, privileged])

    def _foot_forces(self):
        """Vertical contact force on each foot body (cheap: one C call, no Python loop)."""
        mujoco.mj_rnePostConstraint(self.model, self.data)
        return self.data.cfrc_ext[self._foot_body, 5]

    def _non_foot_ground_contact(self):
        """True when anything but a foot touches the ground (knee, torso...): a fall."""
        d = self.data
        if d.ncon == 0:
            return False
        g1, g2 = d.contact.geom[: d.ncon].T
        w1, w2 = self._geom_is_world[g1], self._geom_is_world[g2]
        return bool(np.any((w1 ^ w2) & (self._geom_leg[np.where(w1, g2, g1)] < 0)))

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        cfg, rng = self.cfg, self.np_random
        if cfg.randomization.enabled:
            self._randomize_physics()
        else:
            self._set_nominal_physics()
        self._reset_to_home()
        d = self.data
        # Random start heading: the policy must learn "hold the heading you started with",
        # not "walk along world +x".
        self._heading0 = rng.uniform(-np.pi, np.pi) if cfg.randomization.enabled else 0.0
        yaw_quat = np.array([np.cos(self._heading0 / 2), 0.0, 0.0, np.sin(self._heading0 / 2)])
        quat = np.empty(4)
        mujoco.mju_mulQuat(quat, yaw_quat, d.qpos[3:7].copy())
        d.qpos[3:7] = quat
        self._start_xy = d.qpos[0:2].copy()
        d.qpos[self._qpos_idx] += rng.uniform(-0.05, 0.05, self.nu)
        d.qvel[:] = rng.uniform(-0.05, 0.05, self.model.nv)
        mujoco.mj_forward(self.model, d)

        self._cmd_fixed = bool(options and "speed" in options)
        lo = cfg.command.speed_range[0]
        self._cmd = float(options["speed"]) if self._cmd_fixed else rng.uniform(lo, self._speed_max)
        self._t, self._steps, self._ep_speed_err = 0.0, 0, 0.0
        self._ep_peak = {"torque": 0.0, "speed": 0.0, "impact": 0.0}  # hardware-load maxima this episode
        self._phase_acc = rng.uniform()
        self._target = self._home_joint.copy()
        self._filtered = np.zeros(self.nu)
        self._last_action = np.zeros(self.nu)
        self._last_action2 = np.zeros(self.nu)
        self._last_qvel = d.qvel[self._dof_idx].copy()
        self._contact = np.ones(2)
        self._qd_sign = np.zeros(self.nu)  # last clear direction of each joint, for reversal counting
        self._clearance = self._soles()[0][:, 2] - self._home_sole_z
        self._next_push = rng.uniform(*cfg.randomization.push_interval)
        self._last_push = -np.inf
        self._history.clear()
        self._q_hist.clear()
        self._td_hist.clear()
        self._td_step = [None, None]   # control step of each foot's last touchdown
        self._td_last_leg = None
        self._td_intervals = deque(maxlen=2)
        self._stride_est = None        # measured steps per full stride (from one foot's touchdowns)
        self._timing_asym = 0.0        # latest measured left/right step-timing asymmetry
        frame = self._sensor_frame()
        for _ in range(cfg.history):
            self._history.append(frame)
        return self._get_obs(), {}

    def _episode_was_safe(self):
        w = self.cfg.rewards
        return (self._ep_peak["torque"] < w.safe_torque and self._ep_peak["speed"] < w.safe_joint_speed
                and self._ep_peak["impact"] < w.safe_impact)

    def _update_speed_curriculum(self, fell):
        """Promote the top speed when top commands are tracked through a whole episode;
        demote when the robot falls at the top. Each training env keeps its own."""
        c = self.cfg.command
        if self._cmd_fixed or not self.cfg.randomization.enabled or self._cmd < 0.75 * self._speed_max:
            return
        if fell:
            self._speed_max = max(c.speed_range[1], self._speed_max - c.curriculum_step)
        elif self._ep_speed_err / self._steps < 0.25 * self._cmd and self._episode_was_safe():
            self._speed_max = min(c.speed_limit, self._speed_max + c.curriculum_step)

    def step(self, action):
        cfg, m, d = self.cfg, self.model, self.data
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        a = cfg.action_filter
        self._filtered = a * action + (1 - a) * self._filtered
        new_target = np.clip(self._home_joint + cfg.action_scale * self._filtered, self._jnt_lo, self._jnt_hi)

        # Servo model, run at physics rate. (Plain ufuncs: this loop dominates step time.)
        qi, di, kp, kd, stall = self._qpos_idx, self._dof_idx, self._kp, self._kd, self._stall
        inv_speed = 1.0 / self._speed
        peak_force, tau_sq, power, saturated = np.zeros(2), np.zeros(self.nu), np.zeros(self.nu), 0
        overload, overspeed, peak_tau = np.zeros(self.nu), np.zeros(self.nu), np.zeros(self.nu)
        peak_speed_ratio = 0.0
        xy_before = d.qpos[0:2].copy()
        for k in range(self.frame_skip):
            if k == self._latency_steps:
                self._target = new_target
            qd = d.qvel[di]
            tau = kp * (self._target - d.qpos[qi]) - kd * qd
            # Linear torque-speed curve: no torque left at no-load speed.
            s = qd * inv_speed
            upper = stall * np.minimum(np.maximum(1.0 - s, 0.0), 1.0)
            lower = -stall * np.minimum(np.maximum(1.0 + s, 0.0), 1.0)
            tau_c = np.minimum(np.maximum(tau, lower), upper)
            saturated += np.count_nonzero(tau != tau_c)
            tau_sq += tau_c * tau_c
            power += np.abs(tau_c * qd)
            overload += np.square(np.maximum(np.abs(tau_c) - 0.8 * stall, 0.0) / stall)
            speed_ratio = np.abs(qd) * inv_speed
            overspeed += np.maximum(speed_ratio - 1.0, 0.0)
            peak_speed_ratio = max(peak_speed_ratio, speed_ratio.max())
            peak_tau = np.maximum(peak_tau, np.abs(tau_c))
            d.ctrl[:] = tau_c
            mujoco.mj_step(m, d)
            peak_force = np.maximum(peak_force, self._foot_forces())
        energy = power.sum() * m.opt.timestep
        load = np.mean(tau_sq / stall ** 2)
        bad_contact = self._non_foot_ground_contact()
        self._t += self.dt
        self._steps += 1
        self._phase_acc += self.dt / self._gait_period()

        # ---------------------------------------------------------- state for rewards
        R = self._root_rot()
        grav = self._gravity_body()
        # Velocity along the episode's FIXED target direction, not the robot's current
        # facing: walking in circles averages to zero forward speed. Computed from the
        # distance moved over the whole control step, so no instant can be "timed".
        ch, sh = np.cos(self._heading0), np.sin(self._heading0)
        vel_xy = (d.qpos[0:2] - xy_before) / self.dt
        vx = ch * vel_xy[0] + sh * vel_xy[1]
        vy = -sh * vel_xy[0] + ch * vel_xy[1]
        yaw_err = self._heading_error()
        offset = self._lateral_offset()
        ang_world = R @ d.qvel[3:6]
        qd = d.qvel[self._dof_idx]
        qacc = (qd - self._last_qvel) / self.dt
        prev_contact = self._contact
        self._contact = (peak_force > 0.05 * self._weight).astype(float)
        new_td = (self._contact == 1) & (prev_contact == 0)
        touchdowns = float(np.sum(new_td))
        # Step timing from touchdowns: the robot's real stride length (for the symmetry terms) and
        # whether left->right and right->left take equally long (a lopsided rhythm is a skip).
        for leg in np.flatnonzero(new_td):
            other = 1 - leg
            if self._td_step[leg] is not None and 6 <= self._steps - self._td_step[leg] <= 40:
                self._stride_est = self._steps - self._td_step[leg]
            if self._td_step[other] is not None and self._td_last_leg == other:
                self._td_intervals.append(self._steps - self._td_step[other])
                if len(self._td_intervals) == 2:
                    i1, i2 = self._td_intervals
                    self._timing_asym = abs(i1 - i2) / (i1 + i2)
            elif self._td_last_leg == leg:
                self._td_intervals.clear()  # same foot twice: not an alternating pair, restart
            self._td_step[leg] = self._steps
            self._td_last_leg = leg
        # Direction reversals: a joint clearly moving one way (>0.2 rad/s) then clearly the other.
        sign = np.where(np.abs(qd) > 0.2, np.sign(qd), 0.0)
        reversals = float(np.sum((sign != 0) & (self._qd_sign != 0) & (sign != self._qd_sign)))
        self._qd_sign = np.where(sign != 0, sign, self._qd_sign)
        sole_pos, sole_vel = self._soles()
        self._clearance = sole_pos[:, 2] - self._home_sole_z
        c = cfg.command
        leg_phase = (self._phase() + np.array([0.0, 0.5])) % 1.0
        swing = leg_phase < c.swing_ratio
        q = d.qpos[self._qpos_idx]
        margin = 0.05 * (self._jnt_hi - self._jnt_lo)
        feet_y = (R.T @ (d.xpos[self._foot_body[0]] - d.xpos[self._foot_body[1]]))[1]
        w = cfg.rewards

        # How well the speed command is met, in [0, 1]. Narrow kernel for precision plus a
        # broad one so that creeping forward already beats standing still.
        err = vx - self._cmd
        share = w.tracking_broad_share
        tracking = ((1 - share) * np.exp(-err ** 2 / w.tracking_sigma2)
                    + share * np.exp(-err ** 2 / w.tracking_broad_sigma2))
        self._ep_speed_err += abs(err)

        # Task terms: only pay while the command is being met (gated by `tracking`).
        task = {
            "tracking_vel": w.tracking_vel,
            "tracking_lat_yaw": w.tracking_lat_yaw * np.exp(-(vy ** 2 + 0.25 * ang_world[2] ** 2) / 0.02),
            "heading": w.heading * np.exp(-(yaw_err ** 2) / 0.05),  # ~13deg off -> 0.37
            "gait_contact": w.gait_contact * np.mean(self._contact == ~swing),
            "feet_clearance": w.feet_clearance * np.mean(
                swing * np.exp(-(((self._clearance - c.swing_height) / (0.5 * c.swing_height)) ** 2))),
            "flight": w.flight * float(self._contact.sum() == 0 and vx > 0.5 * self._cmd > 0),
        }
        task = {k: tracking * v for k, v in task.items()}

        # Penalties (>= 0). Their sum scales the whole reward by exp(-sum).
        # Slip counts while the sole is loaded or actually touching (<3 mm): catches light-touch
        # dragging, but a normal step passing near the floor at lift-off/touch-down is not slip.
        on_floor = np.maximum(self._contact, self._clearance < 0.003)
        pen = {
            "path_offset": w.path_offset * min(offset ** 2, 1.0),
            "orientation": w.orientation * np.sum(grav[:2] ** 2),
            "ang_vel_xy": w.ang_vel_xy * np.sum(d.qvel[3:5] ** 2),
            "lin_vel_z": w.lin_vel_z * d.qvel[2] ** 2,
            "base_height": w.base_height * min(self._com_height() - self._home_height, 0.0) ** 2,
            "knee_hyperextension": w.knee_hyperextension * np.sum(np.maximum(-q[self._knee_idx], 0.0)),
            "hip_roll": w.hip_roll * np.sum(q[self._hip_roll_idx] ** 2),
            "hip_roll_inward": w.hip_roll_inward * np.sum(np.maximum(-self._roll_out_sign * q[self._hip_roll_idx], 0.0)),
            "action_rate": w.action_rate * np.mean((action - self._last_action) ** 2),
            "action_acc": w.action_acc * np.mean((action - 2 * self._last_action + self._last_action2) ** 2),
            "joint_acc": w.joint_acc * np.mean(qacc ** 2),
            "servo_load": w.servo_load * load / self.frame_skip,
            "servo_saturation": w.servo_saturation * saturated / (self.frame_skip * self.nu),
            "servo_overload": w.servo_overload * overload.sum() / self.frame_skip,
            "backdrive": w.backdrive * overspeed.sum() / self.frame_skip,
            "impact": w.impact * np.sum(np.clip(peak_force / self._weight - w.impact_threshold, 0, 3)),
            "foot_slip": w.foot_slip * np.sum(on_floor * np.sum(sole_vel[:, :2] ** 2, axis=1)),
            "joint_limits": w.joint_limits * np.sum(np.clip(self._jnt_lo + margin - q, 0, None)
                                                    + np.clip(q - (self._jnt_hi - margin), 0, None)),
            "feet_distance": w.feet_distance * max(w.min_feet_distance - feet_y, 0.0),
            "energy": w.energy * energy / self.dt,
        }
        # Symmetry: compare each leg now with the other leg half a stride ago (mirrored).
        # A symmetric gait scores 0; a limp (one leg held or swung differently) is penalized.
        self._q_hist.append(q.copy())
        stride = self._stride_est or int(round(self._gait_period() / self.dt))  # measured if available
        half = int(round(0.5 * stride))
        if len(self._mirror_l) and len(self._q_hist) > half:
            past = self._q_hist[-1 - half]
            asym = (np.mean((q[self._mirror_l] - self._mirror_sign * past[self._mirror_r]) ** 2)
                    + np.mean((q[self._mirror_r] - self._mirror_sign * past[self._mirror_l]) ** 2))
        else:
            asym = 0.0
        pen["symmetry"] = w.symmetry * asym
        # Posture symmetry: average mirrored left-right difference over the last stride. Catches a
        # leg held at a constant offset (which the half-stride comparison above lets through)
        # without penalizing how big the motions are.
        stride_steps = max(stride, 1)
        if len(self._mirror_l) and len(self._q_hist) >= stride_steps:
            recent_q = np.array(list(self._q_hist)[-stride_steps:])
            posture_gap = (recent_q[:, self._mirror_l] - self._mirror_sign * recent_q[:, self._mirror_r]).mean(axis=0)
            pen["posture_symmetry"] = w.posture_symmetry * float(np.mean(posture_gap ** 2))
        else:
            pen["posture_symmetry"] = 0.0
        pen["step_timing"] = w.step_timing * self._timing_asym  # charged every step, not only at touchdown
        pen["gait_mismatch"] = w.gait_mismatch * float(np.mean(self._contact != ~swing))
        # Footsteps are free up to the gait clock's plan (2 per stride, i.e. one per foot); only
        # extra steps within the last stride cost. Charging every step made walking slower the
        # cheapest way to take fewer steps.
        self._td_hist.append(touchdowns)
        stride_steps = int(round(self._gait_period() / self.dt))
        recent = sum(list(self._td_hist)[-stride_steps:])
        pen["footstep"] = w.footstep * min(touchdowns, max(recent - 2.0, 0.0))
        pen["reversals"] = w.reversals * reversals
        # Not moving while commanded to walk: 0 at >= half the commanded speed, 1 when stopped.
        if self._cmd > 0.1:
            pen["stand_still"] = w.stand_still * float(np.clip(1.0 - vx / (0.5 * self._cmd), 0.0, 1.0))
        else:
            pen["stand_still"] = 0.0
        # Recovery is free: wear and symmetry costs pause for 1 s after a shove or while the torso
        # tilts > 8 deg. Otherwise they punish the quick extra step that catches a fall.
        recovering = (self._t - self._last_push < 1.0) or np.linalg.norm(grav[:2]) > np.sin(np.radians(8))
        if recovering:
            for k in ("symmetry", "posture_symmetry", "step_timing", "gait_mismatch", "footstep", "reversals"):
                pen[k] = 0.0
        penalty = sum(pen.values())
        reward = (w.alive + sum(task.values())) * float(np.exp(-penalty))
        impact_bw = peak_force.max() / self._weight
        torque_ratio = float((peak_tau / self.cfg.robot.stall_torque).max())
        for key, val in (("torque", torque_ratio), ("speed", peak_speed_ratio), ("impact", impact_bw)):
            self._ep_peak[key] = max(self._ep_peak[key], val)
        # Hardware first: a landing or back-drive that could break the robot ends the episode
        # exactly like a fall.
        broke = bool(impact_bw > w.break_impact or peak_speed_ratio > w.break_joint_speed)
        fell = bool(bad_contact or grav[2] > -0.6 or self._com_height() < 0.5 * self._home_height)
        terminated = fell or broke
        if terminated:
            reward += w.termination
        truncated = self._steps >= self._max_steps
        if terminated or truncated:
            self._update_speed_curriculum(fell=terminated)

        # Random shoves (robustness).
        rc = cfg.randomization
        if rc.enabled and self._t >= self._next_push:
            push = self.np_random.uniform(-rc.push_velocity, rc.push_velocity, 2)
            d.qvel[0:2] += push
            self._last_push = self._t
            self._last_push_vec = push  # for drawing the shove in videos
            self._next_push += self.np_random.uniform(*rc.push_interval)

        self._last_action2 = self._last_action
        self._last_action = action
        self._last_qvel = qd.copy()
        self._history.append(self._sensor_frame())

        info = {f"rew/{k}": v for k, v in task.items()}
        info.update({f"pen/{k}": v for k, v in pen.items()})
        info.update({
            "forward_velocity": vx, "command": self._cmd, "x_position": d.qpos[0],
            "speed_error": abs(err), "tracking": tracking, "penalty_scale": float(np.exp(-penalty)),
            "speed_max": self._speed_max,
            "heading_error_deg": abs(np.degrees(yaw_err)), "lateral_offset": abs(offset),
            "energy": energy, "flight": float(self._contact.sum() == 0),
            "double_support": float(self._contact.sum() == 2),
            "peak_force_bw": peak_force.max() / self._weight,
            "saturation": saturated / (self.frame_skip * self.nu),
            "peak_torque_stall": torque_ratio,
            "max_joint_speed_ratio": peak_speed_ratio,
            "broke": float(broke), "fell": float(fell and not broke),
            "action_delta": np.mean(np.abs(action - self._last_action2)),
            "base_height": self._com_height(),
            "recovering": float(recovering),
        })
        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ render
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, *self._render_size)
            self._camera = mujoco.MjvCamera()
            self._camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self._camera.trackbodyid = self._root
            self._camera.distance = 4 * self._home_height
            self._camera.elevation = -15
            self._camera.azimuth = 120
        self._renderer.update_scene(self.data, camera=self._camera)
        self._draw_track_markers(self._renderer.scene)
        self._draw_push_arrow(self._renderer.scene)
        return self._renderer.render()

    def _draw_push_arrow(self, scene):
        """Red arrow at the torso for 0.4 s after each random shove, pointing the way it was pushed;
        longer arrow = harder shove."""
        if self._t - getattr(self, "_last_push", -np.inf) > 0.4 or scene.ngeom >= scene.maxgeom:
            return
        vec = np.array([*self._last_push_vec, 0.0])
        direction = vec / (np.linalg.norm(vec) + 1e-9)
        torso = self.data.subtree_com[self._root].copy()
        length = 0.08 + 0.5 * np.linalg.norm(vec)
        start, end = torso - direction * (0.06 + length), torso - direction * 0.06
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array([1.0, 0.1, 0.1, 1.0], dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.012, start, end)
        scene.ngeom += 1

    def _draw_track_markers(self, scene):
        """The camera follows the robot, so draw the straight target line with a red start
        line and white marks every 0.5 m: progress (and drift) are visible in any video."""
        heading = np.array([np.cos(self._heading0), np.sin(self._heading0), 0.0])
        side = np.array([-heading[1], heading[0], 0.0])
        rot = np.column_stack([heading, side, [0.0, 0.0, 1.0]]).flatten()
        start = np.array([*self._start_xy, 0.001])
        marks = [(0.0, [0.006, 0.15, 0.001], [0.9, 0.1, 0.1, 1])]                      # start line
        marks += [(0.5 * k, [0.004, 0.08, 0.001], [1, 1, 1, 0.8]) for k in range(1, 41)]  # every 0.5 m
        marks += [(10.0, [10.0, 0.002, 0.001], [1, 1, 1, 0.35])]                     # target line, 20 m
        for dist, size, rgba in marks:
            if scene.ngeom >= scene.maxgeom:
                break
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.array(size),
                                start + dist * heading, rot, np.array(rgba, dtype=np.float32))
            scene.ngeom += 1

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
