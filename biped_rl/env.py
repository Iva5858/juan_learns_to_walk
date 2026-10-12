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
        # Push curriculum: one strength per push direction (relative to the robot), kept across episodes.
        self._push_level = np.full(8, cfg.randomization.push_start)
        self._push_tally = np.zeros((8, 2))  # per direction: [stayed up, pushes]
        self._pending_push = None            # (direction, time) of a push whose outcome is not known yet
        # Physics curriculum (see RandomizationConfig): kept across episodes, like the others.
        rc = cfg.randomization
        self._difficulty = rc.difficulty_start if rc.physics_curriculum else 1.0
        self._episode_tally = np.zeros(2)    # [stayed up, episodes] since the last difficulty change

        self._omni = cfg.command.mode == "omni"
        frame_dim = 6 + (2 * self.nu if cfg.robot.joint_feedback else 0) + self.nu
        cmd_dim = 3 if self._omni else 1  # (forward, sideways, turn rate) or forward only
        self.actor_obs_dim = cfg.history * frame_dim + cmd_dim + 2 + 2  # + clock (2), heading error (2)
        privileged_dim = 1 + 3 + 1 + 2 + 2 + 2 * self.nu + 1
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.nu,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, dtype=np.float64,
                                            shape=(self.actor_obs_dim + privileged_dim,))
        # Sensor readings and actions are kept separately: readings can arrive late (obs_delay_steps),
        # the policy's own previous actions never do.
        self._sensor_buf = deque(maxlen=cfg.history + cfg.randomization.obs_delay_steps[1])
        self._action_buf = deque(maxlen=cfg.history)
        self._target_buf = deque(maxlen=cfg.randomization.action_delay_steps[1] + 1)
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
        self._act_delay = self._obs_delay = 0
        self._backlash = np.zeros(self.nu)
        self._joint_off = np.zeros(self.nu)
        self._grav_bias = np.eye(3)
        self._gyro_bias = np.zeros(3)

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
        if c.voltage_coupled:  # one battery voltage: torque and speed rise and fall together
            volt = rng.uniform(*c.servo_strength)
            self._stall = r.stall_torque * volt * rng.uniform(0.95, 1.05, nu)
            self._speed = r.no_load_speed * volt * rng.uniform(0.95, 1.05, nu)
        else:
            self._stall = r.stall_torque * rng.uniform(*c.servo_strength) * rng.uniform(0.95, 1.05, nu)
            self._speed = r.no_load_speed * rng.uniform(*c.servo_speed) * rng.uniform(0.95, 1.05, nu)
        self._kp = r.kp * rng.uniform(*c.kp_scale, size=nu)
        self._kd = r.kd * rng.uniform(*c.kd_scale, size=nu)
        lat = int(rng.uniform(*c.latency) / m.opt.timestep)
        self._latency_steps = min(lat, self.frame_skip - 1)
        # Below: drawn after everything above and only when switched on, so configs without these
        # settings (runs up to juan_v6) get exactly the same random draws as when they were trained.
        # The v7 realism below is scaled by the physics-curriculum difficulty (1 = full ranges).
        k = self._difficulty
        upto = lambda rng_range: (rng_range[0], rng_range[0] + k * (rng_range[1] - rng_range[0]))
        if tuple(c.total_mass_scale) != (1.0, 1.0) or tuple(c.torso_mass_scale) != (1.0, 1.0):
            extra = np.full(m.nbody, rng.uniform(*upto(c.total_mass_scale)))
            extra[self._root] *= rng.uniform(*upto(c.torso_mass_scale))
            m.body_mass[:] *= extra
            m.body_inertia[:] *= extra[:, None]
        if c.action_delay_steps[1] > 0:
            lo, hi = upto(c.action_delay_steps)
            self._act_delay = int(rng.integers(lo, int(round(hi)) + 1))
        if c.obs_delay_steps[1] > 0:
            lo, hi = upto(c.obs_delay_steps)
            self._obs_delay = int(rng.integers(lo, int(round(hi)) + 1))
        if c.backlash[1] > 0:
            self._backlash = rng.uniform(*upto(c.backlash), size=nu)
        if c.joint_offset > 0:
            self._joint_off = rng.uniform(-k * c.joint_offset, k * c.joint_offset, nu)
        if c.imu_tilt_bias_deg > 0:
            bias = k * c.imu_tilt_bias_deg
            roll, pitch = np.radians(rng.uniform(-bias, bias, 2))
            cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
            self._grav_bias = (np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
                               @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]))
        if c.gyro_bias > 0:
            self._gyro_bias = rng.uniform(-k * c.gyro_bias, k * c.gyro_bias, 3)

    def curriculum_state(self):
        """Curriculum levels (top speed, physics difficulty, push strengths), to carry them over to the
        next training run: they live in the env processes and would otherwise start over."""
        return {"speed_max": float(self._speed_max), "difficulty": float(self._difficulty),
                "push_level": self._push_level.tolist()}

    def set_curriculum_state(self, state):
        self._speed_max = float(state["speed_max"])
        self._push_level = np.array(state["push_level"], dtype=float)
        if self.cfg.randomization.physics_curriculum:  # otherwise the robot stays at full difficulty
            self._difficulty = float(state["difficulty"])

    def _score_episode(self, stayed_up):
        """Physics curriculum: every difficulty_window episodes, harder if the robot stayed up in at
        least difficulty_success of them, easier otherwise."""
        rc = self.cfg.randomization
        self._episode_tally += (float(stayed_up), 1.0)
        if self._episode_tally[1] >= rc.difficulty_window:
            ok = self._episode_tally[0] / self._episode_tally[1] >= rc.difficulty_success
            self._difficulty = float(np.clip(self._difficulty + (rc.difficulty_step if ok else -rc.difficulty_step),
                                             0.0, 1.0))
            self._episode_tally[:] = 0.0

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

    def _read_sensors(self):
        """What the real robot measures this control step: IMU gravity direction and gyro (with a
        per-episode bias and noise), and servo angles (with a calibration offset, the servo's
        0.24 deg resolution and noise) and speeds."""
        d, c, rng = self.data, self.cfg.randomization, self.np_random
        noisy = c.enabled
        noise = lambda scale, n: rng.normal(0, scale, n) if noisy else 0.0
        parts = [self._grav_bias @ self._gravity_body() + noise(c.noise_gravity, 3),
                 d.qvel[3:6] + self._gyro_bias + noise(c.noise_gyro, 3)]  # free-joint ang. vel. is in body frame
        if self.cfg.robot.joint_feedback:
            q = d.qpos[self._qpos_idx] - self._joint_off  # what a miscalibrated servo reports
            if noisy and c.joint_resolution > 0:
                q = np.round(q / c.joint_resolution) * c.joint_resolution
            parts += [q - self._home_joint + noise(c.noise_joint_pos, self.nu),
                      0.1 * (d.qvel[self._dof_idx] + noise(c.noise_joint_vel, self.nu))]
        return np.concatenate(parts)

    def _history_frames(self):
        """Last `history` frames, oldest first: each is [sensor reading, action of that step].
        Readings are obs_delay steps old; the oldest available one stands in at episode start."""
        frames = []
        for j in reversed(range(self.cfg.history)):
            back = min(1 + j + self._obs_delay, len(self._sensor_buf))
            frames.append(np.concatenate([self._sensor_buf[-back], self._action_buf[-1 - j]]))
        return frames

    def _gait_period(self):
        """Faster rhythm at higher commands: a fixed rhythm caps top speed."""
        c = self.cfg.command
        speed = np.hypot(self._cmd_vec[0], self._cmd_vec[1])
        frac = min(speed / c.fast_speed, 1.0)
        return c.gait_period - (c.gait_period - c.gait_period_fast) * frac

    def _phase(self):
        return self._phase_acc % 1.0

    def _sample_command(self):
        """Draw (forward, sideways, turn rate) and whether to stand. Forward speeds come from
        [-backward_speed, top of the speed curriculum]; a share of commands is straight walking."""
        c, rng = self.cfg.command, self.np_random
        self._standing = bool(rng.uniform() < c.stand_prob)
        if self._standing:
            self._cmd_vec = np.zeros(3)
        else:
            vx = rng.uniform(-c.backward_speed, self._speed_max)
            if rng.uniform() < c.straight_prob:
                vy = wz = 0.0
            else:
                vy, wz = rng.uniform(*c.lateral_range), rng.uniform(*c.yaw_rate_range)
            self._cmd_vec = np.array([vx, vy, wz])
        self._cmd = float(self._cmd_vec[0])
        self._ep_max_cmd = max(self._ep_max_cmd, self._cmd)

    def _heading_error(self):
        """Yaw relative to the target heading, wrapped to [-pi, pi]. The target is the start
        heading turned by the integrated turn-rate command. On the robot: IMU yaw minus the
        yaw recorded at the start, minus the integrated joystick turn rate."""
        R = self._root_rot()
        yaw = np.arctan2(R[1, 0], R[0, 0])
        return (yaw - self._heading_target + np.pi) % (2 * np.pi) - np.pi

    def _lateral_offset(self):
        """Signed sideways distance from the commanded path (privileged). The reference point
        follows the robot along the target heading and moves sideways only as commanded, so
        drifting off a straight line is measured, but falling behind on speed is not."""
        side = np.array([-np.sin(self._heading_target), np.cos(self._heading_target)])
        return float(side @ (self.data.qpos[0:2] - self._ref_xy))

    def _get_obs(self):
        d = self.data
        ph = 2 * np.pi * self._phase()
        yaw_err = self._heading_error() + (self.np_random.normal(0, self.cfg.randomization.noise_yaw)
                                           if self.cfg.randomization.enabled else 0.0)
        cmd = self._cmd_vec if self._omni else [self._cmd]
        clock = [0.0, 0.0] if self._standing else [np.sin(ph), np.cos(ph)]  # stopped clock = stand
        actor = np.concatenate([*self._history_frames(), cmd, clock, [np.sin(yaw_err), np.cos(yaw_err)]])
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
        self._ref_xy = self._start_xy.copy()
        self._heading_target = self._heading0
        d.qpos[self._qpos_idx] += rng.uniform(-0.05, 0.05, self.nu)
        d.qvel[:] = rng.uniform(-0.05, 0.05, self.model.nv)
        mujoco.mj_forward(self.model, d)
        if cfg.randomization.enabled:
            self._vary_start_state()

        return self._start_episode(options or {})

    def _start_episode(self, options):
        """Everything an episode starts with besides the physics: commands, timers, histories."""
        cfg, rng, d = self.cfg, self.np_random, self.data
        # Commands. options={"speed": v} (forward only) or {"command": (vx, vy, yaw_rate)} or
        # {"stand": True} fix the command for the whole episode (evaluation, videos).
        self._cmd_fixed = any(k in options for k in ("speed", "command", "stand"))
        self._standing, self._ep_max_cmd = False, -np.inf
        if "stand" in options:
            self._standing, self._cmd_vec = bool(options["stand"]), np.zeros(3)
        elif "command" in options:
            self._cmd_vec = np.array(options["command"], dtype=float)
        elif "speed" in options:
            self._cmd_vec = np.array([float(options["speed"]), 0.0, 0.0])
        elif self._omni:
            self._sample_command()
        else:
            self._cmd_vec = np.array([rng.uniform(cfg.command.speed_range[0], self._speed_max), 0.0, 0.0])
        self._cmd = float(self._cmd_vec[0])
        self._ep_max_cmd = max(self._ep_max_cmd, self._cmd)
        # (Only drawn in omni mode, so forward-mode runs see the same random stream as before.)
        self._next_resample = rng.uniform(*cfg.command.resample_interval) if self._omni else np.inf
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
        self._pending_push = None
        self._q_hist.clear()
        self._td_hist.clear()
        self._td_step = [None, None]   # control step of each foot's last touchdown
        self._td_last_leg = None
        self._td_intervals = deque(maxlen=2)
        self._stride_est = None        # measured steps per full stride (from one foot's touchdowns)
        self._timing_asym = 0.0        # latest measured left/right step-timing asymmetry
        reading = self._read_sensors()
        self._sensor_buf.extend([reading] * self._sensor_buf.maxlen)
        self._action_buf.extend([self._last_action] * cfg.history)
        self._target_buf.extend([self._target] * self._target_buf.maxlen)
        return self._get_obs(), {}

    def _vary_start_state(self):
        """Start from a perturbed crouch, moving (RandomizationConfig.reset_*). The robot is then
        lowered or raised so its lowest sole sits where it does in the home pose: bending the legs
        changes their length, and a foot starting inside the floor would be shot out of it."""
        c, d, rng = self.cfg.randomization, self.data, self.np_random
        if tuple(c.reset_joint_scale) == (1.0, 1.0) and c.reset_lin_vel == 0 and c.reset_ang_vel == 0:
            return  # off (runs up to juan_v7): no extra random draws
        qi = self._qpos_idx
        noise = d.qpos[qi] - self._home_joint
        d.qpos[qi] = np.clip(self._home_joint * rng.uniform(*c.reset_joint_scale, self.nu) + noise,
                             self._jnt_lo, self._jnt_hi)
        d.qvel[0:2] += rng.uniform(-c.reset_lin_vel, c.reset_lin_vel, 2)
        d.qvel[3:6] += rng.uniform(-c.reset_ang_vel, c.reset_ang_vel, 3)
        mujoco.mj_forward(self.model, d)
        self._ground()

    def _ground(self):
        """Raise or lower the robot so its lowest sole sits where it does in the home pose."""
        self.data.qpos[2] -= self._soles()[0][:, 2].min() - self._home_sole_z.min()
        mujoco.mj_forward(self.model, self.data)

    def _episode_was_safe(self):
        w = self.cfg.rewards
        return (self._ep_peak["torque"] < w.safe_torque and self._ep_peak["speed"] < w.safe_joint_speed
                and self._ep_peak["impact"] < w.safe_impact)

    def _update_speed_curriculum(self, fell):
        """Promote the top speed when top commands are tracked through a whole episode;
        demote when the robot falls at the top. Each training env keeps its own."""
        c = self.cfg.command
        top = self._ep_max_cmd  # with commands re-drawn mid-episode: the fastest one asked for
        if self._cmd_fixed or not self.cfg.randomization.enabled or top < 0.75 * self._speed_max:
            return
        if fell:
            self._speed_max = max(c.speed_range[1], self._speed_max - c.curriculum_step)
        elif self._ep_speed_err / self._steps < 0.25 * top and self._episode_was_safe():
            self._speed_max = min(c.speed_limit, self._speed_max + c.curriculum_step)

    def _score_push(self, k, stayed_up):
        """Push curriculum: after every push_window pushes from direction k, make that direction
        stronger if the robot stayed up through at least push_success of them, weaker otherwise."""
        rc = self.cfg.randomization
        self._pending_push = None
        self._push_tally[k] += (float(stayed_up), 1.0)
        if self._push_tally[k, 1] >= rc.push_window:
            ok = self._push_tally[k, 0] / self._push_tally[k, 1] >= rc.push_success
            self._push_level[k] = np.clip(self._push_level[k] + (rc.push_step if ok else -rc.push_step),
                                          rc.push_start, rc.push_velocity)
            self._push_tally[k] = 0.0

    def _servo_torque(self, target, slack, inv_speed):
        """The hobby-servo model at one physics step: (joint speed, PD torque, applied torque)."""
        d = self.data
        qd = d.qvel[self._dof_idx]
        # A miscalibrated servo holds its own idea of the angle: physically target + offset.
        err = target + self._joint_off - d.qpos[self._qpos_idx]
        if slack is not None:  # gear slack: no restoring torque until the gap is taken up
            err = np.sign(err) * np.maximum(np.abs(err) - slack, 0.0)
        tau = self._kp * err - self._kd * qd
        # Linear torque-speed curve: no torque left at no-load speed.
        s = qd * inv_speed
        upper = self._stall * np.minimum(np.maximum(1.0 - s, 0.0), 1.0)
        lower = -self._stall * np.minimum(np.maximum(1.0 + s, 0.0), 1.0)
        return qd, tau, np.minimum(np.maximum(tau, lower), upper)

    # ------------------------------------------------------------------ scripted motion (play.py)
    def place_pose(self, joints):
        """Put the robot standing still in joint pose `joints` (absolute angles), soles on the floor."""
        d = self.data
        d.qpos[self._qpos_idx] = np.clip(joints, self._jnt_lo, self._jnt_hi)
        d.qvel[:] = 0.0
        self._target = d.qpos[self._qpos_idx].copy()
        mujoco.mj_forward(self.model, d)
        self._ground()

    def hold_pose(self, joints):
        """One control step with the servos driven straight to joint angles `joints`: no policy,
        no reward. For scripted motions, e.g. the boot sequence (straight legs -> home crouch);
        the policy's actions only reach +-action_scale around home."""
        self._target = np.clip(joints, self._jnt_lo, self._jnt_hi)
        slack = self._backlash if self._backlash.any() else None
        for _ in range(self.frame_skip):
            *_, tau_c = self._servo_torque(self._target, slack, 1.0 / self._speed)
            self.data.ctrl[:] = tau_c
            mujoco.mj_step(self.model, self.data)
        return not (self._non_foot_ground_contact() or self._gravity_body()[2] > -0.6)  # still up?

    def start_policy(self, options=None):
        """Hand over to the policy from wherever the robot is now (after a scripted motion), the
        way the real robot will: the start heading is the current one, histories are filled with
        the current readings, the last action and the filter are zero. Returns the observation."""
        R = self._root_rot()
        self._heading0 = float(np.arctan2(R[1, 0], R[0, 0]))
        self._heading_target = self._heading0
        self._start_xy = self.data.qpos[0:2].copy()
        self._ref_xy = self._start_xy.copy()
        return self._start_episode(options or {"stand": True})[0]

    def set_command(self, forward, sideways=0.0, turn_rate=0.0):
        """Change the command mid-episode (keyboard, joystick). All zero = stand still."""
        self._cmd_vec = np.array([forward, sideways, turn_rate], dtype=float)
        self._cmd = float(forward)
        self._standing = not np.any(self._cmd_vec)

    def step(self, action):
        cfg, m, d = self.cfg, self.model, self.data
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        a = cfg.action_filter
        self._filtered = a * action + (1 - a) * self._filtered
        new_target = np.clip(self._home_joint + cfg.action_scale * self._filtered, self._jnt_lo, self._jnt_hi)
        self._target_buf.append(new_target)
        new_target = self._target_buf[-1 - min(self._act_delay, len(self._target_buf) - 1)]  # bus delay
        w = cfg.rewards
        slack = self._backlash if self._backlash.any() else None
        loaded_overspeed = False

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
            qd, tau, tau_c = self._servo_torque(self._target, slack, inv_speed)
            saturated += np.count_nonzero(tau != tau_c)
            tau_sq += tau_c * tau_c
            power += np.abs(tau_c * qd)
            overload += np.square(np.maximum(np.abs(tau_c) - 0.8 * stall, 0.0) / stall)
            speed_ratio = np.abs(qd) * inv_speed
            overspeed += np.maximum(speed_ratio - 1.0, 0.0)
            peak_speed_ratio = max(peak_speed_ratio, speed_ratio.max())
            if w.break_requires_load > 0:  # over-speed while the servo fights it: gear-stripping load
                loaded_overspeed |= bool(np.any((speed_ratio > w.break_joint_speed)
                                                & (np.abs(tau_c) > w.break_requires_load * stall)))
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
        cvx, cvy, cwz = self._cmd_vec
        self._heading_target += cwz * self.dt

        # ---------------------------------------------------------- state for rewards
        R = self._root_rot()
        grav = self._gravity_body()
        # Velocity along the TARGET heading (the start heading, turned only as commanded), not
        # the robot's current facing: walking in circles earns no forward speed. Computed from
        # the distance moved over the whole control step, so no instant can be "timed".
        ch, sh = np.cos(self._heading_target), np.sin(self._heading_target)
        vel_xy = (d.qpos[0:2] - xy_before) / self.dt
        vx = ch * vel_xy[0] + sh * vel_xy[1]
        vy = -sh * vel_xy[0] + ch * vel_xy[1]
        yaw_err = self._heading_error()
        # Commanded path: the reference moves sideways as commanded and is pulled along the
        # heading to wherever the robot is, so only sideways drift counts.
        ahead, side = np.array([ch, sh]), np.array([-sh, ch])
        self._ref_xy = self._ref_xy + cvy * self.dt * side
        self._ref_xy = self._ref_xy + ahead * (ahead @ (d.qpos[0:2] - self._ref_xy))
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
        if self._standing:
            swing = np.zeros(2, dtype=bool)  # both feet down: contact, clearance, mismatch follow
        q = d.qpos[self._qpos_idx]
        margin = 0.05 * (self._jnt_hi - self._jnt_lo)
        feet_y = (R.T @ (d.xpos[self._foot_body[0]] - d.xpos[self._foot_body[1]]))[1]
        w = cfg.rewards

        # How well the speed command is met, in [0, 1]. Narrow kernel for precision plus a
        # broad one so that creeping forward already beats standing still.
        err = vx - cvx
        err2 = err ** 2 + ((vy - cvy) ** 2 if self._omni else 0.0)  # omni: sideways speed counts too
        share = w.tracking_broad_share
        tracking = ((1 - share) * np.exp(-err2 / w.tracking_sigma2)
                    + share * np.exp(-err2 / w.tracking_broad_sigma2))
        self._ep_speed_err += np.sqrt(err2)

        # Task terms: only pay while the command is being met (gated by `tracking`).
        task = {
            "tracking_vel": w.tracking_vel,
            "tracking_lat_yaw": w.tracking_lat_yaw * np.exp(
                -((vy - cvy) ** 2 + 0.25 * (ang_world[2] - cwz) ** 2) / 0.02),
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
        allowed = 0.0 if self._standing else 2.0  # standing: every step is an extra one
        pen["footstep"] = w.footstep * min(touchdowns, max(recent - allowed, 0.0))
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
        overspeed_break = loaded_overspeed if w.break_requires_load > 0 else peak_speed_ratio > w.break_joint_speed
        broke = bool(impact_bw > w.break_impact or overspeed_break)
        fell = bool(bad_contact or grav[2] > -0.6 or self._com_height() < 0.5 * self._home_height)
        terminated = fell or broke
        if terminated:
            reward += w.termination
        truncated = self._steps >= self._max_steps
        if terminated or truncated:
            self._update_speed_curriculum(fell=terminated)
            if cfg.randomization.enabled and cfg.randomization.physics_curriculum and not self._cmd_fixed:
                self._score_episode(stayed_up=not terminated)

        # Random shoves (robustness).
        rc = cfg.randomization
        if self._pending_push is not None and (terminated or truncated or self._t - self._pending_push[1] >= 2.0):
            self._score_push(self._pending_push[0], stayed_up=not terminated)
        if rc.enabled and rc.push_velocity > 0 and self._t >= self._next_push:
            if rc.push_curriculum:
                k = int(self.np_random.integers(8))
                yaw = np.arctan2(R[1, 0], R[0, 0])
                angle = yaw + (k + self.np_random.uniform()) * np.pi / 4  # direction k: [k*45, (k+1)*45) deg
                strength = min(self._push_level[k] * self.np_random.uniform(0.75, 1.1), rc.push_velocity)
                push = strength * np.array([np.cos(angle), np.sin(angle)])
                self._pending_push = (k, self._t)
            else:
                push = self.np_random.uniform(-rc.push_velocity, rc.push_velocity, 2)
            d.qvel[0:2] += push
            self._last_push = self._t
            self._last_push_vec = push  # for drawing the shove in videos
            self._next_push += self.np_random.uniform(*rc.push_interval)

        # New command every few seconds (omni mode): the policy must change gait, turn, stop
        # and start again mid-walk, like with a joystick.
        if not self._cmd_fixed and self._t >= self._next_resample:
            self._sample_command()
            self._next_resample += self.np_random.uniform(*cfg.command.resample_interval)

        self._last_action2 = self._last_action
        self._last_action = action
        self._last_qvel = qd.copy()
        self._sensor_buf.append(self._read_sensors())
        self._action_buf.append(action)

        info = {f"rew/{k}": v for k, v in task.items()}
        info.update({f"pen/{k}": v for k, v in pen.items()})
        info.update({
            "forward_velocity": vx, "command": cvx, "x_position": d.qpos[0],
            "lateral_velocity": vy, "command_lateral": cvy,
            "yaw_rate": ang_world[2], "command_yaw_rate": cwz, "standing": float(self._standing),
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
            "push_level_mean": float(self._push_level.mean()), "push_level_min": float(self._push_level.min()),
            "physics_difficulty": self._difficulty,
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
