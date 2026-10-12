"""Gymnasium environment: a small servo-driven biped walking on command, in MuJoCo.

Works with any MJCF that has a <freejoint/> root, one actuator per driven joint, foot bodies named
in RobotConfig.foot_bodies and (optionally) a "home" keyframe. Whatever actuator type the MJCF
declares, the joints are driven by a hobby-servo model computed here: PD control, a torque-speed
curve, command delay, gear slack and a zero-calibration error.

The task itself lives in mdp/: commands, observations, rewards, terminations, events (domain
randomization, pushes) and curricula. This class runs the simulation and calls them in order.
"""
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from biped_rl.robots import resolve_model_path
from biped_rl.tasks.locomotion.env_cfg import EnvConfig
from biped_rl.tasks.locomotion.mdp import commands, curriculums, events, observations, rewards, terminations


@dataclass
class RobotModel:
    """What the environment knows about the robot model (fixed for the environment's life)."""
    root: int
    qpos_idx: np.ndarray        # actuated joints, in actuator order
    dof_idx: np.ndarray
    joint_lo: np.ndarray
    joint_hi: np.ndarray
    knee_idx: list
    hip_roll_idx: list
    mirror_l: np.ndarray        # mirror pairs (left index, right index, sign)
    mirror_r: np.ndarray
    mirror_sign: np.ndarray
    foot_bodies: np.ndarray     # (left, right)
    foot_geoms: list            # each foot's collision geoms
    geom_leg: np.ndarray        # per geom: 0 left foot, 1 right foot, -1 not a foot
    geom_is_world: np.ndarray
    home_joint: np.ndarray = None
    home_height: float = 0.0    # centre-of-mass height in the home pose
    home_sole_z: np.ndarray = None
    roll_out_sign: np.ndarray = None
    weight: float = 0.0         # N, nominal: the unit for landing forces
    nominal: dict = None        # the model's own masses, frictions... (randomization scales these)


@dataclass
class ServoState:
    """The servos' numbers this episode (randomized by events.randomize_physics)."""
    stall: np.ndarray = None
    speed: np.ndarray = None
    kp: np.ndarray = None
    kd: np.ndarray = None
    latency_steps: int = 0      # physics steps into a control step before a new target applies
    action_delay: int = 0       # whole control steps of extra delay
    obs_delay: int = 0          # control steps by which readings are late
    backlash: np.ndarray = None
    joint_offset: np.ndarray = None


@dataclass
class DomainState:
    """The rest of this episode's randomization: floor friction and IMU biases."""
    friction_scale: float = 1.0
    imu_tilt: np.ndarray = None
    gyro_bias: np.ndarray = None


@dataclass
class CommandState:
    vec: np.ndarray = field(default_factory=lambda: np.zeros(3))  # forward, sideways, turn rate
    standing: bool = False
    fixed: bool = False         # no new commands this episode
    next_resample: float = np.inf
    max_forward: float = -np.inf
    phase: float = 0.0          # gait clock, accumulated (mdp.commands.phase wraps it)
    heading0: float = 0.0       # start heading
    heading_target: float = 0.0
    start_xy: np.ndarray = None
    ref_xy: np.ndarray = None   # commanded path reference point

    @property
    def forward(self):
        return float(self.vec[0])


@dataclass
class CurriculumState:
    speed_max: float
    difficulty: float
    push_level: np.ndarray
    push_tally: np.ndarray = field(default_factory=lambda: np.zeros((8, 2)))  # per direction: stayed up, pushes
    episode_tally: np.ndarray = field(default_factory=lambda: np.zeros(2))    # stayed up, episodes
    pending_push: tuple = None  # (direction, time) of a push whose outcome is not known yet


@dataclass
class EpisodeState:
    """Everything that starts over with an episode."""
    target: np.ndarray          # servo targets in effect
    filtered: np.ndarray        # action filter state
    target_buf: deque           # targets waiting out the bus delay
    sensor_buf: deque           # readings (can arrive late)
    action_buf: deque           # the policy's own previous actions (never late)
    last_qvel: np.ndarray
    clearance: np.ndarray
    q_hist: deque               # joint angles, for the symmetry terms
    td_hist: deque              # touchdowns per step, for the footstep term
    next_push: float
    t: float = 0.0
    steps: int = 0
    speed_err_sum: float = 0.0
    peaks: dict = field(default_factory=lambda: {"torque": 0.0, "speed": 0.0, "impact": 0.0})
    last_action: np.ndarray = None
    last_action2: np.ndarray = None
    contact: np.ndarray = field(default_factory=lambda: np.ones(2))
    qd_sign: np.ndarray = None  # last clear direction of each joint, for reversal counting
    last_push: float = -np.inf
    last_push_vec: np.ndarray = None
    td_step: list = field(default_factory=lambda: [None, None])  # control step of each foot's last touchdown
    td_last_leg: int = None
    td_intervals: deque = field(default_factory=lambda: deque(maxlen=2))
    stride_est: int = None      # measured steps per stride (from one foot's touchdowns)
    timing_asym: float = 0.0    # latest left/right step-timing asymmetry


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
        self.nu = m.nu
        self.omni = cfg.command.mode == "omni"

        self.robot = self._load_robot()
        self.servo, self.domain = ServoState(), DomainState()
        events.set_nominal_physics(self)
        rc = cfg.randomization
        self.curriculum = CurriculumState(speed_max=cfg.command.speed_range[1],
                                          difficulty=rc.difficulty_start if rc.physics_curriculum else 1.0,
                                          push_level=np.full(8, rc.push_start))
        self.command = CommandState()

        self.actor_obs_dim = observations.actor_dim(cfg, self.nu, self.omni)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.nu,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, dtype=np.float64,
                                            shape=(self.actor_obs_dim + observations.privileged_dim(self.nu),))
        self._max_steps = int(cfg.episode_seconds / self.dt)

    # ------------------------------------------------------------------ setup
    def _load_robot(self):
        """Index the model: joints, feet, mirror pairs, the home pose and its nominal physics."""
        m, cfg = self.model, self.cfg
        if m.njnt == 0 or m.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("The first joint must be the root <freejoint/> of the robot.")
        joint_ids = m.actuator_trnid[:, 0]
        self._convert_actuators_to_torque(joint_ids)
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joint_ids]
        pairs = [(names.index(l), names.index(r), sgn) for l, r, sgn in cfg.robot.mirror_pairs
                 if l in names and r in names]
        feet = []
        for name in cfg.robot.foot_bodies:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"Foot body '{name}' not found; set RobotConfig.foot_bodies.")
            feet.append(bid)
        # Each foot's collision geoms, to find the sole's lowest point (toe and heel aware).
        geom_leg, foot_geoms = np.full(m.ngeom, -1), []
        for leg, bid in enumerate(feet):
            geom_leg[m.geom_bodyid == bid] = leg
            own = np.flatnonzero(m.geom_bodyid == bid)
            colliding = [g for g in own if m.geom_contype[g] or m.geom_conaffinity[g]]
            foot_geoms.append(colliding or list(own))
        lo, hi = m.jnt_range[joint_ids].T.copy()
        robot = RobotModel(
            root=m.jnt_bodyid[0], qpos_idx=m.jnt_qposadr[joint_ids], dof_idx=m.jnt_dofadr[joint_ids],
            joint_lo=lo, joint_hi=hi,
            knee_idx=[names.index(n) for n in cfg.robot.knee_joints if n in names],
            hip_roll_idx=[names.index(n) for n in cfg.robot.hip_roll_joints if n in names],
            mirror_l=np.array([p[0] for p in pairs], dtype=int), mirror_r=np.array([p[1] for p in pairs], dtype=int),
            mirror_sign=np.array([p[2] for p in pairs], dtype=float),
            foot_bodies=np.array(feet), foot_geoms=foot_geoms, geom_leg=geom_leg,
            geom_is_world=m.geom_bodyid == 0)
        self.robot = robot
        self._box_corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
        self._vel6 = np.zeros(6)
        self._home_key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        self._reset_to_home()
        robot.home_joint = self.data.qpos[robot.qpos_idx].copy()
        # Height = centre of mass, not the root body's origin: CAD exports often put the base frame
        # origin somewhere arbitrary (e.g. at floor level).
        robot.home_height = self.com_height()
        robot.home_sole_z = self.soles()[0][:, 2].copy()
        robot.roll_out_sign = self._hip_roll_outward_signs()
        robot.weight = m.body_subtreemass[robot.root] * -m.opt.gravity[2]
        robot.nominal = {"mass": m.body_mass.copy(), "inertia": m.body_inertia.copy(), "ipos": m.body_ipos.copy(),
                         "friction": m.geom_friction.copy(), "armature": m.dof_armature.copy(),
                         "frictionloss": m.dof_frictionloss.copy(), "gravity": m.opt.gravity.copy()}
        return robot

    def _convert_actuators_to_torque(self, joint_ids):
        """Turn every actuator into a plain torque motor; the servo model lives in servo_torque()."""
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
        if self._home_key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def _hip_roll_outward_signs(self):
        """+1 if positive hip roll swings that leg outward, -1 if inward. Measured on the model
        (nudge each roll joint, see whether the feet move away from the centre line), so it works
        whichever way the CAD's roll axes point."""
        m, robot = self.model, self.robot
        d = mujoco.MjData(m)
        signs = []
        for idx in robot.hip_roll_idx:
            spread = []
            for delta in (0.0, 0.05):
                d.qpos[:] = self.data.qpos
                d.qpos[robot.qpos_idx[idx]] += delta
                mujoco.mj_kinematics(m, d)
                R = d.xmat[robot.root].reshape(3, 3)
                feet_y = (d.xpos[robot.foot_bodies] - d.xpos[robot.root]) @ R[:, 1]
                spread.append(np.abs(feet_y).sum())
            signs.append(1.0 if spread[1] > spread[0] else -1.0)
        return np.array(signs)

    # Indirection so tools can swap the physics (e.g. pushes on nominal physics).
    def set_nominal_physics(self):
        events.set_nominal_physics(self)

    def randomize_physics(self):
        events.randomize_physics(self)

    # ------------------------------------------------------------------ geometry and sensing
    def root_rot(self):
        return self.data.xmat[self.robot.root].reshape(3, 3)

    def com_height(self):
        return float(self.data.subtree_com[self.robot.root][2])

    def gravity_body(self):
        g = self.model.opt.gravity
        return self.root_rot().T @ (g / np.linalg.norm(g))

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

    def soles(self):
        """Lowest point of each foot's collision shape (world) and that point's velocity. The sole,
        not the ankle: a foot pitched toe-down lifts the ankle while the toe still scrapes the floor."""
        m, d = self.model, self.data
        pts, vel = np.empty((2, 3)), np.empty((2, 3))
        for leg, bid in enumerate(self.robot.foot_bodies):
            p = min((self._geom_lowest_point(g) for g in self.robot.foot_geoms[leg]), key=lambda c: c[2])
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, bid, self._vel6, 0)
            vel[leg] = self._vel6[3:] + np.cross(self._vel6[:3], p - d.xpos[bid])
            pts[leg] = p
        return pts, vel

    def foot_forces(self):
        """Vertical contact force on each foot body (one C call, no Python loop)."""
        mujoco.mj_rnePostConstraint(self.model, self.data)
        return self.data.cfrc_ext[self.robot.foot_bodies, 5]

    def non_foot_ground_contact(self):
        """True when anything but a foot touches the ground (knee, torso...)."""
        d = self.data
        if d.ncon == 0:
            return False
        g1, g2 = d.contact.geom[: d.ncon].T
        w1, w2 = self.robot.geom_is_world[g1], self.robot.geom_is_world[g2]
        return bool(np.any((w1 ^ w2) & (self.robot.geom_leg[np.where(w1, g2, g1)] < 0)))

    def ground(self):
        """Raise or lower the robot so its lowest sole sits where it does in the home pose."""
        self.data.qpos[2] -= self.soles()[0][:, 2].min() - self.robot.home_sole_z.min()
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ servo model
    def servo_torque(self, target, slack, inv_speed):
        """The hobby-servo model at one physics step: (joint speed, PD torque, applied torque)."""
        d, sv = self.data, self.servo
        qd = d.qvel[self.robot.dof_idx]
        # A miscalibrated servo holds its own idea of the angle: physically target + offset.
        err = target + sv.joint_offset - d.qpos[self.robot.qpos_idx]
        if slack is not None:  # gear slack: no restoring torque until the gap is taken up
            err = np.sign(err) * np.maximum(np.abs(err) - slack, 0.0)
        tau = sv.kp * err - sv.kd * qd
        # Linear torque-speed curve: no torque left at no-load speed.
        s = qd * inv_speed
        upper = sv.stall * np.minimum(np.maximum(1.0 - s, 0.0), 1.0)
        lower = -sv.stall * np.minimum(np.maximum(1.0 + s, 0.0), 1.0)
        return qd, tau, np.minimum(np.maximum(tau, lower), upper)

    def _actuate(self, action):
        """Filter the action, delay it like the servo bus, and run the servos at physics rate.
        Returns what the rewards need from inside the control step (peaks, sums)."""
        cfg, m, d, sv, ep, robot = self.cfg, self.model, self.data, self.servo, self.episode, self.robot
        a = cfg.action_filter
        ep.filtered = a * action + (1 - a) * ep.filtered
        new_target = np.clip(robot.home_joint + cfg.action_scale * ep.filtered, robot.joint_lo, robot.joint_hi)
        ep.target_buf.append(new_target)
        new_target = ep.target_buf[-1 - min(sv.action_delay, len(ep.target_buf) - 1)]  # bus delay
        w = cfg.rewards
        slack = sv.backlash if sv.backlash.any() else None
        stall, inv_speed = sv.stall, 1.0 / sv.speed
        loaded_overspeed = False
        peak_force, tau_sq, power, saturated = np.zeros(2), np.zeros(self.nu), np.zeros(self.nu), 0
        overload, overspeed, peak_tau = np.zeros(self.nu), np.zeros(self.nu), np.zeros(self.nu)
        peak_speed_ratio = 0.0
        xy_before = d.qpos[0:2].copy()
        for k in range(self.frame_skip):  # (plain ufuncs: this loop dominates the step time)
            if k == sv.latency_steps:
                ep.target = new_target
            qd, tau, tau_c = self.servo_torque(ep.target, slack, inv_speed)
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
            peak_force = np.maximum(peak_force, self.foot_forces())
        return SimpleNamespace(peak_force=peak_force, saturated=saturated, overload=overload, overspeed=overspeed,
                               peak_tau=peak_tau, peak_speed_ratio=peak_speed_ratio,
                               loaded_overspeed=loaded_overspeed, xy_before=xy_before,
                               energy=power.sum() * m.opt.timestep, load=np.mean(tau_sq / stall ** 2))

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        cfg, rng, d, cmd = self.cfg, self.np_random, self.data, self.command
        if cfg.randomization.enabled:
            self.randomize_physics()
        else:
            self.set_nominal_physics()
        self._reset_to_home()
        # Random start heading: the policy must learn "hold the heading you started with", not
        # "walk along world +x".
        cmd.heading0 = rng.uniform(-np.pi, np.pi) if cfg.randomization.enabled else 0.0
        yaw_quat = np.array([np.cos(cmd.heading0 / 2), 0.0, 0.0, np.sin(cmd.heading0 / 2)])
        quat = np.empty(4)
        mujoco.mju_mulQuat(quat, yaw_quat, d.qpos[3:7].copy())
        d.qpos[3:7] = quat
        cmd.start_xy = d.qpos[0:2].copy()
        cmd.ref_xy = cmd.start_xy.copy()
        cmd.heading_target = cmd.heading0
        d.qpos[self.robot.qpos_idx] += rng.uniform(-0.05, 0.05, self.nu)
        d.qvel[:] = rng.uniform(-0.05, 0.05, self.model.nv)
        mujoco.mj_forward(self.model, d)
        if cfg.randomization.enabled:
            events.vary_start_state(self)
        return self._start_episode(options or {})

    def _start_episode(self, options):
        """Everything an episode starts with besides the physics: commands, timers, histories."""
        cfg, rng, d, robot = self.cfg, self.np_random, self.data, self.robot
        commands.reset(self, options)
        stride_len = int(cfg.command.gait_period / self.dt) + 2
        self.command.phase = rng.uniform()
        self.episode = ep = EpisodeState(
            target=robot.home_joint.copy(), filtered=np.zeros(self.nu),
            target_buf=deque(maxlen=cfg.randomization.action_delay_steps[1] + 1),
            sensor_buf=deque(maxlen=cfg.history + cfg.randomization.obs_delay_steps[1]),
            action_buf=deque(maxlen=cfg.history),
            last_qvel=d.qvel[robot.dof_idx].copy(), clearance=self.soles()[0][:, 2] - robot.home_sole_z,
            q_hist=deque(maxlen=stride_len), td_hist=deque(maxlen=stride_len),
            next_push=rng.uniform(*cfg.randomization.push_interval),
            last_action=np.zeros(self.nu), last_action2=np.zeros(self.nu), qd_sign=np.zeros(self.nu))
        self.curriculum.pending_push = None
        reading = observations.read_sensors(self)
        ep.sensor_buf.extend([reading] * ep.sensor_buf.maxlen)
        ep.action_buf.extend([ep.last_action] * cfg.history)
        ep.target_buf.extend([ep.target] * ep.target_buf.maxlen)
        return observations.compute(self), {}

    def step(self, action):
        cfg, d, ep, cmd, w = self.cfg, self.data, self.episode, self.command, self.cfg.rewards
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        s = self._measure(action, self._actuate(action))
        reward, task, pen, penalty = rewards.compute(self, s)

        impact_bw = s.peak_force.max() / self.robot.weight
        torque_ratio = float((s.peak_tau / cfg.robot.stall_torque).max())
        for key, val in (("torque", torque_ratio), ("speed", s.peak_speed_ratio), ("impact", impact_bw)):
            ep.peaks[key] = max(ep.peaks[key], val)
        # Hardware first: a landing or back-drive that could break the robot ends the episode
        # exactly like a fall.
        broke, fell = terminations.broke(self, s), terminations.fell(self, s)
        terminated = fell or broke
        if terminated:
            reward += w.termination
        truncated = ep.steps >= self._max_steps
        if terminated or truncated:
            curriculums.update_speed(self, fell=terminated)
            if cfg.randomization.enabled and cfg.randomization.physics_curriculum and not cmd.fixed:
                curriculums.score_episode(self, stayed_up=not terminated)
        events.push_robot(self, s, terminated, truncated)
        commands.resample(self)

        ep.last_action2 = ep.last_action
        ep.last_action = action
        ep.last_qvel = s.qd.copy()
        ep.sensor_buf.append(observations.read_sensors(self))
        ep.action_buf.append(action)

        cvx, cvy, cwz = s.cmd
        info = {f"rew/{k}": v for k, v in task.items()}
        info.update({f"pen/{k}": v for k, v in pen.items()})
        info.update({
            "forward_velocity": s.vx, "command": cvx, "x_position": d.qpos[0],
            "lateral_velocity": s.vy, "command_lateral": cvy,
            "yaw_rate": s.ang_world[2], "command_yaw_rate": cwz, "standing": float(cmd.standing),
            "speed_error": s.speed_error, "tracking": s.tracking, "penalty_scale": float(np.exp(-penalty)),
            "speed_max": self.curriculum.speed_max,
            "heading_error_deg": abs(np.degrees(s.yaw_err)), "lateral_offset": abs(s.offset),
            "energy": s.energy, "flight": float(ep.contact.sum() == 0),
            "double_support": float(ep.contact.sum() == 2),
            "peak_force_bw": impact_bw,
            "saturation": s.saturated / (self.frame_skip * self.nu),
            "peak_torque_stall": torque_ratio,
            "max_joint_speed_ratio": s.peak_speed_ratio,
            "broke": float(broke), "fell": float(fell and not broke),
            "action_delta": np.mean(np.abs(action - ep.last_action2)),
            "base_height": self.com_height(),
            "recovering": float(s.recovering),
            "push_level_mean": float(self.curriculum.push_level.mean()),
            "push_level_min": float(self.curriculum.push_level.min()),
            "physics_difficulty": self.curriculum.difficulty,
        })
        return observations.compute(self), reward, terminated, truncated, info

    def _measure(self, action, phys):
        """Advance the clocks and collect what the rewards and terminations use this step."""
        cfg, d, ep, cmd, robot = self.cfg, self.data, self.episode, self.command, self.robot
        bad_contact = self.non_foot_ground_contact()
        ep.t += self.dt
        ep.steps += 1
        cmd.phase += self.dt / commands.gait_period(self)
        cvx, cvy, cwz = cmd.vec
        cmd.heading_target += cwz * self.dt

        R = self.root_rot()
        grav = self.gravity_body()
        # Velocity along the TARGET heading (the start heading, turned only as commanded), not the
        # robot's current facing: walking in circles earns no forward speed. From the distance moved
        # over the whole control step, so no instant can be "timed".
        ch, sh = np.cos(cmd.heading_target), np.sin(cmd.heading_target)
        vel_xy = (d.qpos[0:2] - phys.xy_before) / self.dt
        vx = ch * vel_xy[0] + sh * vel_xy[1]
        vy = -sh * vel_xy[0] + ch * vel_xy[1]
        yaw_err = commands.heading_error(self)
        # Commanded path: the reference moves sideways as commanded and is pulled along the heading
        # to wherever the robot is, so only sideways drift counts.
        ahead, side = np.array([ch, sh]), np.array([-sh, ch])
        cmd.ref_xy = cmd.ref_xy + cvy * self.dt * side
        cmd.ref_xy = cmd.ref_xy + ahead * (ahead @ (d.qpos[0:2] - cmd.ref_xy))
        offset = commands.lateral_offset(self)
        qd = d.qvel[robot.dof_idx]
        qacc = (qd - ep.last_qvel) / self.dt

        prev_contact = ep.contact
        ep.contact = (phys.peak_force > 0.05 * robot.weight).astype(float)
        new_td = (ep.contact == 1) & (prev_contact == 0)
        touchdowns = float(np.sum(new_td))
        # Step timing from touchdowns: the real stride length (for the symmetry terms) and whether
        # left->right and right->left take equally long (a lopsided rhythm is a skip).
        for leg in np.flatnonzero(new_td):
            other = 1 - leg
            if ep.td_step[leg] is not None and 6 <= ep.steps - ep.td_step[leg] <= 40:
                ep.stride_est = ep.steps - ep.td_step[leg]
            if ep.td_step[other] is not None and ep.td_last_leg == other:
                ep.td_intervals.append(ep.steps - ep.td_step[other])
                if len(ep.td_intervals) == 2:
                    i1, i2 = ep.td_intervals
                    ep.timing_asym = abs(i1 - i2) / (i1 + i2)
            elif ep.td_last_leg == leg:
                ep.td_intervals.clear()  # same foot twice: not an alternating pair, restart
            ep.td_step[leg] = ep.steps
            ep.td_last_leg = leg
        # Direction reversals: a joint clearly moving one way (>0.2 rad/s), then clearly the other.
        sign = np.where(np.abs(qd) > 0.2, np.sign(qd), 0.0)
        reversals = float(np.sum((sign != 0) & (ep.qd_sign != 0) & (sign != ep.qd_sign)))
        ep.qd_sign = np.where(sign != 0, sign, ep.qd_sign)
        sole_pos, sole_vel = self.soles()
        ep.clearance = sole_pos[:, 2] - robot.home_sole_z
        leg_phase = (commands.phase(self) + np.array([0.0, 0.5])) % 1.0
        swing = leg_phase < cfg.command.swing_ratio
        if cmd.standing:
            swing = np.zeros(2, dtype=bool)  # both feet down: contact, clearance, mismatch follow
        q = d.qpos[robot.qpos_idx]
        ep.q_hist.append(q.copy())
        ep.td_hist.append(touchdowns)
        return SimpleNamespace(
            action=action, cmd=(cvx, cvy, cwz), R=R, grav=grav, vx=vx, vy=vy, yaw_err=yaw_err, offset=offset,
            ang_world=R @ d.qvel[3:6], qd=qd, qacc=qacc, q=q, touchdowns=touchdowns, reversals=reversals,
            sole_vel=sole_vel, swing=swing, bad_contact=bad_contact,
            feet_y=(R.T @ (d.xpos[robot.foot_bodies[0]] - d.xpos[robot.foot_bodies[1]]))[1],
            peak_force=phys.peak_force, saturated=phys.saturated, overload=phys.overload,
            overspeed=phys.overspeed, peak_tau=phys.peak_tau, peak_speed_ratio=phys.peak_speed_ratio,
            loaded_overspeed=phys.loaded_overspeed, energy=phys.energy, load=phys.load)

    # ------------------------------------------------------------------ curricula, commands
    def curriculum_state(self):
        """Curriculum levels, to carry them over to the next training run (biped_rl/training/stages.py)."""
        return curriculums.state(self)

    def set_curriculum_state(self, state):
        curriculums.set_state(self, state)

    def set_command(self, forward, sideways=0.0, turn_rate=0.0):
        """Change the command mid-episode (keyboard, joystick). All zero = stand still."""
        commands.set_command(self, forward, sideways, turn_rate)

    # ------------------------------------------------------------------ scripted motion (play.py)
    def place_pose(self, joints):
        """Put the robot standing still in joint pose `joints` (absolute angles), soles on the floor."""
        d, robot = self.data, self.robot
        d.qpos[robot.qpos_idx] = np.clip(joints, robot.joint_lo, robot.joint_hi)
        d.qvel[:] = 0.0
        self.episode.target = d.qpos[robot.qpos_idx].copy()
        mujoco.mj_forward(self.model, d)
        self.ground()

    def hold_pose(self, joints):
        """One control step with the servos driven straight to joint angles `joints`: no policy, no
        reward. For scripted motions, e.g. the boot sequence (straight legs -> home crouch); the
        policy's actions only reach +-action_scale around home. Returns whether it is still up."""
        robot, sv = self.robot, self.servo
        self.episode.target = np.clip(joints, robot.joint_lo, robot.joint_hi)
        slack = sv.backlash if sv.backlash.any() else None
        for _ in range(self.frame_skip):
            *_, tau_c = self.servo_torque(self.episode.target, slack, 1.0 / sv.speed)
            self.data.ctrl[:] = tau_c
            mujoco.mj_step(self.model, self.data)
        return not (self.non_foot_ground_contact() or self.gravity_body()[2] > -0.6)

    def start_policy(self, options=None):
        """Hand over to the policy from wherever the robot is now (after a scripted motion), the way
        the real robot will: the start heading is the current one, histories are filled with the
        current readings, the last action and the filter are zero. Returns the observation."""
        cmd = self.command
        R = self.root_rot()
        cmd.heading0 = float(np.arctan2(R[1, 0], R[0, 0]))
        cmd.heading_target = cmd.heading0
        cmd.start_xy = self.data.qpos[0:2].copy()
        cmd.ref_xy = cmd.start_xy.copy()
        return self._start_episode(options or {"stand": True})[0]

    # ------------------------------------------------------------------ render
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, *self._render_size)
            self._camera = mujoco.MjvCamera()
            self._camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self._camera.trackbodyid = self.robot.root
            self._camera.distance = 4 * self.robot.home_height
            self._camera.elevation = -15
            self._camera.azimuth = 120
        self._renderer.update_scene(self.data, camera=self._camera)
        self._draw_track_markers(self._renderer.scene)
        self._draw_push_arrow(self._renderer.scene)
        return self._renderer.render()

    def _draw_push_arrow(self, scene):
        """Red arrow at the torso for 0.4 s after each random shove, pointing the way it was pushed;
        longer arrow = harder shove."""
        ep = self.episode
        if ep.t - ep.last_push > 0.4 or scene.ngeom >= scene.maxgeom:
            return
        vec = np.array([*ep.last_push_vec, 0.0])
        direction = vec / (np.linalg.norm(vec) + 1e-9)
        torso = self.data.subtree_com[self.robot.root].copy()
        length = 0.08 + 0.5 * np.linalg.norm(vec)
        start, end = torso - direction * (0.06 + length), torso - direction * 0.06
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array([1.0, 0.1, 0.1, 1.0], dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.012, start, end)
        scene.ngeom += 1

    def _draw_track_markers(self, scene):
        """The camera follows the robot, so draw the straight start line with a red start line and
        white marks every 0.5 m: progress (and drift) are visible in any video."""
        cmd = self.command
        heading = np.array([np.cos(cmd.heading0), np.sin(cmd.heading0), 0.0])
        side = np.array([-heading[1], heading[0], 0.0])
        rot = np.column_stack([heading, side, [0.0, 0.0, 1.0]]).flatten()
        start = np.array([*cmd.start_xy, 0.001])
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
