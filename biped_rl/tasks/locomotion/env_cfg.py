"""Configuration of the locomotion task: the robot, commands, rewards and randomization.

Every tunable number lives here. A run saves its EnvConfig as runs/<name>/config.json, and
EnvConfig.load reads it back.
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from biped_rl.robots import juan

__all__ = ["RobotConfig", "CommandConfig", "RewardWeights", "RandomizationConfig", "EnvConfig"]


@dataclass
class RobotConfig:
    """Which robot, and its servos. Defaults: Juan (biped_rl/robots/juan.py)."""
    model_path: str = juan.MODEL_PATH       # relative to the project
    foot_bodies: tuple = juan.FOOT_BODIES   # (left, right) body names in the MJCF
    # The servo model (PD control + torque-speed curve) runs in the environment, so the MJCF's
    # actuator type does not matter.
    stall_torque: float = juan.STALL_TORQUE
    no_load_speed: float = juan.NO_LOAD_SPEED
    kp: float = juan.KP
    kd: float = juan.KD
    # Joints used by posture terms and the mirror map (names in the MJCF; missing names are skipped).
    knee_joints: tuple = juan.KNEE_JOINTS
    hip_roll_joints: tuple = juan.HIP_ROLL_JOINTS
    mirror_pairs: tuple = juan.MIRROR_PAIRS


@dataclass
class CommandConfig:
    """Forward, sideways and turn-rate commands plus "stand", re-drawn every few seconds. Walking many
    ways is what lets a policy step out of trouble, and it is what joystick steering needs."""
    lateral_range: tuple = (-0.15, 0.15)    # sideways speed [m/s], + = left
    yaw_rate_range: tuple = (-0.6, 0.6)     # turn rate [rad/s], + = left (~35 deg/s)
    backward_speed: float = 0.15            # forward speeds are drawn from [-backward_speed, top]
    stand_prob: float = 0.1                 # share of commands that are "stand still, feet down"
    # Shares of the walking commands that do one thing only, so each is practised on its own (mixed
    # in with the rest, turning in place and walking backward are rare and come out weak):
    turn_in_place_prob: float = 0.2         # only turn
    backward_prob: float = 0.1              # only walk backward, at [0, backward_speed]
    sideways_prob: float = 0.1              # only walk sideways, within lateral_range
    straight_prob: float = 0.3              # of the remaining commands: no sideways or turn part
    resample_interval: tuple = (2.0, 8.0)   # s between new commands within an episode
    # Forward speed: the top of the range grows (speed curriculum) while the robot tracks its top
    # commands through whole, safe episodes, up to speed_limit: the sim finds the real top speed.
    speed_range: tuple = (0.0, 0.4)         # starting range [m/s]
    speed_limit: float = 1.0                # curriculum ceiling [m/s]
    curriculum_step: float = 0.05           # m/s per promotion / demotion
    # Gait clock: the rhythm speeds up with the command (a fixed rhythm caps the top speed at
    # longest step / step time).
    gait_period: float = 0.8                # s per full left+right cycle, standing or slow
    gait_period_fast: float = 0.45          # s per cycle at fast_speed and above
    fast_speed: float = 0.8                 # m/s
    swing_ratio: float = 0.45               # fraction of the cycle each foot is in the air (< 0.5: double support)
    swing_height: float = 0.025             # target sole clearance during swing [m]


@dataclass
class RewardWeights:
    """reward = (alive + tracking * sum(task terms)) * exp(-sum(penalties)) [+ termination]

    * tracking = how well the speed command is met (1 = exact). Every task term is multiplied by it,
      so heading, rhythm and foot lift only pay while moving at the commanded speed: standing still
      at a 0.3 m/s command earns ~0.
    * Penalties scale the reward down instead of being subtracted: the reward is never negative
      (falling early never pays) and penalties still bite when things go badly.
    Each term is a function of the same name in mdp/rewards.py.
    """
    # --- task terms (x tracking) ---
    tracking_vel: float = 3.0               # the tracking factor itself
    # tracking = (1-share) * exp(-err^2 / sigma2) + share * exp(-err^2 / broad_sigma2), err in m/s. The
    # broad part makes creeping beat standing (needed to learn to walk at all); once it walks, a
    # forgiving kernel lets it trade speed for small penalty savings, so it is sharpened.
    tracking_sigma2: float = 0.008
    tracking_broad_sigma2: float = 0.08
    tracking_broad_share: float = 0.15
    tracking_lat_yaw: float = 0.5           # sideways speed and turn rate as commanded
    heading: float = 1.0                    # facing the target heading
    gait_contact: float = 1.0               # feet on the ground when the gait clock says stance
    feet_clearance: float = 0.3             # swing sole reaches swing_height (no shuffling, no toe drag)
    # --- survival ---
    alive: float = 0.2                      # small: survival alone must not be worth standing still for
    termination: float = -10.0              # once, on a fall or a hardware limit
    # --- penalties (positive; reward *= exp(-sum)) ---
    # following the command
    path_offset: float = 2.0                # distance^2 from the commanded path [m^2] (0.02 at 10 cm)
    stand_still: float = 1.0                # moving < 50% of a walking command (0 at half speed, 1 stopped)
    stand_pose: float = 20.0                # commanded to stand: joints away from home, mean [rad^2]
                                            # (7 deg off costs ~0.26, 2 deg of balancing ~0.02)
    # posture
    orientation: float = 2.0                # torso tilt (roll, pitch)
    ang_vel_xy: float = 0.05                # torso wobble
    lin_vel_z: float = 0.5                  # torso bouncing
    base_height: float = 50.0               # centre of mass below its home height [m^2] (no crouch-walking)
    knee_hyperextension: float = 2.0        # knee bent backwards [rad] (no bird-legged gait)
    hip_roll: float = 1.0                   # hip roll away from 0 [rad^2] (no splayed, waddling legs)
    hip_roll_inward: float = 5.0            # per rad of inward roll: the 5 deg inward joint limit is a margin,
                                            # not a target (5 deg inward costs ~0.44)
    # smooth motion
    action_rate: float = 0.5                # 1st difference of the actions
    action_acc: float = 0.2                 # 2nd difference of the actions (jerk)
    joint_acc: float = 1e-6                 # joint accelerations [rad/s^2]
    # servo protection
    servo_load: float = 0.5                 # (tau / stall)^2: heat and gear wear
    servo_saturation: float = 1.0           # time spent on the torque-speed limit
    servo_overload: float = 30.0            # ((|tau| - 0.8 stall) / stall)^2 per joint, above 80% of stall
    backdrive: float = 5.0                  # joint forced faster than the servo's no-load speed (landings
                                            # spinning the gearbox backwards: how hobby-servo gears break)
    joint_limits: float = 5.0               # approaching the mechanical end stops
    energy: float = 0.01                    # mechanical power [W]
    # feet
    impact: float = 3.0                     # landing force above impact_threshold
    impact_threshold: float = 1.5           # x body weight
    foot_slip: float = 2.0                  # sole sliding while on the floor
    feet_distance: float = 5.0              # feet closer than min_feet_distance (legs colliding)
    min_feet_distance: float = 0.08         # m, sideways in the torso frame (home 0.118; they touch at 0.05)
    # gait shape and wear (penalties count events: many small motions are not free)
    symmetry: float = 3.0                   # each leg now vs the other half a stride ago, mirrored [rad^2]
    gait_mismatch: float = 0.5              # share of feet on/off the ground against the gait clock
    footstep: float = 1.0                   # steps beyond the gait clock's: fewer steps, fewer impacts
    reversals: float = 0.1                  # per servo direction reversal (backlash hammering, pot wear)
    # --- hardware limits: crossing one ends the episode like a fall (the robot "broke") ---
    break_impact: float = 3.0               # landing above this x body weight
    break_joint_speed: float = 1.3          # joint forced above this x no-load speed...
    break_requires_load: float = 0.5        # ...while the servo pushes back with more than this x stall
                                            # torque (a limb coasting with the servo barely engaged does
                                            # not strip gears)
    # The speed curriculum only promotes after an episode that stayed under all of these:
    safe_torque: float = 0.8                # x stall torque
    safe_joint_speed: float = 1.0           # x no-load speed
    safe_impact: float = 2.0                # x body weight


@dataclass
class RandomizationConfig:
    """What differs from episode to episode (enabled=False: the nominal robot, no noise, no pushes)."""
    enabled: bool = True
    # --- world ---
    friction: tuple = (0.5, 1.2)            # rubber soles: ~0.6-1.0 indoors; lower for dusty or polished floors
    floor_tilt_deg: float = 3.0             # slope, as tilted gravity
    # --- body. Each part x mass_scale, then the whole robot x total_mass_scale, then the torso (battery,
    # Pi, wiring: a heavy, high torso is the hard case) x torso_mass_scale. The CAD model is 1.75 kg;
    # this spans ~1.7-2.25 kg.
    mass_scale: tuple = (0.9, 1.1)          # per part (and its inertia)
    total_mass_scale: tuple = (1.0, 1.15)   # whole robot
    torso_mass_scale: tuple = (1.0, 1.25)   # torso only
    com_offset: float = 0.015               # torso centre-of-mass shift [m]
    # --- servos. Torque and speed scale with battery voltage (rated at 11.1 V; a 3S pack goes from
    # 12.6 V full, x1.13, to ~10.5 V near empty, x0.95, and the lower bound leaves room for servo
    # variance and heat): one draw scales both, +-5% per servo on top.
    servo_strength: tuple = (0.85, 1.13)
    kp_scale: tuple = (0.7, 1.3)
    kd_scale: tuple = (0.7, 1.3)
    armature_scale: tuple = (0.5, 1.5)
    frictionloss_scale: tuple = (0.5, 2.0)
    backlash: tuple = (0.0, 0.017)          # rad (0-1 deg) of gear slack per joint: no restoring torque inside it
    joint_resolution: float = 0.0042        # rad: the HTD-35H reports 0-1000 counts over 240 deg
    joint_offset: float = 0.03              # rad: per-servo zero calibration error (+-1.7 deg)
    # --- delays. A new target applies `latency` into a control step, then `action_delay_steps` whole
    # 20 ms steps later: 25-59 ms in all (the servo starts moving 24-28 ms after a command, and reading
    # 4 servos takes ~15 ms, all 8 on the one bus about twice that). The real robot measures 70-90 ms
    # (EnvConfig.use_measured_servos).
    latency: tuple = (0.005, 0.019)         # s
    action_delay_steps: tuple = (1, 2)
    obs_delay_steps: tuple = (0, 1)         # IMU and joint readings are this many steps old
    # --- IMU. Errors that stay the same through an episode (accelerometer offset and mounting, gyro
    # zero-rate offset) matter more than its white noise.
    imu_tilt_bias_deg: float = 2.5          # +- per axis (roll, pitch) of the gravity reading
    gyro_bias: float = 0.02                 # rad/s, +- per axis
    # Servo buzz shaking the torso (on the robot: 0.5-2.7 rad/s rms on the gyro while standing, 15-20 Hz,
    # mostly pitch): rms on pitch (roll and yaw 0.4x) and frequency drawn per episode.
    gyro_vibration: tuple = (0.0, 1.2)      # rad/s rms
    gyro_vibration_hz: tuple = (14.0, 20.0)
    # The robot low-passes the gyro the policy sees (robot/imu.py, GYRO_LOWPASS_HZ), and so does the
    # simulator, also on the nominal robot. 0 = off.
    gyro_lowpass_hz: float = 4.0
    # --- sensor noise, every step ---
    noise_gravity: float = 0.03
    noise_gyro: float = 0.1                 # rad/s
    noise_yaw: float = 0.03                 # rad, IMU heading
    noise_joint_pos: float = 0.01           # rad
    noise_joint_vel: float = 0.5            # rad/s
    # --- starting state (as in Berkeley Humanoid Lite): a perturbed crouch, already moving, so
    # recovering from odd states is practised from the first step. Joint angles = home x a per-joint
    # scale (hip roll's home is 0: it keeps the small additive noise only).
    reset_joint_scale: tuple = (0.7, 1.3)
    reset_lin_vel: float = 0.2              # m/s, +- per horizontal axis
    reset_ang_vel: float = 0.5              # rad/s, +- per axis
    # --- shoves: a velocity kick to the torso every push_interval ---
    push_interval: tuple = (2.0, 4.0)       # s
    push_velocity: float = 0.4              # m/s: the cap with the push curriculum; without it, the range
                                            # per axis (the evaluation's standard test: 0.3)
    # Push curriculum (Kumar's "region of attraction"): 8 directions relative to the robot, each with its
    # own strength. A direction gets push_step stronger once the robot stays up (no fall within 2 s)
    # through push_success of push_window shoves from it, weaker otherwise; each shove is 0.75-1.1x its
    # direction's strength. The policy trains at the edge of what it can take from each side.
    push_curriculum: bool = True
    push_start: float = 0.05                # m/s, every direction's first strength
    push_step: float = 0.02                 # m/s
    push_window: int = 10
    push_success: float = 0.9
    # Physics curriculum: the hardest realism (extra delay steps, extra mass, gear slack, servo zero error,
    # IMU tilt and gyro bias, gyro buzz) is scaled by a difficulty level from 0 (none of it) to 1 (the
    # full ranges above). Each training env raises it by difficulty_step after difficulty_success of its
    # last difficulty_window episodes stayed up, and lowers it otherwise. Switching all of it on at once
    # made the policy fall in 60-95% of episodes and unlearn the walk it had.
    physics_curriculum: bool = True
    difficulty_start: float = 0.0
    difficulty_step: float = 0.05
    difficulty_window: int = 20
    difficulty_success: float = 0.8


# Settings that saved run configs may still carry, all at the value the current code behaves as;
# dropped when loading.
REMOVED = {"mirror_symmetry", "robot.joint_feedback", "command.mode", "rewards.flight", "rewards.step_timing",
           "rewards.posture_symmetry", "randomization.voltage_coupled", "randomization.servo_speed"}


@dataclass
class EnvConfig:
    robot: RobotConfig = field(default_factory=RobotConfig)
    command: CommandConfig = field(default_factory=CommandConfig)
    rewards: RewardWeights = field(default_factory=RewardWeights)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    control_hz: float = 50.0     # policy rate
    action_scale: float = 0.5    # rad of servo target offset from the home pose per unit action
    action_filter: float = 0.5   # EMA on the actions (1 = off); the robot uses the same filter
    history: int = 3             # past sensor frames stacked into the observation
    episode_seconds: float = 20.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        nested = {"robot": RobotConfig, "command": CommandConfig,
                  "rewards": RewardWeights, "randomization": RandomizationConfig}

        def tup(v):  # JSON turns tuples into lists
            return tuple(map(tup, v)) if isinstance(v, list) else v

        d = {k: v for k, v in d.items() if k not in REMOVED}
        for k in nested:
            d[k] = {kk: v for kk, v in d.get(k, {}).items() if f"{k}.{kk}" not in REMOVED}
        return cls(**{k: nested[k](**{kk: tup(v) for kk, v in d[k].items()}) if k in nested else v
                      for k, v in d.items()})

    @classmethod
    def load(cls, run):
        """A run's saved configuration (<run>/config.json)."""
        return cls.from_dict(json.loads((Path(run) / "config.json").read_text()))

    def use_measured_servos(self):
        """The servo model fitted from the real robot (biped_rl/robots/juan.py), to test how a policy
        will do on it."""
        self.robot.kp = juan.MEASURED_KP
        self.randomization.action_delay_steps = juan.MEASURED_DELAY_STEPS
        return self
