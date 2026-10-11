"""Every tunable number of the robot, task, reward and randomization in one place.

When you bring in your own model, RobotConfig is usually the only part to edit.
"""
from dataclasses import asdict, dataclass, field
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "models"


@dataclass
class RobotConfig:
    model_path: str = str(MODELS / "simple_biped.xml")
    foot_bodies: tuple = ("l_foot", "r_foot")  # (left, right) body names in the MJCF
    # Hiwonder HTD-35H bus servo at 11.1 V. Simulated in the env (PD + torque-speed
    # curve), so the MJCF actuator type does not matter.
    stall_torque: float = 3.43   # Nm (35 kg·cm * 0.0981)
    no_load_speed: float = 5.8   # rad/s (0.18 s/60deg); torque falls linearly to 0 at this speed
    kp: float = 15.0             # internal stiffness, Nm/rad (not published: estimate, full torque at ~13deg)
    kd: float = 0.2              # Nm·s/rad (estimate)
    # HTD-35H reports position (not velocity: the robot computes it by differencing).
    # Set False for servos without feedback.
    joint_feedback: bool = True
    # Joints used by posture terms (names in the MJCF; missing names are skipped).
    # Knee sign: positive = shin swings back (human-like); negative = hyperextension.
    knee_joints: tuple = ("l_knee", "r_knee", "left_knee", "right_knee")
    hip_roll_joints: tuple = ("l_hip_roll", "r_hip_roll", "left_hip_roll", "right_hip_roll")
    # (left joint, right joint, sign) pairs for the symmetry term: sign -1 where mirroring the
    # motion flips the angle (both hip-roll axes point forward, so outward is + left / - right).
    mirror_pairs: tuple = (("l_hip_roll", "r_hip_roll", -1), ("l_hip_pitch", "r_hip_pitch", 1),
                           ("l_knee", "r_knee", 1), ("l_ankle", "r_ankle", 1),
                           ("left_hip_roll", "right_hip_roll", -1), ("left_hip_pitch", "right_hip_pitch", 1),
                           ("left_knee", "right_knee", 1), ("left_ankle", "right_ankle", 1))


@dataclass
class CommandConfig:
    # Forward speed command, sampled per episode. The upper end grows automatically
    # (speed curriculum) while the robot tracks its top commands without falling,
    # up to speed_limit: the sim finds the real top speed instead of a guess.
    speed_range: tuple = (0.0, 0.4)  # starting range [m/s]
    speed_limit: float = 1.0         # curriculum ceiling [m/s]; the hardware gate decides how far below it to stay
    curriculum_step: float = 0.05    # m/s per promotion / demotion
    # Gait rhythm speeds up with the command: a fixed rhythm caps top speed at
    # (max step length / step time).
    gait_period: float = 0.8         # s per full left+right cycle when standing / slow
    gait_period_fast: float = 0.45   # s per cycle at fast_speed and above
    fast_speed: float = 0.8          # m/s
    # Fraction of the cycle each foot is in the air. < 0.5: walking (double support),
    # > 0.5: running (flight phase of (swing_ratio - 0.5) * period per step).
    swing_ratio: float = 0.45
    swing_height: float = 0.025      # target sole clearance during swing [m]


@dataclass
class RewardWeights:
    """reward = (alive + tracking * sum(task terms)) * exp(-sum(penalties)) [+ termination]

    * tracking = how well the speed command is met (1 = exact). Every task term is
      multiplied by it, so heading, rhythm and foot lift only pay while actually
      moving at the commanded speed: standing still at a 0.3 m/s command earns ~0.
    * Penalties scale the reward down instead of being subtracted: the reward is never
      negative (falling early never pays) and penalties still bite when things go badly.
    """
    # --- 1. straight-line speed (task terms, gated by tracking) ---
    tracking_vel: float = 2.0        # the tracking factor itself, as a reward
    # tracking = (1-share) * exp(-err^2 / sigma2) + share * exp(-err^2 / broad_sigma2), err in m/s.
    # The broad part rewards creeping over standing (needed to learn to walk at all); once it walks,
    # a forgiving kernel lets it trade speed away for small penalty savings, so sharpen it.
    tracking_sigma2: float = 0.02
    tracking_broad_sigma2: float = 0.08
    tracking_broad_share: float = 0.5
    tracking_lat_yaw: float = 0.5    # no sideways drift, no turning
    heading: float = 1.0             # keep facing the start direction
    # --- 4. gait shape (task terms, gated by tracking) ---
    gait_contact: float = 1.5        # feet on the ground when the gait clock says stance
    feet_clearance: float = 0.3      # swing sole reaches swing_height (no shuffling / toe drag)
    flight: float = 0.0              # bonus when both feet are off the ground while moving
    # --- 2. don't fall ---
    alive: float = 0.2               # small: survival alone must not be worth standing still for
    termination: float = -10.0       # one-off penalty on falling
    # --- penalties (positive numbers; summed, then reward *= exp(-sum)) ---
    path_offset: float = 2.0         # distance^2 from the straight start line [m^2] (0.02 at 10 cm)
    stand_still: float = 1.0         # moving < 50% of a walking command (0 at half speed, 1 when stopped)
    symmetry: float = 10.0           # left leg now vs right leg half a stride ago (mirrored) [rad^2]
    step_timing: float = 1.5         # every step: latest |left->right - right->left interval| / their sum
                                     # (an even da-da-da-da rhythm costs 0; a lopsided skip is charged
                                     # continuously; charged only at touchdowns it was too weak to matter)
    posture_symmetry: float = 15.0   # left vs right leg AVERAGE angle over the last stride (mirrored)
                                     # [rad^2]: a leg held straighter/bent all the time is a limp too
    # --- wear per cycle (penalties count events, so many small gentle motions are not free) ---
    gait_mismatch: float = 1.0       # fraction of feet on/off the ground against the gait clock
    footstep: float = 1.0            # per touchdown: fewer, rhythm-paced steps = fewer impact cycles
    reversals: float = 0.1           # per servo direction reversal (gear backlash hammering, pot wear)
    orientation: float = 2.0         # torso tilt (roll/pitch)
    ang_vel_xy: float = 0.05         # torso wobble
    lin_vel_z: float = 0.5           # torso bouncing
    base_height: float = 50.0        # pelvis below its home height [m^2] (no crouch-walking)
    knee_hyperextension: float = 2.0 # knee bent backwards [rad] (no bird-legged gait)
    hip_roll: float = 1.0            # hip roll away from 0 [rad^2] (no splayed, waddling legs)
    hip_roll_inward: float = 5.0     # per rad of inward roll: 0 deg inward is the goal, the 5 deg
                                     # joint limit is only a margin (5 deg inward costs ~0.44)
    # --- 3. smooth, hardware-safe motion (penalties) ---
    action_rate: float = 0.5         # 1st difference of actions
    action_acc: float = 0.2          # 2nd difference of actions (jerk)
    joint_acc: float = 1e-6          # joint accelerations [rad/s^2]
    servo_load: float = 0.5          # (tau / stall)^2: heat and gear wear
    servo_saturation: float = 1.0    # time spent on the torque-speed limit
    servo_overload: float = 30.0     # torque above 80% of stall, ((|tau| - 0.8 stall) / stall)^2 per joint
    backdrive: float = 5.0           # joint forced faster than the servo's no-load speed (impacts
                                     # spinning the gearbox backwards: how hobby-servo gears break)
    impact: float = 3.0              # foot landing force above impact_threshold * weight
    foot_slip: float = 2.0           # sole sliding while on / within 1 cm of the floor
    joint_limits: float = 5.0        # approaching the mechanical end stops
    feet_distance: float = 5.0       # feet too close (legs colliding)
    energy: float = 0.01             # mechanical power [W]

    impact_threshold: float = 1.5    # x body weight

    # --- hardware limits: crossing one ends the episode like a fall (the robot "broke") ---
    break_impact: float = 3.0        # foot landing above this x body weight
    break_joint_speed: float = 1.3   # joint forced above this x servo no-load speed (gearbox back-driven)
    # The speed curriculum only promotes after an episode that stayed inside all of these:
    safe_torque: float = 0.8         # x stall torque
    safe_joint_speed: float = 1.0    # x no-load speed
    safe_impact: float = 2.0         # x body weight
    min_feet_distance: float = 0.08  # m, lateral, in torso frame (nominal 0.118; feet touch at 0.05)


@dataclass
class RandomizationConfig:
    enabled: bool = True
    friction: tuple = (0.5, 1.2)            # rubber soles: ~0.6-1.0 indoors, lower bound for dusty/polished floors
    mass_scale: tuple = (0.9, 1.15)         # per link (and inertia); BoM: 1.75 kg, up to ~2 kg with stronger prints
    com_offset: float = 0.01                # torso CoM shift, m
    # Servo torque and speed scale with voltage, rated at 11.1 V: 3S LiPo full 12.6 V (x1.13)
    # to near-empty ~10.5 V (x0.95), lower bound extra for servo variance and heat.
    servo_strength: tuple = (0.85, 1.13)
    servo_speed: tuple = (0.85, 1.13)
    kp_scale: tuple = (0.7, 1.3)
    kd_scale: tuple = (0.7, 1.3)
    armature_scale: tuple = (0.5, 1.5)
    frictionloss_scale: tuple = (0.5, 2.0)
    latency: tuple = (0.005, 0.019)         # s, sensor read -> servo moving (UART bus round trips)
    floor_tilt_deg: float = 3.0             # simulated slope via tilted gravity
    push_interval: tuple = (2.0, 4.0)       # s between random shoves
    push_velocity: float = 0.3              # m/s added to the torso
    noise_gravity: float = 0.03
    noise_gyro: float = 0.1                 # rad/s
    noise_yaw: float = 0.03                 # rad, IMU heading
    noise_joint_pos: float = 0.01           # rad
    noise_joint_vel: float = 0.5            # rad/s


@dataclass
class EnvConfig:
    robot: RobotConfig = field(default_factory=RobotConfig)
    command: CommandConfig = field(default_factory=CommandConfig)
    rewards: RewardWeights = field(default_factory=RewardWeights)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    control_hz: float = 50.0     # same as standard servo PWM rate
    action_scale: float = 0.5    # rad of servo target offset from the home pose per unit action
    action_filter: float = 0.5   # EMA on actions (1 = off). Use the same filter on the robot!
    history: int = 3             # past sensor frames stacked into the observation
    episode_seconds: float = 20.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        nested = {"robot": RobotConfig, "command": CommandConfig,
                  "rewards": RewardWeights, "randomization": RandomizationConfig}

        def build(sub_cls, values):  # JSON turns tuples into lists
            return sub_cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in values.items()})

        return cls(**{k: build(nested[k], v) if k in nested else v for k, v in d.items()})
