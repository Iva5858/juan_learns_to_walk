"""Training stages, and carrying curriculum levels from one run to the next."""
import json

import numpy as np

from biped_rl.tasks.locomotion.env_cfg import CommandConfig, RandomizationConfig, RewardWeights

STAGES = ("walk", "commands", "physics", "pushes", "speed", "speed_physics")

# The free run's action range: every joint can reach both of its end stops from straight legs (deg, the larger
# side of each joint's range in juan/rl_robot.xml: hip roll -26..19, hip pitch -112..104, knee -106..110,
# ankle -78..79); targets past a stop are clipped to it.
FULL_RANGE_DEG = (26.0, 112.0, 110.0, 79.0) * 2
# Weights the free run keeps: speed, survival, going straight, and the robot's physical limits (slipping feet
# don't carry over to the real floor; overload, back-drive, hard landings and end stops break servos). All
# other terms are style or wear: off, for a later fine-tune.
FREE_RUN_WEIGHTS = dict(net_speed=10.0, net_speed_slip=1.0, fall_refund_s=1.0, free_air_time=20.0, alive=0.05, path_offset=2.0, foot_slip=10.0,
                        impact=3.0, servo_overload=30.0, backdrive=5.0, joint_limits=5.0, feet_distance=5.0)
FREE_RUN_OFF = ("tracking_vel", "tracking_lat_yaw", "heading", "gait_contact", "feet_clearance", "step_length",
                "feet_air_time", "swing_lift", "swing_reach", "stand_still", "stand_still_moves", "stand_pose",
                "stand_feet", "orientation", "ang_vel_xy", "lin_vel_z", "base_height", "knee_hyperextension",
                "hip_roll", "hip_roll_inward", "action_rate", "action_acc", "joint_acc", "servo_load",
                "servo_saturation", "energy", "symmetry", "gait_mismatch", "footstep", "reversals", "double_support")


def apply_stage(cfg, stage):
    """Staged training (Li et al. 2024, Cassie): learn the skill first, then the variety of tasks,
    then robustness. Pushes come last: frequent pushes early made Cassie's normal walking worse.
      walk      nominal physics, no pushes; one forward speed (0 to top) per episode; forgiving
                speed tracking
      commands  nominal physics, no pushes; all commands (backward, stand, sideways, turning, changed
                every few seconds); forgiving speed tracking
      physics   randomized physics, with the hardest parts (delays, heavy torso, slack, IMU bias...)
                phased in by the physics curriculum; no pushes; forgiving speed tracking
      pushes    randomized physics at full difficulty + the per-direction push curriculum
    Each stage is a separate run that resumes the previous one (docs/training.md). Settings the stage
    does not mention keep their value."""
    if stage in ("speed", "speed_physics"):
        return apply_speed_stage(cfg, stage)
    rnd, cmd, w = cfg.randomization, cfg.command, cfg.rewards
    rnd0, cmd0, w0 = RandomizationConfig(), CommandConfig(), RewardWeights()
    walk = stage == "walk"
    learning = stage in ("walk", "commands", "physics")
    randomized = stage in ("physics", "pushes")
    rnd.enabled = randomized
    # Speed-tracking kernel: forgiving while learning to walk (creeping must beat standing, or the
    # policy settles for standing still), then sharp once it walks, so it can't trade speed away for
    # small penalty savings.
    w.tracking_vel = 2.0 if learning else w0.tracking_vel
    w.tracking_sigma2 = 0.02 if learning else w0.tracking_sigma2
    w.tracking_broad_share = 0.5 if learning else w0.tracking_broad_share
    # "walk": one forward speed per episode, nothing else. With all the commands mixed in from the start
    # (~30% of them near zero, a new one every few seconds), standing still won.
    cmd.straight_prob = 1.0 if walk else cmd0.straight_prob
    cmd.turn_in_place_prob = 0.0 if walk else cmd0.turn_in_place_prob
    cmd.backward_prob = 0.0 if walk else cmd0.backward_prob
    cmd.sideways_prob = 0.0 if walk else cmd0.sideways_prob
    cmd.lateral_range = cmd0.lateral_range
    cmd.stand_prob = 0.0 if walk else cmd0.stand_prob
    cmd.backward_speed = 0.0 if walk else cmd0.backward_speed
    cmd.resample_interval = (1e9, 1e9) if walk else cmd0.resample_interval
    rnd.push_velocity = rnd0.push_velocity if stage == "pushes" else 0.0
    rnd.push_curriculum = stage == "pushes"
    rnd.physics_curriculum = stage == "physics"  # "pushes" trains on the full-difficulty robot
    # Varied starting states once the physics is randomized.
    rnd.reset_joint_scale = rnd0.reset_joint_scale if randomized else (1.0, 1.0)
    rnd.reset_lin_vel = rnd0.reset_lin_vel if randomized else 0.0
    rnd.reset_ang_vel = rnd0.reset_ang_vel if randomized else 0.0



def apply_speed_stage(cfg, stage):
    """The free run: as fast as it can, from a fresh network, with nothing that suggests a gait.
      speed          legs straight at the start and as the zero of the actions, the full joint range,
                     one "go" command, no gait clock; reward = net speed (+ lifted steps that carry it); only
                     the robot's physical limits cost (the hardware stops stay); nominal physics
      speed_physics  the same on the randomized robot, phased in by the physics curriculum (resumes speed)
    Learned from earlier tries (docs/experiments-2026-11.md): v8/v9 stood still (survival paid, every task
    term gated by speed tracking): survival is small and speed is paid directly; v8j shuffled at ~10 steps/s:
    foot slip x5 and lifted-step air time; v9 marched in place: air time pays only with net progress; v9's
    first step jumped to another pose (4x body-weight landings): the start pose is the actions' zero; v10
    counted lunges and stacked difficulties: net speed over 1 s, a fall takes back the last second's speed reward
    (a dive pays nothing), v7's proven servo model, one stage at a time. Start with train.py --init-std-rad 0.05:
    0.1 action units over the full range (0.2 rad at the hip) ended every episode within 1 s."""
    rnd, cmd, w = cfg.randomization, cfg.command, cfg.rewards
    rnd0 = RandomizationConfig()
    cfg.neutral_pose = "straight"
    cfg.action_scale = tuple(round(float(np.radians(d)), 4) for d in FULL_RANGE_DEG)
    cmd.go_command, cmd.clock = (1.0, 0.0, 0.0), False
    for name in FREE_RUN_OFF:
        setattr(w, name, 0.0)
    for name, value in FREE_RUN_WEIGHTS.items():
        setattr(w, name, value)
    physics = stage == "speed_physics"
    rnd.enabled = rnd.physics_curriculum = physics
    rnd.push_velocity, rnd.push_curriculum = 0.0, False
    rnd.start_pool = None
    rnd.reset_joint_scale = rnd0.reset_joint_scale if physics else (1.0, 1.0)
    rnd.reset_lin_vel = rnd0.reset_lin_vel if physics else 0.0
    rnd.reset_ang_vel = rnd0.reset_ang_vel if physics else 0.0


# Curriculum levels live in the env processes and would start over with every new run; they are
# saved at the end of a run and restored when another run resumes from it. (Through env methods:
# VecEnv.set_attr only reaches the Monitor wrapper, not the environment inside it.)
def save_curriculum(env, run_dir, states=None):
    # A cached snapshot is safe even if Ctrl+C interrupted a pending VecEnv step:
    # asking the same pipes for curriculum state then can read stale step replies.
    if states is None:
        states = env.env_method("curriculum_state")
    state = {k: np.mean([st[k] for st in states], axis=0).tolist() for k in states[0]}
    (run_dir / "curriculum.json").write_text(json.dumps(state, indent=2))


def load_curriculum(env, run_dir):
    path = run_dir / "curriculum.json"
    if path.exists():
        state = json.loads(path.read_text())
        cfg = run_dir / "config.json"
        if cfg.exists() and not json.loads(cfg.read_text())["randomization"].get("physics_curriculum", False):
            state["difficulty"] = None  # it never had the curriculum: start the physics stage easy, not at its 1.0
        env.env_method("set_curriculum_state", state)
        restored = env.env_method("curriculum_state")[0]
        print(f"curriculum from {path}: {restored}", flush=True)
