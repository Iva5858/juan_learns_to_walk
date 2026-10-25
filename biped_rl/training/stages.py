"""Training stages, and carrying curriculum levels from one run to the next."""
import json

import numpy as np

from biped_rl.tasks.locomotion.env_cfg import CommandConfig, RandomizationConfig, RewardWeights

STAGES = ("walk", "commands", "physics", "pushes")


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


# Curriculum levels live in the env processes and would start over with every new run; they are
# saved at the end of a run and restored when another run resumes from it. (Through env methods:
# VecEnv.set_attr only reaches the Monitor wrapper, not the environment inside it.)
def save_curriculum(env, run_dir):
    states = env.env_method("curriculum_state")
    state = {k: np.mean([st[k] for st in states], axis=0).tolist() for k in states[0]}
    (run_dir / "curriculum.json").write_text(json.dumps(state, indent=2))


def load_curriculum(env, run_dir):
    path = run_dir / "curriculum.json"
    if path.exists():
        state = json.loads(path.read_text())
        env.env_method("set_curriculum_state", state)
        print(f"curriculum from {path}: {state}", flush=True)
