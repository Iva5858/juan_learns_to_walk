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
      physics   randomized physics, with the v7 realism (delays, heavy torso, slack, IMU bias...)
                phased in by the physics curriculum; no pushes; forgiving speed tracking
      pushes    randomized physics at full difficulty + the per-direction push curriculum
    Each stage is a separate run that --resume's the previous one (see train_v7.sh)."""
    rnd, cmd, w = cfg.randomization, cfg.command, cfg.rewards
    rnd.enabled = stage in ("physics", "pushes")
    # Speed-tracking kernel: forgiving while learning to walk (creeping must beat standing, or the
    # policy settles for standing still), then sharp (juan_v3's) once it walks, so it can't trade
    # speed away for small penalty savings.
    learning = stage in ("walk", "commands", "physics")
    w.tracking_vel = 2.0 if learning else RewardWeights().tracking_vel
    w.tracking_sigma2 = 0.02 if learning else RewardWeights().tracking_sigma2
    w.tracking_broad_share = 0.5 if learning else RewardWeights().tracking_broad_share
    # Stage 1 is the task the forward-only runs learned to walk on: one forward speed per episode,
    # no stand, no backward, no sideways/turning. With those mixed in from the start (~30% of
    # commands near zero, a new command every few seconds), standing still won.
    default = CommandConfig()
    walk = stage == "walk"
    cmd.straight_prob = 1.0 if walk else default.straight_prob
    cmd.turn_in_place_prob = 0.0 if walk else default.turn_in_place_prob
    cmd.backward_prob = 0.0 if walk else default.backward_prob
    cmd.sideways_prob = 0.0 if walk else default.sideways_prob
    cmd.lateral_range = default.lateral_range
    cmd.stand_prob = 0.0 if walk else default.stand_prob
    cmd.backward_speed = 0.0 if walk else default.backward_speed
    cmd.resample_interval = (1e9, 1e9) if walk else default.resample_interval
    rnd.push_velocity = RandomizationConfig().push_velocity if stage == "pushes" else 0.0
    rnd.push_curriculum = stage == "pushes"
    rnd.physics_curriculum = stage == "physics"  # stage 4 trains on the full-difficulty robot
    # Varied starting states once physics is randomized (also switches them on when resuming a run
    # saved before they existed).
    default_rnd = RandomizationConfig()
    varied = stage in ("physics", "pushes")
    rnd.reset_joint_scale = default_rnd.reset_joint_scale if varied else (1.0, 1.0)
    rnd.reset_lin_vel = default_rnd.reset_lin_vel if varied else 0.0
    rnd.reset_ang_vel = default_rnd.reset_ang_vel if varied else 0.0


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
