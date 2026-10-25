"""Curricula: levels that adapt to how well the robot copes. Each training environment keeps its
own; biped_rl/training/stages.py carries them over between runs (curriculum.json).

* speed: the top of the forward-speed command grows while the top commands are tracked through
  whole, safe episodes (CommandConfig.speed_range .. speed_limit).
* physics difficulty: scales the hardest randomization, 0 -> 1 (RandomizationConfig.physics_curriculum).
* push strength: per push direction relative to the robot (RandomizationConfig.push_curriculum).
"""
import numpy as np


def state(env):
    cur = env.curriculum
    return {"speed_max": float(cur.speed_max), "difficulty": float(cur.difficulty),
            "push_level": cur.push_level.tolist()}


def set_state(env, state):
    cur = env.curriculum
    cur.speed_max = float(state["speed_max"])
    cur.push_level = np.array(state["push_level"], dtype=float)
    if env.cfg.randomization.physics_curriculum:  # otherwise the robot stays at full difficulty
        cur.difficulty = float(state["difficulty"])


def episode_was_safe(env):
    w, peak = env.cfg.rewards, env.episode.peaks
    return peak["torque"] < w.safe_torque and peak["speed"] < w.safe_joint_speed and peak["impact"] < w.safe_impact


def update_speed(env, fell):
    """Promote the top speed when the top commands are tracked through a whole episode; demote
    when the robot falls at the top."""
    c, cur, ep = env.cfg.command, env.curriculum, env.episode
    top = env.command.max_forward  # with commands re-drawn mid-episode: the fastest one asked for
    if env.command.fixed or not env.cfg.randomization.enabled or top < 0.75 * cur.speed_max:
        return
    if fell:
        cur.speed_max = max(c.speed_range[1], cur.speed_max - c.curriculum_step)
    elif ep.speed_err_sum / ep.steps < 0.25 * top and episode_was_safe(env):
        cur.speed_max = min(c.speed_limit, cur.speed_max + c.curriculum_step)


def score_episode(env, stayed_up):
    """Physics difficulty: every difficulty_window episodes, harder if the robot stayed up in at
    least difficulty_success of them, easier otherwise."""
    rc, cur = env.cfg.randomization, env.curriculum
    cur.episode_tally += (float(stayed_up), 1.0)
    if cur.episode_tally[1] >= rc.difficulty_window:
        ok = cur.episode_tally[0] / cur.episode_tally[1] >= rc.difficulty_success
        cur.difficulty = float(np.clip(cur.difficulty + (rc.difficulty_step if ok else -rc.difficulty_step),
                                       0.0, 1.0))
        cur.episode_tally[:] = 0.0


def score_push(env, k, stayed_up):
    """Push strength: after every push_window pushes from direction k, stronger if the robot stayed
    up through at least push_success of them, weaker otherwise."""
    rc, cur = env.cfg.randomization, env.curriculum
    cur.pending_push = None
    cur.push_tally[k] += (float(stayed_up), 1.0)
    if cur.push_tally[k, 1] >= rc.push_window:
        ok = cur.push_tally[k, 0] / cur.push_tally[k, 1] >= rc.push_success
        cur.push_level[k] = np.clip(cur.push_level[k] + (rc.push_step if ok else -rc.push_step),
                                    rc.push_start, rc.push_velocity)
        cur.push_tally[k] = 0.0
