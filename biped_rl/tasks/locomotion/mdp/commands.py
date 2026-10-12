"""Commands: what the robot is asked to do (forward speed, sideways speed, turn rate, or stand
still), plus what follows from them: the gait clock, the target heading and the commanded path.

"omni" mode draws a new command every few seconds; "forward" mode (runs up to juan_v6) draws one
forward speed per episode.
"""
import numpy as np


def reset(env, options):
    """Commands for a new episode. options={"speed": v} (forward only), {"command": (vx, vy,
    yaw_rate)} or {"stand": True} fix the command for the whole episode (evaluation, videos)."""
    cfg, rng, cmd = env.cfg.command, env.np_random, env.command
    cmd.fixed = any(k in options for k in ("speed", "command", "stand"))
    cmd.standing, cmd.max_forward = False, -np.inf
    if "stand" in options:
        cmd.standing, cmd.vec = bool(options["stand"]), np.zeros(3)
    elif "command" in options:
        cmd.vec = np.array(options["command"], dtype=float)
    elif "speed" in options:
        cmd.vec = np.array([float(options["speed"]), 0.0, 0.0])
    elif env.omni:
        sample(env)
    else:
        cmd.vec = np.array([rng.uniform(cfg.speed_range[0], env.curriculum.speed_max), 0.0, 0.0])
    cmd.max_forward = max(cmd.max_forward, cmd.forward)
    # (Only drawn in omni mode, so forward-mode runs see the same random stream as before.)
    cmd.next_resample = rng.uniform(*cfg.resample_interval) if env.omni else np.inf


def sample(env):
    """Draw (forward, sideways, turn rate) and whether to stand. Forward speeds come from
    [-backward_speed, top of the speed curriculum]; a share of commands is straight walking, and
    share each only turns, only walks backward, or only walks sideways."""
    c, rng, cmd = env.cfg.command, env.np_random, env.command
    cmd.standing = bool(rng.uniform() < c.stand_prob)
    # One draw picks the kind of command (none when all shares are off, so older runs see the same
    # random stream as before).
    shares = np.cumsum([c.turn_in_place_prob, c.backward_prob, c.sideways_prob])
    kind = int(np.searchsorted(shares, rng.uniform(), side="right")) if shares[-1] > 0 and not cmd.standing else 3
    if cmd.standing:
        cmd.vec = np.zeros(3)
    elif kind == 0:
        cmd.vec = np.array([0.0, 0.0, rng.uniform(*c.yaw_rate_range)])
    elif kind == 1:
        cmd.vec = np.array([-rng.uniform(0.0, c.backward_speed), 0.0, 0.0])
    elif kind == 2:
        cmd.vec = np.array([0.0, rng.uniform(*c.lateral_range), 0.0])
    else:
        vx = rng.uniform(-c.backward_speed, env.curriculum.speed_max)
        if rng.uniform() < c.straight_prob:
            vy = wz = 0.0
        else:
            vy, wz = rng.uniform(*c.lateral_range), rng.uniform(*c.yaw_rate_range)
        cmd.vec = np.array([vx, vy, wz])
    cmd.max_forward = max(cmd.max_forward, cmd.forward)


def resample(env):
    """A new command every few seconds (omni mode): the policy must change gait, turn, stop and
    start again mid-walk, like with a joystick."""
    cmd = env.command
    if not cmd.fixed and env.episode.t >= cmd.next_resample:
        sample(env)
        cmd.next_resample += env.np_random.uniform(*env.cfg.command.resample_interval)


def set_command(env, forward, sideways=0.0, turn_rate=0.0):
    """Change the command mid-episode (keyboard, joystick). All zero = stand still."""
    cmd = env.command
    cmd.vec = np.array([forward, sideways, turn_rate], dtype=float)
    cmd.standing = not np.any(cmd.vec)


def gait_period(env):
    """Faster rhythm at higher commands: a fixed rhythm caps top speed."""
    c, vec = env.cfg.command, env.command.vec
    frac = min(np.hypot(vec[0], vec[1]) / c.fast_speed, 1.0)
    return c.gait_period - (c.gait_period - c.gait_period_fast) * frac


def phase(env):
    """Gait clock, 0..1: the left foot swings during [0, swing_ratio), the right half a cycle later."""
    return env.command.phase % 1.0


def heading_error(env):
    """Yaw relative to the target heading, wrapped to [-pi, pi]. The target is the start heading
    turned by the integrated turn-rate command. On the robot: IMU yaw minus the yaw recorded at
    the start, minus the integrated joystick turn rate."""
    R = env.root_rot()
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return (yaw - env.command.heading_target + np.pi) % (2 * np.pi) - np.pi


def lateral_offset(env):
    """Signed sideways distance from the commanded path (privileged). The reference point follows
    the robot along the target heading and moves sideways only as commanded, so drifting off a
    straight line is measured, but falling behind on speed is not."""
    heading = env.command.heading_target
    side = np.array([-np.sin(heading), np.cos(heading)])
    return float(side @ (env.data.qpos[0:2] - env.command.ref_xy))
