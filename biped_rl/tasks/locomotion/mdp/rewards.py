"""Reward terms, one function per term. Each function is named like its weight in RewardWeights
and returns the weighted term: f(env, s, w), where s is the step data the environment measured
(see BipedEnv._measure) and w the weights.

    reward = (alive + tracking * sum(task terms)) * exp(-sum(penalties))  [+ termination]

* tracking: how well the speed command is met (1 = exact). Every task term is multiplied by it,
  so heading, rhythm and foot lift only pay while actually moving at the commanded speed.
* Penalties scale the reward down instead of being subtracted: the reward is never negative
  (falling early never pays) and penalties still bite when things go badly.
"""
import numpy as np

from biped_rl.tasks.locomotion.mdp import commands


# --------------------------------------------------------------------- task terms (x tracking)
def tracking_vel(env, s, w):
    """The tracking factor itself, as a reward."""
    return w.tracking_vel


def tracking_lat_yaw(env, s, w):
    """Sideways speed and turn rate as commanded."""
    return w.tracking_lat_yaw * np.exp(-((s.vy - s.cmd[1]) ** 2 + 0.25 * (s.ang_world[2] - s.cmd[2]) ** 2) / 0.02)


def heading(env, s, w):
    """Face the target heading (~13 deg off -> 0.37)."""
    return w.heading * np.exp(-(s.yaw_err ** 2) / 0.05)


def gait_contact(env, s, w):
    """Feet on the ground when the gait clock says stance, in the air when it says swing."""
    return w.gait_contact * np.mean(env.episode.contact == ~s.swing)


def feet_clearance(env, s, w):
    """Swing soles reach swing_height: no shuffling, no toe drag."""
    c = env.cfg.command
    return w.feet_clearance * np.mean(
        s.swing * np.exp(-(((env.episode.clearance - c.swing_height) / (0.5 * c.swing_height)) ** 2)))


def flight(env, s, w):
    """Both feet off the ground while moving at >= half the command (running)."""
    forward = env.command.forward
    return w.flight * float(env.episode.contact.sum() == 0 and s.vx > 0.5 * forward > 0)


TASK_TERMS = (tracking_vel, tracking_lat_yaw, heading, gait_contact, feet_clearance, flight)


# --------------------------------------------------------------------- penalties
def path_offset(env, s, w):
    """Distance^2 from the commanded path [m^2] (0.02 at 10 cm)."""
    return w.path_offset * min(s.offset ** 2, 1.0)


def orientation(env, s, w):
    """Torso tilt (roll, pitch)."""
    return w.orientation * np.sum(s.grav[:2] ** 2)


def ang_vel_xy(env, s, w):
    """Torso wobble."""
    return w.ang_vel_xy * np.sum(env.data.qvel[3:5] ** 2)


def lin_vel_z(env, s, w):
    """Torso bouncing."""
    return w.lin_vel_z * env.data.qvel[2] ** 2


def base_height(env, s, w):
    """Centre of mass below its home height [m^2]: no crouch-walking."""
    return w.base_height * min(env.com_height() - env.robot.home_height, 0.0) ** 2


def knee_hyperextension(env, s, w):
    """Knee bent backwards [rad]: no bird-legged gait."""
    return w.knee_hyperextension * np.sum(np.maximum(-s.q[env.robot.knee_idx], 0.0))


def hip_roll(env, s, w):
    """Hip roll away from 0 [rad^2]: no splayed, waddling legs."""
    return w.hip_roll * np.sum(s.q[env.robot.hip_roll_idx] ** 2)


def hip_roll_inward(env, s, w):
    """Per rad of inward roll: 0 deg inward is the goal, the joint limit only a margin."""
    robot = env.robot
    return w.hip_roll_inward * np.sum(np.maximum(-robot.roll_out_sign * s.q[robot.hip_roll_idx], 0.0))


def action_rate(env, s, w):
    """First difference of the actions."""
    return w.action_rate * np.mean((s.action - env.episode.last_action) ** 2)


def action_acc(env, s, w):
    """Second difference of the actions (jerk)."""
    ep = env.episode
    return w.action_acc * np.mean((s.action - 2 * ep.last_action + ep.last_action2) ** 2)


def joint_acc(env, s, w):
    """Joint accelerations [rad/s^2]."""
    return w.joint_acc * np.mean(s.qacc ** 2)


def servo_load(env, s, w):
    """(torque / stall)^2: heat and gear wear."""
    return w.servo_load * s.load / env.frame_skip


def servo_saturation(env, s, w):
    """Share of time on the torque-speed limit."""
    return w.servo_saturation * s.saturated / (env.frame_skip * env.nu)


def servo_overload(env, s, w):
    """Torque above 80% of stall, ((|tau| - 0.8 stall) / stall)^2 per joint."""
    return w.servo_overload * s.overload.sum() / env.frame_skip


def backdrive(env, s, w):
    """Joint forced faster than the servo's no-load speed: how hobby-servo gears break."""
    return w.backdrive * s.overspeed.sum() / env.frame_skip


def impact(env, s, w):
    """Foot landing force above impact_threshold x body weight."""
    return w.impact * np.sum(np.clip(s.peak_force / env.robot.weight - w.impact_threshold, 0, 3))


def foot_slip(env, s, w):
    """Sole sliding while loaded or actually touching (<3 mm): catches light-touch dragging, but
    a normal step passing near the floor at lift-off and touch-down is not slip."""
    ep = env.episode
    on_floor = np.maximum(ep.contact, ep.clearance < 0.003)
    return w.foot_slip * np.sum(on_floor * np.sum(s.sole_vel[:, :2] ** 2, axis=1))


def joint_limits(env, s, w):
    """Approaching the mechanical end stops (within 5% of the range)."""
    robot = env.robot
    margin = 0.05 * (robot.joint_hi - robot.joint_lo)
    return w.joint_limits * np.sum(np.clip(robot.joint_lo + margin - s.q, 0, None)
                                   + np.clip(s.q - (robot.joint_hi - margin), 0, None))


def feet_distance(env, s, w):
    """Feet closer than min_feet_distance sideways: legs colliding."""
    return w.feet_distance * max(w.min_feet_distance - s.feet_y, 0.0)


def energy(env, s, w):
    """Mechanical power [W]."""
    return w.energy * s.energy / env.dt


def _stride(env):
    """Steps per full stride: measured from touchdowns if available, else the gait clock's."""
    return env.episode.stride_est or int(round(commands.gait_period(env) / env.dt))


def symmetry(env, s, w):
    """Each leg now vs the other leg half a stride ago (mirrored) [rad^2]: a limp scores > 0."""
    robot, hist = env.robot, env.episode.q_hist
    half = int(round(0.5 * _stride(env)))
    if len(robot.mirror_l) and len(hist) > half:
        past = hist[-1 - half]
        asym = (np.mean((s.q[robot.mirror_l] - robot.mirror_sign * past[robot.mirror_r]) ** 2)
                + np.mean((s.q[robot.mirror_r] - robot.mirror_sign * past[robot.mirror_l]) ** 2))
    else:
        asym = 0.0
    return w.symmetry * asym


def posture_symmetry(env, s, w):
    """Left vs right leg AVERAGE angle over the last stride (mirrored) [rad^2]: a leg held
    straighter or more bent all the time is a limp too."""
    robot, hist = env.robot, env.episode.q_hist
    steps = max(_stride(env), 1)
    if len(robot.mirror_l) and len(hist) >= steps:
        recent = np.array(list(hist)[-steps:])
        gap = (recent[:, robot.mirror_l] - robot.mirror_sign * recent[:, robot.mirror_r]).mean(axis=0)
        return w.posture_symmetry * float(np.mean(gap ** 2))
    return 0.0


def step_timing(env, s, w):
    """|left->right - right->left interval| / their sum, charged every step: a lopsided rhythm."""
    return w.step_timing * env.episode.timing_asym


def gait_mismatch(env, s, w):
    """Share of feet on or off the ground against the gait clock."""
    return w.gait_mismatch * float(np.mean(env.episode.contact != ~s.swing))


def footstep(env, s, w):
    """Footsteps beyond the gait clock's plan (2 per stride; standing: 0) within the last stride.
    Charging every step instead made walking slower the cheapest way to take fewer steps."""
    steps = int(round(commands.gait_period(env) / env.dt))
    recent = sum(list(env.episode.td_hist)[-steps:])
    allowed = 0.0 if env.command.standing else 2.0
    return w.footstep * min(s.touchdowns, max(recent - allowed, 0.0))


def reversals(env, s, w):
    """Servo direction reversals (gear backlash hammering, potentiometer wear)."""
    return w.reversals * s.reversals


def stand_still(env, s, w):
    """Not moving while commanded to walk: 0 at >= half the commanded speed, 1 when stopped."""
    forward = env.command.forward
    if forward > 0.1:
        return w.stand_still * float(np.clip(1.0 - s.vx / (0.5 * forward), 0.0, 1.0))
    return 0.0


PENALTY_TERMS = (path_offset, orientation, ang_vel_xy, lin_vel_z, base_height, knee_hyperextension,
                 hip_roll, hip_roll_inward, action_rate, action_acc, joint_acc, servo_load,
                 servo_saturation, servo_overload, backdrive, impact, foot_slip, joint_limits,
                 feet_distance, energy, symmetry, posture_symmetry, step_timing, gait_mismatch,
                 footstep, reversals, stand_still)

# Free while recovering (1 s after a shove, or the torso tilted > 8 deg): otherwise they punish
# the quick extra step that catches a fall.
PAUSED_WHILE_RECOVERING = ("symmetry", "posture_symmetry", "step_timing", "gait_mismatch", "footstep", "reversals")


def compute(env, s):
    """All terms for this step. Sets s.tracking, s.speed_error and s.recovering; returns
    (reward without the termination penalty, task terms, penalties, penalty sum)."""
    w = env.cfg.rewards
    # How well the speed command is met, in [0, 1]: a narrow kernel for precision plus a broad one
    # so that creeping forward already beats standing still. Omni mode: sideways speed counts too.
    err = s.vx - s.cmd[0]
    err2 = err ** 2 + ((s.vy - s.cmd[1]) ** 2 if env.omni else 0.0)
    share = w.tracking_broad_share
    s.tracking = ((1 - share) * np.exp(-err2 / w.tracking_sigma2)
                  + share * np.exp(-err2 / w.tracking_broad_sigma2))
    s.speed_error = abs(err)
    env.episode.speed_err_sum += np.sqrt(err2)

    task = {f.__name__: f(env, s, w) for f in TASK_TERMS}
    task = {k: s.tracking * v for k, v in task.items()}
    pen = {f.__name__: f(env, s, w) for f in PENALTY_TERMS}
    s.recovering = (env.episode.t - env.episode.last_push < 1.0) or np.linalg.norm(s.grav[:2]) > np.sin(np.radians(8))
    if s.recovering:
        for k in PAUSED_WHILE_RECOVERING:
            pen[k] = 0.0
    penalty = sum(pen.values())
    return (w.alive + sum(task.values())) * float(np.exp(-penalty)), task, pen, penalty
