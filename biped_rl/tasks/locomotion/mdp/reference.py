"""Reference gait: an ostrich-like walk scripted for Juan, as joint angles over the gait cycle.

Knees bent backwards (the knees_back crouch: thigh back, shin forward), the torso level and gliding at a
constant height, each foot rising quickly into a high fold, swinging through the air and reaching out ahead
before it lands flat; a light sideways sway over the stance foot. Policies learn from it by imitation
(DeepMimic, Peng et al. 2018; the clock-timed footfalls of Siekmann et al. 2021, on bird-legged Cassie).

Walking backwards, the knees bend forwards instead: the mirror image of the forward walk.

Commands (forward, sideways, turn rate) shape the steps: forward and sideways speed set how far each foot
travels under the hip during stance; turning lengthens the steps on the outside of the turn and shortens
(or reverses) them on the inside (Juan has no hip yaw joint). The rhythm quickens with speed.

Planar leg model in the sagittal plane, measured on juan/rl_robot.xml: hip pitch, knee and ankle axes in a
vertical line 0.100 m apart with the legs straight. Signs (simulator): hip pitch + = leg back, knee + = shin
back (a human knee; negative = bent backwards), ankle + = toes down; hip roll + = foot toward the robot's
left on either leg (left leg outward / right leg inward). A flat foot needs ankle = -(hip + knee).
"""
from dataclasses import dataclass

import numpy as np

THIGH = SHIN = 0.100          # m, hip pitch -> knee -> ankle
ROLL_TO_FOOT = 0.27           # m, hip roll axis down to the sole: sideways reach per rad of roll


@dataclass
class GaitParams:
    # Sized to the servos (5.8 rad/s with no load): every joint stays under their top speed up to ~0.3 m/s;
    # faster than stride_speed_max is the policy's own.
    # A deep crouch lifts the foot with less knee travel; the swing (half the cycle: one foot always down)
    # is the longest a walk allows, since the swing foot's speed, not the rhythm, is what the servos limit.
    knee_crouch_deg: float = 66.0    # backward knee bend standing (deeper than the 46 deg knees_back pose)
    steps_slow: float = 2.4          # steps/s (both feet) at a standstill command ...
    steps_fast: float = 3.4          # ... and at fast_speed
    fast_speed: float = 0.6          # m/s
    # Checked in physics (the servo model, a harness holding the torso up): with half the cycle in the air and
    # 20 mm of lift the swing foot dragged and the body moved at 26% of the command; a short double support
    # (swing 0.4) and 35 mm of lift move it at 59-68% (juan_v9).
    swing_ratio: float = 0.4         # share of the cycle each foot is in the air (0.2 of it both feet down)
    lift: float = 0.035              # m, foot clearance at the top of the swing
    fold_deg: float = 6.0            # toes down mid-swing (the foot folds, then flattens to land)
    stride_speed_max: float = 0.35   # m/s: faster commands keep this stride (~100% servo speed); the rest is the policy's
    sway_deg: float = 2.0            # hip roll shifting the body over the stance foot (small: a steady camera)
    hip_width: float = 0.10          # m between the feet, for turning by step length


def pose_now(env, ahead_steps=0):
    """The reference pose for the env's command at its gait clock, `ahead_steps` control steps ahead."""
    cmd = env.command
    vec = np.zeros(3) if cmd.standing else cmd.vec
    phase = (cmd.phase + ahead_steps * env.dt * cadence(vec) / 2) % 1.0
    return reference_pose(phase, vec)




def leg_angles(x, z, knee_sign=-1.0):
    """Hip pitch and knee for the ankle at (x forward, z up) from the hip pitch axis. knee_sign -1: knee
    bent backwards (the bird leg), +1: forwards."""
    d = np.clip(np.hypot(x, z), 1e-6, THIGH + SHIN - 1e-6)
    phi = np.arctan2(x, -z)                                       # leg direction, + = forward
    alpha = np.arccos(np.clip((THIGH ** 2 + d ** 2 - SHIN ** 2) / (2 * THIGH * d), -1.0, 1.0))
    a_thigh = phi + knee_sign * alpha                              # thigh angle from vertical, + = forward
    beta = np.arcsin(np.clip(THIGH * np.sin(a_thigh - phi) / SHIN, -1.0, 1.0))
    a_shin = phi - beta
    return -a_thigh, a_thigh - a_shin                              # hip pitch, knee


def ankle_position(hip, knee):
    """Ankle (x forward, z up) from the hip pitch axis: the forward model of leg_angles."""
    a_thigh = -hip
    a_shin = a_thigh - knee
    return THIGH * np.sin(a_thigh) + SHIN * np.sin(a_shin), -(THIGH * np.cos(a_thigh) + SHIN * np.cos(a_shin))


def cadence(command, p=None):
    """Steps per second for a command: quicker with speed (sideways and turning count as speed)."""
    p = p or DEFAULT
    vx, vy, wz = command
    speed = np.hypot(vx, vy) + abs(wz) * p.hip_width / 2
    return p.steps_slow + (p.steps_fast - p.steps_slow) * min(speed / p.fast_speed, 1.0)


def _smooth(s, ramp=0.25):
    """0 -> 1 over s in [0, 1], starting and ending at rest: speed ramps up (sin^2) over the first `ramp`,
    holds, and ramps down over the last. Its top speed is 1/(1 - ramp) of the average (1.33x), against
    minimum jerk's 1.875x: the servos' top speed is what limits how fast Juan's foot can swing."""
    s = float(np.clip(s, 0.0, 1.0))
    top = 1.0 / (1.0 - ramp)
    if s < ramp:   # integral of top * sin^2(pi/2 * s/ramp)
        return top * (s / 2 - ramp / (2 * np.pi) * np.sin(np.pi * s / ramp))
    if s > 1 - ramp:
        return 1.0 - _smooth(1.0 - s, ramp)
    return top * (ramp / 2 + (s - ramp))


def foot_path(leg_phase, step, p=None, stance=1.0):
    """(along, up) of one foot relative to its neutral spot under the hip, for a step of `step` m
    (signed: + forward). leg_phase in [0, 1): swing during [0, swing_ratio), stance after. Stance: the foot
    glides back at a constant speed, flat on the floor (the body glides forward). Swing: it rises fast and
    folds, swings through, and reaches forward before landing (an ostrich's quick lift and long reach)."""
    p = p or DEFAULT
    sw = p.swing_ratio
    if leg_phase < sw:
        s = leg_phase / sw
        along = -step / 2 + step * _smooth(s)
        up = p.lift * np.sin(np.pi * s) ** 2             # rises and lowers smoothly, highest mid-swing
        fold = np.radians(p.fold_deg) * np.sin(np.pi * s) ** 2
        return along, up, fold
    s = (leg_phase - sw) / (1 - sw)
    return step / 2 - step * s, 0.0, 0.0


def reference_pose(phase, command, p=None):
    """Joint angles (simulator order: left hip roll, hip pitch, knee, ankle, then right) at gait phase
    [0, 1) for a command (forward m/s, sideways m/s, turn rad/s). Left foot swings first, the right half a
    cycle later. A zero command: standing in the crouch, feet down."""
    p = p or DEFAULT
    vx, vy, wz = command
    knee0 = np.radians(p.knee_crouch_deg)
    z0 = -(THIGH + SHIN) * np.cos(knee0 / 2)          # hip -> ankle height in the crouch
    standing = abs(vx) < 1e-3 and abs(vy) < 1e-3 and abs(wz) < 1e-3
    backwards = vx < -0.02 and abs(vx) >= abs(vy)
    cycle = 2.0 / cadence(command, p)
    stance_time = (1 - p.swing_ratio) * cycle
    q = np.zeros(8)
    for leg, offset, side in ((0, 0.0, 1.0), (1, 0.5, -1.0)):   # side: + = left
        leg_phase = (phase + offset) % 1.0
        # turning while walking: the outside foot steps longer (turn left = right foot further). Turning in
        # place is left to the policy: the reference just steps on the spot.
        walking = min(np.hypot(vx, vy) / 0.1, 1.0)
        v_leg = np.clip(vx - walking * side * wz * p.hip_width / 2, -p.stride_speed_max, p.stride_speed_max)
        if standing:
            along, up, fold, lateral = 0.0, 0.0, 0.0, 0.0
        else:
            along, up, fold = foot_path(leg_phase, v_leg * stance_time, p, stance_time)
            lateral, _, _ = foot_path(leg_phase, np.clip(vy, -p.stride_speed_max, p.stride_speed_max) * stance_time, p, stance_time)
        # Backwards: knees bent forwards, the mirror image of the forward bird walk (Juan's leg is nearly
        # symmetric front to back), so it walks backwards as well as forwards; the toe fold becomes a heel fold.
        # Switching direction flips the knees through straight legs: the policy learns that transition.
        knee_sign = 1.0 if backwards else -1.0
        hip, knee = leg_angles(along, z0 + up, knee_sign)
        fold = -knee_sign * fold
        # Sway: both feet move the same way under the body, shifting it over the stance foot (left swing ->
        # body over the right foot -> both feet toward +y), peaking mid-swing.
        wave = side * np.cos(2 * np.pi * (leg_phase - p.swing_ratio / 2))
        sway = 0.0 if standing else np.radians(p.sway_deg) * wave
        roll = lateral / ROLL_TO_FOOT + sway                     # + = foot toward +y (left), on either leg
        q[4 * leg:4 * leg + 4] = (roll, hip, knee, -(hip + knee) + fold)
    return q


DEFAULT = GaitParams()
