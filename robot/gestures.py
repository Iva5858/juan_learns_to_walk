"""Expressive gestures: short keyframe motions in the simulator's joint angles (degrees, legs straight
= 0, signs in robot/joints.py), played at 50 Hz on the robot (robot.dashboard) and checked in MuJoCo
(scripts/check_gestures.py: stays upright, servos within limits) before they touch the robot.

Every gesture starts and ends at home, with both feet on the ground. Juan has no ankle roll, so
anything needing balance on one foot is a trained skill, not a gesture.

A leg pose is built from its knee bend k with the foot kept flat (hip pitch = ankle = -k/2: the
three sum to 0, so the sole stays parallel to the torso), plus optional offsets.
"""
import math

HOME_KNEE = 45.84
CONTROL_HZ = 50


def leg(side, k, lean=0.0, ankle=0.0, roll=0.0):
    """One leg: knee bend k (deg), torso lean forward `lean` (deg, hips fold), extra ankle (deg,
    + toes down), hip roll `roll` (deg, + = outward on either side)."""
    return {f"{side}_hip_roll": roll if side == "left" else -roll,
            f"{side}_hip_pitch": -k / 2 - lean,
            f"{side}_knee": k,
            f"{side}_ankle": -k / 2 + ankle}


def pose(k=HOME_KNEE, k_right=None, lean=0.0, ankle_left=0.0, ankle_right=0.0, roll=0.0, sway=0.0):
    """Both legs. `sway` (deg, + = toward the robot's right) tilts the hips sideways with hip roll."""
    p = leg("left", k, lean, ankle_left, roll + sway)
    p.update(leg("right", k if k_right is None else k_right, lean, ankle_right, roll - sway))
    return p


HOME = pose()
STRAIGHT = pose(k=4)   # (not quite 0: a knee locked straight can snap past it)


# ---- easing: how a move speeds up and slows down (0..1 -> 0..1) ----
def smooth(u):
    return 0.5 - 0.5 * math.cos(math.pi * u)


def spring(u, overshoot=1.6):
    """Ease out with a small overshoot and settle: reads as springy, alive."""
    u -= 1.0
    return 1.0 + u * u * ((overshoot + 1.0) * u + overshoot)


def snappy(u):
    """Fast start, soft landing."""
    return 1.0 - (1.0 - u) ** 3


EASINGS = {"smooth": smooth, "spring": spring, "snappy": snappy}


# ---- the library: name -> (label, description, sound, keyframes) ----
# Torso tilt allowed when checking in sim (default 15 deg): a bow leans on purpose.
MAX_TILT = {"bow": 25.0}
# keyframes: (seconds to get there, target pose, easing)
def _happy_bounce():
    # Rising fast to straight legs with an overshoot throws the robot (checked in sim); the
    # liveliness comes from a quick rise and a small extra dip on landing instead.
    keys = []
    for _ in range(3):
        keys += [(0.4, pose(k=8), "smooth"), (0.4, pose(k=HOME_KNEE + 6), "smooth"), (0.2, HOME, "smooth")]
    return keys + [(0.2, HOME, "smooth")]


def _wake_up():
    keys = [(1.4, STRAIGHT, "smooth"), (0.5, STRAIGHT, "smooth")]
    for side in (1, -1, 1, -1):  # a little shake, hips side to side
        keys.append((0.12, pose(k=4, sway=3 * side), "smooth"))
    return keys + [(0.15, STRAIGHT, "smooth"), (1.0, HOME, "spring"), (0.3, HOME, "smooth")]


def _sad():
    return [(1.6, pose(k=72, lean=-4), "smooth"), (1.5, pose(k=72, lean=-4), "smooth"),
            (1.4, HOME, "smooth"), (0.3, HOME, "smooth")]


def _toe_tap():
    keys = []  # toes up 5 deg: more pushes the heel into the floor and tips the robot (sim)
    for _ in range(3):
        keys += [(0.25, pose(ankle_right=-5), "smooth"), (0.25, HOME, "smooth")]
    return keys + [(0.3, HOME, "smooth")]


def _bow():
    # Hips fold forward 12 deg; knees bend more so the hips go back and the weight stays over the
    # feet (15 deg of fold without it tipped over in sim).
    low = pose(k=62, lean=12)
    return [(1.0, low, "smooth"), (0.7, low, "smooth"), (1.0, HOME, "smooth"), (0.3, HOME, "smooth")]


def _sway(cycles=4, period=1.4):
    keys = []  # one knee bends 6 deg more as the other straightens 6 deg (7 + hip sway tipped it, sim)
    for _ in range(cycles):
        keys += [(period / 2, pose(k=HOME_KNEE + 6, k_right=HOME_KNEE - 6, sway=-1), "smooth"),
                 (period / 2, pose(k=HOME_KNEE - 6, k_right=HOME_KNEE + 6, sway=1), "smooth")]
    return keys + [(0.6, HOME, "smooth"), (0.3, HOME, "smooth")]


def dance(bpm=100, beats=16):
    """Bounce on every beat, sway left / right every two beats."""
    beat = 60.0 / bpm
    keys = []
    for b in range(beats):
        side = 1 if (b // 2) % 2 == 0 else -1
        keys += [(beat * 0.45, pose(k=HOME_KNEE + 14, sway=3 * side), "snappy"),
                 (beat * 0.55, pose(k=HOME_KNEE - 8, sway=3 * side), "smooth")]
    return keys + [(0.6, HOME, "smooth"), (0.3, HOME, "smooth")]


def _settle():  # idle: a small breath-like dip
    return [(0.7, pose(k=HOME_KNEE + 8), "smooth"), (0.9, HOME, "smooth")]


def _fidget():  # idle: one small toe tap with the left foot
    return [(0.2, pose(ankle_left=-8), "snappy"), (0.25, HOME, "smooth"), (0.3, HOME, "smooth")]


def _shift():  # idle: a slow weight shift to one side and back
    side = 1
    return [(0.8, pose(k=HOME_KNEE + 6, k_right=HOME_KNEE - 6, sway=-2 * side), "smooth"),
            (0.6, pose(k=HOME_KNEE + 6, k_right=HOME_KNEE - 6, sway=-2 * side), "smooth"),
            (0.8, HOME, "smooth")]


GESTURES = {
    "happy_bounce": ("Happy bounce", "Springs up to straight legs and back, three times", "happy", _happy_bounce()),
    "wake_up": ("Wake-up stretch", "Rises tall, a little shake, settles into home", "sleepy", _wake_up()),
    "sad": ("Sad", "Slow deep crouch, leaning back a little, then back up", "sad", _sad()),
    "toe_tap": ("Toe tap", "Taps the right toes three times", "curious", _toe_tap()),
    "bow": ("Bow", "Folds forward at the hips, holds, comes back up", "hello", _bow()),
    "sway": ("Sway", "Rocks side to side, one leg bending as the other straightens", "happy", _sway()),
    "dance": ("Dance", "Bounces on the beat and sways (100 BPM)", "excited", dance()),
}
IDLE = {
    "settle": ("Settle", "Idle: a small breath-like dip", None, _settle()),
    "fidget": ("Fidget", "Idle: one small toe tap", None, _fidget()),
    "shift": ("Shift", "Idle: slow weight shift and back", None, _shift()),
}
ALL = {**GESTURES, **IDLE}


def frames(keys, start=None, hz=CONTROL_HZ):
    """Keyframes -> one target pose per control step (dict name -> deg), from `start` (default home)."""
    cur = dict(HOME if start is None else start)
    out = []
    for seconds, target, easing in keys:
        ease = EASINGS[easing]
        n = max(1, round(seconds * hz))
        a = dict(cur)
        for i in range(1, n + 1):
            u = ease(i / n)
            out.append({j: a[j] + (target[j] - a[j]) * u for j in a})
        cur = dict(target)
    return out


def duration(keys):
    return sum(k[0] for k in keys)
