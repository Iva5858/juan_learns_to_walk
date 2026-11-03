"""Expressive gestures: short keyframe motions in the simulator's joint angles (degrees, legs straight
= 0, signs in robot/joints.py), played at 50 Hz on the robot (robot.dashboard) and checked in MuJoCo
(scripts/check_gestures.py: stays upright, servos within limits) before they touch the robot.

Every gesture starts and ends at home, with both feet on the ground. Juan has no ankle roll, so
anything needing balance on one foot is a trained skill, not a gesture.

A leg pose is built from its knee bend k with the foot kept flat (hip pitch = ankle = -k/2: the
three sum to 0, so the sole stays parallel to the torso), plus optional offsets. A negative k bends the
knee backwards, like a bird's (the "bird" gestures); they reach it through straight legs.
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
MAX_TILT = {"bow": 25.0, "deep_bow": 25.0, "look_up": 22.0}
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


# ---- with the full joint ranges (deep knee bends, bigger leans); each checked in sim like the rest.
# Lifting a foot (stomp, kick) fell on every robot in sim: balancing on one foot needs a trained skill.
def _hold(p, t_in, t_hold, t_out):
    return [(t_in, p, "smooth"), (t_hold, p, "smooth"), (t_out, HOME, "smooth"), (0.3, HOME, "smooth")]


def _squat():
    low = pose(k=100)
    keys = []
    for _ in range(3):
        keys += [(1.0, low, "smooth"), (0.3, low, "smooth"), (0.9, pose(k=10), "smooth"), (0.2, pose(k=10), "smooth")]
    return keys + [(0.8, HOME, "smooth"), (0.3, HOME, "smooth")]


def _nod():  # yes: the hips fold forward and back
    keys = []
    for _ in range(3):
        keys += [(0.25, pose(k=HOME_KNEE + 8, lean=8), "smooth"), (0.25, HOME, "smooth")]
    return keys + [(0.3, HOME, "smooth")]


def _shake():  # no: quick hip sway side to side
    return [(0.15, pose(sway=4 * s), "smooth") for s in (1, -1, 1, -1, 1, -1)] + [(0.2, HOME, "smooth"), (0.3, HOME, "smooth")]


def _look_up():  # the head (camera) tilts up: lean back, knees bend to keep the weight over the feet
    return _hold(pose(k=70, lean=-12), 0.8, 1.2, 0.8)


def _look_down():  # 12 deg of lean with less knee bend tipped one robot in 7 (sim)
    return _hold(pose(k=95, lean=10), 0.9, 1.2, 0.9)


def _deep_bow():  # 22 deg of fold tipped one robot in 7 (sim)
    return _hold(pose(k=100, lean=14), 1.2, 0.9, 1.2)


def _scared():  # drop low, tremble, slowly rise (a faster drop reached stall torque in sim)
    low = pose(k=88, lean=3)
    keys = [(0.45, low, "smooth")] + [(0.07, pose(k=88 + 2 * s, lean=3), "smooth") for s in (1, -1) * 6]
    return keys + [(0.6, low, "smooth"), (1.6, HOME, "smooth"), (0.3, HOME, "smooth")]


def _tired():  # slump, a sigh, rise
    slump = pose(k=85, lean=8)
    return [(2.0, slump, "smooth"), (0.6, pose(k=92, lean=10), "smooth"), (1.0, slump, "smooth"),
            (1.8, HOME, "smooth"), (0.3, HOME, "smooth")]


def _proud():  # stand tall, chest out
    return [(1.0, pose(k=6, lean=-4), "spring"), (1.5, pose(k=6, lean=-4), "smooth"), (0.9, HOME, "smooth"),
            (0.3, HOME, "smooth")]


def _head_tilt():  # curious: tilts to one side (+-10 deg of knee tipped one robot in 7, sim)
    return _hold(pose(k=HOME_KNEE + 7, k_right=HOME_KNEE - 7, sway=-2), 0.7, 1.0, 0.7)


def _wiggle():
    return [(0.12, pose(k=HOME_KNEE + 6, sway=3 * s), "smooth") for s in (1, -1) * 5] + [(0.3, HOME, "smooth"), (0.3, HOME, "smooth")]


def _power_pose():  # wide stance: hips roll outward, sink, hold
    return _hold(pose(k=60, roll=8), 0.8, 1.4, 0.8)


def _excited():  # little bounces getting faster (faster than 0.2 s a half reached stall torque, sim)
    keys = []
    for t in (0.3, 0.26, 0.23, 0.2, 0.2, 0.2):
        keys += [(t, pose(k=HOME_KNEE + 12), "smooth"), (t, pose(k=HOME_KNEE - 8), "smooth")]
    return keys + [(0.4, HOME, "smooth"), (0.3, HOME, "smooth")]


def _stretch():  # tall and leaning back, then a slow deep squat
    return [(1.0, STRAIGHT, "smooth"), (0.6, pose(k=4, lean=-6), "smooth"), (1.6, pose(k=100), "smooth"),
            (0.8, pose(k=100), "smooth"), (1.4, HOME, "smooth"), (0.3, HOME, "smooth")]


# ---- knees bending backwards (k < 0): the home crouch mirrored. Checked in sim like the rest; swapping
# which knee is reversed (through straight legs or through home) tipped one robot in 7, so it isn't here.
BIRD = -HOME_KNEE


def _into_bird(t=0.9):
    return [(t, STRAIGHT, "smooth"), (t, pose(k=BIRD), "smooth")]


def _out_of_bird(t=0.9):
    return [(t, STRAIGHT, "smooth"), (t, HOME, "smooth"), (0.3, HOME, "smooth")]


def _bird():
    return _into_bird() + [(1.5, pose(k=BIRD), "smooth")] + _out_of_bird()


def _bird_deep():
    return _into_bird() + [(1.0, pose(k=-95), "smooth"), (0.8, pose(k=-95), "smooth"),
                           (1.0, pose(k=BIRD), "smooth")] + _out_of_bird()


def _bird_bounce():
    keys = _into_bird()
    for _ in range(4):
        keys += [(0.3, pose(k=BIRD - 14), "smooth"), (0.3, pose(k=BIRD + 10), "smooth")]
    return keys + [(0.3, pose(k=BIRD), "smooth")] + _out_of_bird()


def _knee_flip():  # home -> bird -> home, three times, quicker each time
    keys = []
    for t in (0.8, 0.6, 0.5):
        keys += [(t, STRAIGHT, "smooth"), (t, pose(k=BIRD), "smooth"), (0.3, pose(k=BIRD), "smooth"),
                 (t, STRAIGHT, "smooth"), (t, HOME, "smooth"), (0.3, HOME, "smooth")]
    return keys


def _twisted():  # left knee the normal way, right knee reversed
    odd = pose(k=HOME_KNEE, k_right=BIRD)
    return [(0.9, pose(k=HOME_KNEE, k_right=4), "smooth"), (0.9, odd, "smooth"), (1.5, odd, "smooth"),
            (0.9, pose(k=HOME_KNEE, k_right=4), "smooth"), (0.9, HOME, "smooth"), (0.3, HOME, "smooth")]


def _bird_bow():  # 10 deg of lean came within 0.2 deg of the tilt limit (sim)
    low = pose(k=-80, lean=7)
    return _into_bird() + [(1.0, low, "smooth"), (0.8, low, "smooth"), (1.0, pose(k=BIRD), "smooth")] + _out_of_bird()


def _bird_dance(beats=12, bpm=100):
    beat = 60.0 / bpm
    keys = _into_bird()
    for b in range(beats):
        side = 1 if (b // 2) % 2 == 0 else -1
        keys += [(beat * 0.45, pose(k=BIRD - 12, sway=3 * side), "snappy"),
                 (beat * 0.55, pose(k=BIRD + 8, sway=3 * side), "smooth")]
    return keys + [(0.4, pose(k=BIRD), "smooth")] + _out_of_bird()


def _bird_sway():
    keys = _into_bird()
    for _ in range(3):
        keys += [(0.7, pose(k=BIRD - 6, k_right=BIRD + 6, sway=-1), "smooth"),
                 (0.7, pose(k=BIRD + 6, k_right=BIRD - 6, sway=1), "smooth")]
    return keys + [(0.5, pose(k=BIRD), "smooth")] + _out_of_bird()


def _bird_look():  # bird crouch, then lean back to look up
    up = pose(k=-70, lean=-8)
    return _into_bird() + [(0.9, up, "smooth"), (1.0, up, "smooth"), (0.9, pose(k=BIRD), "smooth")] + _out_of_bird()


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
    "squat": ("Squats", "Three deep squats, feet flat", "ok", _squat()),
    "nod": ("Nod (yes)", "Folds forward and back three times", "ok", _nod()),
    "shake": ("Shake (no)", "Quick hip sways side to side", "alert", _shake()),
    "look_up": ("Look up", "Leans back so the head and camera tilt up, holds", "curious", _look_up()),
    "look_down": ("Look down", "Leans forward so the head and camera tilt down, holds", "curious", _look_down()),
    "deep_bow": ("Deep bow", "A low bow with deep knees", "goodbye", _deep_bow()),
    "scared": ("Scared", "Drops low, trembles, slowly rises", "scared", _scared()),
    "tired": ("Tired", "Slumps low, sighs, gets back up", "sleepy", _tired()),
    "proud": ("Proud", "Stands tall, chest out, holds", "happy", _proud()),
    "head_tilt": ("Head tilt", "Tilts to one side, curious", "curious", _head_tilt()),
    "wiggle": ("Wiggle", "Quick little side-to-side wiggles", "happy", _wiggle()),
    "power_pose": ("Power pose", "Wide stance, sinks, holds", "excited", _power_pose()),
    "excited": ("Excited", "Little bounces getting faster", "excited", _excited()),
    "stretch": ("Stretch", "Tall and leaning back, then a slow deep squat", "sleepy", _stretch()),
    "bird": ("Bird legs", "Knees flip backwards into a bird crouch, hold, flip back", "chirp", _bird()),
    "bird_deep": ("Deep bird", "A deep squat on backward knees", "curious", _bird_deep()),
    "bird_bounce": ("Bird bounce", "Bounces on backward knees", "happy", _bird_bounce()),
    "knee_flip": ("Knee flip", "Knees flip backwards and forwards three times, quicker each time", "excited", _knee_flip()),
    "twisted": ("Twisted", "One knee bends the normal way, the other backwards", "curious", _twisted()),
    "bird_bow": ("Bird bow", "A bow on backward knees", "hello", _bird_bow()),
    "bird_dance": ("Bird dance", "The dance on backward knees", "excited", _bird_dance()),
    "bird_sway": ("Bird sway", "Rocks side to side on backward knees", "happy", _bird_sway()),
    "bird_look": ("Bird look", "Bird crouch, then leans back to look up", "curious", _bird_look()),
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
