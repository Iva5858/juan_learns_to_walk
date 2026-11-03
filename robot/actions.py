"""Everything the page (or anything else on the network) can make the robot do, declared once.

Each action has a name (also its URL: POST /api/<name> with JSON parameters, or the old GET /<name>?...),
typed parameters, a label and group for the page's buttons, and, when it is risky, the confirmation
the page shows first. GET /api/actions lists them all, so a new feature only adds an @action here and
its button can appear on the page by itself.

A handler gets the robot and its parameters and returns None (done), an error message, or a dict of
extra results.
"""
import time
from dataclasses import dataclass, field

from robot.joints import BY_NAME, HOME, STRAIGHT


@dataclass
class Action:
    name: str
    fn: object
    label: str
    group: str                 # drive, poses, personality, voice, calibration, system
    params: dict = field(default_factory=dict)   # name -> (type, default); no default = required
    confirm: str = None        # asked on the page first
    danger: bool = False

    def describe(self):
        return {"name": self.name, "label": self.label, "group": self.group, "confirm": self.confirm,
                "danger": self.danger, "doc": (self.fn.__doc__ or "").strip(),
                "params": {k: {"type": t.__name__, **({"default": d} if d is not REQUIRED else {})}
                           for k, (t, d) in self.params.items()}}

    def parse(self, raw):
        """Parameters from a query string or JSON, typed; KeyError / ValueError if one is missing or bad."""
        out = {}
        for k, (t, default) in self.params.items():
            if k not in raw or raw[k] in (None, ""):
                if default is REQUIRED:
                    raise KeyError(k)
                out[k] = default
                continue
            v = raw[k]
            if t is bool:
                v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "on", "yes")
            elif t is Joint:
                if v not in BY_NAME:
                    raise ValueError(f"no joint {v}")
            else:
                v = t(v)
            out[k] = v
        return out


class Joint(str):
    """A joint name from robot/joints.py."""


REQUIRED = object()
ACTIONS = {}


def action(action_name, label, group, /, confirm=None, danger=False, **params):
    def register(fn):
        ACTIONS[action_name] = Action(action_name, fn, label, group, {k: v if isinstance(v, tuple) else (v, REQUIRED)
                                                        for k, v in params.items()}, confirm, danger)
        return fn
    return register


# ---------------------------------------------------------------- system and poses
@action("stop", "Stop all", "system", danger=True)
def stop(robot):
    """Stop the policy and any gesture; the servos hold where they are."""
    robot.stop()


@action("shutdown", "Shut down dashboard", "system", danger=True, relax=(bool, False),
        confirm="Shut down the dashboard? The policy stops and its log is saved, then the dashboard exits "
                "(start it again on the Pi with python3 -m robot.dashboard).")
def shutdown(robot, relax):
    """Stop everything, save the walk log, then exit the dashboard; the servos hold (or relax)."""
    log = robot.shutdown(relax)
    exit_dashboard = getattr(robot, "exit_dashboard", None)
    if exit_dashboard:
        exit_dashboard()   # after this reply is sent
    return {"log": log}


@action("relax", "Relax", "system", joint=(Joint, None))
def relax(robot, joint):
    """Torque off: every servo, or one joint."""
    robot.relax(joint)


@action("scan", "Rescan", "system")
def scan(robot):
    """Find the servos on every BusLinker again."""
    return {"found": robot.scan()}


@action("heartbeat", "Heartbeat", "system")
def heartbeat(robot):
    """The page is still there (the walking policy stands still, then stops, without it)."""
    robot.walker.heartbeat()


@action("boot", "Boot: straight → home", "poses")
def boot(robot):
    """Straight legs over 2 s, then the home crouch over 2.5 s."""
    return robot.boot()


@action("pose", "Pose", "poses", name=str, ms=(int, 2500))
def pose(robot, name, ms):
    """All joints to a named pose: straight or home."""
    return robot.pose({"straight": STRAIGHT, "home": HOME}[name], ms)


@action("move", "Move joint", "calibration", joint=Joint, deg=float, ms=(int, 300))
def move(robot, joint, deg, ms):
    """One joint to an angle (degrees, simulator convention), clamped to its limits."""
    return robot.move(joint, deg, ms)


@action("hold", "Hold joint", "calibration", joint=Joint)
def hold(robot, joint):
    """Torque on where the joint is now."""
    return robot.hold(joint)


# ---------------------------------------------------------------- walking
@action("policy/start", "Start", "drive", name=str,
        confirm="Start the policy? The robot must be at Home, hanging or standing on the floor (harness on). "
                "It starts standing still.")
def policy_start(robot, name):
    """Start a walking policy (robot/policies/<name>.npz)."""
    return robot.walker.start(name)


@action("policy/stop", "Stop (hold)", "drive", danger=True, why=(str, None))
def policy_stop(robot, why):
    """Stop the policy, servos hold; why=pickup logs it as a pickup, not a fall."""
    robot.walker.stop("stopped: picked up" if why == "pickup" else "stopped from the page")


@action("policy/cmd", "Command", "drive", vx=float, vy=(float, 0.0), wz=(float, 0.0), source=(str, "page"))
def policy_cmd(robot, vx, vy, wz, source):
    """The walking command: forward m/s, sideways m/s (+ = left), turn rad/s (+ = left)."""
    return robot.walker.set_command(vx, vy, wz, source=source)


@action("policy/move_ms", "Servo move time", "drive", ms=int)
def policy_move_ms(robot, ms):
    """Servo move time per policy step (0 = as in training)."""
    robot.walker.move_ms = max(0, min(100, ms))


@action("spin", "Victory spin", "personality")
def spin(robot):
    """A full turn in place with the walking policy, then a bounce."""
    return robot.personality.victory_spin()


# ---------------------------------------------------------------- personality and voice
@action("gesture", "Gesture", "personality", name=str, bpm=(float, None))
def gesture(robot, name, bpm):
    """Play a gesture (robot/gestures.py) from home."""
    return robot.personality.play(name, bpm=bpm)


@action("gesture/stop", "Stop gesture", "personality")
def gesture_stop(robot):
    robot.personality.stop()


@action("playful", "Playful mode", "personality", on=bool)
def playful(robot, on):
    """Small idle moves and chirps while standing at home."""
    robot.personality.playful = on
    robot.personality.last_activity = time.monotonic()


@action("faces", "Greet faces", "personality", on=bool)
def faces(robot, on):
    """A happy bounce when the camera sees someone."""
    return robot.personality.set_faces(on)


@action("listen", "Voice commands", "voice", on=bool)
def listen(robot, on):
    """Listen for "Juan, ..." through the microphone."""
    return robot.personality.set_listening(on)


@action("voice", "Voice phrase", "voice", text=str)
def voice(robot, text):
    """A phrase heard by a microphone elsewhere (python -m robot.voice_commands --robot)."""
    from robot.voice_commands import parse
    act = parse(text)
    if act:
        robot.personality._voice_command(act, text)
    return {"action": act} if act else "not a known command"


@action("sound", "Sound", "voice", file=(str, None), emotion=(str, None))
def sound(robot, file, emotion):
    """Play a sound file (robot/sounds/) or an emotion's chirp."""
    v = robot.personality.voice
    if file:
        v.play_file(file)
    elif emotion in v.state()["emotions"]:
        v.say(emotion)
    else:
        raise ValueError("file or emotion")
    return v.error


@action("say", "Say", "voice", text=str)
def say(robot, text):
    """Speak text (needs espeak-ng)."""
    robot.personality.voice.speak(text)
    return robot.personality.voice.error


@action("volume", "Volume", "voice", v=float)
def volume(robot, v):
    robot.personality.voice.volume = max(0.0, min(1.0, v))


# ---------------------------------------------------------------- calibration
@action("cal/zero", "Set zero: legs straight", "calibration",
        confirm="Legs straight and still? Every joint's current position becomes 0°.")
def cal_zero(robot):
    return robot.set_zero()


@action("cal/flip", "Flip", "calibration", joint=Joint, confirm="Flip this joint's direction?")
def cal_flip(robot, joint):
    robot.flip(joint)


@action("cal/limit", "Limit here", "calibration", joint=Joint, end=str)
def cal_limit(robot, joint, end):
    """This joint's lower (end=lo) or upper (end=hi) limit at its current position."""
    if end not in ("lo", "hi"):
        raise ValueError("end must be lo or hi")
    return robot.set_limit(joint, end)


@action("cal/clear_limits", "Clear limits", "calibration", joint=Joint)
def cal_clear_limits(robot, joint):
    robot.clear_limits(joint)


@action("cal/write_limits", "Write limits to servos", "calibration", reset=(bool, False),
        confirm="Write the limits into every servo's memory? The servos then refuse targets outside them, "
                "whatever program sends them.")
def cal_write_limits(robot, reset):
    return {"written": robot.write_servo_limits(reset=reset)}


@action("cal/imu", "IMU axes and bus", "calibration", axes=(str, None), bus=(int, None))
def cal_imu(robot, axes, bus):
    """Torso axes from sensor axes (e.g. -y,x,z) and the I2C bus."""
    if axes is not None:
        axes = axes.split(",")
        if len(axes) != 3 or sorted(a[-1] for a in axes) != ["x", "y", "z"]:
            return "axes must use x, y and z once each, e.g. -y,x,z"
    robot.set_imu(axes, bus)


@action("cal/imu_level", "Level here", "calibration")
def cal_imu_level(robot):
    """The lean shown on the page is relative to now (display only)."""
    return robot.level_imu()


@action("cal/zero_gyro", "Zero gyro", "calibration")
def cal_zero_gyro(robot):
    """Measure the gyro's bias over 2 s (robot still)."""
    robot.imu.zero_gyro()


def run(robot, name, raw):
    """(HTTP status, JSON reply) for an action by name with raw parameters."""
    a = ACTIONS.get(name)
    if a is None:
        return 404, {"error": f"no action {name}"}
    try:
        out = a.fn(robot, **a.parse(raw))
    except (KeyError, ValueError) as e:
        return 400, {"error": f"bad or missing parameter: {e}"}
    if isinstance(out, dict):
        return 200, {"error": None, **out}
    return (409 if out else 200), {"error": out}
