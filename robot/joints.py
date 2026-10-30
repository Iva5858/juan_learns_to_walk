"""Juan's joints on the real robot, in the simulator's conventions, and the saved calibration.

Angles are the simulator's (biped_rl/robots/juan.py): degrees, legs straight = 0, signs:
    hip roll    + = left leg outward, right leg inward
    hip pitch   + = leg swings back
    knee        + = the knee bends (shin swings back)
    ankle       + = toes down
A servo reports counts, 0-1000 over 240 deg. Per joint, the calibration stores the count at angle 0
(legs straight), the sign (which way the counts grow), and the physical limits found on the robot:
    counts = zero + sign * angle * COUNTS_PER_DEG
"""
import json
import os
from dataclasses import asdict, dataclass

COUNTS_PER_DEG = 1000 / 240

# name, servo ID, simulator joint range (deg), home pose (deg), what + means.
# Ranges and home must match juan/rl_robot.xml (checked by scripts/check_robot_joints.py).
JOINTS = (
    ("left_hip_roll", 1, -7.0, 25.0, 0.0, "leg outward"),
    ("left_hip_pitch", 2, -110.0, 68.0, -22.92, "leg back"),
    ("left_knee", 3, -40.0, 114.0, 45.84, "knee bends"),
    ("left_ankle", 4, -74.0, 72.0, -22.92, "toes down"),
    ("right_hip_roll", 5, -25.0, 7.0, 0.0, "leg inward"),
    ("right_hip_pitch", 6, -110.0, 68.0, -22.92, "leg back"),
    ("right_knee", 7, -40.0, 114.0, 45.84, "knee bends"),
    ("right_ankle", 8, -74.0, 72.0, -22.92, "toes down"),
)
NAMES = tuple(j[0] for j in JOINTS)
BY_NAME = {j[0]: j for j in JOINTS}
BY_ID = {j[1]: j for j in JOINTS}
HOME = {j[0]: j[4] for j in JOINTS}
STRAIGHT = {j[0]: 0.0 for j in JOINTS}

CAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")


@dataclass
class JointCal:
    zero: int = 500        # counts at angle 0 (legs straight)
    sign: int = 1          # +1: counts grow with the angle, -1: they shrink
    lo: int = None         # physical limits, in counts (None = not set yet)
    hi: int = None

    def counts(self, deg):
        return self.zero + self.sign * deg * COUNTS_PER_DEG

    def degrees(self, counts):
        return (counts - self.zero) * self.sign / COUNTS_PER_DEG

    def limits_deg(self, name):
        """Allowed range in degrees: the physical limits if set, else the simulator's range."""
        sim_lo, sim_hi = BY_NAME[name][2:4]
        if self.lo is None or self.hi is None:
            return sim_lo, sim_hi
        a, b = self.degrees(self.lo), self.degrees(self.hi)
        return min(a, b), max(a, b)

    def clamp_counts(self, name, deg):
        """Target counts for an angle, clamped to the allowed range and to 0-1000."""
        lo, hi = self.limits_deg(name)
        return int(round(min(1000, max(0, self.counts(min(hi, max(lo, deg)))))))


@dataclass
class Calibration:
    joints: dict = None    # name -> JointCal
    zero_set: bool = False
    imu_bus: int = 1       # I2C bus number (1 = header pins 3/5; i2c-gpio overlays add others)
    imu_axes: tuple = ("x", "y", "z")  # sensor axis giving the torso's x (forward), y (left), z (up)
    imu_level: tuple = (0.0, 0.0)      # roll, pitch (deg) read when the torso stood level
    # (lean right, lean forward) in deg added to the tilt the walking policy sees (not the dashboard or the
    # fall check): a deliberate IMU mounting offset, e.g. to reproduce one a policy walked well with. 0 = off.
    imu_offset: tuple = (0.0, 0.0)

    def __post_init__(self):
        joints = self.joints or {}
        self.joints = {n: JointCal(**joints[n]) if isinstance(joints.get(n), dict) else joints.get(n, JointCal())
                       for n in NAMES}

    def save(self, path=CAL_FILE):
        data = asdict(self)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, path)  # never leave a half-written file

    @classmethod
    def load(cls, path=CAL_FILE):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return cls()
        data["imu_axes"] = tuple(data.get("imu_axes", ("x", "y", "z")))
        data["imu_level"] = tuple(data.get("imu_level", (0.0, 0.0)))
        data["imu_offset"] = tuple(data.get("imu_offset", (0.0, 0.0)))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
