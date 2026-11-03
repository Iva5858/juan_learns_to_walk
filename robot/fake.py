"""Stand-in hardware for the dashboard on a computer without the robot: python3 -m robot.dashboard --fake

  * servos: two buses (IDs 1-4, 5-8) that answer every read; a held servo glides to its target at the
    real servo's top speed (or over the move time), a relaxed one stays where it is;
  * IMU: upright and still, its heading turning with the walking policy's turn command;
  * camera: a test pattern (with OpenCV) or nothing;
  * Pi: fixed, plausible health numbers.
The robot doesn't fall or walk anywhere: this is for working on the page and the API, not the policy.
"""
import math
import random
import threading
import time

from robot.joints import BY_ID, HOME, NAMES, Calibration
from robot.stabilization import OrientationHistory, sensor_time

SPEED_COUNTS_S = 5.8 * 180 / math.pi * 1000 / 240   # the servo's no-load speed, 5.8 rad/s
PORTS = ("fake0", "fake1")
BUS_IDS = {"fake0": (1, 2, 3, 4), "fake1": (5, 6, 7, 8)}


class World:
    """What every fake part shares: the servos' positions and the policy's turn command."""
    def __init__(self, cal):
        self.lock = threading.Lock()
        self.servos = {}
        for sid, (name, *_rest) in BY_ID.items():
            counts = cal.joints[name].clamp_counts(name, HOME[name])
            self.servos[sid] = {"pos": float(counts), "target": float(counts), "rate": SPEED_COUNTS_S, "torque": False,
                                "t": time.monotonic()}
        self.turn_rate = lambda: 0.0

    def advance(self, sid):
        s, now = self.servos[sid], time.monotonic()
        dt, s["t"] = now - s["t"], now
        if s["torque"]:
            step = s["rate"] * dt
            s["pos"] += max(-step, min(step, s["target"] - s["pos"]))
        return s


class Bus:
    """hiwonder.Bus's interface, on the World."""
    world = None   # set by install()

    def __init__(self, port):
        self.port, self._ids = port, BUS_IDS[port]
        self.lock = threading.RLock()

    def close(self):
        pass

    def _io(self):
        time.sleep(0.0012)   # one query on the real 115200 baud bus

    def ping(self, sid):
        return sid in self._ids

    def position(self, sid):
        self._io()
        with self.world.lock:
            return int(round(self.world.advance(sid)["pos"])) + random.choice((-1, 0, 0, 1))

    def temperature(self, sid):
        return 38 + sid % 3

    def voltage(self, sid):
        return 11.6

    def torque_on(self, sid):
        return self.world.servos[sid]["torque"]

    def target(self, sid):
        return int(self.world.servos[sid]["target"])

    def angle_limits(self, sid):
        return self.world.servos[sid].get("limits", (0, 1000))

    def info(self, sid):
        return {"offset": 0, "angle_limits": self.angle_limits(sid), "vin_limits": (4.5, 14.0), "temp_max": 85}

    def move(self, sid, counts, ms=0):
        self.move_many({sid: counts}, ms)

    def move_many(self, targets, ms=0):
        with self.world.lock:
            for sid, counts in targets.items():
                s = self.world.advance(sid)
                lo, hi = s.get("limits", (0, 1000))
                s["target"] = float(max(lo, min(hi, counts)))
                s["torque"] = True
                s["rate"] = max(SPEED_COUNTS_S if ms <= 0 else min(SPEED_COUNTS_S, abs(s["target"] - s["pos"]) / (ms / 1000)), 1.0)

    def set_torque(self, sid, on):
        with self.world.lock:
            for i in (self._ids if sid == 254 else (sid,)):
                self.world.advance(i)["torque"] = bool(on)

    def stop(self, sid=254):
        with self.world.lock:
            for i in (self._ids if sid == 254 else (sid,)):
                s = self.world.advance(i)
                s["target"] = s["pos"]

    def write_angle_limits(self, sid, lo, hi):
        self.world.servos[sid]["limits"] = (min(lo, hi), max(lo, hi))


def ports():
    return list(PORTS)


class IMU:
    """robot.imu.IMU's interface: upright, a little sensor noise, heading from the turn command."""
    world = None

    def __init__(self, bus=1, axes=("x", "y", "z")):
        self.bus, self.axes = bus, tuple(axes)
        self.error, self.device, self.state = None, "fake IMU", None
        self.reads = self.failures = 0
        self.read_ms = 0.3
        self.orientation = OrientationHistory()
        self.yaw = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    def set_axes(self, axes):
        self.axes = tuple(axes)

    def set_bus(self, bus):
        self.bus = int(bus)

    def zero_gyro(self):
        pass

    def reset(self):
        self.yaw = 0.0

    def _loop(self):
        dt = 0.01
        while True:
            time.sleep(dt)
            wz = self.world.turn_rate() if self.world else 0.0
            self.yaw = (self.yaw + wz * dt + math.pi) % (2 * math.pi) - math.pi
            n = lambda s: random.gauss(0.0, s)
            gyro = [n(0.01), n(0.01), wz + n(0.01)]
            g = [n(0.002), n(0.002), -1.0]
            self.reads += 1
            self.orientation.append(sensor_time(), gyro)
            self.state = {"gravity": g, "gyro": gyro, "yaw": self.yaw, "accel_g": 1.0, "gyro_raw_rms": 0.01,
                          "gyro_lowpass_hz": 4.0, "biasing": False, "gyro_bias": [0.0, 0.0, 0.0], "t": time.time(),
                          "device": self.device}


class Camera:
    """robot.camera.Camera's interface: a moving test pattern if OpenCV is installed."""
    try:
        import cv2 as _cv2
        import numpy as _np
        available = True
    except ImportError:
        available = False

    def __init__(self):
        self.error = None if self.available else "fake camera needs OpenCV (pip install opencv-python)"
        self.stabilizer, self.stabilization_error, self.corrected = None, None, False

    def configure_stabilization(self, imu, path=None):
        pass

    def stream(self, stabilized=False):
        if not self.available:
            return
        cv2, np = self._cv2, self._np
        t0 = time.monotonic()
        while True:
            t = time.monotonic() - t0
            img = np.full((480, 640, 3), 40, np.uint8)
            x = int(320 + 200 * math.sin(t))
            cv2.circle(img, (x, 240), 40, (0, 180, 255), -1)
            cv2.putText(img, f"fake camera {t:5.1f} s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 230, 230), 2)
            yield cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
            time.sleep(1 / 15)


class PiHealth:
    """robot.pi_health's interface."""
    @staticmethod
    def temperature():
        return 47.0

    @staticmethod
    def addresses():
        return ["127.0.0.1"]

    @staticmethod
    def power():
        return {"undervolt": False, "throttled": False, "undervolt_before": False}

    class SysStats:
        def read(self):
            return {"cpu": 12.0, "self_cpu": 30.0, "load": 0.9, "mem_used_mb": 900, "mem_total_mb": 3800,
                    "arm_mhz": 800.0, "gpu_mhz": 500.0, "gpu_mem_mb": 76, "wifi_dbm": -55.0, "cores": [10, 15, 8, 12]}


def hardware(cal_path):
    """The fake parts for robot.body.Robot, sharing one World, and their calibration file (a copy, so
    calibrating the fake robot never touches the real robot's calibration)."""
    from types import SimpleNamespace
    cal = Calibration.load(cal_path)
    if not cal.zero_set:  # a fresh fake robot: zero at 500, ready to drive
        cal.zero_set = True
        for name in NAMES:
            cal.joints[name].zero = 500
    cal.save(cal_path)
    world = World(cal)
    Bus.world = IMU.world = world
    return SimpleNamespace(ports=ports, Bus=Bus, IMU=IMU, Camera=Camera, pi=PiHealth, world=world, broadcast=254)
