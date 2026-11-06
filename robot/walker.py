"""Runs the walking policy on the robot at 50 Hz, inside the dashboard (robot.dashboard.Robot):
latest servo positions and IMU estimate in, joint targets out, clamped to the calibrated limits.

Safety, checked every tick:
  * tilt over TILT_RELAX_DEG: the robot is falling: policy off and every servo relaxed;
  * stale sensors (a servo or the IMU silent for STALE_S): policy off, servos hold their targets;
  * lost connection (no page or joystick for LINK_STAND_S): command set to "stand still";
    after LINK_STOP_S: policy off, servos hold. (Long enough to ride out a backgrounded tab or a
    Wi-Fi gap.)
Commands come from a source (the page, a joystick, voice, a behavior such as the victory spin); a
behavior can take the steering (steer_owner) so nothing else changes the command until it is done.
Every run is logged to robot/logs/ (observations, actions, targets, joints, IMU, timing, whether the
camera and face detection were running) for comparing with the simulator.
"""
import json
import math
import os
import threading
import time

import numpy as np

from robot.imu import tilt_deg, tilted
from robot.joints import BY_NAME, COUNTS_PER_DEG, NAMES
from robot.policy import Controller, Policy

TILT_RELAX_DEG = 40
STALE_S = 0.1
LINK_STAND_S = 2.0
LINK_STOP_S = 6.0
START_TOLERANCE_DEG = 10   # every joint within this of home to start
START_TILT_DEG = 20        # and the torso within this of upright
GETUP_TIMEOUT_S = 10.0     # getting up (start_getup): not up by then -> stop and relax
UP_TILT_DEG, UP_RATE, UP_HOLD_S = 15.0, 1.0, 0.5   # "up": torso within 15 deg, turning under 1 rad/s, for 0.5 s
HERE = os.path.dirname(os.path.abspath(__file__))
POLICY_DIR = os.path.join(HERE, "policies")
LOG_DIR = os.path.join(HERE, "logs")
# The Pi's health, logged every tick (the dashboard refreshes it once a second; NaN if unknown): CPU %
# (all cores, and this program as % of one core), load, memory, CPU and GPU clocks, temperature, Wi-Fi
# signal, power flags (1 = undervoltage / throttled right now), the slowest servo-bus pass of the last
# few, and the IMU's failed reads so far.
PI_FIELDS = ("cpu", "self_cpu", "load", "mem_used_mb", "arm_mhz", "gpu_mhz", "temp", "wifi_dbm", "undervolt",
             "throttled", "bus_ms", "imu_failures")


WALK_ROLES = ("walk", "speed")   # the walking policies' roles, in the page's order; plus "getup"


def roles():
    """robot/policies/roles.json: the one policy per role the page offers ({"walk": name, "speed": name or null,
    "getup": name}; scripts/policies.py sets it). Every other exported policy stays in the folder, unlisted (the
    library). None if there is no roles file: then every policy is listed, as before."""
    try:
        with open(os.path.join(POLICY_DIR, "roles.json")) as f:
            r = json.load(f)
        return r if isinstance(r, dict) else None
    except (OSError, ValueError):
        return None


def _exported(sub=""):
    d = os.path.join(POLICY_DIR, sub)
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".npz")) if os.path.isdir(d) else []


def policies():
    """The walking policies the page offers: the roles' (walk, then speed), or every exported one without roles.json."""
    r, have = roles(), _exported()
    if r is None:
        return have
    return [n for n in dict.fromkeys(r.get(k) for k in WALK_ROLES) if n in have]


def getup_policies():
    """The getup policies (robot/policies/getup/, scripts/export_getup.py): never offered as walking policies."""
    r, have = roles(), _exported("getup")
    return have if r is None else [n for n in [r.get("getup")] if n in have]


def policy_labels():
    """{policy name: its role} for the page's lists."""
    return {n: k for k, n in (roles() or {}).items() if n}


def default_policy():
    """The policy preselected on the page: the walk role (or, without roles.json, robot/policies/default.txt)."""
    r = roles()
    if r is not None:
        return r.get("walk") if r.get("walk") in policies() else None
    try:
        with open(os.path.join(POLICY_DIR, "default.txt")) as f:
            name = f.read().strip()
        return name if name in policies() else None
    except OSError:
        return None


class Walker:
    def __init__(self, robot):
        self.robot = robot
        self.running, self.thread = False, None
        self.status, self.policy_name = "off", None
        self.ctrl = None
        self.move_ms = 0              # servo move time per tick (0 = full speed, as in the simulator)
        self.last_link = time.monotonic()
        self.stats = {}
        self.listener = None          # called with "start", "stop", "fall" or "lost" (personality sounds)
        self.command_source = None    # who sent the current command
        self.steer_owner = None       # a behavior steering: other sources are refused until it lets go
        self.log_dir = LOG_DIR
        self.phase = None             # "getup" (start_getup: the getup policy, until up) or "walk"
        self.next_policy = None       # the walking policy that takes over once up

    # ---- from the page ----
    def heartbeat(self):
        self.last_link = time.monotonic()

    def set_command(self, vx, vy, wz, source="page"):
        if self.ctrl is None:
            return "policy not running"
        if self.phase == "getup":
            return "getting up: commands once it is up"
        if self.steer_owner and source != self.steer_owner:
            return f"{self.steer_owner} is steering: stop it first"
        self.command_source = source
        m = self.ctrl.policy.meta
        clip = lambda v, r: float(min(max(v, float(r[0])), float(r[1])))
        self.ctrl.set_command(clip(vx, m["forward_range"]), clip(vy, m["lateral_range"]), clip(wz, m["yaw_rate_range"]))
        self.heartbeat()
        return None

    def start(self, name):
        if self.running:
            return "already running"
        r = self.robot
        if not r.cal.zero_set:
            return "not calibrated: set zero first"
        pers = r.personality
        if pers.busy() and pers.idle_playing():
            pers.stop()  # a playful-mode idle animation: the walk takes over
            for _ in range(25):
                if not pers.busy():
                    break
                time.sleep(0.02)
        if pers.busy():
            return "a gesture is playing"
        if any(BY_NAME[n][1] not in r.bus_of for n in NAMES):
            return "not all 8 servos found"
        if any(r.locked(BY_NAME[n][1]) for n in NAMES):
            return "a servo is locked (too hot)"
        try:
            policy = Policy(os.path.join(POLICY_DIR, name + ".npz"))
        except (OSError, KeyError, ValueError) as e:
            return f"can't load {name}: {e}"
        sensors, why = self._sensors()
        if sensors is None:
            return why
        lean = math.hypot(*tilt_deg(sensors["gravity_raw"]))
        if lean > START_TILT_DEG:
            return f"not upright (tilted {lean:.0f} deg)"
        home = policy.meta["home"]
        far = [n for n, q, h in zip(NAMES, sensors["q"], home) if abs(math.degrees(q - h)) > START_TOLERANCE_DEG]
        if far:
            pose = "Straight" if not np.any(home) else "Home"
            return f"not at its start pose ({', '.join(far)}): press {pose} first"
        self.ctrl = Controller(policy)
        self.ctrl.start(sensors)
        if str(policy.meta.get("task", "walk")) == "getup":
            return f"{name} is a getup policy: use Get up"
        self.policy_name, self.running, self.status = name, True, "running"
        self.command_source, self.steer_owner = None, None
        self.phase, self.next_policy = "walk", None
        self.heartbeat()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        if self.listener:
            self.listener("start")
        return None

    def start_getup(self, getup_name, walker_name):
        """From lying (placed face down): the getup policy until the robot is up (UP_*), then the walking
        policy, commanded to stand. No home or upright check to start, and no tilt stop while getting up;
        not up within GETUP_TIMEOUT_S -> stop and relax."""
        if self.running:
            return "already running"
        r = self.robot
        if not r.cal.zero_set:
            return "not calibrated: set zero first"
        if r.personality.busy():
            return "a gesture is playing"
        if any(BY_NAME[n][1] not in r.bus_of for n in NAMES):
            return "not all 8 servos found"
        if any(r.locked(BY_NAME[n][1]) for n in NAMES):
            return "a servo is locked (too hot)"
        try:
            getup = Policy(os.path.join(POLICY_DIR, "getup", getup_name + ".npz"))
            walker = Policy(os.path.join(POLICY_DIR, walker_name + ".npz"))
        except (OSError, KeyError, ValueError) as e:
            return f"can't load: {e}"
        if str(walker.meta.get("task", "walk")) == "getup":
            return f"{walker_name} is a getup policy, not a walking one"
        sensors, why = self._sensors()
        if sensors is None:
            return why
        self.ctrl = Controller(getup)
        self.ctrl.start(sensors)
        self.phase, self.next_policy = "getup", walker
        self.getup_deadline = time.monotonic() + GETUP_TIMEOUT_S
        self.up_since = None
        self.policy_name, self.running, self.status = f"{getup_name}+{walker_name}", True, "getting up"
        self.command_source, self.steer_owner = None, "getup"
        self.heartbeat()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        if self.listener:
            self.listener("start")
        return None

    def _getup_step(self, sensors):
        """While getting up: hand over to the walking policy once up; give up after GETUP_TIMEOUT_S."""
        right, fwd = tilt_deg(sensors["gravity_raw"])
        upright = math.hypot(right, fwd) < UP_TILT_DEG and math.hypot(*sensors["gyro"][:2]) < UP_RATE
        now = time.monotonic()
        self.up_since = (self.up_since or now) if upright else None
        if self.up_since is not None and now - self.up_since >= UP_HOLD_S:
            self.ctrl = Controller(self.next_policy)
            self.ctrl.start(sensors)
            self.phase, self.status, self.steer_owner = "walk", "up: walking policy standing", None
            return True
        if now > self.getup_deadline:
            self.stop("stopped: not up within the getup time", relax=True)
            return False
        return True

    def stop(self, why="stopped", relax=False):
        was = self.running
        self.running = False
        self.status = why
        if relax:
            self.robot.relax()
        if was and self.listener:
            self.listener("fall" if "tilted" in why else "lost" if "connection" in why else "stop")

    # ---- the loop ----
    def _sensors(self):
        """Latest joint angles (rad) and IMU, or (None, reason) if anything is stale."""
        r, now = self.robot, time.time()
        q, qd = [], []
        for n in NAMES:
            s, jc = r.servo.get(BY_NAME[n][1], {}), r.cal.joints[n]
            if s.get("pos") is None or now - s.get("t", 0) > STALE_S:
                return None, f"no fresh reading from {n}"
            q.append(math.radians(jc.degrees(s["pos"])))
            qd.append(math.radians(s.get("vel", 0.0) * jc.sign / COUNTS_PER_DEG))  # measured on the bus thread
        imu = r.imu.state
        if imu is None or now - imu["t"] > STALE_S:
            return None, "no fresh IMU reading"
        gravity = imu["gravity"]
        if any(r.cal.imu_offset):
            gravity = tilted(gravity, *r.cal.imu_offset)
        return {"q": np.array(q), "qd": np.array(qd), "gravity": np.array(gravity), "gyro": np.array(imu["gyro"]),
                "gravity_raw": np.array(imu["gravity"]), "yaw": imu["yaw"], "t": now}, None

    def _slow(self, key):
        """Each servo's latest supply voltage (V) or temperature (C), NaN until read: the bus thread reads
        one servo's every few passes, so a value can be up to ~2 s old."""
        return [np.nan if (v := self.robot.servo.get(BY_NAME[n][1], {}).get(key)) is None else v for n in NAMES]

    def _pi(self):
        r = self.robot
        pi = dict(getattr(r, "pi", None) or {})
        stats = getattr(r, "bus_stats", {})
        passes = [p[0] for st in stats.values() for p in list(st)[-10:]]
        pi["bus_ms"] = max(passes) if passes else None
        pi["imu_failures"] = getattr(getattr(r, "imu", None), "failures", None)
        return [np.nan if pi.get(k) is None else float(pi[k]) for k in PI_FIELDS]

    def _vision(self):
        """(camera streamed in the last 2 s, face detection on): what else was using the Pi."""
        r = self.robot
        cam = getattr(r, "camera", None)
        seen = getattr(cam, "last_view", 0.0)
        return [float(bool(seen) and time.monotonic() - seen < 2.0),
                float(getattr(getattr(r, "personality", None), "faces", None) is not None)]

    def _send(self, targets_rad):
        r, per_bus = self.robot, {}
        for n, rad in zip(NAMES, targets_rad):
            sid = BY_NAME[n][1]
            counts = r.cal.joints[n].clamp_counts(n, math.degrees(rad))
            per_bus.setdefault(r.bus_of[sid], {})[sid] = counts
            r.servo[sid]["target"], r.servo[sid]["torque"] = counts, True
        for bus, targets in per_bus.items():
            bus.move_many(targets, self.move_ms)

    def _loop(self):
        dt = self.ctrl.dt
        log = {k: [] for k in ("t", "obs", "action", "target", "q", "qd", "gravity", "gyro", "yaw", "command", "tick_ms",
                               "link_s", "vin", "temp", "pi", "gravity_raw", "vision")}
        link_max = 0.0
        stale, overruns, ticks = 0, 0, 0
        next_t = time.monotonic()
        t0 = time.time()
        while self.running:
            start = time.perf_counter()
            sensors, why = self._sensors()
            if sensors is None:
                stale += 1
                if stale >= 3:
                    self.stop(f"stopped: {why}")
                    break
            else:
                stale = 0
                right, fwd = tilt_deg(sensors["gravity_raw"])  # the real tilt, whatever the offset
                if self.phase == "getup":
                    if not self._getup_step(sensors):
                        break
                elif math.hypot(right, fwd) > TILT_RELAX_DEG:
                    self.stop(f"stopped: tilted {math.hypot(right, fwd):.0f} deg (falling): relaxed", relax=True)
                    break
                quiet = time.monotonic() - self.last_link
                link_max = max(link_max, quiet)
                if quiet > LINK_STOP_S:
                    self.stop("stopped: lost the connection")
                    break
                if quiet > LINK_STAND_S and np.any(self.ctrl.command):
                    self.ctrl.set_command(0.0, 0.0, 0.0)  # connection lost: stand still
                targets = self.ctrl.tick(sensors)
                self._send(targets)
                ticks += 1
                for k, v in (("t", sensors["t"] - t0), ("obs", self.ctrl.last_obs), ("action", self.ctrl.last_action),
                             ("target", targets), ("q", sensors["q"]), ("qd", sensors["qd"]),
                             ("gravity", sensors["gravity"]), ("gyro", sensors["gyro"]), ("yaw", sensors["yaw"]),
                             ("command", self.ctrl.command.copy()), ("link_s", quiet),
                             ("vin", self._slow("vin")), ("temp", self._slow("temp")), ("pi", self._pi()),
                             ("gravity_raw", sensors["gravity_raw"]), ("vision", self._vision())):
                    log[k].append(np.array(v, dtype=float))
            took = (time.perf_counter() - start) * 1000
            log["tick_ms"].append(took)
            next_t += dt
            late = time.monotonic() - next_t
            if late > 0:
                overruns += 1
                if late > 0.1:
                    next_t = time.monotonic()  # far behind: start counting again
            else:
                time.sleep(-late)
            recent = log["tick_ms"][-50:]
            self.stats = {"ticks": ticks, "overruns": overruns, "tick_ms": sum(recent) / len(recent),
                          "link_s": time.monotonic() - self.last_link, "link_max": link_max,
                          "tick_max": max(recent), "command": self.ctrl.command.tolist(), "seconds": ticks * dt}
        self.running = False
        self._save(log)

    def _save(self, log):
        if not log["t"]:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d_%H%M%S") + f"{time.time() % 1:.3f}"[1:]
        path = os.path.join(self.log_dir, f"{stamp}_{self.policy_name}.npz")
        np.savez_compressed(path, joints=np.array(NAMES), status=self.status, move_ms=self.move_ms,
                            pi_fields=np.array(PI_FIELDS), imu_offset=np.array(self.robot.cal.imu_offset),
                            **{k: np.array(v) for k, v in log.items()})
        self.stats["log"] = os.path.relpath(path, os.path.dirname(HERE))

    def state(self):
        return {"running": self.running, "status": self.status, "policy": self.policy_name,
                "link_s": time.monotonic() - self.last_link, "link_limits": [LINK_STAND_S, LINK_STOP_S],
                "policies": policies(), "default": default_policy(), "move_ms": self.move_ms, **self.stats,
                "command_source": self.command_source, "steer_owner": self.steer_owner,
                "phase": self.phase, "getup_policies": getup_policies(), "labels": policy_labels(),
                "ranges": None if self.ctrl is None else {k: np.asarray(self.ctrl.policy.meta[k]).tolist()
                                                          for k in ("forward_range", "lateral_range", "yaw_rate_range")}}
