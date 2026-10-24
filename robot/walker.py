"""Runs the walking policy on the robot at 50 Hz, inside the dashboard (robot.dashboard.Robot):
latest servo positions and IMU estimate in, joint targets out, clamped to the calibrated limits.

Safety, checked every tick:
  * tilt over TILT_RELAX_DEG: the robot is falling: policy off and every servo relaxed;
  * stale sensors (a servo or the IMU silent for STALE_S): policy off, servos hold their targets;
  * lost connection (no page or joystick for LINK_STAND_S): command set to "stand still";
    after LINK_STOP_S: policy off, servos hold.
Every run is logged to robot/logs/ (observations, actions, targets, joints, IMU, timing) for
comparing with the simulator.
"""
import math
import os
import threading
import time

import numpy as np

from robot.joints import BY_NAME, COUNTS_PER_DEG, NAMES
from robot.policy import Controller, Policy

TILT_RELAX_DEG = 40
STALE_S = 0.1
LINK_STAND_S = 1.0
LINK_STOP_S = 3.0
START_TOLERANCE_DEG = 10   # every joint within this of home to start
START_TILT_DEG = 20        # and the torso within this of upright
HERE = os.path.dirname(os.path.abspath(__file__))
POLICY_DIR = os.path.join(HERE, "policies")
LOG_DIR = os.path.join(HERE, "logs")


def policies():
    return sorted(f[:-4] for f in os.listdir(POLICY_DIR) if f.endswith(".npz")) if os.path.isdir(POLICY_DIR) else []


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

    # ---- from the page ----
    def heartbeat(self):
        self.last_link = time.monotonic()

    def set_command(self, vx, vy, wz):
        if self.ctrl is None:
            return "policy not running"
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
        pers = getattr(r, "personality", None)
        if pers and pers.busy() and pers.idle_playing():
            pers.stop()  # a playful-mode idle animation: the walk takes over
            for _ in range(25):
                if not pers.busy():
                    break
                time.sleep(0.02)
        if pers and pers.busy():
            return "a gesture is playing"
        if any(BY_NAME[n][1] not in r.bus_of for n in NAMES):
            return "not all 8 servos found"
        if any(r.locked(BY_NAME[n][1]) for n in NAMES):
            return "a servo is locked (too hot)"
        sensors, why = self._sensors()
        if sensors is None:
            return why
        lean = math.hypot(*tilt(sensors["gravity"]))
        if lean > START_TILT_DEG:
            return f"not upright (tilted {lean:.0f} deg)"
        far = [n for n, q, h in zip(NAMES, sensors["q"], self._home(name)) if abs(math.degrees(q - h)) > START_TOLERANCE_DEG]
        if far:
            return "not at home (" + ", ".join(far) + "): press Home first"
        try:
            policy = Policy(os.path.join(POLICY_DIR, name + ".npz"))
        except (OSError, KeyError, ValueError) as e:
            return f"can't load {name}: {e}"
        self.ctrl = Controller(policy)
        self.ctrl.start(sensors)
        self.policy_name, self.running, self.status = name, True, "running"
        self.heartbeat()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        if self.listener:
            self.listener("start")
        return None

    def stop(self, why="stopped", relax=False):
        was = self.running
        self.running = False
        self.status = why
        if relax:
            self.robot.relax()
        if was and self.listener:
            self.listener("fall" if "tilted" in why else "lost" if "connection" in why else "stop")

    # ---- the loop ----
    def _home(self, name):
        try:
            return Policy(os.path.join(POLICY_DIR, name + ".npz")).meta["home"]
        except Exception:
            return np.zeros(len(NAMES))

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
        return {"q": np.array(q), "qd": np.array(qd), "gravity": np.array(imu["gravity"]), "gyro": np.array(imu["gyro"]),
                "yaw": imu["yaw"], "t": now}, None

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
        log = {k: [] for k in ("t", "obs", "action", "target", "q", "qd", "gravity", "gyro", "yaw", "command", "tick_ms")}
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
                right, fwd = tilt(sensors["gravity"])
                if math.hypot(right, fwd) > TILT_RELAX_DEG:
                    self.stop(f"stopped: tilted {math.hypot(right, fwd):.0f} deg (falling): relaxed", relax=True)
                    break
                quiet = time.monotonic() - self.last_link
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
                             ("command", self.ctrl.command.copy())):
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
                          "tick_max": max(recent), "command": self.ctrl.command.tolist(), "seconds": ticks * dt}
        self.running = False
        self._save(log)

    def _save(self, log):
        if not log["t"]:
            return
        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d_%H%M%S") + f"{time.time() % 1:.3f}"[1:]
        path = os.path.join(LOG_DIR, f"{stamp}_{self.policy_name}.npz")
        np.savez_compressed(path, joints=np.array(NAMES), status=self.status, move_ms=self.move_ms,
                            **{k: np.array(v) for k, v in log.items()})
        self.stats["log"] = os.path.relpath(path, os.path.dirname(HERE))

    def state(self):
        return {"running": self.running, "status": self.status, "policy": self.policy_name,
                "policies": policies(), "move_ms": self.move_ms, **self.stats,
                "ranges": None if self.ctrl is None else {k: np.asarray(self.ctrl.policy.meta[k]).tolist()
                                                          for k in ("forward_range", "lateral_range", "yaw_rate_range")}}


def tilt(gravity):
    gx, gy, gz = gravity
    return math.degrees(math.atan2(-gy, -gz)), math.degrees(math.atan2(gx, -gz))
