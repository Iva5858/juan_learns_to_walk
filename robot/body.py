"""Juan's body: the servo buses, the IMU, the camera and the Pi's health, read continuously by background
threads (one per servo bus, one for the IMU, one for the Pi), with the safety rules (temperature locks,
every move clamped to the joint's limits: the physical limits once set, the simulator's range until
then), calibration, and the walking policy and personality that drive the servos. The web page
(robot/dashboard.py) only reads the latest values and calls into it, so a slow browser never slows the
buses.

The hardware is pluggable: real_hardware() on the Pi, robot.fake.hardware() anywhere else.
"""
import math
import threading
import time
from collections import deque
from types import SimpleNamespace

from robot.imu import tilt_deg
from robot.joints import BY_ID, BY_NAME, CAL_FILE, HOME, NAMES, SQUAT, STRAIGHT, Calibration
from robot.personality import Personality
from robot.walker import Walker

SERVO_TEMP_LIMIT = 60   # deg C: a servo this hot is relaxed and locked
PI_TEMP_LIMIT = 75      # deg C: the Pi this hot relaxes every servo
COOL_MARGIN = 5         # unlock once this many degrees below the limit
SLOW_EVERY = 8          # bus passes between temperature / voltage / torque reads (one servo per pass)
BOOT_MS = (2000, 2500)  # boot: to straight legs, then to home
SHUTDOWN_MS = (1000, 2500, 500)  # shutdown: to home, down into the squat, settle there; then relax
SHUTDOWN_TILT_DEG = 30           # tilted more (fallen): no squat, relax at once


def real_hardware():
    """The robot's own parts (imported here: pyserial, smbus2 and picamera2 exist only on the Pi)."""
    from robot import hiwonder, pi_health
    from robot.camera import Camera
    from robot.imu import IMU
    return SimpleNamespace(ports=hiwonder.ports, Bus=hiwonder.Bus, IMU=IMU, Camera=Camera, pi=pi_health,
                           broadcast=hiwonder.BROADCAST)


class Robot:
    def __init__(self, hw=None, cal_path=CAL_FILE):
        self.hw = hw or real_hardware()
        self.cal_path = cal_path
        self.cal = Calibration.load(cal_path)
        self.buses = []        # hiwonder.Bus, one per BusLinker with servos on it
        self.bus_of = {}       # servo ID -> Bus
        self.servo = {}        # servo ID -> latest readings
        self.info = {}         # servo ID -> settings read at scan time
        self.bus_stats = {}    # port -> recent pass times
        self.pi = {}
        self.hot, self.pi_hot = {}, False
        self.message = ""
        self.booting = False
        self.camera = self.hw.Camera()
        self.walker = Walker(self)
        self.personality = Personality(self)
        self.walker.listener = self.personality.walker_event
        if getattr(self.hw, "world", None):  # the fake IMU turns with the policy's turn command
            self.hw.world.turn_rate = lambda: float(self.walker.ctrl.command[2]) if self.walker.running else 0.0
        self.scan()
        self.imu = self.hw.IMU(self.cal.imu_bus, self.cal.imu_axes)
        self.camera.configure_stabilization(self.imu)
        threading.Thread(target=self._pi_loop, daemon=True).start()

    # ---------------------------------------------------------------- servos
    def scan(self):
        """Open every BusLinker and find which servo IDs answer on it."""
        if self.walker.running:
            self.walker.stop("stopped: rescan")
        for bus in self.buses:
            bus.closed = True
            bus.close()
        self.buses, self.bus_of, self.servo, self.info = [], {}, {}, {}
        for port in self.hw.ports():
            try:
                bus = self.hw.Bus(port)
            except Exception as e:
                self.message = f"Can't open {port}: {e}"
                continue
            ids = [sid for sid in BY_ID if sid not in self.bus_of and bus.ping(sid)]
            if not ids:
                bus.close()
                continue
            bus.ids, bus.closed = ids, False
            self.buses.append(bus)
            for sid in ids:
                self.bus_of[sid] = bus
                self.info[sid] = bus.info(sid)
                self.servo[sid] = {"pos": None, "temp": None, "vin": None, "torque": None, "t": 0.0}
            self.bus_stats[port] = deque(maxlen=200)
            threading.Thread(target=self._bus_loop, args=(bus,), daemon=True).start()
        return sorted(self.bus_of)

    def _bus_loop(self, bus):
        """Positions of every servo on this bus each pass; one slow read (temperature, voltage or
        torque) per pass, round-robin. The pass time is what a control loop would see."""
        n = 0
        while not bus.closed:
            t0 = time.perf_counter()
            fails = 0
            for sid in bus.ids:
                try:
                    pos = bus.position(sid)
                except Exception:
                    pos = None
                if pos is None:
                    fails += 1
                else:
                    # Speed from this read and the previous one, at their own times: the control
                    # loop runs on another clock, so differencing at its ticks would alias.
                    s, now = self.servo[sid], time.time()
                    if s["pos"] is not None and now - s["t"] < 0.1:
                        s["vel"] = (pos - s["pos"]) / (now - s["t"])  # counts/s
                    else:
                        s["vel"] = 0.0
                    s.update(pos=pos, t=now)
            self.bus_stats[bus.port].append(((time.perf_counter() - t0) * 1000, fails, len(bus.ids)))
            n += 1
            if n % SLOW_EVERY == 0:
                sid = bus.ids[(n // SLOW_EVERY) % len(bus.ids)]
                try:
                    s = self.servo[sid]
                    s["temp"], s["vin"], s["torque"] = bus.temperature(sid), bus.voltage(sid), bus.torque_on(sid)
                    self._check_temp(sid, s["temp"])
                except Exception:
                    pass
            time.sleep(0.002)  # room for commands (the control loop's writes, the page)

    def _check_temp(self, sid, temp):
        if temp is None:
            return
        if temp >= SERVO_TEMP_LIMIT or (sid in self.hot and temp >= SERVO_TEMP_LIMIT - COOL_MARGIN):
            if sid not in self.hot:
                print(f"[protect] servo {sid} at {temp} C: relaxed and locked")
            self.hot[sid] = temp
            self.bus_of[sid].set_torque(sid, False)
        elif sid in self.hot:
            del self.hot[sid]
            print(f"[protect] servo {sid} cooled to {temp} C: unlocked")

    def locked(self, sid):
        return self.pi_hot or sid in self.hot

    def all_buses(self, fn):
        for bus in self.buses:
            fn(bus)

    def relax(self, name=None):
        self.personality.stop()
        if self.walker.running:
            self.walker.stop("stopped: relaxed from the page")
        if name is None:
            self.all_buses(lambda b: b.set_torque(self.hw.broadcast, False))
            sids = list(self.servo)
        else:
            sids = [BY_NAME[name][1]] if BY_NAME[name][1] in self.bus_of else []
            for sid in sids:
                self.bus_of[sid].set_torque(sid, False)
        for sid in sids:  # show it now; the bus loop re-reads it within a second
            self.servo[sid]["torque"] = False

    def stop(self):
        self.personality.stop()
        self.walker.stop("stopped from the page")
        self.all_buses(lambda b: b.stop())

    def shutdown(self):
        """The stop sequence before the dashboard exits: the walking policy stops and its log is written
        (its loop saves it on the way out: waited for), gestures stop, face greeting and voice listening
        close; the robot goes to home and down into the deep squat (SQUAT), then every servo relaxes, always:
        also when the squat can't be done (fallen, a servo missing, locked or uncalibrated) or fails. Returns
        the walk log saved, if any."""
        was_walking = self.walker.running
        try:
            self.personality.stop()
            self.walker.stop("stopped: dashboard shut down")
            if self.walker.thread is not None:
                self.walker.thread.join(timeout=5.0)
            self.personality.set_faces(False)
            self.personality.set_listening(False)
            why = self._why_no_squat()
            if why is None:
                self.message = "Shutting down: squatting"
                for pose, ms in ((HOME, SHUTDOWN_MS[0]), (SQUAT, SHUTDOWN_MS[1])):
                    why = self.pose(pose, ms)
                    if why:
                        break
                    time.sleep(ms / 1000)
                else:
                    time.sleep(SHUTDOWN_MS[2] / 1000)
        finally:
            self.relax()
        self.message = "Dashboard shut down: squatted, servos relaxed" if why is None else \
            f"Dashboard shut down: servos relaxed (no squat: {why})"
        return self.walker.stats.get("log") if was_walking else None

    def _why_no_squat(self):
        if not self.cal.zero_set:
            return "not calibrated"
        if any(BY_NAME[n][1] not in self.bus_of for n in NAMES):
            return "not all 8 servos found"
        if any(self.locked(BY_NAME[n][1]) for n in NAMES):
            return "a servo is locked (too hot)"
        st = self.imu.state
        if st is None or time.time() - st["t"] > 0.5:
            return "no IMU reading"
        lean = math.hypot(*(a - b for a, b in zip(tilt_deg(st["gravity"]), self.cal.imu_level)))
        if lean > SHUTDOWN_TILT_DEG:
            return f"tilted {lean:.0f} deg"
        return None

    def busy(self):
        """Why the servos can't be moved from the page right now (None: they can)."""
        if self.walker.running:
            return "the policy is running: stop it first"
        if self.personality.busy():
            return "a gesture is playing"
        if self.booting:
            return "booting: wait for home"
        return None

    def mode(self):
        """Who has the servos: walking, gesture, boot, or idle (the page's buttons follow this)."""
        return ("walking" if self.walker.running else "gesture" if self.personality.busy()
                else "boot" if self.booting else "idle")

    def move(self, name, deg, ms=300, _boot=False):
        """Move one joint to an angle (simulator convention, degrees), clamped to its limits."""
        sid = BY_NAME[name][1]
        if self.busy() and not (_boot and self.booting):
            return self.busy()
        if sid not in self.bus_of:
            return f"servo {sid} ({name}) not found"
        if self.locked(sid):
            return f"servo {sid} is locked (too hot)"
        counts = self.cal.joints[name].clamp_counts(name, deg)
        self.bus_of[sid].move(sid, counts, ms)
        self.servo[sid]["torque"] = True
        self.servo[sid]["target"] = counts
        return None

    def hold(self, name):
        """Torque on at the current position, so the joint doesn't jump."""
        sid = BY_NAME[name][1]
        pos = self.servo.get(sid, {}).get("pos")
        if pos is None:
            return "no position reading"
        return self.move(name, self.cal.joints[name].degrees(pos), 0)

    def pose(self, angles, ms, _boot=False):
        errors = dict.fromkeys(e for e in (self.move(n, angles[n], ms, _boot) for n in NAMES) if e)
        return "; ".join(errors) or None

    def boot(self):
        """Straight legs, then the home crouch, as on the robot at power-up (feet stay flat: the
        home angles sum to 0 on each leg)."""
        if self.busy():
            return self.busy()

        def run():
            try:
                self.message = "Boot: straightening the legs"
                self.pose(STRAIGHT, BOOT_MS[0], _boot=True)
                time.sleep(BOOT_MS[0] / 1000 + 0.5)
                self.message = "Boot: crouching to home"
                self.pose(HOME, BOOT_MS[1], _boot=True)
                time.sleep(BOOT_MS[1] / 1000)
                self.message = "Boot done: holding home"
            finally:
                self.booting = False
        self.booting = True
        threading.Thread(target=run, daemon=True).start()
        return None

    # ---------------------------------------------------------------- calibration
    def set_zero(self):
        """Legs straight right now: every joint's current count becomes its angle 0."""
        missing = []
        for name in NAMES:
            pos = self.servo.get(BY_NAME[name][1], {}).get("pos")
            if pos is None:
                missing.append(name)
            else:
                self.cal.joints[name].zero = pos
        if missing:
            return "no reading from " + ", ".join(missing) + ": nothing saved"
        self.cal.zero_set = True
        self.cal.save(self.cal_path)
        return None

    def flip(self, name):
        self.cal.joints[name].sign *= -1
        self.cal.save(self.cal_path)

    def set_limit(self, name, end):
        pos = self.servo.get(BY_NAME[name][1], {}).get("pos")
        if pos is None:
            return "no position reading"
        setattr(self.cal.joints[name], end, pos)
        self.cal.save(self.cal_path)
        return None

    def clear_limits(self, name):
        jc = self.cal.joints[name]
        jc.lo = jc.hi = None
        self.cal.save(self.cal_path)

    def write_servo_limits(self, reset=False):
        """Store the limits in each servo's memory (reset: back to the full 0-1000)."""
        done = []
        for name in NAMES:
            sid, jc = BY_NAME[name][1], self.cal.joints[name]
            if sid not in self.bus_of or (not reset and (jc.lo is None or jc.hi is None)):
                continue
            lo, hi = (0, 1000) if reset else (min(jc.lo, jc.hi), max(jc.lo, jc.hi))
            bus = self.bus_of[sid]
            bus.write_angle_limits(sid, lo, hi)
            time.sleep(0.02)
            self.info[sid]["angle_limits"] = bus.angle_limits(sid)
            done.append(name)
        return done

    def set_imu(self, axes=None, bus=None):
        if axes is not None:
            self.cal.imu_axes = tuple(axes)
            self.imu.set_axes(axes)
        if bus is not None:
            self.cal.imu_bus = int(bus)
            self.imu.set_bus(bus)  # the IMU thread reconnects on the new bus
        self.cal.save(self.cal_path)

    def level_imu(self):
        """Display only: the lean shown on the page is relative to this. The policy always gets
        the raw gravity estimate, as in the simulator."""
        if not self.imu.state:
            return "no IMU reading"
        self.cal.imu_level = tilt_deg(self.imu.state["gravity"])
        self.cal.save(self.cal_path)
        return None

    # ---------------------------------------------------------------- Pi
    def _pi_loop(self):
        stats = self.hw.pi.SysStats()
        loops, addrs = 0, []
        while True:
            t = self.hw.pi.temperature()
            if t is not None and not self.pi_hot and t >= PI_TEMP_LIMIT:
                self.pi_hot = True
                print(f"[protect] Pi at {t} C: all servos relaxed")
            elif self.pi_hot and (t is None or t < PI_TEMP_LIMIT - COOL_MARGIN):
                self.pi_hot = False
            if self.pi_hot:
                self.relax()
            if loops % 10 == 0:   # addresses change rarely: every 10 s
                addrs = self.hw.pi.addresses()
            self.pi = {"temp": t, **self.hw.pi.power(), **stats.read(), "addresses": addrs}
            loops += 1
            time.sleep(1)

    # ---------------------------------------------------------------- page state
    def state(self, fast=False):
        """Everything the page shows. fast: only what changes quickly (joints, IMU, walking, mode,
        alerts), for the 10 Hz updates; the rest (health, settings, lists) once a second."""
        joints = {}
        for name in NAMES:
            _, sid, sim_lo, sim_hi, home, meaning = BY_NAME[name]
            jc, s = self.cal.joints[name], self.servo.get(sid)
            lo, hi = jc.limits_deg(name)
            j = {"id": sid, "found": s is not None, "meaning": meaning, "sim": [sim_lo, sim_hi], "home": home,
                 "limits": [lo, hi], "limits_set": jc.lo is not None and jc.hi is not None,
                 "zero": jc.zero, "sign": jc.sign, "locked": self.locked(sid)}
            if not fast:
                j["info"] = self.info.get(sid)
            if s is not None:
                j.update({k: s.get(k) for k in ("pos", "temp", "vin", "torque")})
                j["deg"] = None if s["pos"] is None else jc.degrees(s["pos"])
                j["target_deg"] = None if s.get("target") is None else jc.degrees(s["target"])
                j["age"] = time.time() - s["t"] if s["t"] else None
            joints[name] = j
        buses = []
        for bus in self.buses:
            st = list(self.bus_stats[bus.port])
            ms = [x[0] for x in st]
            reads = sum(x[2] for x in st)
            buses.append({"port": bus.port, "ids": bus.ids,
                          "pass_ms": sum(ms) / len(ms) if ms else None, "pass_max": max(ms) if ms else None,
                          "fail_pct": 100 * sum(x[1] for x in st) / reads if reads else None})
        imu, st = None, self.imu.state
        if st:
            right, fwd = tilt_deg(st["gravity"])
            imu = dict(st, lean_right=right - self.cal.imu_level[0], lean_forward=fwd - self.cal.imu_level[1],
                       age=time.time() - st["t"])
        fast_part = {"mode": self.mode(), "joints": joints, "message": self.message, "hot": self.hot,
                     "pi_hot": self.pi_hot, "zero_set": self.cal.zero_set, "imu": imu, "imu_error": self.imu.error,
                     "walker": self.walker.state(), "gesture": self.personality.gesture, "t": time.time()}
        if fast:
            return fast_part
        return {
            **fast_part, "buses": buses, "pi": self.pi,
            "imu_stats": {"bus": self.cal.imu_bus, "axes": self.cal.imu_axes,
                          "fail_pct": 100 * self.imu.failures / self.imu.reads if self.imu.reads else None,
                          "read_ms": self.imu.read_ms},
            "personality": self.personality.state(),
            "camera": {"available": self.camera.available, "error": self.camera.error,
                       "stabilization_configured": self.camera.stabilizer is not None,
                       "stabilization_corrected": self.camera.corrected,
                       "stabilization_error": self.camera.stabilization_error},
            "limits": {"servo_temp": SERVO_TEMP_LIMIT, "pi_temp": PI_TEMP_LIMIT},
        }
