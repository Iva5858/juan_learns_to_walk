#!/usr/bin/env python3
"""Juan's robot dashboard: all 8 servos, the IMU, the camera, battery and Pi health, plus
calibration (zero, directions), physical joint limits, the boot pose, and the walking policy
(robot/walker.py) steered from the page.

Run on the Pi, from the project folder:   python3 -m robot.dashboard
Then open                                  http://<pi-name>.local:8000

Background threads read the hardware continuously (one per servo bus, one for the IMU, one for the
Pi); the page only shows the latest values, so a slow browser never slows the buses. Every move is
clamped to the joint's limits: the physical limits once set, the simulator's range until then.
"""
import json
import os
import socket
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from robot import hiwonder
from robot.camera import Camera
from robot.imu import IMU, tilt_deg
from robot.joints import BY_ID, BY_NAME, HOME, NAMES, STRAIGHT, Calibration
from robot.walker import Walker

PORT = 8000
SERVO_TEMP_LIMIT = 60   # deg C: a servo this hot is relaxed and locked
PI_TEMP_LIMIT = 75      # deg C: the Pi this hot relaxes every servo
COOL_MARGIN = 5         # unlock once this many degrees below the limit
SLOW_EVERY = 8          # bus passes between temperature / voltage / torque reads (one servo per pass)
BOOT_MS = (2000, 2500)  # boot: to straight legs, then to home
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")


class Robot:
    def __init__(self):
        self.cal = Calibration.load()
        self.buses = []        # hiwonder.Bus, one per BusLinker with servos on it
        self.bus_of = {}       # servo ID -> Bus
        self.servo = {}        # servo ID -> latest readings
        self.info = {}         # servo ID -> settings read at scan time
        self.bus_stats = {}    # port -> recent pass times
        self.pi = {}
        self.hot, self.pi_hot = {}, False
        self.message = ""
        self.camera = Camera()
        self.walker = Walker(self)
        self.scan()
        self.imu = IMU(self.cal.imu_bus, self.cal.imu_axes)
        threading.Thread(target=self._pi_loop, daemon=True).start()

    # ---------------------------------------------------------------- servos
    def scan(self):
        """Open every BusLinker and find which servo IDs answer on it."""
        self.walker.stop("stopped: rescan")
        for bus in self.buses:
            bus.closed = True
            bus.close()
        self.buses, self.bus_of, self.servo, self.info = [], {}, {}, {}
        for port in hiwonder.ports():
            try:
                bus = hiwonder.Bus(port)
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
        if self.walker.running:
            self.walker.stop("stopped: relaxed from the page")
        if name is None:
            self.all_buses(lambda b: b.set_torque(hiwonder.BROADCAST, False))
            sids = list(self.servo)
        else:
            sids = [BY_NAME[name][1]] if BY_NAME[name][1] in self.bus_of else []
            for sid in sids:
                self.bus_of[sid].set_torque(sid, False)
        for sid in sids:  # show it now; the bus loop re-reads it within a second
            self.servo[sid]["torque"] = False

    def stop(self):
        self.walker.stop("stopped from the page")
        self.all_buses(lambda b: b.stop())

    def move(self, name, deg, ms=300):
        """Move one joint to an angle (simulator convention, degrees), clamped to its limits."""
        sid = BY_NAME[name][1]
        if self.walker.running:
            return "the policy is running: stop it first"
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

    def pose(self, angles, ms):
        errors = [e for e in (self.move(n, angles[n], ms) for n in NAMES) if e]
        return "; ".join(errors) or None

    def boot(self):
        """Straight legs, then the home crouch, as on the robot at power-up (feet stay flat: the
        home angles sum to 0 on each leg)."""
        if self.walker.running:
            return "the policy is running: stop it first"

        def run():
            self.message = "Boot: straightening the legs"
            self.pose(STRAIGHT, BOOT_MS[0])
            time.sleep(BOOT_MS[0] / 1000 + 0.5)
            self.message = "Boot: crouching to home"
            self.pose(HOME, BOOT_MS[1])
            time.sleep(BOOT_MS[1] / 1000)
            self.message = "Boot done: holding home"
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
        self.cal.save()
        return None

    def flip(self, name):
        self.cal.joints[name].sign *= -1
        self.cal.save()

    def set_limit(self, name, end):
        pos = self.servo.get(BY_NAME[name][1], {}).get("pos")
        if pos is None:
            return "no position reading"
        setattr(self.cal.joints[name], end, pos)
        self.cal.save()
        return None

    def clear_limits(self, name):
        jc = self.cal.joints[name]
        jc.lo = jc.hi = None
        self.cal.save()

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
        self.cal.save()

    def level_imu(self):
        """Display only: the lean shown on the page is relative to this. The policy always gets
        the raw gravity estimate, as in the simulator."""
        if not self.imu.state:
            return "no IMU reading"
        self.cal.imu_level = tilt_deg(self.imu.state["gravity"])
        self.cal.save()
        return None

    # ---------------------------------------------------------------- IMU and Pi
    def _pi_loop(self):
        while True:
            t = pi_temp()
            if t is not None and not self.pi_hot and t >= PI_TEMP_LIMIT:
                self.pi_hot = True
                print(f"[protect] Pi at {t} C: all servos relaxed")
            elif self.pi_hot and (t is None or t < PI_TEMP_LIMIT - COOL_MARGIN):
                self.pi_hot = False
            if self.pi_hot:
                self.relax()
            self.pi = {"temp": t, **throttled()}
            time.sleep(1)

    # ---------------------------------------------------------------- page state
    def state(self):
        joints = {}
        for name in NAMES:
            _, sid, sim_lo, sim_hi, home, meaning = BY_NAME[name]
            jc, s = self.cal.joints[name], self.servo.get(sid)
            lo, hi = jc.limits_deg(name)
            j = {"id": sid, "found": s is not None, "meaning": meaning, "sim": [sim_lo, sim_hi], "home": home,
                 "limits": [lo, hi], "limits_set": jc.lo is not None and jc.hi is not None,
                 "zero": jc.zero, "sign": jc.sign, "locked": self.locked(sid),
                 "info": self.info.get(sid)}
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
        return {
            "joints": joints, "buses": buses, "pi": self.pi, "message": self.message,
            "hot": self.hot, "pi_hot": self.pi_hot, "zero_set": self.cal.zero_set,
            "imu": imu, "imu_error": self.imu.error,
            "imu_stats": {"bus": self.cal.imu_bus, "axes": self.cal.imu_axes,
                          "fail_pct": 100 * self.imu.failures / self.imu.reads if self.imu.reads else None,
                          "read_ms": self.imu.read_ms},
            "walker": self.walker.state(),
            "camera": {"available": Camera.available, "error": self.camera.error},
            "limits": {"servo_temp": SERVO_TEMP_LIMIT, "pi_temp": PI_TEMP_LIMIT},
        }


def pi_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return round(int(f.read()) / 1000, 1)
    except (OSError, ValueError):
        return None


def throttled():
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=1).stdout
        flags = int(out.strip().split("=")[1], 16)
        return {"undervolt": bool(flags & 0x1), "throttled": bool(flags & 0x4),
                "undervolt_before": bool(flags & 0x10000)}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return {}


# -------------------------------------------------------------------- web server
robot = None


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body=b"", ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def result(self, error=None, **extra):
        self.reply(200 if error is None else 409, json.dumps({"error": error, **extra}).encode())

    def camera(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            for frame in robot.camera.stream():
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(frame))
                self.wfile.write(frame + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        name = q.get("joint")
        if name is not None and name not in BY_NAME:
            return self.reply(400)
        try:
            p = url.path
            if p == "/":
                with open(PAGE, "rb") as f:
                    self.reply(200, f.read(), "text/html; charset=utf-8")
            elif p == "/state":
                robot.walker.heartbeat()  # the page is still there
                self.reply(200, json.dumps(robot.state()).encode())
            elif p == "/camera.mjpg":
                self.camera()
            elif p == "/scan":
                self.result(found=robot.scan())
            elif p == "/move":
                self.result(robot.move(name, float(q["deg"]), int(q.get("ms", 300))))
            elif p == "/hold":
                self.result(robot.hold(name))
            elif p == "/relax":
                robot.relax(name)
                self.result()
            elif p == "/stop":
                robot.stop()
                self.result()
            elif p == "/pose":
                angles = {"straight": STRAIGHT, "home": HOME}[q["name"]]
                self.result(robot.pose(angles, int(q.get("ms", 2500))))
            elif p == "/boot":
                self.result(robot.boot())
            elif p == "/policy/start":
                self.result(robot.walker.start(q["name"]))
            elif p == "/policy/stop":
                robot.walker.stop("stopped from the page")
                self.result()
            elif p == "/policy/cmd":
                self.result(robot.walker.set_command(float(q["vx"]), float(q["vy"]), float(q["wz"])))
            elif p == "/policy/move_ms":
                robot.walker.move_ms = max(0, min(100, int(q["ms"])))
                self.result()
            elif p == "/cal/zero_gyro":
                robot.imu.zero_gyro()
                self.result()
            elif p == "/cal/zero":
                self.result(robot.set_zero())
            elif p == "/cal/flip":
                robot.flip(name)
                self.result()
            elif p == "/cal/limit":
                if q["end"] not in ("lo", "hi"):
                    return self.reply(400)
                self.result(robot.set_limit(name, q["end"]))
            elif p == "/cal/clear_limits":
                robot.clear_limits(name)
                self.result()
            elif p == "/cal/write_limits":
                self.result(written=robot.write_servo_limits(reset=q.get("reset") == "1"))
            elif p == "/cal/imu":
                axes = q["axes"].split(",") if "axes" in q else None
                if axes is not None and (len(axes) != 3 or sorted(a[-1] for a in axes) != ["x", "y", "z"]):
                    return self.result("axes must use x, y and z once each, e.g. -y,x,z")
                robot.set_imu(axes, q.get("bus"))
                self.result()
            elif p == "/cal/imu_level":
                self.result(robot.level_imu())
            else:
                self.reply(404)
        except (KeyError, ValueError):
            self.reply(400)

    def log_message(self, *args):
        pass


class DualStackServer(ThreadingHTTPServer):
    """IPv4 and IPv6, so http://name.local:8000 works either way."""
    address_family = socket.AF_INET6
    daemon_threads = True

    def server_bind(self):
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


def main():
    global robot
    try:
        server = DualStackServer(("::", PORT), Handler)
    except OSError:
        try:
            server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
        except OSError:
            raise SystemExit(f"Port {PORT} is busy: stop any other servo script first (Ctrl+C in its window).")
    robot = Robot()
    for bus in robot.buses:
        print(f"Bus {bus.port}: servos {', '.join(map(str, bus.ids))}")
    missing = sorted(set(BY_ID) - set(robot.bus_of))
    if missing:
        print(f"Not found: servos {', '.join(map(str, missing))}")
    print(f"Calibration: {'zero set' if robot.cal.zero_set else 'NOT calibrated yet (set zero first)'}")
    print(f"Protection:  servos relax above {SERVO_TEMP_LIMIT} C, everything relaxes if the Pi reaches {PI_TEMP_LIMIT} C")
    print(f"Dashboard:   http://{socket.gethostname().removesuffix('.local')}.local:{PORT}   (Ctrl+C stops and holds position)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        robot.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()
