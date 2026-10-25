#!/usr/bin/env python3
"""Hardware self-test. Nothing moves: servos are only read, never commanded.

    python3 -m robot.selftest              # everything
    python3 -m robot.selftest --no-camera  # skip the camera

Stop the dashboard first (both want the serial ports). Prints PASS / WARN / FAIL per check and
ends with a summary; paste the output to compare runs or to ask for help.
"""
import argparse
import glob
import math
import os
import statistics
import subprocess
import sys
import threading
import time

from robot import pi_health
from robot.joints import BY_ID, Calibration

results = []


def report(status, name, detail=""):
    results.append(status)
    print(f"  [{status}] {name}" + (f": {detail}" if detail else ""), flush=True)


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def pct(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, int(p / 100 * len(v)))]


# ---------------------------------------------------------------- 1. system
def system():
    print("\n1. System")
    v = sys.version_info
    report("PASS" if v >= (3, 9) else "FAIL", "Python", f"{v.major}.{v.minor}")
    for mod in ("serial", "smbus2"):
        try:
            __import__(mod)
            report("PASS", f"import {mod}")
        except ImportError:
            report("FAIL", f"import {mod}", "sudo apt install python3-serial python3-smbus2")
    groups = run(["id", "-nG"]) or ""
    report("PASS" if "dialout" in groups.split() else "WARN", "user in group dialout",
           "" if "dialout" in groups.split() else "sudo usermod -aG dialout $USER, then log out and in")
    usb = run(["lsusb"]) or ""
    n = usb.count("1a86:7523")
    report("PASS" if n >= 1 else "FAIL", "BusLinker (CH340, 1a86:7523) on USB", f"{n} found")
    ports = sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))
    report("PASS" if len(ports) == n else "WARN" if ports else "FAIL", "serial ports", ", ".join(ports) or "none")
    i2c = sorted(glob.glob("/dev/i2c-*"))
    report("PASS" if i2c else "FAIL", "I2C buses", ", ".join(i2c) or "none: enable I2C (docs/bringup.md, step 1.2)")
    flags = pi_health.throttle_flags()
    if flags is not None:
        if flags & 0x5:
            report("FAIL", "Pi power", f"undervoltage or throttling now (flags {flags:#x})")
        elif flags & 0x50000:
            report("WARN", "Pi power", f"undervoltage or throttling happened since boot (flags {flags:#x})")
        else:
            report("PASS", "Pi power", "no undervoltage or throttling since boot")
    t = pi_health.temperature()
    if t is None:
        report("WARN", "Pi temperature", "not readable")
    else:
        report("PASS" if t < 65 else "WARN", "Pi temperature", f"{t:.1f} C")
    return ports


# ---------------------------------------------------------------- 2. servos
def servos(ports):
    print("\n2. Servos (read only)")
    from robot import hiwonder
    buses, found = [], {}
    for port in ports:
        try:
            bus = hiwonder.Bus(port)
        except Exception as e:
            report("FAIL", f"open {port}", f"{e} (is the dashboard still running?)")
            continue
        ids = [sid for sid in range(1, 9) if bus.ping(sid)]
        bus.ids = ids
        buses.append(bus)
        for sid in ids:
            found.setdefault(sid, []).append(port)
        print(f"     {port}: IDs {ids}")
    dup = {sid: p for sid, p in found.items() if len(p) > 1}
    report("FAIL" if dup else "PASS", "each ID on one bus only", f"duplicates: {dup}" if dup else "")
    missing = sorted(set(BY_ID) - set(found))
    report("FAIL" if missing else "PASS", "all 8 servos answer", f"missing {missing}" if missing else "")
    for bus in buses:
        legs = {"left" if sid <= 4 else "right" for sid in bus.ids}
        report("PASS" if len(legs) == 1 else "FAIL", f"{bus.port} carries one leg",
               ", ".join(sorted(legs)) + ("" if len(legs) == 1 else ": IDs 1-4 = left leg, 5-8 = right leg"))
    vins = []
    for bus in buses:
        for sid in bus.ids:
            name = BY_ID[sid][0]
            info, pos = bus.info(sid), bus.position(sid)
            temp, vin, torque = bus.temperature(sid), bus.voltage(sid), bus.torque_on(sid)
            if vin:
                vins.append(vin)
            problems = []
            if info["vin_limits"] and info["vin_limits"][1] < 12.6:
                problems.append(f"voltage limit {info['vin_limits'][1]} V < 12.6 V (full pack)")
            if info["vin_limits"] and info["vin_limits"][0] > 9.5:
                problems.append(f"low-voltage cutoff {info['vin_limits'][0]} V")
            if temp is not None and temp > 45:
                problems.append(f"warm: {temp} C")
            if info["offset"]:
                problems.append(f"angle offset {info['offset']} set in the servo")
            status = "WARN" if problems else "PASS"
            report(status, f"servo {sid} {name}",
                   f"pos {pos}, {temp} C, {vin} V, torque {'on' if torque else 'off'}, "
                   f"limits {info['angle_limits']}, vin {info['vin_limits']}, max temp {info['temp_max']}"
                   + ("; " + "; ".join(problems) if problems else ""))
    if vins:
        spread = max(vins) - min(vins)
        report("PASS" if spread < 0.3 else "WARN", "servo voltages agree",
               f"{min(vins):.2f}-{max(vins):.2f} V" + ("" if spread < 0.3 else ": check power wiring / connectors"))
        report("PASS" if min(vins) > 10.5 else "WARN", "battery", f"{max(vins):.2f} V")
    return buses


def timing(buses):
    print("\n3. Bus timing (the walker uses the latest reading; a full pass adds to the delay the policy sees)")

    def passes(bus, n, out):
        times, fails = [], 0
        for _ in range(n):
            t0 = time.perf_counter()
            for sid in bus.ids:
                if bus.position(sid) is None:
                    fails += 1
            times.append((time.perf_counter() - t0) * 1000)
        out[bus.port] = (times, fails, n * len(bus.ids))

    for parallel in (False, True) if len(buses) > 1 else (False,):
        out = {}
        if parallel:
            th = [threading.Thread(target=passes, args=(b, 300, out)) for b in buses]
            [t.start() for t in th]
            [t.join() for t in th]
        else:
            for b in buses:
                passes(b, 300, out)
        for port, (times, fails, reads) in out.items():
            n = reads // 300
            p99 = pct(times, 99)
            per_servo = p99 / max(n, 1)  # ~4 ms per servo read is normal
            status = "PASS" if per_servo < 4.5 and fails == 0 else "WARN" if per_servo < 6 else "FAIL"
            report(status, f"{'all buses at once' if parallel else 'alone'}, {port}",
                   f"read {n} positions: median {statistics.median(times):.1f} ms, 99% {p99:.1f}, max {max(times):.1f}; "
                   f"failed {fails}/{reads}")


# ---------------------------------------------------------------- 4. IMU
def imu():
    print("\n4. IMU (keep the robot upright and still for 10 s)")
    cal = Calibration.load()
    out = run(["i2cdetect", "-y", str(cal.imu_bus)]) or ""
    seen = [a for a in ("6a", "6b", "1c", "1e", "28", "29") if f" {a}" in out]
    report("PASS" if any(a in seen for a in ("6a", "6b", "28", "29")) else "FAIL", f"I2C bus {cal.imu_bus} devices",
           ", ".join("0x" + a for a in seen) or "nothing found: check wiring (3.3 V, GND, SDA pin 3, SCL pin 5) and that I2C is enabled")
    from robot.imu import IMU, tilt_deg
    dev = IMU(cal.imu_bus, cal.imu_axes)
    time.sleep(3.0)  # connect + 2 s gyro zeroing
    if dev.state is None:
        report("FAIL", "IMU", dev.error or "no reading")
        return
    report("PASS", f"{dev.state['device']} on I2C bus {cal.imu_bus}")
    samples, raw_gyro = [], []
    t_end = time.time() + 10
    while time.time() < t_end:
        s = dev.state
        samples.append(s)
        raw_gyro.append([s["gyro"][k] + s["gyro_bias"][k] for k in range(3)])
        time.sleep(0.02)
    bad = 100 * dev.failures / max(dev.reads, 1)
    report("PASS" if bad < 0.5 else "WARN" if bad < 3 else "FAIL", "read reliability",
           f"{bad:.2f}% bad of {dev.reads}, {dev.read_ms:.2f} ms per read")
    rate = dev.reads / 13.0
    report("PASS" if rate > 150 else "WARN", "IMU loop rate", f"~{rate:.0f} reads/s (target 200)")
    for k, axis in enumerate("xyz"):
        g = [s["gyro"][k] for s in samples]
        raw = statistics.mean(r[k] for r in raw_gyro)
        sd = statistics.pstdev(g)
        report("PASS" if abs(raw) < 0.05 and sd < 0.02 else "WARN", f"gyro {axis} at rest",
               f"bias {raw:+.4f} rad/s (removed), noise {sd:.4f} (sim: bias up to 0.02, noise 0.1)")
    right, fwd = zip(*(tilt_deg(s["gravity"]) for s in samples))
    report("PASS" if statistics.pstdev(fwd) < 0.3 and statistics.pstdev(right) < 0.3 else "WARN", "tilt steady",
           f"forward {statistics.mean(fwd):+.2f} deg (+-{statistics.pstdev(fwd):.2f}), "
           f"right {statistics.mean(right):+.2f} (+-{statistics.pstdev(right):.2f})")
    gz = statistics.mean(s["gravity"][2] for s in samples)
    report("PASS" if gz < -0.9 else "FAIL", "gravity points down along torso z (robot upright)",
           f"z = {gz:+.2f}" + ("" if gz < -0.9 else ": wrong axis mapping (dashboard calibration step 5) or robot not upright"))
    acc = statistics.mean(s["accel_g"] for s in samples)
    report("PASS" if abs(acc - 1) < 0.05 else "WARN", "accelerometer reads 1 g at rest", f"{acc:.3f} g")
    drift = math.degrees(samples[-1]["yaw"] - samples[0]["yaw"])
    report("PASS" if abs(drift) < 2 else "WARN", "yaw drift at rest", f"{drift:+.2f} deg in 10 s")


# ---------------------------------------------------------------- 5. camera
def camera():
    print("\n5. Camera")
    out = run(["rpicam-hello", "--list-cameras"]) or run(["libcamera-hello", "--list-cameras"]) or ""
    report("PASS" if "imx219" in out.lower() else "FAIL", "camera detected (IMX219, v2.1)",
           "" if "imx219" in out.lower() else "check the ribbon (contacts facing the right way) and the CSI port")
    from robot.camera import Camera
    if not Camera.available:
        report("FAIL", "picamera2", "sudo apt install python3-picamera2")
        return
    cam, frames, t0 = Camera(), 0, time.monotonic()
    for _ in cam.stream():
        frames += 1
        if time.monotonic() - t0 > 3:
            break
    fps = frames / (time.monotonic() - t0)
    report("PASS" if fps > 10 else "FAIL", "MJPEG stream", f"{fps:.1f} frames/s" + (f", {cam.error}" if cam.error else ""))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-camera", action="store_true")
    a = p.parse_args()
    print("Juan self-test: nothing moves.")
    ports = system()
    buses = servos(ports) if ports else []
    if buses:
        timing(buses)
        for b in buses:
            b.close()
    imu()
    if not a.no_camera:
        camera()
    print(f"\n{results.count('PASS')} passed, {results.count('WARN')} warnings, {results.count('FAIL')} failed")
    os._exit(1 if "FAIL" in results else 0)  # (exits even if the camera thread is still winding down)


if __name__ == "__main__":
    main()
