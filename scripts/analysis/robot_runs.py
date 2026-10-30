"""What the real robot did in its runs (robot/logs/*.npz): per run and per stretch of constant command,
the speed it actually walked, drift, turning, heading weave, foot lift, double support and lean, next to
what was commanded. Position comes from the planted-foot estimate (biped_rl/robot_logs.py) and heading
from the IMU. The estimate can't see feet slipping, so speeds are a lower bound: checked on simulated
runs logged the same way, it found 87% of the true distance at 0.2 m/s and 57% at 0.3 m/s. Measure
one real run with a tape to know how far it is off on the floor you use.

    python scripts/analysis/robot_runs.py robot/logs/*.npz
    python scripts/analysis/robot_runs.py robot/logs/2026-10-31_*.npz --min-seconds 3 --json out/runs.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from biped_rl.robot_logs import Replay, kinematic_env

DOUBLE_SUPPORT_MM = 3.0  # both soles within this of the floor


def segments(command, t, min_seconds):
    """(start, end) index ranges where the command stays the same for at least min_seconds."""
    out, start = [], 0
    for i in range(1, len(command) + 1):
        if i == len(command) or np.any(np.abs(command[i] - command[start]) > 1e-6):
            if t[i - 1] - t[start] >= min_seconds:
                out.append((start, i))
            start = i
    return out


def weave_deg(yaw_deg, dt):
    """Peak-to-peak heading swing with the steady drift removed, after the first 2 s."""
    y = yaw_deg[int(2 / dt):]
    if len(y) < int(3 / dt):
        return float("nan")
    x = np.arange(len(y))
    return float(np.ptp(y - np.polyval(np.polyfit(x, y, 1), x)))


def pi_summary(log):
    """The Pi during the run (logs from 2026-10-30 on): mean and peak of each health field."""
    if "pi" not in log:
        return {}
    out = {}
    with np.errstate(all="ignore"):
        for k, col in zip(log["pi_fields"], np.asarray(log["pi"], dtype=float).T):
            if np.isfinite(col).any():
                out[str(k)] = {"mean": float(np.nanmean(col)), "max": float(np.nanmax(col)), "min": float(np.nanmin(col))}
    return out


def analyse(path, env, min_seconds):
    log = dict(np.load(path))
    t, cmd = log["t"], log["command"]
    if len(t) < 10:
        return None
    dt = float(np.median(np.diff(t)))
    replay = Replay(log, env)
    xy, lift, both, roll, pitch = [], [], [], [], []
    for k in range(len(t)):
        replay.pose(k)
        z = replay.corrected(replay.soles[:, 2])
        xy.append(replay.base_xy.copy())
        lift.append(1000 * (z.max() - z.min()))
        both.append(1000 * (z.max() - z.min()) < DOUBLE_SUPPORT_MM)
        roll.append(replay.roll)
        pitch.append(replay.pitch)
    xy, lift, both = np.array(xy), np.array(lift), np.array(both)
    yaw = np.unwrap(log["yaw"] - log["yaw"][0])
    rows = []
    for a, b in segments(cmd, t, min_seconds):
        seconds = float(t[b - 1] - t[a])
        h = yaw[a]  # the heading the stretch started with
        d = xy[b - 1] - xy[a]
        rows.append({
            "command": [float(x) for x in cmd[a]], "seconds": seconds,
            "forward_mps": float((d[0] * math.cos(h) + d[1] * math.sin(h)) / seconds),
            "sideways_mps": float((-d[0] * math.sin(h) + d[1] * math.cos(h)) / seconds),
            "turn_degps": float(math.degrees(yaw[b - 1] - yaw[a]) / seconds),
            "weave_deg": weave_deg(np.degrees(yaw[a:b]), dt),
            "lift_mm": float(np.percentile(lift[a:b], 90)), "both_feet_pct": float(100 * both[a:b].mean()),
            "lean_fwd_deg": float(np.median(pitch[a:b])), "lean_right_deg": float(np.median(roll[a:b])),
        })
    tick = log.get("tick_ms", np.array([np.nan]))
    link = log.get("link_s", np.array([0.0]))
    battery = [float("nan")] * 3  # V: start, end, lowest (median over the servos; logs from 2026-10-30 on)
    if "vin" in log and np.isfinite(log["vin"]).any():
        with np.errstate(all="ignore"):
            v = np.nanmedian(log["vin"], axis=1)
        n = max(1, int(3 / dt))
        battery = [float(np.nanmedian(v[:n])), float(np.nanmedian(v[-n:])), float(np.nanmin(v))]
    return {"log": Path(path).name, "policy": Path(path).stem.split("_", 2)[2], "seconds": float(t[-1] - t[0]),
            "status": str(log.get("status", "")), "sole_bias_mm": 1000 * replay.sole_bias, "tick_ms_median": float(np.nanmedian(tick)),
            "tick_ms_p99": float(np.nanpercentile(tick, 99)), "max_link_gap_s": float(np.max(link)), "battery_v": battery,
            "servo_temp_max": float(np.nanmax(log["temp"])) if "temp" in log and np.isfinite(log["temp"]).any() else float("nan"),
            "pi": pi_summary(log),
            "segments": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+")
    p.add_argument("--min-seconds", type=float, default=3.0, help="shortest stretch of constant command to report")
    p.add_argument("--json", help="save every number to this file")
    a = p.parse_args()
    env = kinematic_env()
    runs = []
    for path in sorted(a.logs):
        r = analyse(path, env, a.min_seconds)
        if r is None:
            continue
        runs.append(r)
        print(f"\n{r['log']}  ({r['policy']}, {r['seconds']:.0f} s, {r['status'] or 'no status'}; "
              f"control loop {r['tick_ms_median']:.1f} ms median, {r['tick_ms_p99']:.1f} ms 99%; longest link gap {r['max_link_gap_s']:.1f} s)")
        if np.isfinite(r["battery_v"][0]):
            print(f"  battery {r['battery_v'][0]:.2f} V at the start, {r['battery_v'][1]:.2f} V at the end, lowest "
                  f"{r['battery_v'][2]:.2f} V; hottest servo {r['servo_temp_max']:.0f} C")
        p = r["pi"]
        if p:
            g = lambda k, s="mean", p=p: p.get(k, {}).get(s, float("nan"))
            flags = [w for w, k in (("UNDERVOLTAGE", "undervolt"), ("THROTTLED", "throttled")) if g(k, "max") > 0.5]
            print(f"  Pi: CPU {g('cpu'):.0f}% (peak {g('cpu', 'max'):.0f}%), dashboard {g('self_cpu'):.0f}% of a core, "
                  f"{g('temp', 'max'):.0f} C, clock {g('arm_mhz', 'min'):.0f}-{g('arm_mhz', 'max'):.0f} MHz, "
                  f"Wi-Fi {g('wifi_dbm'):.0f} dBm, servo bus pass up to {g('bus_ms', 'max'):.0f} ms"
                  + (f"; {' and '.join(flags)} during the run" if flags else ""))
        if not r["segments"]:
            continue
        print(f"  {'command (fwd, side, turn)':>27s} {'s':>5s} {'fwd m/s':>8s} {'(% cmd)':>8s} {'side':>6s} {'turn°/s':>8s}"
              f" {'weave':>6s} {'lift':>6s} {'2 feet':>7s} {'lean fwd':>9s} {'right':>6s}")
        for s in r["segments"]:
            c = s["command"]
            share = f"{100 * s['forward_mps'] / c[0]:6.0f}%" if abs(c[0]) > 0.01 else "      -"
            print(f"  {c[0]:+9.2f} {c[1]:+7.2f} {c[2]:+7.2f}  {s['seconds']:5.1f} {s['forward_mps']:+8.3f} {share:>8s}"
                  f" {s['sideways_mps']:+6.3f} {s['turn_degps']:+8.1f} {s['weave_deg']:5.0f}° {s['lift_mm']:4.0f}mm"
                  f" {s['both_feet_pct']:6.0f}% {s['lean_fwd_deg']:+8.1f}° {s['lean_right_deg']:+5.1f}°")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(runs, indent=2))


if __name__ == "__main__":
    main()
