"""Carry a calibration's joint limits and directions over to a new zero (after re-horning the servos).

The limits are where the leg parts touch: the same joint angles whatever the horns' position, so they
move with the zero. Run after "Set zero" on the dashboard (legs straight), with the dashboard stopped
(it would overwrite the file with what it holds):

    python3 -m robot.carry_limits robot/calibration_old.json

Keeps robot/calibration.json's zeros; takes the old file's signs, limits (as angles) and IMU settings.
Then start the dashboard and press "Write limits to servos".
"""
import argparse

from robot.joints import CAL_FILE, NAMES, Calibration


def main():
    p = argparse.ArgumentParser()
    p.add_argument("old", help="the calibration with the right limits (before re-horning)")
    a = p.parse_args()
    old, cal = Calibration.load(a.old), Calibration.load(CAL_FILE)
    if not cal.zero_set:
        raise SystemExit("robot/calibration.json has no zero: Set zero on the dashboard first (legs straight)")
    print(f"{'joint':16s} {'old zero':>8s} {'new zero':>8s} {'range (deg)':>16s}")
    for name in NAMES:
        o, n = old.joints[name], cal.joints[name]
        n.sign = o.sign
        n.lo = None if o.lo is None else n.zero + (o.lo - o.zero)
        n.hi = None if o.hi is None else n.zero + (o.hi - o.zero)
        lo, hi = n.limits_deg(name)
        print(f"{name:16s} {o.zero:8d} {n.zero:8d} {lo:+7.1f} .. {hi:+6.1f}")
    cal.imu_bus, cal.imu_axes, cal.imu_level, cal.imu_offset = old.imu_bus, old.imu_axes, old.imu_level, old.imu_offset
    cal.save(CAL_FILE)
    print(f"saved {CAL_FILE}: start the dashboard, press Write limits to servos, then Rescan")


if __name__ == "__main__":
    main()
