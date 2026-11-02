"""Leg servos to the middle of their range (500 counts), one at a time, each held there while its horn
goes back on straight (re-zeroing the legs). Hang the robot first: the leg swings to wherever 500 is.

Stop the dashboard first (it holds the serial ports). Run on the Pi:

    python3 -m robot.center_servos 3                # only servo 3 (left knee)
    python3 -m robot.center_servos 3 7              # servo 3, then 7
    python3 -m robot.center_servos                  # all 8, one after the other
    python3 -m robot.center_servos 3 --clear-limits # also widen its stored limits to 0-1000

IDs: 1 left hip roll, 2 left hip pitch, 3 left knee, 4 left ankle, 5-8 the same on the right.
Each servo is held until Enter, then goes limp (--hold: keeps holding) before the next one moves.

A servo refuses a target outside the angle limits stored in its memory (set from the dashboard's
calibration); --clear-limits widens them to the full range so it can reach 500. After re-horning, the
old zero counts and limits in robot/calibration.json are wrong: calibrate again in the dashboard
(joints page), then python scripts/model/sync_joint_limits.py robot/calibration.json on the computer.
"""
import argparse
import time

from robot import hiwonder
from robot.joints import BY_ID

CENTER = 500


def find(ids):
    """{id: bus} for the servos that answer."""
    bus_of = {}
    for port in hiwonder.ports():
        try:
            bus = hiwonder.Bus(port)
        except Exception as e:
            print(f"can't open {port}: {e} (is the dashboard still running?)")
            continue
        for sid in ids:
            if sid not in bus_of and bus.ping(sid):
                bus_of[sid] = bus
    return bus_of


def center(sid, bus, a):
    name = BY_ID[sid][0]
    lim = bus.angle_limits(sid)
    if lim and not lim[0] <= CENTER <= lim[1]:
        if not a.clear_limits:
            print(f"{sid} {name}: stored limits {lim} exclude {CENTER}, it won't move: skipped "
                  "(run again with --clear-limits)")
            return
        bus.write_angle_limits(sid, 0, 1000)
        time.sleep(0.02)
        print(f"{sid} {name}: stored limits {lim} -> 0-1000")
    input(f"\n{sid} {name}: Enter moves it to {CENTER} over {a.ms} ms... ")
    bus.move(sid, CENTER, a.ms)
    time.sleep(a.ms / 1000 + 0.5)
    pos = bus.position(sid)
    print(f"{sid} {name}: at {'?' if pos is None else pos}" + ("" if pos is None else f" ({pos - CENTER:+d} from {CENTER})"))
    print("Holding: take the horn off and put it back with this joint straight (leg straight, foot flat).")
    input("Enter when done" + (" (it keeps holding)" if a.hold else " (it goes limp)") + "... ")
    if not a.hold:
        bus.set_torque(sid, False)


def main():
    p = argparse.ArgumentParser(description="Center leg servos at 500, one at a time.")
    p.add_argument("ids", type=int, nargs="*", help="servo IDs 1-8, done in this order (default: all 8)")
    p.add_argument("--ms", type=int, default=2000, help="time for the move (default 2000 ms: slow)")
    p.add_argument("--clear-limits", action="store_true", help="widen the stored angle limits to 0-1000")
    p.add_argument("--hold", action="store_true", help="leave each servo holding (default: limp when done)")
    a = p.parse_args()
    ids = a.ids or sorted(BY_ID)
    bad = [sid for sid in ids if sid not in BY_ID]
    if bad:
        p.error(f"not a leg servo: {bad} (IDs 1-8)")

    bus_of = find(ids)
    missing = [sid for sid in ids if sid not in bus_of]
    if missing:
        print(f"no answer from IDs {missing}")
    print("Robot hanging, legs free?")
    for sid in ids:
        if sid in bus_of:
            center(sid, bus_of[sid], a)
    for bus in set(bus_of.values()):
        bus.close()


if __name__ == "__main__":
    main()
