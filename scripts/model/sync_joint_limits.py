"""Give the simulator the joint ranges measured on the robot: reads a calibration file (the limits set by
hand on the dashboard, in servo counts), and writes the ranges into the model (juan/rl_robot.xml), the
robot's joint table (robot/joints.py) and the model script's constants (scripts/model/postprocess_onshape.py).

Each range is what the servo can really reach (its set limits and its 0-1000 counts), the narrower of the two
legs (the policy is mirror-symmetric), minus a safety margin, in whole degrees. Without --write it only
prints the ranges next to the current ones.

    python scripts/model/sync_joint_limits.py robot/calibration.json             # compare
    python scripts/model/sync_joint_limits.py robot/calibration.json --write     # and write them
"""
import argparse
import math
import re
from pathlib import Path

from robot.joints import BY_NAME, Calibration

ROOT = Path(__file__).resolve().parents[2]
MODEL, TABLE, POSTPROCESS = ROOT / "juan/rl_robot.xml", ROOT / "robot/joints.py", ROOT / "scripts/model/postprocess_onshape.py"


def reachable_deg(cal, name):
    """(lo, hi) degrees this servo can reach: its set limits, within 0-1000 counts."""
    j = cal.joints[name]
    if j.lo is None or j.hi is None:
        raise SystemExit(f"{name}: limits not set in the calibration (set both ends on the dashboard)")
    lo_c, hi_c = sorted((j.lo, j.hi))
    a, b = j.degrees(max(lo_c, 0)), j.degrees(min(hi_c, 1000))
    return min(a, b), max(a, b)


def ranges(cal, margin):
    """Simulator ranges (deg) in the joint table's conventions, both legs mirrored."""
    shrink = lambda lo, hi: (math.ceil(lo + margin), math.floor(hi - margin))
    out = {}
    # hip roll: left + = outward, right + = inward
    l_lo, l_hi = reachable_deg(cal, "left_hip_roll")
    r_lo, r_hi = reachable_deg(cal, "right_hip_roll")
    inward, outward = shrink(-min(-l_lo, r_hi), min(l_hi, -r_lo))
    inward = -inward
    out["left_hip_roll"], out["right_hip_roll"] = (-inward, outward), (-outward, inward)
    for part in ("hip_pitch", "knee", "ankle"):
        (a_lo, a_hi), (b_lo, b_hi) = reachable_deg(cal, f"left_{part}"), reachable_deg(cal, f"right_{part}")
        out[f"left_{part}"] = out[f"right_{part}"] = shrink(max(a_lo, b_lo), min(a_hi, b_hi))
    return out


def write(new):
    xml = MODEL.read_text()
    for name, (lo, hi) in new.items():
        pat = rf'(<joint [^>]*name="{name}"[^>]*range=")[^"]*(")'
        xml, n = re.subn(pat, rf"\g<1>{math.radians(lo):.6g} {math.radians(hi):.6g}\g<2>", xml)
        assert n == 1, f"{name} not found once in {MODEL.name}"
    MODEL.write_text(xml)
    table = TABLE.read_text()
    for name, (lo, hi) in new.items():
        pat = rf'(\("{name}", \d+, )[-0-9.]+, [-0-9.]+(,)'
        table, n = re.subn(pat, rf"\g<1>{float(lo)}, {float(hi)}\g<2>", table)
        assert n == 1, f"{name} not found once in {TABLE.name}"
    TABLE.write_text(table)
    post = POSTPROCESS.read_text()
    inward, outward = new["right_hip_roll"][1], new["left_hip_roll"][1]
    post = re.sub(r"ROLL_INWARD_DEG, ROLL_OUTWARD_DEG = [-0-9.]+, [-0-9.]+",
                  f"ROLL_INWARD_DEG, ROLL_OUTWARD_DEG = {float(inward)}, {float(outward)}", post)
    pitch = ", ".join(f'"{p}": ({float(new["left_" + p][0])}, {float(new["left_" + p][1])})'
                      for p in ("hip_pitch", "knee", "ankle"))
    post = re.sub(r"PITCH_RANGE_DEG = \{[^}]*\}", f"PITCH_RANGE_DEG = {{{pitch}}}", post)
    POSTPROCESS.write_text(post)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("calibration", help="a robot calibration.json (copied from the Pi)")
    p.add_argument("--margin", type=float, default=5.0, help="degrees kept clear of each measured end")
    p.add_argument("--write", action="store_true")
    a = p.parse_args()
    cal = Calibration.load(a.calibration)
    new = ranges(cal, a.margin)
    print(f"{'joint':16s} {'robot can reach':>20s} {'simulator now':>16s} {'new simulator':>16s}")
    for name, (lo, hi) in new.items():
        r = reachable_deg(cal, name)
        print(f"{name:16s} {r[0]:+8.1f} .. {r[1]:+6.1f}   {BY_NAME[name][2]:+6.0f} .. {BY_NAME[name][3]:+4.0f}   "
              f"{lo:+6.0f} .. {hi:+4.0f}" + ("" if (lo, hi) == tuple(BY_NAME[name][2:4]) else "   changed"))
    home = {n: BY_NAME[n][4] for n in new}
    outside = [n for n, (lo, hi) in new.items() if not lo <= home[n] <= hi]
    if outside:
        raise SystemExit(f"home pose outside the new range for {outside}: not written")
    if a.write:
        write(new)
        print("written: juan/rl_robot.xml, robot/joints.py, scripts/model/postprocess_onshape.py "
              "(check with: python scripts/check_robot_joints.py)")


if __name__ == "__main__":
    main()
