"""Getting up, and staying up, on the getup policy alone (no walker), the way the robot's "Get up" without a walker
runs it (robot/walker.py): one try. From lying face down on randomized robots, the getup policy runs for --seconds;
"up" is the robot's own test (torso within 15 deg of upright, turning under 1 rad/s, for 0.5 s; here also only the
feet touching the floor), within 6 s or it failed; once up, tilting past 40 deg is a fall (the robot relaxes).

Rows: "main" (juan_v7_13_straight's randomization) and stress rows, each one factor fixed at a hard value:
weak servos (stall 2.6 Nm), slow servos (3.6 rad/s), a slippery floor (friction 0.25), the torso's centre of mass
3 cm off.

    python scripts/analysis/getup_balance.py runs/getup/juan_getup_38_servos runs/getup/<other> ...
    python scripts/analysis/getup_balance.py <runs> --json out/balance.json

Per run and row: got up, standing at the end, share of the time standing once up, falls once up, median time to up,
peak impact (x body weight), servo load while getting up. PASS: standing at the end in >= 16/20 (main) and >= 6/10
in every stress row. Runs are ranked by standing at the end over all rows, then getting up at all, then time to up, then peak impact.
"""
import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np

PHYSICS = "runs/v7/juan_v7_13_straight"
UP_TILT_DEG, UP_RATE, UP_HOLD_S, UP_WITHIN_S, FALL_DEG = 15.0, 1.0, 0.5, 6.0, 40.0   # robot/walker.py
STRESS = {"stall 2.6 Nm": ("robot.stall_torque", 2.6), "speed 3.6 rad/s": ("robot.no_load_speed", 3.6),
          "friction 0.25": ("randomization.friction", (0.25, 0.25)), "CoM 3 cm": ("randomization.com_offset", 0.03)}
PASS_MAIN, PASS_STRESS = 0.8, 0.6


def episode(args):
    run, row, seed, seconds = args
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from biped_rl.tasks.getup import GetUpEnv, config
    d = Path(run)
    saved = json.loads((d / "getup.json").read_text())
    cfg = config(seconds, PHYSICS)
    if row in STRESS:
        key, val = STRESS[row]
        group, name = key.split(".")
        setattr(getattr(cfg, group), name, val)
    start = {"side_deg": 20.0, "joint_rad": 0.25, "any_heading": True, "knee_min_deg": saved.get("start", {}).get("knee_min_deg")}
    env = GetUpEnv(cfg.to_dict(), saved["weights"], start)
    model = PPO.load(d / "final_model", device="cpu")
    norm = VecNormalize.load(str(d / "vecnormalize.pkl"), DummyVecEnv([lambda: env]))
    norm.training = False
    obs, _ = env.reset(seed=seed)
    up_at = fell_at = None
    upright_for, standing_steps, after_up, impact, load = 0.0, 0, 0, 0.0, []
    for i in range(env._max_steps):
        obs, _, _, _, info = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
        t = (i + 1) * env.dt
        g = env.gravity_body()
        tilt = math.degrees(math.acos(min(max(-g[2], -1.0), 1.0)))
        rate = float(np.hypot(*env.robot_readings()["gyro"][:2]))
        on_feet = not env.non_foot_ground_contact()
        impact = max(impact, info["peak_force_bw"])
        if up_at is None:
            load.append(info["load_fraction"])
            upright_for = upright_for + env.dt if tilt < UP_TILT_DEG and rate < UP_RATE and on_feet else 0.0
            if upright_for >= UP_HOLD_S:
                up_at = t
            elif t >= UP_WITHIN_S:
                break   # one try: not up in time, the robot relaxes
        else:
            after_up += 1
            standing_steps += tilt < UP_TILT_DEG and on_feet
            if tilt > FALL_DEG:
                fell_at = t
                break
    standing_end = up_at is not None and fell_at is None and tilt < UP_TILT_DEG and on_feet
    return {"run": run, "row": row, "up_at": up_at, "fell_at": fell_at, "standing_end": bool(standing_end),
            "standing_share": standing_steps / after_up if after_up else 0.0, "impact": impact,
            "load": float(np.mean(load)) if load else 0.0}


def summarize(eps):
    ups = [e["up_at"] for e in eps if e["up_at"] is not None]
    return {"n": len(eps), "up": len(ups), "standing_end": sum(e["standing_end"] for e in eps),
            "falls_after_up": sum(e["fell_at"] is not None for e in eps),
            "standing_share": float(np.mean([e["standing_share"] for e in eps if e["up_at"] is not None])) if ups else 0.0,
            "time_to_up": float(np.median(ups)) if ups else float("nan"),
            "impact": float(max(e["impact"] for e in eps)), "load": float(np.mean([e["load"] for e in eps]))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--seeds", type=int, default=20, help="robots in the main row")
    p.add_argument("--stress-seeds", type=int, default=10, help="robots per stress row (0: none)")
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--json", help="save every number to this file")
    a = p.parse_args()
    rows = {"main": a.seeds, **({k: a.stress_seeds for k in STRESS} if a.stress_seeds else {})}
    jobs = [(run, row, 9000 + s, a.seconds) for run in a.runs for row, n in rows.items() for s in range(n)]
    with Pool(a.procs) as pool:
        res = pool.map(episode, jobs)
    out = {}
    for run in a.runs:
        out[run] = {row: summarize([e for e in res if e["run"] == run and e["row"] == row]) for row in rows}
        r = out[run]
        ok = r["main"]["standing_end"] >= PASS_MAIN * r["main"]["n"] and all(
            r[k]["standing_end"] >= PASS_STRESS * r[k]["n"] for k in rows if k != "main")
        r["pass"] = bool(ok)
        r["score"] = (sum(r[k]["standing_end"] / r[k]["n"] for k in rows), sum(r[k]["up"] / r[k]["n"] for k in rows),
                      -np.nan_to_num(r["main"]["time_to_up"], nan=99.0), -r["main"]["impact"])
        print(f"== {run}: {'PASS' if ok else 'fail'}")
        for row in rows:
            s = r[row]
            print(f"   {row:16s} up {s['up']:2d}/{s['n']}  standing at the end {s['standing_end']:2d}/{s['n']}  "
                  f"falls once up {s['falls_after_up']:2d}  standing once up {100 * s['standing_share']:3.0f}%  "
                  f"time to up {s['time_to_up']:.1f} s  peak impact {s['impact']:.1f}x  load {s['load']:.3f}")
    print("\nranking (standing at the end over all rows, then got up, then time to up, then peak impact):")
    for i, run in enumerate(sorted(a.runs, key=lambda r: out[r]["score"], reverse=True), 1):
        m = out[run]["main"]
        print(f"  {i}. {Path(run).name:28s} {'PASS' if out[run]['pass'] else 'fail'}  main standing {m['standing_end']}/{m['n']}, "
              f"stress {sum(out[run][k]['standing_end'] for k in rows if k != 'main')}/"
              f"{sum(out[run][k]['n'] for k in rows if k != 'main')}, up in {m['time_to_up']:.1f} s, impact {m['impact']:.1f}x")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
