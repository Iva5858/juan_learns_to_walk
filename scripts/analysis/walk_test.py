"""Walking the way the real robot does it: standing still at home, the policy taking over (start_policy),
1 s standing, then a straight forward command for 8 s. On randomized robots with the measured servos,
without and with 0.3 m/s shoves. Every policy is tested on the same robots: the randomization of
--physics-from (default juan_v7_13_straight), not the run's own (a run trained on harder robots would
otherwise also be tested on them).

The other tests start each episode from the training start state (a perturbed crouch, already moving):
juan_v7_13 lost 58 of its 77 failures in their first 1.5 s, which the real robot never sees. This one
counts only what happens while walking, and how straight it went.

Per speed: the speed reached (share of the command), falls while walking (hardware-limit stops
included), sideways drift, heading drift and the weave (heading swing peak to peak, drift removed);
in the summary also the mean speed at commands of 0.4 m/s and up.

    python scripts/analysis/walk_test.py runs/v7/juan_v7_13_straight runs/v7/juan_v7_21_steady_best
    python scripts/analysis/walk_test.py runs/a --json runs/logs/v7/a_walk.json
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from biped_rl.runs import load_policy
from biped_rl.tasks.locomotion import BipedEnv, EnvConfig

SPEEDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
PHYSICS_FROM = "runs/v7/juan_v7_13_straight"
STAND_S, WALK_S = 1.0, 8.0


def episode(args):
    run, speed, seed, shoves, physics_from = args
    cfg = EnvConfig.load(run)
    cfg.randomization = EnvConfig.load(physics_from).randomization
    cfg.use_measured_servos()
    r = cfg.randomization
    r.physics_curriculum, r.push_curriculum = False, False
    r.push_velocity = 0.3 if shoves else 0.0
    model, norm = load_policy(run, cfg)
    env = BipedEnv(cfg.to_dict())
    env.reset(seed=seed)
    env.settle_at_home()
    obs = env.start_policy()
    env.episode.next_push = STAND_S + 1.0   # no shove while still standing
    for _ in range(int(STAND_S / env.dt)):
        obs, _, term, _, _ = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
        if term:
            return None   # fell standing: not a walking result (counted apart)
    env.set_command(speed)
    y0, p0, yaws, fell = env.yaw(), env.data.qpos[:2].copy(), [], False
    t0 = env.episode.t
    for _ in range(int(WALK_S / env.dt)):
        obs, _, term, _, _ = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
        if term:
            fell = True
            break
        yaws.append(env.yaw())
    fwd, left, turn = env.moved_since(p0, y0)
    y = np.degrees(np.unwrap(yaws[50:])) if len(yaws) > 100 else np.zeros(2)  # after the first second
    x = np.arange(len(y))
    weave = float(np.ptp(y - np.polyval(np.polyfit(x, y, 1), x)))
    t = max(env.episode.t - t0, env.dt)
    env.close()
    return {"speed": fwd / t, "fell": fell, "left": left, "turn": abs(turn), "weave": weave}


def measure(run, seeds, pool, physics_from=PHYSICS_FROM, speeds=SPEEDS):
    out = {}
    for shoves in (False, True):
        jobs = [(run, v, 7000 + s, shoves, physics_from) for v in speeds for s in range(seeds)]
        res = pool.map(episode, jobs)
        rows, standing_falls = {}, 0
        for v in speeds:
            eps = [x for (_, vv, *_), x in zip(jobs, res) if vv == v]
            standing_falls += sum(e is None for e in eps)
            eps = [e for e in eps if e is not None]
            done = [e for e in eps if not e["fell"]] or eps
            rows[v] = {"speed": float(np.mean([e["speed"] for e in eps])),
                       "share": float(np.mean([e["speed"] for e in eps]) / v),
                       "falls": sum(e["fell"] for e in eps), "episodes": len(eps),
                       "left_m": float(np.mean([abs(e["left"]) for e in done])),
                       "turn_deg": float(np.mean([e["turn"] for e in done])),
                       "weave_deg": float(np.mean([e["weave"] for e in done]))}
        falls = sum(r["falls"] for r in rows.values())
        out["shoves" if shoves else "calm"] = {
            "rows": {str(k): v for k, v in rows.items()}, "standing_falls": standing_falls,
            "summary": {"share": float(np.mean([r["share"] for r in rows.values()])), "falls": falls,
                        "episodes": sum(r["episodes"] for r in rows.values()),
                        "turn_deg": float(np.mean([r["turn_deg"] for r in rows.values()])),
                        "weave_deg": float(np.mean([r["weave_deg"] for r in rows.values()])),
                        "left_m": float(np.mean([r["left_m"] for r in rows.values()])),
                        "fast_speed": float(np.mean([r["speed"] for v, r in rows.items() if v >= 0.4] or [0.0]))}}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--seeds", type=int, default=12, help="randomized robots per speed (default 12)")
    p.add_argument("--physics-from", default=PHYSICS_FROM, help="the run whose randomization makes the test robots")
    p.add_argument("--speeds", type=float, nargs="+", default=SPEEDS, help="forward commands, m/s (default 0.1-0.6)")
    p.add_argument("--json", help="save every number to this file (one run: its numbers; several: by run)")
    a = p.parse_args()
    results = {}
    print(f"{'policy':32s} {'':6s} {'cmd':>5s} {'reached':>8s} {'share':>6s} {'falls':>6s} {'side m':>7s} {'turn':>6s} {'weave':>6s}")
    with Pool(12) as pool:
        for run in a.runs:
            res = results[run] = measure(run, a.seeds, pool, a.physics_from, a.speeds)
            for mode, m in res.items():
                for v, r in m["rows"].items():
                    print(f"{Path(run).name:32s} {mode:6s} {float(v):5.2f} {r['speed']:+8.3f} {100 * r['share']:5.0f}% "
                          f"{r['falls']:2d}/{r['episodes']:<3d} {r['left_m']:7.3f} {r['turn_deg']:5.1f}° {r['weave_deg']:5.1f}°")
                s = m["summary"]
                print(f"{Path(run).name:32s} {mode:6s} summary: {100 * s['share']:.0f}% of the command, falls {s['falls']}/{s['episodes']}"
                      f" (+{m['standing_falls']} while standing), turn {s['turn_deg']:.1f}°, weave {s['weave_deg']:.1f}°,"
                      f" side {s['left_m']:.3f} m, speed at 0.4+ {s['fast_speed']:.3f} m/s", flush=True)
            print()
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(results[a.runs[0]] if len(a.runs) == 1 else results, indent=2))


if __name__ == "__main__":
    main()
