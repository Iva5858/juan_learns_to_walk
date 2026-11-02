"""Walking the way the real robot does it: standing still at home, the policy taking over (start_policy),
1 s standing, then a straight command for 8 s. On randomized robots with the measured servos, without and
with 0.3 m/s shoves. Every policy is tested on the same robots: the randomization of --physics-from (default
juan_v7_13_straight), not the run's own (a run trained on harder robots would otherwise also be tested on them).

Tests that start from the training start state (a perturbed crouch, already moving) count falls the real
robot never sees: juan_v7_13 lost 58 of its 77 failures in their first 1.5 s. This one counts only what
happens while walking.

Per speed: the speed reached (share of the command), falls while walking (of them, hardware-limit stops),
sideways drift, heading drift, the weave (heading swing peak to peak, drift removed), and the gait: steps per
second, step length and foot lift. In the summary also the mean speed at commands of 0.4 m/s and up.

    python scripts/analysis/walk_test.py runs/v7/juan_v7_13_straight runs/v7/juan_v7_24_speed
    python scripts/analysis/walk_test.py runs/a --speeds -0.1 -0.15 --nominal     # backward, nominal robot
    python scripts/analysis/walk_test.py runs/base runs/new runs/new_best --verdict --json out/a.json

--verdict judges every run against the first (scripts/finetune.sh): KEEP if it falls no more (calm +2, with
shoves +3), weaves at most 3 deg more, and is better at something: >= 3 cm/s faster at 0.4 m/s and up, 3 fewer
falls, or 2 deg less weave.
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
    if physics_from:
        cfg.randomization = EnvConfig.load(physics_from).randomization
    else:
        cfg.randomization.enabled = False
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
    act = lambda o: model.predict(norm.normalize_obs(o), deterministic=True)[0]
    for _ in range(int(STAND_S / env.dt)):
        obs, _, term, _, _ = env.step(act(obs))
        if term:
            return None   # fell standing: not a walking result (counted apart)
    env.set_command(speed)
    y0, p0, yaws, lift, info = env.yaw(), env.data.qpos[:2].copy(), [], [], {}
    t0, steps = env.episode.t, 0
    for _ in range(int(WALK_S / env.dt)):
        obs, _, term, _, info = env.step(act(obs))
        steps += int(env.episode.td_hist[-1])
        lift.append(env.episode.clearance.max())
        if term:
            break
        yaws.append(env.yaw())
    fwd, left, turn = env.moved_since(p0, y0)
    y = np.degrees(np.unwrap(yaws[50:])) if len(yaws) > 100 else np.zeros(2)  # after the first second
    x = np.arange(len(y))
    weave = float(np.ptp(y - np.polyval(np.polyfit(x, y, 1), x)))
    t = max(env.episode.t - t0, env.dt)
    env.close()
    return {"speed": fwd / t, "fell": bool(term), "broke": bool(term and info.get("broke")), "left": left,
            "turn": abs(turn), "weave": weave, "steps_per_s": steps / t, "step_m": abs(fwd) / max(steps, 1),
            "lift_m": float(np.percentile(lift, 90))}


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
            mean = lambda key, of=eps: float(np.mean([e[key] for e in of]))
            rows[v] = {"speed": mean("speed"), "share": mean("speed") / v,
                       "falls": sum(e["fell"] for e in eps), "stops": sum(e["broke"] for e in eps), "episodes": len(eps),
                       "left_m": float(np.mean([abs(e["left"]) for e in done])), "turn_deg": mean("turn", done),
                       "weave_deg": mean("weave", done), "steps_per_s": mean("steps_per_s", done),
                       "step_m": mean("step_m", done), "lift_m": mean("lift_m", done)}
        avg = lambda key, rows=rows: float(np.mean([r[key] for r in rows.values()]))
        out["shoves" if shoves else "calm"] = {
            "rows": {str(k): v for k, v in rows.items()}, "standing_falls": standing_falls,
            "summary": {"share": avg("share"), "falls": sum(r["falls"] for r in rows.values()),
                        "episodes": sum(r["episodes"] for r in rows.values()), "turn_deg": avg("turn_deg"),
                        "weave_deg": avg("weave_deg"), "left_m": avg("left_m"),
                        "fast_speed": float(np.mean([r["speed"] for v, r in rows.items() if v >= 0.4] or [0.0]))}}
    return out


def verdict(base, new):
    """KEEP or NOT BETTER for `new` against `base` (see the module docstring), with the numbers."""
    P, C = ({m: r[m]["summary"] for m in ("calm", "shoves")} for r in (base, new))
    safe = (C["calm"]["falls"] <= P["calm"]["falls"] + 2 and C["shoves"]["falls"] <= P["shoves"]["falls"] + 3
            and C["calm"]["weave_deg"] <= P["calm"]["weave_deg"] + 3)
    falls = lambda S: S["calm"]["falls"] + S["shoves"]["falls"]
    better = (C["calm"]["fast_speed"] >= P["calm"]["fast_speed"] + 0.03 or falls(C) <= falls(P) - 3
              or C["calm"]["weave_deg"] <= P["calm"]["weave_deg"] - 2)
    return (f"{'KEEP' if safe and better else 'NOT BETTER'} (speed at 0.4+ {P['calm']['fast_speed']:.3f} -> "
            f"{C['calm']['fast_speed']:.3f} m/s; falls calm {P['calm']['falls']} -> {C['calm']['falls']}, shoves "
            f"{P['shoves']['falls']} -> {C['shoves']['falls']}; weave {P['calm']['weave_deg']:.1f} -> "
            f"{C['calm']['weave_deg']:.1f} deg; share {100 * P['calm']['share']:.0f} -> {100 * C['calm']['share']:.0f}%)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--seeds", type=int, default=12, help="randomized robots per speed (default 12)")
    p.add_argument("--speeds", type=float, nargs="+", default=SPEEDS, help="forward commands, m/s (default 0.1-0.6; "
                   "negative = backward)")
    p.add_argument("--physics-from", default=PHYSICS_FROM, help="the run whose randomization makes the test robots")
    p.add_argument("--nominal", action="store_true", help="the nominal robot instead (no randomization)")
    p.add_argument("--verdict", action="store_true", help="judge every run against the first")
    p.add_argument("--json", help="save every number to this file (one run: its numbers; several: by run)")
    a = p.parse_args()
    physics_from = None if a.nominal else a.physics_from
    seeds = 1 if a.nominal else a.seeds
    results = {}
    print(f"{'policy':28s} {'':6s} {'cmd':>5s} {'reached':>8s} {'share':>6s} {'falls':>7s} {'side m':>7s} {'turn':>6s} "
          f"{'weave':>6s} {'steps/s':>7s} {'step':>6s} {'lift':>5s}")
    with Pool(12) as pool:
        for run in a.runs:
            res = results[run] = measure(run, seeds, pool, physics_from, a.speeds)
            for mode, m in res.items():
                for v, r in m["rows"].items():
                    stops = f"({r['stops']})" if r["stops"] else ""
                    print(f"{Path(run).name:28s} {mode:6s} {float(v):5.2f} {r['speed']:+8.3f} {100 * r['share']:5.0f}% "
                          f"{r['falls']:2d}/{r['episodes']:<3d}{stops:4s} {r['left_m']:6.3f} {r['turn_deg']:5.1f}° "
                          f"{r['weave_deg']:5.1f}° {r['steps_per_s']:7.1f} {100 * r['step_m']:4.1f}cm {1000 * r['lift_m']:3.0f}mm")
                s = m["summary"]
                print(f"{Path(run).name:28s} {mode:6s} summary: {100 * s['share']:.0f}% of the command, falls {s['falls']}/"
                      f"{s['episodes']} (+{m['standing_falls']} while standing), turn {s['turn_deg']:.1f}°, weave "
                      f"{s['weave_deg']:.1f}°, side {s['left_m']:.3f} m, speed at 0.4+ {s['fast_speed']:.3f} m/s", flush=True)
            print()
    if a.verdict:
        for run in a.runs[1:]:
            print(f"=== {Path(run).name} vs {Path(a.runs[0]).name}: {verdict(results[a.runs[0]], results[run])}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(results[a.runs[0]] if len(a.runs) == 1 else results, indent=2))


if __name__ == "__main__":
    main()
