"""Speed per command, forward and backward, with the gait it uses: on randomized robots with the
measured real servos (EnvConfig.use_measured_servos), no shoves, 10 s per episode.

For each command: the speed reached along the commanded direction (and as a share of the command),
falls, torso lean, how often both feet are on the ground, and how high the feet lift.

    python scripts/analysis/speeds.py checkpoints/juan_v7_13_straight runs/juan_v7_16_best
    python scripts/analysis/speeds.py runs/a --json out/a_speeds.json     # also save the numbers
"""
import argparse
import json
from pathlib import Path

import numpy as np

from biped_rl.runs import load_policy
from biped_rl.tasks.locomotion import BipedEnv, EnvConfig

FORWARD = (0.1, 0.2, 0.3, 0.4)
BACKWARD = (-0.1, -0.15, -0.2)


def measure(run, seeds=6, seconds=10.0, best=False):
    cfg = EnvConfig.load(run).use_measured_servos()
    cfg.randomization.push_velocity = 0.0
    cfg.randomization.physics_curriculum = False
    model, norm = load_policy(run, cfg, best=best)
    rows = {}
    for cmd in FORWARD + BACKWARD:
        speed, falls, lean, both, lift, n = [], 0, [], 0, [], 0
        for seed in range(seeds):
            env = BipedEnv(cfg.to_dict())
            obs, _ = env.reset(seed=100 + seed, options={"command": (cmd, 0.0, 0.0)})
            p0, y0 = env.data.qpos[:2].copy(), env.yaw()
            for _ in range(int(seconds / env.dt)):
                obs, _, term, _, _ = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
                g = env.gravity_body()
                lean.append(np.degrees(np.arcsin(min(1.0, np.linalg.norm(g[:2])))))
                both += env.episode.contact.sum() == 2
                lift.append(env.episode.clearance.max())
                n += 1
                if term:
                    falls += 1
                    break
            speed.append(env.moved_since(p0, y0)[0] / env.episode.t)
        rows[cmd] = {"speed": float(np.mean(speed)), "share": float(np.mean(speed) / cmd), "falls": falls,
                     "episodes": seeds, "lean_deg": float(np.median(lean)), "both_feet_pct": 100.0 * both / n,
                     "lift_mm": 1000.0 * float(np.median(lift))}
    return rows


def summary(rows):
    """Mean share of the command reached, forward and backward (falls count as what they reached)."""
    fwd = float(np.mean([rows[c]["share"] for c in FORWARD]))
    back = float(np.mean([rows[c]["share"] for c in BACKWARD]))
    falls = sum(r["falls"] for r in rows.values())
    return {"forward_share": fwd, "backward_share": back, "falls": falls, "episodes": sum(r["episodes"] for r in rows.values())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--best", action="store_true", help="the runs' best_model instead of final_model")
    p.add_argument("--json", help="save every number to this file")
    a = p.parse_args()
    out = {}
    print(f"{'policy':30s} {'command':>8s} {'reached':>8s} {'share':>6s} {'falls':>6s} {'lean':>6s} {'both feet':>9s} {'lift':>7s}")
    for run in a.runs:
        rows = measure(run, a.seeds, best=a.best)
        for cmd, r in rows.items():
            print(f"{Path(run).name:30s} {cmd:+8.2f} {r['speed']:+8.3f} {100 * r['share']:5.0f}% {r['falls']:3d}/{r['episodes']:<2d}"
                  f" {r['lean_deg']:5.1f}° {r['both_feet_pct']:8.0f}% {r['lift_mm']:5.1f}mm", flush=True)
        s = summary(rows)
        print(f"{Path(run).name:30s} summary: forward {100 * s['forward_share']:.0f}% of the command, "
              f"backward {100 * s['backward_share']:.0f}%, falls {s['falls']}/{s['episodes']}\n", flush=True)
        out[run] = {"rows": {str(k): v for k, v in rows.items()}, "summary": s}
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
