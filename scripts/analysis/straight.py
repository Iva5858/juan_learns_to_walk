"""Straight walking: forward distance, sideways drift, turning and heading swing (the weave: peak to
peak, with the slow drift removed) over 10 s at fixed speeds, on the nominal robot and on randomized
robots (no pushes). By default on the measured real servos (EnvConfig.use_measured_servos).

    python scripts/analysis/straight.py checkpoints/juan_v7_12b_pushes_best
    python scripts/analysis/straight.py runs/a runs/b --speeds 0.2 --seeds 4 --as-trained
"""
import argparse
from pathlib import Path

import numpy as np

from biped_rl.runs import load_policy
from biped_rl.tasks.locomotion import BipedEnv, EnvConfig


def walk(model, norm, cfg, speed, seed, seconds=10.0):
    env = BipedEnv(cfg.to_dict())
    obs, _ = env.reset(seed=seed, options={"speed": speed})
    y0, p0 = env.yaw(), env.data.qpos[:2].copy()
    fell, yaws = False, []
    for _ in range(int(seconds / env.dt)):
        obs, _, term, _, _ = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
        if term:
            fell = True
            break
        yaws.append(env.yaw())
    fwd, left, turn = env.moved_since(p0, y0)
    # weave: after the first 2 s, with the steady drift removed
    y = np.degrees(np.unwrap(yaws[100:])) if len(yaws) > 150 else np.zeros(2)
    x = np.arange(len(y))
    swing = float(np.ptp(y - np.polyval(np.polyfit(x, y, 1), x)))
    return fwd, left, turn, swing, fell


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--speeds", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--as-trained", action="store_true", help="each run's own servo model, not the measured one")
    a = p.parse_args()
    print(f"{'policy':36s} {'m/s':>4s} | nominal: {'fwd m':>5s} {'left m':>6s} {'turn':>6s} | "
          f"randomized x{a.seeds}: {'fwd m':>5s} {'left m':>6s} {'turn':>6s} (range)  {'swing':>6s}  falls")
    for run in map(Path, a.runs):
        cfg = EnvConfig.load(run)
        if not a.as_trained:
            cfg.use_measured_servos()
        cfg.randomization.push_velocity = 0.0
        cfg.randomization.physics_curriculum = False
        model, norm = load_policy(run, cfg)
        for speed in a.speeds:
            rows = []
            for enabled, seeds in ((False, [0]), (True, range(a.seeds))):
                c = EnvConfig.from_dict(cfg.to_dict())
                c.randomization.enabled = enabled
                rows.append(np.array([walk(model, norm, c, speed, s) for s in seeds], dtype=float))
            n, r = rows
            print(f"{run.name:36s} {speed:4.2f} | {'':9s}{n[0,0]:5.2f} {n[0,1]:+6.2f} {n[0,2]:+5.1f}° | "
                  f"{'':14s}{r[:,0].mean():5.2f} {r[:,1].mean():+6.2f} {r[:,2].mean():+5.1f}° ({r[:,2].min():+.0f}..{r[:,2].max():+.0f}°)"
                  f"  {r[:,3].mean():5.1f}°  {int(n[:,4].sum() + r[:,4].sum())}/{1 + a.seeds}", flush=True)


if __name__ == "__main__":
    main()
