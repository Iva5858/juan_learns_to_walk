"""Fall test: episodes at 0.4 m/s on randomized robots, with and without random shoves, and what the
falls have in common (fell vs "broke" a hardware limit, the push before each one). The standard test,
the same for every policy: shoves up to 0.3 m/s per axis (no push curriculum), the full-difficulty
robot, the same seeds. By default on the measured real servos (EnvConfig.use_measured_servos).

    python scripts/analysis/falls.py checkpoints/juan_v7_12b_pushes_best
    python scripts/analysis/falls.py runs/a runs/b -n 50 --modes pushes      # just with shoves
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from biped_rl.runs import load_policy
from biped_rl.tasks.locomotion import BipedEnv, EnvConfig

MODES = {"pushes": 0.3, "no-pushes": 0.0}


def episode(env, model, norm, seed, speed):
    """Returns (outcome, seconds, mean speed, last shove before the end: (time, forward, left) or None)."""
    obs, _ = env.reset(seed=seed, options={"speed": speed})
    vx, last_push, seen = [], None, env.episode.last_push
    while True:
        obs, _, term, trunc, info = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
        if env.episode.last_push != seen:  # a shove this step
            seen = env.episode.last_push
            yaw = env.yaw()
            v = env.episode.last_push_vec
            last_push = (seen, np.cos(yaw) * v[0] + np.sin(yaw) * v[1], -np.sin(yaw) * v[0] + np.cos(yaw) * v[1])
        if term or trunc:
            break
        vx.append(info["forward_velocity"])
    outcome = "survived" if not term else "broke" if info["broke"] > 0.5 else "fell"
    return outcome, env.episode.t, float(np.mean(vx)) if vx else 0.0, last_push


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("-n", type=int, default=30, help="episodes per policy and mode")
    p.add_argument("--speed", type=float, default=0.4)
    p.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    p.add_argument("--best", action="store_true", help="the run's best_model instead of final_model")
    p.add_argument("--as-trained", action="store_true", help="each run's own servo model, not the measured one")
    a = p.parse_args()
    for run in map(Path, a.runs):
        cfg = EnvConfig.load(run)
        if not a.as_trained:
            cfg.use_measured_servos()
        cfg.randomization.enabled = True
        cfg.randomization.push_curriculum = False
        cfg.randomization.physics_curriculum = False
        model, norm = load_policy(run, cfg, best=a.best)
        for mode in a.modes:
            cfg.randomization.push_velocity = MODES[mode]
            env = BipedEnv(cfg.to_dict())
            rows = [episode(env, model, norm, 10_000 + i, a.speed) for i in range(a.n)]
            c = Counter(r[0] for r in rows)
            print(f"{run.name:36s} {mode:9s}  survived {c['survived']:2d}/{a.n}  fell {c['fell']:2d}  "
                  f"broke {c['broke']:2d}  mean speed {np.mean([r[2] for r in rows]):.2f} m/s", flush=True)
            # falls within 1.5 s of a shove, by the direction it came from (robot frame)
            dirs = Counter()
            for outcome, t, _, push in rows:
                if outcome != "survived" and push is not None and t - push[0] < 1.5:
                    fwd, left = push[1], push[2]
                    dirs["forward" if abs(fwd) >= abs(left) and fwd > 0 else "backward" if abs(fwd) >= abs(left)
                         else "left" if left > 0 else "right"] += 1
            if dirs:
                print(f"{'':47s}after a shove: {sum(dirs.values())} "
                      f"({', '.join(f'{k} {v}' for k, v in dirs.most_common())})", flush=True)


if __name__ == "__main__":
    main()
