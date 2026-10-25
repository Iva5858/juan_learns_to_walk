"""Straight walking: forward distance, sideways drift and turning over 10 s at fixed speeds, on the
nominal robot and on randomized robots (no pushes). Policies are run folders (final_model.zip +
vecnormalize.pkl + config.json); the robot comes from --physics-from (servo model, randomization).

    python scripts/analysis/straight.py runs/juan_v7_12b_pushes runs/juan_v7_8_best \
        --physics-from runs/juan_v7_12b_pushes
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from biped_rl.tasks.locomotion import BipedEnv, EnvConfig  # noqa: E402


def walk(model, norm, cfg, speed, seed, seconds=10.0):
    env = BipedEnv(cfg.to_dict())
    obs, _ = env.reset(seed=seed, options={"speed": speed})
    R = env.root_rot()
    y0, p0 = math.atan2(R[1, 0], R[0, 0]), env.data.qpos[:2].copy()
    fell = False
    for _ in range(int(seconds / env.dt)):
        obs, _, term, _, _ = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
        if term:
            fell = True
            break
    d, R = env.data.qpos[:2] - p0, env.root_rot()
    fwd = d[0] * math.cos(y0) + d[1] * math.sin(y0)
    left = -d[0] * math.sin(y0) + d[1] * math.cos(y0)
    turn = math.degrees((math.atan2(R[1, 0], R[0, 0]) - y0 + math.pi) % (2 * math.pi) - math.pi)
    return fwd, left, turn, fell


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--physics-from", required=True)
    p.add_argument("--speeds", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    p.add_argument("--seeds", type=int, default=6)
    a = p.parse_args()
    other = EnvConfig.from_dict(json.loads((Path(a.physics_from) / "config.json").read_text()))
    print(f"{'policy':42s} {'m/s':>4s} | nominal: {'fwd m':>5s} {'left m':>6s} {'turn':>6s} | "
          f"randomized x{a.seeds}: {'fwd m':>5s} {'left m':>6s} {'turn':>6s} (range)      falls")
    for run in map(Path, a.runs):
        cfg = EnvConfig.from_dict(json.loads((run / "config.json").read_text()))
        cfg.randomization = other.randomization
        for k in ("stall_torque", "no_load_speed", "kp", "kd"):
            setattr(cfg.robot, k, getattr(other.robot, k))
        cfg.randomization.push_velocity = 0.0
        cfg.randomization.push_curriculum = False
        cfg.randomization.physics_curriculum = False
        model = PPO.load(run / "final_model", device="cpu")
        for speed in a.speeds:
            rows = []
            for enabled, seeds in ((False, [0]), (True, range(a.seeds))):
                c = EnvConfig.from_dict(cfg.to_dict())
                c.randomization.enabled = enabled
                norm = VecNormalize.load(str(run / "vecnormalize.pkl"), DummyVecEnv([lambda: BipedEnv(c.to_dict())]))
                norm.training = False
                rows.append(np.array([walk(model, norm, c, speed, s) for s in seeds], dtype=float))
            n, r = rows
            print(f"{run.name:42s} {speed:4.2f} | {'':9s}{n[0,0]:5.2f} {n[0,1]:+6.2f} {n[0,2]:+5.1f}° | "
                  f"{'':14s}{r[:,0].mean():5.2f} {r[:,1].mean():+6.2f} {r[:,2].mean():+5.1f}° ({r[:,2].min():+.0f}..{r[:,2].max():+.0f}°)"
                  f"  {int(n[:,3].sum() + r[:,3].sum())}/{1 + a.seeds}", flush=True)


if __name__ == "__main__":
    main()
