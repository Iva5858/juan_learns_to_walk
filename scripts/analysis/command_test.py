"""Does the policy do what it's told? Fixed commands (turning, sideways, backward, standing), on
nominal and on randomized physics (no pushes), and what the robot actually does.

    python scripts/analysis/command_test.py runs/juan_v7_4_pushes            # 3 episodes per command and physics
    python scripts/analysis/command_test.py runs/juan_v7_4_pushes -n 5

Speeds are measured along / across the target heading (the start heading, turned at the commanded
rate), averaged after the first 2 s (the change of gait). "heading err" is how far the robot's
facing ended from where the commanded turning should have brought it.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from biped_rl.tasks.locomotion import BipedEnv, EnvConfig

COMMANDS = [  # (label, forward m/s, sideways m/s, turn rad/s); None = stand
    ("straight 0.3", 0.3, 0.0, 0.0),
    ("turn in place left 0.3", 0.0, 0.0, 0.3),
    ("turn in place right 0.3", 0.0, 0.0, -0.3),
    ("turn in place left 0.6", 0.0, 0.0, 0.6),
    ("walk 0.2 + turn left 0.3", 0.2, 0.0, 0.3),
    ("walk 0.2 + turn right 0.3", 0.2, 0.0, -0.3),
    ("walk 0.2 + turn left 0.6", 0.2, 0.0, 0.6),
    ("sideways left 0.1", 0.0, 0.1, 0.0),
    ("sideways right 0.1", 0.0, -0.1, 0.0),
    ("sideways left 0.15", 0.0, 0.15, 0.0),
    ("sideways right 0.15", 0.0, -0.15, 0.0),
    ("backward 0.1", -0.1, 0.0, 0.0),
    ("backward 0.15", -0.15, 0.0, 0.0),
    ("stand", None, None, None),
]


def run(model, norm, cfg, command, seed, seconds=10.0, settle=2.0):
    env = BipedEnv(cfg.to_dict())
    options = {"stand": True} if command is None else {"command": command}
    obs, _ = env.reset(seed=seed, options=options)
    log = {"vx": [], "vy": [], "wz": [], "double": []}
    fell = False
    for step in range(int(seconds / env.dt)):
        obs, _, terminated, _, info = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
        if terminated:
            fell = True
            break
        if step * env.dt >= settle:
            log["vx"].append(info["forward_velocity"])
            log["vy"].append(info["lateral_velocity"])
            log["wz"].append(info["yaw_rate"])
            log["double"].append(info["double_support"])
    means = {k: float(np.mean(v)) if v else np.nan for k, v in log.items()}
    return means, fell, info["heading_error_deg"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("-n", type=int, default=3, help="episodes per command and physics")
    a = p.parse_args()
    run_dir = Path(a.run)
    base = EnvConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
    if base.command.mode != "omni":
        raise SystemExit("this run was trained on forward commands only")
    model = PPO.load(run_dir / "final_model", device="cpu")
    norm = VecNormalize.load(str(run_dir / "vecnormalize.pkl"), DummyVecEnv([lambda: BipedEnv(base.to_dict())]))
    norm.training = False

    for physics in ("nominal", "randomized"):
        cfg = EnvConfig.from_dict(base.to_dict())
        cfg.randomization.enabled = physics == "randomized"
        cfg.randomization.push_velocity = 0.0
        cfg.randomization.physics_curriculum = False  # full difficulty
        print(f"\n=== {physics} physics{' (full difficulty, no pushes)' if physics == 'randomized' else ''}, "
              f"{a.n} x 10 s per command ===")
        print(f"{'command':28s} {'fwd m/s':>14s} {'side m/s':>14s} {'turn rad/s':>14s} {'heading err':>11s} "
              f"{'double%':>7s} {'falls':>5s}")
        for label, vx, vy, wz in COMMANDS:
            command = None if vx is None else (vx, vy, wz)
            results = [run(model, norm, cfg, command, seed) for seed in range(a.n)]
            mean = lambda k: np.nanmean([r[0][k] for r in results])
            cvx, cvy, cwz = command or (0.0, 0.0, 0.0)
            falls = sum(r[1] for r in results)
            print(f"{label:28s} {mean('vx'):+6.2f} ({cvx:+.2f}) {mean('vy'):+6.2f} ({cvy:+.2f}) "
                  f"{mean('wz'):+6.2f} ({cwz:+.2f}) {np.mean([r[2] for r in results]):9.0f}° "
                  f"{100 * mean('double'):6.0f}% {falls:>3d}/{a.n}")
    print("\n(numbers in brackets: the command)")


if __name__ == "__main__":
    main()
