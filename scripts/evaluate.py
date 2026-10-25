"""Evaluate a trained policy: gait quality, smoothness, robustness, video (out/eval/<run>.mp4).

    python scripts/evaluate.py --run runs/<name>                       # nominal physics
    python scripts/evaluate.py --run runs/<name> --robust -n 50        # randomized physics + pushes: fall rate
    python scripts/evaluate.py --run runs/<name> --speed 0.3           # fixed speed command
    python scripts/evaluate.py --run runs/<name> --sweep               # speed actually reached per command
(To drive a policy yourself in the viewer: scripts/play.py.)
"""
import argparse
from pathlib import Path

import imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from biped_rl.tasks.locomotion import BipedEnv, EnvConfig
from biped_rl.training.callbacks import end_of_clip


def sweep(env, policy, raw, cfg, episodes=3):
    """Find the real top speed: run each command and report what the robot actually does."""
    print(f"{'command':>8} {'reached':>8} {'falls':>6} {'max yaw°':>9} {'sat%':>5} {'impact×BW':>10}")
    for cmd in np.arange(0.1, cfg.command.speed_limit + 1e-9, 0.1):
        speeds, falls, yaw, sat, impact = [], 0, [], [], []
        for _ in range(episodes):
            env.set_options({"speed": float(cmd)})  # VecEnv options only apply to the next reset
            obs, steps = env.reset(), 0
            while True:
                action, _ = policy.predict(obs, deterministic=True)
                obs, _, done, info = env.step(action)
                info = info[0]
                steps += 1
                speeds.append(info["forward_velocity"])
                yaw.append(info["heading_error_deg"])
                sat.append(info["saturation"])
                impact.append(info["peak_force_bw"])
                if done[0]:
                    falls += steps * raw.dt < cfg.episode_seconds - 1e-6
                    break
        print(f"{cmd:>8.1f} {np.mean(speeds):>8.2f} {falls:>3}/{episodes} {np.max(yaw):>9.1f} "
              f"{100 * np.mean(sat):>5.1f} {np.percentile(impact, 99):>10.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="checkpoints/juan_v7_12b_pushes_best")
    p.add_argument("--best", action="store_true", help="use best_model instead of final_model")
    p.add_argument("-n", "--episodes", type=int, default=5)
    p.add_argument("--speed", type=float, help="fixed speed command (default: max of the range)")
    p.add_argument("--robust", action="store_true", help="keep domain randomization and pushes on")
    p.add_argument("--sweep", action="store_true",
                   help="command 0.1, 0.2, ... up to speed_limit and report the speed actually reached")
    args = p.parse_args()

    run = Path(args.run)
    cfg = EnvConfig.load(run)
    cfg.randomization.enabled = args.robust
    speed = args.speed if args.speed is not None else cfg.command.speed_range[1]

    env = DummyVecEnv([lambda: BipedEnv(cfg.to_dict(), render_mode="rgb_array")])
    env = VecNormalize.load(str(run / ("best_vecnormalize.pkl" if args.best else "vecnormalize.pkl")), env)
    env.training, env.norm_reward = False, False
    policy = PPO.load(run / ("best_model" if args.best else "final_model"), device="cpu")
    raw = env.venv.envs[0]

    if args.sweep:
        sweep(env, policy, raw, cfg)
        return

    mass_g = raw.model.body_subtreemass[1] * 9.81
    rows, frames = [], []
    for ep in range(args.episodes):
        env.set_options({"speed": speed})  # VecEnv options only apply to the next reset
        obs = env.reset()
        log = {k: [] for k in ("forward_velocity", "flight", "double_support", "peak_force_bw",
                               "saturation", "action_delta", "energy", "heading_error_deg", "lateral_offset")}
        while True:
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, done, info = env.step(action)
            info = info[0]
            for k in log:
                log[k].append(info[k])
            if done[0]:
                break  # VecEnv auto-resets; final values come from info
            if ep == 0:
                frames.append(raw.render())
        steps = len(log["energy"])
        fell = steps * raw.dt < cfg.episode_seconds - 1e-6
        dist = np.sum(log["forward_velocity"]) * raw.dt  # along the heading, robust to slow turns
        rows.append({
            "fell": fell, "time": steps * raw.dt, "speed": np.mean(log["forward_velocity"]),
            "flight%": 100 * np.mean(log["flight"]), "double%": 100 * np.mean(log["double_support"]),
            "impact": np.percentile(log["peak_force_bw"], 99), "sat%": 100 * np.mean(log["saturation"]),
            "jerk": np.mean(log["action_delta"]), "power": np.sum(log["energy"]) / (steps * raw.dt),
            "CoT": np.sum(log["energy"]) / (mass_g * dist) if dist > 0.2 else np.nan,
            "head": np.max(log["heading_error_deg"]), "off": 100 * log["lateral_offset"][-1],
        })

    print(f"\ncommand {speed:.2f} m/s, {'randomized + pushes' if args.robust else 'nominal physics'}")
    header = f"{'ep':>3} {'fell':>5} {'time s':>7} {'speed':>6} {'flight%':>8} {'double%':>8} " \
             f"{'impact×BW':>10} {'sat%':>5} {'|Δa|':>6} {'power W':>8} {'CoT':>6} {'max yaw°':>8} {'drift cm':>8}"
    print(header)
    for i, r in enumerate(rows):
        print(f"{i:>3} {str(r['fell']):>5} {r['time']:>7.1f} {r['speed']:>6.2f} {r['flight%']:>8.1f} "
              f"{r['double%']:>8.1f} {r['impact']:>10.2f} {r['sat%']:>5.1f} {r['jerk']:>6.3f} "
              f"{r['power']:>8.2f} {r['CoT']:>6.2f} {r['head']:>8.1f} {r['off']:>8.1f}")
    print(f"\nfall rate: {100 * np.mean([r['fell'] for r in rows]):.0f}%  "
          f"mean speed: {np.mean([r['speed'] for r in rows]):.2f} m/s")

    out = Path(__file__).resolve().parents[1] / "out" / "eval" / f"{run.name}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, end_of_clip(frames, round(1 / raw.dt)), fps=round(1 / raw.dt))
    print(f"Video of episode 0: {out}")


if __name__ == "__main__":
    main()
