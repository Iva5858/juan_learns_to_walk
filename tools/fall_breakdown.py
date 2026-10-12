"""Break a policy's robust-eval failures down by cause: fell vs "broke" (hardware limit), and the
direction of the push before each failure. Same seeds in every mode, so modes are comparable.

    python tools/fall_breakdown.py runs/juan_v3 full     -n 50 --out out/v3_full.jsonl
    python tools/fall_breakdown.py runs/juan_v3 nopush   -n 50 --out out/v3_nopush.jsonl    # randomized physics only
    python tools/fall_breakdown.py runs/juan_v3 pushonly -n 50 --out out/v3_pushonly.jsonl  # pushes on nominal physics
    python tools/summarize_falls.py out/
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from biped_rl.config import EnvConfig  # noqa: E402
from biped_rl.env import BipedEnv  # noqa: E402


def yaw_of(raw):
    R = raw._root_rot()
    return float(np.arctan2(R[1, 0], R[0, 0]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("mode", choices=["full", "nopush", "pushonly"])
    p.add_argument("-n", type=int, default=50)
    p.add_argument("--speed", type=float, default=0.4)
    p.add_argument("--best", action="store_true")
    p.add_argument("--physics-from", help="test on another run's robot: its randomization and hardware-limit "
                   "rule (e.g. runs/juan_v7_4_pushes, to test an older policy on the realistic robot)")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    run = PROJECT / a.run
    text = (run / "config.json").read_text()
    cfg = EnvConfig.from_dict(json.loads(text))
    cfg.randomization.enabled = True
    if a.physics_from:
        other = EnvConfig.from_dict(json.loads((PROJECT / a.physics_from / "config.json").read_text()))
        cfg.randomization = other.randomization
        for k in ("break_impact", "break_joint_speed", "break_requires_load"):
            setattr(cfg.rewards, k, getattr(other.rewards, k))
    # The standard test, the same for every run: random shoves up to 0.3 m/s per axis (no push
    # curriculum), and the full-difficulty robot (no physics curriculum).
    cfg.randomization.push_curriculum = False
    cfg.randomization.physics_curriculum = False
    cfg.randomization.push_velocity = 0.0 if a.mode == "nopush" else 0.3
    env = DummyVecEnv([lambda: BipedEnv(cfg.to_dict())])
    env = VecNormalize.load(str(run / ("best_vecnormalize.pkl" if a.best else "vecnormalize.pkl")), env)
    env.training, env.norm_reward = False, False
    policy = PPO.load(run / ("best_model" if a.best else "final_model"), device="cpu")
    raw = env.venv.envs[0]
    if a.mode == "pushonly":  # nominal physics, but pushes and random start heading stay on
        raw._randomize_physics = raw._set_nominal_physics

    with open(a.out, "w") as f:
        for ep in range(a.n):
            env.seed(10_000 + ep)
            env.set_options({"speed": a.speed})
            obs = env.reset()
            phys = {"stall": float(np.mean(raw._stall)), "servo_speed": float(np.mean(raw._speed)),
                    "latency_ms": raw._latency_steps * raw.model.opt.timestep * 1000,
                    "friction_scale": float(raw._friction_scale),
                    "gravity_tilt_deg": float(np.degrees(np.arccos(-raw.model.opt.gravity[2] / 9.81)))}
            pushes, vx, last_push_t, steps = [], [], -np.inf, 0
            peak_tilt = 0.0
            while True:
                action, _ = policy.predict(obs, deterministic=True)
                obs, _, done, info = env.step(action)
                info = info[0]
                steps += 1
                if done[0]:
                    break
                vx.append(info["forward_velocity"])
                g = raw._gravity_body()
                peak_tilt = max(peak_tilt, float(np.degrees(np.arcsin(min(1.0, np.linalg.norm(g[:2]))))))
                if raw._last_push != last_push_t:  # a shove was applied this step
                    last_push_t = raw._last_push
                    yaw = yaw_of(raw)
                    v = raw._last_push_vec
                    body = [np.cos(yaw) * v[0] + np.sin(yaw) * v[1], -np.sin(yaw) * v[0] + np.cos(yaw) * v[1]]
                    pushes.append({"t": float(last_push_t), "fwd": float(body[0]), "left": float(body[1])})
            t = steps * raw.dt
            ended_early = t < cfg.episode_seconds - 1e-6
            outcome = "survived"
            if ended_early:
                outcome = "broke" if info.get("broke", 0) > 0.5 else "fell"
            rec = {"ep": ep, "outcome": outcome, "time": t, "speed": float(np.mean(vx)) if vx else 0.0,
                   "n_pushes": len(pushes), "peak_tilt_deg": peak_tilt,
                   "end_impact_bw": float(info["peak_force_bw"]),
                   "end_joint_speed_ratio": float(info["max_joint_speed_ratio"]), **phys}
            if pushes:
                lp = pushes[-1]
                rec.update({"since_last_push": t - lp["t"], "last_push_fwd": lp["fwd"],
                            "last_push_left": lp["left"],
                            "last_push_mag": float(np.hypot(lp["fwd"], lp["left"]))})
            f.write(json.dumps(rec) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
