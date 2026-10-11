"""Which joint trips the 'back-driven' hardware limit, when in the stride, and what happens if
the limit doesn't end the episode.

python tools/joint_breakdown.py runs/juan_v3 full|nopush|pushonly|nominal -n 50 --out x.jsonl [--keep-limits]
Without --keep-limits the break limits are disabled, so we see whether the robot would have walked on.
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import biped_rl.env as envmod  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from biped_rl.config import EnvConfig  # noqa: E402
from biped_rl.env import BipedEnv  # noqa: E402

LIMIT = 1.3


class StepRecorder:
    """Stands in for the mujoco module inside env.py and records every physics substep."""

    def __init__(self):
        self.raw = None
        self.reset_step()

    def reset_step(self):
        self.peak = None       # per-joint peak |qd| / no-load speed this control step
        self.backdrive = None  # per-joint: torque opposed motion at that peak
        self.tau_at_peak = None

    def __getattr__(self, name):
        return getattr(mujoco, name)

    def mj_step(self, m, d):
        mujoco.mj_step(m, d)
        r = self.raw
        if r is None:
            return
        qd = d.qvel[r._dof_idx]
        ratio = np.abs(qd) / r._speed
        tau = d.ctrl[:]  # env writes the clipped servo torque into ctrl before stepping
        if self.peak is None:
            self.peak, self.backdrive, self.tau_at_peak = ratio.copy(), tau * qd < 0, tau.copy()
        else:
            up = ratio > self.peak
            self.peak[up] = ratio[up]
            self.backdrive[up] = (tau * qd < 0)[up]
            self.tau_at_peak[up] = tau[up]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("mode", choices=["full", "nopush", "pushonly", "nominal"])
    p.add_argument("-n", type=int, default=50)
    p.add_argument("--speed", type=float, default=0.4)
    p.add_argument("--keep-limits", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    rec = StepRecorder()
    envmod.mujoco = rec

    run = PROJECT / a.run
    text = (run / "config.json").read_text()
    cfg = EnvConfig.from_dict(json.loads(text))
    cfg.randomization.enabled = a.mode != "nominal"
    if a.mode == "nopush":
        cfg.randomization.push_velocity = 0.0
    if not a.keep_limits:
        cfg.rewards.break_joint_speed = 1e9
        cfg.rewards.break_impact = 1e9
    env = DummyVecEnv([lambda: BipedEnv(cfg.to_dict())])
    env = VecNormalize.load(str(run / "vecnormalize.pkl"), env)
    env.training, env.norm_reward = False, False
    policy = PPO.load(run / "final_model", device="cpu")
    raw = env.venv.envs[0]
    rec.raw = raw
    if a.mode == "pushonly":
        raw._randomize_physics = raw._set_nominal_physics
    m = raw.model
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.dof_jntid[i]) for i in raw._dof_idx]
    leg_of = [0 if n.startswith("l") else 1 for n in names]

    with open(a.out, "w") as f:
        for ep in range(a.n):
            env.seed(10_000 + ep)
            env.set_options({"speed": a.speed})
            obs = env.reset()
            di = raw._dof_idx
            phys = {"kp": raw._kp.tolist(), "kd": raw._kd.tolist(), "servo_speed": raw._speed.tolist(),
                    "stall": raw._stall.tolist(), "armature": m.dof_armature[di].tolist(),
                    "frictionloss": m.dof_frictionloss[di].tolist(),
                    "mass": float(m.body_subtreemass[1]), "latency_ms": raw._latency_steps * m.opt.timestep * 1e3}
            events, peaks, impacts, steps, vx = [], np.zeros(raw.nu), [], 0, []
            contact_hist, last_push = [], -np.inf
            while True:
                rec.reset_step()
                action, _ = policy.predict(obs, deterministic=True)
                obs, _, done, info = env.step(action)
                info = info[0]
                steps += 1
                if done[0]:
                    break
                vx.append(info["forward_velocity"])
                contact = raw._contact.copy()
                contact_hist.append(contact)
                if raw._last_push != last_push:
                    last_push = raw._last_push
                peaks = np.maximum(peaks, rec.peak)
                impacts.append(info["peak_force_bw"])
                over = np.where(rec.peak > LIMIT)[0]
                for j in over:
                    leg = leg_of[j]
                    # was this foot touching down right now (swing->stance in the last 2 steps)?
                    prev = contact_hist[-3][leg] if len(contact_hist) >= 3 else contact[leg]
                    events.append({"t": round(steps * raw.dt, 2), "joint": names[j], "ratio": float(rec.peak[j]),
                                   "backdriven": bool(rec.backdrive[j]),
                                   "tau_frac": float(abs(rec.tau_at_peak[j]) / raw._stall[j]),
                                   "foot": "stance" if contact[leg] else "swing",
                                   "touchdown": bool(contact[leg] and not prev),
                                   "since_push": float(steps * raw.dt - last_push),
                                   "tilt_deg": float(np.degrees(np.arcsin(min(1, np.linalg.norm(raw._gravity_body()[:2])))))})
            t = steps * raw.dt
            out = "survived" if t >= cfg.episode_seconds - 1e-6 else ("broke" if info.get("broke", 0) > .5 else "fell")
            f.write(json.dumps({"ep": ep, "outcome": out, "time": t, "speed": float(np.mean(vx)) if vx else 0,
                                "peak_ratio": dict(zip(names, peaks.tolist())),
                                "impact_p99": float(np.percentile(impacts, 99)) if impacts else 0,
                                "impact_max": float(max(impacts)) if impacts else 0,
                                "events": events, **phys}) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
