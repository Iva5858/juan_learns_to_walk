"""Left vs right gait comparison: step timing, step length, foot lift, joint posture.

    python scripts/analysis/gait_symmetry.py runs/walk_v4
"""
import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from biped_rl.tasks.locomotion import BipedEnv, EnvConfig
from biped_rl.tasks.locomotion.mdp import commands

run = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/walk_v3")
speed = 0.4
cfg = EnvConfig.from_dict(json.loads((run / "config.json").read_text()))
cfg.randomization.enabled = False
pol = PPO.load(run / "final_model", device="cpu")
norm = VecNormalize.load(str(run / "vecnormalize.pkl"), DummyVecEnv([lambda: BipedEnv(cfg.to_dict())]))
stats = {s: {k: [] for k in ("stance_s", "swing_s", "step_cm", "lift_mm")} for s in ("left", "right")}
joints, speeds = {}, []
for seed in (0, 3, 7):
    env = BipedEnv(cfg)
    obs, _ = env.reset(seed=seed, options={"speed": speed})
    names = [env.model.actuator(i).name for i in range(env.nu)]
    contact, sole_x, clear, q = [], [], [], []
    for i in range(1000):
        a, _ = pol.predict(norm.normalize_obs(obs), deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        contact.append(env.episode.contact.copy()); clear.append(env.episode.clearance.copy())
        sole_x.append(env.data.xpos[env.robot.foot_bodies, 0].copy()); q.append(env.data.qpos[env.robot.qpos_idx].copy())
        speeds.append(info["forward_velocity"])
        if term or trunc:
            break
    contact, clear, sole_x, q = map(np.array, (contact, clear, sole_x, q))
    for j, n in enumerate(names):
        joints.setdefault(n, []).append(q[100:, j])
    for leg, side in enumerate(("left", "right")):
        c = contact[100:, leg]
        edges = np.flatnonzero(np.diff(c)) + 1
        starts = np.concatenate([[0], edges])
        for vals, st in list(zip(np.split(c, edges), starts))[1:-1]:
            if vals[0] == 1:
                stats[side]["stance_s"].append(len(vals) * env.dt)
            else:
                stats[side]["swing_s"].append(len(vals) * env.dt)
                seg = slice(100 + st, 100 + st + len(vals) + 1)
                stats[side]["step_cm"].append(100 * (sole_x[seg, leg][-1] - sole_x[seg, leg][0]))
                stats[side]["lift_mm"].append(1000 * clear[seg, leg].max())
print(f"{run.name} final model, nominal physics, {speed} m/s command: walked {np.mean(speeds):.2f} m/s")
print(f"{'':16}{'left':>8}{'right':>8}")
for k, label in (("stance_s", "stance time s"), ("swing_s", "swing time s"),
                 ("step_cm", "step length cm"), ("lift_mm", "foot lift mm")):
    print(f"{label:16}{np.median(stats['left'][k]):>8.2f}{np.median(stats['right'][k]):>8.2f}")
print(f"steps per second per leg: {len(stats['left']['swing_s']) / (len(speeds) * env.dt - 6):.1f} "
      f"(gait clock asks for {1 / commands.gait_period(env):.1f})")
print("mean joint angle (deg), left vs right:")
for jn in ("hip_roll", "hip_pitch", "knee", "ankle"):
    sides = ("l", "r") if "l_knee" in joints else ("left", "right")  # placeholder or Onshape names
    l, r = (np.degrees(np.concatenate(joints[f"{s}_{jn}"])).mean() for s in sides)
    print(f"  {jn:<10}{l:>8.1f}{r:>8.1f}" + ("   (mirror: should be opposite sign)" if jn == "hip_roll" else ""))
