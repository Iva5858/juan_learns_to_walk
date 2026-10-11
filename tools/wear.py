"""Wear per meter walked: servo direction reversals, footsteps, joint range. Compare runs.

    python tools/wear.py runs/walk_v3 runs/walk_v4 runs/walk_v5
"""
import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biped_rl.config import EnvConfig  # noqa: E402
from biped_rl.env import BipedEnv  # noqa: E402

runs = sys.argv[1:] or ["runs/walk_v3"]
speed = 0.4
rows = {}
for run in runs:
    cfg = EnvConfig.from_dict(json.loads(Path(run, "config.json").read_text()))
    cfg.randomization.enabled = False
    pol = PPO.load(Path(run, "final_model"), device="cpu")
    norm = VecNormalize.load(str(Path(run, "vecnormalize.pkl")), DummyVecEnv([lambda: BipedEnv(cfg.to_dict())]))
    env = BipedEnv(cfg)
    obs, _ = env.reset(seed=0, options={"speed": speed})
    names = [env.model.actuator(i).name for i in range(env.nu)]
    qd, q, dist, touchdowns, prev = [], [], 0.0, 0, env._contact.copy()
    for i in range(1000):
        a, _ = pol.predict(norm.normalize_obs(obs), deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        if i >= 100:  # skip the first 2 s (starting up)
            qd.append(env.data.qvel[env._dof_idx].copy()); q.append(env.data.qpos[env._qpos_idx].copy())
            dist += info["forward_velocity"] * env.dt
            touchdowns += int(((env._contact == 1) & (prev == 0)).sum())
        prev = env._contact.copy()
        if term or trunc:
            break
    qd, q = np.array(qd), np.array(q)
    sign = np.sign(np.where(np.abs(qd) > 0.2, qd, 0))
    rev = np.mean([np.sum(x[1:] != x[:-1]) for x in (sign[sign[:, j] != 0, j] for j in range(env.nu))])
    rng = np.degrees(np.percentile(q, 95, axis=0) - np.percentile(q, 5, axis=0))
    side = ("l_", "r_") if "l_" + "knee" in names else ("left_", "right_")  # placeholder or Onshape names
    pick = lambda j: rng[[names.index(side[0] + j), names.index(side[1] + j)]].mean()
    rows[Path(run).name] = {
        "walked m/s": dist / (len(qd) * env.dt),
        "reversals per joint per m": rev / dist,
        "footsteps per m": touchdowns / dist,
        "step length cm (per foot)": 100 * 2 * dist / max(touchdowns, 1),
        "hip pitch range deg": pick("hip_pitch"),
        "knee range deg": pick("knee"),
        "ankle range deg": pick("ankle"),
    }
print(f"wear at a {speed} m/s command, nominal physics, final models (lower per-meter = less wear)")
print(f"{'':28}" + "".join(f"{n:>10}" for n in rows))
for k in next(iter(rows.values())):
    print(f"{k:28}" + "".join(f"{rows[n][k]:>10.2f}" for n in rows))
