"""Left/right step timing of a checkpoint: is it walking evenly or skipping?

    python scripts/analysis/step_rhythm.py runs/juan_v6                     # newest checkpoint
    python scripts/analysis/step_rhythm.py runs/juan_v6 path/to/rl_model_X_steps.zip
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from biped_rl.tasks.locomotion import BipedEnv, EnvConfig

run = Path(sys.argv[1])
ck = sys.argv[2] if len(sys.argv) > 2 else sorted(glob.glob(str(run / "checkpoints/rl_model_[0-9]*_steps.zip")),
                                                   key=lambda p: int(p.split("_")[-2]))[-1]
step = ck.split("_")[-2]
cfg = EnvConfig.from_dict(json.loads((run / "config.json").read_text()))
cfg.randomization.enabled = False
pol = PPO.load(ck, device="cpu")
norm = VecNormalize.load(ck.replace("rl_model_", "rl_model_vecnormalize_").replace(".zip", ".pkl"),
                         DummyVecEnv([lambda: BipedEnv(cfg.to_dict())]))
env = BipedEnv(cfg)
obs, _ = env.reset(seed=0, options={"speed": 0.4})
seq, dist = [], 0.0
for i in range(600):
    a, _ = pol.predict(norm.normalize_obs(obs), deterministic=True)
    obs, r, t, tr, info = env.step(a)
    c = env.episode.contact
    seq.append({(1, 1): "B", (1, 0): "L", (0, 1): "R", (0, 0): "-"}[(int(c[0]), int(c[1]))])
    if i >= 100:
        dist += info["forward_velocity"] * env.dt
full = seq[100:]
td = {0: [], 1: []}
for i in range(1, len(full)):
    for leg, on in ((0, "LB"), (1, "RB")):
        if full[i] in on and full[i - 1] not in on:
            td[leg].append(i)
ev = sorted([(i, 0) for i in td[0]] + [(i, 1) for i in td[1]])
lr, rl = [], []
for (i1, l1), (i2, l2) in zip(ev, ev[1:]):
    if l1 != l2:
        (lr if l1 == 0 else rl).append(i2 - i1)
s = "".join(full[50:150])
runs, cur, n = [], s[0], 0
for ch in s:
    if ch == cur:
        n += 1
    else:
        runs.append(f"{cur}{n}")
        cur, n = ch, 1
runs.append(f"{cur}{n}")
a, b = np.mean(lr) * 20, np.mean(rl) * 20
print(f"{int(step) / 1e6:.1f}M: {dist / (len(full) * env.dt):.2f} m/s | left->right {a:.0f} ms, right->left {b:.0f} ms "
      f"| asymmetry {abs(a - b) / (a + b):.0%} | pattern {' '.join(runs[:12])}")
