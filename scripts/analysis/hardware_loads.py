"""Hardware load check: per-joint servo torque/speed, foot impacts, current draw."""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from biped_rl.tasks.locomotion import BipedEnv, EnvConfig

run = Path(next((a for a in sys.argv[1:] if not a.startswith("--")), "runs/walk_v3")).resolve()
robust = "--robust" in sys.argv

# Record torque and joint speed at every physics substep (500 Hz), not just per control step.
rec = {"tau": [], "qd": []}
_mj_step = mujoco.mj_step


def recording_step(m, d):
    rec["tau"].append(d.ctrl.copy())
    rec["qd"].append(d.qvel[env_raw.robot.dof_idx].copy())
    _mj_step(m, d)


mujoco.mj_step = recording_step

cfg = EnvConfig.from_dict(json.loads((run / "config.json").read_text()))
cfg.randomization.enabled = robust
venv = DummyVecEnv([lambda: BipedEnv(cfg.to_dict())])
venv = VecNormalize.load(str(run / "best_vecnormalize.pkl"), venv)
venv.training, venv.norm_reward = False, False
policy = PPO.load(run / "best_model", device="cpu")
env_raw = venv.venv.envs[0]
names = [mujoco.mj_id2name(env_raw.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(env_raw.nu)]
stall, w_max, weight = cfg.robot.stall_torque, cfg.robot.no_load_speed, env_raw.robot.weight

print(f"policy: {run.name}/best_model, {'randomized physics + pushes' if robust else 'nominal physics'}, "
      f"robot weight {weight:.1f} N\n")
for speed in (0.3, 0.5, 0.8):
    rec["tau"].clear(); rec["qd"].clear()
    impacts, fell, dist = [], 0, 0.0
    for ep in range(3):
        venv.set_options({"speed": speed})  # VecEnv options only apply to the next reset
        obs = venv.reset()
        while True:
            a, _ = policy.predict(obs, deterministic=True)
            obs, _, done, info = venv.step(a)
            impacts.append(info[0]["peak_force_bw"] * weight)
            dist += info[0]["forward_velocity"] * env_raw.dt
            if done[0]:
                fell += info[0].get("TimeLimit.truncated", False) is False
                break
    tau, qd = np.abs(np.array(rec["tau"])), np.abs(np.array(rec["qd"]))
    t_sim = len(tau) * env_raw.model.opt.timestep
    current = (tau / stall * 3.0).sum(axis=1) + 8 * 0.1  # A: stall 3 A each at stall torque, +0.1 A idle
    print(f"=== command {speed} m/s: walked {dist / t_sim:.2f} m/s avg, falls {fell}/3 ===")
    print(f"{'joint':<12}{'peak Nm':>8}{'99% Nm':>8}{'peak %stall':>12}{'time >80%':>10}{'peak rad/s':>11}")
    for j, n in enumerate(names):
        print(f"{n:<12}{tau[:, j].max():>8.2f}{np.percentile(tau[:, j], 99):>8.2f}"
              f"{100 * tau[:, j].max() / stall:>11.0f}%{100 * np.mean(tau[:, j] > 0.8 * stall):>9.1f}%"
              f"{qd[:, j].max():>11.1f}")
    imp = np.array(impacts)
    print(f"foot landing force: typical (median) {np.median(imp):.0f} N, 99th pct {np.percentile(imp, 99):.0f} N, "
          f"max {imp.max():.0f} N  (= {imp.max() / weight:.1f}x body weight)")
    print(f"battery current (estimate): mean {current.mean():.1f} A, peak {current.max():.1f} A\n")
