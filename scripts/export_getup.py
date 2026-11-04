"""Export a getup policy (biped_rl/tasks/getup.py) for the robot, and check it against the simulator.

    python scripts/export_getup.py runs/getup/juan_getup_21     # -> robot/policies/getup/juan_getup_21.npz

The same file format as the walking policies (robot/policy.py runs it unchanged), in its own folder so it is
never offered as a walking policy, with meta_task = "getup", its wide per-joint action range and its knee limit
(joint_lo). Checks, before saving:
  1. the numpy network gives the same actions as the trained PyTorch policy;
  2. the robot's controller builds exactly the simulator's observations from the face-down start (the pose
     you place the robot in), step by step, and the simulated robot gets up and stands.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from biped_rl.robots import PROJECT
from biped_rl.symmetry import mirror_maps
from biped_rl.tasks.getup import GetUpEnv, config
from biped_rl.tasks.locomotion.mdp import observations
from robot.joints import NAMES
from robot.policy import Controller, Policy


def export(run, out):
    saved = json.loads((run / "getup.json").read_text())
    knee = saved.get("start", {}).get("knee_min_deg")
    env = GetUpEnv(config(8.0).to_dict(), saved["weights"], {"knee_min_deg": knee})   # face down, nominal robot
    env.servo.action_delay = 3
    model = PPO.load(run / "final_model", device="cpu")
    norm = VecNormalize.load(str(run / "vecnormalize.pkl"), DummyVecEnv([lambda: env]))
    norm.training, norm.norm_reward = False, False
    p = model.policy
    linears = [m for m in p.mlp_extractor.policy_net if hasattr(m, "weight")] + [p.action_net]
    data = {"n_layers": len(linears), "actor_dim": env.actor_obs_dim, "obs_mean": norm.obs_rms.mean,
            "obs_var": norm.obs_rms.var, "obs_eps": norm.epsilon, "clip_obs": norm.clip_obs,
            "mirror": bool(getattr(p, "mirror", False))}
    for i, lin in enumerate(linears):
        data[f"w{i}"] = lin.weight.detach().numpy().astype(np.float64)
        data[f"b{i}"] = lin.bias.detach().numpy().astype(np.float64)
    if data["mirror"]:
        obs_perm, obs_sign, act_perm, act_sign = mirror_maps(env)
        data.update(obs_perm=np.array(obs_perm), obs_sign=np.array(obs_sign),
                    act_perm=np.array(act_perm), act_sign=np.array(act_sign))
    c = env.cfg.command
    meta = {"task": "getup", "control_hz": env.cfg.control_hz, "history": env.cfg.history,
            "action_scale": env.action_scale, "action_filter": env.cfg.action_filter, "home": env.robot.home_joint,
            "joint_lo": env.robot.joint_lo, "joint_hi": env.robot.joint_hi, "gait_period": c.gait_period,
            "gait_period_fast": c.gait_period_fast, "fast_speed": c.fast_speed, "joints": np.array(NAMES),
            "run": run.name, "forward_range": (0.0, 0.0), "lateral_range": (0.0, 0.0), "yaw_rate_range": (0.0, 0.0)}
    data.update({f"meta_{k}": np.asarray(v) for k, v in meta.items()})
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **data)
    policy = Policy(tmp)

    # 1. network
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    worst = 0.0
    for _ in range(300):
        torch_a = model.predict(norm.normalize_obs(obs), deterministic=True)[0]
        worst = max(worst, np.abs(policy(obs) - np.clip(torch_a, -1, 1)).max())
        obs, _, term, trunc, _ = env.step(torch_a + 0.3 * rng.standard_normal(env.nu))
        if term or trunc:
            obs, _ = env.reset()
    print(f"1. network: max action difference numpy vs PyTorch {worst:.2e}")

    # 2. closed loop through the robot controller, from the face-down start
    env.reset(seed=1)
    ctrl = Controller(policy)
    ctrl.start(env.robot_readings())
    obs = observations.compute(env)
    obs_err = act_err = 0.0
    up_at = None
    for _ in range(env._max_steps):
        ctrl.tick(env.robot_readings())
        obs_err = max(obs_err, np.abs(ctrl.last_obs - obs[:env.actor_obs_dim]).max())
        sim_a = np.clip(model.predict(norm.normalize_obs(obs), deterministic=True)[0], -1, 1)
        act_err = max(act_err, np.abs(ctrl.last_action - sim_a).max())
        obs, _, term, trunc, info = env.step(sim_a)
        act_err = max(act_err, np.abs(ctrl.target - env.episode.target_buf[-1]).max())
        if up_at is None and env.episode.stood_for >= 0.5:
            up_at = env.episode.t
        if term or trunc:
            break
    print(f"2. robot controller in the simulator: max observation difference {obs_err:.2e}, max action / target "
          f"difference {act_err:.2e}, " + (f"stood up at {up_at:.1f} s" if up_at else "DID NOT GET UP"))
    if not (worst < 1e-5 and obs_err < 1e-6 and act_err < 1e-5 and up_at):
        tmp.unlink()
        raise SystemExit("MISMATCH or no getup: not exported")
    tmp.replace(out)
    print(f"saved {out} ({out.stat().st_size // 1024} kB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    run = Path(p.parse_args().run)
    export(run, PROJECT / "robot" / "policies" / "getup" / f"{run.name}.npz")


if __name__ == "__main__":
    main()
