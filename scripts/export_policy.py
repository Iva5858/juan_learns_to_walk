"""Export a trained policy for the robot (numpy only), and check it against the simulator.

    python scripts/export_policy.py runs/juan_v7_8_best      # -> robot/policies/juan_v7_8_best.npz

Checks, before saving:
  1. the numpy network gives the same actions as the trained PyTorch policy;
  2. the robot's controller (robot/policy.py) builds exactly the simulator's observations and
     drives the simulated Juan through stand / walk / turn / sideways / back / stand, step by step
     the same as the training code.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))  # the robot/ package isn't installed
from biped_rl.symmetry import mirror_maps  # noqa: E402
from biped_rl.tasks.locomotion import BipedEnv, EnvConfig  # noqa: E402
from biped_rl.tasks.locomotion.mdp import observations  # noqa: E402
from robot.joints import NAMES  # noqa: E402
from robot.policy import Controller, Policy  # noqa: E402

# (seconds, forward, sideways, turn): the closed-loop check
SCRIPT = [(2, 0, 0, 0), (4, 0.3, 0, 0), (3, 0.2, 0, 0.4), (3, 0, 0.1, 0), (2, -0.1, 0, 0), (2, 0, 0, 0)]


def export(run, out):
    cfg = EnvConfig.from_dict(json.loads((run / "config.json").read_text()))
    cfg.randomization.enabled = False
    env = BipedEnv(cfg.to_dict())
    model = PPO.load(run / "final_model", device="cpu")
    norm = VecNormalize.load(str(run / "vecnormalize.pkl"), DummyVecEnv([lambda: BipedEnv(cfg.to_dict())]))
    norm.training = False
    p = model.policy
    names = [env.model.joint(int(j)).name for j in env.model.dof_jntid[env.robot.dof_idx]]
    if tuple(names) != NAMES:
        raise SystemExit(f"joint order differs from robot/joints.py: {names}")

    linears = [m for m in p.mlp_extractor.policy_net if hasattr(m, "weight")] + [p.action_net]
    data = {"n_layers": len(linears), "actor_dim": env.actor_obs_dim,
            "obs_mean": norm.obs_rms.mean, "obs_var": norm.obs_rms.var, "obs_eps": norm.epsilon,
            "clip_obs": norm.clip_obs, "mirror": bool(getattr(p, "mirror", False))}
    for i, lin in enumerate(linears):
        data[f"w{i}"] = lin.weight.detach().numpy().astype(np.float64)
        data[f"b{i}"] = lin.bias.detach().numpy().astype(np.float64)
    if data["mirror"]:
        obs_perm, obs_sign, act_perm, act_sign = mirror_maps(env)
        data.update(obs_perm=np.array(obs_perm), obs_sign=np.array(obs_sign),
                    act_perm=np.array(act_perm), act_sign=np.array(act_sign))
    c = cfg.command
    meta = {"control_hz": cfg.control_hz, "history": cfg.history, "action_scale": cfg.action_scale,
            "action_filter": cfg.action_filter, "home": env.robot.home_joint, "joint_lo": env.robot.joint_lo,
            "joint_hi": env.robot.joint_hi, "gait_period": c.gait_period, "gait_period_fast": c.gait_period_fast,
            "fast_speed": c.fast_speed, "joints": np.array(NAMES), "run": run.name,
            "forward_range": (-c.backward_speed, c.speed_range[1]), "lateral_range": c.lateral_range,
            "yaw_rate_range": c.yaw_rate_range}
    data.update({f"meta_{k}": np.asarray(v) for k, v in meta.items()})
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **data)
    policy = Policy(tmp)

    # 1. network
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    worst = 0.0
    for _ in range(500):
        torch_a = model.predict(norm.normalize_obs(obs), deterministic=True)[0]
        worst = max(worst, np.abs(policy(obs) - np.clip(torch_a, -1, 1)).max())
        obs, _, term, trunc, _ = env.step(torch_a + 0.3 * rng.standard_normal(env.nu))
        if term or trunc or env.episode.t > 9:
            obs, _ = env.reset()
    print(f"1. network: max action difference numpy vs PyTorch {worst:.2e}")
    ok = worst < 1e-5

    # 2. closed loop through the robot controller
    env.reset(seed=1)
    env.place_pose(env.robot.home_joint)
    for _ in range(25):
        env.hold_pose(env.robot.home_joint)
    obs = env.start_policy()
    ctrl = Controller(policy)

    def sensors():
        d, R = env.data, env.root_rot()
        return {"q": d.qpos[env.robot.qpos_idx].copy(), "qd": d.qvel[env.robot.dof_idx].copy(),
                "gravity": env.gravity_body(), "gyro": d.qvel[3:6].copy(), "yaw": float(np.arctan2(R[1, 0], R[0, 0]))}

    ctrl.start(sensors(), phase=env.command.phase)
    obs_err = act_err = 0.0
    fell = False
    for seconds, vx, vy, wz in SCRIPT:
        env.set_command(vx, vy, wz)
        ctrl.set_command(vx, vy, wz)
        obs = observations.compute(env)  # the sim's observation with the new command
        for _ in range(int(seconds / env.dt)):
            ctrl.tick(sensors())
            diff = np.abs(ctrl.last_obs - obs[:env.actor_obs_dim])
            if diff.max() > 1e-6 and obs_err <= 1e-6:
                i = np.flatnonzero(diff > 1e-6)
                print(f"   first observation mismatch at t={env.episode.t:.2f} s, command {ctrl.command}, entries {i.tolist()}:"
                      f"\n   robot {np.round(ctrl.last_obs[i], 4).tolist()}\n   sim   {np.round(obs[i], 4).tolist()}")
            obs_err = max(obs_err, diff.max())
            sim_a = np.clip(model.predict(norm.normalize_obs(obs), deterministic=True)[0], -1, 1)
            act_err = max(act_err, np.abs(ctrl.last_action - sim_a).max())
            obs, _, term, _, _ = env.step(sim_a)
            target_err = np.abs(ctrl.target - env.episode.target).max()
            act_err = max(act_err, target_err)
            if term:
                fell = True
                break
        if fell:
            break
    print(f"2. robot controller in the simulator: max observation difference {obs_err:.2e}, "
          f"max action / target difference {act_err:.2e}, walked {np.hypot(*env.data.qpos[:2]):.2f} m"
          f"{', FELL' if fell else ', stayed up'}")
    ok = ok and obs_err < 1e-6 and act_err < 1e-5 and not fell
    if not ok:
        tmp.unlink()
        raise SystemExit("MISMATCH: not exported")
    tmp.replace(out)
    print(f"saved {out.relative_to(PROJECT)} ({out.stat().st_size // 1024} kB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    a = p.parse_args()
    run = Path(a.run)
    out = PROJECT / "robot" / "policies" / f"{run.name}.npz"
    out.parent.mkdir(exist_ok=True)
    export(run, out)


if __name__ == "__main__":
    main()
