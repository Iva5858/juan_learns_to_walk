"""Export a trained policy for the robot (numpy only), and check it against the simulator.

    python scripts/export_policy.py runs/<name>      # -> robot/policies/<name>.npz

Checks, before saving:
  1. the numpy network gives the same actions as the trained PyTorch policy;
  2. the robot's controller (robot/policy.py) builds exactly the simulator's observations and
     drives the simulated Juan through stand / walk / turn / sideways / back / stand, step by step
     the same as the training code, without falling. On the nominal robot, but with the shortest
     servo delay the policy was trained with: a policy trained only with long delays can be
     unstable without any, which says nothing about the real robot (it has them).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from biped_rl.robots import PROJECT
from biped_rl.symmetry import mirror_maps
from biped_rl.tasks.locomotion import BipedEnv, EnvConfig
from biped_rl.tasks.locomotion.mdp import observations
from robot.joints import NAMES
from robot.policy import Controller, Policy

# (seconds, forward, sideways, turn): the closed-loop check
SCRIPT = [(2, 0, 0, 0), (4, 0.3, 0, 0), (3, 0.2, 0, 0.4), (3, 0, 0.1, 0), (2, -0.1, 0, 0), (2, 0, 0, 0)]


def top_speed(run, c):
    """The fastest forward command it trained on: where the speed curriculum got to, not where it started."""
    path = run / "curriculum.json"
    reached = json.loads(path.read_text())["speed_max"] if path.exists() else 0.0
    return round(max(c.speed_range[1], min(reached, c.speed_limit)), 2)


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
            "forward_range": (-c.backward_speed, top_speed(run, c)), "lateral_range": c.lateral_range,
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
    env.servo.action_delay = int(cfg.randomization.action_delay_steps[0])
    env.settle_at_home(25)
    obs = env.start_policy()
    ctrl = Controller(policy)
    ctrl.start(env.robot_readings(), phase=env.command.phase)
    obs_err = act_err = 0.0
    fell = False
    for seconds, vx, vy, wz in SCRIPT:
        env.set_command(vx, vy, wz)
        ctrl.set_command(vx, vy, wz)
        obs = observations.compute(env)  # the sim's observation with the new command
        for _ in range(int(seconds / env.dt)):
            ctrl.tick(env.robot_readings())
            diff = np.abs(ctrl.last_obs - obs[:env.actor_obs_dim])
            if diff.max() > 1e-6 and obs_err <= 1e-6:
                i = np.flatnonzero(diff > 1e-6)
                print(f"   first observation mismatch at t={env.episode.t:.2f} s, command {ctrl.command}, entries {i.tolist()}:"
                      f"\n   robot {np.round(ctrl.last_obs[i], 4).tolist()}\n   sim   {np.round(obs[i], 4).tolist()}")
            obs_err = max(obs_err, diff.max())
            sim_a = np.clip(model.predict(norm.normalize_obs(obs), deterministic=True)[0], -1, 1)
            act_err = max(act_err, np.abs(ctrl.last_action - sim_a).max())
            obs, _, term, _, _ = env.step(sim_a)
            target_err = np.abs(ctrl.target - env.episode.target_buf[-1]).max()  # newest, before the delay
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
    print(f"saved {out} ({out.stat().st_size // 1024} kB)")


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
