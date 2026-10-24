"""How a policy stands and walks straight, run through the robot's own controller (robot/policy.py)
on the robot with the servos measured on the bench:
  * standing still: torso lean, joint offset from home, and the jump when it takes over from home
  * straight walking at 0.15 m/s for 10 s: forward / sideways distance, turning

    python scripts/analysis/standing.py juan_v7_10a_real_servos juan_v7_11_upright
(names of exported policies in robot/policies/)
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from biped_rl.tasks.locomotion import BipedEnv, EnvConfig  # noqa: E402
from robot.policy import Controller, Policy  # noqa: E402

ROBOT_RUN = PROJECT / "runs" / "juan_v7_10a_real_servos"  # measured servos: 3.0 rad/s, kp 6


def episode(policy, cfg, seed, command, seconds):
    env = BipedEnv(cfg.to_dict())
    env.reset(seed=seed)
    env.place_pose(env.robot.home_joint)
    for _ in range(50):
        env.hold_pose(env.robot.home_joint)
    env.start_policy()
    ctrl = Controller(policy)
    yaw = lambda: math.atan2(env.root_rot()[1, 0], env.root_rot()[0, 0])
    sens = lambda: {"q": env.data.qpos[env.robot.qpos_idx].copy(), "qd": env.data.qvel[env.robot.dof_idx].copy(),
                    "gravity": env.gravity_body(), "gyro": env.gyro_reading.copy(), "yaw": yaw()}
    ctrl.start(sens())
    ctrl.set_command(*command)
    env.set_command(*command)
    y0, p0 = yaw(), env.data.qpos[:2].copy()
    lean, offset, jump = [], [], 0.0
    for t in range(int(seconds * 50)):
        ctrl.tick(sens())
        if t < 25:  # the first half second after taking over from home
            jump = max(jump, float(np.degrees(np.abs(ctrl.target - env.robot.home_joint)).max()))
        if env.step(ctrl.last_action)[2]:
            return None
        if t >= 50:
            g = env.gravity_body()
            lean.append(math.degrees(math.atan2(g[0], -g[2])))  # + = forward
            offset.append(float(np.degrees(np.sqrt(np.mean((env.data.qpos[env.robot.qpos_idx] - env.robot.home_joint) ** 2)))))
    d = env.data.qpos[:2] - p0
    fwd = d[0] * math.cos(y0) + d[1] * math.sin(y0)
    left = -d[0] * math.sin(y0) + d[1] * math.cos(y0)
    turn = math.degrees((yaw() - y0 + math.pi) % (2 * math.pi) - math.pi)
    return float(np.mean(lean)), float(np.mean(offset)), jump, fwd, left, turn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("policies", nargs="+")
    p.add_argument("--seeds", type=int, default=6)
    a = p.parse_args()
    base = EnvConfig.from_dict(json.loads((ROBOT_RUN / "config.json").read_text()))
    base.randomization.push_velocity = 0.0
    print(f"{'policy':38s} {'physics':10s} | standing: {'lean fwd':>8s} {'off home':>8s} {'jump':>6s} | "
          f"walk 0.15 m/s, 10 s: {'fwd m':>6s} {'left m':>7s} {'turn':>6s} | falls")
    for name in a.policies:
        policy = Policy(PROJECT / "robot" / "policies" / f"{name}.npz")
        for physics in ("nominal", "randomized"):
            cfg = EnvConfig.from_dict(base.to_dict())
            cfg.randomization.enabled = physics == "randomized"
            seeds = range(1 if physics == "nominal" else a.seeds)
            stand = [episode(policy, cfg, s, (0.0, 0.0, 0.0), 6) for s in seeds]
            walk = [episode(policy, cfg, s, (0.15, 0.0, 0.0), 10) for s in seeds]
            falls = sum(r is None for r in stand + walk)
            st = np.array([r for r in stand if r]) if any(stand) else np.full((1, 6), np.nan)
            wk = np.array([r for r in walk if r]) if any(walk) else np.full((1, 6), np.nan)
            print(f"{name:38s} {physics:10s} | {'':9s} {st[:, 0].mean():+7.1f}° {st[:, 1].mean():7.1f}° {st[:, 2].mean():5.1f}° | "
                  f"{'':20s} {wk[:, 3].mean():6.2f} {wk[:, 4].mean():+7.2f} {wk[:, 5].mean():+5.1f}° | {falls}/{2 * len(seeds)}",
                  flush=True)


if __name__ == "__main__":
    main()
