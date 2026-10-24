"""Check every gesture (robot/gestures.py) on the simulated Juan before it runs on the robot: played
from home with the servos measured on the bench (the v7_10 robot), nominal and randomized physics.

    python scripts/check_gestures.py                 # all gestures, table + verdicts
    python scripts/check_gestures.py --video         # also out/gestures/<name>.mp4 (nominal robot)
    python scripts/check_gestures.py happy_bounce    # just one

Passes when: never falls, torso tilt under 15 deg (gestures.MAX_TILT for ones that lean on purpose,
including the ~3-6 deg the robot leans standing at home), servo torque under 90% of stall.
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))  # the robot/ package isn't installed
from biped_rl.tasks.locomotion import BipedEnv, EnvConfig  # noqa: E402
from robot import gestures as G  # noqa: E402
from robot.joints import NAMES  # noqa: E402

ROBOT_RUN = PROJECT / "runs" / "juan_v7_10a_real_servos"  # measured servos: 3.0 rad/s, kp 6
MAX_TILT, MAX_TORQUE = 15.0, 0.9


def play(cfg, keys, seed, video=None):
    env = BipedEnv(cfg.to_dict(), render_mode="rgb_array" if video else None, width=480, height=360)
    env.reset(seed=seed)
    env.place_pose(env.robot.home_joint)
    for _ in range(50):
        env.hold_pose(env.robot.home_joint)
    sv, m, d = env.servo, env.model, env.data
    slack = sv.backlash if sv.backlash.any() else None
    tilt = peak_tau = peak_speed = 0.0
    fell = False
    imgs = []
    targets = [np.radians([f[n] for n in NAMES]) for f in G.frames(keys)] + [env.robot.home_joint] * 50
    for target in targets:
        env.episode.target = np.clip(target, env.robot.joint_lo, env.robot.joint_hi)
        for _ in range(env.frame_skip):
            qd, tau, tau_c = env.servo_torque(env.episode.target, slack, 1.0 / sv.speed)
            d.ctrl[:] = tau_c
            mujoco.mj_step(m, d)
            peak_tau = max(peak_tau, float(np.max(np.abs(tau_c) / sv.stall)))
            peak_speed = max(peak_speed, float(np.max(np.abs(qd) / sv.speed)))
        g = env.gravity_body()
        tilt = max(tilt, float(np.degrees(np.arccos(np.clip(-g[2], -1, 1)))))
        if env.non_foot_ground_contact() or g[2] > -0.6:
            fell = True
            break
        if video is not None:
            imgs.append(env.render())
    if video is not None and imgs:
        import imageio
        video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video, imgs, fps=G.CONTROL_HZ)
    return fell, tilt, peak_tau, peak_speed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="*")
    p.add_argument("--video", action="store_true")
    p.add_argument("--seeds", type=int, default=6)
    a = p.parse_args()
    base = EnvConfig.from_dict(json.loads((ROBOT_RUN / "config.json").read_text()))
    base.randomization.push_velocity = 0.0
    names = a.names or list(G.ALL)
    print(f"{'gesture':14s} {'s':>4s}  {'falls':>6s} {'max tilt':>9s} {'torque/stall':>12s} {'speed/max':>9s}  verdict")
    ok_all = True
    for name in names:
        label, _, _, keys = G.ALL[name]
        results = []
        for physics in ("nominal", "randomized"):
            cfg = EnvConfig.from_dict(base.to_dict())
            cfg.randomization.enabled = physics == "randomized"
            for seed in range(1 if physics == "nominal" else a.seeds):
                video = PROJECT / "out" / "gestures" / f"{name}.mp4" if a.video and physics == "nominal" else None
                results.append(play(cfg, keys, seed, video))
        falls = sum(r[0] for r in results)
        tilt = max(r[1] for r in results)
        tau = max(r[2] for r in results)
        speed = max(r[3] for r in results)
        ok = falls == 0 and tilt < G.MAX_TILT.get(name, MAX_TILT) and tau < MAX_TORQUE
        ok_all &= ok
        print(f"{name:14s} {G.duration(keys):4.1f}  {falls:>3d}/{len(results):<2d} {tilt:8.1f}° {tau:12.2f} {speed:9.2f}  "
              f"{'OK' if ok else 'NOT SAFE'}")
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
