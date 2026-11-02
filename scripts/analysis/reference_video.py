"""Film the reference gait (biped_rl/tasks/locomotion/mdp/reference.py) on the robot model: the poses played
back with the body carried along at the commanded speed and heading, a clip per command, one video.

    python scripts/analysis/reference_video.py                       # runs/v9/media/ostrich_reference.mp4
"""
import argparse
from pathlib import Path

import imageio
import mujoco
import numpy as np

from biped_rl.tasks.locomotion import BipedEnv, EnvConfig
from biped_rl.tasks.locomotion.mdp import reference as ref

CLIPS = [("forward 0.2 m/s", (0.2, 0.0, 0.0), 4.0), ("forward 0.35 m/s", (0.35, 0.0, 0.0), 4.0),
         ("backward 0.2 m/s (knees forward)", (-0.2, 0.0, 0.0), 4.0), ("sideways left 0.15 m/s", (0.0, 0.15, 0.0), 3.0),
         ("turning in place (left to the policy)", (0.0, 0.0, 0.8), 3.0), ("walk and turn", (0.25, 0.0, 0.5), 4.0),
         ("standing", (0.0, 0.0, 0.0), 1.5)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/v9/media/ostrich_reference.mp4")
    p.add_argument("--azimuth", type=float, default=90.0, help="camera angle: 90 = from the side")
    a = p.parse_args()
    env = BipedEnv(EnvConfig().to_dict(), render_mode="rgb_array", width=640, height=480)
    env.reset(seed=0)
    d = env.data
    fps, frames = 50, []
    x, y, yaw, phase = 0.0, 0.0, 0.0, 0.0
    for label, cmd, seconds in CLIPS:
        print(f"{label}: {ref.cadence(cmd):.1f} steps/s")
        for _ in range(int(seconds * fps)):
            dt = 1.0 / fps
            phase = (phase + dt * ref.cadence(cmd) / 2) % 1.0
            vx, vy, wz = cmd
            x += (vx * np.cos(yaw) - vy * np.sin(yaw)) * dt
            y += (vx * np.sin(yaw) + vy * np.cos(yaw)) * dt
            yaw += wz * dt
            env.place_pose(ref.reference_pose(phase, cmd))   # joint angles, soles grounded
            d.qpos[0:2] = (x, y)
            d.qpos[3:7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
            mujoco.mj_forward(env.model, d)
            env.ground()
            frame = env.render()
            if env._camera.azimuth != a.azimuth:  # the camera exists after the first render
                env._camera.azimuth = a.azimuth
                frame = env.render()
            frames.append(frame)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=fps)
    print(f"wrote {out} ({len(frames) / fps:.0f} s)")


if __name__ == "__main__":
    main()
