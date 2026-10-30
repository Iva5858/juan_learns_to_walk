"""3D video of what the real robot did: a robot log (robot/logs/*.npz) played back in the simulator's
renderer, with the telemetry overlaid.

  * joint angles: as measured by the servos, every 20 ms
  * torso tilt: from the IMU's gravity estimate; heading: the IMU's integrated yaw
  * position on the floor: not measured (no odometry), estimated by keeping the standing foot
    planted (the lower foot stays where it touched down; the body moves around it). Foot slip is
    not captured, so treat distances as approximate.

No physics: this shows the real motion, it does not simulate it (for that, see replay.py).

    python scripts/analysis/replay_video.py robot/logs/<log>.npz              # -> out/replay/<log>_3d.mp4
    python scripts/analysis/replay_video.py <log> --azimuth 180 --start 20 --seconds 30
"""
import argparse
import math
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw

from biped_rl.robot_logs import Replay, kinematic_env
from biped_rl.robots import PROJECT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("--start", type=float, default=0.0, help="seconds into the run")
    p.add_argument("--seconds", type=float, default=None, help="how much to render (default: all)")
    p.add_argument("--azimuth", type=float, default=120.0, help="camera angle around the robot (180 = from the front)")
    p.add_argument("--fps", type=int, default=25, help="video frame rate (the log is 50 Hz)")
    p.add_argument("--size", default="640x480")
    a = p.parse_args()
    log = dict(np.load(a.log))
    W, H = map(int, a.size.split("x"))
    env = kinematic_env(render_mode="rgb_array", width=W, height=H)
    env._draw_track_markers = lambda scene: None  # (the simulator's commanded-path and push markers)
    env._draw_push_arrow = lambda scene: None

    dt = float(np.median(np.diff(log["t"]))) if len(log["t"]) > 1 else 0.02
    k0 = int(a.start / dt)
    k1 = len(log["t"]) if a.seconds is None else min(len(log["t"]), k0 + int(a.seconds / dt))
    every = max(1, round(1 / (dt * a.fps)))
    yaw0 = float(log["yaw"][k0])
    status = str(log.get("status", ""))
    link = log.get("link_s")

    replay = Replay(log, env, yaw0=yaw0)
    frames = []
    for k in range(k0, k1):
        replay.pose(k)
        roll, pitch, base_xy = replay.roll, replay.pitch, replay.base_xy
        if (k - k0) % every:
            continue
        if getattr(env, "_camera", None) is None:
            env.render()  # (creates the camera)
        env._camera.azimuth = a.azimuth
        img = Image.fromarray(env.render())
        dr = ImageDraw.Draw(img)
        c = log["command"][k]
        cmd = "stand still" if not np.any(c) else f"fwd {c[0]:+.2f} m/s  side {c[1]:+.2f}  turn {c[2]:+.2f} rad/s"
        lines = [f"t = {log['t'][k]:6.1f} s    {cmd}",
                 f"lean fwd {pitch:+5.1f} deg   right {roll:+5.1f} deg   heading {math.degrees(float(log['yaw'][k]) - yaw0):+6.0f} deg",
                 f"distance (est.) {np.hypot(*base_xy):4.2f} m"]
        if link is not None and link[k] > 1.0:
            lines.append(f"LINK GAP {link[k]:.1f} s")
        if k >= len(log["t"]) - 50 and status:  # (the run's last second)
            lines.append(status)
        dr.rectangle([0, 0, W, 16 * len(lines) + 8], fill=(0, 0, 0))
        for i, line in enumerate(lines):
            color = (255, 90, 90) if line.startswith(("LINK", "stopped: tilted")) else (255, 230, 120)
            dr.text((8, 4 + 16 * i), line, fill=color)
        frames.append(np.array(img))
    out = PROJECT / "out" / "replay" / f"{Path(a.log).stem}_3d.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=a.fps, macro_block_size=1)
    print(f"{len(frames)} frames ({(k1 - k0) * dt:.0f} s of the run), estimated distance {np.hypot(*base_xy):.2f} m -> {out.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
