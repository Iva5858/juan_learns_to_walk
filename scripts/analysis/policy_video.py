"""Film a policy following a sequence of commands in one episode (nominal robot, from its start pose), side view.

    python scripts/analysis/policy_video.py runs/v9/juan_v9_walk_at52M runs/v9/media/juan_v9_walk_at52M.mp4
"""
import sys

import imageio

from biped_rl.runs import load_policy
from biped_rl.tasks.locomotion import BipedEnv, EnvConfig

SEQUENCE = [("stand", (0.0, 0.0, 0.0), 2.0), ("forward 0.15", (0.15, 0.0, 0.0), 5.0), ("forward 0.3", (0.3, 0.0, 0.0), 5.0),
            ("sideways left 0.1", (0.0, 0.1, 0.0), 4.0), ("turn left 0.6", (0.0, 0.0, 0.6), 4.0), ("stand", (0.0, 0.0, 0.0), 3.0)]


def main():
    run, out = sys.argv[1], sys.argv[2]
    cfg = EnvConfig.load(run)
    model, norm = load_policy(run, cfg)
    env = BipedEnv(cfg.to_dict(), render_mode="rgb_array", width=640, height=480)
    obs, _ = env.reset(seed=1, options={"command": SEQUENCE[0][1]})
    frames = [env.render()]
    env._camera.azimuth = 90
    for label, cmd, seconds in SEQUENCE:
        env.set_command(*cmd)
        for _ in range(int(seconds / env.dt)):
            obs, _, term, _, _ = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
            frames.append(env.render())
            if term:
                print(f"fell during: {label}")
                imageio.mimsave(out, frames, fps=int(1 / env.dt))
                return
        print(f"{label}: done")
    imageio.mimsave(out, frames, fps=int(1 / env.dt))
    print(f"wrote {out} ({len(frames) * env.dt:.0f} s)")


if __name__ == "__main__":
    main()
