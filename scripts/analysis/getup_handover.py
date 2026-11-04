"""Getting up, then the walking policy takes over: does the robot end up standing? On randomized robots
(the walking runs' physics randomization) from varied lying poses. Per robot: the getup policy runs until
it has stood on its feet for --stand-for s (or its 8 s run out); the walking policy then takes over the
same robot (same masses, friction, servos, sensor biases), commanded to stand, for --hold s, optionally
with a shove of --shove m/s at 2 s.

    python scripts/analysis/getup_handover.py runs/getup/juan_getup_8_best runs/v7/juan_v7_24_speed
    python scripts/analysis/getup_handover.py runs/getup/juan_getup_9 runs/v7/juan_v7_25_stand --seeds 50 --video out/handover

Prints the share of robots that got up, that the walker then kept standing, the time to stand, and how
each failure happened.
"""
import argparse
import copy
import json
from pathlib import Path

import imageio
import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from biped_rl.runs import load_policy
from biped_rl.tasks.getup import GetUpEnv, config
from biped_rl.tasks.locomotion import BipedEnv, EnvConfig

MODEL_FIELDS = ("body_mass", "body_inertia", "body_ipos", "geom_friction", "dof_armature", "dof_frictionloss")


def take_over(g, w):
    """Put walking env `w` in getup env `g`'s exact situation: the same robot and the same moment."""
    for f in MODEL_FIELDS:
        getattr(w.model, f)[:] = getattr(g.model, f)
    w.model.opt.gravity[:] = g.model.opt.gravity
    w.servo, w.domain = copy.deepcopy(g.servo), copy.deepcopy(g.domain)
    w.data.qpos[:], w.data.qvel[:] = g.data.qpos, g.data.qvel
    mujoco.mj_forward(w.model, w.data)
    return w.start_policy({"stand": True})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("getup")
    p.add_argument("walker")
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--physics-from", default="runs/v7/juan_v7_13_straight", help="physics randomization ('' = nominal robot)")
    p.add_argument("--side-deg", type=float, default=90.0)
    p.add_argument("--joint-rad", type=float, default=0.5)
    p.add_argument("--stand-for", type=float, default=0.5, help="s on its feet before the walker takes over")
    p.add_argument("--at-idle", action="store_true", help="hand over when getup reaches idle (tasks/getup.py) instead")
    p.add_argument("--hold", type=float, default=5.0, help="s the walker must keep it standing")
    p.add_argument("--shove", type=float, default=0.0, help="m/s shove 2 s after the handover, random direction")
    p.add_argument("--video", help="folder for videos of the first 4 robots")
    a = p.parse_args()

    gdir = Path(a.getup)
    saved = json.loads((gdir / "getup.json").read_text())
    weights = saved["weights"]
    start = {"side_deg": a.side_deg, "joint_rad": a.joint_rad, "any_heading": True,
             "knee_min_deg": saved.get("start", {}).get("knee_min_deg")}   # the run's own knee limit, if any
    render = {"render_mode": "rgb_array", "width": 320, "height": 240} if a.video else {}
    g = GetUpEnv(config(8.0, a.physics_from or None).to_dict(), weights, start, **render)
    gm = PPO.load(gdir / "final_model", device="cpu")
    gn = VecNormalize.load(str(gdir / "vecnormalize.pkl"), DummyVecEnv([lambda: g]))
    gn.training = False
    wc = EnvConfig.load(a.walker)
    wc.use_measured_servos()
    wc.randomization.push_velocity = 0.0
    wm, wn = load_policy(a.walker, wc)
    w = BipedEnv(wc.to_dict(), **render)
    if a.video:
        Path(a.video).mkdir(parents=True, exist_ok=True)

    got_up, kept, times, fails = 0, 0, [], {}
    for seed in range(a.seeds):
        frames = []
        obs, _ = g.reset(seed=seed)
        handed = None
        for i in range(g._max_steps):
            obs, *_ = g.step(gm.predict(gn.normalize_obs(obs), deterministic=True)[0])
            if a.video and seed < 4 and i % 2:
                frames.append(g.render())
            if (getattr(g.episode, "idle_for", 0.0) >= 0.5) if a.at_idle else g.episode.stood_for >= a.stand_for:
                handed = (i + 1) * g.dt
                break
        if handed is None:
            fails["never stood up"] = fails.get("never stood up", 0) + 1
            outcome = "never stood up"
        else:
            got_up += 1
            times.append(handed)
            w.reset(seed=seed)
            wobs = take_over(g, w)
            fell = None
            for k in range(int(a.hold / w.dt)):
                if a.shove and k == int(2.0 / w.dt):
                    ang = w.np_random.uniform(-np.pi, np.pi)
                    w.data.qvel[0:2] += a.shove * np.array([np.cos(ang), np.sin(ang)])
                    w.episode.last_push = w.episode.t
                wobs, _, term, _, _ = w.step(wm.predict(wn.normalize_obs(wobs), deterministic=True)[0])
                if a.video and seed < 4 and k % 2:
                    frames.append(w.render())
                if term:
                    fell = (k + 1) * w.dt
                    break
            if fell is None:
                kept += 1
                outcome = f"up at {handed:.1f} s, kept standing"
            else:
                key = "walker fell in its first 0.5 s" if fell < 0.5 else "walker fell later"
                fails[key] = fails.get(key, 0) + 1
                outcome = f"up at {handed:.1f} s, walker fell after {fell:.1f} s"
        print(f"robot {seed:3d}: {outcome}", flush=True)
        if a.video and seed < 4 and frames:
            imageio.mimsave(Path(a.video) / f"robot{seed}.mp4", frames, fps=25, macro_block_size=1)

    n = a.seeds
    print(f"\n{gdir.name} -> {Path(a.walker).name}: got up {got_up}/{n}, then kept standing {kept}/{n} "
          f"({100 * kept / n:.0f}%); time to standing (incl. {a.stand_for} s on its feet) median "
          f"{np.median(times) if times else float('nan'):.2f} s; failures {fails}")


if __name__ == "__main__":
    main()
