"""Standing shove test: commanded to stand still, the robot gets one shove from each of 8 directions
(robot frame) at growing strength, and has 2 s to stay up (stepping is allowed). Three starts: standing
from the beginning, and stopped after 2 s of walking forward (0.2 m/s) or backward (-0.15 m/s), so the
feet may be staggered; the shove comes 1 s after the stop. Also how far the joints end up from home when
nobody pushes and how far one foot stays ahead of the other. Randomized robots at full difficulty, the same seeds
for every policy; by default on the measured real servos.

    python scripts/analysis/stand_pushes.py runs/juan_v7_19_topspeed_best runs/juan_v7_20_stand_best
"""
import argparse
from pathlib import Path

import numpy as np

from biped_rl.runs import load_policy
from biped_rl.tasks.locomotion import BipedEnv, EnvConfig

DIRECTIONS = ("fwd", "fwd-L", "left", "back-L", "back", "back-R", "right", "fwd-R")  # where the shove pushes to
STRENGTHS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)  # m/s added to the torso
STARTS = {"standing": None, "stop fwd": 0.2, "stop back": -0.15}
WALK_S = 2.0


def episode(env, model, norm, seed, walk, k=None, strength=0.0):
    """Stays up? the joints' rms offset from home at the end [deg], and one foot ahead of the other [cm]."""
    if walk is None:
        obs, _ = env.reset(seed=seed, options={"stand": True})
        stop = 0.0
    else:
        obs, _ = env.reset(seed=seed, options={"command": (walk, 0.0, 0.0)})
        stop = WALK_S
    stopped = walk is None
    while env.episode.t < stop + 3.0:
        if not stopped and env.episode.t >= stop:
            env.set_command(0.0)
            stopped = True
        if k is not None and env.episode.t >= stop + 1.0:
            angle = env.yaw() + k * np.pi / 4
            env.data.qvel[0:2] += strength * np.array([np.cos(angle), np.sin(angle)])
            env.episode.last_push = env.episode.t  # as in training: gait penalties pause while recovering
            k = None
        obs, _, term, trunc, _ = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
        if term:
            return False, np.nan, np.nan
        if trunc:
            break
    q = env.data.qpos[env.robot.qpos_idx] - env.robot.home_joint
    feet = env.data.xpos[env.robot.foot_bodies]
    yaw = env.yaw()
    ahead = np.cos(yaw) * (feet[0, 0] - feet[1, 0]) + np.sin(yaw) * (feet[0, 1] - feet[1, 1])
    return True, float(np.degrees(np.sqrt(np.mean(q ** 2)))), 100 * abs(float(ahead))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--seeds", type=int, default=3, help="robots per direction and strength")
    p.add_argument("--best", action="store_true", help="the run's best_model instead of final_model")
    p.add_argument("--as-trained", action="store_true", help="each run's own servo model, not the measured one")
    a = p.parse_args()
    print(f"Stays up after one shove while standing ({a.seeds} randomized robots each). Per direction: the "
          f"strongest shove [m/s] all of them survived; then the share up per strength, all directions; "
          f"then, unpushed, 3 s after the stop: joints' rms offset from home and one foot ahead of the other.")
    print(f"{'policy':30s} {'start':9s} " + " ".join(f"{d:>6s}" for d in DIRECTIONS) + " | "
          + " ".join(f"{s:>4.1f}" for s in STRENGTHS) + " | off home  stagger")
    for run in map(Path, a.runs):
        cfg = EnvConfig.load(run)
        if not a.as_trained:
            cfg.use_measured_servos()
        cfg.randomization.enabled = True
        cfg.randomization.push_velocity = 0.0
        cfg.randomization.push_curriculum = False
        cfg.randomization.physics_curriculum = False
        model, norm = load_policy(run, cfg, best=a.best)
        env = BipedEnv(cfg.to_dict())
        for start, walk in STARTS.items():
            seeds = [20_000 + i for i in range(a.seeds)]
            up = np.array([[[episode(env, model, norm, sd, walk, k, s)[0] for sd in seeds]
                            for s in STRENGTHS] for k in range(8)])  # direction x strength x seed
            calm = [episode(env, model, norm, sd, walk) for sd in seeds]
            held = []
            for k in range(8):
                ok = [s for s, row in zip(STRENGTHS, up[k]) if row.all()]
                held.append(f"{max(ok):6.1f}" if ok else f"{'<' + str(STRENGTHS[0]):>6s}")
            share = up.mean(axis=(0, 2))
            off = np.nanmean([c[1] for c in calm]) if any(c[0] for c in calm) else np.nan
            stagger = np.nanmean([c[2] for c in calm]) if any(c[0] for c in calm) else np.nan
            falls = sum(not c[0] for c in calm)
            print(f"{run.name:30s} {start:9s} " + " ".join(held) + " | "
                  + " ".join(f"{100 * x:3.0f}%" for x in share)
                  + f" | {off:7.1f}° {stagger:5.1f} cm" + (f" ({falls} fell unpushed)" if falls else ""), flush=True)


if __name__ == "__main__":
    main()
